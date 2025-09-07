from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune
from datumaro.plugins.validators import DetectionValidator, SegmentationValidator
import os
from typing import Literal, Union, List
import torch
import clip
from PIL import Image
import faiss
import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp
import logging
import copy
import sklearn.cluster as skc
from scipy.spatial import cKDTree




def sample_data(img_dir: str, 
                cluster_method: Literal["cluster_random", 
                                        "query_clust", 
                                        "centroid",
                                        "entropy",
                                        "ndr",
                                        "random"
                                        ]="cluster_random", 
                reduce_proportion: float=0.5,
                output_dir: str = "samples", 
                save_format: str= "coco_instances",
                **kwargs
                ):
    """_summary_

    Args:
        img_dir (str): Directory containing images
        cluster_method (str, optional): Clustering method. Defaults to "cluster_random".
        reduce_proportion (float, optional): Ratio of data to select (0 - 1). Defaults to 0.5.
        output_dir (str, optional): Directory to store selected data. Defaults to "samples".
        save_format (str, optional): format to save data. Defaults to "coco_instances".
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    env = Environment()
    detected_format = env.detect_dataset(path=img_dir)
    dataset = Dataset.import_from(img_dir, detected_format[0])
    prune = Prune(dataset, cluster_method=cluster_method)
    cluster_random_result = prune.get_pruned(ratio=reduce_proportion)
    cluster_random_result.export(output_dir, format=save_format, save_media=True)
    

class ImageUniqueness(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir 
    
    
    
    
from typing import List 


def get_embeddings(img_list, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_type, device=device)

    embeddings = []
    for img_path in img_list:
        img = Image.open(img_path)
        image = preprocess(img=img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            embeddings.append(image_features)

    return embeddings


logger = logging.getLogger(__name__)

def _compute_representativeness(embeddings, 
                                method: Literal["cluster-center", 
                                                "cluster-center-downweight"
                                                ]="cluster-center",
                                cluster_algorithm="kmeans", 
                                N=20, 
                                quantile=0.8, n_samples=500,
                                bandwidth: Union[float, None, 
                                                Literal["auto"]
                                                ]=None,
                                norm_method: Literal["local", "global"]="local"
                                ):
    #
    # @todo experiment on which method for assessing representativeness
    #
    num_embeddings = len(embeddings)
    logger.info(
        "Computing clusters for %d embeddings; this may take awhile...",
        num_embeddings,
    )

    initial_ranking, _ = _cluster_ranker(embeddings,
                                         cluster_algorithm=cluster_algorithm,
                                         N=N, quantile=quantile,
                                         n_samples=n_samples,
                                         bandwidth=bandwidth,
                                         norm_method=norm_method
                                         )

    if method == "cluster-center":
        final_ranking = initial_ranking
    elif method == "cluster-center-downweight":
        logger.info("Applying iterative downweighting...")
        final_ranking = _adjust_rankings(
            embeddings, initial_ranking, ball_radius=0.5
        )
    else:
        raise ValueError(
            (
                "Method '%s' not supported. Please use one of "
                "['cluster-center', 'cluster-center-downweight']"
            )
            % method
        )

    return final_ranking


def _cluster_ranker(embeddings, 
                    cluster_algorithm="kmeans", 
                    N=20, 
                    quantile=0.8, n_samples=500,
                    bandwidth: Union[float, None, 
                                     Literal["auto"]
                                     ]=None,
                    norm_method: Literal["local", "global"]="local"
                    ):
    if cluster_algorithm == "meanshift":
        if bandwidth is None:
            bandwidth = skc.estimate_bandwidth(embeddings, 
                                               quantile=quantile, 
                                               n_samples=n_samples
                                                )
        if bandwidth == "auto":
            bandwidth = None
        clusterer = skc.MeanShift(bandwidth=bandwidth, 
                                  bin_seeding=True).fit(embeddings)   
    elif cluster_algorithm == "kmeans":
        clusterer = skc.KMeans(n_clusters=N, 
                               random_state=1234
                               ).fit(embeddings)
    else:
        raise ValueError(
            (
                "Clustering algorithm '%s' not supported. Please use one of "
                "['meanshift', 'kmeans']"
            )
            % cluster_algorithm
        )

    cluster_centers = clusterer.cluster_centers_
    cluster_ids = clusterer.labels_

    # Get distance from each point to it's closest cluster center
    sample_dists = np.linalg.norm(
        embeddings - cluster_centers[cluster_ids], axis=1
    )

    centerness_ranking = 1 / (1 + sample_dists)

    # Normalize per cluster vs globally
    #norm_method = "local"
    if norm_method == "global":
        centerness_ranking = centerness_ranking / centerness_ranking.max()
    elif norm_method == "local":
        unique_ids = np.unique(cluster_ids)
        for unique_id in unique_ids:
            cluster_indices = np.where(cluster_ids == unique_id)[0]
            cluster_dists = sample_dists[cluster_indices]
            cluster_dists /= cluster_dists.max()
            sample_dists[cluster_indices] = cluster_dists
        centerness_ranking = sample_dists

    return centerness_ranking, clusterer


# Step 3: Adjust rankings to avoid redundancy
def _adjust_rankings(embeddings, initial_ranking, ball_radius=0.5):
    tree = cKDTree(embeddings)
    new_ranking = copy.deepcopy(initial_ranking)

    ordered_ranking = np.argsort(new_ranking)[::-1]
    visited_indices = set()

    for ranked_index in ordered_ranking:
        visited_indices.add(ranked_index)
        query_embedding = embeddings[ranked_index, :]
        nearby_indices = tree.query_ball_point(
            query_embedding, ball_radius, return_sorted=True
        )
        filtered_indices = [idx for idx in nearby_indices 
                            if idx not in visited_indices
                            ]
        visited_indices |= set(filtered_indices)
        new_ranking[filtered_indices] = new_ranking[filtered_indices] * 0.7

    new_ranking = new_ranking / new_ranking.max()
    return new_ranking

class RepresentativenessSampler(object):
    def __init__(self, img_list: List[str],
                method: Literal["cluster-center", 
                                "cluster-center-downweight"
                                ]="cluster-center",
                cluster_algorithm: Literal["kmeans", "meanshift"]="kmeans", 
                n_clusters=20, 
                quantile=0.8, n_samples=500,
                bandwidth: Union[float, None, 
                                Literal["auto"]
                                ]=None,
                norm_method: Literal["local", "global"]="local"
                ):
        self.img_list = img_list
        self.method = method
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.norm_method = norm_method
        
        
    def extract_img_features(self):
        embeddings = get_embeddings(img_list=self.img_list, 
                                    model_type="ViT-B/32"
                                    )
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        self.normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                       axis=1
                                       )
        return self.normalized_embedding
    
    def compute_rep(self, normalized_embedding=None):
        if not normalized_embedding:
            if hasattr(self, "normalized_embedding"):
                normalized_embedding = self.normalized_embedding
            else:
                normalized_embedding = self.extract_img_features()
        repr_score = _compute_representativeness(embeddings=normalized_embedding,
                                                 N=self.n_clusters,
                                                 method=self.method,
                                                 cluster_algorithm=self.cluster_algorithm,
                                                 quantile=self.quantile,
                                                 n_samples=self.n_samples,
                                                 bandwidth=self.bandwidth,
                                                 norm_method=self.norm_method
                                                 )

        self.img_representativeness_score = [(img, rep) for img, rep, in 
                                            zip(self.img_list, repr_score)
                                            ]
        self.img_representativeness_score
        return self.img_representativeness_score
        
    def sample(self, sample_ratio: float = 0.5, 
               img_representativeness_score=None
               ):
        if not img_representativeness_score:
            if hasattr(self, "img_representativeness_score"):
                img_representativeness_score = self.img_representativeness_score
            else:
                img_representativeness_score = self.compute_rep()
        img_repr_score_sorted = sorted(img_representativeness_score, 
                                       key=lambda x : x[1], 
                                       reverse=True
                                       )
        
        sample_end_index = int(sample_ratio * len(img_repr_score_sorted)) +1 

        self.sample_imgs = img_repr_score_sorted[:sample_end_index]
        return self.sample_imgs


class ImageUniqueness(object):
    def __init__(self, img_list: List[str], 
                 model_type: str ="ViT-B/32",
                 nearest_neighbour_influence: int =3,
                neighbour_weights: Union[List[float],None,
                                        ]= [0.6, 0.3, 0.1]
                ):
        self.img_list = img_list
        self.model_type = model_type
        self.nearest_neighbour_influence = nearest_neighbour_influence
        
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        self.neighbour_weights = neighbour_weights
        
    def extract_img_features(self):
        embeddings = get_embeddings(img_list=self.img_list, 
                                    model_type=self.model_type
                                    )
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        self.normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                       axis=1
                                       )
        return self.normalized_embedding

    def compute_uniqueness(self, nearest_neighbour_influence: Union[int, None]=None,
                           neighbour_weights: Union[List[float],None,
                                                   ]= None,
                           normalized_embedding = None
                           ):
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        else:
            neighbour_weights = self.neighbour_weights
        
        if not nearest_neighbour_influence:
            nearest_neighbour_influence = self.nearest_neighbour_influence
            
        nn = skn.NearestNeighbors(metric="cosine")
        if not normalized_embedding:
            if hasattr(self, "normalized_embedding"):
                normalized_embedding = self.normalized_embedding
            else:
                normalized_embedding = self.extract_img_features()
                
        nn.fit(X=normalized_embedding)
        dist, _ = nn.kneighbors(X=normalized_embedding, 
                                   n_neighbors=nearest_neighbour_influence + 1, 
                                   return_distance=True
                                   )
        
        if neighbour_weights:
            sample_dists = np.mean(dist[:, 1:] * neighbour_weights, axis=1)
        else:
            sample_dists = np.mean(dist[:, 1:], axis=1)

        sample_dists /= sample_dists.max()
        self.img_uniquness = [(img, uniq) for img, uniq 
                                in zip(self.img_list, sample_dists)
                                ]
        return self.img_uniquness
    
    def sample(self, sample_ratio: float = 0.5, 
               img_uniqueness=None
               ):
        assert sample_ratio <= 1.0, "sample_ratio cannot be greater than 1.0"
        if not img_uniqueness:
            if hasattr(self, "img_uniqueness"):
                img_uniqueness = self.img_uniquness
            else:
                img_uniqueness = self.compute_uniqueness()
        
        img_uniqueness_sorted = sorted(img_uniqueness, 
                                       key=lambda x : x[1], 
                                       reverse=True
                                       )        
        sample_end_index = int(sample_ratio * len(img_uniqueness_sorted)) + 1
        self.sample_imgs = img_uniqueness_sorted[:sample_end_index]
        return self.sample_imgs
        
