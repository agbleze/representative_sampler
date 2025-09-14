from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune
import os
from typing import Literal, Union, List, Optional
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
from sklearn.cluster import KMeans
from scipy.stats import entropy
from dataclasses import dataclass, field
from copy import deepcopy    
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)

np.random.seed(0)

# create an abstract class sampler that other representative samplers subclass
# sample must have abstract methods - sample

class Sampler(ABC):
    
    @abstractmethod
    def sample(self):
        pass
    
    
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
    

# class ImageUniqueness(object):
#     def __init__(self, img_dir):
#         self.img_dir = img_dir 

@dataclass
class ClusterMetadata:
    cluster_name: int = field(default_factory=int)
    cluster_size: int = field(default_factory=int)
    cluster_entropy: float = field(default_factory=float)
    cluster_weight: float = field(default_factory=float)
    cluster_sampling_size: int = field(default_factory=int)
    cluster_indices: Union[np.ndarray, int] = field(default_factory=int)
    
    #def __post_init__(self):
        #self.cluster_sampling_size = 
        

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


class EmbeddingExtractor(object):
    def __init__(self, img_list, model_type):
        self.img_list = img_list
        self.model_type = model_type
        
    def _extract_img_features(self):
        embeddings = get_embeddings(img_list=self.img_list, 
                                    model_type=self.model_type
                                    )
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        self.normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                                    axis=1
                                                    )
        return self.normalized_embedding
    
    
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
        


# Adapted from datumaro
import math
import random
class Centroid(EmbeddingExtractor):
    """
    Select items through clustering with centers targeting the desired number.
    
    Number of items to subset are used as n_clusters and the centroid of each selected
    """
    def __init__(self, img_list: List[str], 
                 n_clusters: int,
                 model_type: str ="ViT-B/32",
                 
                ):
        super.__init__(img_list, model_type)
        self.img_list = img_list
        self.model_type = model_type
        self.n_clusters = n_clusters

    def extract_img_features(self):
        self.normalized_embedding = self._extract_img_features()
        
    def base(self, 
             #ratio, num_centers, labels, 
             #database_keys, item_list,
             ):

        #num_selected_centers = int(len(self.img_list) * ratio) #math.ceil(len(item_list) * ratio)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        clusters = kmeans.fit_predict(self.normalized_embedding) #(database_keys)
        cluster_centers = kmeans.cluster_centers_
        cluster_ids = np.unique(clusters)

        selected_items = []
        dist_tuples = []
        for cluster_id in cluster_ids:
            cluster_center = cluster_centers[cluster_id]
            cluster_items_idx = np.where(clusters == cluster_id)[0]
            num_selected_items = 1
            cluster_items = self.normalized_embedding[cluster_items_idx,] #database_keys[cluster_items_idx,]
            dist = calculate_hamming(cluster_center, cluster_items)
            ind = np.argsort(dist)
            item_idx_list = cluster_items_idx[ind]
            for i, idx in enumerate(item_idx_list[:num_selected_items]):
                selected_items.append(self.img_list[idx]) #(item_list[idx])
                # dist_tuples.append(
                #     (cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i])
                # )
        return selected_items#, dist_tuples


class ClusteredRandom(EmbeddingExtractor):
    """
    Select items through clustering and choose randomly within each cluster.
    """
    
    def __init__(self, img_list: List[str], 
                 n_clusters: int,
                 model_type: str ="ViT-B/32",
                 sample_ratio: float = 0.5,
                ):
        super.__init__(img_list, model_type)
        self.img_list = img_list
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio

    def extract_img_features(self):
        self.normalized_embedding = self._extract_img_features()
        
    def base(self, #ratio, num_centers, labels, 
             #database_keys, item_list, source
             ):
        if hasattr(self, "normalized_embedding"):
            normalized_embedding = self.normalized_embedding
        else:
            normalized_embedding = self.extract_img_features()

        kmeans = KMeans(n_clusters=self.n_clusters, #num_centers, 
                        random_state=0)
        clusters = kmeans.fit_predict(normalized_embedding) #(database_keys)
        cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

        norm_cluster_num_item_list = match_num_item_for_cluster(self.sample_ratio,
                                                                #ratio, 
                                                                len(normalized_embedding),#len(database_keys), 
                                                                cluster_num_item_list
                                                                )

        selected_items = []
        random.seed(0)
        for i, cluster_id in enumerate(cluster_ids):
            cluster_items_idx = np.where(clusters == cluster_id)[0]
            num_selected_items = norm_cluster_num_item_list[i]
            random.shuffle(cluster_items_idx)
            #selected_items.extend(item_list[idx] for idx in cluster_items_idx[:num_selected_items])
            selected_items.extend(self.img_list[idx] for idx in cluster_items_idx[:num_selected_items])
        return selected_items #, None


# class QueryClust(object):
#     """
#     Select items through clustering with inits that imply each label.
#     """

#     def base(self, ratio, num_centers, labels, database_keys, item_list, source):
#         from sklearn.cluster import KMeans

#         center_dict = {i: None for i in range(1, num_centers)}
#         for item in item_list:
#             for anno in item.annotations:
#                 if isinstance(anno, Label):
#                     label_ = anno.label
#                     if center_dict.get(label_) is None:
#                         center_dict[label_] = item
#             if all(center_dict.values()):
#                 break

#         item_id_list = [item.id.split("/")[-1] for item in item_list]
#         centroids = [
#             database_keys[item_id_list.index(item.id)] for item in center_dict.values() if item
#         ]
#         kmeans = KMeans(
#             n_clusters=num_centers, n_init=1, init=np.stack(centroids, axis=0), random_state=0
#         )

#         clusters = kmeans.fit_predict(database_keys)
#         cluster_centers = kmeans.cluster_centers_
#         cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)

#         norm_cluster_num_item_list = match_num_item_for_cluster(
#             ratio, len(database_keys), cluster_num_item_list
#         )

#         selected_items = []
#         dist_tuples = []
#         for i, cluster_id in enumerate(cluster_ids):
#             cluster_center = cluster_centers[cluster_id]
#             cluster_items_idx = np.where(clusters == cluster_id)[0]
#             num_selected_item = norm_cluster_num_item_list[i]

#             cluster_items = database_keys[cluster_items_idx]
#             dist = calculate_hamming(cluster_center, cluster_items)
#             ind = np.argsort(dist)
#             item_idx_list = cluster_items_idx[ind]
#             for i, idx in enumerate(item_idx_list[:num_selected_item]):
#                 selected_items.append(item_list[idx])
#                 dist_tuples.append(
#                     (cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i])
#                 )
#         return selected_items, dist_tuples


class Entropy(EmbeddingExtractor):
    """
    Select items through clustering and choose them based on label entropy in each cluster.
    """
    
    def __init__(self, img_list: List[str], 
                 n_clusters: int,
                 model_type: str ="ViT-B/32",
                 sample_ratio: float = 0.5,
                ):
        super().__init__(img_list=img_list, model_type=model_type)
        self.img_list = img_list
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.total_sample_size = int(len(img_list) * sample_ratio)

    def extract_img_features(self):
        self.normalized_embedding = self._extract_img_features()
        return self.normalized_embedding

    def base(self, #labels,
             alpha: Optional[float]=None,
             #ratio, num_centers, labels, database_keys, item_list, source
             ):
        if hasattr(self, "normalized_embedding"):
            normalized_embedding = self.normalized_embedding
        else:
            normalized_embedding = self.extract_img_features()
        kmeans = KMeans(n_clusters=self.n_clusters, #num_centers, 
                        random_state=0
                        )
        clusters = kmeans.fit_predict(normalized_embedding) #fit_predict(database_keys)

        cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)
        
        cluster_metadata = [ClusterMetadata(cluster_name=id, 
                                            cluster_size=count,
                                            cluster_indices=np.where(clusters == id)[0],                                            
                                            ) 
                            for id, count in zip(cluster_ids, cluster_num_item_list)
                            ]
        #print(cluster_metadata[0].cluster_indices)
        # for cl in cluster_metadata:
        #     print(cl.cluster_name)
        #     print(cl.cluster_indices)
        #     print(cl.cluster_size)
        # exit()
        
        cluster_metadata = compute_cluster_entropy(cluster_metadata=cluster_metadata,
                                                    embeddings=normalized_embedding
                                                    )
        if alpha is None:
            cluster_metadata = compute_weights(cluster_metadata=cluster_metadata)
        else:
            cluster_metadata = normalize_entropy(cluster_metadata=cluster_metadata)
            cluster_metadata = compute_combine_weight(cluster_metadata=cluster_metadata,
                                                    alpha=alpha
                                                    )
        cluster_metadata = set_cluster_sampling_size(cluster_metadata=cluster_metadata,
                                                    total_sample_size=self.total_sample_size
                                                    ) 
        selected_imgs = self.sample(img_list=self.img_list,
                               cluster_metadata=cluster_metadata
                               )
        return selected_imgs
    
    def sample(self, img_list, cluster_metadata):
        selected_items = _entropy_based_sampling(img_list=img_list, 
                                                cluster_metadata=cluster_metadata
                                                )
        return selected_items
        
        
        # norm_cluster_num_item_list = match_num_item_for_cluster(
        #                                                         #ratio, len(database_keys), 
        #                                                         self.sample_ratio, len(normalized_embedding),
        #                                                         cluster_num_item_list
        #                                                         )

        # selected_item_indexes = []
        # for cluster_id, num_selected_item in zip(cluster_ids, norm_cluster_num_item_list):
        #     cluster_items_idx = np.where(clusters == cluster_id)[0]

        #     cluster_classes = np.array(labels)[cluster_items_idx]
        #     _, inv, cnts = np.unique(cluster_classes, return_inverse=True, return_counts=True)
        #     weights = 1 / cnts
        #     probs = weights[inv]
        #     probs /= probs.sum()

        #     choices = np.random.choice(len(inv), size=num_selected_item, 
        #                                p=probs, replace=False
        #                                )
        #     selected_item_indexes.extend(cluster_items_idx[choices])

        # selected_items = np.array(item_list)[selected_item_indexes].tolist()
        # return selected_items #, None



def match_num_item_for_cluster(ratio, dataset_len, cluster_num_item_list):
    total_num_selected_item = int(dataset_len * ratio) #math.ceil(dataset_len * ratio)

    cluster_weights = np.array(cluster_num_item_list) / sum(cluster_num_item_list)
    norm_cluster_num_item_list = (cluster_weights * total_num_selected_item).astype(int)
    remaining_items = total_num_selected_item - sum(norm_cluster_num_item_list)

    if remaining_items > 0:
        zero_cluster_indexes = np.where(norm_cluster_num_item_list == 0)[0]
        add_clust_dist = np.sort(cluster_weights[zero_cluster_indexes])[::-1][:remaining_items]

        for dist in set(add_clust_dist):
            indices = np.where(cluster_weights == dist)[0]
            for index in indices:
                norm_cluster_num_item_list[index] += 1

    elif remaining_items < 0:
        diff_num_item_list = np.argsort(cluster_weights - norm_cluster_num_item_list)
        for diff_idx in diff_num_item_list[: abs(remaining_items)]:
            norm_cluster_num_item_list[diff_idx] -= 1

    return norm_cluster_num_item_list.tolist()



def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    return np.count_nonzero(B1 != B2, axis=1)


def match_query_subset(query_id, dataset, subset=None):
    if subset:
        return dataset.get(query_id, subset)

    subset_names = dataset.subsets().keys()
    for subset_name in subset_names:
        try:
            query_datasetitem = dataset.get(query_id, subset_name)
            if query_datasetitem:
                return query_datasetitem
        except Exception:
            pass
    return None


def compute_cluster_entropy(cluster_metadata: Union[List[ClusterMetadata], 
                                                    ClusterMetadata
                                                    ], 
                            embeddings
                            ):
    if isinstance(cluster_metadata, ClusterMetadata):
        cluster_metadata = [cluster_metadata]
        
    for clust_met in  cluster_metadata:
        cluster_embeddings = embeddings[clust_met.cluster_indices]
        cluster_centroid = cluster_embeddings.mean(axis=0),

        dists =  np.linalg.norm(cluster_embeddings - cluster_centroid, axis=1)  
        hist, _ = np.histogram(dists, bins=10, density=True)
        cluster_entropy = entropy(hist)
        setattr(clust_met, "cluster_entropy", cluster_entropy)
    if len(cluster_metadata) == 1:
        return cluster_metadata[0]
    else:
        return cluster_metadata


def compute_weights(cluster_metadata: List[ClusterMetadata]):
    entropies = np.array([clust.cluster_entropy for clust in cluster_metadata])
    total_entropy = entropies.sum()
    for clust in cluster_metadata:
        weight = clust.cluster_entropy / total_entropy
        setattr(clust, "cluster_weight", weight)
    return cluster_metadata


def get_sample_indices(embeddings, weights, cluster_entropies, sample_ratio=0.5):
    target_size = int(sample_ratio * len(embeddings))  # 50% reduction
    samples_per_cluster = (weights * target_size).astype(int)
    selected_indices = []
    for (k, _, idxs), n_samples in zip(cluster_entropies, samples_per_cluster):
        chosen = np.random.choice(idxs, size=min(n_samples, len(idxs)), replace=False)
        selected_indices.extend(chosen)



def normalize_entropy(cluster_metadata: List[ClusterMetadata]):
    raw_entropies = np.array([clust.cluster_entropy for clust in cluster_metadata])
    max_entropy = np.max(raw_entropies)
    for clust in cluster_metadata:
        normalized_entropy = clust.cluster_entropy / max_entropy
        setattr(clust, "cluster_entropy", normalized_entropy)
    
    return cluster_metadata


def compute_combine_weight(cluster_metadata: List[ClusterMetadata], 
                           alpha = 0.7, # controls how much entropy influences sampling                           
                           ):
    # use normalized entropy
    cluster_sizes = np.array([clust.cluster_size for clust in cluster_metadata])
    total_cluster_sizes = cluster_sizes.sum() 
    for clust in cluster_metadata:
        relative_cluster_size = clust.cluster_size / total_cluster_sizes
        weight = alpha * clust.entropy + (1 - alpha) * relative_cluster_size
        setattr(clust, "relative_cluster_size", relative_cluster_size)
        setattr(clust, "cluster_weight", weight)
    return cluster_metadata

def set_cluster_sampling_size(cluster_metadata: List[ClusterMetadata],
                              total_sample_size
                              ):
    for clust in cluster_metadata:
        cluster_sample_size = int(clust.cluster_weight * total_sample_size)
        setattr(clust, "cluster_sampling_size", cluster_sample_size)
        
        if cluster_sample_size > clust.cluster_size:
            setattr(clust, "sample_shortage", True)
        else:
            setattr(clust, "sample_shortage", False)
    
    return cluster_metadata


def _entropy_based_sampling(img_list: List[str], 
                            cluster_metadata: Union[List[ClusterMetadata], 
                                                    ClusterMetadata
                                                    ]
                            ):            
    _img_list = deepcopy(img_list)
    selected_imgs = []
    #remaining = total_sample_size
    shortage = 0
    _cluster_metadata = deepcopy(cluster_metadata)
    for clust in _cluster_metadata:
        if clust.sample_shortage == True:
            print(f"shortage in cluster {clust.cluster_name}")
            cluster_indices = clust.cluster_indices.tolist()
            imgs = [_img_list[i] for i in cluster_indices]
            selected_imgs.extend(imgs)
            #remaining -= clust.cluster_sampling_size
            shortage += clust.cluster_sampling_size - clust.cluster_size
            _cluster_metadata.remove(clust)
            
    if shortage > 0:
        print(f"total cluster shortage: {shortage}")
        # sample rest of samples from the largest cluster
        _cluster_sizes = np.array([clust.cluster_size for clust in _cluster_metadata])
        largest_cluster_index = np.argmax(_cluster_sizes)
        largest_cluster = _cluster_metadata[largest_cluster_index]
        print(f"largest cluster {largest_cluster.cluster_name} --- size {largest_cluster.cluster_size}")
        np.random.seed(0)
        seleted_shortage_indices = np.random.choice(largest_cluster.cluster_indices, 
                                                    size=shortage, replace=False
                                                    )
        _imgs = [_img_list[i] for i in seleted_shortage_indices]
        selected_imgs.extend(_imgs)
    
    for _clust in _cluster_metadata:
        print(f"cluster name {_clust.cluster_name} --- size {_clust.cluster_size} --- cluster sample size {_clust.cluster_sampling_size}")
        
        if _clust.cluster_size == _clust.cluster_sampling_size:
            _imgs = [_img_list[i] for i in _clust.cluster_indices]
            selected_imgs.extend(_imgs)
        else:
            selected_indices = np.random.choice(_clust.cluster_indices, size=_clust.cluster_sampling_size,
                                                replace=False
                                                )
            _imgs = [_img_list[i] for i in selected_indices]
            selected_imgs.extend(_imgs)
    return selected_imgs
        
        
        
