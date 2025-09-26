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
from representative_sampler.coco_annotation_utils import subset_coco_annotations
import shutil
import json
from sklearn.mixture import GaussianMixture
from pandas import json_normalize

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

def _sample_with_image_entropy_score(img_list: List[str], 
                                    cluster_metadata: Union[List[ClusterMetadata], 
                                                            ClusterMetadata
                                                            ],
                                    maximize_entropy=True
                                    ):
    selected_imgs = []
    for clust_met in cluster_metadata:
        clustimg_entropy = clust_met.cluster_images_entropy
        if maximize_entropy:
        # entropies are expected to sorted in ascending order
        # to get location of higest entroies we reverse the array
            clustimg_entropy_indices = np.argsort(clustimg_entropy)[::-1]
        else:
            clustimg_entropy_indices = np.argsort(clustimg_entropy)
        selected_entropy_indices = clustimg_entropy_indices[:clust_met.cluster_sampling_size + 1]
        clust_selected_imgs = np.array(img_list)[selected_entropy_indices].tolist()
        selected_imgs.extend(clust_selected_imgs)
    return selected_imgs

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
        cluster_centroid = cluster_embeddings.mean(axis=0)

        dists =  np.linalg.norm(cluster_embeddings - cluster_centroid, axis=1)  
        hist, _ = np.histogram(dists, bins=10, density=False)
        if hist.sum() == 0:
            raise ValueError(f"Entropy cannot be computed on 0")
        else:
            hist = hist / hist.sum()
            cluster_entropy = entropy(hist)
        #cluster_entropy = entropy(hist)
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

def compute_image_entropy_score(cluster_metadata: List[ClusterMetadata],
                                embeddings: np.ndarray
                                ):
    entropy_computer = ImageEntropyScore(img_list=[], n_clusters=0)
    for clust_met in cluster_metadata:
        cluster_embeddings = embeddings[clust_met.cluster_indices]
        clust_imgentropy = entropy_computer.compute_image_feature_entropy(embedding=cluster_embeddings)
        setattr(clust_met, "cluster_images_entropy", clust_imgentropy)
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
    print(np.isnan(raw_entropies).sum())
    print(f"raw_entropies: {raw_entropies}")
    print(f"max_entropy: {max_entropy}")
    for clust in cluster_metadata:
        normalized_entropy = clust.cluster_entropy / max_entropy
        print(f"normalized_entropy: {normalized_entropy}")
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
        weight = alpha * clust.cluster_entropy + (1 - alpha) * relative_cluster_size
        print(f"weight: {weight}")
        setattr(clust, "relative_cluster_size", relative_cluster_size)
        setattr(clust, "cluster_weight", weight)
    return cluster_metadata

def set_cluster_sampling_size(cluster_metadata: List[ClusterMetadata],
                              total_sample_size
                              ):
    for clust in cluster_metadata:
        print(f"clust.cluster_weight: {clust.cluster_weight}")
        print(f"total_sample_size: {total_sample_size}")
        print(f"clust.cluster_weight * total_sample_size: {clust.cluster_weight * total_sample_size}")
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
    shortage = 0
    _cluster_metadata = deepcopy(cluster_metadata)
    for clust in _cluster_metadata:
        if clust.sample_shortage == True:
            print(f"shortage in cluster {clust.cluster_name}")
            cluster_indices = clust.cluster_indices.tolist()
            imgs = [_img_list[i] for i in cluster_indices]
            selected_imgs.extend(imgs)
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
        print(f"cluster name {_clust.cluster_name} --- size {_clust.cluster_size} --- cluster sample size {_clust.cluster_sampling_size} -- entropy {_clust.cluster_entropy}")
        
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
        
        
class ImageEntropyScore(EmbeddingExtractor):
    def __init__(self, img_list: List[str], 
                n_clusters: int,
                model_type: str ="ViT-B/32",
                sample_ratio: float = 0.5,
                ):
        super().__init__(img_list=img_list, model_type=model_type)
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.total_sample_size = int(len(img_list) * sample_ratio)
        
    def _extract_img_features(self):
        normalized_embedding = super()._extract_img_features()
        return normalized_embedding
    
    def compute_image_feature_entropy(self, embedding: np.ndarray):
        if embedding.size == 1:
            image_entropy = entropy(embedding)
        else:
            image_entropy = np.array([entropy(emb) for emb in embedding])
        return image_entropy
        




def coco_annotation_to_df(coco_annotation_file):
    with open(coco_annotation_file, "r") as annot_file:
        annotation = json.load(annot_file)
    annotations_df = json_normalize(annotation, "annotations")
    annot_imgs_df = json_normalize(annotation, "images")
    annot_cat_df = json_normalize(annotation, "categories")
    annotations_images_merge_df = annotations_df.merge(annot_imgs_df, left_on='image_id', 
                                                        right_on='id',
                                                        suffixes=("_annotation", "_image"),
                                                        how="outer"
                                                        )
    annotations_imgs_cat_merge = annotations_images_merge_df.merge(annot_cat_df, left_on="category_id", right_on="id",
                                                                    suffixes=(None, '_categories'),
                                                                    how="outer"
                                                                    )
    all_merged_df = annotations_imgs_cat_merge[['id_annotation', 'image_id','category_id', 'bbox', 'area', 'segmentation', 'iscrowd',
                                'file_name', 'height', 'width', 'name', 'supercategory'
                                ]]
    all_merged_df.rename(columns={"name": "category_name",
                                  "height": "image_height",
                                  "width": "image_width"}, 
                         inplace=True
                         )
    all_merged_df.dropna(subset=["file_name"], inplace=True)
    return all_merged_df


from typing import Literal

def sample_rep_data(img_list,
                    destination_img_dir,
                    cluster_algorithm="meanshift",
                    bandwidth=0.9,
                    sample_ratio=0.5,
                    coco_annotation_file=None,
                    save_annotations_as=None,
                    scoring_method: Literal['cluster-center', 'cluster-center-downweight'] = "cluster-center",
                    kmeans_n_clusters: int = 20,
                    bandwidth_quantile: float = 0.8,
                    bandwidth_n_samples: int = 500,
                    norm_method: Literal['local', 'global'] = "local"
                    ):
    os.makedirs(destination_img_dir, exist_ok=True)
    save_ann_dir = os.path.dirname(save_annotations_as)
    os.makedirs(save_ann_dir, exist_ok=True)
    repsampler = RepresentativenessSampler(img_list=img_list,
                                            cluster_algorithm=cluster_algorithm,
                                            bandwidth=bandwidth,
                                            norm_method=norm_method,
                                            n_samples=bandwidth_n_samples,
                                            quantile=bandwidth_quantile,
                                            n_clusters=kmeans_n_clusters,
                                            method=scoring_method
                                            )
    samples_obj = repsampler.sample(sample_ratio=sample_ratio)

    selected_imgs = [obj[0] for obj in 
                    samples_obj
                    ]
    if coco_annotation_file:
        subset_coco_annotations(img_list=selected_imgs,
                                coco_annotation_file=coco_annotation_file,
                                save_annotations_as=save_annotations_as
                                )

    for img in selected_imgs:
        shutil.copy(img, destination_img_dir)
        
        
def export_data(coco_annotation_file, destination_img_dir,
                save_annotations_as, img_list
                ):
    os.makedirs(destination_img_dir, exist_ok=True)
    save_ann_dir = os.path.dirname(save_annotations_as)
    os.makedirs(save_ann_dir, exist_ok=True)
    if coco_annotation_file:
        subset_coco_annotations(img_list=img_list,
                                coco_annotation_file=coco_annotation_file,
                                save_annotations_as=save_annotations_as
                                )

    for img in img_list:
        shutil.copy(img, destination_img_dir)

