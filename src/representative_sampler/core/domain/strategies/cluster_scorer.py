

from representative_sampler.core.domain.strategies.base_scorer import BaseScorer
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.domain.strategies.object_collections import ScoreCollection
from typing import List, Literal, Union
import numpy as np
from sklearn.cluster import KMeans
import copy
import sklearn.cluster as skc
from scipy.spatial import cKDTree
import logging
from scipy.stats import entropy

from ..entities import ClusterMetadata

class ClusterScorer(BaseScorer):
    scorer_name = "cluster_scorer"
    status = "experimental"
    
    
    def __init__(self, alpha,
                 n_clusters=20,
                 *args, **kwargs):
        self.alpha = alpha
        self.n_clusters = n_clusters
        
    def score(self, embeddings: EmbeddingResult, **kwargs) -> ScoreCollection:
        self.score_collection = self.compute_cluster_entropy_score(embedding_result=embeddings)
        return self.score_collection
    
    
    def compute_cluster_entropy_score(self, embedding_result: EmbeddingResult,
                                      alpha: float):
        embedding = embedding_result.embedding
        embedding_names = embedding_result.embedding_name
        
        kmeans = KMeans(n_clusters=self.n_clusters,
                        random_state=0
                        )
        clusters = kmeans.fit_predict(embedding) 
        cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)
        
        cluster_metadata = [ClusterMetadata(cluster_name=id, 
                                            cluster_size=count,
                                            cluster_indices=np.where(clusters == id)[0], 
                                            name=id                                           
                                            ) 
                            for id, count in zip(cluster_ids, cluster_num_item_list)
                            ]
        
        cluster_metadata = [setattr(clust, "cluster_items", [embedding_names[idx] for idx 
                                                            in clust.cluster_indices
                                                            ]
                                    )
                            for clust in cluster_metadata
                            ]
        cluster_metadata = [setattr(clust, "population", embedding_names) 
                            for clust in cluster_metadata
                            ]
        
        # for id, count in zip(cluster_ids, cluster_num_item_list):
        #     score_res = ScoringResult(object_name=id, 
        #                             score=None,
        #                             scorer_name=self.scorer_name
        #                             )
        #     setattr(score_res, "cluster_size", count)
        #     setattr(score_res, "cluster_indices", np.where(clusters == id)[0])
        #     setattr(score_res, "cluster_name", id)
            
            
            
        cluster_metadata = compute_cluster_entropy(cluster_metadata=cluster_metadata,
                                                    embeddings=embedding
                                                    )
        if alpha is None:
            cluster_metadata = compute_weights(cluster_metadata=cluster_metadata)
        else:
            print(f"alpha: {alpha}")
            cluster_metadata = normalize_entropy(cluster_metadata=cluster_metadata)
            cluster_metadata = compute_combine_weight(cluster_metadata=cluster_metadata,
                                                    alpha=alpha
                                                    )
            
        score_col = [ScoringResult(object_name=clust.cluster_name, 
                       score=clust.cluster_entropy if hasattr(clust, "cluster_entropy") else None,
                       scorer_name=self.scorer_name,
                       metadata=clust
                       ) 
                        
                    for clust in cluster_metadata
                    ]
        return ScoreCollection(score_col)
         
            
        # cluster_metadata = set_cluster_sampling_size(cluster_metadata=cluster_metadata,
        #                                             total_sample_size=self.total_sample_size
        #                                             )

        
    
    








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


       