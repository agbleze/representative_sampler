from .base_scorer import BaseScorer
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.strategies.object_collections import ScoreCollection
from typing import List, Literal, Union
import numpy as np
import copy
import sklearn.cluster as skc
from scipy.spatial import cKDTree
import logging
import logger


class RepresentativeScorer(BaseScorer):
    scorer_name = "representative_scorer"
    status = "stable"
    valid_methods = ["cluster-center", "cluster-center-downweight"]
    valid_cluster_algorithms = ["meanshift", "kmeans"]
    valid_norm_methods = ["local", "global"]
    
    def __init__(self, 
                 method: Literal["cluster-center", 
                                "cluster-center-downweight"
                                ]="cluster-center",
                cluster_algorithm="kmeans", 
                n_clusters=20, 
                quantile=0.8, n_samples=500,
                bandwidth: Union[float, None, 
                                Literal["auto"]
                                ]=None,
                norm_method: Literal["local", "global"]="local"
                ):
        if method not in self.valid_methods:
            raise ValueError(f"Invalid method '{method}'. Valid methods are {self.valid_methods}.")
        if cluster_algorithm not in self.valid_cluster_algorithms:
            raise ValueError(f"Invalid cluster_algorithm '{cluster_algorithm}'. Valid algorithms are {self.valid_cluster_algorithms}.")
        self.method = method
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.norm_method = norm_method
    
    def score(self, embeddings, **kwargs) -> ScoreCollection:
        self.score_collection = self.compute_representativeness_score(embedding_result=embeddings)
        return self.score_collection
    
    
    def compute_representativeness_score(self, embedding_result: EmbeddingResult) -> ScoreCollection:
        embedding_names = embedding_result.embedding_name
        embedding = embedding_result.embedding
        
        repr_scores = _compute_representativeness(embeddings=embedding,
                                                 N=self.n_clusters,
                                                 method=self.method,
                                                 cluster_algorithm=self.cluster_algorithm,
                                                 quantile=self.quantile,
                                                 n_samples=self.n_samples,
                                                 bandwidth=self.bandwidth,
                                                 norm_method=self.norm_method
                                                 )

        self.img_representativeness_score = [ScoringResult(object_name=embedding_nm, 
                                                           score=repscore,
                                                           scorer_name=self.scorer_name
                                                           ) 
                                             for embedding_nm, repscore, in zip(embedding_names, repr_scores)
                                            ]
        self.img_representativeness_score
        return ScoreCollection(self.img_representativeness_score)
    


# Adapted from     
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

