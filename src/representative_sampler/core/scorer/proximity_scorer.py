from representative_sampler.core.scorer.base_scorer import Scorer
from representative_sampler.core.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.object_collections import ScoreCollection
from typing import List, Literal, Union
import numpy as np
from representative_sampler.core.scorer.representative_scorer import RepresentativeScorer
from ..utils.utils import get_cls_init_params


class ProximityScorer(Scorer):
    scorer_name = "proximity_scorer"
    status = "experimental"
    
    def __init__(self, alpha=None, ensemble=False, *args, **kwargs):
        if alpha is not None:
            assert 0 <= alpha <= 1, "alpha must be between 0 and 1"
        if ensemble:
            assert alpha is not None, "alpha must be provided when ensemble is True"
            
        self.alpha = alpha
        self.ensemble = ensemble
        represcorer_init_params = get_cls_init_params(RepresentativeScorer)
        represcorer_kwargs = {k:v for k,v in kwargs.items() if k in represcorer_init_params}
        if represcorer_kwargs:
            self.representative_scorer = RepresentativeScorer(**represcorer_kwargs)
    
    def score(self, embeddings: EmbeddingResult, **kwargs) -> ScoreCollection:
        if not self.ensemble:
            self.score_collection = self.compute_proximity_score(embeddings_obj=embeddings)
            return self.score_collection
        else:
            self.score_collection = self.compute_ensemble_score(embeddings_obj=embeddings)
            return self.score_collection
        
    def compute_ensemble_score(self, embeddings_obj: EmbeddingResult, *args, **kwargs) -> ScoreCollection:
        representative_scores = self.representative_scorer.score(embeddings=embeddings_obj)
        proximity_scores = self.compute_proximity_score(embeddings_obj=embeddings_obj)
        repre_weight = 1- self.alpha
        combine_scores = []
        for proximity_item in proximity_scores:
            item_name = proximity_item.object_name
            rep_score_item = next((rep_item for rep_item in representative_scores if rep_item.object_name == item_name), None)
            if rep_score_item:
                ensembel_score = (self.alpha * proximity_item.score) + (repre_weight * rep_score_item.score) 
                combine_scores.append(ScoringResult(object_name=item_name,
                                                    score=ensembel_score,
                                                    scorer_name=self.scorer_name
                                                    )
                                      )
        return ScoreCollection(combine_scores)
    
    
    def compute_centroid(self, embeddings_obj, *args, **kwargs):
        embeddings = embeddings_obj.embedding
        centroid = embeddings.mean(axis=0)
        return centroid
    
    def compute_distance_to_centroid(self, embeddings, centroid, *args, **kwargs):
         # Get distance from each point to it's closest cluster center
        dist = np.linalg.norm(embeddings - centroid, axis=1)
        dist /= np.max(dist)  # normalize to [0,1]
        return dist
    
    def compute_proximity_score(self, embeddings_obj: EmbeddingResult, *args, **kwargs) -> ScoreCollection:
        embedding_names = embeddings_obj.embedding_name
        embeddings = embeddings_obj.embedding
        centroid = self.compute_centroid(embeddings_obj=embeddings_obj)
        distances = self.compute_distance_to_centroid(embeddings=embeddings, centroid=centroid)
        
        proximity_scores = [ScoringResult(object_name=embedding_nm, 
                                            score=score,
                                            scorer_name=self.scorer_name
                                            ) 
                                             for embedding_nm, score, in zip(embedding_names, distances)
                                            ]
        return ScoreCollection(proximity_scores)