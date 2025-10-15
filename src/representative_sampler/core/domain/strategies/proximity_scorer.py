from representative_sampler.core.domain.strategies.base_scorer import Scorer
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.domain.strategies.object_collections import ScoreCollection
from typing import List, Literal, Union
import numpy as np



class ProximityScorer(Scorer):
    scorer_name = "proximity_scorer"
    status = "experimental"
    
    def __init__(self, *args, **kwargs):
        pass
    
    def score(self, embeddings: EmbeddingResult, **kwargs) -> ScoreCollection:
        self.score_collection = self.compute_proximity_score(embeddings_obj=embeddings)
        return self.score_collection
        
    
    def compute_centroid(self, embeddings_obj, *args, **kwargs):
        embeddings = embeddings_obj.embedding
        centroid = embeddings.mean(axis=0)
        return centroid
    
    def compute_distance_to_centroid(self, embeddings, centroid, *args, **kwargs):
         # Get distance from each point to it's closest cluster center
        dist = np.linalg.norm(embeddings - centroid, axis=1)
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