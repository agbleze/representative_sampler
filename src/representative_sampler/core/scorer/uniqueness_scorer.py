from representative_sampler.core.scorer.base_scorer import BaseScorer
from representative_sampler.core.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.object_collections import ScoreCollection
import sklearn.preprocessing as skp
import sklearn.neighbors as skn
import numpy as np
from typing import List, Union


class UniquenessScorer(BaseScorer):
    scorer_name = "uniqueness_scorer"
    status = "experimental"
    
    def __init__(self,
                 model_type: str ="ViT-B/32",
                 nearest_neighbour_influence: int =3,
                neighbour_weights: Union[List[float],None,
                                        ]= [0.6, 0.3, 0.1]
                ):
        self.model_type = model_type
        self.nearest_neighbour_influence = nearest_neighbour_influence
        
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        self.neighbour_weights = neighbour_weights
    
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:
        self.score_collection = self._compute_uniqueness(embedding_obj=embedding_obj,
                                                   nearest_neighbour_influence=self.nearest_neighbour_influence,
                                                   neighbour_weights=self.neighbour_weights
                                                   )
        return self.score_collection
    
    def _compute_uniqueness(self, embedding_obj: EmbeddingResult, # use normlalized embedding from embedding obj
                            nearest_neighbour_influence: Union[int, None]=None,
                           neighbour_weights: Union[List[float],None,
                                                   ]= None,
                           
                           ):
        embedding = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        else:
            neighbour_weights = self.neighbour_weights
        
        if not nearest_neighbour_influence:
            nearest_neighbour_influence = self.nearest_neighbour_influence
            
        nn = skn.NearestNeighbors(metric="cosine")
                
        nn.fit(X=embedding)
        dist, _ = nn.kneighbors(X=embedding, 
                                n_neighbors=nearest_neighbour_influence + 1, 
                                return_distance=True
                                )
        
        if neighbour_weights:
            sample_dists = np.mean(dist[:, 1:] * neighbour_weights, axis=1)
        else:
            sample_dists = np.mean(dist[:, 1:], axis=1)

        sample_dists /= sample_dists.max()
        uniquness_scores = [ScoringResult(scorer_name=self.scorer_name,
                                            object_name=nm, 
                                            score=score
                                            ) 
                              for nm, score in zip(embedding_names, sample_dists)
                            ]
        return ScoreCollection(uniquness_scores)
    
    
            
            