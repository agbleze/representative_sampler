

from representative_sampler.core.domain.strategies.base_scorer import BaseScorer
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.domain.strategies.object_collections import ScoreCollection
from scipy.stats import entropy
import numpy as np


class FeatureEntropyScorer(BaseScorer):
    scorer_name = "feature_entropy_scorer"
    status = "experimental"
    
    def __init__(self, *args, **kwargs):
        pass
    
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:
        self.score_collection = self.compute_image_feature_entropy(embedding_obj=embedding_obj)
        return self.score_collection
    
    def compute_image_feature_entropy(self, embedding_obj: EmbeddingResult):
        embedding = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        if embedding.size == 1:
            image_entropy = entropy(embedding)
        else:
            image_entropy = np.array([entropy(emb) for emb in embedding])
            
        image_entropy_scores = [ScoringResult(object_name=embedding_nm, 
                                            score=score,
                                            scorer_name=self.scorer_name
                                            ) 
                                             for embedding_nm, score, in zip(embedding_names, image_entropy)
                                            ]
        return ScoreCollection(image_entropy_scores)
        

