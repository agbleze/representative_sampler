from representative_sampler.core.domain.strategies.base_scorer import Scorer
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.domain.strategies.object_collections import ScoreCollection
from typing import List, Literal, Union



class ProximityScorer(Scorer):
    scorer_name = "proximity_scorer"
    status = "experimental"
    
    def __init__(self, *args, **kwargs):
        pass
    
    def score(self):
        pass