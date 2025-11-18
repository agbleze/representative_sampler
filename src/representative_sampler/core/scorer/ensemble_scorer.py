from .base_scorer import Scorer
from representative_sampler.core.entities import EmbeddingResult, ScoringResult
from representative_sampler.core.object_collections import ScoreCollection
from typing import List, Literal, Union
import numpy as np


class EnsembelScorer(Scorer):
    scorer_name = "ensembel_scorer"
    status = "experimental"
    
    def __init__(self, scorer_names: List, weights: List,
                 *args, **kwargs
                 ):
        pass
    
    def score(self, embeddings: EmbeddingResult, **kwargs) -> ScoreCollection:
        pass