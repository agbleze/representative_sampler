from representative_sampler.core.domain.strategies.base_sampler import Sampler
from representative_sampler.core.domain.strategies.object_collections import SampleCollection, ScoreCollection
from typing import List, Literal
import numpy as np

_SENTINEL = object()

class GenericSampler(Sampler):
    sampler_name = "generic_sampler"
    status = "experimental"
    valid_modes = ["min", "max"]
    
    def __init__(self, sample_ratio: float = 0.1,
                 mode: Literal["min", "max"]="max"
                 ):
        if mode not in self.valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are {self.valid_modes}.")
        self.sample_ratio = sample_ratio
        self.mode = mode
        
    def sample(self, score_collection: ScoreCollection,
               sample_ratio: float = _SENTINEL
               ) -> SampleCollection:
        if sample_ratio is _SENTINEL:
            sample_ratio = self.sample_ratio
            
        scores_sorted = sorted(score_collection, 
                                key=lambda x : x.score, 
                                reverse=True if self.mode=="max" else False
                                )
        
        sample_end_index = int(sample_ratio * len(scores_sorted)) +1 

        samples = scores_sorted[:sample_end_index]
        return SampleCollection(samples)
    