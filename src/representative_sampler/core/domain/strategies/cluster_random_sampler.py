

from representative_sampler.core.domain.strategies.base_sampler import Sampler
from representative_sampler.core.domain.strategies.object_collections import SampleCollection, ScoreCollection
from representative_sampler.core.domain.entities import SamplingResult
from typing import Union, List, Literal
import numpy as np
import random
import os


_SENTINEL = object()

class ClusterRandomSampler(Sampler):
    sampler_name = "cluster_random_sampler"
    status = "experimental"
    
    
    def __init__(self, sample_ratio: float = 0.5,
                 validity_check: bool = True,
                 seed: int = 123,
                 *args, **kwargs
                 ):
        self.validity_check = validity_check
        self.sample_ratio = sample_ratio
        self.seed = seed
        
        
    def sample(self, score_collection: ScoreCollection,
               seed: int = _SENTINEL
               ):
        if seed is _SENTINEL:
            seed = self.seed
            
        np.random.seed(seed)
        random.seed(seed)
        
        if self.validity_check:
            score_collection.validity_check()
            
        cluster_metadata = [sc.metadata for sc in score_collection]
        
        sampled_items = []
        for clust in cluster_metadata:
            cluster_sampling_size = int(self.sample_ratio * clust.cluster_size)
            cluster_items = clust.cluster_items
            random.shuffle(clust.cluster_items)
            selected_items = np.random.choice(cluster_items, size=cluster_sampling_size, 
                                              replace=False
                                              )
            sampled_items.extend(selected_items)
        samples = [SamplingResult(name=os.path.basename(img),
                               file_path=img)
                    for img in sampled_items
                    ]
        return SampleCollection(samples)
 
