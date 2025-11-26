from representative_sampler.core.sampler.base_sampler import Sampler
from representative_sampler.core.object_collections import SampleCollection, ScoreCollection
from representative_sampler.core.entities import EmbeddingResult, ScoringResult, SamplingResult
from typing import Union, List, Literal
from copy import deepcopy
import os
from pathlib import Path
from ..entities import ClusterMetadata


class ClusterEntropySampler(Sampler):
    sampler_name = "cluster_entropy_sampler"
    status = "experimental"
    
    def __init__(self,
                 in_cluster_sampling_strategy,
                 maximize_entropy,
                 sample_ratio: float = 0.5,
                 validity_check: bool = True,
                 *args, **kwargs
                 ):
        self.validty_check = validity_check
        self.sample_ratio = sample_ratio
        self.in_cluster_sampling_strategy = in_cluster_sampling_strategy
        self.maximize_entropy = maximize_entropy
    
    def sample(self, score_collection: ScoreCollection,
               ) -> SampleCollection:
        if self.validity_check:
            score_collection.validity_check()
        cluster_metadata = [sc.metadata for sc in score_collection]
        total_population = len(cluster_metadata[0].population)
        total_sample_size = int(self.sample_ratio * total_population)
        selected_imgs = self._sample(#img_list=self.img_list,
                                    cluster_metadata=cluster_metadata,
                                    in_cluster_sampling_strategy=self.in_cluster_sampling_strategy,
                                    maximize_entropy=self.maximize_entropy,
                                    total_sample_size=total_sample_size
                                    )
        samples = [SamplingResult(name=os.path.basename(img),
                               file_path=img)
                    for img in selected_imgs
                    ]
        return SampleCollection(samples)
    
    
    def _sample(self,
                #img_list, 
                cluster_metadata,
                total_sample_size,
               in_cluster_sampling_strategy: Literal["random", "entropy"],
               maximize_entropy: Union[bool, None] = True,
               
               ):
        cluster_metadata = set_cluster_sampling_size(cluster_metadata=cluster_metadata,
                                                    total_sample_size=total_sample_size
                                                    )
        
        if in_cluster_sampling_strategy not in ["random", "entropy"]:
            raise ValueError(f"{in_cluster_sampling_strategy} is not a valid value for in_cluster_sampling_strategy. in_cluster_sampling_strategy has to be one of random, entropy")
        
        if in_cluster_sampling_strategy == "entropy" and maximize_entropy == None:
            raise ValueError(f"maximize_entropy parameter is required when in_cluster_sampling_strategy is set to entropy")
        
        if in_cluster_sampling_strategy == "random":
            selected_items = _entropy_based_sampling(#img_list=img_list, 
                                                    cluster_metadata=cluster_metadata
                                                    )
        elif in_cluster_sampling_strategy == "entropy":
            selected_items = _sample_with_image_entropy_score(#img_list=img_list, 
                                                            cluster_metadata=cluster_metadata,
                                                            maximize_entropy=maximize_entropy
                                                            )
        return selected_items
        
        


def _sample_with_image_entropy_score(#img_list: List[str], 
                                    cluster_metadata: Union[List[ClusterMetadata], 
                                                            ClusterMetadata
                                                            ],
                                 