from representative_sampler.core.domain.strategies.base_sampler import Sampler


class GenericSampler(Sampler):
    sampler_name = "generic_sampler"
    status = "experimental"
    
    def __init__(self, sample_ratio: float = 0.1):
        self.sample_ratio = sample_ratio
        
    def sample(self, sample_collection):
        pass
    
    