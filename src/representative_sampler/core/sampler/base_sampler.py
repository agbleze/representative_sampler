from abc import ABC, abstractmethod
from representative_sampler.core.domain.entities import Sample
from typing import List, Literal, Union
from representative_sampler.core.domain.strategies.registry import Registry
from ..registry import registry

#SAMPLER_REGISTRY = Registry()


class Sampler(ABC):
    sampler_name = "base_sampler"
    status: Literal["stable", "experimental"] = "experimental"
    #version: str = "0.1.0"
    required = ["sampler_name", "status"] #, "version"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} requires definining a 'name' class attribute because it is a Subclass of Sampler.")
        
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")
        
        registry.register(cls.sampler_name, cls, cls.status)

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the sample method.")
 
 
    
    
class SamplingStrategy(ABC):
    name: str
    
    @abstractmethod
    def sample(self, items: List[Sample], k: int) -> List[Sample]:
        pass
    