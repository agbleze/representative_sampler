from abc import ABC, abstractmethod
from representative_sampler.core.domain.entities import Sample
from typing import List, Literal, Union


class Sampler(ABC):
    
    @abstractmethod
    def sample(self):
        pass
    
    
class SamplingStrategy(ABC):
    name: str
    
    @abstractmethod
    def sample(self, items: List[Sample], k: int) -> List[Sample]:
        pass
    