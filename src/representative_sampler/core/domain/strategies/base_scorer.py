from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from abc import ABC, abstractmethod
from typing import Literal
from representative_sampler.core.domain.strategies.registry import Registry

STABLE_SCORER_REGISTRY = Registry()
EXPERIMENTAL_SCORER_REGISTRY = Registry()


@dataclass
class BaseScorer(ABC):
    scorer_name: str = "base_scorer"
    status: Literal["stable", "experimental"] = "stable"
    required = ["name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of BaseScorer.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")

        if cls.status == "stable":
            STABLE_SCORER_REGISTRY.register(cls.scorer_name, cls)
        elif cls.status == "experimental":
            EXPERIMENTAL_SCORER_REGISTRY.register(cls.scorer_name, cls)
            
    @abstractmethod
    def score(self, embeddings: np.ndarray, **kwargs) -> List[float]:
        raise NotImplementedError("Subclasses must implement the score method.")
    
    