from typing import List, Optional
import numpy as np
from abc import ABC, abstractmethod
from typing import Literal
from ..registry import registry


class BaseScorer(ABC):
    scorer_name: str = "base_scorer"
    status: Literal["stable", "experimental"] = "stable"
    required = ["scorer_name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if attr not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of BaseScorer.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")

        registry.register(cls.scorer_name, cls, cls.status)
        
    @abstractmethod
    def score(self, embeddings: np.ndarray, **kwargs) -> List[float]:
        raise NotImplementedError("Subclasses must implement the score method.")
    
    