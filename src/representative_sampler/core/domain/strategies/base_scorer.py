from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from abc import ABC, abstractmethod



@dataclass
class BaseScorer(ABC):
    name: str = "base_scorer"
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "name"):
            raise NotImplementedError(f"Subclassing {cls.__name__} requires definining a 'name' class attribute.")
    
    
    @abstractmethod
    def score(self, embeddings: np.ndarray, **kwargs) -> List[float]:
        raise NotImplementedError("Subclasses must implement the score method.")
    