from abc import ABC, abstractmethod
from typing import List, Literal
from ..registry import registry


class Embedder(ABC):
    embedder_name = "base_embedder"
    status: Literal["stable", "experimental"] = "experimental"
    required = ["embedder_name", "status"]
    valid_status_values = ["stable", "experimental"]
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr in cls.required:
            if not hasattr(cls, attr):
                raise NotImplementedError(f"{cls.__name__} requires definining a '{attr}' class attribute because it is a Subclass of Embedder.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")
        if not hasattr(cls, "name"):
            raise NotImplementedError(f"Subclassing {cls.__name__} requires definining a 'name' class attribute.")
        registry.register(cls.embedder_name, cls, cls.status)
        
    @abstractmethod
    def get_embeddings(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the embed method.")