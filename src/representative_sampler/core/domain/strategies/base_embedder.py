from abc import ABC, abstractmethod



class Embedder(ABC):
    name = "base_embedder"
    
    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, "name"):
            raise NotImplementedError(f"Subclassing {cls.__name__} requires definining a 'name' class attribute.")
    
    @abstractmethod
    def embed(self, img_list):
        raise NotImplementedError("Subclasses must implement the embed method.")