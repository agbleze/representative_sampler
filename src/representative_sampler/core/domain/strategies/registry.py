

class Registry:
    
    def __init__(self):
        self._registry = {}
        
    def register(self, name: str, cls: type):
        if name in self._registry:
            raise ValueError(f"{name} is already registered.")
        
        self._registry[name] = cls
        
    def get(self, name: str):
        if name not in self._registry:
            raise ValueError(f"{name} is not in registry.")
        return self._registry[name]
    
    def list_available(self):
        return set(self._registry.keys())
        