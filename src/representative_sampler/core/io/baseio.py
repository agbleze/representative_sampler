from abc import ABC, abstractmethod
from ..registry import registry


class BaseImporter(object):
    importer_name: str = "base_importer"
    status: str = "stable"
    valid_status_values = ["stable", "experimental"]
    required = ["importer_name", "status"]
    
    def __init_subclass__(self, cls):
        super().__init_subclass__()
        
        for nm in cls.required:
            if nm not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{nm}' class attribute because it is a Subclass of BaseImporter.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")
        registry.register(cls.importer_name, cls, cls.status)
      
        
    @abstractmethod
    def import_data(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the import_data method.")


class BaseExporter(object):
    exporter_name: str = "base_exporter"
    status: str = "stable"
    valid_status_values = ["stable", "experimental"]
    required = ["exporter_name", "status"]
    
    def __init_subclass__(self, cls):
        super().__init_subclass__()
        
        for nm in cls.required:
            if nm not in cls.__dict__:
                raise NotImplementedError(f"{cls.__name__} requires definining a '{nm}' class attribute because it is a Subclass of BaseExporter.")
        if cls.status not in cls.valid_status_values:
            raise ValueError(f"{cls.__name__} has invalid status '{cls.status}'. Valid values are {cls.valid_status_values}.")
        
    @abstractmethod
    def export_data(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the export_data method.")