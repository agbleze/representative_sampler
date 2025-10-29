from abc import ABC, abstractmethod

class BaseImporter(object):
    importer_name: str = "base_importer"
    status: str = "experimental"
    valid_status_values = ["stable", "experimental"]
    required = ["importer_name", "status"]
    
    def __init_subclass__(self, cls):
        super().__init_subclass__()
    