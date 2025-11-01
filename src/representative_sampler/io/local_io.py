from ..core.io.baseio import BaseImporter


class LocalImporter(BaseImporter):
    importer_name = "local_importer"
    status = "experimental"
    
    def __init__(self, folder, *args, **kwargs):
        pass
    
    def import_data(self, folder: str, *args, **kwargs):
        pass