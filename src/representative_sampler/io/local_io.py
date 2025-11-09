from ..core.io.baseio import BaseImporter
import shutil
import os
from representative_sampler import logger
from representative_sampler.core.entities import SamplingResult
from glob import glob
import concurrent.futures
import itertools


def copy_to_dest_dir(sample: SamplingResult, dest_dir: str):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.copy(sample.file_path, dest_dir)
    logger.info(f"Copied {sample.file_path} to {dest_dir}") 

def multiprocess_copy(sample_collection, dest_dir: str, num_workers: int = 4):
    cpu_count = os.cpu_count()
    logger.info(f"max_workers will be set to {cpu_count}")
    logger.info(f"started copying samples to {dest_dir}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        executor.map(copy_to_dest_dir, sample_collection, 
                     itertools.repeat(dest_dir)
                     )
    logger.info(f"Finished copying samples to {dest_dir}")
    
    
class LocalImporter(BaseImporter):
    importer_name = "local_importer"
    status = "experimental"
    
    def __init__(self, folder, *args, **kwargs):
        pass
    
    def import_data(self, folder: str, *args, **kwargs):
        pass