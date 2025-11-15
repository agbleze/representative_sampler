from ..core.io.baseio import BaseImporter, BaseExporter
import shutil
import os
from representative_sampler import logger
from representative_sampler.core.entities import SamplingResult
from representative_sampler.core.object_collections import SampleCollection
from glob import glob
import concurrent.futures
import itertools

_SENTINEL = object()


def copy_to_dest_dir(sample: SamplingResult, dest_dir: str):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    shutil.copy(sample.file_path, dest_dir)
    logger.info(f"Copied {sample.file_path} to {dest_dir}") 

def multiprocess_copy(sample_collection, dest_dir: str,):
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
    
    def __init__(self, directory, *args, **kwargs):
        if not os.path.exists(directory):
            msg = f"Provided directory does not exist: {directory}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        self.directory = directory
        logger.info(f"{self.importer_name} initialized with directory: {self.directory}")
    
    def import_data(self, *args, **kwargs):
        logger.info(f"{self.importer_name}: Importing data from folder: {self.directory}")
        img_list = glob(f"{self.directory}/*")
        logger.info(f"{self.importer_name}: Found {len(img_list)} images in {self.directory}")
        logger.info(f"{self.importer_name}: Data import successfully completed.")
        return img_list
    
    
class LocalExporter(BaseExporter):
    exporter_name = "local_exporter"
    status = "experimental"
    
    def __init__(self, directory, 
                validity_check: bool = True,
                coco_annotation_file: str = None,
                 *args, **kwargs):
        self.directory = directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.validity_check = validity_check
        
        if coco_annotation_file:
            msg = f"coco_annotation_file: {coco_annotation_file} not found."
            if not os.path.exists(coco_annotation_file):
                logger.error(msg)
                raise FileNotFoundError(msg)
            else:
                self.coco_annotation_file = coco_annotation_file
        else:
            self.coco_annotation_file = coco_annotation_file
        logger.info(f"{self.exporter_name} initialized with output directory: {self.output_dir}")
    
    def export_data(self, samples: SampleCollection, 
                    validity_check: bool = _SENTINEL,
                    *args, **kwargs
                    ):
        if validity_check is _SENTINEL:
            validity_check = self.validity_check
        if validity_check:
            samples.validity_check()
            
        logger.info(f"{self.exporter_name}: Exporting {len(samples)} samples to: {self.directory}")
        multiprocess_copy(sample_collection=samples, 
                          dest_dir=self.directory,
                          )
        logger.info(f"{self.exporter_name}: Data export successfully completed.")