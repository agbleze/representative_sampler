from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.algorithms.hash_key_inference.prune import Prune
from datumaro.plugins.validators import DetectionValidator, SegmentationValidator
import os
from typing import Literal, Union

def sample_data(img_dir: str, 
                cluster_method: Literal["cluster_random", 
                                        "query_clust", 
                                        "centroid",
                                        "entropy",
                                        "ndr",
                                        "random"
                                        ]="cluster_random", 
                reduce_proportion: float=0.5,
                output_dir: str = "samples", 
                save_format: str= "coco_instances",
                **kwargs
                ):
    """_summary_

    Args:
        img_dir (str): Directory containing images
        cluster_method (str, optional): Clustering method. Defaults to "cluster_random".
        reduce_proportion (float, optional): Ratio of data to select (0 - 1). Defaults to 0.5.
        output_dir (str, optional): Directory to store selected data. Defaults to "samples".
        save_format (str, optional): format to save data. Defaults to "coco_instances".
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    env = Environment()
    detected_format = env.detect_dataset(path=img_dir)
    dataset = Dataset.import_from(img_dir, detected_format[0])
    prune = Prune(dataset, cluster_method=cluster_method)
    cluster_random_result = prune.get_pruned(ratio=reduce_proportion)
    cluster_random_result.export(output_dir, format=save_format, save_media=True)