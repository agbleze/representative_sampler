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
    

class ImageUniqueness(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir 
    
    
    
    
from typing import List 
class RepresentativenessSampler(object):
    def __init__(self, img_list: List[str],
                method: Literal["cluster-center", 
                                "cluster-center-downweight"
                                ]="cluster-center",
                cluster_algorithm: Literal["kmeans", "meanshift"]="kmeans", 
                n_clusters=20, 
                quantile=0.8, n_samples=500,
                bandwidth: Union[float, None, 
                                Literal["auto"]
                                ]=None,
                norm_method: Literal["local", "global"]="local"
                ):
        self.img_list = img_list
        self.method = method
        self.cluster_algorithm = cluster_algorithm
        self.n_clusters = n_clusters
        self.quantile = quantile
        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.norm_method = norm_method
        
        
    def extract_img_features(self):
        embeddings = get_embeddings(img_list=img_list, 
                                    model_type="ViT-B/32"
                                    )
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        self.normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                       axis=1
                                       )
        return self.normalized_embedding
    
    def compute_rep(self, normalized_embedding=None):
        if not normalized_embedding:
            if hasattr(self, "normalized_embedding"):
                normalized_embedding = self.normalized_embedding
            else:
                normalized_embedding = self.extract_img_features()
        repr_score = _compute_representativeness(embeddings=normalized_embedding,
                                                 N=self.n_clusters,
                                                 method=self.method,
                                                 cluster_algorithm=self.cluster_algorithm,
                                                 quantile=self.quantile,
                                                 n_samples=self.n_samples,
                                                 bandwidth=self.bandwidth,
                                                 norm_method=self.norm_method
                                                 )

        self.img_representativeness_score = [(img, rep) for img, rep, in 
                                            zip(img_list, repr_score)
                                            ]
        self.img_representativeness_score
        return self.img_representativeness_score
        
    def sample(self, sample_ratio: float = 0.5, 
               img_representativeness_score=None
               ):
        if not img_representativeness_score:
            if hasattr(self, "img_representativeness_score"):
                img_representativeness_score = self.img_representativeness_score
            else:
                img_representativeness_score = self.compute_rep()
        img_repr_score_sorted = sorted(img_representativeness_score, 
                                       key=lambda x : x[1], 
                                       reverse=True
                                       )
        
        
        sample_end_index = int(sample_ratio * len(img_repr_score_sorted)) +1 

        self.sample_imgs = img_repr_score_sorted[:sample_end_index]
        return self.sample_imgs
