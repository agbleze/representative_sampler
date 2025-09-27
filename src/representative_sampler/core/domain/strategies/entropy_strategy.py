
from typing import List, Literal, Optional, Union
from sklearn.cluster import KMeans
import numpy as np
from ..entities import ClusterMetadata

class Entropy(EmbeddingExtractor):
    """
    Select items through clustering and choose them based on label entropy in each cluster.
    """
    
    def __init__(self, img_list: List[str], 
                 n_clusters: int,
                 model_type: str ="ViT-B/32",
                 sample_ratio: float = 0.5,
                ):
        super().__init__(img_list=img_list, model_type=model_type)
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.total_sample_size = int(len(img_list) * sample_ratio)
        
        if self.total_sample_size == n_clusters:
            raise ValueError(f"n_clusters {n_clusters} is same as total sample size and entropy is not needed for this. Ensure n_clusters is much lesser")

    def extract_img_features(self):
        self.normalized_embedding = self._extract_img_features()
        return self.normalized_embedding

    def base(self, #labels,
            in_cluster_sampling_strategy: Literal["random", "entropy"],
             alpha: Optional[float]=None,
            maximize_entropy: Union[bool, None] = True
             #ratio, num_centers, labels, database_keys, item_list, source
             ):
        if hasattr(self, "normalized_embedding"):
            normalized_embedding = self.normalized_embedding
        else:
            normalized_embedding = self.extract_img_features()
        kmeans = KMeans(n_clusters=self.n_clusters, #num_centers, 
                        random_state=0
                        )
        clusters = kmeans.fit_predict(normalized_embedding) #fit_predict(database_keys)

        cluster_ids, cluster_num_item_list = np.unique(clusters, return_counts=True)
        
        cluster_metadata = [ClusterMetadata(cluster_name=id, 
                                            cluster_size=count,
                                            cluster_indices=np.where(clusters == id)[0],                                            
                                            ) 
                            for id, count in zip(cluster_ids, cluster_num_item_list)
                            ]
        cluster_metadata = compute_cluster_entropy(cluster_metadata=cluster_metadata,
                                                    embeddings=normalized_embedding
                                                    )
        if alpha is None:
            cluster_metadata = compute_weights(cluster_metadata=cluster_metadata)
        else:
            print(f"alpha: {alpha}")
            cluster_metadata = normalize_entropy(cluster_metadata=cluster_metadata)
            cluster_metadata = compute_combine_weight(cluster_metadata=cluster_metadata,
                                                    alpha=alpha
                                                    )
        cluster_metadata = set_cluster_sampling_size(cluster_metadata=cluster_metadata,
                                                    total_sample_size=self.total_sample_size
                                                    ) 
        selected_imgs = self.sample(img_list=self.img_list,
                                    cluster_metadata=cluster_metadata,
                                    in_cluster_sampling_strategy=in_cluster_sampling_strategy,
                                    maximize_entropy=maximize_entropy
                                    )
        return selected_imgs
    
    def sample(self, img_list, cluster_metadata,
               in_cluster_sampling_strategy: Literal["random", "entropy"],
               maximize_entropy: Union[bool, None] = True
               ):
        if in_cluster_sampling_strategy not in ["random", "entropy"]:
            raise ValueError(f"{in_cluster_sampling_strategy} is not a valid value for in_cluster_sampling_strategy. in_cluster_sampling_strategy has to be one of random, entropy")
        
        if in_cluster_sampling_strategy == "entropy" and maximize_entropy == None:
            raise ValueError(f"maximize_entropy parameter is required when in_cluster_sampling_strategy is set to entropy")
        
        if in_cluster_sampling_strategy == "random":
            selected_items = _entropy_based_sampling(img_list=img_list, 
                                                    cluster_metadata=cluster_metadata
                                                    )
        elif in_cluster_sampling_strategy == "entropy":
            selected_items = _sample_with_image_entropy_score(img_list=img_list, 
                                                            cluster_metadata=cluster_metadata,
                                                            maximize_entropy=maximize_entropy
                                                            )
        return selected_items
        
        
        
        
class ImageEntropyScore(EmbeddingExtractor):
    def __init__(self, img_list: List[str], 
                n_clusters: int,
                model_type: str ="ViT-B/32",
                sample_ratio: float = 0.5,
                ):
        super().__init__(img_list=img_list, model_type=model_type)
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.total_sample_size = int(len(img_list) * sample_ratio)
        
    def _extract_img_features(self):
        normalized_embedding = super()._extract_img_features()
        return normalized_embedding
    
    def compute_image_feature_entropy(self, embedding: np.ndarray):
        if embedding.size == 1:
            image_entropy = entropy(embedding)
        else:
            image_entropy = np.array([entropy(emb) for emb in embedding])
        return image_entropy
        

