

from typing import List, Literal, Union
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
        embeddings = get_embeddings(img_list=self.img_list, 
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
                                            zip(self.img_list, repr_score)
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

