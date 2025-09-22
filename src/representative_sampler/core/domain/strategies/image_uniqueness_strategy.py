

from typing import List, Union

class ImageUniqueness(object):
    def __init__(self, img_list: List[str], 
                 model_type: str ="ViT-B/32",
                 nearest_neighbour_influence: int =3,
                neighbour_weights: Union[List[float],None,
                                        ]= [0.6, 0.3, 0.1]
                ):
        self.img_list = img_list
        self.model_type = model_type
        self.nearest_neighbour_influence = nearest_neighbour_influence
        
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        self.neighbour_weights = neighbour_weights
        
    def extract_img_features(self):
        embeddings = get_embeddings(img_list=self.img_list, 
                                    model_type=self.model_type
                                    )
        embeddings_cpu = [emb.cpu() for emb in embeddings]
        embeddings_cpu_concat = np.concatenate(embeddings_cpu)
        self.normalized_embedding = skp.normalize(embeddings_cpu_concat, 
                                       axis=1
                                       )
        return self.normalized_embedding

    def compute_uniqueness(self, nearest_neighbour_influence: Union[int, None]=None,
                           neighbour_weights: Union[List[float],None,
                                                   ]= None,
                           normalized_embedding = None
                           ):
        if neighbour_weights:
            assert len(neighbour_weights) == nearest_neighbour_influence, "neighbour_weights must be of equal length as num_sample_influence"
            assert round(sum(neighbour_weights),1) == 1, "neighbour_weights must sum up to 1"
        else:
            neighbour_weights = self.neighbour_weights
        
        if not nearest_neighbour_influence:
            nearest_neighbour_influence = self.nearest_neighbour_influence
            
        nn = skn.NearestNeighbors(metric="cosine")
        if not normalized_embedding:
            if hasattr(self, "normalized_embedding"):
                normalized_embedding = self.normalized_embedding
            else:
                normalized_embedding = self.extract_img_features()
                
        nn.fit(X=normalized_embedding)
        dist, _ = nn.kneighbors(X=normalized_embedding, 
                                   n_neighbors=nearest_neighbour_influence + 1, 
                                   return_distance=True
                                   )
        
        if neighbour_weights:
            sample_dists = np.mean(dist[:, 1:] * neighbour_weights, axis=1)
        else:
            sample_dists = np.mean(dist[:, 1:], axis=1)

        sample_dists /= sample_dists.max()
        self.img_uniquness = [(img, uniq) for img, uniq 
                                in zip(self.img_list, sample_dists)
                                ]
        return self.img_uniquness
    
    def sample(self, sample_ratio: float = 0.5, 
               img_uniqueness=None
               ):
        assert sample_ratio <= 1.0, "sample_ratio cannot be greater than 1.0"
        if not img_uniqueness:
            if hasattr(self, "img_uniqueness"):
                img_uniqueness = self.img_uniquness
            else:
                img_uniqueness = self.compute_uniqueness()
        
        img_uniqueness_sorted = sorted(img_uniqueness, 
                                       key=lambda x : x[1], 
                                       reverse=True
                                       )        
        sample_end_index = int(sample_ratio * len(img_uniqueness_sorted)) + 1
        self.sample_imgs = img_uniqueness_sorted[:sample_end_index]
        return self.sample_imgs
        
