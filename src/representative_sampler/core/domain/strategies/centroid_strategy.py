import math
import random
from typing import List, Literal, Union
import numpy as np


class Centroid(EmbeddingExtractor):
    """
    Select items through clustering with centers targeting the desired number.
    
    Number of items to subset are used as n_clusters and the centroid of each selected
    """
    def __init__(self, img_list: List[str], 
                 n_clusters: int,
                 model_type: str ="ViT-B/32",
                 
                ):
        super.__init__(img_list, model_type)
        self.img_list = img_list
        self.model_type = model_type
        self.n_clusters = n_clusters

    def extract_img_features(self):
        self.normalized_embedding = self._extract_img_features()
        
    def base(self,
             ):

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        clusters = kmeans.fit_predict(self.normalized_embedding) #(database_keys)
        cluster_centers = kmeans.cluster_centers_
        cluster_ids = np.unique(clusters)

        selected_items = []
        dist_tuples = []
        for cluster_id in cluster_ids:
            cluster_center = cluster_centers[cluster_id]
            cluster_items_idx = np.where(clusters == cluster_id)[0]
            num_selected_items = 1
            cluster_items = self.normalized_embedding[cluster_items_idx,] #database_keys[cluster_items_idx,]
            dist = calculate_hamming(cluster_center, cluster_items)
            ind = np.argsort(dist)
            item_idx_list = cluster_items_idx[ind]
            for i, idx in enumerate(item_idx_list[:num_selected_items]):
                selected_items.append(self.img_list[idx]) #(item_list[idx])
                # dist_tuples.append(
                #     (cluster_id, item_list[idx].id, item_list[idx].subset, dist[ind][i])
                # )
        return selected_items#, dist_tuples
