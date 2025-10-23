from .base_scorer import BaseScorer
from ..object_collections import ScoreCollection
from sklearn.cluster import KMeans
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from scipy.special import softmax
from scipy.stats import entropy
import numpy as np




class KMeansEntropyScorer(BaseScorer):
    scorer_name = "kmeans_entropy_scorer"
    status = "experimental"
    
    def __init__(self, n_clusters, *args, **kwargs):
        self.n_clusters = n_clusters
        
        
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:
        self.score_collection = self.compute_kmeans_entropy(embedding_obj=embedding_obj,
                                                           n_clusters=self.n_clusters
                                                           )
        return self.score_collection
    
    def compute_kmeans_entropy(self, embedding_obj: EmbeddingResult, 
                               n_clusters,
                               **kwargs
                               ) -> ScoreCollection:
        embeddings = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        kmeans = KMeans(n_clusters=n_clusters, 
                        random_state=0
                        )
        clusters = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        all_distances = []
        for emb in embeddings:
            distances = np.linalg.norm(emb - cluster_centers, axis=1)
            all_distances.append(distances)
        all_distances = np.array(all_distances)
        
        all_probabilities = softmax(-all_distances, axis=1)
        
        all_entropies = entropy(all_probabilities.T)
        normalized_entropies = all_entropies / np.log(n_clusters)
        
        entropy_scores = [ScoringResult(object_name=embedding_nm, 
                                        score=score,
                                        scorer_name=self.scorer_name
                                        ) 
                          for embedding_nm, score, in zip(embedding_names, normalized_entropies)
                          ]
        return ScoreCollection(entropy_scores)