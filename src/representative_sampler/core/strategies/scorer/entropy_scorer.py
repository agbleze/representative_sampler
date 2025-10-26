from .base_scorer import BaseScorer
from ..object_collections import ScoreCollection
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from scipy.stats import entropy
import numpy as np
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture

_SENTINEL = object()

class GMMEntropyScorer(BaseScorer):
    scorer_name = "gmm_entropy_scorer"
    status = "experimental"
    
    def __init__(self, n_components, *args, **kwargs):
        self.n_components = n_components
    
    def score(self, embedding_obj, 
              *args, **kwargs
              ) -> ScoreCollection:
        score_collection = self.compute_entropy_score(embedding_obj=embedding_obj,
                                                      n_components=self.n_components
                                                      )
        return score_collection
    
    def compute_entropy_score(self, embedding_obj: EmbeddingResult, 
                              n_components,
                              **kwargs
                              ) -> ScoreCollection:
        embeddings = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        
        gmm = GaussianMixture(n_components=10).fit(embeddings)
        _probs = gmm.predict_proba(embeddings)
        _sample_entropies = entropy(_probs.T) 
        normalized_entropies = _sample_entropies / np.log(n_components)
        
        entropy_scores = [ScoringResult(object_name=embedding_nm, 
                                        score=score,
                                        scorer_name=self.scorer_name
                                        ) 
                          for embedding_nm, score, in zip(embedding_names, normalized_entropies)
                          ]
        return ScoreCollection(entropy_scores)

    

class FCMEntropyScorer(BaseScorer):
    scorer_name = "entropy_scorer"
    status = "experimental"
    
    def __init__(self, n_clusters,
                 m=4, error=0.005, 
                 maxiter=1000,
                 init=None,
                 *args, **kwargs
                 ):
        self.n_clusters = n_clusters
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.init = init
    
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:       
            
        self.score_collection = self.compute_fuzzy_cmeans_entropy(embedding_obj=embedding_obj)
        return self.score_collection
    
    def compute_fuzzy_cmeans_entropy(self, embedding_obj: EmbeddingResult, *args, **kwargs) -> ScoreCollection:
        if "m" in kwargs:
            m = kwargs.get("m")
        else:
            m = self.m
        if "error" in kwargs:
            error = kwargs.get("error")
        else:
            error = self.error
        if "maxiter" in kwargs:
            maxiter = kwargs.get("maxiter")
        else:
            maxiter = self.maxiter
        if "init" in kwargs:
            init = kwargs.get("init")
        else:
            init = self.init
        
        if "n_clusters" in kwargs:
            n_clusters = kwargs.get("n_clusters")
        else:
            n_clusters = self.n_clusters
            
        embeddings = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(embeddings.T, 
                                                     c=n_clusters, 
                                                     m=m,
                                                    error=error, 
                                                    maxiter=maxiter, 
                                                    init=init
                                                    )
        sample_entropies = np.array([entropy(proba) for proba in u.T])
        normalized_entropies = sample_entropies / np.log(self.n_clusters)

        entropy_scores = [ScoringResult(object_name=embedding_nm, 
                                        score=score,
                                        scorer_name=self.scorer_name
                                        ) 
                          for embedding_nm, score, in zip(embedding_names, normalized_entropies)
                          ]
        return ScoreCollection(entropy_scores)
    
    