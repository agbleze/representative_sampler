from .base_scorer import BaseScorer
from ..object_collections import ScoreCollection
from representative_sampler.core.domain.entities import EmbeddingResult, ScoringResult
from scipy.stats import entropy
import numpy as np
import skfuzzy as fuzz
from sklearn.mixture import GaussianMixture


class GMMEntropyScorer(BaseScorer):
    scorer_name = "gmm_entropy_scorer"
    status = "experimental"
    
    def __init__(self, *args, **kwargs):
        pass
    
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:
        embedding = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        
        gmm = GaussianMixture(n_components=10).fit(embeddings)
        _probs = gmm.predict_proba(embeddings)
        _sample_entropies = entropy(_probs.T)  # entropy per sample

    

class FCMEntropyScorer(BaseScorer):
    scorer_name = "entropy_scorer"
    status = "experimental"
    
    def __init__(self, n_clusters,
                 m, error, maxiter,
                 init,
                 *args, **kwargs
                 ):
        pass
    
    def score(self, embedding_obj: EmbeddingResult, **kwargs) -> ScoreCollection:
        pass
    
    def compute_fuzzy_cmeans_entropy(self, embedding_obj: EmbeddingResult):
        embeddings = embedding_obj.embedding
        embedding_names = embedding_obj.embedding_name
        
        fuzz.cluster.cmeans(
        embeddings.T, c=n_centers, m=4,
        error=0.005, maxiter=1000, init=None
    )
    
    
    
    