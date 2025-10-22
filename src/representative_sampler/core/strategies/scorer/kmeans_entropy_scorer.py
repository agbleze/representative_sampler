from .base_scorer import BaseScorer
from ..object_collections import ScoreCollection




class KMeansEntropyScorer(BaseScorer):
    scorer_name = "kmeans_entropy_scorer"
    status = "experimental"