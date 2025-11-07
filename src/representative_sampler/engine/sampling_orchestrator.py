
from ..core.sampler.base_sampler import Sampler
from ..core.scorer.base_scorer import BaseScorer
from ..core.embedder.base_embedder import Embedder
from ..core.io.baseio import BaseImporter, BaseExporter


# class SamplingOrchestrator(object):
#     def __init__(self, sampler: Sampler, 
#                 scorer: BaseScorer, 
#                 embedder: Embedder
#                 ):
#         embeddings = embedder.get_embeddings()
#         scores = scorer.score(embeddings=embeddings)
#         samples = sampler.sample(score_collection=scores)
#         return samples
   

def run_sampler(sampler: Sampler, 
                scorer: BaseScorer, 
                embedder: Embedder,
                exporter: BaseExporter,
                importer: BaseImporter,
                *args, **kwargs
                ):
    embeddings = embedder.get_embeddings()
    scores = scorer.score(embeddings=embeddings)
    samples = sampler.sample(score_collection=scores)
    return samples