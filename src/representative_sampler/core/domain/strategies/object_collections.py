
from collections import UserList
from representative_sampler.core.domain.entities import ScoringResult, SamplingResult

class ScoreCollection(UserList):
    def __init__(self, initlist=None):
        super().__init__(initlist or [])
    
    def add_score(self, score):
        self.data.append(score)
    
    def get_scores(self):
        return self.data    
    
    def validity_check(self):
        for score in self.data:
            if not isinstance(score, ScoringResult):
                raise ValueError(f"{score} is not a valid instance of ScoringResult. ScoreCollection requires all items to be of type ScoringResult.")
            

class SampleCollection(UserList):

    def add_sample(self, sample):
        self.data.append(sample)
    
    def get_samples(self):
        return self.data    
    
    def validity_check(self):
        for sample in self.data:
            if not isinstance(sample, SamplingResult):
                raise ValueError(f"{sample} is not a valid instance of Sample. SampleCollection requires all items to be of type SamplingResult.")         