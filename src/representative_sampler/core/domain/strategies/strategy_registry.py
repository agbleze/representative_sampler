from typing import Dict, Type
from representative_sampler.core.domain.strategies.base_strategy import SamplingStrategy

STRATEGY_REGISTRY: Dict[str, Type[SamplingStrategy]] = {}

def register_strategy(cls: Type[SamplingStrategy]):
    STRATEGY_REGISTRY[cls.name] = cls
    return cls