from dataclasses import dataclass, field
from typing import Union
import numpy as np


@dataclass
class ClusterMetadata:
    cluster_name: int = field(default_factory=int)
    cluster_size: int = field(default_factory=int)
    cluster_entropy: float = field(default_factory=float)
    cluster_weight: float = field(default_factory=float)
    cluster_sampling_size: int = field(default_factory=int)
    cluster_indices: Union[np.ndarray, int] = field(default_factory=int)