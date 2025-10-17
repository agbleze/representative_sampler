

import numpy as np
import random
from typing import List, Dict
from collections import defaultdict
from representative_sampler.core.domain.strategies.registry import Registry

SAMPLER_REGISTRY = Registry()

class HistogramSampler:
    sampler_name: str = "histogram_sampler"
    status: str = "experimental"
    
    def __init__(self, scores: Dict[str, float], num_bins: int = 10):
        """
        Args:
            scores: dict mapping image_id -> score
            num_bins: number of histogram bins
        """
        self.scores = scores
        self.num_bins = num_bins
        self.counts, self.bin_edges = np.histogram(
            list(scores.values()), bins=num_bins
        )
        self.bin_to_images = self._assign_bins()

    def _assign_bins(self) -> Dict[int, List[str]]:
        """Assign each image to a histogram bin using np.digitize."""
        values = np.array(list(self.scores.values()))
        image_ids = list(self.scores.keys())

        # Assign each score to a bin index
        bin_indices = np.digitize(values, self.bin_edges, right=False)

        bin_to_images = defaultdict(list)
        for img_id, b in zip(image_ids, bin_indices):
            bin_to_images[b].append(img_id)

        return bin_to_images

    def sample(self, k: int, strategy: str = "uniform") -> List[str]:
        """
        Sample images from bins.
        Args:
            k: number of images to sample
            strategy: "uniform" (equal chance per bin) or "proportional" (weighted by bin size)
        """
        sampled = []

        if strategy == "uniform":
            for _ in range(k):
                chosen_bin = random.choice(list(self.bin_to_images.keys()))
                if self.bin_to_images[chosen_bin]:
                    sampled.append(random.choice(self.bin_to_images[chosen_bin]))

        elif strategy == "proportional":
            all_images = [img for imgs in self.bin_to_images.values() for img in imgs]
            sampled = random.sample(all_images, k)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return sampled


# Auto-register in the SAMPLER_REGISTRY
SAMPLER_REGISTRY.register(
    HistogramSampler.sampler_name, HistogramSampler, HistogramSampler.status
)
