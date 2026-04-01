from __future__ import annotations
import torch


class StatsCollector:
    def __init__(self, feature_dim: int, continous_indices: list[int]) -> None:
        self.feature_dim = feature_dim
        self.continous_indices = continous_indices
        
        self.count = 0
        self.sum = torch.zeros(len(continous_indices), dtype=torch.float64)
        self.sum_sq = torch.zeros(len(continous_indices), dtype=torch.float64)

    def update(self, window: torch.Tensor) -> None:
        cont = window[:, self.continous_indices]

        self.sum += cont.sum(dim=0).double()
        self.sum_sq += (cont ** 2).sum(dim=0).double()
        
        self.count += cont.shape[0]

    def finalize(self) -> dict[str, torch.Tensor]:
        mean = self.sum / self.count
        var = (self.sum_sq / self.count) - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-8))

        return {
            "mean": mean.float(),
            "std": std.float(),
            "continuous_indices": torch.tensor(self.continous_indices)
        }

