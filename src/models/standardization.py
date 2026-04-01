from __future__ import annotations
from typing import cast

from torch import nn, Tensor


class StandardizationLayer(nn.Module):
    def __init__(
        self,
        mean: Tensor,
        std: Tensor,
        continuous_indices: Tensor
    ) -> None:
        super().__init__()

        if mean.ndim != 1:
            raise ValueError("mean must be 1D tensor")

        if std.ndim != 1:
            raise ValueError("std must be 1D tensor")

        if mean.shape != std.shape:
            raise ValueError("mean and std must have same shape")

        if continuous_indices.ndim != 1:
            raise ValueError("continuous_indices must be 1D tensor")

        if len(continuous_indices) != len(mean):
            raise ValueError(
                "continuous_indices size must match mean/std size"
            )
        
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("continuous_indices", continuous_indices.long())

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, T, F), got {tuple(x.shape)}"
            )

        x = x.clone()
        
        mean: Tensor = cast(Tensor, self.mean)
        std: Tensor = cast(Tensor, self.std)
        idx: Tensor = cast(Tensor, self.continuous_indices)
        
        x[:, :, idx] = (x[:, :, idx] - mean) / std

        return x
