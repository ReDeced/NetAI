from __future__ import annotations
from typing import Optional

import torch
from torch import nn, Tensor

from .backbone import LSTMBackbone, HiddenState


class NetAI(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.standardization: nn.Module | None = None

        self.heads = nn.ModuleDict()

        self.heads["reconstruction"] = nn.Linear(
            hidden_dim,
            input_dim
        )

    def forward(
        self,
        tensor: Tensor,
        hidden: Optional[HiddenState] = None,
        return_hidden: bool = False,
    ) -> dict[str, Tensor | HiddenState]:
        if self.standardization is not None:
            tensor = self.standardization(tensor)

        features, hidden_out = self.backbone(tensor, hidden)

        outputs: dict[str, Tensor | HiddenState] = {}

        reconstruction = self.heads["reconstruction"](features)
        outputs["reconstruction"] = reconstruction

        if return_hidden:
            outputs["hidden"] = hidden_out

        return outputs

