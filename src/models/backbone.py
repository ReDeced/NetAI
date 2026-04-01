from __future__ import annotations
from typing import Optional

import torch
from torch import nn, Tensor


HiddenState = tuple[Tensor, Tensor]


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        if num_layers < 2 and dropout > 0.0:
            # dropout won't be applied by PyTorch
            import warnings
            warnings.warn(
                "Dropout is ignored when num_layers < 2 in nn.LSTM"
            )

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.num_layers: int = num_layers

        self.lstm: nn.LSTM = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: Tensor,
        hidden: Optional[HiddenState] = None,
    ) -> tuple[Tensor, HiddenState]:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (B, T, F), got {tuple(x.shape)}"
            )

        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected feature dim {self.input_dim}, got {x.size(-1)}"
            )

        output, hidden = self.lstm(x, hidden)
        
        if hidden is None:
            raise RuntimeError(
                "Unexpected LSTM state"
            )

        return output, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> HiddenState:
        
        h = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
        )
        c = torch.zeros_like(h)
        
        return h, c

