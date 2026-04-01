from __future__ import annotations
from pathlib import Path
import torch


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int = 100_000) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.windows: list[torch.Tensor] = []
        self.timestamps: list[float] = []
        self.shard_index = 0

    def add(self, window: torch.Tensor, timestamp: float) -> None:
        self.windows.append(window)
        self.timestamps.append(timestamp)

        if len(self.windows) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.windows:
            return

        data = {
            "windows": torch.stack(self.windows),
            "timestamps": torch.tensor(self.timestamps)
        }

        shard_path = self.output_dir / f"shard_{self.shard_index:05d}.pt"
        torch.save(data, shard_path)

        self.windows.clear()
        self.timestamps.clear()
        self.shard_index += 1

