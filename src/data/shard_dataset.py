from __future__ import annotations
from pathlib import Path
import random 
from typing import Iterable, Iterator
import torch

from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data._utils.worker import WorkerInfo


class ShardIterableDataset(IterableDataset):
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        train_ratio: float = 0.8,
        shuffle_shards: bool = True,
        window_shuffle_buffer: int = 0
    ) -> None:
        super().__init__()

        if split not in {"train", "val", "all"}:
            raise ValueError("split must be one of: train, val, all")

        if not (0.0 < train_ratio < 1.0):
            raise ValueError("train_ratio must be between 0 and 1")

        self.root_dir = root_dir
        self.split = split
        self.train_ratio = train_ratio
        self.shuffle_shards = shuffle_shards
        self.window_shuffle_buffer = window_shuffle_buffer

        self.shard_paths = self._collect_shards()

    def _collect_shards(self) -> list[Path]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")

        shards = sorted(self.root_dir.glob("shard_*.pt"))

        if not shards:
            raise RuntimeError(f"No shard_*.pt files found in {self.root_dir}")

        return shards

    def _split_shards(self) -> list[Path]:
        if self.split == "all":
            return self.shard_paths
        
        total = len(self.shard_paths)
        split_idx = int(total * self.train_ratio)

        if self.split == "train":
            return self.shard_paths[:split_idx]
        else:  # val
            return self.shard_paths[split_idx:]
        
    def _shards_for_worker(self) -> list[Path]:
        shards = self._split_shards()

        worker_info: WorkerInfo | None = get_worker_info()

        if worker_info is None:
            return shards

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        return shards[worker_id::num_workers]

    def __iter__(self) -> Iterator[dict]:
       shard_paths = list(self._shards_for_worker())

       if self.shuffle_shards:
           random.shuffle(shard_paths)

       for shard_path in shard_paths:
           data = torch.load(shard_path, map_location="cpu")

           windows = data["windows"]
           timestamps = data["timestamps"]

           if self.window_shuffle_buffer > 0:
               yield from self._buffered_window_iterator(
                   windows, timestamps
               )
           else:
               for window, ts in zip(windows, timestamps):
                   yield {
                       "input": window,
                       "timestamp": ts
                   }
            
    def _buffered_window_iterator(
        self,
        windows: torch.Tensor,
        timestamps: torch.Tensor
    ) -> Iterable[dict]:

        buffer: list[tuple[torch.Tensor, float]] = []

        for window, ts in zip(windows, timestamps):
            
            ts_val = ts.item() if hasattr(ts, "item") else ts

            buffer.append((window, ts_val))

            if len(buffer) > self.window_shuffle_buffer:
                idx = random.randint(0, len(buffer) - 1)
                w, t = buffer.pop(idx)
                yield {"input": w, "timestamp": t}

        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            w, t = buffer.pop(idx)
            yield {"input": w, "timestamp": t}

            
