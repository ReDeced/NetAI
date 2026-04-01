from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.shard_dataset import ShardIterableDataset
from src.data.stats import StatsCollector
from src.models.netai_model import NetAI
from src.models.standardization import StandardizationLayer
from src.features.constants import FEATURE_DIM

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 10
LR = 10e-3
NUM_WORKERS = 4

CONTINUOUS_INDICES = [0, 8]


def compute_stats(dataset: ShardIterableDataset):
    collector = StatsCollector(FEATURE_DIM, CONTINUOUS_INDICES)
    
    print("Computing train statistics...")

    for sample in dataset:
        window = sample["input"]
        collector.update(window)

    stats = collector.finalize()
    print("Statistics computed.")

    return stats

