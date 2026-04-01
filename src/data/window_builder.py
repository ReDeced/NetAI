from __future__ import annotations
import torch
from src.session.session_manager import SessionManager
from src.network.network_event import NetworkEvent


class WindowBuilder:
    def __init__(
        self,
        timeout: float,
        max_len: int = 128,
        min_len: int = 32,
        cap_per_session: int = 500,
        cap_total: int = 5_000_000
    ) -> None:
        self.manager = SessionManager(timeout, max_len)
        self.max_len = max_len
        self.min_len = min_len
        self.cap_per_session = cap_per_session
        self.cap_total = cap_total

        self.total_windows = 0
        self.session_counts: dict = {}

    def process(
        self,
        event: NetworkEvent
    ) -> tuple[torch.Tensor, float] | None:
        
        sequence = self.manager.process_event(event)
        
        if len(sequence) < self.min_len:
            return None

        key = event.make_key()

        if key not in self.session_counts:
            self.session_counts[key] = 0

        if self.session_counts[key] >= self.cap_per_session:
            return None

        if self.total_windows >= self.cap_total:
            return None

        if len(sequence) == self.max_len:
            window_tensor = torch.tensor(sequence, dtype=torch.float32)
            timestamp = event.timestamp

            self.session_counts[key] += 1
            self.total_windows += 1

            return window_tensor, timestamp

        return None

