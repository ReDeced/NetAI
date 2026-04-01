from __future__ import annotations
from collections import deque


class SessionState:
    def __init__(self, max_len: int = 50) -> None:
        self.buffer: deque[list[float]] = deque(maxlen=max_len)
        self.start_time: float | None = None
        self.last_seen: float | None = None
        self.last_direction : bool | None = None
        self.total_bytes: int = 0
        self.packet_count: int = 0

    def update(
        self, 
        feature_vector: list[float],
        packet_size: int,
        timestamp: float,
        last_direction: bool 
    ) -> None:
        if self.start_time is None:
            self.start_time = timestamp

        self.last_seen = timestamp
        self.last_direction = last_direction
        self.buffer.append(feature_vector)
        self.packet_count += 1
        self.total_bytes += packet_size

    def is_expired(self, timeout: float) -> bool:
        if self.last_seen is None:
            return False

        import time
        return (self.last_seen + timeout) < time.time()

    def get_sequence(self) -> list[list[float]]:
        return list(self.buffer)
