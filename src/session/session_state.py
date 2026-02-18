import time
from collections import deque


class SessionState:
    def __init__(self, max_len: int = 50) -> None:
        self.buffer: deque[list[float]] = deque(maxlen=max_len)
        self.start_time: float = time.time()
        self.last_seen: float = time.time()
        self.total_bytes: int = 0
        self.packet_count: int = 0

    def update(self, feature_vector: list[float], packet_size: int) -> None:
        self.buffer.append(feature_vector)
        self.packet_count += 1
        self.total_bytes += packet_size
        self.last_seen = time.time()

    def is_expired(self, timeout: float) -> bool:
        return (time.time() - self.last_seen) > timeout

    def get_sequence(self) -> list[list[float]]:
        return list(self.buffer)
