from dataclasses import dataclass
from src.network.network_event import NetworkEvent


@dataclass(slots=True)
class SessionEvent:
    event: NetworkEvent
    is_first: bool
    direction_changed: bool
    delay: float

