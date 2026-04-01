from __future__ import annotations
from dataclasses import dataclass
from .types import Endpoint, ProtocolType, SessionKey


@dataclass(slots=True)
class NetworkEvent:
    protocol: ProtocolType
    source: Endpoint
    destination: Endpoint
    size: int
    timestamp: float

    def make_key(self) -> SessionKey:
        endpoints = sorted([
            self.source.to_key_part(),
            self.destination.to_key_part()
        ])
        
        return (
            self.protocol.value,
            endpoints[0],
            endpoints[1]
        )

