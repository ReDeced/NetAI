from math import log1p
from src.network.types import ProtocolType
from src.network.network_event import NetworkEvent
from src.session.session_event import SessionEvent
from typing import Protocol


class FeatureModule(Protocol):
    def extract(self, session_event: SessionEvent) -> list[float]:
        ...

class PacketSizeFeature:
    def extract(self, session_event: SessionEvent) -> list[float]:
        event = session_event.event
        return [float(event.size) / 1500.0]
    
class ProtocolOneHotFeature:
    def extract(self, session_event: SessionEvent) -> list[float]:
        event = session_event.event
        proto = event.protocol

        return [
            1.0 if proto is ProtocolType.TCP else 0.0,
            1.0 if proto is ProtocolType.UDP else 0.0,
            1.0 if proto is ProtocolType.ICMP else 0.0,
            1.0 if proto is ProtocolType.OTHER else 0.0,
        ]

class DirectionFeature:
    def extract(self, session_event: SessionEvent) -> list[float]:
        event = session_event.event

        src = event.source.to_key_part()
        dst = event.destination.to_key_part()

        canonical = sorted((src, dst))
        return [1.0 if src == canonical[0] else 0.0]

class RelativityFeature:
    def extract(self, session_event: SessionEvent) -> list[float]:
        return [
            1.0 if session_event.is_first else 0.0,
            1.0 if session_event.direction_changed else 0.0,
            log1p(session_event.delay)
        ]

