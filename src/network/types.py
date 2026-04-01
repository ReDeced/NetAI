from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


SessionKey = tuple[str, str, str]


class ProtocolType(Enum):
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    OTHER = "OTHER"


@dataclass(slots=True, frozen=True)
class Endpoint:
    ip: str
    port: int | None = None

    def to_key_part(self) -> str:
        if self.port is None:
            return self.ip
        return f"{self.ip}:{self.port}"


