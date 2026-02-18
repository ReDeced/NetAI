from typing import Protocol


class PacketProtocol(Protocol):
    src: str
    dst: str
    sport: int
    dport: int
    prot: int
