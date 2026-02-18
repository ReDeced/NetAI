from .packet_protocol import PacketProtocol
from .session_state import SessionState
from typing import Any, Hashable, Tuple


class SessionManager:
    def __init__(self, timeout: float = 60, max_len: int = 50) -> None:
        self.sessions: dict[Hashable, SessionState] = {}
        self.timeout: float = timeout
        self.max_len: int = max_len

    def _make_key(self, packet: PacketProtocol) -> Tuple:
        endpoint_a = (packet.src, packet.sport)
        endpoint_b = (packet.dst, packet.dport)

        ordered = tuple(sorted([endpoint_a, endpoint_b]))
        return ordered + (packet.prot,)

    def process_packet(
        self, 
        packet: PacketProtocol, 
        feature_vector: list[float], 
        packet_size: int
    ) -> list[list[float]]:
        key = self._make_key(packet)

        if key not in self.sessions:
            self.sessions[key] = SessionState(max_len=self.max_len)
        
        session = self.sessions[key]
        session.update(feature_vector, packet_size)
        
        return session.get_sequence()

    def cleanup(self) -> None:
        expired = [
            key
            for key, session in self.sessions.items()
            if session.is_expired(self.timeout)
        ]
        
        for key in expired:
            del self.sessions[key]


