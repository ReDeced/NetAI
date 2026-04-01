from src.network.network_event import NetworkEvent
from src.network.types import SessionKey
from src.features.feature_extractor import FeatureExtractor
from .session_state import SessionState
from .session_event import SessionEvent


class SessionManager:
    def __init__(self, timeout: float = 60, max_len: int = 50) -> None:
        self.sessions: dict[SessionKey, SessionState] = {}
        self.timeout: float = timeout
        self.max_len: int = max_len

        self.feature_extractor: FeatureExtractor = FeatureExtractor()

    def process_event(
        self, 
        event: NetworkEvent 
    ) -> list[list[float]]:

        key: SessionKey = event.make_key()
        
        is_first = False
        if key not in self.sessions:
            self.sessions[key] = SessionState(max_len=self.max_len)
            is_first = True
        
        session = self.sessions[key]
        
        # направление
        src = event.source.to_key_part()
        dst = event.destination.to_key_part()
        canonical = sorted((src, dst))
        direction = src == canonical[0]
        
        # задержка
        delay = 0.0
        if session.last_seen is not None:
            delay = event.timestamp - session.last_seen 
        
        direction_changed = (
            False if is_first or session.last_direction is None
            else session.last_direction != direction
        )

        session_event = SessionEvent(
            event,
            is_first,
            direction_changed,
            delay
        )

        feature_vector = self.feature_extractor.extract(session_event)

        session.update(feature_vector, event.size, event.timestamp, direction)
        
        return session.get_sequence()

    def cleanup(self) -> None:
        expired = [
            key
            for key, session in self.sessions.items()
            if session.is_expired(self.timeout)
        ]
        
        for key in expired:
            del self.sessions[key]


