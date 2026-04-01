from src.session.session_event import SessionEvent
from .features import FeatureModule, PacketSizeFeature, ProtocolOneHotFeature, DirectionFeature, RelativityFeature


class FeatureExtractor:
    def __init__(self) -> None:
        self._modules: tuple[FeatureModule, ...] = (
            PacketSizeFeature(),
            ProtocolOneHotFeature(),
            DirectionFeature(),
            RelativityFeature(),
        )

    def extract(self, session_event: SessionEvent) -> list[float]:
        features: list[float] = []

        for module in self._modules:
            features.extend(module.extract(session_event))

        return features

