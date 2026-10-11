from dataclasses import dataclass
from typing import Any
import time


@dataclass
class HumanIntentSignal:
    modality: str
    intent: str
    target: Any = None
    confidence: float = 1.0
    timestamp: float = 0.0
    raw: dict[str, Any] | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.raw is None:
            self.raw = {}
