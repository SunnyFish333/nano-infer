from dataclasses import dataclass
import math


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert math.isfinite(self.temperature) and self.temperature >= 0, "temperature must be finite and nonnegative"
