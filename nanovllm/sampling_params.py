from dataclasses import dataclass
import math
from numbers import Integral, Real


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        self.validate()

    def validate(self):
        if (isinstance(self.temperature, bool) or not isinstance(self.temperature, Real)
                or not math.isfinite(self.temperature) or self.temperature <= 1e-10):
            raise ValueError("temperature must be finite and greater than 1e-10; greedy sampling is not supported")
        if (isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, Integral)
                or self.max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean")
