"""Optional CPU-side observations; the engine never imports a benchmark package."""

from time import perf_counter_ns
from typing import Any, Protocol


class EngineObserver(Protocol):
    def on_event(self, event: str, timestamp_ns: int, payload: dict[str, Any]) -> None:
        ...


def emit(observer: EngineObserver | None, event: str, **payload: Any) -> None:
    if observer is not None:
        observer.on_event(event, perf_counter_ns(), payload)
