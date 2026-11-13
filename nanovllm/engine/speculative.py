"""CPU transactions; GPU probabilities/hidden states remain owned by the runner."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetForwardOutput:
    logit_hidden: Any
    mtp_hidden: Any


@dataclass
class CapacityReservation:
    seq: Any
    old_blocks: int
    required_capacity: int
    closed: bool = False


@dataclass
class SpeculativeStepPlan:
    step_id: int
    seq_id: int
    k: int
    target_start: int
    draft_start: int
    reservation: CapacityReservation
    source_slot: int
    state_slots: tuple[int, ...]
    state: str = "RESERVED"
    fallback_reason: str | None = None


@dataclass
class DraftOutput:
    token_ids: list[int]
    probabilities: Any
    draft_spec_end: int


@dataclass
class VerifyOutput:
    logits: Any
    hidden: Any
    state_slots: tuple[int, ...]
    completed: bool = False


@dataclass
class SpeculativeResult:
    seq_id: int
    token_ids: list[int]
    accepted: int
    target_cached_len: int
    draft_cached_len: int
    selected_slot: int
    finished: bool = False
    stats: dict = field(default_factory=dict)
