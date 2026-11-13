"""CPU 投机事务与窗口策略；GPU 概率和隐藏状态由执行器持有。"""
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


class SpeculationPolicy:
    def __init__(self):
        self.trials = {}
        self.costs = {}
        self.current = {}

    @staticmethod
    def bucket(batch):
        return 1 << (max(1, batch)-1).bit_length()

    def choose(self, cap, batch):
        load = self.bucket(batch)
        choices = [k for k in (1, 2, 4, 0) if k <= cap]
        if cap not in choices:
            choices.insert(-1, cap)
        for k in choices:
            if not self.trials.get((load, k), 0):
                self.trials[load, k] = 1
                return k
        tick = sum(self.trials.get((load, k), 0) for k in choices)
        if tick % 32 == 0:
            chosen = choices[(tick // 32) % len(choices)]
        else:
            measured = [k for k in choices if (load, k) in self.costs]
            best = min(measured, key=lambda k: self.costs[load, k]) if measured else choices[0]
            previous = self.current.get(load, best)
            chosen = (previous if previous in measured and
                      self.costs[load, previous] <= self.costs[load, best] * 1.05 else best)
        self.current[load] = chosen
        self.trials[load, chosen] = self.trials.get((load, chosen), 0) + 1
        return chosen

    def observe(self, batch, k, emitted, elapsed_ms):
        if emitted <= 0 or elapsed_ms <= 0:
            return
        key = self.bucket(batch), k
        cost = elapsed_ms / emitted
        self.costs[key] = cost if key not in self.costs else self.costs[key] * 0.8 + cost * 0.2
