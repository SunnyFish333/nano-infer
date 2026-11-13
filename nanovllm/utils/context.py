from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    gdn_slots: torch.Tensor | None = None   # GDN：本轮每个请求使用的状态槽位，[请求数]
    gdn_has_initial: list[bool] | None = None   # GDN：每个请求是否有可续算的历史状态，供 prefill 分支使用
    mode: str = "normal"
    gdn_verify_segments: list[tuple[int, int, int, tuple[int, ...]]] | None = None
    state_io_events: list | None = None  # Optional diagnostic profiling; never synchronizes here.
    gdn_verify_destinations: torch.Tensor | None = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    gdn_slots=None,
    gdn_has_initial=None,
    mode="normal",
    gdn_verify_segments=None,
    state_io_events=None,
    gdn_verify_destinations=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables, gdn_slots, gdn_has_initial,
        mode, gdn_verify_segments, state_io_events, gdn_verify_destinations,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
