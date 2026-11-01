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

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, gdn_slots=None, gdn_has_initial=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, gdn_slots, gdn_has_initial)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
