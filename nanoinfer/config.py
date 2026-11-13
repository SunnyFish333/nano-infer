from dataclasses import dataclass, field

from transformers import AutoConfig, PretrainedConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: PretrainedConfig | None = field(default=None, init=False)
    text_config: PretrainedConfig | None = field(default=None, init=False)
    vision_config: PretrainedConfig | None = field(default=None, init=False)
    image_token_id: int = field(default=-1, init=False)
    video_token_id: int = field(default=-1, init=False)
    vision_start_token_id: int = field(default=-1, init=False)
    vision_end_token_id: int = field(default=-1, init=False)
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_gdn_slots: int | None = None
    cache_plan: dict = field(default_factory=dict, init=False)
    auto_gdn_cache: bool = field(default=False, init=False)
    adapt_cache_to_workload: bool = True
    enable_gdn_prefix_cache: bool = True
    gdn_prefix_cache_retention_interval: int | None = 0
    max_image_pixels: int = 262144
    max_video_frames: int = 8
    video_fps: float = 2.0
    reference_numerics: bool = False
    enable_mtp: bool = False
    num_speculative_tokens: int = 1
    speculative_max_batch_size: int = 1
    mtp_batch_execution: bool = True
    adaptive_speculation: bool = True
    enable_gdn_cuda_graph: bool = False
    cuda_graph_max_batch_size: int = 8
    gdn_graph_slot: int = field(default=-1, init=False)
    has_gdn: bool = field(default=False, init=False)  # ModelRunner 根据实际模型层设置。

    def __post_init__(self):
        self.auto_gdn_cache = self.num_gdn_slots is None
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.vision_config = getattr(self.hf_config, "vision_config", None)
        if self.enable_mtp:
            assert self.hf_config.model_type == "qwen3_5", "MTP requires Qwen3.5 Dense"
            assert self.tensor_parallel_size == 1 and (self.enforce_eager or self.enable_gdn_cuda_graph), "MTP requires TP=1 and eager or explicit GDN CUDA Graph support"
            assert getattr(self.text_config, "num_experts", 0) <= 0, "MTP requires Dense weights"
            assert getattr(self.text_config, "mtp_num_hidden_layers", 0) == 1, "MTP requires one prediction layer"
            assert not getattr(self.text_config, "mtp_use_dedicated_embeddings", False), "Dedicated MTP embeddings are unsupported"
            assert self.num_speculative_tokens >= 0 and self.speculative_max_batch_size > 0
        if self.enable_gdn_cuda_graph:
            assert self.hf_config.model_type == "qwen3_5" and not self.reference_numerics
            assert self.tensor_parallel_size == 1 and not self.enforce_eager
            assert self.cuda_graph_max_batch_size > 0
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            setattr(self, name, getattr(self.hf_config, name, -1))
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
