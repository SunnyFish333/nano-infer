import os
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
    num_gdn_slots: int = 64
    has_gdn: bool = field(default=False, init=False)  # ModelRunner 根据实际模型层设置。

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.num_gdn_slots > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.vision_config = getattr(self.hf_config, "vision_config", None)
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            setattr(self, name, getattr(self.hf_config, name, -1))
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
