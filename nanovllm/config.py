import os
import math
from numbers import Integral, Real
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
    max_images_per_prompt: int = 4
    max_videos_per_prompt: int = 1
    max_image_pixels: int = 262144
    max_video_frames: int = 8
    video_fps: float = 2.0
    vision_cpu_cache_mb: int = 256
    reference_numerics: bool = False
    has_gdn: bool = field(default=False, init=False)  # ModelRunner 根据实际模型层设置。

    def __post_init__(self):
        # Validate before reading a checkpoint or starting any GPU allocation.
        for name in ("max_num_batched_tokens", "max_num_seqs", "max_model_len",
                     "tensor_parallel_size", "kvcache_block_size", "num_gdn_slots",
                     "max_images_per_prompt", "max_videos_per_prompt", "max_image_pixels",
                     "max_video_frames", "vision_cpu_cache_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.tensor_parallel_size > 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if self.max_image_pixels < 1024:
            raise ValueError("max_image_pixels must be >=1024")
        if (isinstance(self.num_kvcache_blocks, bool)
                or not isinstance(self.num_kvcache_blocks, Integral)
                or (self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0)):
            raise ValueError("num_kvcache_blocks must be -1 or a positive integer")
        for name in ("gpu_memory_utilization", "video_fps"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be in (0,1]")
        if not os.path.isdir(self.model):
            raise ValueError("model must be an existing local directory")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.text_config = getattr(self.hf_config, "text_config", self.hf_config)
        self.vision_config = getattr(self.hf_config, "vision_config", None)
        for name in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
            setattr(self, name, getattr(self.hf_config, name, -1))
        self.max_model_len = min(self.max_model_len, self.text_config.max_position_embeddings)
