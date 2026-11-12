"""HF image/video preprocessing and explicit visual intervals for the scheduler."""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from struct import pack
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
import torch

from nanovllm.utils.video_processor import load_video


@dataclass(frozen=True)
class VisualItem:
    start: int
    end: int
    grid_thw: tuple[int, int, int]
    pixel_values: torch.Tensor
    modality: str = "image"
    content_digest: bytes = field(init=False)

    def __post_init__(self):
        pixels = self.pixel_values.detach().cpu().contiguous()
        digest = sha256(b"nanovllm.visual.v1")
        for value in (self.modality.encode(), pack("<3q", *self.grid_thw),
                      str(pixels.dtype).encode(),
                      pack("<Q", pixels.ndim) + pack(f"<{pixels.ndim}q", *pixels.shape),
                      pixels.reshape(-1).view(torch.uint8).numpy().tobytes()):
            digest.update(pack("<Q", len(value)))
            digest.update(value)
        object.__setattr__(self, "content_digest", digest.digest())


@dataclass
class PreparedPrompt:
    token_ids: list[int]
    visual_items: list[VisualItem]


class ImageProcessor:
    def __init__(self, config, processor=None):
        self.config = config
        if processor is None:
            processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
        self.processor = processor

    def process(self, request):
        messages = deepcopy(request["messages"])
        images, videos, metadata = [], [], []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            for part in content:
                kind = part.get("type")
                if kind == "image":
                    image = part["image"]
                    if isinstance(image, (str, Path)):
                        with Image.open(image) as source:
                            image = source.convert("RGB")
                    images.append(image.convert("RGB"))
                elif kind == "video":
                    frames, info = load_video(part["video"], self.config.max_video_frames,
                                              self.config.video_fps)
                    videos.append(frames)
                    metadata.append(info)
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        kwargs = dict(text=[text], return_tensors="pt", padding=False,
                      images_kwargs={"size": {"shortest_edge": 32 * 32,
                                               "longest_edge": self.config.max_image_pixels}})
        if images:
            kwargs["images"] = images
        if videos:
            kwargs.update(videos=videos, video_metadata=metadata,
                          videos_kwargs={"do_sample_frames": False,
                                         "size": {"shortest_edge": 32 * 32,
                                                  "longest_edge": self.config.max_image_pixels}})
        output = self.processor(**kwargs)
        return self.from_output(output)

    def from_output(self, output):
        ids = output["input_ids"][0].tolist()
        merge = self.config.vision_config.spatial_merge_size
        # Queues are kept separate: the prompt may interleave images and videos.
        queues = {}
        for modality, grid_key, pixel_key, token_id in (
            ("image", "image_grid_thw", "pixel_values", self.config.image_token_id),
            ("video", "video_grid_thw", "pixel_values_videos", self.config.video_token_id),
        ):
            frames, offset = [], 0
            grids = output.get(grid_key, [])
            pixels = output.get(pixel_key)
            for t, h, w in (grids.tolist() if isinstance(grids, torch.Tensor) else grids):
                for _ in range(t):
                    frames.append(((1, h, w), pixels[offset:offset+h*w].cpu()))
                    offset += h*w
            queues[token_id] = (modality, frames, 0)
        items, cursor = [], 0
        while cursor < len(ids):
            if ids[cursor] not in queues:
                cursor += 1
                continue
            token_id = ids[cursor]
            modality, frames, index = queues[token_id]
            grid, pixels = frames[index]
            count = grid[1] * grid[2] // merge**2
            items.append(VisualItem(cursor, cursor+count, grid, pixels, modality))
            queues[token_id] = (modality, frames, index+1)
            cursor += count
        return PreparedPrompt(ids, items)
