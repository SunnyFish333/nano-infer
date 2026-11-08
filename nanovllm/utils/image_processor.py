"""HF image/video preprocessing and explicit visual intervals for the scheduler."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class VisualItem:
    start: int
    end: int
    grid_thw: tuple[int, int, int]
    pixel_values: torch.Tensor
    modality: str = "image"


@dataclass
class PreparedPrompt:
    token_ids: list[int]
    visual_items: list[VisualItem]


class ImageProcessor:
    def __init__(self, config, processor=None):
        self.config = config
        if processor is None:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(config.model, local_files_only=True)
        self.processor = processor

    def process(self, request):
        from PIL import Image

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
                    if not isinstance(image, Image.Image):
                        raise ValueError("Images must be PIL images or local file paths")
                    images.append(image.convert("RGB"))
                elif kind == "video":
                    from nanovllm.utils.video_processor import load_video
                    frames, info = load_video(part["video"], self.config.max_video_frames,
                                              self.config.video_fps)
                    videos.append(frames)
                    metadata.append(info)
                elif kind != "text":
                    raise ValueError(f"Unsupported content type: {kind!r}")
        if len(images) > self.config.max_images_per_prompt or len(videos) > self.config.max_videos_per_prompt:
            raise ValueError("Request exceeds image/video count limit")
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
                if min(t, h, w) <= 0 or h % merge or w % merge or (modality == "image" and t != 1):
                    raise ValueError("Invalid visual grid")
                patch = getattr(self.config.vision_config, "patch_size", 16)
                if h*w*patch**2 > getattr(self.config, "max_image_pixels", float("inf")):
                    raise ValueError("Processed image/frame exceeds max_image_pixels")
                for _ in range(t):
                    frames.append(((1, h, w), pixels[offset:offset+h*w].cpu()))
                    offset += h*w
            if pixels is not None and offset != len(pixels):
                raise ValueError("Pixel count does not match visual grids")
            queues[token_id] = (modality, frames, 0)
        items, cursor = [], 0
        while cursor < len(ids):
            if ids[cursor] not in queues:
                cursor += 1
                continue
            token_id = ids[cursor]
            modality, frames, index = queues[token_id]
            if index >= len(frames):
                raise ValueError("Visual placeholder has no matching image/frame")
            grid, pixels = frames[index]
            count = grid[1] * grid[2] // merge**2
            if ids[cursor:cursor+count] != [token_id] * count:
                raise ValueError("Visual placeholder count does not match grid")
            items.append(VisualItem(cursor, cursor+count, grid, pixels, modality))
            queues[token_id] = (modality, frames, index+1)
            cursor += count
        if any(index != len(frames) for _, frames, index in queues.values()):
            raise ValueError("Image/frame has no matching visual placeholder")
        return PreparedPrompt(ids, items)
