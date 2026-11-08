"""Bounded local video decoding with original frame indices for timestamps."""

from pathlib import Path

import numpy as np


def load_video(path, max_frames=16, fps=2.0):
    import av
    from transformers.video_utils import VideoMetadata

    if max_frames < 1 or fps <= 0:
        raise ValueError("Video frame limit and sampling fps must be positive")
    with av.open(str(Path(path))) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 0)
        if source_fps <= 0:
            raise ValueError("Video must provide a positive frame rate")
        total = stream.frames
        if not total and stream.duration is not None:
            total = round(float(stream.duration * stream.time_base) * source_fps)
        if total < 1:
            raise ValueError("Video must provide frame count or duration metadata")
        count = min(max_frames, total, max(1, int(total / source_fps * fps)))
        indices = np.linspace(0, total - 1, count, dtype=int).tolist()
        targets = set(indices)
        frames, actual = [], []
        for index, frame in enumerate(container.decode(stream)):
            if index in targets:
                frames.append(frame.to_ndarray(format="rgb24"))
                actual.append(index)
            if index >= indices[-1]:
                break
    if not frames:
        raise ValueError("Video contains no decodable sampled frames")
    return np.stack(frames), VideoMetadata(
        total_num_frames=total, fps=source_fps, frames_indices=actual,
        duration=total / source_fps)
