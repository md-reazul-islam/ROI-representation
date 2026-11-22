
from dataclasses import dataclass
import math
import cv2
from .video import VideoReader

@dataclass(frozen=True)
class SampledFrame:
    frame_index: int
    timestamp_sec: float
    frame: object

class FrameSampler:
    """Deterministic timestamp-based sampler. Native FPS is never changed."""
    def __init__(self, video_reader: VideoReader, target_fps: float = 20.0):
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.video_reader = video_reader
        self.target_fps = float(target_fps)

    def selected_indices(self, metadata=None):
        m = metadata or self.video_reader.metadata()
        if m.native_fps <= 0 or m.frame_count <= 0:
            return []
        duration = m.frame_count / m.native_fps
        count = max(1, math.ceil(duration * self.target_fps))
        out, seen = [], set()
        for k in range(count):
            t = k / self.target_fps
            if t >= duration:
                break
            i = min(m.frame_count - 1, int(round(t * m.native_fps)))
            if i not in seen:
                out.append(i); seen.add(i)
        return out

    def iter_sampled(self):
        m = self.video_reader.metadata()
        wanted = set(self.selected_indices(m))
        cap = cv2.VideoCapture(str(self.video_reader.path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Unable to open video: {self.video_reader.path}")
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                if i in wanted:
                    yield SampledFrame(i, i/m.native_fps, frame)
                i += 1
        finally:
            cap.release()
