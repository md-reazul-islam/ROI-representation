
from dataclasses import dataclass
from pathlib import Path
import cv2

@dataclass(frozen=True)
class VideoMetadata:
    path: str
    filename: str
    width: int
    height: int
    native_fps: float
    frame_count: int
    duration_sec: float
    codec: str
    opened_successfully: bool

class VideoReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def metadata(self) -> VideoMetadata:
        cap = cv2.VideoCapture(str(self.path))
        opened = bool(cap.isOpened())
        if not opened:
            cap.release()
            return VideoMetadata(str(self.path.resolve()), self.path.name, 0, 0, 0.0, 0, 0.0, "", False)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8*i)) & 255) for i in range(4)).strip("\x00")
        cap.release()
        return VideoMetadata(str(self.path.resolve()), self.path.name, width, height, fps, frames,
                             frames/fps if fps > 0 else 0.0, codec, True)
