from dataclasses import dataclass


@dataclass(frozen=True)
class Keypoint:
    index: int
    x: float
    y: float
    confidence: float
    visibility: float | None = None


@dataclass(frozen=True)
class PoseResult:
    keypoints: tuple[Keypoint, ...]
    confidence: float
    bbox: tuple[float, float, float, float]
    pose_quality: float
    timestamp: float
    frame_index: int
    checks: dict
    status: str
