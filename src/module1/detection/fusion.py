"""Independent, deterministic fusion of two detectors in original-frame pixels."""
from dataclasses import dataclass
import math

from .base import Detection

STRATEGIES = ('or', 'and', 'confidence_weighted', 'weighted_box_fusion')


@dataclass(frozen=True)
class FusionConfig:
    strategy: str = 'or'
    iou_threshold: float = 0.5
    yolo_weight: float = 1.0
    rtdetr_weight: float = 1.0
    min_confidence: float = 0.0
    agreement_bonus: float = 0.0

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f'Unknown fusion strategy: {self.strategy}')
        if not math.isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1:
            raise ValueError('iou_threshold must be in (0, 1]')
        for weight in (self.yolo_weight, self.rtdetr_weight):
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError('Detector weights must be finite and positive')
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError('min_confidence must be in [0, 1]')
        if not math.isfinite(self.agreement_bonus) or not 0 <= self.agreement_bonus <= 1:
            raise ValueError('agreement_bonus must be in [0, 1]')


def validate_detection(d):
    if len(d.bbox) != 4 or not all(math.isfinite(v) for v in d.bbox):
        raise ValueError('Bounding box must contain four finite coordinates')
    x1, y1, x2, y2 = d.bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Bounding box must have positive area')
    if not math.isfinite(d.confidence) or not 0 <= d.confidence <= 1:
        raise ValueError('Detection confidence must be in [0, 1]')
    if d.frame_index < 0 or int(d.frame_index) != d.frame_index:
        raise ValueError('frame_index must be a nonnegative integer')
    if not math.isfinite(d.timestamp) or d.timestamp < 0:
        raise ValueError('timestamp must be finite and nonnegative')
    if int(d.class_id) != d.class_id or d.class_id < 0:
        raise ValueError('class_id must be a nonnegative integer')


def box_iou(a, b):
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def detection_key(d):
    return (-d.confidence, d.class_id, tuple(d.bbox), d.class_name)


@dataclass(frozen=True)
class FusionCandidate:
    yolo: Detection | None
    rtdetr: Detection | None
    iou: float | None
    detection: Detection | None
    selection_score: float

    @property
    def agreement(self):
        return self.yolo is not None and self.rtdetr is not None

    @property
    def source(self):
        return 'agreement' if self.agreement else ('yolo_only' if self.yolo else 'rtdetr_only')


@dataclass(frozen=True)
class FusionResult:
    candidates: tuple[FusionCandidate, ...]
    selected: FusionCandidate | None


class DetectorFusion:
    def __init__(self, config=None):
        self.config = config or FusionConfig()

    def fuse(self, yolo: list[Detection], rtdetr: list[Detection]) -> FusionResult:
        cfg = self.config
        for d in [*yolo, *rtdetr]:
            validate_detection(d)
        inputs = [*yolo, *rtdetr]
        if inputs and any(d.frame_index != inputs[0].frame_index or
                          abs(d.timestamp - inputs[0].timestamp) > 1e-8 for d in inputs):
            raise ValueError('Cannot fuse detections from different frames or timestamps')
        yolo, rtdetr = sorted(yolo, key=detection_key), sorted(rtdetr, key=detection_key)
        edges = []
        for i, y in enumerate(yolo):
            for j, r in enumerate(rtdetr):
                overlap = box_iou(y.bbox, r.bbox)
                if y.class_id == r.class_id and overlap >= cfg.iou_threshold:
                    edges.append((-overlap, i, j))
        used_y, used_r, pairs = set(), set(), []
        # Greedy highest-IoU matching: each detector contributes at most one box per pair.
        for negative_iou, i, j in sorted(edges):
            if i not in used_y and j not in used_r:
                pairs.append((yolo[i], rtdetr[j], -negative_iou))
                used_y.add(i)
                used_r.add(j)
        pairs.extend((d, None, None) for i, d in enumerate(yolo) if i not in used_y)
        pairs.extend((None, d, None) for j, d in enumerate(rtdetr) if j not in used_r)
        candidates = tuple(self._combine(y, r, overlap) for y, r, overlap in pairs)
        accepted = [c for c in candidates if c.detection is not None]
        accepted.sort(key=lambda c: (-c.selection_score, not c.agreement, detection_key(c.detection), c.source))
        return FusionResult(candidates, accepted[0] if accepted else None)

    def _combine(self, y, r, overlap):
        cfg = self.config
        agreement = y is not None and r is not None
        if cfg.strategy == 'and' and not agreement:
            return FusionCandidate(y, r, overlap, None, 0.0)
        supported = [(d, w) for d, w in ((y, cfg.yolo_weight), (r, cfg.rtdetr_weight)) if d is not None]
        strongest = min((d for d, _ in supported), key=detection_key)
        box, confidence = strongest.bbox, strongest.confidence
        if cfg.strategy == 'and':
            confidence = min(d.confidence for d, _ in supported)
        elif cfg.strategy in ('confidence_weighted', 'weighted_box_fusion'):
            mass = sum(d.confidence * w for d, w in supported)
            if mass:
                box = tuple(sum(d.bbox[k] * d.confidence * w for d, w in supported) / mass for k in range(4))
            denominator = sum(w for _, w in supported)
            if cfg.strategy == 'weighted_box_fusion':
                denominator = cfg.yolo_weight + cfg.rtdetr_weight
            confidence = mass / denominator
        detection = None
        if confidence >= cfg.min_confidence:
            detection = Detection(tuple(box), confidence, strongest.class_id, strongest.class_name,
                                  strongest.frame_index, strongest.timestamp)
        return FusionCandidate(y, r, overlap, detection, confidence + cfg.agreement_bonus * agreement)
