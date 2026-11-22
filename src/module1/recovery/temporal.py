from dataclasses import dataclass
import math

from module1.detection.base import Detection
from module1.detection.fusion import box_iou, validate_detection


@dataclass(frozen=True)
class RecoveryConfig:
    max_recovery_frames: int = 10
    min_recovery_confidence: float = .35
    max_box_change: float = .3
    max_center_displacement: float = .5
    max_trajectory_error: float = .25
    min_previous_iou: float = .25
    min_anchor_iou: float = .5
    confidence_decay: float = .9
    min_pose_quality: float = .5

    def __post_init__(self):
        if not isinstance(self.max_recovery_frames, int) or isinstance(self.max_recovery_frames, bool) or self.max_recovery_frames < 0:
            raise ValueError('max_recovery_frames must be a nonnegative integer')
        for name in ('max_box_change', 'max_center_displacement', 'max_trajectory_error'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('min_recovery_confidence', 'min_previous_iou', 'min_anchor_iou', 'min_pose_quality'):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not math.isfinite(self.confidence_decay) or not 0 < self.confidence_decay < 1:
            raise ValueError('confidence_decay must be in (0,1)')


@dataclass(frozen=True)
class TrackerObservation:
    track_id: int | None
    bbox: tuple[float, float, float, float] | None
    velocity: tuple[float, float] | None
    state: str
    observed: bool
    confidence: float
    reliability: float


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def valid_box(box):
    return (box is not None and len(box) == 4 and all(math.isfinite(v) for v in box)
            and box[2] > box[0] and box[3] > box[1])


class TemporalRecovery:
    def __init__(self, config=None):
        self.config = config or RecoveryConfig()
        self.reset()

    def reset(self):
        self.anchor = None
        self.previous_box = None
        self.gap = 0
        self.last_index = None
        self.last_timestamp = None

    def update(self, detection: Detection | None, tracker: TrackerObservation,
               frame_index: int, timestamp: float, image_size: tuple[int, int],
               pose_quality: float | None = None):
        cfg = self.config
        if not isinstance(frame_index, int) or frame_index < 0 or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid frame metadata')
        if self.last_index is not None and (frame_index <= self.last_index or timestamp <= self.last_timestamp):
            raise ValueError('Frames and timestamps must increase strictly; reset between videos')
        if len(image_size) != 2 or any(not math.isfinite(v) or v <= 0 for v in image_size):
            raise ValueError('Image size must be positive (width, height)')
        if pose_quality is not None and (not math.isfinite(pose_quality) or not 0 <= pose_quality <= 1):
            raise ValueError('pose_quality must be in [0,1] when available')
        if detection:
            validate_detection(detection)
            if detection.frame_index != frame_index or abs(detection.timestamp - timestamp) > 1e-8:
                raise ValueError('Detection and frame metadata mismatch')
        self.last_index, self.last_timestamp = frame_index, timestamp
        result = dict(frame_index=frame_index, timestamp_sec=timestamp,
                      detector_status='DETECTED' if detection else 'MISSED', tracker_status=tracker.state,
                      track_id=tracker.track_id, tracker_observed=tracker.observed,
                      recovered_status='NOT_NEEDED' if detection else 'REJECTED', recovery_confidence=0.,
                      final_status='DETECTED' if detection else 'MISSED',
                      final_infant_bbox=detection.bbox if detection else None,
                      final_infant_confidence=detection.confidence if detection else 0.,
                      candidate_bbox=None, rejection_reason='', gap_age=0,
                      anchor_frame_index=None, anchor_timestamp=None, anchor_track_id=None,
                      previous_iou=None, scale_change=None, center_displacement=None,
                      trajectory_error=None, pose_quality=pose_quality,
                      pose_validation_status='AVAILABLE' if pose_quality is not None else 'UNAVAILABLE',
                      ground_truth_status='UNAVAILABLE')
        if detection:
            self.gap = 0
            self.anchor = None
            self.previous_box = detection.bbox
            if (tracker.state == 'CONFIRMED' and tracker.observed and tracker.track_id is not None
                    and valid_box(tracker.bbox) and tracker.velocity is not None
                    and len(tracker.velocity) == 2 and all(math.isfinite(v) for v in tracker.velocity)
                    and box_iou(detection.bbox, tracker.bbox) >= cfg.min_anchor_iou
                    and 0 <= tracker.confidence <= 1 and 0 <= tracker.reliability <= 1):
                self.anchor = dict(box=detection.bbox, timestamp=timestamp, frame_index=frame_index,
                                   track_id=tracker.track_id, velocity=tracker.velocity,
                                   confidence=min(detection.confidence, tracker.confidence, tracker.reliability))
            return result
        self.gap += 1
        result['gap_age'] = self.gap

        def reject(reason):
            result['rejection_reason'] = reason
            return result

        if self.anchor is None:
            return reject('NO_CONFIRMED_DETECTION_ANCHOR')
        anchor = self.anchor
        result.update(anchor_frame_index=anchor['frame_index'], anchor_timestamp=anchor['timestamp'],
                      anchor_track_id=anchor['track_id'])
        if self.gap > cfg.max_recovery_frames:
            return reject('RECOVERY_LIMIT_EXCEEDED')
        if tracker.track_id != anchor['track_id']:
            return reject('TRACK_ID_CHANGED_OR_MISSING')
        if tracker.state != 'LOST' or tracker.observed:
            return reject('NO_LOST_TRACK_PREDICTION')
        box = tracker.bbox
        if not valid_box(box):
            return reject('INVALID_PREDICTION')
        result['candidate_bbox'] = box
        width, height = image_size
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            return reject('PREDICTION_OUTSIDE_IMAGE')
        original = anchor['box']
        w, h = original[2] - original[0], original[3] - original[1]
        ratio_w, ratio_h = (box[2] - box[0]) / w, (box[3] - box[1]) / h
        scale = max(abs(ratio_w - 1), abs(ratio_h - 1), abs(ratio_w * ratio_h - 1))
        diagonal = math.hypot(w, h)
        point, origin = center(box), center(original)
        displacement = math.dist(point, origin) / diagonal
        elapsed = timestamp - anchor['timestamp']
        expected = tuple(origin[k] + anchor['velocity'][k] * elapsed for k in range(2))
        trajectory = math.dist(point, expected) / diagonal
        overlap = box_iou(box, self.previous_box)
        result.update(previous_iou=overlap, scale_change=scale, center_displacement=displacement,
                      trajectory_error=trajectory)
        if displacement > cfg.max_center_displacement:
            return reject('CENTER_DISPLACEMENT')
        if overlap < cfg.min_previous_iou:
            return reject('LOW_PREVIOUS_IOU')
        if scale > cfg.max_box_change:
            return reject('BOX_SCALE_CHANGE')
        if trajectory > cfg.max_trajectory_error:
            return reject('TRAJECTORY_INCONSISTENCY')
        quality = (overlap + math.exp(-scale / cfg.max_box_change)
                   + math.exp(-displacement / cfg.max_center_displacement)
                   + math.exp(-trajectory / cfg.max_trajectory_error)) / 4
        confidence = anchor['confidence'] * cfg.confidence_decay ** self.gap * quality
        if pose_quality is not None:
            confidence *= pose_quality
            result['recovery_confidence'] = confidence
            if pose_quality < cfg.min_pose_quality:
                return reject('POSE_VALIDATION_FAILED')
        result['recovery_confidence'] = confidence
        if confidence < cfg.min_recovery_confidence:
            return reject('LOW_RECOVERY_CONFIDENCE')
        result.update(recovered_status='ACCEPTED', final_status='RECOVERED',
                      final_infant_bbox=box, final_infant_confidence=confidence)
        # Only continuity geometry changes: recoveries never refresh the anchor or gap budget.
        self.previous_box = box
        return result
