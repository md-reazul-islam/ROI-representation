from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PoseQualityConfig:
    confidence_threshold: float = .5
    min_confident_fraction: float = .1
    min_body_keypoints: int = 4
    min_bone_length: float = .005
    max_bone_length: float = .8
    max_bilateral_ratio: float = 6.
    max_keypoint_speed: float = 3.
    max_temporal_gap_seconds: float = .5
    min_temporal_keypoints: int = 4

    def __post_init__(self):
        for name in ('confidence_threshold', 'min_confident_fraction'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        for name in ('min_body_keypoints', 'min_temporal_keypoints'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 308:
                raise ValueError(f'{name} must be an integer in [1,308]')
        for name in ('min_bone_length', 'max_bone_length', 'max_bilateral_ratio',
                     'max_keypoint_speed', 'max_temporal_gap_seconds'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.min_bone_length >= self.max_bone_length or self.max_bilateral_ratio < 1:
            raise ValueError('Invalid bone-length limits')


BODY_EDGES = tuple((f'{side}_{a}', f'{side}_{b}')
                   for side in ('left', 'right')
                   for a, b in (('shoulder', 'elbow'), ('elbow', 'wrist'),
                                ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')))


class PoseSanityChecker:
    """Conservative 2D diagnostics, not a neonatal anatomical validity classifier."""

    def __init__(self, names, config=None):
        if len(names) != 308 or len(set(names)) != 308:
            raise ValueError('Expected 308 unique official keypoint names')
        self.ids = {name: i for i, name in enumerate(names)}
        self.config = config or PoseQualityConfig()
        self.reset()

    def reset(self):
        self.previous = None
        self.last = None

    def check(self, xy, scores, bbox, image_size, frame_index, timestamp):
        cfg = self.config
        if xy.shape != (308, 2) or scores.shape != (308,):
            raise ValueError('Sapiens2 must return exactly 308 keypoints and scores')
        if self.last is not None and (frame_index <= self.last[0] or timestamp <= self.last[1]):
            raise ValueError('Pose frames must increase strictly; reset between videos')
        self.last = (frame_index, timestamp)
        width, height = image_size
        finite = np.isfinite(xy).all(axis=1) & np.isfinite(scores)
        inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        confident = inside & (scores >= cfg.confidence_threshold)
        fraction = float(confident.mean())
        checks = dict(keypoint_count='PASS', coordinate_validity='PASS' if inside.all() else 'FAIL',
                      invalid_coordinate_count=int((~inside).sum()), confident_keypoints=int(confident.sum()),
                      confident_fraction=fraction,
                      confidence_threshold='PASS' if fraction >= cfg.min_confident_fraction else 'FAIL')
        diagonal = math.hypot(bbox[2]-bbox[0], bbox[3]-bbox[1])
        lengths = {}
        body = {name for edge in BODY_EDGES for name in edge}
        body_count = sum(bool(confident[self.ids[name]]) for name in body if name in self.ids)
        for a, b in BODY_EDGES:
            if a in self.ids and b in self.ids and confident[self.ids[a]] and confident[self.ids[b]]:
                lengths[(a, b)] = float(np.linalg.norm(xy[self.ids[a]]-xy[self.ids[b]]) / diagonal)
        anatomical = 'NOT_ASSESSED'
        if body_count >= cfg.min_body_keypoints and len(lengths) >= 2:
            good = all(cfg.min_bone_length <= length <= cfg.max_bone_length for length in lengths.values())
            for (a, b), length in lengths.items():
                if a.startswith('left_'):
                    opposite = lengths.get((a.replace('left_', 'right_'), b.replace('left_', 'right_')))
                    if opposite is not None:
                        good &= max(length, opposite) / max(min(length, opposite), 1e-12) <= cfg.max_bilateral_ratio
            anatomical = 'PASS' if good else 'FAIL'
        checks.update(anatomical_plausibility=anatomical, confident_body_keypoints=body_count,
                      assessed_bones=len(lengths), temporal_plausibility='NOT_ASSESSED',
                      temporal_reason='NO_PREVIOUS_VALID_POSE', temporal_speed=None)
        if self.previous is not None:
            old_xy, old_mask, old_time, old_diag = self.previous
            dt = timestamp-old_time
            common = confident & old_mask
            if dt > cfg.max_temporal_gap_seconds:
                checks['temporal_reason'] = 'GAP_TOO_LONG'
                self.previous = None
            elif common.sum() < cfg.min_temporal_keypoints:
                checks['temporal_reason'] = 'INSUFFICIENT_COMMON_KEYPOINTS'
            else:
                speed = float(np.quantile(np.linalg.norm(xy[common]-old_xy[common], axis=1), .9)
                              / max(diagonal, old_diag) / dt)
                checks.update(temporal_speed=speed, temporal_reason='',
                              temporal_plausibility='PASS' if speed <= cfg.max_keypoint_speed else 'FAIL')
        failed = any(checks[key] == 'FAIL' for key in
                     ('coordinate_validity', 'confidence_threshold', 'anatomical_plausibility', 'temporal_plausibility'))
        confidence = float(np.mean(scores[np.isfinite(scores)])) if np.isfinite(scores).any() else 0.
        # Unknown anatomy is not promoted to a valid pose; raw keypoints remain available.
        quality = float(np.clip(scores[confident], 0, 1).mean() * fraction) if confident.any() else 0.
        if failed or anatomical == 'NOT_ASSESSED':
            quality = 0.
        if not failed and anatomical == 'PASS':
            self.previous = (xy.copy(), confident.copy(), timestamp, diagonal)
        status = 'FLAGGED' if failed else ('INSUFFICIENT_EVIDENCE' if anatomical == 'NOT_ASSESSED' else 'ESTIMATED')
        return confidence, quality, checks, status
