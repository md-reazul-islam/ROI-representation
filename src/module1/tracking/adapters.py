"""Adapters around the installed Ultralytics tracking engines; no detector inference."""
from dataclasses import dataclass, asdict, replace
import inspect
import math
from types import SimpleNamespace

import numpy as np

from module1.detection.fusion import box_iou, detection_key, validate_detection
from .base import BaseTracker, Track, TrackingFrame


@dataclass(frozen=True)
class TrackingConfig:
    track_high_thresh: float = .25
    track_low_thresh: float = .1
    new_track_thresh: float = .25
    match_thresh: float = .8
    track_buffer: int = 30
    min_hits: int = 3
    primary_hold_frames: int = 15
    nms_iou: float = .8
    class_id: int = 0
    fuse_score: bool = True
    gmc_method: str = 'sparseOptFlow'
    proximity_thresh: float = .5
    appearance_thresh: float = .8

    def __post_init__(self):
        for name in ('track_low_thresh', 'track_high_thresh', 'new_track_thresh', 'match_thresh',
                     'nms_iou', 'proximity_thresh', 'appearance_thresh'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if not self.track_low_thresh < self.track_high_thresh <= self.new_track_thresh:
            raise ValueError('Require low < high <= new track thresholds')
        for name in ('track_buffer', 'min_hits', 'primary_hold_frames', 'class_id'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.min_hits < 1 or self.primary_hold_frames > self.track_buffer:
            raise ValueError('min_hits must be positive and hold must not exceed track_buffer')
        if self.gmc_method not in ('sparseOptFlow', 'orb', 'sift', 'ecc', 'none'):
            raise ValueError('Unknown global motion compensation method')


class _UltralyticsAdapter(BaseTracker):
    backend_name = ''

    def __init__(self, config=None):
        self.config = config or TrackingConfig()
        self.reset()

    def reset(self):
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.bot_sort import BOTSORT
        backend = BYTETracker if self.backend_name == 'bytetrack' else BOTSORT
        args = SimpleNamespace(**asdict(self.config), with_reid=False, model='auto')
        # Upstream constructors reset a process-global counter. Preserve it so
        # constructing another adapter cannot collide with an active engine's IDs.
        previous_counter = BaseTrack._count
        kwargs = {'frame_rate': 30} if 'frame_rate' in inspect.signature(backend).parameters else {}
        self.engine = backend(args, **kwargs)
        BaseTrack._count = max(previous_counter, BaseTrack._count)
        self.engine.max_frames_lost = self.config.track_buffer
        self._history = {}
        self._step = 0
        self._last_frame = None
        self._primary = None

    def _filter(self, detections):
        kept = []
        for d in sorted(detections, key=detection_key):
            if d.class_id != self.config.class_id:
                continue
            if not any(box_iou(d.bbox, old.bbox) > self.config.nms_iou for old in kept):
                kept.append(d)
        return kept

    def update(self, detections, frame):
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.basetrack import TrackState
        if not isinstance(frame, TrackingFrame):
            raise TypeError('frame must be TrackingFrame(image, frame_index, timestamp)')
        if not isinstance(frame.image, np.ndarray) or frame.image.ndim != 3 or frame.image.shape[2] != 3 or not frame.image.size:
            raise ValueError('frame.image must be a nonempty HxWx3 BGR image')
        if not isinstance(frame.frame_index, int) or frame.frame_index < 0 or not math.isfinite(frame.timestamp) or frame.timestamp < 0:
            raise ValueError('Invalid frame index or timestamp')
        if self._last_frame and (frame.frame_index <= self._last_frame.frame_index or frame.timestamp <= self._last_frame.timestamp):
            raise ValueError('Frames and timestamps must increase strictly; reset for a new video')
        for d in detections:
            validate_detection(d)
            if d.frame_index != frame.frame_index or abs(d.timestamp - frame.timestamp) > 1e-8:
                raise ValueError('Detection and frame metadata differ')
        filtered = self._filter(detections)
        array = np.asarray([[*d.bbox, d.confidence, d.class_id] for d in filtered], dtype=np.float32).reshape(-1, 6)
        self.engine.update(Boxes(array, frame.image.shape[:2]), frame.image)
        self._step += 1
        self._last_frame = frame
        active = {int(t.track_id): t for t in self.engine.tracked_stracks if t.state == TrackState.Tracked}
        lost = {int(t.track_id): t for t in self.engine.lost_stracks if t.state == TrackState.Lost}
        outputs = []
        for backend_id, item in {**lost, **active}.items():
            observed = backend_id in active and item.frame_id == self.engine.frame_id
            h = self._history.get(backend_id)
            if h is None:
                h = dict(id=len(self._history) + 1, start=self._step, hits=0, misses=0,
                         streak=0, last_seen=self._step, confirmed=False, score_sum=0.,
                         center=None, timestamp=None, velocity=(0., 0.), retired=False)
                self._history[backend_id] = h
            if h['retired']:
                continue
            box = tuple(float(v) for v in item.xyxy)
            if observed:
                center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                if h['center'] is not None:
                    elapsed = frame.timestamp - h['timestamp']
                    h['velocity'] = tuple((center[k] - h['center'][k]) / elapsed for k in range(2))
                h['center'], h['timestamp'] = center, frame.timestamp
                h['hits'] += 1
                h['streak'] += 1
                h['last_seen'] = self._step
                h['score_sum'] += float(item.score)
                h['confirmed'] |= h['streak'] >= self.config.min_hits and bool(item.is_activated)
            else:
                h['misses'] += 1
                h['streak'] = 0
            consecutive = self._step - h['last_seen']
            age = self._step - h['start'] + 1
            state = 'CONFIRMED' if h['confirmed'] else 'TENTATIVE'
            if not observed:
                state = 'LOST'
            if consecutive > self.config.track_buffer:
                state = 'REMOVED'
                h['retired'] = True
            reliability = (h['score_sum'] / max(1, h['hits'])) * h['hits'] / age
            outputs.append(Track(h['id'], box, float(item.score) if observed else 0., frame.frame_index,
                                 frame.timestamp, h['velocity'], age, h['hits'], h['misses'], state,
                                 consecutive, False, observed, reliability, int(item.cls)))
        # Emit removal once for tracks that the backend discarded (including tentative tracks).
        visible_ids = {t.track_id for t in outputs}
        for h in self._history.values():
            if h['id'] not in visible_ids and not h['retired']:
                h['retired'] = True
                old = h.get('last_output')
                if old:
                    h['misses'] += 1
                    outputs.append(replace(old, frame_index=frame.frame_index, timestamp=frame.timestamp,
                                           age=self._step - h['start'] + 1, number_of_misses=h['misses'],
                                           consecutive_misses=self._step - h['last_seen'], track_state='REMOVED',
                                           observed=False, confidence=0., is_primary=False))
        primary = next((t for t in outputs if t.track_id == self._primary), None)
        if primary is None or primary.track_state == 'REMOVED' or primary.consecutive_misses > self.config.primary_hold_frames:
            self._primary = None
        if self._primary is None:
            eligible = [t for t in outputs if t.track_state == 'CONFIRMED' and t.observed]
            if eligible:
                self._primary = max(eligible, key=lambda t: (t.reliability, t.number_of_hits, t.age, -t.track_id)).track_id
        outputs = [replace(t, is_primary=t.track_id == self._primary) for t in outputs]
        for h in self._history.values():
            match = next((t for t in outputs if t.track_id == h['id']), None)
            if match:
                h['last_output'] = match
        return sorted(outputs, key=lambda t: t.track_id)


class ByteTrackAdapter(_UltralyticsAdapter):
    backend_name = 'bytetrack'


class BoTSORTAdapter(_UltralyticsAdapter):
    backend_name = 'botsort'
