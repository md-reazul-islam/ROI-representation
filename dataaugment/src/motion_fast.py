"""Fast, large repositioning motion: the subject jumps quickly to new positions and stays there.

Purpose
    The slow-drift ``motion`` category (cut-off 0.08 Hz) never moves the face out of a fixed ROI, so it cannot
    separate a fixed box from a tracked ROI. This category does: the image content is shifted by up to
    ``translation_px`` in quick moves (0.25-0.6 s), so the face leaves part (small), most (medium) or all (large)
    of a box fixed on the first frame, while staying inside the image.

Design
    * Repositioning events at random (Poisson) times, mean interval ``event_interval_s`` with a minimum gap
      ``min_gap_s``; no regular rhythm, so no spectral line (checked by the narrowband test in main.py).
    * Each move follows a minimum-jerk profile (smooth start/stop, peak speed ~1.9x mean speed) to a target drawn
      uniformly in an ellipse (x radius 1, y radius ``y_ratio``); rotation and zoom jump with the same events.
    * The trajectory starts at the original position (frame 0 unchanged geometry), so a box fixed on the first
      frame fits the subject at the start.
    * Small slow drift (0.08 Hz) and tremor (6-10 Hz) as in ``motion`` are added on top.
    * One set of unit trajectories for all severities (same seed); severities differ only in amplitude.
    * Warping is identical to ``motion`` (one Lanczos-4 affine warp per frame from the original pixels).
"""
from __future__ import annotations

import numpy as np

from .motion import MotionAugmenter, identity_check  # noqa: F401  (identity_check re-exported)
from .video_utils import bandlimited_noise, smooth_noise


def _event_times(n, fps, interval, min_gap, rng, first_s):
    t, times, T = first_s, [], n / fps
    while True:
        t += min_gap + rng.exponential(max(interval - min_gap, 1e-3))
        if t >= T - 0.5:
            return times
        times.append(t)


def _min_jerk(u):
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def _jump_track(n, fps, times, durations, targets):
    """Piecewise hold / minimum-jerk transitions; starts at 0."""
    t = np.arange(n) / fps
    x = np.zeros(n)
    prev = 0.0
    for s, d, tgt in zip(times, durations, targets):
        u = np.clip((t - s) / d, 0, 1)
        x = np.where(t >= s, prev + (tgt - prev) * _min_jerk(u), x)
        prev = tgt
    return x


def make_units(n, fps, cfg, rng):
    times = _event_times(n, fps, cfg['event_interval_s'], cfg['min_gap_s'], rng, cfg['first_event_s'])
    k = len(times)
    lo, hi = cfg['move_duration_s']
    durations = rng.uniform(lo, hi, k)
    # targets uniform in the unit disc, then y squeezed (keeps the face inside the image vertically)
    r, a = np.sqrt(rng.uniform(0, 1, k)), rng.uniform(0, 2 * np.pi, k)
    tx, ty = r * np.cos(a), r * np.sin(a) * cfg['y_ratio']
    trot, tzoom = rng.uniform(-1, 1, k), rng.uniform(-1, 1, k)
    c = cfg['drift_cutoff_hz']
    tlo, thi = cfg['tremor_band_hz']
    return {'dx': _jump_track(n, fps, times, durations, tx), 'dy': _jump_track(n, fps, times, durations, ty),
            'rot': _jump_track(n, fps, times, durations, trot), 'zoom': _jump_track(n, fps, times, durations, tzoom),
            'drift_x': smooth_noise(n, fps, c, rng), 'drift_y': smooth_noise(n, fps, c, rng),
            'tx': bandlimited_noise(n, fps, tlo, thi, rng), 'ty': bandlimited_noise(n, fps, tlo, thi, rng),
            'event_times_s': np.array(times), 'move_durations_s': durations}


class FastMotionAugmenter(MotionAugmenter):
    category = 'motion_fast'

    def __init__(self, cfg, sev, units, meta, subject=None):
        u = dict(units)
        A, D = sev['translation_px'], sev['drift_px']
        # base class computes dx = A*dx + tremor*tx; fold the drift in through the unit tracks
        u['dx'] = units['dx'] + (D / A) * units['drift_x']
        u['dy'] = units['dy'] + (D / A) * units['drift_y']
        super().__init__(cfg, sev, u, meta, subject)
        self.events = units['event_times_s']
        self.durations = units['move_durations_s']
        self.subject = subject

    def summary(self):
        s = super().summary()
        v = np.hypot(np.diff(self.dx), np.diff(self.dy))
        s.update({'n_events': int(len(self.events)),
                  'mean_event_interval_s': round(float(np.diff(np.r_[0, self.events]).mean()), 3)
                  if len(self.events) else None,
                  'move_duration_s_range': [round(float(self.durations.min()), 3),
                                            round(float(self.durations.max()), 3)] if len(self.events) else None,
                  'peak_speed_px_per_frame': round(float(v.max()), 2),
                  'event_times_s': [round(float(t), 3) for t in self.events]})
        if self.subject:
            fw = float(self.subject['w'])
            s['peak_shift_in_face_widths'] = round(float(np.hypot(self.dx, self.dy).max()) / fw, 3)
        return s
