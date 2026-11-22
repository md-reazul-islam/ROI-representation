"""Simulated hand-held camera motion: smooth translation, rotation and zoom.

Design
    * One set of unit trajectories per video (same random seed for all severities), scaled by the severity
      amplitudes, so small/medium/large differ only in amplitude (clean dose-response comparison).
    * Drift: aperiodic low-pass noise (default cut-off 0.08 Hz), below the respiratory band, so the camera motion
      does not create respiration-like periodic movement.
    * Tremor: small band-limited jitter (default 6-10 Hz), above the heart-rate band.
    * Each frame is warped once from the original pixels (no accumulated resampling) with Lanczos-4
      interpolation; uncovered borders are filled according to ``border_mode``.
"""
from __future__ import annotations

import cv2
import numpy as np

from .video_utils import bandlimited_noise, smooth_noise

INTERP = {'lanczos4': cv2.INTER_LANCZOS4, 'cubic': cv2.INTER_CUBIC, 'linear': cv2.INTER_LINEAR}
BORDER = {'constant': cv2.BORDER_CONSTANT, 'reflect': cv2.BORDER_REFLECT_101, 'replicate': cv2.BORDER_REPLICATE}


def make_units(n, fps, cfg, rng):
    c = cfg['drift_cutoff_hz']
    lo, hi = cfg['tremor_band_hz']
    return {'dx': smooth_noise(n, fps, c, rng), 'dy': smooth_noise(n, fps, c, rng),
            'rot': smooth_noise(n, fps, c, rng), 'zoom': smooth_noise(n, fps, c, rng),
            'tx': bandlimited_noise(n, fps, lo, hi, rng), 'ty': bandlimited_noise(n, fps, lo, hi, rng)}


class MotionAugmenter:
    category = 'motion'

    def __init__(self, cfg, sev, units, meta, subject=None):
        self.cfg, self.sev = cfg, sev
        self.W, self.H = meta['width'], meta['height']
        self.center = (self.W / 2.0, self.H / 2.0)
        self.interp = INTERP[cfg['interpolation']]
        self.border = BORDER[cfg['border_mode']]
        self.dx = sev['translation_px'] * units['dx'] + sev['tremor_px_rms'] * units['tx']
        self.dy = sev['translation_px'] * units['dy'] + sev['tremor_px_rms'] * units['ty']
        self.angle = sev['rotation_deg'] * units['rot']
        self.scale = 1.0 + sev['zoom_frac'] * units['zoom']
        self.ones = np.full((self.H, self.W), 255, np.uint8)
        self.rows = []

    def matrix(self, i):
        M = cv2.getRotationMatrix2D(self.center, float(self.angle[i]), float(self.scale[i]))
        M[0, 2] += self.dx[i]
        M[1, 2] += self.dy[i]
        return M

    def apply(self, frame, i):
        M = self.matrix(i)
        out = cv2.warpAffine(frame, M, (self.W, self.H), flags=self.interp, borderMode=self.border,
                             borderValue=(0, 0, 0))
        cover = cv2.warpAffine(self.ones, M, (self.W, self.H), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.rows.append({'frame': i, 'dx_px': round(float(self.dx[i]), 4), 'dy_px': round(float(self.dy[i]), 4),
                          'rotation_deg': round(float(self.angle[i]), 5), 'scale': round(float(self.scale[i]), 6),
                          'border_fraction': round(1.0 - float(cover.mean()) / 255.0, 5)})
        return out

    def control_signals(self):
        return {'dx_px': self.dx, 'dy_px': self.dy, 'rotation_deg': self.angle, 'scale': self.scale}

    def summary(self):
        b = np.array([r['border_fraction'] for r in self.rows])
        return {'peak_abs_dx_px': round(float(np.abs(self.dx).max()), 2),
                'peak_abs_dy_px': round(float(np.abs(self.dy).max()), 2),
                'peak_abs_rotation_deg': round(float(np.abs(self.angle).max()), 3),
                'scale_range': [round(float(self.scale.min()), 4), round(float(self.scale.max()), 4)],
                'border_fraction_mean': round(float(b.mean()), 5), 'border_fraction_max': round(float(b.max()), 5)}


def identity_check(frame, cfg):
    """A zero-amplitude warp must return the input unchanged (checks that the warp adds nothing by itself)."""
    W, H = frame.shape[1], frame.shape[0]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), 0.0, 1.0)
    out = cv2.warpAffine(frame, M, (W, H), flags=INTERP[cfg['interpolation']], borderMode=BORDER[cfg['border_mode']])
    return bool(np.array_equal(out, frame))
