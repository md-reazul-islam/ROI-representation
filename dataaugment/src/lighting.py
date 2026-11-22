"""Simulated illumination changes: exposure drift, slow spatial/colour fluctuations, flicker and a moving shadow.

All changes are multiplicative gains applied in (approximately) linear light (sRGB-like gamma 2.2), so a scene
point keeps its relative (AC/DC) pulse modulation except where pixels clip at 0 or 255; clipping is reported.

Components (per frame t, pixel x, channel c):
    G_c(x, t) = exposure(t) * flicker(t) * colour_c(t) * gradient(x, t) * (1 - shadow(x, t))
    * exposure: 2**(EV(t)), EV an aperiodic slow drift (default cut-off 0.05 Hz)
    * gradient: a slowly rotating linear illumination gradient (side lamp)
    * colour: slow warm/cool shift (blue and red gains in opposite directions)
    * flicker: aperiodic, (a) small frame-wise gain noise and (b) sparse random dips (lamp instability) with fast
      recovery. Mains flicker (100/120 Hz) aliased by a ~29 fps camera produces a narrow line at a few Hz that
      can mimic a heart rate; it is therefore OFF by default (``mains_alias.enabled``).
    * shadow: a soft elliptical shadow drifting across the face/body.
"""
from __future__ import annotations

import numpy as np

from .video_utils import smooth_noise

GAMMA = 2.2
TO_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** GAMMA).astype(np.float32)


def make_units(n, fps, cfg, rng):
    c_slow, c_shadow = cfg['drift_cutoff_hz'], cfg['shadow_cutoff_hz']
    u = {'ev': smooth_noise(n, fps, c_slow, rng), 'grad': smooth_noise(n, fps, c_slow, rng),
         'theta': smooth_noise(n, fps, c_slow, rng), 'ct': smooth_noise(n, fps, c_slow, rng),
         'sx': smooth_noise(n, fps, c_shadow, rng), 'sy': smooth_noise(n, fps, c_shadow, rng),
         'flicker': rng.standard_normal(n)}
    # dip events at the maximum rate with a uniform mark; each severity keeps marks < rate/max_rate (nested sets)
    rmax = cfg['dip_rate_max_hz']
    times, t = [], rng.exponential(1.0 / rmax)
    while t < n / fps:
        times.append(t)
        t += rng.exponential(1.0 / rmax)
    u['dip_t'] = np.array(times)
    u['dip_mark'] = rng.uniform(0, 1, len(times))
    u['theta0'] = float(rng.uniform(0, 2 * np.pi))
    return u


class LightingAugmenter:
    category = 'lighting'

    def __init__(self, cfg, sev, units, meta, subject):
        self.cfg, self.sev = cfg, sev
        W, H, fps = self.W, self.H, self.fps = meta['width'], meta['height'], meta['fps']
        n = len(units['ev'])
        t = np.arange(n) / fps
        self.ev = sev['exposure_ev'] * units['ev']
        # flicker: frame-wise gain noise + nested sparse dips
        fl = 1.0 + sev['flicker_sigma'] * units['flicker']
        keep = units['dip_mark'] < sev['dip_rate_hz'] / cfg['dip_rate_max_hz']
        dip = np.zeros(n)
        for t0 in units['dip_t'][keep]:
            m = t >= t0
            dip[m] += np.exp(-(t[m] - t0) / cfg['dip_recovery_s'])
        self.dip = sev['dip_depth'] * np.clip(dip, 0, 1)
        self.n_dips = int(keep.sum())
        self.flicker = fl * (1.0 - self.dip)
        mains = cfg.get('mains_alias', {})
        if mains.get('enabled', False):
            f_alias = abs(mains['mains_hz'] * 2 - round(mains['mains_hz'] * 2 / fps) * fps)
            self.flicker *= 1.0 + sev['flicker_sigma'] * mains['relative_amplitude'] * np.sin(2 * np.pi * f_alias * t)
            self.mains_alias_hz = float(f_alias)
        else:
            self.mains_alias_hz = None
        self.global_gain = 2.0 ** self.ev * self.flicker
        self.grad_amp = sev['gradient_amp'] * units['grad']
        self.theta = units['theta0'] + 0.5 * np.pi * units['theta']
        self.ct = sev['color_temp_amp'] * units['ct']
        fx, fy = subject['x'] + subject['w'] / 2, subject['y'] + subject['h'] / 2
        self.sx = fx + cfg['shadow_path_x_face'] * subject['w'] * units['sx']
        self.sy = fy + cfg['shadow_path_y_face'] * subject['h'] * units['sy']
        self.sig = (cfg['shadow_sigma_face'][0] * subject['w'], cfg['shadow_sigma_face'][1] * subject['h'])
        self.shadow_strength = float(sev['shadow_strength'])
        self.xn = ((np.arange(W, dtype=np.float32) - W / 2) / (W / 2))[None, :]
        self.yn = ((np.arange(H, dtype=np.float32) - H / 2) / (H / 2))[:, None]
        self.X = np.arange(W, dtype=np.float32)[None, :]
        self.Y = np.arange(H, dtype=np.float32)[:, None]
        self.rows = []

    def gain_map(self, i):
        g = 1.0 + self.grad_amp[i] * (self.xn * np.cos(self.theta[i]) + self.yn * np.sin(self.theta[i]))
        r2 = ((self.X - self.sx[i]) / self.sig[0]) ** 2 + ((self.Y - self.sy[i]) / self.sig[1]) ** 2
        sh = self.shadow_strength * np.exp(-0.5 * r2 ** 1.5)          # flat-topped soft shadow
        return np.maximum(g, 0).astype(np.float32) * (1.0 - sh).astype(np.float32) * np.float32(self.global_gain[i])

    def apply(self, frame, i):
        G = self.gain_map(i)
        ch = np.array([1.0 - self.ct[i], 1.0, 1.0 + self.ct[i]], np.float32)   # B, G, R
        lin = TO_LIN[frame] * G[:, :, None] * ch[None, None, :]
        out = np.rint(255.0 * np.clip(lin, 0, 1) ** (1.0 / GAMMA)).astype(np.uint8)
        self.rows.append({'frame': i, 'exposure_ev': round(float(self.ev[i]), 5),
                          'flicker_gain': round(float(self.flicker[i]), 5), 'dip': round(float(self.dip[i]), 5),
                          'global_gain': round(float(self.global_gain[i]), 5),
                          'color_temp': round(float(self.ct[i]), 5), 'gradient_amp': round(float(self.grad_amp[i]), 5),
                          'gradient_theta_rad': round(float(self.theta[i]), 4),
                          'shadow_x': round(float(self.sx[i]), 2), 'shadow_y': round(float(self.sy[i]), 2),
                          'saturated_frac_in': round(float((frame.max(axis=2) == 255).mean()), 5),
                          'saturated_frac_out': round(float((out.max(axis=2) == 255).mean()), 5),
                          'black_frac_out': round(float((out.min(axis=2) == 0).mean()), 5)})
        return out

    def control_signals(self):
        return {'log_global_gain': np.log(self.global_gain), 'exposure_ev': self.ev, 'gradient_amp': self.grad_amp,
                'color_temp': self.ct, 'shadow_x': self.sx, 'shadow_y': self.sy}

    def summary(self):
        r = self.rows
        return {'gain_range': [round(float(self.global_gain.min()), 4), round(float(self.global_gain.max()), 4)],
                'exposure_ev_range': [round(float(self.ev.min()), 3), round(float(self.ev.max()), 3)],
                'dip_events': self.n_dips,
                'mains_alias_hz': self.mains_alias_hz,
                'saturated_frac_in_mean': round(float(np.mean([x['saturated_frac_in'] for x in r])), 5),
                'saturated_frac_out_mean': round(float(np.mean([x['saturated_frac_out'] for x in r])), 5),
                'saturated_frac_out_max': round(float(np.max([x['saturated_frac_out'] for x in r])), 5),
                'black_frac_out_max': round(float(np.max([x['black_frac_out'] for x in r])), 5)}


def identity_check(frame):
    """Unit gain must return the input unchanged (checks the linear-light round trip)."""
    out = np.rint(255.0 * np.clip(TO_LIN[frame], 0, 1) ** (1.0 / GAMMA)).astype(np.uint8)
    return bool(np.array_equal(out, frame))
