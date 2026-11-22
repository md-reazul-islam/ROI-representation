"""Simulated partial occlusion: a moving hand-shaped occluder plus a blanket/cloth layer rising from the bottom.

These are SIMULATED occlusions (synthetic, procedurally generated textures composited over the video). They are
not real hands, real blankets or real swaddling: the occluders carry no pulse, cast only an approximate shadow and
do not deform with the body.

Severity = fraction of the image area covered (union of the hand and blanket alpha masks). The hand has a fixed
share of the target area; the blanket level is solved per frame so that the union covers the target fraction.
The hand path and blanket edge drift slowly (aperiodic low-pass noise, default < 0.08 Hz), so the occluders do not
move up and down at respiratory or cardiac rates. Textures are fixed in object coordinates (no per-frame noise).
"""
from __future__ import annotations

import cv2
import numpy as np

from .video_utils import smooth_noise


def make_units(n, fps, cfg, rng):
    c = cfg['drift_cutoff_hz']
    u = {k: smooth_noise(n, fps, c, rng) for k in ('hx', 'hy', 'hrot', 'btx', 'w0', 'w1', 'w2')}
    u['tex_seed'] = int(rng.integers(0, 2 ** 31 - 1))
    return u


# --------------------------------------------------------------------------------------------------- textures
def _lowfreq(shape, sigma, rng):
    x = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return x / max(float(np.abs(x).max()), 1e-6)


def render_hand(size, skin_bgr, rng, ss=2):
    """RGBA hand template (palm + four fingers + thumb + wrist), anti-aliased, with shading for a 3-D look."""
    s = size * ss
    Wc, Hc = int(1.7 * s), int(2.3 * s)
    m = np.zeros((Hc, Wc), np.uint8)
    cx, cy = Wc * 0.5, Hc * 0.58
    cv2.ellipse(m, (int(cx), int(cy)), (int(0.5 * s), int(0.56 * s)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    top = cy - 0.42 * s
    for off, ln, ang in zip((-0.35, -0.12, 0.12, 0.35), (0.72, 0.92, 0.86, 0.66), (-11, -3, 3, 10)):
        bx, by = cx + off * s, top + 0.12 * s
        a = np.deg2rad(ang)
        tx, ty = bx + np.sin(a) * ln * s, by - np.cos(a) * ln * s
        th = int(0.21 * s)
        cv2.line(m, (int(bx), int(by)), (int(tx), int(ty)), 255, th, cv2.LINE_AA)
        cv2.circle(m, (int(tx), int(ty)), th // 2, 255, -1, cv2.LINE_AA)
    bx, by = cx - 0.42 * s, cy + 0.05 * s
    tx, ty = bx - 0.45 * s, by - 0.42 * s
    cv2.line(m, (int(bx), int(by)), (int(tx), int(ty)), 255, int(0.25 * s), cv2.LINE_AA)
    cv2.circle(m, (int(tx), int(ty)), int(0.125 * s), 255, -1, cv2.LINE_AA)
    cv2.rectangle(m, (int(cx - 0.36 * s), int(cy + 0.3 * s)), (int(cx + 0.36 * s), Hc - 1), 255, -1)
    # wrist fades out over its last part so the template has no hard cut
    fade = np.ones(Hc, np.float32)
    f0 = int(Hc - 0.35 * s)
    fade[f0:] = np.linspace(1, 0, Hc - f0)
    alpha = (m.astype(np.float32) / 255.0) * fade[:, None]
    alpha = cv2.resize(alpha, (Wc // ss, Hc // ss), interpolation=cv2.INTER_AREA)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)
    h, w = alpha.shape
    # shading: rounded edges darker, light from the top, low-frequency skin variation, fine fixed grain
    dist = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    edge = 0.82 + 0.18 * np.clip(dist / (0.08 * size), 0, 1)
    grad = np.linspace(1.05, 0.9, h, dtype=np.float32)[:, None]
    shade = edge * grad * (1 + 0.06 * _lowfreq((h, w), size * 0.15, rng))
    rgb = np.clip(np.array(skin_bgr, np.float32)[None, None, :] * shade[:, :, None]
                  + rng.normal(0, 1.5, (h, w, 3)).astype(np.float32), 0, 255)
    anchor = (w * 0.5, h * 0.58)
    return rgb, alpha.astype(np.float32), anchor


def render_cloth(h, w, base_bgr, rng):
    """Blanket/cloth texture: fine weave, thread noise and soft folds."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.035 * np.sin(2 * np.pi * x / 5.0) * np.sin(2 * np.pi * y / 5.0)
    theta = rng.uniform(0, np.pi)
    folds = 0.07 * np.sin(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / 90.0
                          + 3.0 * _lowfreq((h, w), 40, rng))
    soft = 0.10 * _lowfreq((h, w), 30, rng)
    shade = 1 + weave + folds + soft
    rgb = np.array(base_bgr, np.float32)[None, None, :] * shade[:, :, None] + rng.normal(0, 2.0, (h, w, 3))
    return np.clip(rgb, 0, 255).astype(np.float32)


# ------------------------------------------------------------------------------------------------- augmenter
class OcclusionAugmenter:
    category = 'occlusion'

    def __init__(self, cfg, sev, units, meta, subject):
        self.cfg, self.sev = cfg, sev
        W, H = self.W, self.H = meta['width'], meta['height']
        self.area_target = float(sev['area_fraction'])
        rng = np.random.default_rng(units['tex_seed'])
        # hand: scale template so its alpha area is hand_share of the target area
        hand_area = sev['hand_share'] * self.area_target * W * H
        _, a_ref, _ = render_hand(100, cfg['hand_skin_bgr'], np.random.default_rng(0))
        size = 100 * np.sqrt(hand_area / a_ref.sum())
        self.hand_rgb, self.hand_a, self.anchor = render_hand(size, cfg['hand_skin_bgr'], rng)
        self.hand_size = float(size)
        # hand path around the detected face
        fx, fy = subject['x'] + subject['w'] / 2, subject['y'] + subject['h'] / 2
        self.hx = np.clip(fx + cfg['hand_path_x_face'] * subject['w'] * units['hx'], 0, W - 1)
        self.hy = np.clip(fy + cfg['hand_offset_y_face'] * subject['h']
                          + cfg['hand_path_y_face'] * subject['h'] * units['hy'], 0, H - 1)
        self.hang = cfg['hand_base_angle_deg'] + cfg['hand_rotation_deg'] * units['hrot']
        # blanket
        m = int(cfg['blanket_margin_px'])
        self.cloth = render_cloth(H + 2 * m, W + 2 * m, cfg['blanket_bgr'], rng)
        self.margin = m
        self.btx = cfg['blanket_drift_px'] * units['btx']
        sc = H / 480.0
        self.wave_amp = [a * sc for a in cfg['blanket_wave_amp_px']]
        self.wave_len = [l * sc for l in cfg['blanket_wave_len_px']]
        self.wave_ph = [units['w0'] * cfg['blanket_wave_phase_drift_rad'], units['w1'] * cfg['blanket_wave_phase_drift_rad'],
                        units['w2'] * cfg['blanket_wave_phase_drift_rad']]
        self.ph0 = rng.uniform(0, 2 * np.pi, 3)
        self.feather = float(cfg['blanket_feather_px'])
        self.xs = np.arange(W, dtype=np.float32)
        self.ys = np.arange(H, dtype=np.float32)[:, None]
        self.shadow_strength = float(cfg['hand_shadow_strength'])
        self.shadow_offset = cfg['hand_shadow_offset_px']
        self.rows, self.level = [], []

    def _hand(self, i):
        M = cv2.getRotationMatrix2D(self.anchor, float(self.hang[i]), 1.0)
        M[0, 2] += self.hx[i] - self.anchor[0]
        M[1, 2] += self.hy[i] - self.anchor[1]
        a = cv2.warpAffine(self.hand_a, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=0)
        rgb = cv2.warpAffine(self.hand_rgb, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return a, rgb

    def _edge(self, i):
        e = np.zeros(self.W, np.float32)
        for k in range(3):
            e += self.wave_amp[k] * np.sin(2 * np.pi * self.xs / self.wave_len[k] + self.ph0[k] + self.wave_ph[k][i])
        return e

    def _blanket_alpha(self, level, edge):
        return np.clip((self.ys - (level + edge[None, :])) / self.feather + 0.5, 0, 1)

    def _solve_level(self, a_hand, edge):
        """Blanket level so that the union of hand and blanket covers the target area fraction."""
        inv_h = 1.0 - a_hand
        lo, hi = -float(np.abs(edge).max()) - self.feather, self.H + float(np.abs(edge).max()) + self.feather
        cov = lambda L: 1.0 - float((inv_h * (1.0 - self._blanket_alpha(L, edge))).mean())
        if cov(hi) >= self.area_target:
            return hi, False
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            if cov(mid) > self.area_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), True

    def apply(self, frame, i):
        a_h, rgb_h = self._hand(i)
        edge = self._edge(i)
        level, ok = self._solve_level(a_h, edge)
        a_b = self._blanket_alpha(level, edge)
        m = self.margin
        cloth = cv2.getRectSubPix(self.cloth, (self.W, self.H),
                                  (m + self.W / 2.0 + float(self.btx[i]), m + self.H / 2.0))
        out = frame.astype(np.float32)
        # approximate soft shadow of the hand on the scene (the shadow region stays visible, it is not counted)
        if self.shadow_strength > 0:
            S = np.float32([[1, 0, self.shadow_offset[0]], [0, 1, self.shadow_offset[1]]])
            sh = cv2.GaussianBlur(cv2.warpAffine(a_h, S, (self.W, self.H)), (0, 0), 7) * self.shadow_strength
            out *= (1.0 - sh)[:, :, None]
        out = out * (1 - a_b[:, :, None]) + cloth * a_b[:, :, None]
        out = out * (1 - a_h[:, :, None]) + rgb_h * a_h[:, :, None]
        union = 1.0 - (1.0 - a_h) * (1.0 - a_b)
        self.level.append(level)
        self.rows.append({'frame': i, 'hand_x': round(float(self.hx[i]), 3), 'hand_y': round(float(self.hy[i]), 3),
                          'hand_angle_deg': round(float(self.hang[i]), 4), 'blanket_level_px': round(level, 3),
                          'blanket_edge_center_px': round(level + float(edge[self.W // 2]), 3),
                          'coverage': round(float(union.mean()), 5), 'coverage_hand': round(float(a_h.mean()), 5),
                          'coverage_blanket': round(float(a_b.mean()), 5),
                          'coverage_binary': round(float((union > 0.5).mean()), 5), 'hand_alone_exceeds_target': not ok})
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    def control_signals(self):
        r = self.rows
        return {'hand_x': self.hx[:len(r)], 'hand_y': self.hy[:len(r)], 'hand_angle_deg': self.hang[:len(r)],
                'blanket_level_px': np.array([x['blanket_level_px'] for x in r]),
                'blanket_edge_center_px': np.array([x['blanket_edge_center_px'] for x in r]),
                'coverage': np.array([x['coverage'] for x in r])}

    def summary(self):
        c = np.array([x['coverage'] for x in self.rows])
        return {'target_area_fraction': self.area_target, 'coverage_mean': round(float(c.mean()), 4),
                'coverage_min': round(float(c.min()), 4), 'coverage_max': round(float(c.max()), 4),
                'coverage_hand_mean': round(float(np.mean([x['coverage_hand'] for x in self.rows])), 4),
                'coverage_blanket_mean': round(float(np.mean([x['coverage_blanket'] for x in self.rows])), 4),
                'frames_hand_alone_exceeds_target': int(sum(x['hand_alone_exceeds_target'] for x in self.rows)),
                'hand_palm_width_px': round(self.hand_size, 1)}
