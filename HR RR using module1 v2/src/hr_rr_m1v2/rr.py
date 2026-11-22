"""RR branch of Module 2 v2.

Signal: Module 1 v2 camera-compensated residual motion of every chest / abdomen candidate (all candidates, not only
the HR selection), at the sampled frame rate and native resolution: velocity -> displacement -> principal breathing
axis -> band-pass 20-90 breaths/min (Module 2 v1 respiratory_signal).
Fusion: softmax(q_v2 / T) of the regions present in the window, gated by a minimum breathing amplitude; no
same-window SNR weighting. Windows are causal (growing 8 -> 15 s, 1 s step; Module 2 v1 rr_windows).
Validity: breathing-present test against block-shuffled surrogates of the same traces fused with the same weights,
and chest-vs-abdomen agreement; Kalman track over validated windows only.
Camera confound (added after NEW_1104, a hand-held phone video): periodic camera sway passes the shuffle null and moves
chest and abdomen together, so it passed both tests. A present window whose rate is also the rate of the camera motion
(Module 1 v2 global_dx/dy) or of a non-breathing body ROI (head) is not_assessable (camera_confounded = True).
"""
import json
import math

import numpy as np

from module2 import respiration as rs
from module2.rppg import kalman_smooth
from hr_rr_m1v2 import validity as va


def traces(rows, rois, frames):
    """{roi: (K, 2) residual velocity}, {roi: (K,) q_v2} on the sampled-frame grid (NaN where unavailable)."""
    pos = {f: i for i, f in enumerate(frames)}
    vel, q = {}, {}
    for roi in rois:
        part = rows[(rows.roi_name == roi) & rows.available]
        v = np.full((len(frames), 2), np.nan)
        s = np.full(len(frames), np.nan)
        idx = [pos[f] for f in part.frame_index]
        v[idx, 0], v[idx, 1] = part.residual_dx.to_numpy(float), part.residual_dy.to_numpy(float)
        s[idx] = part.q_v2.to_numpy(float)
        vel[roi], q[roi] = v, s
    return vel, q


def camera_trace(rows, frames):
    """(K, 2) camera (global) velocity per sampled frame from Module 1 v2 (identical for every ROI of a frame)."""
    g = rows[rows.available].groupby('frame_index')[['global_dx', 'global_dy']].first()
    return g.reindex(frames).to_numpy(float)


def _narrow_amp(x, fs, rate_bpm, halfwidth_hz):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return 0.
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1. / fs)
    X[np.abs(f - rate_bpm / 60.) > halfwidth_hz] = 0
    return float(np.std(np.fft.irfft(X, len(x))))


def _concentration(f, p, halfwidth_hz=.1):
    k = int(np.argmax(p))
    return float(p[np.abs(f - f[k]) <= halfwidth_hz].sum() / max(p.sum(), 1e-20))


def confound_check(rr, a, b, fs, torso_amp, refs, cc, rr_config):
    """Is the window rate also the rate of camera motion or of a non-breathing body ROI? (see RR+ confound_check)

    refs: {name: (z-scored signal, raw band-passed signal in px, kind)} with kind 'camera' or 'body'.
    Confounded when the reference peak is within tolerance of rr AND its motion at rr is >= min_amplitude_ratio_camera
    (camera, also concentrated >= min_concentration) or min_amplitude_ratio_body (head) times the torso's.
    """
    if rr is None or not cc.get('enabled', False):
        return False, None, None
    # body references catch camera motion the compensation missed. When Module 1 v2 measured the camera (camera_known)
    # and it does not move at rr, there is nothing to miss: a head moving with breathing (swaddled infant, NEW_1105)
    # must not veto real breathing. When the camera estimate is degenerate (exactly 0: black bars, NEW_1104) the camera
    # is unknown and the body references are used.
    if 'camera' in refs and refs['camera'][2] == 'camera' and \
            _narrow_amp(refs['camera'][1][a:b], fs, rr, cc['narrowband_hz']) < cc.get('body_requires_camera_ratio', 0.) * torso_amp:
        refs = {'camera': refs['camera']}
    for name, (z, raw, kind) in refs.items():
        sp = rs.window_spectrum(z[a:b], fs, rr_config)
        if sp is None:
            continue
        r0, _ = rs.estimate_rr(sp[0], sp[1], (b - a) / fs, rr_config)
        if r0 is None or abs(r0 - rr) > cc['tolerance_brpm']:
            continue
        ratio = cc['min_amplitude_ratio_camera'] if kind == 'camera' else (
            cc['min_amplitude_ratio_body'] if 'camera' in refs else cc['min_amplitude_ratio_body_unknown_camera'])
        if kind == 'camera' and _concentration(*sp) < cc['min_concentration']:
            continue
        if _narrow_amp(raw[a:b], fs, rr, cc['narrowband_hz']) >= ratio * torso_amp:
            return True, name, r0
    return False, None, None


def gross_motion(rows, frames, fs, config):
    """Per sampled frame: True when the mean residual speed of the infant exceeds gross_motion_box_per_sec box heights."""
    speed = rows[rows.available].assign(sp=np.hypot(rows.residual_dx, rows.residual_dy)).groupby('frame_index').sp.mean()
    heights = {}
    for f, box in zip(rows.frame_index, rows.infant_box):
        if f not in heights and isinstance(box, str) and box not in ('null', 'None'):
            b = [float(x) for x in box.strip('[]').split(',')]
            heights[f] = b[3] - b[1]
    return np.array([(speed.get(f, np.nan) * fs / heights[f] > config.gross_motion_box_per_sec)
                     if f in heights and np.isfinite(speed.get(f, np.nan)) else False for f in frames])


def signals(vel, fs, config, scale):
    sig, amp = {}, {}
    for roi, v in vel.items():
        if np.isfinite(v).all(axis=1).sum() < config.min_window_sec * fs * .5:
            continue
        z, raw = rs.respiratory_signal(v, fs, 'FLOW', config, return_raw=True)
        sig[roi], amp[roi] = z, raw * scale
    return sig, amp


def window_weights(sig, amp, q, a, b, fs, config, temperature):
    """{roi: normalized fusion weight} of one window (same gating as fuse), for display."""
    names, prior = [], []
    for roi, z in sig.items():
        sp = rs.window_spectrum(z[a:b], fs, config)
        level = np.nanstd(amp[roi][a:b]) if np.isfinite(amp[roi][a:b]).any() else 0.
        value = np.nanmean(q[roi][a:b]) if np.isfinite(q[roi][a:b]).any() else np.nan
        if sp is not None and level >= config.min_flow_amplitude_px and np.isfinite(value):
            names.append(roi)
            prior.append(value)
    if not names:
        return {}
    w = np.exp((np.asarray(prior) - max(prior)) / temperature)
    return {n: float(x) for n, x in zip(names, w / w.sum())}


def fuse(sig, amp, q, a, b, fs, config, temperature, group=None):
    spectra, weights = [], []
    for roi, z in sig.items():
        if group is not None and roi not in group:
            continue
        sp = rs.window_spectrum(z[a:b], fs, config)
        level = np.nanstd(amp[roi][a:b]) if np.isfinite(amp[roi][a:b]).any() else 0.
        prior = np.nanmean(q[roi][a:b]) if np.isfinite(q[roi][a:b]).any() else np.nan
        if sp is None or level < config.min_flow_amplitude_px or not np.isfinite(prior):
            continue
        spectra.append(sp)
        weights.append(prior)
    if not spectra:
        return None
    w = np.exp((np.asarray(weights) - max(weights)) / temperature)
    return rs.fuse_spectra(spectra, w)


def analyse(record, rows, cfg, rr_config, seed, waves=None):
    frames = sorted(rows.frame_index.unique())
    if len(frames) < 3:
        return []
    fs = len(frames) / float(record['duration_sec'])
    scale = cfg['reference_flow_width'] / float(record['width'])
    vel, q = traces(rows, cfg['rois'], frames)
    sig, amp = signals(vel, fs, rr_config, scale)
    windows = rs.rr_windows(len(frames), fs, rr_config)
    gross = gross_motion(rows, frames, fs, rr_config)
    cc = cfg.get('confound', {})
    refs = {}
    if cc.get('enabled', False):
        rvel, _ = traces(rows, cc.get('reference_regions', []), frames)
        rsig, ramp = signals(rvel, fs, rr_config, scale)
        refs = {f'body_{n}': (rsig[n], ramp[n], 'body') for n in rsig}
        cam = camera_trace(rows, frames)
        known = np.isfinite(cam).all(axis=1) & (np.abs(cam).sum(axis=1) > 0)
        csig, camp = signals({'camera': cam}, fs, rr_config, scale) if known.mean() >= .5 else ({}, {})
        if 'camera' in csig:
            refs = {'camera': (csig['camera'], camp['camera'], 'camera'), **refs}

    def estimate(s, am, a, b, group=None):
        f = fuse(s, am, q, a, b, fs, rr_config, cfg['weight_temperature'], group)
        return (None, None) if f is None else rs.estimate_rr(f[0], f[1], (b - a) / fs, rr_config)

    block = max(1, int(round(cfg['shuffle_block_sec'] * fs)))
    null = []
    for s in range(cfg['surrogates']):
        rng = np.random.default_rng([seed, int(record['video_id'][-3:]) if record['video_id'][-3:].isdigit() else 0, s])
        sv = {roi: va.block_shuffle_2d(v, rng, block) for roi, v in vel.items()}
        ss, sa = signals(sv, fs, rr_config, scale)
        for a, b in windows:
            _, snr = estimate(ss, sa, a, b)
            null.append(snr if snr is not None and math.isfinite(snr) else np.nan)
    null = np.asarray(null, float)
    threshold = np.nanpercentile(null, cfg['null_percentile']) if np.isfinite(null).any() else np.nan
    out = []
    for w, (a, b) in enumerate(windows):
        rr, snr = estimate(sig, amp, a, b)
        rc, _ = estimate(sig, amp, a, b, set(cfg['chest_group']))
        ra, _ = estimate(sig, amp, a, b, set(cfg['abdomen_group']))
        motion_fraction = float(np.mean(gross[a:b])) if b > a else None
        sqi = rs.signal_quality(snr, motion_fraction, rr_config) if snr is not None else 0.
        is_present = snr is not None and math.isfinite(snr) and math.isfinite(threshold) and snr > threshold
        agreement = va.agree(rc, ra, cfg['agreement_brpm'])
        weights_w = window_weights(sig, amp, q, a, b, fs, rr_config, cfg['weight_temperature'])
        conf, source, ref_rate = False, None, None
        if is_present and refs:
            torso_amp = max([_narrow_amp(amp[n][a:b], fs, rr, cc['narrowband_hz']) for n in weights_w] or [0.])
            conf, source, ref_rate = confound_check(rr, a, b, fs, torso_amp, refs, cc, rr_config)
        out.append(dict(window=w, window_start_sec=a / fs, window_end_sec=b / fs, window_center_sec=(a + b) / 2 / fs,
                        window_sec=(b - a) / fs, rr_weights=json.dumps(weights_w), rr_brpm=rr, snr_db=snr, sqi=sqi, gross_motion_fraction=motion_fraction,
                        null_threshold_db=threshold, breathing_present=is_present, rr_chest_brpm=rc, rr_abdomen_brpm=ra,
                        chest_abdomen_agree=agreement, label=va.label(is_present and not conf, agreement),
                        camera_confounded=bool(conf), confound_source=source, confound_rate_brpm=ref_rate,
                        regions=','.join(sorted(sig))))
    kal = va.smoothed([r['window_end_sec'] for r in out], [r['rr_brpm'] for r in out], [r['label'] for r in out],
                      [r['sqi'] for r in out], rr_config, kalman_smooth)
    for r, k in zip(out, kal):
        r['rr_kalman_brpm'] = k
    if waves is not None and sig:
        # display waveform: sign-aligned, prior-weighted mean of the z-scored region signals
        names = sorted(sig)
        stack = np.column_stack([sig[n] for n in names])
        prior = np.array([np.nanmean(q[n]) if np.isfinite(q[n]).any() else -np.inf for n in names])
        weights = np.exp((prior - prior.max()) / cfg['weight_temperature']) if np.isfinite(prior).any() else np.ones(len(names))
        signs = rs.align_signs(stack, int(np.nanargmax(weights)))
        finite = np.isfinite(stack)
        num = np.where(finite, stack * signs * weights, 0.).sum(1)
        den = np.where(finite, weights, 0.).sum(1)
        waves.update(breath=np.divide(num, den, out=np.full(len(num), np.nan), where=den > 0), breath_fs=fs,
                     breath_frames=np.asarray(frames))
    return out
