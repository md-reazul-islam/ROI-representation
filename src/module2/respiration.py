"""Camera respiration rate (RR) from the Module 1 dynamic ROIs (Module 2, RR branch).

Respiratory signals per ROI (every native frame, ROI geometry held from Module 1 as for HR):
  FLOW       breathing motion: dense optical flow (Farneback) between consecutive frames, averaged
             inside the ROI, with the camera / whole-scene motion removed by subtracting the median
             flow of the background (outside the infant box); the velocity is integrated into a 2-D
             displacement and projected on its principal axis (the local breathing direction).
             Motion-based neonatal RR: e.g. Villarroel et al., npj Digit. Med. 2019; Lorato et al.,
             Sensors 2021; arXiv:2207.11886 (abdomen / chest regions best).
  INTENSITY  mean grey level inside the ROI (chest-wall shading changes; the classic pixel-intensity
             method, e.g. Tarassenko et al., Physiol. Meas. 2014).
Breathing motion is visible through clothing and blankets, so no skin mask is used for RR.

Per window, every ROI signal gives a power spectrum, a spectral peak (breaths/min) and a spectral SNR.
ROIs are fused at the SPECTRUM level (weighted mean of normalized spectra): chest and abdomen often
move in anti-phase in neonates (paradoxical breathing), which would cancel in a waveform average but
not in a spectral average. The weights use the same dynamic fusion as HR:
w_r ∝ softmax(Q_r / T) x 10^(SNR_r / 10). The displayed waveform is the weighted, sign-aligned sum.
SQI = sigmoid((SNR - threshold) / slope) x (1 - fraction of the window with gross body motion);
the RR series is smoothed with the same SQI-weighted causal Kalman filter as HR.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from scipy import signal as sps

from module2.rppg import detrend_smoothness_priors, interpolate_gaps, segments


RR_METHODS = ('FLOW', 'INTENSITY')
RESPIRATORY_POOL = ('NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN',
                    'THORACO_ABDOMINAL')


@dataclass(frozen=True)
class RRConfig:
    # Neonatal breathing: normal 30-60 /min, tachypnoea > 60; the band stops where the HR band starts.
    rr_min_bpm: float = 20.
    rr_max_bpm: float = 90.
    filter_order: int = 3
    # Smoothness-priors detrending at 30 fps (scaled by (fs/30)^2). The HR value (100) would remove
    # breathing (a 0.5 Hz sine keeps 54 % of its amplitude); 1000 keeps 93 % at 0.33 Hz.
    detrend_lambda_30fps: float = 1000.
    estimation_window_sec: float = 15.
    estimation_step_sec: float = 1.
    # Windows grow from min_window_sec to estimation_window_sec at the start of a clip (causal), so
    # short clips still get an RR track; shorter windows have a coarser resolution and lower SNR.
    min_window_sec: float = 8.
    min_valid_fraction: float = .9
    max_gap_sec: float = .5
    frequency_resolution_bpm: float = .25
    # SNR: power within +-max(halfwidth, 1/window) of f0 and of 2 f0 vs the rest of the RR band, in dB
    # above white noise. Calibrated on synthetic data: the peak of pure noise reaches 9.6 dB (95th pct)
    # in 8 s windows and 8.8 dB in 15 s windows; recalibrate on validation data with a reference RR.
    snr_fundamental_halfwidth_hz: float = .05
    snr_threshold_db: float = 9.
    snr_slope_db: float = 1.5
    sqi_threshold: float = .5
    # Optical flow (computed on frames resized to this width).
    flow_width: int = 320
    compensate_camera_motion: bool = True
    # A FLOW ROI is used only if its band-passed breathing displacement has at least this standard
    # deviation (pixels at flow_width). Z-scoring hides amplitude, so without this gate a static ROI
    # with a tiny periodic residual (e.g. imperfectly removed camera shake) looks like breathing.
    # Neonatal chest excursion is about 1-2 px at 320 px width; flow noise gives about 0.05 px.
    min_flow_amplitude_px: float = .1
    min_background_fraction: float = .1
    # Gross motion: mean |flow| in the infant box above this many box heights per second (breathing
    # moves the box mean far less; limb movement, crying and handling exceed it). Heuristic.
    gross_motion_box_per_sec: float = .1
    # Kalman smoothing of the RR series (same filter as HR; units breaths/min).
    kalman_process_bpm_per_sec: float = 1.5
    kalman_measurement_bpm: float = 3.
    kalman_min_sqi: float = .05
    kalman_max_gap_sec: float = 6.

    @property
    def band_hz(self):
        return self.rr_min_bpm/60., self.rr_max_bpm/60.


def flow_frame(previous_gray, gray):
    return cv2.calcOpticalFlowFarneback(previous_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)


def resize_gray(image, width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1., width / gray.shape[1])
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def polygon_mask(shape, polygon, scale):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(polygon) * scale).astype(np.int32)], 1)
    return mask.astype(bool)


def background_flow(flow, box_mask, config):
    """Median flow outside the (dilated) infant box: camera / whole-scene motion; zero if too little background."""
    if not config.compensate_camera_motion or box_mask is None:
        return np.zeros(2)
    outside = ~box_mask
    if outside.mean() < config.min_background_fraction:
        return np.zeros(2)
    return np.median(flow[outside], axis=0)


def bandpass(x, fs, config):
    low, high = config.band_hz
    sos = sps.butter(config.filter_order, [low, min(high, .95 * fs / 2)], btype='bandpass', fs=fs, output='sos')
    if len(x) <= 3 * (2 * len(sos) + 1):
        return x - np.mean(x)
    return sps.sosfiltfilt(sos, x)


def _filtered(x, fs, config):
    lam = config.detrend_lambda_30fps * (fs / 30.) ** 2
    filtered = bandpass(detrend_smoothness_priors(x - x.mean(), lam), fs, config)
    return filtered - filtered.mean()


def _zscore(x):
    std = np.std(x)
    return x / std if std > 0 else x * 0.


def respiratory_signal(trace, fs, method, config, return_raw=False):
    """Band-passed, z-scored respiratory signal from one ROI trace that may contain NaN gaps.

    FLOW: trace (n, 2) = per-frame ROI velocity (px/frame) -> displacement -> principal axis.
    INTENSITY: trace (n,) = mean grey level.
    Short gaps are interpolated; longer gaps split the trace into segments processed separately.
    With return_raw, also the band-passed signal before z-scoring (FLOW: displacement in pixels at
    flow_width; INTENSITY: relative grey level), whose amplitude gates FLOW ROIs.
    """
    trace = interpolate_gaps(np.asarray(trace, dtype=float), fs, config.max_gap_sec)
    flat = trace.reshape(len(trace), -1)
    valid = np.isfinite(flat).all(axis=1)
    out, raw = np.full(len(trace), np.nan), np.full(len(trace), np.nan)
    minimum = int(math.ceil(min(config.min_window_sec, 4.) * fs))
    for start, stop in segments(valid, minimum):
        part = flat[start:stop]
        if method == 'FLOW':
            displacement = np.cumsum(part, axis=0)
            cleaned = np.column_stack([_filtered(displacement[:, c], fs, config) for c in range(2)])
            # Principal axis of the band-passed 2-D displacement = the local breathing direction.
            _, _, vt = np.linalg.svd(cleaned, full_matrices=False)
            axis = vt[0] if vt[0][np.argmax(np.abs(vt[0]))] > 0 else -vt[0]
            raw[start:stop] = cleaned @ axis
        else:
            values = part[:, 0]
            mean = values.mean()
            raw[start:stop] = _filtered(values / mean if mean > 0 else values, fs, config)
        out[start:stop] = _zscore(raw[start:stop])
    return (out, raw) if return_raw else out


def rr_windows(n, fs, config):
    """Causal analysis windows: growing from min_window_sec to estimation_window_sec, then sliding."""
    length = int(round(config.estimation_window_sec * fs))
    minimum = int(round(config.min_window_sec * fs))
    step = max(1, int(round(config.estimation_step_sec * fs)))
    if n < minimum:
        return []
    ends = list(range(minimum, n + 1, step))
    if ends[-1] != n and n - ends[-1] >= step / 2:
        ends.append(n)
    return [(max(0, end - length), end) for end in ends]


def spectrum(x, fs, config):
    """Hann-windowed, zero-padded power spectrum normalized to unit power inside the RR band."""
    x = sps.detrend(np.asarray(x, dtype=float))
    nfft = 1 << int(math.ceil(math.log2(max(len(x), fs * 60. / config.frequency_resolution_bpm))))
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, 1. / fs)
    band = in_band(frequencies, config)
    total = power[band].sum()
    return frequencies, power / total if total > 0 else power


def in_band(frequencies, config):
    low, high = config.band_hz
    return (frequencies >= low) & (frequencies <= high)


def peak_rate(frequencies, power, config):
    """Interpolated spectral peak (breaths/min) inside the RR band."""
    indices = np.flatnonzero(in_band(frequencies, config))
    if indices.size == 0 or not np.any(power[indices] > 0):
        return None
    peak = indices[np.argmax(power[indices])]
    f0 = frequencies[peak]
    if 0 < peak < len(power) - 1:
        a, b, c = np.log(power[peak-1:peak+2] + 1e-20)
        denominator = a - 2*b + c
        if denominator != 0:
            f0 += .5 * (a - c) / denominator * (frequencies[1] - frequencies[0])
    return float(f0 * 60.)


def spectral_snr(frequencies, power, rate_bpm, duration_sec, config):
    """Power near f0 and 2 f0 vs the rest of the RR band, in dB above the white-noise level.

    The half-width follows the window's resolution (max(halfwidth, 1/window)), so short windows claim a
    larger part of the band; subtracting the white-noise value 10 log10(signal bins / noise bins) makes
    noise score about 0 dB at every window length, and one threshold fits growing windows.
    """
    band = in_band(frequencies, config)
    f0 = rate_bpm / 60.
    width = max(config.snr_fundamental_halfwidth_hz, 1. / max(duration_sec, 1e-9))
    wanted = band & ((np.abs(frequencies - f0) <= width) | (np.abs(frequencies - 2*f0) <= width))
    signal, noise = power[wanted].sum(), power[band & ~wanted].sum()
    if signal <= 0:
        return -float('inf')
    if noise <= 0:
        return float('inf')
    return 10 * math.log10(signal / noise) - 10 * math.log10(wanted.sum() / max((band & ~wanted).sum(), 1))


def usable(x, config):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.mean(np.isfinite(x)) < config.min_valid_fraction:
        return None
    if not np.isfinite(x).all():
        x = interpolate_gaps(x, 1., len(x))
        x = x[np.isfinite(x)]
    return x if len(x) >= 3 and np.std(x) > 0 else None


def window_spectrum(x, fs, config):
    """(frequencies, normalized power) of one ROI window, or None if the window is unusable."""
    x = usable(x, config)
    return None if x is None else spectrum(x, fs, config)


def estimate_rr(frequencies, power, duration_sec, config):
    """(rr_bpm, snr_db) from a (fused) spectrum."""
    rate = peak_rate(frequencies, power, config)
    if rate is None:
        return None, None
    return rate, float(spectral_snr(frequencies, power, rate, duration_sec, config))


def fuse_spectra(spectra, weights):
    """Weighted mean of normalized spectra (phase-invariant ROI fusion); None if nothing is weighted."""
    total, fused = 0., None
    for item, weight in zip(spectra, weights):
        if item is None or weight <= 0:
            continue
        fused = item[1] * weight if fused is None else fused + item[1] * weight
        total += weight
        frequencies = item[0]
    return None if fused is None else (frequencies, fused / total)


def align_signs(signals, reference):
    """+-1 per column so that every ROI signal correlates positively with the reference column."""
    signs = np.ones(signals.shape[1])
    if reference is None:
        return signs
    ref = signals[:, reference]
    for r in range(signals.shape[1]):
        both = np.isfinite(ref) & np.isfinite(signals[:, r])
        if both.sum() > 3 and np.std(signals[both, r]) > 0 and np.std(ref[both]) > 0:
            signs[r] = 1. if np.corrcoef(ref[both], signals[both, r])[0, 1] >= 0 else -1.
    return signs


def signal_quality(snr, motion_fraction, config):
    """SQI in [0, 1]: logistic SNR term times (1 - fraction of the window with gross body motion)."""
    if snr is None or (not math.isfinite(snr) and snr < 0):
        return 0.
    snr_term = 1. if snr == float('inf') else 1. / (1. + math.exp(-(snr - config.snr_threshold_db) / config.snr_slope_db))
    motion_term = 1. if motion_fraction is None or not math.isfinite(motion_fraction) else float(np.clip(1. - motion_fraction, 0., 1.))
    return snr_term * motion_term
