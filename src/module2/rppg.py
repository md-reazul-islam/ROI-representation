"""Training-free rPPG pulse extraction and neonatal heart-rate estimation.

Pulse methods (input: spatially averaged RGB trace, shape (n, 3), channel order R, G, B):
  GREEN  Verkruysse et al., Opt. Express 2008; the channel used in the NICU study of Villarroel et al. 2019.
  CHROM  de Haan & Jeanne, IEEE TBME 2013 (chrominance, Hann overlap-add).
  POS    Wang et al., IEEE TBME 2017 (plane-orthogonal-to-skin, overlap-add).
Pre-processing: smoothness-priors detrending (Tarvainen et al., IEEE TBME 2002; as in rPPG-Toolbox)
and a zero-phase Butterworth band-pass on the neonatal band.
Heart rate per window (Villarroel et al., npj Digit. Med. 2019: 8 s windows, 1 s step, 1.5-4.5 Hz):
  fft      highest periodogram peak with parabolic interpolation (default);
  ar_best  autoregressive "best model" of that study (orders 6-12; the dominant pole must agree with
           the AR spectral peak within 1 bpm), fitted with Burg's method. It was the most accurate
           estimator on their NICU data (MAE 2.3 vs 2.6 bpm for FFT), but on our synthetic stress tests
           the interpolated FFT was more robust at low SNR, so both are always reported and the primary
           estimator should be chosen on validation data with a reference HR.
Signal quality: spectral SNR (de Haan & Jeanne 2013) combined with Module 1 motion into an SQI; the HR
series is then smoothed by an SQI-weighted Kalman filter (as in Villarroel et al. 2019).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg as spl
from scipy import signal as sps
from scipy import sparse
from scipy.sparse import linalg as spsl


METHODS = ('POS', 'CHROM', 'GREEN')
ESTIMATORS = ('fft', 'ar_best')


@dataclass(frozen=True)
class HRConfig:
    # Preterm/term neonates: >99% of reference HR in 90-270 bpm (Villarroel et al. 2019).
    hr_min_bpm: float = 90.
    hr_max_bpm: float = 270.
    filter_order: int = 3
    detrend_lambda: float = 100.
    pos_window_sec: float = 1.6
    chrom_window_sec: float = 1.6
    estimation_window_sec: float = 8.
    estimation_step_sec: float = 1.
    min_window_sec: float = 5.
    min_valid_fraction: float = .9
    max_gap_sec: float = .5
    estimator: str = 'fft'
    ar_orders: tuple = (6, 7, 8, 9, 10, 11, 12)
    ar_agreement_bpm: float = 1.
    frequency_resolution_bpm: float = .25
    snr_fundamental_halfwidth_hz: float = .1
    # SQI = sigmoid((SNR - snr_threshold_db) / snr_slope_db) * (1 - Module 1 motion magnitude)
    snr_threshold_db: float = -3.
    snr_slope_db: float = 2.
    sqi_threshold: float = .5
    # Kalman random-walk smoothing of the HR series.
    kalman_process_bpm_per_sec: float = 3.
    kalman_measurement_bpm: float = 3.
    kalman_min_sqi: float = .05
    # Without a new window HR for this long the estimate is dropped (shown as '--'), not extrapolated.
    kalman_max_gap_sec: float = 4.

    @property
    def band_hz(self):
        return self.hr_min_bpm/60., self.hr_max_bpm/60.


def interpolate_gaps(trace, fs, max_gap_sec):
    """Linearly fill internal NaN runs no longer than max_gap_sec; longer runs and edges stay NaN."""
    trace = np.array(trace, dtype=float, copy=True)
    flat = trace.reshape(len(trace), -1)
    max_gap = int(round(max_gap_sec * fs))
    for column in range(flat.shape[1]):
        values = flat[:, column]
        missing = ~np.isfinite(values)
        if not missing.any() or missing.all():
            continue
        index = np.arange(len(values))
        filled = np.interp(index, index[~missing], values[~missing])
        for start, stop in segments(missing, 1):
            if start > 0 and stop < len(values) and stop - start <= max_gap:
                values[start:stop] = filled[start:stop]
    return flat.reshape(trace.shape)


def segments(valid, min_length):
    """Contiguous True runs of at least min_length samples, as (start, stop) pairs."""
    runs, start = [], None
    for i, ok in enumerate(list(valid) + [False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_length:
                runs.append((start, i))
            start = None
    return runs


def detrend_smoothness_priors(x, lam):
    """Tarvainen et al. 2002: x - (I + lam^2 D2'D2)^-1 x, with D2 the second-difference operator."""
    n = len(x)
    if n < 3 or lam <= 0:
        return x - np.mean(x)
    d2 = sparse.diags([np.ones(n-2), -2*np.ones(n-2), np.ones(n-2)], [0, 1, 2], shape=(n-2, n))
    trend = spsl.spsolve((sparse.identity(n) + lam**2 * (d2.T @ d2)).tocsc(), x)
    return x - trend


def bandpass(x, fs, config):
    low, high = config.band_hz
    high = min(high, .95 * fs / 2)
    b, a = sps.butter(config.filter_order, [low, high], btype='bandpass', fs=fs)
    if len(x) <= 3 * max(len(a), len(b)):
        return x - np.mean(x)
    return sps.filtfilt(b, a, x)


def _normalized(rgb):
    mean = rgb.mean(axis=0)
    mean[mean == 0] = 1.
    return rgb / mean


def pulse_green(rgb, fs, config):
    return _normalized(rgb)[:, 1] - 1.


def pulse_pos(rgb, fs, config):
    n = len(rgb)
    length = min(n, max(2, int(math.ceil(config.pos_window_sec * fs))))
    h = np.zeros(n)
    projection = np.array([[0., 1., -1.], [-2., 1., 1.]])
    for end in range(length, n + 1):
        s = projection @ _normalized(rgb[end-length:end]).T
        std1 = np.std(s[1])
        p = s[0] + (np.std(s[0]) / std1 if std1 > 0 else 0.) * s[1]
        h[end-length:end] += p - p.mean()
    return h


def pulse_chrom(rgb, fs, config):
    n = len(rgb)
    length = max(4, int(math.ceil(config.chrom_window_sec * fs)))
    length += length % 2
    if n < length:
        length = n - n % 2
    step = length // 2
    h = np.zeros(n)
    hann = np.hanning(length)
    for start in range(0, n - length + 1, step):
        window = _normalized(rgb[start:start+length])
        x = 3*window[:, 0] - 2*window[:, 1]
        y = 1.5*window[:, 0] + window[:, 1] - 1.5*window[:, 2]
        xf, yf = bandpass(x, fs, config), bandpass(y, fs, config)
        std_y = np.std(yf)
        s = xf - (np.std(xf) / std_y if std_y > 0 else 0.) * yf
        h[start:start+length] += hann * (s - s.mean())
    return h


PULSE_FUNCTIONS = dict(POS=pulse_pos, CHROM=pulse_chrom, GREEN=pulse_green)


def extract_pulse(rgb, fs, method, config):
    """Detrended, band-passed, z-scored pulse for a (n, 3) RGB trace that may contain NaN gaps.

    Short gaps are interpolated; each remaining contiguous segment is processed on its own
    and gaps stay NaN, so no pulse is invented where the ROI was not observed.
    """
    rgb = interpolate_gaps(np.asarray(rgb, dtype=float), fs, config.max_gap_sec)
    valid = np.isfinite(rgb).all(axis=1) & (rgb > 0).all(axis=1)
    pulse = np.full(len(rgb), np.nan)
    minimum = max(int(math.ceil(2 * config.pos_window_sec * fs)), 3 * (2*config.filter_order + 1) + 1)
    for start, stop in segments(valid, minimum):
        raw = PULSE_FUNCTIONS[method](rgb[start:stop], fs, config)
        filtered = bandpass(detrend_smoothness_priors(raw, config.detrend_lambda), fs, config)
        std = np.std(filtered)
        pulse[start:stop] = (filtered - filtered.mean()) / std if std > 0 else 0.
    return pulse


def spectrum(x, fs, config):
    """Hann-windowed, zero-padded power spectrum (frequencies in Hz)."""
    x = sps.detrend(np.asarray(x, dtype=float))
    nfft = 1 << int(math.ceil(math.log2(max(len(x), fs * 60. / config.frequency_resolution_bpm))))
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    return np.fft.rfftfreq(nfft, 1. / fs), power


def _band(frequencies, fs, config):
    low, high = config.band_hz
    return (frequencies >= low) & (frequencies <= min(high, fs / 2))


def hr_fft(x, fs, config):
    frequencies, power = spectrum(x, fs, config)
    indices = np.flatnonzero(_band(frequencies, fs, config))
    if indices.size == 0:
        return None
    peak = indices[np.argmax(power[indices])]
    f0 = frequencies[peak]
    if 0 < peak < len(power) - 1:  # parabolic interpolation of the log-power peak
        a, b, c = np.log(power[peak-1:peak+2] + 1e-20)
        denominator = a - 2*b + c
        if denominator != 0:
            f0 += .5 * (a - c) / denominator * (frequencies[1] - frequencies[0])
    return float(f0 * 60.)


def _ar_fit(x, order):
    """Burg AR coefficients a_1..a_p (x_t = sum a_k x_{t-k} + e_t) and the prediction-error variance."""
    x = np.asarray(x, dtype=float)
    forward, backward = x[1:].copy(), x[:-1].copy()
    polynomial, variance = np.array([1.]), float(np.dot(x, x) / len(x))
    for _ in range(order):
        denominator = np.dot(forward, forward) + np.dot(backward, backward)
        if denominator <= 0:
            return None, None
        reflection = -2 * np.dot(forward, backward) / denominator
        extended = np.r_[polynomial, 0.]
        polynomial = extended + reflection * extended[::-1]
        variance *= 1 - reflection ** 2
        forward, backward = forward[1:] + reflection * backward[1:], backward[:-1] + reflection * forward[:-1]
    return -polynomial[1:], variance


def hr_ar_best(x, fs, config):
    """Villarroel et al. 2019 'AR best model' heart rate (bpm) for one window."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    low, high = config.band_hz
    high = min(high, fs / 2)
    grid = np.arange(low, high, config.frequency_resolution_bpm / 60.)
    candidates, fallback = [], None
    for order in config.ar_orders:
        if len(x) <= 2 * order:
            continue
        a, variance = _ar_fit(x, order)
        if a is None or not np.all(np.isfinite(a)):
            continue
        poles = np.roots(np.r_[1., -a])
        frequencies = np.abs(np.angle(poles)) * fs / (2*np.pi)
        inside = (frequencies >= low) & (frequencies <= high) & (np.imag(poles) >= 0)
        response = variance / np.abs(1 - np.exp(-2j*np.pi*np.outer(grid/fs, np.arange(1, order+1))) @ a) ** 2
        peak_bpm = float(grid[np.argmax(response)] * 60.)
        if fallback is None or order == 8:
            fallback = peak_bpm
        if not inside.any():
            continue
        dominant = np.flatnonzero(inside)[np.argmax(np.abs(poles[inside]))]
        pole_bpm = float(frequencies[dominant] * 60.)
        if abs(pole_bpm - peak_bpm) < config.ar_agreement_bpm:
            candidates.append((abs(poles[dominant]), pole_bpm))
    if candidates:
        return max(candidates)[1]
    return fallback


def snr_db(x, fs, hr_bpm, config):
    """de Haan & Jeanne SNR: power within +-halfwidth of f0 and +-2*halfwidth of 2 f0 vs the rest."""
    frequencies, power = spectrum(x, fs, config)
    low, high = config.band_hz
    f0, width = hr_bpm / 60., config.snr_fundamental_halfwidth_hz
    evaluation = (frequencies >= low) & (frequencies <= min(2.2 * high, fs / 2))
    signal_bins = evaluation & ((np.abs(frequencies - f0) <= width) | (np.abs(frequencies - 2*f0) <= 2*width))
    noise, wanted = power[evaluation & ~signal_bins].sum(), power[signal_bins].sum()
    if wanted <= 0:
        return -float('inf')
    return 10 * math.log10(wanted / noise) if noise > 0 else float('inf')


def _usable(x, config):
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.mean(np.isfinite(x)) < config.min_valid_fraction:
        return None
    if not np.isfinite(x).all():
        x = interpolate_gaps(x, 1., len(x))
        x = x[np.isfinite(x)]
    return x if len(x) >= 3 and np.std(x) > 0 else None


def estimate_hr(x, fs, config, estimator=None):
    """Return (hr_bpm, snr_db) for one analysis window, or (None, None) if it is unusable."""
    x = _usable(x, config)
    if x is None:
        return None, None
    estimator = estimator or config.estimator
    bpm = hr_ar_best(x, fs, config) if estimator == 'ar_best' else hr_fft(x, fs, config)
    if bpm is None:
        return None, None
    return float(bpm), float(snr_db(x, fs, bpm, config))


def window_snr(x, fs, config):
    """SNR (dB) at the FFT peak; used as the per-ROI 'goodness' for maximal-ratio fusion."""
    x = _usable(x, config)
    if x is None:
        return None
    bpm = hr_fft(x, fs, config)
    return None if bpm is None else float(snr_db(x, fs, bpm, config))


def signal_quality(snr, motion, config):
    """SQI in [0, 1]: logistic SNR term times (1 - Module 1 motion magnitude)."""
    if snr is None or (not math.isfinite(snr) and snr < 0):
        return 0.
    snr_term = 1. if snr == float('inf') else 1. / (1. + math.exp(-(snr - config.snr_threshold_db) / config.snr_slope_db))
    motion_term = 1. if motion is None or not math.isfinite(motion) else float(np.clip(1. - motion, 0., 1.))
    return snr_term * motion_term


def kalman_smooth(times, hr, sqi, config):
    """Causal random-walk Kalman filter; measurement noise grows as the SQI falls.

    Short gaps without a window HR are bridged by prediction; after kalman_max_gap_sec without a
    measurement the output is None and the filter restarts at the next measurement.
    """
    out, state, variance, previous, last_measurement = [], None, None, None, None
    for t, value, quality in zip(times, hr, sqi):
        if state is not None:
            variance += (config.kalman_process_bpm_per_sec * (t - previous)) ** 2
        previous = t
        measured = value is not None and math.isfinite(value)
        if not measured and state is not None and t - last_measurement > config.kalman_max_gap_sec:
            state, variance = None, None
        if measured:
            last_measurement = t
            r = (config.kalman_measurement_bpm / max(quality, config.kalman_min_sqi)) ** 2
            if state is None:
                state, variance = value, r
            else:
                gain = variance / (variance + r)
                state += gain * (value - state)
                variance *= 1 - gain
        out.append(state)
    return out


def analysis_windows(n, fs, config):
    """(start, stop) sample windows; a clip shorter than one window yields one whole-clip window."""
    length = int(round(config.estimation_window_sec * fs))
    step = max(1, int(round(config.estimation_step_sec * fs)))
    if n < length:
        return [(0, n)] if n >= config.min_window_sec * fs else []
    return [(start, start + length) for start in range(0, n - length + 1, step)]
