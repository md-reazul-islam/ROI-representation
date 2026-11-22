"""Window-level validity used by both branches: a value is reported only when the signal beats a surrogate null
built from the same data, and two disjoint parts of the body agree on the rate."""
import math

import numpy as np

from module1_v2.baselines import agree, phase_surrogate, present, shuffle_surrogate  # noqa: F401  (re-exported)


def label(present_flag, agreement):
    """'validated' (beats the null and the two parts agree), 'present' (beats the null; agreement failed or not
    testable), or 'not_assessable'."""
    if not present_flag:
        return 'not_assessable'
    return 'validated' if agreement is True else 'present'


def label_fallback(present_flag, agreement):
    """gate2_mode 'fallback': agreement is True / False for the first available pair, None when no pair was available.
    'validated' (null beaten, pair agrees), 'present' (null beaten, pair disagrees), 'validated_one_side' (null beaten,
    no second body part visible: gate 1 only), or 'not_assessable'."""
    if not present_flag:
        return 'not_assessable'
    if agreement is None:
        return 'validated_one_side'
    return 'validated' if agreement else 'present'


def smoothed(times, values, labels, quality, config, kalman, one_side_noise_factor=None):
    """Kalman track over the validated windows only (others are gaps the filter may bridge).
    With one_side_noise_factor (gate2_mode 'fallback'), 'validated_one_side' windows are measured too, with their
    measurement noise variance multiplied by that factor; without it the original filter is used unchanged."""
    if one_side_noise_factor is None:
        measured = [v if lab == 'validated' else None for v, lab in zip(values, labels)]
        return kalman(list(times), measured, [q if q is not None else 0. for q in quality], config)
    measured = [v if lab in ('validated', 'validated_one_side') else None for v, lab in zip(values, labels)]
    factor = [one_side_noise_factor if lab == 'validated_one_side' else 1. for lab in labels]
    return kalman_scaled(list(times), measured, [q if q is not None else 0. for q in quality], factor, config)


def kalman_scaled(times, hr, sqi, factor, config):
    """module2.rppg.kalman_smooth (same causal random walk, gap reset and SQI noise) with a per-measurement factor on
    the measurement noise variance. With all factors 1 it equals kalman_smooth."""
    out, state, variance, previous, last_measurement = [], None, None, None, None
    for t, value, quality, f in zip(times, hr, sqi, factor):
        if state is not None:
            variance += (config.kalman_process_bpm_per_sec * (t - previous)) ** 2
        previous = t
        measured = value is not None and math.isfinite(value)
        if not measured and state is not None and t - last_measurement > config.kalman_max_gap_sec:
            state, variance = None, None
        if measured:
            last_measurement = t
            r = f * (config.kalman_measurement_bpm / max(quality, config.kalman_min_sqi)) ** 2
            if state is None:
                state, variance = value, r
            else:
                gain = variance / (variance + r)
                state += gain * (value - state)
                variance *= 1 - gain
        out.append(state)
    return out


def block_shuffle_2d(trace, rng, block):
    """Joint block shuffle of an (n, 2) velocity trace (NaN rows stay in place)."""
    return shuffle_surrogate(np.asarray(trace, float), rng, block)
