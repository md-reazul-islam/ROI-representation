"""Phase 2 building blocks: baseline ROI geometries (M0-M3), causal window-level selection, surrogate nulls
for a pulse-present test, and a split-half agreement test.

Everything here is label-free. Selection that depends on colour signals (M1, M2) only looks at an EARLIER,
non-overlapping analysis window (causal); M3 looks at the same window and is an oracle upper bound only.
"""
from __future__ import annotations

import numpy as np


def box_polygon(box):
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def grid_tiles(box, rows=4, cols=4):
    """rows x cols rectangular tiles covering the infant box, row-major; returns [(polygon, column)]."""
    x1, y1, x2, y2 = box
    xs, ys = np.linspace(x1, x2, cols + 1), np.linspace(y1, y2, rows + 1)
    return [(box_polygon((xs[c], ys[r], xs[c + 1], ys[r + 1])), c) for r in range(rows) for c in range(cols)]


def causal_selection(scores, lag, k):
    """Per window w: indices of the k best units by score in window w - lag (NaN = unusable).

    scores: (W, R). Windows with no earlier window (w < lag) or no usable unit there get None (= use all).
    """
    scores = np.asarray(scores, float)
    out = []
    for w in range(len(scores)):
        if w < lag:
            out.append(None)
            continue
        row = scores[w - lag]
        usable = np.flatnonzero(np.isfinite(row))
        out.append(None if not len(usable) else usable[np.argsort(-row[usable], kind='stable')[:k]])
    return out


def oracle_selection(scores):
    """Per window: the single unit with the best score in the SAME window (upper bound, not a method)."""
    out = []
    for row in np.asarray(scores, float):
        usable = np.flatnonzero(np.isfinite(row))
        out.append(None if not len(usable) else np.array([usable[np.argmax(row[usable])]]))
    return out


def window_mean(pulses, a, b, units):
    """Equal-weight mean of the pulses of `units` (None = all) over samples [a, b), ignoring NaN."""
    block = pulses[a:b] if units is None else pulses[a:b][:, units]
    finite = np.isfinite(block)
    count = finite.sum(axis=1)
    return np.divide(np.where(finite, block, 0.).sum(axis=1), count, out=np.full(b - a, np.nan), where=count > 0)


def fill_gaps(rgb):
    """Linear interpolation over NaN rows of an (n, 3) trace; returns (filled, finite mask) or (None, mask)."""
    finite = np.isfinite(rgb).all(axis=1)
    if finite.sum() < 3:
        return None, finite
    index = np.arange(len(rgb))
    return np.column_stack([np.interp(index, index[finite], rgb[finite, c]) for c in range(rgb.shape[1])]), finite


def phase_surrogate(rgb, rng):
    """Each colour channel gets INDEPENDENT random Fourier phases (amplitude spectrum and mean kept).

    This keeps every channel's power spectrum but destroys the cross-channel phase coherence that a blood
    volume pulse produces and that POS exploits. Gaps (NaN) stay where they were.
    """
    filled, finite = fill_gaps(rgb)
    if filled is None:
        return np.full_like(rgb, np.nan)
    n = len(filled)
    out = np.empty_like(filled)
    for c in range(filled.shape[1]):
        spectrum = np.fft.rfft(filled[:, c])
        phases = rng.uniform(0, 2 * np.pi, len(spectrum))
        phases[0] = 0.
        if n % 2 == 0:
            phases[-1] = 0.
        out[:, c] = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n)
    out[~finite] = np.nan
    return out


def shuffle_surrogate(rgb, rng, block):
    """Blocks of `block` samples permuted in time, jointly for all channels (temporal structure destroyed)."""
    filled, finite = fill_gaps(rgb)
    if filled is None:
        return np.full_like(rgb, np.nan)
    starts = np.arange(0, len(filled), max(1, int(block)))
    order = rng.permutation(len(starts))
    out = np.concatenate([filled[starts[i]:starts[i] + max(1, int(block))] for i in order])[:len(filled)]
    out[~finite] = np.nan
    return out


def present(observed, null, percentile=95.):
    """Window passes when its SNR exceeds the given percentile of the pooled null SNRs (NaN -> False)."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not len(null):
        return np.zeros(len(observed), bool)
    threshold = np.percentile(null, percentile)
    observed = np.asarray(observed, float)
    return np.isfinite(observed) & (observed > threshold)


def empirical_p(observed, null):
    """One-sided permutation p-value that a null statistic is >= observed: (1 + #null>=obs) / (1 + N)."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(null):
        return np.nan
    return (1. + np.sum(null >= observed)) / (1. + len(null))


def agree(hr_a, hr_b, tolerance=6.):
    """True / False when both halves give a heart rate, None otherwise."""
    if hr_a is None or hr_b is None or not np.isfinite(hr_a) or not np.isfinite(hr_b):
        return None
    return bool(abs(hr_a - hr_b) <= tolerance)


def units_or_all(choice, columns):
    """Trace columns used in one window: the chosen positions within `columns`, or ALL of `columns` when nothing
    could be chosen (e.g. no earlier window exists yet). Never falls back to traces of another unit kind."""
    columns = list(columns)
    return columns if choice is None else [columns[j] for j in choice]


def clip_half(polygon, side):
    """Part of a polygon left ('L') or right ('R') of its own vertical midline (Sutherland-Hodgman).

    The two halves of one region are disjoint, so their pulses are independent pixel samples of that region.
    Returns None when the half is degenerate.
    """
    points = np.asarray(polygon, float)
    if len(points) < 3:
        return None
    mid = (points[:, 0].min() + points[:, 0].max()) / 2
    inside = (lambda p: p[0] <= mid) if side == 'L' else (lambda p: p[0] >= mid)
    out = []
    for i in range(len(points)):
        current, previous = points[i], points[i - 1]
        if inside(current):
            if not inside(previous):
                out.append(_cross(previous, current, mid))
            out.append(current)
        elif inside(previous):
            out.append(_cross(previous, current, mid))
    return np.asarray(out) if len(out) >= 3 else None


def _cross(a, b, x):
    t = (x - a[0]) / (b[0] - a[0])
    return np.array([x, a[1] + t * (b[1] - a[1])])


__all__ = ['box_polygon', 'grid_tiles', 'causal_selection', 'oracle_selection', 'window_mean', 'phase_surrogate',
           'shuffle_surrogate', 'present', 'empirical_p', 'agree', 'fill_gaps', 'clip_half', 'units_or_all']
