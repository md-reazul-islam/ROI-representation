"""Phase 1a: temporal stabilization of region selection and fusion weights.

All functions work on one video's score matrix Q[t, r] (t = sampled frames in time order, r = the 17 ROI
names; NaN = region unavailable at that frame). Q is the per-frame confidence (v1: the stored Phase 10 Q).

Settings that reproduce v1 exactly: ema_beta = 0, method 'topk', alpha_ema = 0, max_step = inf.
"""
from __future__ import annotations

import math

import numpy as np


def ema(Q, beta):
    """Per-region exponential moving average. A region keeps its smoothed value while unavailable (it is
    not selectable then); frames where no region is available (infant lost) reset all memory."""
    Q = np.asarray(Q, float)
    if beta <= 0:
        return Q.copy()
    S = np.full_like(Q, np.nan)
    memory = np.full(Q.shape[1], np.nan)
    for t in range(len(Q)):
        row = Q[t]
        valid = np.isfinite(row)
        if not valid.any():
            memory[:] = np.nan
            continue
        seen = valid & np.isfinite(memory)
        memory[seen] = beta * memory[seen] + (1 - beta) * row[seen]
        memory[valid & ~seen] = row[valid & ~seen]
        S[t, valid] = memory[valid]
    return S


def _order(values, candidates, names):
    """Candidates sorted by descending value, ties by ROI name (the v1 lexsort order)."""
    return sorted(candidates, key=lambda r: (-values[r], names[r]))


def topk_sets(S, k, names):
    """v1 selection: the k best available regions of every frame, no temporal state."""
    out = []
    for row in np.asarray(S, float):
        available = [r for r in range(len(row)) if math.isfinite(row[r])]
        out.append(_order(row, available, names)[:k])
    return out


def hysteresis_sets(S, k, names, delta, n_frames, fixed=None):
    """Top-k with hysteresis: an unselected region replaces the weakest selected one only after its score
    exceeded the incumbent's by more than `delta` on `n_frames` consecutive frames. Vacancies (a selected
    region became unavailable) are filled immediately with the best available region. `fixed[t]` (optional)
    is a region forced into the set at frame t (e.g. a Viterbi anchor); it counts towards k."""
    S = np.asarray(S, float)
    selected, counters, out = [], {}, []
    for t, row in enumerate(S):
        available = {r for r in range(len(row)) if math.isfinite(row[r])}
        if not available:
            selected, counters = [], {}
            out.append([])
            continue
        anchor = fixed[t] if fixed is not None and fixed[t] is not None and fixed[t] in available else None
        size = min(k, len(available))
        selected = [r for r in selected if r in available and r != anchor]
        free = size - (1 if anchor is not None else 0)
        for r in _order(row, available - set(selected) - {anchor}, names):
            if len(selected) >= free:
                break
            selected.append(r)
        selected = selected[:max(free, 0)]
        challengers = available - set(selected) - {anchor}
        if selected:
            weakest = _order(row, selected, names)[-1]
            for r in list(counters):
                if r not in challengers:
                    del counters[r]
            for r in challengers:
                counters[r] = counters.get(r, 0) + 1 if row[r] > row[weakest] + delta else 0
            while True:
                weakest = _order(row, selected, names)[-1]
                ready = [r for r in _order(row, challengers, names)
                         if counters.get(r, 0) >= n_frames and row[r] > row[weakest] + delta]
                if not ready:
                    break
                selected[selected.index(weakest)] = ready[0]
                challengers = (challengers - {ready[0]}) | {weakest}
                counters.pop(ready[0], None)
                counters[weakest] = 0
        current = ([anchor] if anchor is not None else []) + selected
        out.append(_order(row, current, names))
    return out


def viterbi_anchor(Q, penalty, lag=None):
    """Top-1 region per frame maximizing sum(Q of the chosen region) - penalty * (number of switches).
    lag=None: offline (whole-video) Viterbi. lag=L (frames): online fixed-lag decoding, the decision for
    frame t uses frames up to t + L only. Frames without any available region split the video into
    independent segments. Returns a list of region indices (None where nothing is available)."""
    Q = np.asarray(Q, float)
    T, R = Q.shape
    out = [None] * T
    t = 0
    while t < T:
        if not np.isfinite(Q[t]).any():
            t += 1
            continue
        end = t
        while end < T and np.isfinite(Q[end]).any():
            end += 1
        E = np.where(np.isfinite(Q[t:end]), Q[t:end], -np.inf)
        n = len(E)
        score = np.full((n, R), -np.inf)
        back = np.zeros((n, R), int)
        score[0] = E[0]
        for i in range(1, n):
            stay = score[i - 1]
            best = int(np.argmax(stay))
            switch = stay[best] - penalty
            use_switch = switch > stay
            back[i] = np.where(use_switch, best, np.arange(R))
            score[i] = np.where(use_switch, switch, stay) + E[i]

        def backtrack(frm, to):
            r = int(np.argmax(score[frm]))
            for i in range(frm, to, -1):
                r = int(back[i, r])
            return r
        if lag is None:
            path = [int(np.argmax(score[n - 1]))]
            for i in range(n - 1, 0, -1):
                path.append(int(back[i, path[-1]]))
            decisions = path[::-1]
        else:
            decisions = [backtrack(min(n - 1, i + lag), i) for i in range(n)]
        for i, decision in enumerate(decisions):
            out[t + i] = decision if math.isfinite(E[i, decision]) else int(np.argmax(E[i]))
        t = end
    return out


def softmax(values, temperature):
    values = np.asarray(values, float) / temperature
    values = np.exp(values - values.max())
    return values / values.sum()


def fusion_weights(S, sets, temperature=1., alpha_ema=0., max_step=math.inf):
    """Per-frame fusion weights W[t, r]. Target = softmax(S / T) over the selected set (v1 rule). Optional
    temporal smoothing (EMA with factor alpha_ema) and a taper limiting every weight change to max_step per
    frame, so a region entering or leaving the set is cross-faded instead of switched. Weights are
    renormalized over the regions available at that frame; a region that becomes unavailable drops out."""
    S = np.asarray(S, float)
    W = np.zeros_like(S)
    previous = None
    for t, members in enumerate(sets):
        target = np.zeros(S.shape[1])
        if members:
            target[members] = softmax(S[t, members], temperature)
        if previous is None or not members:
            current = target
        else:
            current = alpha_ema * previous + (1 - alpha_ema) * target
            if math.isfinite(max_step):
                current = previous + np.clip(current - previous, -max_step, max_step)
            current = np.where(np.isfinite(S[t]), current, 0.)
            total = current.sum()
            current = current / total if total > 0 else target
        W[t] = current
        previous = current if members else None
    return W
