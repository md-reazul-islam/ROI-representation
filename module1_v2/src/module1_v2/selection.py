"""Phase 1 configurations: per-video score matrices, D-config selection and v1-compatible outputs.

A configuration is a dict of flags (see configs/module1_v2.yaml). `select_video` returns
  geometry: {frame: [(roi, weight, polygon, q01)]} - the selected set (feeds v1 paper_eval.geometry_metrics)
  fusion:   {frame: [(roi, weight, None, nan)]}   - per-frame fusion weights incl. cross-fading regions
                                                     (feeds v1 paper_eval.native_weights)
With the v1_equivalent flags both are identical to v1 paper_eval.selections(candidates, 'Full').
"""
from __future__ import annotations

import math

import numpy as np

from module1 import paper_eval as pe
from module1_v2 import stabilization as st
from module1_v2 import subject as sj

NAMES = list(pe.ROI_NAMES)
W_EQUAL = .2                      # v1 equal_fixed weights (normalized)


def pixel_polygon(polygon, box_mode='skip'):
    """Polygon used for PIXEL extraction. v1 ('skip'): a 2-point bounding box [[x1, y1], [x2, y2]] has no area and is
    dropped (None), so such a region can be selected but contributes no pixels. v2 ('rectangle'): the box becomes its
    4-corner rectangle. Polygons with >= 3 points are returned unchanged in both modes."""
    if not polygon:
        return None
    if len(polygon) >= 3:
        return polygon
    if box_mode == 'rectangle' and len(polygon) == 2:
        (xa, ya), (xb, yb) = polygon
        x1, x2, y1, y2 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
        if x2 > x1 and y2 > y1:
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return None


def score_column(table, visibility='v1', motion='total'):
    """Per ROI-frame score. v1 flags: the stored Phase 10 Q (bit-identical). Otherwise the same linear rule
    with V replaced by V' (keypoint visibility) and/or M replaced by nuisance motion."""
    if visibility == 'v1' and motion == 'total':
        return table.q.to_numpy(float)
    v = table.visibility_keypoint if visibility == 'keypoint' else table.visibility_term
    m = table.motion_nuisance if motion == 'residual_nuisance' else table.motion_term
    parts = [table.pose_term, v, table.skin_term, table.temporal_term.fillna(1.)]
    q = W_EQUAL * sum(np.asarray(p, float) for p in parts) - W_EQUAL * np.nan_to_num(np.asarray(m, float), nan=0.)
    return q


def score_matrix(candidates, frames, values):
    """Q[t, r] over the video's sampled frames (NaN = unavailable) plus polygons[t][r]."""
    row_of = {f: i for i, f in enumerate(frames)}
    Q = np.full((len(frames), len(NAMES)), np.nan)
    polygons = [dict() for _ in frames]
    col = {n: j for j, n in enumerate(NAMES)}
    available = candidates.roi_status.to_numpy() == 'AVAILABLE'
    for frame, roi, polygon, value, ok in zip(candidates.frame_index.to_numpy(), candidates.roi_name.to_numpy(),
                                              candidates.polygon.to_numpy(), values, available):
        if not ok or not math.isfinite(value) or roi not in col:
            continue
        t, r = row_of[int(frame)], col[roi]
        Q[t, r] = value
        polygons[t][r] = polygon
    return Q, polygons


def select_video(candidates, frames, flags, fps_sampled, k=3, temperature=1., reject=None):
    """Selection of one video under one configuration (flags dict). `reject[t]` True = subject check
    failed at frame t (abstain: no ROI)."""
    values = score_column(candidates, flags.get('visibility', 'v1'), flags.get('motion', 'total'))
    Q, polygons = score_matrix(candidates, frames, values)
    if reject is not None:
        Q[np.asarray(reject, bool)] = np.nan
    S = st.ema(Q, flags.get('ema_beta', 0.))
    method = flags.get('selection', 'topk')
    if method == 'topk':
        sets = st.topk_sets(S, k, NAMES)
    elif method == 'hysteresis':
        sets = st.hysteresis_sets(S, k, NAMES, flags['hysteresis_delta'], flags['hysteresis_frames'])
    elif method == 'viterbi':
        lag = flags.get('viterbi_lag')
        anchors = st.viterbi_anchor(Q, flags['viterbi_penalty'], None if lag is None else int(round(lag * fps_sampled)))
        sets = st.hysteresis_sets(S, k, NAMES, flags['hysteresis_delta'], flags['hysteresis_frames'], fixed=anchors)
        sets = [([a] + [r for r in s if r != a]) if a is not None and a in s else s for a, s in zip(anchors, sets)]
    else:
        raise ValueError(f'unknown selection {method}')
    crossfade = flags.get('crossfade_sec', 0.)
    max_step = 1 / (crossfade * fps_sampled) if crossfade and fps_sampled > 0 else math.inf
    W = st.fusion_weights(S, sets, temperature, flags.get('alpha_ema', 0.), max_step)
    geometry, fusion = {}, {}
    for t, frame in enumerate(frames):
        members = sets[t]
        ordered = members          # selection order: top-1 = best smoothed score (Viterbi: the anchor)
        total = W[t, members].sum() if members else 0.
        geometry[frame] = [(NAMES[r], float(W[t, r] / total) if total > 0 else 0., pe.parse(polygons[t][r]),
                            float(pe.rescaled(Q[t, r], 'Full'))) for r in ordered]
        fusion[frame] = [(NAMES[r], float(W[t, r]), None, np.nan) for r in np.flatnonzero(W[t] > 0)]
    return geometry, fusion, Q, sets


def retention(Q, sets, k=3):
    """Mean over frames of (sum of rescaled Q of the selected set) / (sum of the per-frame top-k rescaled Q):
    how much instantaneous confidence a stabilized selection gives up (1 = none)."""
    values = []
    for row, members in zip(np.asarray(Q, float), sets):
        finite = np.sort(row[np.isfinite(row)])[::-1]
        if not len(finite) or not members:
            continue
        best = (finite[:k] + .2).sum()
        values.append((row[members] + .2).sum() / best if best > 0 else 1.)
    return float(np.mean(values)) if values else np.nan


def switches_per_sec(sets, fps_sampled):
    tops = [s[0] if s else None for s in sets]
    pairs = [(a, b) for a, b in zip(tops, tops[1:]) if a is not None and b is not None]
    return float(np.mean([a != b for a, b in pairs]) * fps_sampled) if pairs else np.nan


def subject_reject(subject_rows, frames):
    """Boolean per sampled frame from the per-frame subject table (REJECT = True)."""
    status = dict(zip(subject_rows.frame_index.astype(int), subject_rows.subject_status))
    return np.array([status.get(int(f)) == 'REJECT' for f in frames])


__all__ = ['score_column', 'score_matrix', 'select_video', 'retention', 'switches_per_sec', 'subject_reject', 'sj', 'pixel_polygon']
