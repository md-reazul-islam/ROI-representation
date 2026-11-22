"""D8 rules R1-R3 (see ../PREREGISTRATION.md; every constant here is fixed there, not tuned).

All functions work on one video's sampled frames in time order (t) and the Module 1 ROI names (r):
  s[t, r]   visible_skin_proportion (NaN = unavailable)
  A[t, r]   region available with a finite D5 score
  C[t, r]   (rn, gn, lum) colour of all polygon pixels (NaN = no colour feature)
"""
from __future__ import annotations

import numpy as np

BASELINE_SEC = 10.0          # causal baseline window
MIN_BASELINE_SEC = 2.0       # clean baseline frames needed for a decision
R1_FLAG, R1_RECOVER = 0.5, 0.75
R2_FLAG, R2_RECOVER = 3.0, 1.5
R2_PERSIST_SEC = 0.5
SC_FLOOR, SL_FLOOR = 0.005, 0.02
MAD_SCALE = 1.4826
R3_RECENT_SEC = 2.0
R3_MIN_CLEAN = 3
HR_GROUP = ('FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE', 'UPPER_CHEST')


def _mad(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def flag_regions(s, A, C, fps):
    """Return covered (R1), occluded (R2), excess (R2 statistic) arrays of shape (T, R)."""
    T, R = s.shape
    win = max(1, int(round(BASELINE_SEC * fps)))
    min_base = max(1, int(round(MIN_BASELINE_SEC * fps)))
    persist = max(1, int(round(R2_PERSIST_SEC * fps)))
    covered = np.zeros((T, R), bool)
    occluded = np.zeros((T, R), bool)
    excess = np.full((T, R), np.nan)
    clean = np.zeros((T, R), bool)              # available and not flagged: the frozen baseline pool
    cov_state = np.zeros(R, bool)
    cov_ref = np.full(R, np.nan)
    occ_state = np.zeros(R, bool)
    run_len = np.zeros(R, int)
    for t in range(T):
        lo = max(0, t - win)
        dev = np.full(R, np.nan)
        for r in range(R):
            if not A[t, r]:
                continue
            base = np.flatnonzero(clean[lo:t, r]) + lo
            # R1 coverage
            if cov_state[r]:
                if np.isfinite(s[t, r]) and s[t, r] >= R1_RECOVER * cov_ref[r]:
                    cov_state[r] = False
            elif len(base) >= min_base and np.isfinite(s[t, r]):
                med = np.nanmedian(s[base, r])
                if np.isfinite(med) and med > 0 and s[t, r] < R1_FLAG * med:
                    cov_state[r], cov_ref[r] = True, med
            # R2 deviation from the region's own clean baseline
            cb = C[base, r] if len(base) else np.empty((0, 3))
            ok = np.all(np.isfinite(cb), axis=1) if len(cb) else np.zeros(0, bool)
            if ok.sum() >= min_base and np.all(np.isfinite(C[t, r])):
                cb = cb[ok]
                med = np.median(cb, axis=0)
                if med[2] > 0:
                    dcs = np.hypot(cb[:, 0] - med[0], cb[:, 1] - med[1])
                    dls = np.abs(cb[:, 2] / med[2] - 1)
                    sc = max(MAD_SCALE * _mad(dcs), SC_FLOOR)
                    sl = max(MAD_SCALE * _mad(dls), SL_FLOOR)
                    dc = np.hypot(C[t, r, 0] - med[0], C[t, r, 1] - med[1])
                    dl = abs(C[t, r, 2] / med[2] - 1)
                    dev[r] = max(dc / sc, dl / sl)
        if np.isfinite(dev).any():
            excess[t] = dev - np.nanmedian(dev)
        for r in range(R):
            if not A[t, r]:
                run_len[r] = 0
                continue
            e = excess[t, r]
            if occ_state[r]:
                if np.isfinite(e) and e < R2_RECOVER:
                    occ_state[r], run_len[r] = False, 0
            else:
                run_len[r] = run_len[r] + 1 if (np.isfinite(e) and e > R2_FLAG) else 0
                if run_len[r] >= persist:
                    occ_state[r] = True
            covered[t, r], occluded[t, r] = cov_state[r], occ_state[r]
            clean[t, r] = not (cov_state[r] or occ_state[r])
    return covered, occluded, excess


def redundancy(flagged, A, s, area, names, fps):
    """R3: affected[t] and the fusion weights W[t, r] used in affected frames (area-weighted clean HR-group regions)."""
    T, R = flagged.shape
    group = np.array([n in HR_GROUP for n in names])
    recent = max(1, int(round(R3_RECENT_SEC * fps)))
    any_flag = (flagged & group).any(axis=1)
    clean_now = A & ~flagged & group
    affected = np.zeros(T, bool)
    W = np.zeros((T, R))
    for t in range(T):
        affected[t] = any_flag[max(0, t - recent + 1):t + 1].any() or clean_now[t].sum() < R3_MIN_CLEAN
        if affected[t]:
            w = np.where(clean_now[t], np.nan_to_num(s[t]) * np.nan_to_num(area[t]), 0.)
            if w.sum() > 0:
                W[t] = w / w.sum()
    return affected, W
