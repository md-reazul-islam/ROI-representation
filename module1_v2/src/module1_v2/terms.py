"""Phase 1b: an independent visibility term and the condition-term correlation audit.

v1 computes V = 1 - occlusion_proxy with occlusion_proxy = 1 - skin likelihood, so V and S are the same
quantity and skin is counted twice in Q. V' here is computed from pose geometry instead: the fraction of a
region's *template* keypoints (all keypoints that define the region, not only the confident ones that
v1 kept as support) whose confidence is >= 0.5. It does not look at colour at all.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def keypoint_table(keypoints):
    """{index: (x, y, confidence)} for every keypoint with finite coordinates and confidence."""
    out = {}
    for point in keypoints:
        try:
            x, y, c = float(point['x']), float(point['y']), float(point['confidence'])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (x, y, c)):
            out[int(point['index'])] = (x, y, c)
    return out


def keypoint_visibility(mapper, region, points, threshold=.5):
    """V' of one region: share of its template keypoints with confidence >= threshold (NaN if no template).
    `mapper` is the unchanged v1 AnatomicalROIMapper; its region templates are reused read-only."""
    template = list(dict.fromkeys(mapper._region_support(region, points)))
    if not template:
        return np.nan
    return float(np.mean([points[i][2] >= threshold if i in points else False for i in template]))


TERM_COLUMNS = dict(P='pose_term', V='visibility_term', V_prime='visibility_keypoint', S='skin_term',
                    C='temporal_term', M='motion_term', M_nuisance='motion_nuisance', B='blur_score',
                    L='illumination_score', Sat='saturation_score', W='swaddling_proxy', G='SAM2_mask_support')


def correlation_audit(table, limit=.8):
    """Pearson and Spearman correlation matrices of the condition terms (rows = ROI-frames) and the list of
    term pairs with |r| > limit under either coefficient."""
    columns = {k: v for k, v in TERM_COLUMNS.items() if v in table}
    data = table[list(columns.values())].apply(pd.to_numeric, errors='coerce')
    data.columns = list(columns)
    pearson, spearman = data.corr('pearson'), data.corr('spearman')
    flags = []
    names = list(columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            p, s = pearson.loc[a, b], spearman.loc[a, b]
            if (math.isfinite(p) and abs(p) > limit) or (math.isfinite(s) and abs(s) > limit):
                flags.append(dict(term_a=a, term_b=b, pearson=p, spearman=s))
    return pearson, spearman, pd.DataFrame(flags, columns=['term_a', 'term_b', 'pearson', 'spearman'])
