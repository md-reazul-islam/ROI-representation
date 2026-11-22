"""Label-free and label-based evaluation of the Module 1 dynamic ROI representation (paper experiments).

Configurations (the same sampled frames, Phase 2-9 outputs and selection size k for all):
  B0  fixed box ROI       centre crop of the first-frame infant box, held for the whole video (no anatomy, no tracking)
  B1  fixed anatomical    the region with the highest anatomical confidence on the first frame, geometry held fixed
  B2  region dynamic      per frame, top-k regions by pose (anatomical) confidence only
  B3  + condition         per frame, top-k by mean(pose, visibility, skin)
  B4  + temporal/motion   the implemented Phase 10 score Q = w.(P, V, S, C) - w_M M, equal weights (no SAM 2)
  Full                    B4 on the SAM 2 branch (mask-constrained pose, ROI n infant mask, mask support)
Fusion weights inside the selection are softmax(Q / T), as in Phase 10.

Label-free metrics, per video (the statistical unit):
  coverage           % sampled frames with a selected ROI
  temporal IoU       IoU of the selected-ROI union box between consecutive sampled frames
  centroid jitter    |centroid change| / infant-box diagonal between consecutive sampled frames
  selection Jaccard  overlap of the selected region names between consecutive sampled frames
  mask support       fraction of selected ROI pixels inside the SAM 2 infant mask (weighted by fusion weight)
  pulse SNR          task-based validity: POS pulse of the fused ROI colour, 8 s windows, de Haan & Jeanne SNR (dB);
                     it uses no input of Q, so Q-vs-SNR agreement is an independent check of the confidence
Q validity: per ROI and window, Spearman(Q, SNR), AUROC of Q for "pulse-informative" (SNR >= tau), and the
calibration (ECE, Brier) of the affinely rescaled Q in [0, 1]. These are proxy checks, not expert ground truth.
With expert labels (annotation kit): expert-valid ROI rate, selection accuracy, AUROC / AUPRC / ECE / Brier of Q.
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pandas as pd
from scipy import stats

ROI_NAMES = ('HEAD', 'FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE', 'NECK', 'LEFT_SHOULDER',
             'RIGHT_SHOULDER', 'UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN', 'THORACO_ABDOMINAL', 'LEFT_ARM', 'RIGHT_ARM',
             'LEFT_LEG', 'RIGHT_LEG')
CONFIGS = ('B0', 'B1', 'B2', 'B3', 'B4', 'Full')
CONFIG_NAMES = dict(B0='B0 fixed box ROI', B1='B1 fixed anatomical ROI', B2='B2 region-level dynamic (anatomy)',
                    B3='B3 + visibility / skin', B4='B4 + temporal / motion (no SAM 2)',
                    Full='Full: B4 + SAM 2 mask support')
TERMS = ('pose_term', 'visibility_term', 'skin_term', 'temporal_term', 'motion_term')


def parse(value):
    if isinstance(value, str) and value[:1] in '[{':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def score(frame, config):
    """Per-ROI score of a configuration on one frame's candidate rows (NaN where not computable)."""
    p, v, s, c, m = (frame[t].to_numpy(float) for t in TERMS)
    if config == 'B2':
        return p
    if config == 'B3':
        return (p + v + s) / 3
    return frame.q.to_numpy(float)          # B4 / Full: the stored Phase 10 Q


def rescaled(q, config):
    """Score mapped to [0, 1] (B4 / Full: (Q + w_M) / sum(w) with equal weights; B2 / B3 are already in [0, 1])."""
    return (q + .2) / 1. if config in ('B4', 'Full') else q


def softmax(values, temperature=1.):
    values = np.asarray(values, float) / temperature
    values = np.exp(values - values.max())
    return values / values.sum()


def box_of(polygon):
    polygon = np.asarray(polygon, float)
    return polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else np.nan


def fixed_rois(candidates, infant_boxes, fraction=.5):
    """B0 / B1 geometry of one video: centre crop of the first infant box; first-frame best anatomical region."""
    b0 = b1 = None
    for frame in sorted(infant_boxes):
        box = infant_boxes[frame]
        if box is not None:
            x1, y1, x2, y2 = box
            dx, dy = (x2 - x1) * (1 - fraction) / 2, (y2 - y1) * (1 - fraction) / 2
            b0 = [[x1 + dx, y1 + dy], [x2 - dx, y1 + dy], [x2 - dx, y2 - dy], [x1 + dx, y2 - dy]]
            break
    for frame, rows in sorted(candidates.groupby('frame_index')):
        rows = rows[rows.roi_status == 'AVAILABLE']
        if len(rows):
            best = rows.sort_values('anatomical_confidence', ascending=False).iloc[0]
            b1 = (best.roi_name, parse(best.polygon))
            break
    return b0, b1


def selections(candidates, config, k=3, temperature=1., fixed=None):
    """{frame_index: [(roi_key, weight, polygon, q_rescaled)]} for one video and configuration."""
    out = {}
    for frame, rows in candidates.groupby('frame_index', sort=True):
        if config in ('B0', 'B1'):
            polygon = fixed[0] if config == 'B0' else (fixed[1][1] if fixed[1] else None)
            out[frame] = [(config, 1., polygon, np.nan)] if polygon is not None else []
            continue
        rows = rows[rows.roi_status == 'AVAILABLE']
        values = score(rows, config)
        keep = np.isfinite(values)
        rows, values = rows[keep], values[keep]
        if not len(rows):
            out[frame] = []
            continue
        order = np.lexsort((rows.roi_name.to_numpy(), -values))[:k]
        weights = softmax(values[order], temperature)
        out[frame] = [(rows.roi_name.iloc[i], float(w), parse(rows.polygon.iloc[i]), float(rescaled(values[i], config)))
                      for i, w in zip(order, weights)]
    return out


def geometry_metrics(selected, infant_boxes, fps_sampled, masks=None):
    """Coverage, temporal IoU, centroid jitter, selection Jaccard, top-1 switch rate and mask support of one video."""
    frames = sorted(selected)
    covered = [f for f in frames if selected[f]]
    ious, jitter, jaccard, switches, support = [], [], [], [], []
    previous = None
    for frame in frames:
        rows = [r for r in selected[frame] if r[2] is not None]
        if not rows:
            previous = None
            continue
        boxes = [box_of(r[2]) for r in rows]
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        weights = np.array([r[1] for r in rows])
        centre = np.average([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes], axis=0, weights=weights)
        names = {r[0] for r in rows}
        top = rows[0][0]
        if previous is not None:
            ious.append(box_iou(previous[0], union))
            infant = infant_boxes.get(frame)
            if infant is not None:
                diagonal = math.hypot(infant[2] - infant[0], infant[3] - infant[1])
                jitter.append(float(np.linalg.norm(centre - previous[1]) / diagonal) if diagonal > 0 else np.nan)
            jaccard.append(len(names & previous[2]) / len(names | previous[2]))
            switches.append(top != previous[3])
        previous = (union, centre, names, top)
        if masks is not None and frame in masks:
            support.append(mask_support(rows, masks[frame]))
    return dict(coverage_percent=100. * len(covered) / len(frames) if frames else np.nan,
                temporal_iou=np.nanmean(ious) if ious else np.nan,
                centroid_jitter=np.nanmean(jitter) if jitter else np.nan,
                selection_jaccard=np.mean(jaccard) if jaccard else np.nan,
                switches_per_sec=np.mean(switches) * fps_sampled if switches else np.nan,
                mask_support=np.nanmean(support) if support else np.nan)


def mask_support(rows, mask):
    """Fusion-weighted fraction of ROI pixels inside the binary infant mask."""
    values, weights = [], []
    for _, weight, polygon, _ in rows:
        canvas = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(canvas, [np.rint(np.asarray(polygon)).astype(np.int32)], 1)
        area = canvas.sum()
        if area:
            values.append(float((canvas.astype(bool) & mask).sum() / area))
            weights.append(weight)
    return float(np.average(values, weights=weights)) if values else np.nan


def window_bounds(n, fps, window_sec=8., step_sec=1.):
    length, step = int(round(window_sec * fps)), max(1, int(round(step_sec * fps)))
    return [(s, s + length) for s in range(0, n - length + 1, step)] if n >= length else ([(0, n)] if n >= 5 * fps else [])


def fused_pulse(pulses, weights):
    finite = np.isfinite(pulses)
    w = np.where(finite, weights, 0.)
    total = w.sum(axis=1)
    return np.divide((np.where(finite, pulses, 0.) * w).sum(axis=1), total, out=np.full(len(total), np.nan),
                     where=total > 0)


def native_weights(selected, keys, hold, n):
    """(n, len(keys)) per-native-frame fusion weights from the per-sampled-frame selection (hold index)."""
    index = {k: i for i, k in enumerate(keys)}
    by_frame = {}
    for frame, rows in selected.items():
        vector = np.zeros(len(keys))
        for key, weight, _, _ in rows:
            if key in index:
                vector[index[key]] += weight
        by_frame[frame] = vector
    out = np.zeros((n, len(keys)))
    for i in range(n):
        source = int(hold[i]) if i < len(hold) else -1
        if source in by_frame:
            out[i] = by_frame[source]
    return out


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    if labels.all() or not labels.any():
        return np.nan
    ranks = stats.rankdata(scores)
    positives = labels.sum()
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * (len(labels) - positives)))


def auprc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    if not labels.any():
        return np.nan
    order = np.argsort(-scores)
    hits = labels[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / hits.sum())


def calibration(probabilities, labels, bins=10):
    """(ECE, Brier, reliability-diagram table) of probabilities in [0, 1] against binary labels."""
    p, y = np.asarray(probabilities, float), np.asarray(labels, float)
    keep = np.isfinite(p)
    p, y = np.clip(p[keep], 0, 1), y[keep]
    if not len(p):
        return np.nan, np.nan, pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.
    for b in range(bins):
        inside = index == b
        if inside.any():
            ece += inside.mean() * abs(p[inside].mean() - y[inside].mean())
            rows.append(dict(bin_low=edges[b], bin_high=edges[b + 1], n=int(inside.sum()), mean_confidence=p[inside].mean(),
                             observed_rate=y[inside].mean()))
    return float(ece), float(np.mean((p - y) ** 2)), pd.DataFrame(rows)


def expert_metrics(labels, selected_by_config, scores):
    """Expert-label metrics: labels has video_id, frame_index, roi_name, valid (0/1) and optionally best (0/1).

    selected_by_config: {config: {(video, frame): [roi names in selection order]}}; scores: DataFrame video_id,
    frame_index, roi_name, config, q01 (rescaled score) for the AUROC / calibration of each scored configuration.
    """
    rows = []
    valid = labels.set_index(['video_id', 'frame_index', 'roi_name']).valid.astype(float)
    best = (labels[labels.get('best', 0).astype(float) > 0].groupby(['video_id', 'frame_index']).roi_name.apply(set)
            if 'best' in labels else pd.Series(dtype=object))
    for config, selections_ in selected_by_config.items():
        hits, total, correct, judged = 0, 0, 0, 0
        for (video, frame), names in selections_.items():
            for name in names:
                key = (video, frame, name)
                if key in valid.index:
                    hits += valid[key]
                    total += 1
            if names and (video, frame) in best.index:
                correct += names[0] in best[(video, frame)]
                judged += 1
        row = dict(configuration=config, labelled_selected_rois=total,
                   expert_valid_roi_percent=100. * hits / total if total else np.nan,
                   selection_accuracy_percent=100. * correct / judged if judged else np.nan, judged_frames=judged)
        part = scores[scores.config == config].merge(labels, on=['video_id', 'frame_index', 'roi_name'])
        if len(part):
            ece, brier, _ = calibration(part.q01, part.valid)
            row.update(auroc=auroc(part.q01, part.valid.astype(bool)), auprc=auprc(part.q01, part.valid.astype(bool)),
                       ece=ece, brier=brier, labelled_candidates=len(part))
        rows.append(row)
    return pd.DataFrame(rows)
