"""Ablation study of the v1 / v2 / v3 camera vitals pipelines: version definitions, per-video metrics, statistics.

The unit of analysis is the video (windows of one video are strongly autocorrelated): every metric is computed
per video first and versions are compared with paired, non-parametric tests over videos (Wilcoxon signed-rank,
Holm-corrected; Friedman across v1 / v2 / v3), effect size as the matched-pairs rank-biserial correlation and
95 % bootstrap confidence intervals of the mean over videos.

Without a reference, the metrics describe signal quality, stability and coverage, not accuracy:
  reliable_percent      windows whose SQI passes the branch threshold (v2 / v3 also need agreeing sites)
  sqi_median, snr_median_db
  coverage_percent      windows with any estimate
  jump_median           median |rate change| between consecutive 1 s windows (lower = steadier)
  plausible_percent     estimates inside the physiological neonatal range
With reference HR / RR, window-level accuracy (MAE, RMSE, MAPE, within 5 / 10, Bland-Altman, Pearson, CCC, ICC)
comes from module2.evaluation.agreement.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from module2.evaluation import agreement


# key, label, pulse method, fusion strategy, removed component (None for the full versions)
HR_VERSIONS = (
    ('v1', 'v1: POS + Module 1 dynamic fusion', 'POS', 'dynamic_fusion', None),
    ('v2', 'v2: + multi-site consensus + harmonic-aware', 'POS', 'cardiorespiratory_fusion', None),
    ('v3', 'v3: + neonate-adapted neural rPPG', 'NEURAL_TTA', 'cardiorespiratory_fusion', None),
)
HR_COMPONENTS = (
    # v1 / v2 components
    ('v2 - harmonic awareness', 'POS', 'consensus_fusion', 'v2', 'respiratory-harmonic-aware peak selection'),
    ('v2 - multi-site consensus (= v1)', 'POS', 'dynamic_fusion', 'v2', 'multi-site consensus + harmonic awareness'),
    ('v1 - Module 1 prior', 'POS', 'snr_fusion', 'v1', 'Module 1 confidence prior (SNR weights only)'),
    ('v1 - signal evidence', 'POS', 'confidence_fusion', 'v1', 'SNR goodness (Module 1 prior only)'),
    ('v1 - anatomical ROIs (infant box)', 'POS', 'infant_box', 'v1', 'anatomical ROIs (whole infant box)'),
    # v3 components
    ('v3 - adaptation (source model)', 'NEURAL_SRC', 'cardiorespiratory_fusion', 'v3', 'test-time adaptation'),
    ('v3 - Module 1 guidance', 'NEURAL_TTA_UNGUIDED', 'cardiorespiratory_fusion', 'v3', 'all Module 1 guidance'),
    ('v3 - Q-sampling', 'NEURAL_TTA_NO_QSAMPLE', 'cardiorespiratory_fusion', 'v3', 'confidence-guided clip sampling'),
    ('v3 - Q-weighting', 'NEURAL_TTA_NO_QWEIGHT', 'cardiorespiratory_fusion', 'v3', 'confidence-weighted loss'),
    ('v3 - cross-site loss', 'NEURAL_TTA_NO_SITES', 'cardiorespiratory_fusion', 'v3', 'cross-site PSD consistency'),
    ('v3 - harmonic awareness', 'NEURAL_TTA', 'consensus_fusion', 'v3', 'respiratory-harmonic-aware peak selection'),
    ('v3 - multi-site fusion', 'NEURAL_TTA', 'dynamic_fusion', 'v3', 'multi-site consensus + harmonic awareness'),
    ('v3 - neural pulse (= v2)', 'POS', 'cardiorespiratory_fusion', 'v3', 'neural pulse model (POS instead)'),
)
RR_VERSIONS = (
    ('v1', 'v1: FLOW + Module 1 dynamic fusion', 'FLOW', 'dynamic_fusion', None),
    ('v2', 'v2 / v3: + thoraco-abdominal phase', 'FLOW', 'dynamic_fusion', None),
)
HR_RANGE, RR_RANGE = (100., 200.), (20., 90.)


def per_video_quality(windows, rate, plausible_range, keys=('dataset', 'video_id', 'method', 'strategy')):
    """One row per video x configuration with the reference-free metrics (see module docstring)."""
    rows = []
    for key, group in windows.groupby(list(keys), sort=False):
        group = group.sort_values('window_end_sec')
        values = group[rate].astype(float)
        steps = values.diff().abs()
        rows.append(dict(zip(keys, key), windows=len(group),
                         coverage_percent=100. * values.notna().mean(),
                         reliable_percent=100. * group.reliable.astype(bool).mean(),
                         sqi_median=group.sqi.median(), snr_median_db=group.snr_db.replace([np.inf, -np.inf], np.nan).median(),
                         rate_median=values.median(), rate_median_reliable=values[group.reliable.astype(bool)].median(),
                         jump_median=steps.median() if steps.notna().any() else np.nan,
                         plausible_percent=100. * values.between(*plausible_range).mean()))
    return pd.DataFrame(rows)


def bootstrap_ci(values, statistic=np.mean, n=5000, seed=0, level=.95):
    """(estimate, low, high) percentile bootstrap over the finite values."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    estimate = float(statistic(values))
    if len(values) < 2:
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    tail = (1 - level) / 2 * 100
    return estimate, float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def paired_test(a, b):
    """Wilcoxon signed-rank of b - a over pairs with both values; rank-biserial r in [-1, 1] (> 0: b larger)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    difference = b[keep] - a[keep]
    result = dict(n=int(keep.sum()), mean_difference=float(difference.mean()) if keep.any() else np.nan,
                  median_difference=float(np.median(difference)) if keep.any() else np.nan, p_value=np.nan,
                  rank_biserial=np.nan)
    nonzero = difference[difference != 0]
    if len(nonzero) == 0:
        result.update(rank_biserial=0., p_value=1. if keep.any() else np.nan)
        return result
    ranks = stats.rankdata(np.abs(nonzero))
    result['rank_biserial'] = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())
    if len(nonzero) >= 2:
        result['p_value'] = float(stats.wilcoxon(nonzero).pvalue)
    return result


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p_values, float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * p[index]))
        out[index] = running
    return out


def friedman(table):
    """Friedman test over the columns of a (videos x conditions) table, complete rows only."""
    complete = table.dropna()
    if len(complete) < 2 or complete.shape[1] < 3:
        return dict(n=len(complete), statistic=np.nan, p_value=np.nan)
    result = stats.friedmanchisquare(*[complete[c] for c in complete])
    return dict(n=len(complete), statistic=float(result.statistic), p_value=float(result.pvalue))


def compare(per_video, baseline, candidates, metrics, label='version'):
    """Paired comparisons of every candidate against the baseline for every metric, Holm-corrected per metric."""
    rows = []
    for metric in metrics:
        wide = per_video.pivot_table(index='video_key', columns=label, values=metric, aggfunc='first')
        block = []
        for candidate in candidates:
            if candidate not in wide or baseline not in wide:
                continue
            block.append(dict(metric=metric, baseline=baseline, candidate=candidate,
                              **paired_test(wide[baseline], wide[candidate])))
        for row, adjusted in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = adjusted
        rows += block
    return pd.DataFrame(rows)


def summarize(per_video, group, metrics, seed=0):
    """Mean [95 % bootstrap CI] over videos of every metric per group value."""
    rows = []
    for name, part in per_video.groupby(group, sort=False, observed=True):
        row = {group: name, 'videos': part.video_key.nunique()}
        for metric in metrics:
            row[metric], row[f'{metric}_ci_low'], row[f'{metric}_ci_high'] = bootstrap_ci(part[metric], seed=seed)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_reference(windows, reference, value):
    """Mean reference value inside every window (reference: dataset?, video_id, timestamp_sec, <value>)."""
    keys = ['dataset', 'video_id'] if 'dataset' in reference and 'dataset' in windows else ['video_id']
    groups = {k if isinstance(k, tuple) else (k,): g for k, g in reference.groupby(keys)}
    out = []
    for row in windows.itertuples():
        group = groups.get(tuple(getattr(row, k) for k in keys))
        if group is None:
            out.append(np.nan)
            continue
        inside = group[(group.timestamp_sec >= row.window_start_sec) & (group.timestamp_sec <= row.window_end_sec)]
        out.append(inside[value].mean() if len(inside) else np.nan)
    return windows.assign(reference=out)


def accuracy(pairs, estimate):
    """Window-level agreement of `estimate` with the paired reference (module2.evaluation.agreement)."""
    return agreement(pairs[estimate], pairs.reference)


def coverage_accuracy(pairs, estimate, thresholds=np.round(np.arange(0., 1.01, .1), 2)):
    """MAE and coverage when only windows with SQI >= threshold are kept (the reliability-accuracy trade-off)."""
    rows = []
    usable = pairs[pairs[estimate].notna() & pairs.reference.notna()]
    for threshold in thresholds:
        part = usable[usable.sqi >= threshold]
        rows.append(dict(sqi_threshold=float(threshold), coverage_percent=100. * len(part) / len(usable) if len(usable) else 0.,
                         mae=float((part[estimate] - part.reference).abs().mean()) if len(part) else np.nan, n=len(part)))
    return pd.DataFrame(rows)


def fmt_ci(value, low, high, digits=1):
    if value is None or not np.isfinite(value):
        return 'n/a'
    if low is None or not np.isfinite(low):
        return f'{value:.{digits}f}'
    return f'{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]'


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return 'n/a'
    return '<0.001' if p < .001 else f'{p:.3f}'


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if math.isfinite(value) else np.nan
