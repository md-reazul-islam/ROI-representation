"""Agreement metrics between camera HR and a reference HR (ECG / pulse oximeter / bedside monitor)."""
from __future__ import annotations

import math

import numpy as np


def icc_2_1(a, b):
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss 1979)."""
    data = np.column_stack([a, b]).astype(float)
    n, k = data.shape
    if n < 2:
        return None
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_error = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator > 0 else None


def agreement(estimate, reference):
    estimate, reference = np.asarray(estimate, dtype=float), np.asarray(reference, dtype=float)
    keep = np.isfinite(estimate) & np.isfinite(reference)
    estimate, reference = estimate[keep], reference[keep]
    n = len(estimate)
    if n == 0:
        return dict(n=0)
    error = estimate - reference
    result = dict(n=int(n), mae_bpm=float(np.mean(np.abs(error))), rmse_bpm=float(math.sqrt(np.mean(error ** 2))),
                  mape_percent=float(np.mean(np.abs(error) / reference) * 100),
                  within_5bpm_percent=float(np.mean(np.abs(error) <= 5) * 100),
                  within_10bpm_percent=float(np.mean(np.abs(error) <= 10) * 100),
                  bland_altman_bias_bpm=float(error.mean()))
    if n >= 2:
        sd = float(error.std(ddof=1))
        result.update(bland_altman_loa_low_bpm=result['bland_altman_bias_bpm'] - 1.96 * sd,
                      bland_altman_loa_high_bpm=result['bland_altman_bias_bpm'] + 1.96 * sd)
        if estimate.std() > 0 and reference.std() > 0:
            result['pearson_r'] = float(np.corrcoef(estimate, reference)[0, 1])
        covariance = np.mean((estimate - estimate.mean()) * (reference - reference.mean()))
        denominator = estimate.var() + reference.var() + (estimate.mean() - reference.mean()) ** 2
        result['ccc'] = float(2 * covariance / denominator) if denominator > 0 else None
        result['icc_2_1'] = icc_2_1(estimate, reference)
    return result
