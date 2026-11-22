"""Reference HR per camera window, matched the way the UBFC check notebook does it (sections 7, 10, 13).

reference_hr_ppg.csv already holds ONE value per 8-s window: the FFT HR of the fingertip PPG in the 8-s window centred
at timestamp_sec (1-s step). The comparison tool (compare_roi_fixed_vs_dynamic.attach_reference) averages every
reference value whose timestamp lies inside the camera window; for this file that averages ~8 overlapping 8-s values,
i.e. smooths the reference over ~16 s. The right match here is the reference value whose centre is nearest to the
camera window centre (within 1 s). The tool's value is kept in column hr_ref_window_mean for transparency.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REF_CSV = Path(__file__).resolve().parent/'reference_hr_ppg.csv'
MAX_SEC = 1.0


def nearest_reference(video_id, center_sec, ref_csv=REF_CSV, max_sec=MAX_SEC):
    """Array of reference HR (bpm) for the given windows: nearest reference centre within max_sec, else NaN."""
    ref = pd.read_csv(ref_csv)
    video_id, center_sec = np.asarray(video_id), np.asarray(center_sec, float)
    out = np.full(len(center_sec), np.nan)
    for v in np.unique(video_id):
        r = ref[ref.video_id == v].sort_values('timestamp_sec')
        if r.empty:
            continue
        t, h = r.timestamp_sec.to_numpy(float), r.hr_bpm.to_numpy(float)
        sel = np.flatnonzero(video_id == v)
        c = center_sec[sel]
        j = np.clip(np.searchsorted(t, c), 1, len(t) - 1) if len(t) > 1 else np.zeros(len(c), int)
        if len(t) > 1:
            j = np.where(np.abs(t[j - 1] - c) <= np.abs(t[j] - c), j - 1, j)
        out[sel] = np.where(np.abs(t[j] - c) <= max_sec, h[j], np.nan)
    return out


def fix_reference(df, ref_csv=REF_CSV):
    """Replace df.hr_ref (tool: window mean) by the nearest-centre reference; keep the old one as hr_ref_window_mean."""
    df = df.copy()
    if 'hr_ref' in df:
        df['hr_ref_window_mean'] = df['hr_ref']
    df['hr_ref'] = nearest_reference(df.video_id, df.window_center_sec, ref_csv)
    return df
