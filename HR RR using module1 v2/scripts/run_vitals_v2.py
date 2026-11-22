"""Module 2 v2: HR and RR from the Module 1 v2 dynamic ROI representation (module1_v2 export_dynamic_roi_v2.py).

Usage: python "HR RR using module1 v2/scripts/run_vitals_v2.py --run runs/final34_paper --module1-config D5
       [--videos VID_001 ...] [--reference-hr ref_hr.csv] [--reference-rr ref_rr.csv] [--workers 2]
Reads   module1_v2/runs/<run>/dynamic_roi_v2_<config>/dynamic_roi_v2.csv, the run's videos and SAM 2 masks.
Writes  HR RR using module1 v2/runs/<run>/<config>/{hr_windows.csv, rr_windows.csv, summary_per_video.csv, metadata.json}
        and, with references, evaluation_hr.csv / evaluation_rr.csv (MAE, RMSE, within +-5 bpm / brpm).
Window labels: validated (beats the surrogate null AND two disjoint body parts agree), present (beats the null),
not_assessable. Only validated windows feed the Kalman track and the reported rate.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(p))
from module2.respiration import RRConfig                       # noqa: E402  (Module 2 v1 constants, read-only)
from module2.rppg import HRConfig                              # noqa: E402
from run_paper_evaluation import EXTRACT, load_mask, pulse     # noqa: E402  (Module 1 pixel / pulse helpers)
from hr_rr_m1v2 import hr as hr_branch, rr as rr_branch        # noqa: E402

CFG = yaml.safe_load((M2/'configs/hr_rr_m1v2.yaml').read_text())
HR, RR = HRConfig(), RRConfig()


def video_job(job):
    record, rows, sam2_rows, out, branches, wave_dir = job
    started = time.perf_counter()
    result = dict(hr=[], rr=[])
    waves = {}
    if 'hr' in branches:
        sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
        masks = {}
        for f in rows.frame_index.unique():
            if len(sam2_rows) and f in sam2_rows.index:
                m = load_mask(Path(out)/'sam2', sam2_rows.loc[f])
                if m is not None:
                    masks[int(f)] = m
        result['hr'] = hr_branch.analyse(record, rows, masks, CFG['hr'], HR, EXTRACT, pulse, CFG['seed'], waves)
    if 'rr' in branches:
        result['rr'] = rr_branch.analyse(record, rows, CFG['rr'], RR, CFG['seed'], waves)
    for key in result:
        for r in result[key]:
            r.update(video_id=record['video_id'], split=record['split'])
    if waves:
        Path(wave_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(wave_dir)/f"{record['video_id']}.npz", **waves)
    return result, time.perf_counter() - started


def summarize(table, value, low, high):
    rows = []
    one_side = 'gate2_mode' in table and table.gate2_mode.eq('fallback').any()
    for video, p in table.groupby('video_id'):
        p = p.sort_values('window_end_sec')
        v = p[p.label == 'validated']
        rows.append({'video_id': video, 'split': p.split.iloc[0], 'windows': len(p),
                     'validated_percent': 100 * len(v) / len(p), 'present_percent': 100 * p.label.ne('not_assessable').mean(),
                     f'{value}_median_validated': v[value].median() if len(v) else np.nan,
                     f'{value}_median_all': p[value].median(),
                     f'{value}_jump_median_validated': v[value].diff().abs().median() if len(v) > 1 else np.nan,
                     'in_physiological_band_percent': 100 * p[value].between(low, high).mean(),
                     'sqi_median': p.sqi.median(), 'snr_median_db': p.snr_db.median()})
        if one_side:                                   # gate2_mode fallback only (column absent in strict runs)
            rows[-1]['validated_one_side_percent'] = 100 * p.label.eq('validated_one_side').mean()
    return pd.DataFrame(rows)


def evaluate(table, reference, value, ref_col, tolerance):
    """Nearest reference sample to every window centre (within 1 s); errors for all and for validated windows."""
    ref = pd.read_csv(reference)
    rows = []
    for subset, part in (('all_windows', table), ('validated_windows', table[table.label == 'validated'])):
        err = []
        for video, p in part.groupby('video_id'):
            r = ref[ref.video_id == video].sort_values('timestamp_sec')
            if not len(r):
                continue
            idx = np.searchsorted(r.timestamp_sec.to_numpy(), p.window_center_sec.to_numpy()).clip(0, len(r) - 1)
            near = r.iloc[idx]
            ok = (np.abs(near.timestamp_sec.to_numpy() - p.window_center_sec.to_numpy()) <= 1.) & p[value].notna().to_numpy()
            err += list(p[value].to_numpy()[ok] - near[ref_col].to_numpy()[ok])
        e = np.asarray(err, float)
        rows.append(dict(subset=subset, windows=len(e), mae=np.abs(e).mean() if len(e) else np.nan,
                         rmse=np.sqrt((e ** 2).mean()) if len(e) else np.nan,
                         within_tolerance_percent=100 * (np.abs(e) <= tolerance).mean() if len(e) else np.nan))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--module1-config', default=CFG['input']['module1_config'])
    ap.add_argument('--videos', nargs='+')
    ap.add_argument('--branches', default='hr,rr')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--reference-hr')
    ap.add_argument('--reference-rr')
    ap.add_argument('--out-name', default=None)
    ap.add_argument('--hr-region-set', choices=['module1', 'hr_specific'], default=None,
                    help='HR regions: module1 (Module 1 v2 top 3, as Module 2 v2) or hr_specific (own HR set)')
    ap.add_argument('--gate2-mode', choices=['strict', 'fallback'], default=None,
                    help='HR gate 2: strict (left/right, default) or fallback (left_right -> face_chest -> top_bottom, '
                         'else validated_one_side); hr_specific region set only')
    a = ap.parse_args(argv)
    if a.gate2_mode:
        CFG['hr']['gate2_mode'] = a.gate2_mode
    if a.hr_region_set:
        CFG['hr']['region_set'] = a.hr_region_set
    run = (ROOT/a.run).resolve()
    src = ROOT/'module1_v2/runs'/run.name/f'dynamic_roi_v2_{a.module1_config}'/'dynamic_roi_v2.csv'
    if not src.is_file():
        raise SystemExit(f'Missing Module 1 v2 representation {src}: run module1_v2/scripts/export_dynamic_roi_v2.py first')
    suffix = '_hr_specific' if CFG['hr'].get('region_set', 'module1') == 'hr_specific' else ''
    target = M2/'runs'/run.name/(a.out_name or a.module1_config + suffix)
    target.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(src, low_memory=False)
    manifest = pd.read_csv(run/'manifest.csv')
    if a.videos:
        manifest = manifest[manifest.video_id.isin(a.videos)]
    sam = pd.read_csv(run/'outputs/sam2/sam2_masks.csv', low_memory=False)
    branches = a.branches.split(',')
    jobs = [(r._asdict(), table[table.video_id == r.video_id], sam[sam.video_id == r.video_id], str(run/'outputs'), branches,
             str(target/'waveforms'))
            for r in manifest.itertuples()]
    started = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(video_job, jobs))
    meta = dict(run=str(run), module1_config=a.module1_config, representation=str(src), config=CFG,
                videos=len(jobs), wall_time_sec=time.perf_counter() - started)
    for key, value, low, high, ref, ref_col, tol in (('hr', 'hr_bpm', 100, 200, a.reference_hr, 'hr_bpm', 5.),
                                                     ('rr', 'rr_brpm', 20, 90, a.reference_rr, 'rr_brpm', 5.)):
        if key not in branches:
            continue
        t = pd.DataFrame([row for r in results for row in r[0][key]])
        if not len(t):
            continue
        t.to_csv(target/f'{key}_windows.csv', index=False)
        s = summarize(t, value, low, high)
        s.to_csv(target/f'{key}_summary_per_video.csv', index=False)
        meta[f'{key}_validated_percent_mean'] = float(s.validated_percent.mean())
        print(f'== {key.upper()}: {len(t)} windows, validated {100 * t.label.eq("validated").mean():.1f}%, '
              f'present {100 * t.label.ne("not_assessable").mean():.1f}%')
        print(s.round(2).to_string(index=False))
        if ref:
            ev = evaluate(t, ref, value, ref_col, tol)
            ev.to_csv(target/f'evaluation_{key}.csv', index=False)
            print(ev.round(2).to_string(index=False))
    (target/'metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
