"""Label HR / RR / RR+ windows that overlap caregiver contact as 'caregiver_intervention' and exclude them from
the summaries and the accuracy evaluation (standard practice in NICU camera studies).

The HR / RR estimates themselves are NOT changed: the window tables are copied with extra columns.
Usage:  python "HR RR using module1 v2/scripts/apply_caregiver_labels.py" --run runs/<RUN>
            --results "HR RR using module1 v2/runs/<RUN>/D5_hr_specific" [--rr-plus rr_plus/runs/<RUN>/D5]
            [--reference-hr ref_hr.csv] [--reference-rr ref_rr.csv]
Reads   HR RR using module1 v2/runs/<RUN>/caregiver/{caregiver_segments.csv, caregiver_frames.csv}
        (from detect_caregiver_contact.py) and the window tables in --results (and --rr-plus).
Writes  <results>/caregiver/
          hr_windows_caregiver.csv, rr_windows_caregiver.csv, rr_plus_windows_caregiver.csv
              original columns + caregiver_overlap_sec, caregiver_overlap_fraction, adult_in_view_fraction,
              caregiver_intervention (bool), label_with_caregiver (= 'caregiver_intervention' or the original label)
          caregiver_window_summary.csv   per signal and video: windows, intervention windows, validated % of all windows
              and of the windows WITHOUT intervention, median validated rate with and without intervention windows
          evaluation_<hr|rr|rr_plus>_caregiver.csv   (with references) MAE / RMSE / within tolerance for all and validated
              windows, with and without the intervention windows (the paper reports the 'excl_intervention' rows)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
CFG = yaml.safe_load((M2/'configs/caregiver.yaml').read_text())


def evaluate(table, reference, value, ref_col, tolerance, label_col='label'):
    """Same matching as run_vitals_v2.evaluate: nearest reference sample to every window centre (within 1 s)."""
    ref = pd.read_csv(reference)
    rows = []
    for subset, part in (('all_windows', table), ('validated_windows', table[table[label_col] == 'validated'])):
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


def label_windows(table, segs, frames):
    t = table.copy()
    checked = set(frames.video_id) if len(frames) else set()
    overlap, frac, adult = [], [], []
    for r in t.itertuples():
        a, b = float(r.window_start_sec), float(r.window_end_sec)
        s = segs[segs.video_id == r.video_id]
        o = float(sum(max(0., min(b, pe) - max(a, ps)) for ps, pe in zip(s.pad_start_sec, s.pad_end_sec)))
        overlap.append(o)
        frac.append(o / (b - a) if b > a else 0.)
        f = frames[(frames.video_id == r.video_id) & (frames.timestamp_sec >= a) & (frames.timestamp_sec <= b)] if len(frames) else frames
        adult.append(float(f.adult_in_view.astype(bool).mean()) if len(f) else np.nan)
    t['caregiver_checked'] = t.video_id.isin(checked)
    t['caregiver_overlap_sec'] = overlap
    t['caregiver_overlap_fraction'] = frac
    t['adult_in_view_fraction'] = adult
    t['caregiver_intervention'] = t.caregiver_overlap_fraction > CFG['windows']['min_overlap_fraction']
    t['label_with_caregiver'] = np.where(t.caregiver_intervention, 'caregiver_intervention', t.label)
    return t


def summarize(t, signal, value):
    rows = []
    for video, p in t.groupby('video_id'):
        keep = p[~p.caregiver_intervention]
        v_all, v_keep = p[p.label == 'validated'], keep[keep.label == 'validated']
        rows.append({'signal': signal, 'video_id': video, 'caregiver_checked': bool(p.caregiver_checked.iloc[0]),
                     'windows': len(p), 'intervention_windows': int(p.caregiver_intervention.sum()),
                     'intervention_percent': 100 * p.caregiver_intervention.mean(),
                     'windows_without_intervention': len(keep),
                     'validated_windows_all': len(v_all), 'validated_percent_all': 100 * len(v_all) / len(p),
                     'validated_windows_excl_intervention': len(v_keep),
                     'validated_percent_excl_intervention': 100 * len(v_keep) / len(keep) if len(keep) else np.nan,
                     'validated_inside_intervention': len(v_all) - len(v_keep),
                     'present_percent_excl_intervention': 100 * keep.label.ne('not_assessable').mean() if len(keep) else np.nan,
                     f'{value}_median_validated_all': v_all[value].median() if len(v_all) else np.nan,
                     f'{value}_median_validated_excl_intervention': v_keep[value].median() if len(v_keep) else np.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--results', required=True, help='folder with hr_windows.csv / rr_windows.csv')
    ap.add_argument('--rr-plus', help='RR+ result folder with rr_windows.csv (optional)')
    ap.add_argument('--reference-hr')
    ap.add_argument('--reference-rr')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    src = M2/'runs'/run.name/'caregiver'
    if not (src/'caregiver_segments.csv').is_file():
        raise SystemExit(f'Missing {src}/caregiver_segments.csv: run detect_caregiver_contact.py first')
    segs = pd.read_csv(src/'caregiver_segments.csv')
    frames = pd.read_csv(src/'caregiver_frames.csv', usecols=['video_id', 'timestamp_sec', 'adult_in_view'])
    res = Path(a.results).resolve()
    out = res/'caregiver'
    out.mkdir(parents=True, exist_ok=True)
    tables = [('hr', res/'hr_windows.csv', 'hr_bpm', a.reference_hr, 5.),
              ('rr', res/'rr_windows.csv', 'rr_brpm', a.reference_rr, 5.)]
    if a.rr_plus:
        tables.append(('rr_plus', Path(a.rr_plus).resolve()/'rr_windows.csv', 'rr_brpm', a.reference_rr, 5.))
    summaries = []
    for signal, path, value, reference, tol in tables:
        if not path.is_file():
            print(f'skip {signal}: no {path}')
            continue
        t = label_windows(pd.read_csv(path), segs, frames)
        t.to_csv(out/f'{signal}_windows_caregiver.csv', index=False)
        summaries.append(summarize(t, signal.upper().replace('_PLUS', '+'), value))
        if reference:
            ev = pd.concat([evaluate(t, reference, value, value, tol),
                            evaluate(t[~t.caregiver_intervention], reference, value, value, tol).assign(
                                subset=lambda d: d.subset + '_excl_intervention')], ignore_index=True)
            ev.to_csv(out/f'evaluation_{signal}_caregiver.csv', index=False)
            print(f'== evaluation {signal}\n{ev.round(2).to_string(index=False)}')
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(out/'caregiver_window_summary.csv', index=False)
    (out/'metadata.json').write_text(json.dumps(dict(run=str(run), caregiver=str(src), results=str(res), rr_plus=a.rr_plus,
                                                     window_rule=CFG['windows'], temporal=CFG['temporal']), indent=2))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(summary.round(1).to_string(index=False))
    print('->', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
