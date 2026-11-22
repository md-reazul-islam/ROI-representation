"""Module 1 v2 configuration D8 = D5 + occlusion / coverage rules (../PREREGISTRATION.md).

    python module1_v2_d8/scripts/export_d8.py --run runs/ubfc_aug --tuning-log module1_v2/runs/final34_paper/tuning_log.json

Same table format as module1_v2/scripts/export_dynamic_roi_v2.py (which is imported, not changed), so the unchanged HR
code reads it with --module1-config D8. Steps per video:
  1. D5 scores and flags (frozen final34 stabilizer), exactly as the D5 export;
  2. R1/R2 flags per region (colour features from colour_features.py); flagged regions -> score unavailable;
  3. the D5 selection / weights on the rule-filtered scores (identical code path to D5);
  4. R3: in affected frames the fusion set = all clean HR-group regions, weighted by visible skin area.
Outputs (all new): module1_v2_d8/runs/<run>/dynamic_roi_v2_D8/{dynamic_roi_v2.csv, metadata.json, d8_flags.csv} and a
symlink module1_v2/runs/<run>/dynamic_roi_v2_D8 -> that folder (the comparison tool looks there).
--no-rules: write the same table with the rules switched off (regression test: must equal D5).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D8 = Path(__file__).resolve().parents[1]
ROOT = D8.parent
V2 = ROOT/'module1_v2'
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts', D8):
    sys.path.insert(0, str(p))
import export_dynamic_roi_v2 as ex                   # noqa: E402  (read only: FEATURES)
import run_v2_evaluation as ev                       # noqa: E402
from module1 import paper_eval as pe                 # noqa: E402
from module1_v2 import selection as sel              # noqa: E402
from module1_v2 import stabilization as st           # noqa: E402
from run_paper_evaluation import boxes_of            # noqa: E402
from d8 import rules                                 # noqa: E402

NAMES = sel.NAMES


def shoelace(poly):
    p = np.asarray(poly, float)
    if p.ndim != 2 or len(p) < 3:
        return np.nan
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def select_masked(c, frames, flags, fps_sampled, exclude, k=3, temperature=1.):
    """module1_v2.selection.select_video with an extra per-region exclusion mask (same code otherwise)."""
    values = sel.score_column(c, flags.get('visibility', 'v1'), flags.get('motion', 'total'))
    Q, polygons = sel.score_matrix(c, frames, values)
    Q[exclude] = np.nan
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
    return sets, W


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tuning-log', required=True, help='frozen stabilizer (final34) tuning_log.json')
    ap.add_argument('--no-rules', action='store_true', help='regression test: rules off (output must equal D5)')
    ap.add_argument('--name', default='D8')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    flags = ev.configurations(json.loads(Path(a.tuning_log).resolve().read_text())['chosen'])['D5']
    box_mode = flags.get('box_roi_pixels', 'skip')
    target = D8/'runs'/run.name/f'dynamic_roi_v2_{a.name}'
    target.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=ev.CAND_COLUMNS + ['timestamp_sec', 'visible_skin_proportion', 'motion_magnitude'])
    feat = pd.read_csv(work/'features/roi_features_median.csv', usecols=['video_id', 'frame_index', 'roi_name'] + ex.FEATURES)
    cand = cand.merge(feat, on=['video_id', 'frame_index', 'roi_name'], how='left')
    base = out/'confidence/roi_confidence.csv'
    frames0 = pd.read_csv(base if base.is_file() else out/'confidence_sam2/roi_confidence.csv',
                          usecols=['video_id', 'frame_index'], low_memory=False)
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    colour = None if a.no_rules else pd.read_csv(D8/'runs'/run.name/'colour_features.csv')
    col = {n: j for j, n in enumerate(NAMES)}
    parts, flag_rows, summary = [], [], []
    for video, c in cand.groupby('video_id'):
        frames = sorted(c.frame_index.unique())
        row_of = {f: i for i, f in enumerate(frames)}
        T, R = len(frames), len(NAMES)
        fps = len(set(frames0[frames0.video_id == video].frame_index) | set(frames)) / float(manifest.loc[video, 'duration_sec'])
        values = sel.score_column(c, flags.get('visibility', 'v1'), flags.get('motion', 'total'))
        Q0, polygons = sel.score_matrix(c, frames, values)
        A = np.isfinite(Q0)
        s = np.full((T, R), np.nan)
        area = np.full((T, R), np.nan)
        pix = {}
        for f, r, sk, p in zip(c.frame_index, c.roi_name, c.visible_skin_proportion, c.polygon):
            if r not in col:
                continue
            t, j = row_of[int(f)], col[r]
            s[t, j] = sk
            if isinstance(p, str):
                pp = sel.pixel_polygon(pe.parse(p), box_mode)
                pix[(int(f), r)] = pp
                area[t, j] = shoelace(pp) if pp else np.nan
        if a.no_rules:
            covered = occluded = np.zeros((T, R), bool)
            excess = np.full((T, R), np.nan)
            affected, WR = np.zeros(T, bool), np.zeros((T, R))
        else:
            C = np.full((T, R, 3), np.nan)
            g = colour[colour.video_id == video]
            for f, r, rn, gn, lum in zip(g.frame_index, g.roi_name, g.rn, g.gn, g.lum):
                if int(f) in row_of and r in col:
                    C[row_of[int(f)], col[r]] = (rn, gn, lum)
            covered, occluded, excess = rules.flag_regions(s, A, C, fps)
            affected, WR = rules.redundancy(covered | occluded, A, s, area, NAMES, fps)
        sets, W = select_masked(c, frames, flags, fps, covered | occluded)
        rank, weight = {}, {}
        for t, f in enumerate(frames):
            if affected[t]:
                order = [j for j in np.argsort(-WR[t]) if WR[t, j] > 0]
                for i, j in enumerate(order):
                    rank[(f, NAMES[j])], weight[(f, NAMES[j])] = i, float(WR[t, j])
            else:
                members = sets[t]
                for i, j in enumerate(members):
                    rank[(f, NAMES[j])] = i
                for j in np.flatnonzero(W[t] > 0):
                    weight[(f, NAMES[j])] = float(W[t, j])
        boxes = boxes_of(recovery[recovery.video_id == video])
        c = c.copy()
        c['q_v1'] = c.q
        c['q_v2'] = values
        keys = list(zip(c.frame_index, c.roi_name))
        c['selected'] = [k in rank for k in keys]
        c['rank'] = [rank.get(k, -1) for k in keys]
        c['weight'] = [weight.get(k, 0.) for k in keys]
        c['available'] = c.roi_status.eq('AVAILABLE')
        c['pixel_polygon'] = [json.dumps(sel.pixel_polygon(pe.parse(p), box_mode)) if isinstance(p, str) else 'null'
                              for p in c.polygon]
        c['infant_box'] = [json.dumps(boxes.get(int(f))) for f in c.frame_index]
        c['fps_sampled'] = fps
        parts.append(c)
        for t, f in enumerate(frames):
            for j in np.flatnonzero(A[t]):
                flag_rows.append((video, f, NAMES[j], bool(covered[t, j]), bool(occluded[t, j]),
                                  None if not np.isfinite(excess[t, j]) else round(float(excess[t, j]), 3), bool(affected[t])))
        summary.append(dict(video_id=video, frames=T, affected_pct=round(100 * affected.mean(), 1),
                            covered_region_frames=int(covered.sum()), occluded_region_frames=int(occluded.sum())))
    table = pd.concat(parts, ignore_index=True)
    columns = ['video_id', 'split', 'frame_index', 'timestamp_sec', 'fps_sampled', 'roi_name', 'available', 'pixel_polygon',
               'q_v2', 'q_v1', 'selected', 'rank', 'weight', 'visible_skin_proportion', 'motion_magnitude'] + ex.FEATURES + ['infant_box']
    table[columns].to_csv(target/'dynamic_roi_v2.csv', index=False)
    pd.DataFrame(flag_rows, columns=['video_id', 'frame_index', 'roi_name', 'covered_R1', 'occluded_R2', 'excess_R2',
                                     'affected_R3']).to_csv(target/'d8_flags.csv', index=False)
    summ = pd.DataFrame(summary)
    (target/'metadata.json').write_text(json.dumps(dict(
        run=str(run), config=a.name, base_config='D5', d5_flags=flags, rules='off' if a.no_rules else 'R1+R2+R3',
        preregistration=str(D8/'PREREGISTRATION.md'), tuning_log=str(Path(a.tuning_log).resolve()), rows=len(table),
        selected_rows=int(table.selected.sum()), per_video=summary), indent=2, default=str))
    link = work/f'dynamic_roi_v2_{a.name}'
    if not a.no_rules and not link.exists():           # the regression test stays inside module1_v2_d8/
        link.symlink_to(target, target_is_directory=True)
    print(summ.to_string(index=False))
    print(target/'dynamic_roi_v2.csv', len(table), 'rows,', int(table.selected.sum()), 'selected')


if __name__ == '__main__':
    main()
