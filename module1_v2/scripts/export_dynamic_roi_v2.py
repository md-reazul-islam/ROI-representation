"""Module 1 v2 output: the dynamic ROI representation of one run, as a table Module 2 v2 reads.

One row per candidate ROI and sampled frame (all candidates, so downstream RR can use chest/abdomen regions even
when the HR selection is on the face):
  video_id, split, frame_index, timestamp_sec, roi_name, available, pixel_polygon (2-point boxes as rectangles),
  q_v2 (score of the configuration), q_v1, selected, rank, weight (stabilized fusion weight; 0 if not selected),
  visible_skin_proportion, motion_magnitude (v1), visibility_keypoint (V'), motion_nuisance (M_n),
  respiratory_ratio, residual_dx, residual_dy (camera-compensated ROI velocity, px per sampled frame),
  global_dx, global_dy (camera motion), infant_box
Stabilizer settings come from a tuning log (default: this run's; for new videos pass the final34 log, so nothing is
re-tuned on the videos being analysed).
Usage: python module1_v2/scripts/export_dynamic_roi_v2.py --run runs/final34_paper --config D5
       [--tuning-log module1_v2/runs/final34_paper/tuning_log.json]
Output: module1_v2/runs/<run>/dynamic_roi_v2_<config>/dynamic_roi_v2.csv (+ metadata.json)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
import run_v2_evaluation as ev                       # noqa: E402
from module1 import paper_eval as pe                 # noqa: E402
from run_paper_evaluation import boxes_of            # noqa: E402
from module1_v2 import selection as sel              # noqa: E402

FEATURES = ['visibility_keypoint', 'motion_nuisance', 'motion_total', 'motion_residual', 'respiratory_ratio',
            'residual_dx', 'residual_dy', 'global_dx', 'global_dy']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', default='D5')
    ap.add_argument('--tuning-log', help='tuning_log.json with the frozen stabilizer settings (default: this run)')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    log_path = Path(a.tuning_log).resolve() if a.tuning_log else work/'tuning_log.json'
    flags = ev.configurations(json.loads(log_path.read_text())['chosen'])[a.config]
    box_mode = flags.get('box_roi_pixels', 'skip')
    target = work/f'dynamic_roi_v2_{a.config}'
    target.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=ev.CAND_COLUMNS + ['timestamp_sec', 'visible_skin_proportion', 'motion_magnitude'])
    feat = pd.read_csv(work/'features/roi_features_median.csv', usecols=['video_id', 'frame_index', 'roi_name'] + FEATURES)
    cand = cand.merge(feat, on=['video_id', 'frame_index', 'roi_name'], how='left')
    base = out/'confidence/roi_confidence.csv'             # absent when only the SAM 2 branch was run (new videos)
    frames0 = pd.read_csv(base if base.is_file() else out/'confidence_sam2/roi_confidence.csv',
                          usecols=['video_id', 'frame_index'], low_memory=False)
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    parts = []
    for video, c in cand.groupby('video_id'):
        frames = sorted(c.frame_index.unique())
        fps = len(set(frames0[frames0.video_id == video].frame_index) | set(frames)) / float(manifest.loc[video, 'duration_sec'])
        geometry, fusion, _, _ = sel.select_video(c, frames, flags, fps, 3, 1., None)
        rank = {(f, r[0]): i for f, rows in geometry.items() for i, r in enumerate(rows)}
        weight = {(f, r[0]): r[1] for f, rows in fusion.items() for r in rows}
        boxes = boxes_of(recovery[recovery.video_id == video])
        c = c.copy()
        c['q_v1'] = c.q
        c['q_v2'] = sel.score_column(c, flags.get('visibility', 'v1'), flags.get('motion', 'total'))
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
    table = pd.concat(parts, ignore_index=True)
    columns = ['video_id', 'split', 'frame_index', 'timestamp_sec', 'fps_sampled', 'roi_name', 'available', 'pixel_polygon',
               'q_v2', 'q_v1', 'selected', 'rank', 'weight', 'visible_skin_proportion', 'motion_magnitude'] + FEATURES + ['infant_box']
    table[columns].to_csv(target/'dynamic_roi_v2.csv', index=False)
    (target/'metadata.json').write_text(json.dumps(dict(
        run=str(run), config=a.config, flags=flags, tuning_log=str(log_path), rows=len(table),
        selected_rows=int(table.selected.sum()), videos=int(table.video_id.nunique())), indent=2, default=str))
    print(target/'dynamic_roi_v2.csv', len(table), 'rows,', int(table.selected.sum()), 'selected')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
