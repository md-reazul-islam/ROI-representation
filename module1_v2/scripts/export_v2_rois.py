"""Export a Module 1 v2 configuration in the input format Module 2 already reads (roi_confidence.csv), so
run_hr_rppg.py / run_rr.py can consume v2 through --confidence-dir without any change to Module 2.

Per candidate ROI-frame (same rows and columns as v1's confidence_sam2/roi_confidence.csv):
  q            <- the configuration's own region score (v1 rule for D3; Q' = 0.2(P+V'+S+C) - 0.2 M_n for D5/D6)
  roi_status   <- AVAILABLE only for the regions the v2 stabilized selection keeps at that frame, otherwise
                  NOT_SELECTED_V2 (Module 2 uses AVAILABLE rows only), so the stabilization reaches Module 2
  added        q_v1, roi_status_v1 (originals), v2_selected, v2_rank, v2_weight (stabilized fusion weight)
Polygons, bbox (Module 2 reads 2-point boxes as rectangles via bbox), skin and motion columns are unchanged.
Usage: python module1_v2/scripts/export_v2_rois.py --run runs/final34_paper --config D5
Output: module1_v2/runs/<run>/module2_input_<config>/roi_confidence.csv
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
from module1_v2 import selection as sel              # noqa: E402

FEATURES = ['visibility_keypoint', 'motion_nuisance', 'motion_total', 'motion_residual', 'respiratory_ratio']


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', default='D5')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    flags = ev.configurations(json.loads((work/'tuning_log.json').read_text())['chosen'])[a.config]
    target = work/f'module2_input_{a.config}'
    target.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False)
    feat = pd.read_csv(work/'features/roi_features_median.csv', usecols=['video_id', 'frame_index', 'roi_name'] + FEATURES)
    table = table.merge(feat, on=['video_id', 'frame_index', 'roi_name'], how='left')
    for col in ('pose_term', 'visibility_term', 'skin_term', 'temporal_term', 'motion_term'):
        if col not in table:
            raise KeyError(col)
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'], low_memory=False)
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    table['q_v1'] = table['q']
    table['roi_status_v1'] = table['roi_status']
    table['q'] = sel.score_column(table, flags.get('visibility', 'v1'), flags.get('motion', 'total'))
    table['v2_selected'], table['v2_rank'], table['v2_weight'] = False, -1, 0.
    index = {k: i for i, k in enumerate(zip(table.video_id, table.frame_index, table.roi_name))}
    selected, rank, weight = table.v2_selected.to_numpy(), table.v2_rank.to_numpy(), table.v2_weight.to_numpy(float)
    for video, cand in table.groupby('video_id'):
        frames = sorted(cand.frame_index.unique())
        fps = len(set(frames0[frames0.video_id == video].frame_index) | set(frames)) / float(manifest.loc[video, 'duration_sec'])
        geometry, fusion, _, _ = sel.select_video(cand.assign(q=cand.q_v1), frames, flags, fps, 3, 1., None)
        for f, rows in geometry.items():
            for r, (name, _, _, _) in enumerate(rows):
                i = index[(video, f, name)]
                selected[i], rank[i] = True, r
        for f, rows in fusion.items():
            for name, w, _, _ in rows:
                weight[index[(video, f, name)]] = w
    table['v2_selected'], table['v2_rank'], table['v2_weight'] = selected, rank, weight
    table['roi_status'] = table.roi_status_v1.where(table.v2_selected, 'NOT_SELECTED_V2')
    table.to_csv(target/'roi_confidence.csv', index=False)
    (target/'export_metadata.json').write_text(json.dumps(dict(
        run=str(run), config=a.config, flags=flags, rows=len(table), selected_rows=int(table.v2_selected.sum()),
        note='q = configuration score; roi_status AVAILABLE only where the v2 stabilized selection keeps the region'),
        indent=2, default=str))
    print(target/'roi_confidence.csv', len(table), 'rows,', int(table.v2_selected.sum()), 'selected')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
