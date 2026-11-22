"""Internal-consistency checks that need no new perception run (label-free; video = unit).

C1 Term independence within each score: largest |r| among the terms that enter the SAME score (v1: P, V, S, C, M;
   v2: P, V', S, C, M_n) on training ROI-frames, and where M_n departs from M (respiratory regions; frames with
   strong global/camera motion).
C2 Score-selection agreement: quality retention per split for D1-D7.
C3 Cross-level agreement of the selected regions (fusion-weighted): share of region area inside the SAM 2 mask, share
   of the region's pose landmarks inside the mask, share of 2-point-box ("dead") selections, and refined-pixel yield
   (pixels kept by ROI n mask n adaptive skin / shrunk-polygon pixels) on every 10th sampled frame.
C4 Rate consistency: selection with every 2nd sampled frame (time constants kept) vs every frame, compared on the
   frames both process.
Usage: python module1_v2/scripts/run_consistency_checks.py --run runs/final34_paper
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for p in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(p))
import run_v2_evaluation as ev                                   # noqa: E402
from run_phase5 import strided                                   # noqa: E402
from module1 import paper_eval as pe                             # noqa: E402
from module2.ablation import bootstrap_ci                        # noqa: E402
from module2.extraction import mean_rgb, shrink                  # noqa: E402
from run_paper_evaluation import EXTRACT, load_mask              # noqa: E402
from module1_v2 import selection as sel                          # noqa: E402

CONFIGS = ('D0', 'I-B', 'D3', 'D3L', 'D5')
RESP = ('THORACO_ABDOMINAL', 'UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN')


def summarize(v, nd=3):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 'n/a'
    _, lo, hi = bootstrap_ci(v)
    return f'{v.mean():.{nd}f} ± {v.std(ddof=1) if len(v) > 1 else 0:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]'


def video_job(args):
    video, record, out, cand, frames0, sam2_rows, configs = args
    out = Path(out)
    frames = sorted(cand.frame_index.unique())
    fps = len(sorted(set(frames0) | set(frames))) / float(record['duration_sec'])
    info = cand.set_index(['frame_index', 'roi_name'])
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    sample = set(frames[::10])
    images = {}
    if sample:
        cap = cv2.VideoCapture(str(record['video_path']))
        for i in range(int(record['frame_count'])):
            ok, image = cap.read()
            if not ok:
                break
            if i in sample:
                images[i] = image
        cap.release()
    rows, rate = [], []
    for cid in CONFIGS:
        geometry, _, _, _ = sel.select_video(cand, frames, configs[cid], fps, 3, 1., None)
        inside, landmarks, dead, yields, weights = [], [], [], [], []
        for f, sel_rows in geometry.items():
            for name, w, polygon, _ in sel_rows:
                r = info.loc[(f, name)]
                weights.append(w)
                inside.append(r.roi_inside_mask_ratio)
                landmarks.append(r.landmark_inside_mask_ratio)
                pix = sel.pixel_polygon(polygon, configs[cid].get('box_roi_pixels', 'skip'))
                is_dead = pix is None
                dead.append(float(is_dead))
                if f in images and not is_dead:
                    image = images[f]
                    mask = load_mask(out/'sam2', sam2_rows.loc[f]) if len(sam2_rows) and f in sam2_rows.index else None
                    if mask is not None and mask.shape == image.shape[:2]:
                        image = image.copy()
                        image[~mask] = 0
                    poly = np.asarray(pix, float)
                    _, count = mean_rgb(image, poly, EXTRACT)
                    area = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillPoly(area, [np.rint(shrink(poly, EXTRACT.roi_shrink)).astype(np.int32)], 1)
                    yields.append(count / max(1, int(area.sum())))
        w = np.asarray(weights, float)
        wavg = lambda x: float(np.nansum(w * np.asarray(x, float)) / np.nansum(w * np.isfinite(np.asarray(x, float))))  # noqa: E731
        rows.append(dict(video_id=video, split=record['split'], config=cid, area_inside_mask=wavg(inside),
                         landmarks_inside_mask=wavg(landmarks), dead_selection_percent=100 * float(np.mean(dead)),
                         pixel_yield=float(np.mean(yields)) if yields else np.nan))
        # rate consistency: stride 2 vs stride 1 on the frames both process
        keep = frames[::2]
        g2, _, _, _ = sel.select_video(cand[cand.frame_index.isin(set(keep))], keep, strided(configs[cid], 2), fps / 2,
                                       3, 1., None)
        jac, top = [], []
        for f in keep:
            a = [r[0] for r in geometry.get(f) or []]
            b = [r[0] for r in g2.get(f) or []]
            if a or b:
                jac.append(len(set(a) & set(b)) / len(set(a) | set(b)))
                top.append(float(bool(a) and bool(b) and a[0] == b[0]))
        rate.append(dict(video_id=video, split=record['split'], config=cid, jaccard_s1_s2=float(np.mean(jac)) if jac else np.nan,
                         top_agreement_s1_s2=100 * float(np.mean(top)) if top else np.nan))
    return rows, rate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    target = V2/'reports'/run.name/'consistency'
    target.mkdir(parents=True, exist_ok=True)
    configs = ev.configurations(json.loads((work/'tuning_log.json').read_text())['chosen'])
    manifest = pd.read_csv(run/'manifest.csv')
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=ev.CAND_COLUMNS + ['roi_inside_mask_ratio', 'landmark_inside_mask_ratio'])
    feat = pd.read_csv(work/'features/roi_features_median.csv')
    cand = cand.merge(feat[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance', 'motion_total',
                            'motion_residual', 'respiratory_ratio', 'global_dx', 'global_dy']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')

    # C1 term independence (training ROI-frames)
    train = cand[(cand.split == 'train') & (cand.roi_status == 'AVAILABLE')]
    sets = dict(v1=['pose_term', 'visibility_term', 'skin_term', 'temporal_term', 'motion_term'],
                v2=['pose_term', 'visibility_keypoint', 'skin_term', 'temporal_term', 'motion_nuisance'])
    c1 = []
    for name, cols in sets.items():
        for method in ('pearson', 'spearman'):
            r = train[cols].corr(method=method).abs().where(~np.eye(len(cols), dtype=bool))
            pair = r.stack().idxmax()
            c1.append(dict(score=name, method=method, max_abs_r=float(r.max().max()), pair=f'{pair[0]} - {pair[1]}'))
    g = np.hypot(train.global_dx, train.global_dy)
    edges = np.nanpercentile(g, [100 / 3, 200 / 3])
    for label, part in (('all', train), ('respiratory regions', train[train.roi_name.isin(RESP)]),
                        ('other regions', train[~train.roi_name.isin(RESP)]),
                        ('low camera motion', train[g <= edges[0]]), ('high camera motion', train[g > edges[1]])):
        ok = part[['motion_term', 'motion_nuisance']].dropna()
        c1.append(dict(score='M vs M_n', method=label, max_abs_r=float(ok.motion_term.corr(ok.motion_nuisance)),
                       pair=f'mean |M - M_n| = {float((ok.motion_term - ok.motion_nuisance).abs().mean()):.3f}; n = {len(ok)}'))
    pd.DataFrame(c1).to_csv(target/'table_c1_term_independence.csv', index=False)

    # C2 retention per split
    pv = pd.read_csv(V2/'reports'/run.name/'phase1/per_video_metrics.csv')
    d = pv[pv.config.isin(['D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7'])]
    c2 = [dict(config=c, **{s: summarize(part[part.split == s].retention) for s in ('train', 'validation', 'test')},
               all=summarize(part.retention)) for c, part in d.groupby('config')]
    pd.DataFrame(c2).to_csv(target/'table_c2_retention.csv', index=False)

    # C3 + C4
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'], low_memory=False)
    sam = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = [(r.video_id, r._asdict(), str(out), cand[cand.video_id == r.video_id],
             frames0[frames0.video_id == r.video_id].frame_index.tolist(), sam[sam.video_id == r.video_id], configs)
            for r in manifest.itertuples()]
    with ProcessPoolExecutor(a.workers) as pool:
        res = list(pool.map(video_job, jobs))
    c3 = pd.DataFrame([x for r in res for x in r[0]])
    c4 = pd.DataFrame([x for r in res for x in r[1]])
    c3.to_csv(target/'c3_per_video.csv', index=False)
    c4.to_csv(target/'c4_per_video.csv', index=False)
    pd.DataFrame([dict(config=c, **{m: summarize(p[m], 3 if m != 'dead_selection_percent' else 1)
                                    for m in ('area_inside_mask', 'landmarks_inside_mask', 'dead_selection_percent', 'pixel_yield')})
                  for c, p in c3.groupby('config')]).to_csv(target/'table_c3_cross_level.csv', index=False)
    pd.DataFrame([dict(config=c, jaccard_s1_s2=summarize(p.jaccard_s1_s2), top_agreement_s1_s2=summarize(p.top_agreement_s1_s2, 1))
                  for c, p in c4.groupby('config')]).to_csv(target/'table_c4_rate_consistency.csv', index=False)
    for f in ('c1_term_independence', 'c2_retention', 'c3_cross_level', 'c4_rate_consistency'):
        print('##', f); print(pd.read_csv(target/f'table_{f}.csv').to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
