"""Module 1 v2 configuration D9 = D5 + subject-mask repair for the face (+ optional ROI-geometry stabilisation).

    python module1_v2_d9/scripts/export_d9.py --run runs/ubfc_swaddle [--stabilise] [--name D9]

Why (development video UBFC subject1 swaddle-large, cloth up to the mouth): the person detector then sees only the
head and SAM 2 segments the HAIR, not the face. The HR extraction keeps only pixels inside the SAM 2 mask, so the face
ROIs D5 selects lose (almost) all pixels and the pulse is lost (no HR in 37 / 46 windows), while the fixed box
(which ignores the mask) still works. The face landmarks also fail (keypoint visibility < 0.5), so the landmark ROIs
jitter 10-30 px per frame.

Rules (constants fixed in ../PREREGISTRATION.md):
  R1 mask repair: per sampled frame, coverage = share of the face-skin ROI pixels (union of FOREHEAD, UPPER_FACE,
     LEFT_CHEEK, RIGHT_CHEEK polygons) inside the SAM 2 mask. If coverage < 0.5, the union is added to the mask.
     The extraction's own adaptive skin filter (YCrCb) still decides which of those pixels are used.
  R2 (only with --stabilise) geometry: in frames where the mean keypoint visibility of the face group is < 0.5, the
     face-group ROIs and HEAD get a temporal majority-vote shape (pixel inside in >= 50 % of the frames within +-1 s).
  Selection, ranks and weights are D5's, unchanged (read from the D5 export of the same run).

Outputs (all new, nothing existing is changed):
  module1_v2_d9/runs/<run>/<name>/{dynamic_roi_v2.csv, d9_frames.csv, metadata.json, masks/*.png}
  runs/<run>_d9/  mirror of the Module 1 run: every item is a symlink to the original, except outputs/sam2, which is
                  a new folder whose sam2_masks.csv points to the repaired masks (changed frames) or to the original
                  masks (via the symlink 'orig').
  module1_v2/runs/<run>_d9/  symlinks to the original v2 run items + dynamic_roi_v2_<name> -> the D9 table.
The unchanged HR comparison tool then runs on --run runs/<run>_d9 --module1-config <name>. The fixed box ignores the
mask, so its results in the mirror run equal the original run's.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

D9 = Path(__file__).resolve().parents[1]
ROOT = D9.parent
FACE_GROUP = ('FOREHEAD', 'UPPER_FACE', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'FACE')
FACE_SKIN = ('FOREHEAD', 'UPPER_FACE', 'LEFT_CHEEK', 'RIGHT_CHEEK')
STABILISED = FACE_GROUP + ('HEAD',)
MASK_COVERAGE_MIN = 0.5
V_UNRELIABLE, HALF_WIN_SEC, VOTE, SCALE = 0.5, 1.0, 0.5, 2


def poly_of(p):
    if not isinstance(p, str) or p == 'null':
        return None
    q = json.loads(p)
    return q if q and len(q) >= 3 else None


def stabilise(polys, frames, unreliable, fps, W, H):
    w, h = W // SCALE, H // SCALE
    fr = np.array(frames)
    masks = {}
    for f in frames:
        p = polys.get(f)
        if p:
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.round(np.asarray(p, float) / SCALE).astype(np.int32)], 1)
            masks[f] = m
    out = {}
    for i, f in enumerate(frames):
        if not unreliable[i] or f not in masks:
            continue
        near = [g for g in fr[np.abs(fr - f) <= HALF_WIN_SEC * fps] if g in masks]
        if len(near) < 3:
            continue
        m = (np.sum([masks[g] for g in near], axis=0, dtype=np.int32) >= VOTE * len(near)).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs and cv2.contourArea(c := max(cs, key=cv2.contourArea)) >= 4:
            out[f] = (c.reshape(-1, 2).astype(float) * SCALE + SCALE / 2 - .5).round(1).tolist()
    return out


def mirror(src, dst, skip=()):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip or (dst/item.name).exists() or (dst/item.name).is_symlink():
            continue
        (dst/item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--stabilise', action='store_true', help='also apply R2 (geometry stabilisation)')
    ap.add_argument('--name', default='D9')
    a = ap.parse_args(argv)
    run = (ROOT/a.run).resolve()
    v2 = ROOT/'module1_v2/runs'/run.name
    t = pd.read_csv(v2/'dynamic_roi_v2_D5'/'dynamic_roi_v2.csv', low_memory=False)
    vis = pd.read_csv(v2/'features/roi_features_median.csv', usecols=['video_id', 'frame_index', 'roi_name', 'visibility_keypoint'])
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    sam = pd.read_csv(run/'outputs/sam2/sam2_masks.csv', low_memory=False)
    target = D9/'runs'/run.name/a.name
    (target/'masks').mkdir(parents=True, exist_ok=True)
    new_poly = t.pixel_polygon.copy()
    frame_rows, summary = [], []
    sam['mask_path_new'] = np.where(sam.mask_path.apply(lambda x: isinstance(x, str)), 'orig/' + sam.mask_path.astype(str), None)
    for video, g in t.groupby('video_id'):
        frames = sorted(g.frame_index.unique())
        fps = float(g.fps_sampled.iloc[0])
        W, H = int(manifest.loc[video, 'width']), int(manifest.loc[video, 'height'])
        v = vis[(vis.video_id == video) & vis.roi_name.isin(FACE_GROUP)]
        rel = v.groupby('frame_index').visibility_keypoint.mean().reindex(frames).fillna(0.).to_numpy()
        unreliable = rel < V_UNRELIABLE
        stab = 0
        if a.stabilise:
            for r in STABILISED:
                rows = g[(g.roi_name == r) & g.available]
                st = stabilise({int(f): poly_of(p) for f, p in zip(rows.frame_index, rows.pixel_polygon)},
                               frames, unreliable, fps, W, H)
                idx = rows.index[rows.frame_index.isin(list(st))]
                new_poly.loc[idx] = [json.dumps(st[int(f)]) for f in t.loc[idx, 'frame_index']]
                stab += len(idx)
        # R1 mask repair on the final geometry
        s = sam[sam.video_id == video].set_index('frame_index')
        polys = {f: [poly_of(new_poly.loc[i]) for i in rows.index]
                 for f, rows in g[g.roi_name.isin(FACE_SKIN) & g.available].groupby('frame_index')}
        repaired = 0
        for i, f in enumerate(frames):
            cov = np.nan
            if f in s.index and f in polys and isinstance(s.loc[f, 'mask_path'], str):
                m = cv2.imread(str(run/'outputs/sam2'/s.loc[f, 'mask_path']), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    u = np.zeros(m.shape, np.uint8)
                    for p in polys[f]:
                        if p:
                            cv2.fillPoly(u, [np.round(np.asarray(p, float)).astype(np.int32)], 1)
                    if u.any():
                        cov = float((m[u > 0] > 0).mean())
                        if cov < MASK_COVERAGE_MIN:
                            m = np.where(u > 0, 255, m).astype(np.uint8)
                            name = f'masks/{video}_frame_{f:08d}.png'
                            cv2.imwrite(str(target/name), m)
                            sam.loc[(sam.video_id == video) & (sam.frame_index == f), 'mask_path_new'] = 'repaired/' + name
                            repaired += 1
            frame_rows.append((video, f, round(float(rel[i]), 3), bool(unreliable[i]),
                               None if not np.isfinite(cov) else round(cov, 3)))
        summary.append(dict(video_id=video, frames=len(frames), landmarks_unreliable_pct=round(100 * unreliable.mean(), 1),
                            region_frames_stabilised=stab, mask_repaired_pct=round(100 * repaired / len(frames), 1)))
    t['pixel_polygon'] = new_poly
    t.to_csv(target/'dynamic_roi_v2.csv', index=False)
    pd.DataFrame(frame_rows, columns=['video_id', 'frame_index', 'face_keypoint_visibility', 'landmarks_unreliable',
                                      'sam2_face_coverage']).to_csv(target/'d9_frames.csv', index=False)
    (target/'metadata.json').write_text(json.dumps(dict(
        run=str(run), config=a.name, base='D5 export of the same run (selection and weights unchanged)',
        rules=dict(R1_mask_repair=dict(face_skin=FACE_SKIN, coverage_min=MASK_COVERAGE_MIN),
                   R2_stabilise=dict(on=a.stabilise, face_group=FACE_GROUP, v_unreliable=V_UNRELIABLE,
                                     half_window_sec=HALF_WIN_SEC, vote=VOTE)),
        preregistration=str(D9/'PREREGISTRATION.md'), per_video=summary), indent=2))
    # mirror run with repaired masks
    mrun = ROOT/'runs'/f'{run.name}_d9'
    mirror(run, mrun, skip=('outputs',))
    mirror(run/'outputs', mrun/'outputs', skip=('sam2',))
    sdir = mrun/'outputs/sam2'
    sdir.mkdir(parents=True, exist_ok=True)
    for link, dest in (('orig', run/'outputs/sam2'), ('repaired', target)):
        if (sdir/link).is_symlink():
            os.remove(sdir/link)
        (sdir/link).symlink_to(dest, target_is_directory=True)
    out = sam.drop(columns='mask_path').rename(columns={'mask_path_new': 'mask_path'})[sam.columns.drop('mask_path_new')]
    out.to_csv(sdir/'sam2_masks.csv', index=False)
    mirror(v2, ROOT/'module1_v2/runs'/mrun.name, skip=(f'dynamic_roi_v2_{a.name}',))
    link = ROOT/'module1_v2/runs'/mrun.name/f'dynamic_roi_v2_{a.name}'
    if link.is_symlink():
        os.remove(link)
    link.symlink_to(target, target_is_directory=True)
    print(pd.DataFrame(summary).to_string(index=False))
    print('mirror run:', mrun)


if __name__ == '__main__':
    main()
