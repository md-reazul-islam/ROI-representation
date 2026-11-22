"""Phase 1b/1c/1d features for a finished v1 run (SAM 2 branch): keypoint visibility V', decomposed motion
(global / residual / respiratory periodicity / nuisance) per ROI-frame, and infant-vs-adult subject status
per frame. Reads v1 outputs read-only; writes module1_v2/runs/<run>/features/.

Usage (from the Module1 folder):
    python module1_v2/scripts/run_v2_features.py --run runs/dataset5_paper [--global-motion median|homography]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for path in (V2/'src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(path))

from module1 import paper_eval as pe                         # noqa: E402  (v1, read-only)
from module1.roi.anatomical import AnatomicalROIMapper         # noqa: E402
from module1_v2 import motion as mo, subject as sj, terms      # noqa: E402
from run_paper_evaluation import boxes_of, load_mask           # noqa: E402  (v1 helpers)

CFG = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())['phase1']


def video_features(job):
    video, record, out, cand, recovery, sam2_rows, pose_rows, names, global_mode = job
    out = Path(out)
    started = time.perf_counter()
    mapper = AnatomicalROIMapper(names, CFG['keypoint_confidence'])
    parts = sj.part_indices(names)
    boxes = boxes_of(recovery)
    tracks = dict(zip(recovery.frame_index.astype(int), recovery.track_id))
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    pose_rows = pose_rows.set_index('frame_index') if len(pose_rows) else pose_rows
    cand = cand[cand.roi_status == 'AVAILABLE']
    by_frame = {int(f): rows for f, rows in cand.groupby('frame_index')}
    frames = sorted(set(int(f) for f in recovery.frame_index))
    wanted = set(frames)
    fps = float(record['fps'])
    capture = cv2.VideoCapture(str(record['video_path']))
    roi_rows, subject_rows = [], []
    previous_gray, index = None, 0
    last = max(frames) if frames else -1
    while index <= last:
        ok, image = capture.read()
        if not ok:
            break
        frame, index = index, index + 1
        if frame not in wanted:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = load_mask(out/'sam2', sam2_rows.loc[frame]) if len(sam2_rows) and frame in sam2_rows.index else None
        points = {}
        if len(pose_rows) and frame in pose_rows.index and isinstance(pose_rows.loc[frame, 'keypoints_path'], str):
            payload = json.loads((out/'pose_sam2'/pose_rows.loc[frame, 'keypoints_path']).read_text())
            points = terms.keypoint_table(payload.get('keypoints') or [])
        r1, r2 = sj.proportions(points, parts, CFG['keypoint_confidence']) if points else (np.nan, np.nan)
        subject_rows.append(dict(video_id=video, frame_index=frame, time_sec=frame / fps, track_id=tracks.get(frame),
                                 head_shoulder_ratio=r1, arm_torso_ratio=r2,
                                 frame_decision=sj.frame_decision(r1, r2, CFG['head_shoulder_ratio_min'],
                                                                  CFG['arm_torso_ratio_max'])))
        flow = residual = None
        status, global_vector = 'FIRST_FRAME', (np.nan, np.nan)
        if previous_gray is not None:
            flow = mo.dense_flow(previous_gray, gray)
            background = mo.background_mask(gray.shape, mask, boxes.get(frame), CFG['background_dilation_px'])
            field, status = mo.global_flow(flow, background, global_mode)
            residual = flow - field
            global_vector = tuple(float(v) for v in field.reshape(-1, 2).mean(0))
        for row in by_frame.get(frame, pd.DataFrame()).itertuples():
            polygon = pe.parse(row.polygon)
            visibility = terms.keypoint_visibility(mapper, row.roi_name, points, CFG['keypoint_confidence']) if points else np.nan
            total = res = np.nan
            vector = (np.nan, np.nan)
            if flow is not None and polygon and len(polygon) >= 3:
                total, res, vector = mo.region_motion(flow, residual, mo.polygon_mask(gray.shape, polygon))
            roi_rows.append(dict(video_id=video, frame_index=frame, time_sec=frame / fps, roi_name=row.roi_name,
                                 visibility_keypoint=visibility, motion_total_px=total, motion_residual_px=res,
                                 residual_dx=vector[0], residual_dy=vector[1], global_status=status,
                                 global_dx=global_vector[0], global_dy=global_vector[1]))
        previous_gray = gray
    capture.release()

    rois = pd.DataFrame(roi_rows)
    if len(rois):
        scale = CFG['motion_scale_pixels']
        rois['motion_total'] = [mo.bounded(v, scale) for v in rois.motion_total_px]
        rois['motion_residual'] = [mo.bounded(v, scale) for v in rois.motion_residual_px]
        fs_sampled = len(frames) / float(record['duration_sec']) if float(record['duration_sec']) > 0 else fps
        window = CFG['periodicity_window_sec']
        ratios = np.full(len(rois), np.nan)
        for roi, part in rois.groupby('roi_name'):
            times, vectors = part.time_sec.to_numpy(), part[['residual_dx', 'residual_dy']].to_numpy()
            for j, (position, t) in enumerate(zip(part.index, times)):
                keep = (times > t - window) & (times <= t)
                ratios[position] = mo.periodic_ratio(vectors[keep], fs_sampled, tuple(CFG['respiratory_band_hz']),
                                                     min_samples=max(8, int(.5 * window * fs_sampled)))
        rois['respiratory_ratio'] = ratios
        respiratory = set(CFG['respiratory_rois'])
        rois['motion_nuisance'] = [mo.nuisance(m, r, roi in respiratory)
                                   for m, r, roi in zip(rois.motion_residual, rois.respiratory_ratio, rois.roi_name)]
    subjects = pd.DataFrame(subject_rows)
    if len(subjects):
        subjects['subject_status'] = sj.track_decisions(subjects.frame_decision, subjects.track_id, subjects.time_sec,
                                                        CFG['subject_window_sec'])
    return rois, subjects, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 v2 Phase 1 features (visibility, motion, subject).')
    parser.add_argument('--run', required=True, help='finished v1 run, e.g. runs/dataset5_paper')
    parser.add_argument('--global-motion', default=CFG['global_motion'], choices=('median', 'homography'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-videos', type=int)
    args = parser.parse_args(argv)
    run = (ROOT/args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    out = run/'outputs'
    target = V2/'runs'/run.name/'features'
    target.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run/'manifest.csv')
    if args.max_videos:
        manifest = manifest.head(args.max_videos)
    names = json.loads((out/'pose_sam2/experiment_metadata.json').read_text())['keypoint_names']
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon'])
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    pose = pd.read_csv(out/'pose_sam2/pose_results.csv', low_memory=False, usecols=['video_id', 'frame_index', 'keypoints_path'])
    jobs = [(r.video_id, r._asdict(), str(out), cand[cand.video_id == r.video_id], recovery[recovery.video_id == r.video_id],
             sam2[sam2.video_id == r.video_id], pose[pose.video_id == r.video_id], names, args.global_motion)
            for r in manifest.itertuples()]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(video_features, jobs))
    rois = pd.concat([r[0] for r in results], ignore_index=True)
    subjects = pd.concat([r[1] for r in results], ignore_index=True)
    rois.to_csv(target/f'roi_features_{args.global_motion}.csv', index=False)
    subjects.to_csv(target/'subject_frames.csv', index=False)
    (target/f'features_metadata_{args.global_motion}.json').write_text(json.dumps(dict(
        run=str(run), global_motion=args.global_motion, videos=len(jobs), roi_frames=len(rois),
        wall_time_sec=time.perf_counter() - started, per_video_sec={j[0]: r[2] for j, r in zip(jobs, results)},
        config=CFG), indent=2, default=str))
    print(f'Wrote {target} ({len(rois):,} ROI-frames, {len(subjects):,} frames) in {time.perf_counter() - started:.0f} s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
