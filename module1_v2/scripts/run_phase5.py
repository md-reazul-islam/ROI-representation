"""Phase 5: efficiency.

1. Stage profile: measured wall time per pipeline stage (runs/<run>/phase_timings.json), per sampled frame,
   and peak GPU memory (runs/<run>/gpu_monitor.csv); plus the v2 add-on stages (features, evaluation).
2. Frame-stride variants: the v2 selection and pulse extraction are re-run with only every s-th sampled frame
   (s = 1..4) carrying detection / pose / ROI / quality; ROI geometry is held (zero-order) in between. The
   time constants of the tuned stabilizer are kept (EMA beta -> beta^s, hysteresis frames N -> round(N / s)).
   Accuracy is MEASURED on the real videos; run time is PROJECTED by scaling the measured per-frame stages
   by 1/s (the stages were not re-run).
3. Optical flow: measured time of the Farneback dense flow used by the v2 motion features vs pyramidal
   Lucas-Kanade on sparse corners, on real frame pairs.
A lighter pose model was not evaluated: no checkpoint other than Sapiens2-0.4B is available.

Usage: python module1_v2/scripts/run_phase5.py --run runs/final34_paper --v1-report reports/paper_final34_paper
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for path in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(path))

import run_v2_evaluation as ev                                               # noqa: E402
from module2.ablation import holm, paired_test                               # noqa: E402
from module1_v2.motion import dense_flow                                     # noqa: E402

PER_FRAME = ('yolo', 'rtdetr', 'sam2', 'pose', 'pose_sam2', 'roi', 'roi_sam2', 'roi_quality', 'roi_quality_sam2',
             'confidence', 'confidence_sam2')
STRIDES = (1, 2, 3, 4)
ORIGINAL_EXTRACT = ev.EXTRACT


def strided(flags, s):
    out = dict(flags)
    if out.get('ema_beta'):
        out['ema_beta'] = float(out['ema_beta']) ** s
    if out.get('hysteresis_frames'):
        out['hysteresis_frames'] = max(1, int(round(out['hysteresis_frames'] / s)))
    return out


def stride_job(job):
    stride, base_job = job
    video, record, out, cand, frames, recovery, sam2_rows, subjects, configs = base_job
    keep = frames[::stride]
    cand = cand[cand.frame_index.isin(set(keep))]
    fps_sampled = len(frames) / float(record['duration_sec'])
    # hold geometry until the next processed frame (zero-order propagation)
    ev.EXTRACT = ORIGINAL_EXTRACT if stride == 1 else dataclasses.replace(
        ORIGINAL_EXTRACT, max_hold_sec=max(ORIGINAL_EXTRACT.max_hold_sec, (stride + .5) / fps_sampled))
    configs = {c: strided(f, stride) for c, f in configs.items()}
    per_video, windows, _, seconds = ev.evaluate_video((video, record, out, cand, keep, recovery, sam2_rows,
                                                        subjects.iloc[0:0], configs))
    for row in per_video:
        row['stride'] = stride
    for row in windows:
        row['stride'] = stride
    return per_video, windows, seconds


def flow_timing(manifest, pairs=60):
    rows = []
    for r in manifest.itertuples():
        capture = cv2.VideoCapture(r.video_path)
        ok, first = capture.read()
        grays = []
        while ok and len(grays) < pairs + 1:
            grays.append(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
            ok, first = capture.read()
        capture.release()
        if len(grays) < 2:
            continue
        started = time.perf_counter()
        for a, b in zip(grays, grays[1:]):
            dense_flow(a, b)
        dense = (time.perf_counter() - started) / (len(grays) - 1)
        started = time.perf_counter()
        for a, b in zip(grays, grays[1:]):
            corners = cv2.goodFeaturesToTrack(a, 200, .01, 7)
            if corners is not None:
                cv2.calcOpticalFlowPyrLK(a, b, corners, None)
        sparse = (time.perf_counter() - started) / (len(grays) - 1)
        rows.append(dict(video_id=r.video_id, resolution=f'{r.width}x{r.height}', pairs=len(grays) - 1,
                         farneback_ms=1e3 * dense, lucas_kanade_ms=1e3 * sparse))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 v2 Phase 5 efficiency.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--v1-report', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--configs', default='D0,D3,D4')
    parser.add_argument('--flow-videos', type=int, default=6)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--out-name', default='phase5')
    args = parser.parse_args(argv)
    run = (ROOT/args.run).resolve()
    v1 = (ROOT/args.v1_report).resolve()
    work = V2/'runs'/run.name
    target = V2/'reports'/run.name/args.out_name
    target.mkdir(parents=True, exist_ok=True)
    out = run/'outputs'
    manifest = pd.read_csv(run/'manifest.csv')
    if args.max_videos:
        manifest = manifest.head(args.max_videos)

    # 1. stage profile
    timings = json.loads((run/'phase_timings.json').read_text())
    stamps = [l[:19] for l in (run/'pipeline.log').read_text(errors='replace').splitlines() if l[:4].isdigit()]
    base_wall = (pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[0])).total_seconds() / 3600 if stamps else None
    sampled = pd.read_csv(out/'confidence/roi_confidence.csv', usecols=['video_id', 'frame_index'],
                          low_memory=False).drop_duplicates()
    n_sampled = len(sampled)
    gpu = pd.read_csv(run/'gpu_monitor.csv', header=None, names=['time', 'memory', 'util'])
    gpu_peak = float(gpu.memory.str.extract(r'(\d+)')[0].astype(float).max())
    extra = {'v2_features': json.loads((work/'features/features_metadata_median.json').read_text())['wall_time_sec']}
    p1 = json.loads((V2/'reports'/run.name/'phase1/phase1_metadata.json').read_text())
    extra['v2_evaluation'] = p1['wall_time_sec']
    rows = [dict(stage=k, seconds=v, sec_per_sampled_frame=v / n_sampled, scales_with_frames=k in PER_FRAME)
            for k, v in {**timings, **extra}.items()]
    profile = pd.DataFrame(rows)
    profile['share_percent'] = 100 * profile.seconds / profile.seconds.sum()
    profile.to_csv(target/'table_phase5_profile.csv', index=False)

    # 2. stride variants
    log = json.loads((work/'tuning_log.json').read_text())
    configs = {c: f for c, f in ev.configurations(log['chosen']).items() if c in args.configs.split(',')}
    tau = json.loads((v1/'evaluation_metadata.json').read_text())['thresholds']['informative_snr_db']
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False, usecols=ev.CAND_COLUMNS)
    features = pd.read_csv(work/'features/roi_features_median.csv')
    cand = cand.merge(features[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                                'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    subjects = pd.read_csv(work/'features/subject_frames.csv')
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = []
    for r in manifest.itertuples():
        c = cand[cand.video_id == r.video_id]
        frames = sorted(set(sampled[sampled.video_id == r.video_id].frame_index) | set(c.frame_index))
        base = (r.video_id, r._asdict(), str(out), c, frames, recovery[recovery.video_id == r.video_id],
                sam2[sam2.video_id == r.video_id], subjects[subjects.video_id == r.video_id], configs)
        jobs += [(s, base) for s in STRIDES]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(stride_job, jobs))
    per_video = pd.DataFrame([row for r in results for row in r[0]])
    windows = pd.DataFrame([row for r in results for row in r[1]])
    for (stride, cid), part in per_video.groupby(['stride', 'config']):
        rate = windows[windows.stride == stride].groupby('video_id')[f'snr_{cid}'].apply(lambda s: 100. * (s >= tau).mean())
        per_video.loc[part.index, 'informative_window_percent'] = part.video_id.map(rate)
    per_video.to_csv(target/'per_video_stride.csv', index=False)

    scaled = profile[profile.scales_with_frames].seconds.sum()
    fixed = profile[~profile.scales_with_frames].seconds.sum()
    metrics = ['switches_per_sec', 'temporal_iou', 'pulse_snr_db', 'informative_window_percent']
    rows, tests = [], []
    for cid in configs:
        ref = per_video[(per_video.config == cid) & (per_video.stride == 1)].set_index('video_id')
        for s in STRIDES:
            part = per_video[(per_video.config == cid) & (per_video.stride == s)].set_index('video_id')
            row = dict(config=cid, stride=s, projected_hours=(scaled / s + fixed) / 3600,
                       speedup=(scaled + fixed) / (scaled / s + fixed),
                       effective_fps=float(np.nanmean(part.fps_sampled)))
            for m in metrics:
                row[m] = float(part[m].mean())
            rows.append(row)
            if s > 1:
                block = []
                for m in metrics:
                    block.append(dict(config=cid, stride=s, metric=m, **paired_test(ref[m], part[m].reindex(ref.index))))
                for b, adjusted in zip(block, holm([b['p_value'] for b in block])):
                    b['p_holm'] = adjusted
                tests += block
    pd.DataFrame(rows).to_csv(target/'table_phase5_stride.csv', index=False)
    pd.DataFrame(tests).to_csv(target/'paired_stride_vs_1.csv', index=False)

    # 3. optical flow timing
    flow = flow_timing(manifest.sort_values('width').iloc[np.linspace(0, len(manifest) - 1, min(args.flow_videos, len(manifest))).astype(int)])
    flow.to_csv(target/'table_phase5_flow.csv', index=False)

    (target/'phase5_metadata.json').write_text(json.dumps(dict(
        run=str(run), sampled_frames=n_sampled, gpu_peak_mib=gpu_peak, strides=STRIDES, configs=list(configs),
        projection='summed stage time (both SAM 2 / no-SAM 2 branches, which ran concurrently, so this exceeds wall clock); per-frame stages scaled by 1/stride; other stages fixed; stages NOT re-run',
        base_run_wall_clock_hours=base_wall,
        stabilizer_rescaling='ema_beta -> beta**s; hysteresis_frames -> round(N/s)', tau_db=tau,
        lighter_pose='not evaluated: no checkpoint other than Sapiens2-0.4B available',
        wall_time_sec=time.perf_counter() - started), indent=2, default=str))
    print(profile.round(3).to_string(index=False))
    print(pd.read_csv(target/'table_phase5_stride.csv').round(3).to_string(index=False))
    print(flow.round(2).to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
