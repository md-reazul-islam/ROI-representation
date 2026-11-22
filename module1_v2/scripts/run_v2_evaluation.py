"""Phase 1e evaluation: v1 configurations B0-Full (taken unchanged from the v1 evaluation report of the same
run) plus the new D-configurations, evaluated with identical pixels, pulse extraction, windows and the v1
informative-window threshold tau (fixed on training videos by the v1 evaluation).

D0  = v1-equivalent flags through the v2 code path (must reproduce Full exactly; checked and logged)
D1  = Full + EMA of Q                          D2  = D1 + hysteresis
D3  = Full + offline Viterbi anchor (+ D2 companions)   D3L = online fixed-lag Viterbi variant
D4  = best of D2/D3/D3L (training objective) + fusion-weight smoothing + cross-fade
D5  = D4 + keypoint visibility V'              D6  = D5 + nuisance-only motion penalty
D7  = D6 + infant-vs-adult subject verification (abstain on adult)
I-V, I-M, I-S = the single fixes of D5/D6/D7 applied alone on Full (isolated effects)
I-B = Full + 2-point (BBOX) ROIs read as rectangles for pixels; D1-D7 include this fix

Usage: python module1_v2/scripts/run_v2_evaluation.py --run runs/dataset5_paper --v1-report <dir>
"""
from __future__ import annotations

import argparse
import json
import math
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

from module1 import paper_eval as pe                                         # noqa: E402  (v1, read-only)
from module2.ablation import bootstrap_ci, holm, paired_test                 # noqa: E402
from module2.extraction import hold_indices, mean_rgb                        # noqa: E402
from module2.rppg import analysis_windows, window_snr                        # noqa: E402
from run_paper_evaluation import EXTRACT, HR, boxes_of, load_mask, pulse     # noqa: E402  (v1 helpers)
from module1_v2 import selection as sel, subject as sj, terms                # noqa: E402

CFG = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())
V1_CONFIGS = ('B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'Full')
METRICS = ('coverage_percent', 'temporal_iou', 'centroid_jitter', 'selection_jaccard', 'switches_per_sec',
           'mask_support', 'pulse_snr_db', 'informative_window_percent')
DIGITS = dict(coverage_percent=1, temporal_iou=3, centroid_jitter=3, selection_jaccard=3, switches_per_sec=2,
              mask_support=3, pulse_snr_db=2, informative_window_percent=1, retention=3)
CAND_COLUMNS = ['video_id', 'split', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'q', 'pose_term',
                'visibility_term', 'skin_term', 'temporal_term', 'motion_term', 'blur_score', 'illumination_score',
                'saturation_score', 'swaddling_proxy', 'SAM2_mask_support']


def configurations(chosen):
    """{ID: flags} from the training-tuned values (tuning_log.json['chosen'])."""
    base = dict(CFG['v1_equivalent'])
    p1 = CFG['phase1']
    d1 = dict(base, ema_beta=chosen['D1']['ema_beta'])
    d2 = dict(base, ema_beta=chosen['D2']['ema_beta'], selection='hysteresis',
              hysteresis_delta=chosen['D2']['hysteresis_delta'], hysteresis_frames=chosen['D2']['hysteresis_frames'])
    d3 = dict(d2, selection='viterbi', viterbi_penalty=chosen['D3']['viterbi_penalty'], viterbi_lag=None)
    d3l = dict(d2, selection='viterbi', viterbi_penalty=chosen['D3L']['viterbi_penalty'],
               viterbi_lag=chosen['D3L']['viterbi_lag'])
    d4 = dict(dict(D2=d2, D3=d3, D3L=d3l)[chosen['D4_base']], alpha_ema=p1['alpha_ema'], crossfade_sec=p1['crossfade_sec'])
    d5 = dict(d4, visibility='keypoint')
    d6 = dict(d5, motion='residual_nuisance')
    d7 = dict(d6, subject_verify=True)
    out = {'D0': base, 'D1': d1, 'D2': d2, 'D3': d3, 'D3L': d3l, 'D4': d4, 'D5': d5, 'D6': d6, 'D7': d7,
           'I-V': dict(base, visibility='keypoint'), 'I-M': dict(base, motion='residual_nuisance'),
           'I-S': dict(base, subject_verify=True), 'I-B': dict(base, box_roi_pixels='rectangle')}
    for cid in ('D1', 'D2', 'D3', 'D3L', 'D4', 'D5', 'D6', 'D7'):   # every v2 configuration reads box ROIs as rectangles
        out[cid] = dict(out[cid], box_roi_pixels='rectangle')
    return out


def evaluate_video(job):
    video, record, out, cand, frames, recovery, sam2_rows, subjects, configs = job
    started = time.perf_counter()
    out = Path(out)
    fps = float(record['fps'])
    duration = float(record['duration_sec'])
    fps_sampled = len(frames) / duration if duration > 0 else np.nan
    boxes = boxes_of(recovery)
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    reliable = {}
    for f in frames:
        if len(sam2_rows) and f in sam2_rows.index:
            mask = load_mask(out/'sam2', sam2_rows.loc[f])
            if mask is not None:
                reliable[f] = mask
    sel_frames = sorted(cand.frame_index.unique())
    reject = sel.subject_reject(subjects, sel_frames) if len(subjects) else np.zeros(len(sel_frames), bool)

    # Mask-restricted colour of every candidate polygon (v1 'm1' pixels, identical code path).
    capture = cv2.VideoCapture(str(record['video_path']))
    n = int(record['frame_count'])
    hold = hold_indices(frames, n, fps, EXTRACT.max_hold_sec)
    by_frame = {f: rows for f, rows in cand[cand.roi_status == 'AVAILABLE'].groupby('frame_index')}
    names = sel.NAMES
    col = {r: j for j, r in enumerate(names)}
    rgb = np.full((n, len(names), 3), np.nan)          # v1 pixels: 2-point boxes carry none
    fix = any(f.get('box_roi_pixels') == 'rectangle' for f in configs.values())
    rgb_box = rgb.copy() if fix else None             # v2 pixels: 2-point boxes read as rectangles
    for i in range(n):
        ok, image = capture.read()
        if not ok:
            break
        source = int(hold[i])
        rows = by_frame.get(source) if source >= 0 else None
        if rows is None:
            continue
        mask = reliable.get(source)
        masked = None
        if mask is not None and mask.shape == image.shape[:2]:
            masked = image.copy()
            masked[~mask] = 0
        for row in rows.itertuples():
            polygon = pe.parse(row.polygon)
            source_image = masked if masked is not None else image
            if polygon and len(polygon) >= 3:
                value, _ = mean_rgb(source_image, np.asarray(polygon, float), EXTRACT)
                if value is not None:
                    rgb[i, col[row.roi_name]] = value
                    if fix:
                        rgb_box[i, col[row.roi_name]] = value
            elif fix:
                rect = sel.pixel_polygon(polygon, 'rectangle')
                if rect is not None:
                    value, _ = mean_rgb(source_image, np.asarray(rect, float), EXTRACT)
                    if value is not None:
                        rgb_box[i, col[row.roi_name]] = value
    capture.release()
    pulses = np.column_stack([pulse(rgb[:, j], fps) for j in range(len(names))])
    pulses_box = None
    if fix:
        pulses_box = pulses.copy()
        for j in range(len(names)):                   # recompute only the regions whose traces differ
            if not np.array_equal(np.isnan(rgb[:, j]), np.isnan(rgb_box[:, j])):
                pulses_box[:, j] = pulse(rgb_box[:, j], fps)
    windows = analysis_windows(n, fps, HR)

    per_video, window_rows = [], [dict(video_id=video, split=record['split'], window=w) for w in range(len(windows))]
    for cid, flags in configs.items():
        geometry, fusion, Q, sets = sel.select_video(cand, sel_frames, flags, fps_sampled, CFG['phase1']['top_k'],
                                                     CFG['phase1']['temperature'],
                                                     reject if flags.get('subject_verify') else None)
        g = pe.geometry_metrics(geometry, boxes, fps_sampled, masks=reliable)
        g['mask_support'] = 1. if reliable else np.nan        # pixels outside a reliable mask are never used
        source = pulses_box if flags.get('box_roi_pixels') == 'rectangle' else pulses
        fused = pe.fused_pulse(source, pe.native_weights(fusion, names, hold, n))
        snr = []
        for w, (a, b) in enumerate(windows):
            value = window_snr(fused[a:b], fps, HR)
            window_rows[w][f'snr_{cid}'] = value
            snr.append(value)
        snr = np.asarray([s for s in snr if s is not None and math.isfinite(s)], float)
        per_video.append(dict(video_id=video, split=record['split'], config=cid, fps_sampled=fps_sampled,
                              sampled_frames=len(sel_frames), reliable_mask_frames=len(reliable),
                              pulse_snr_db=float(np.median(snr)) if len(snr) else np.nan,
                              retention=sel.retention(Q, sets, CFG['phase1']['top_k']), **g))
    rejected = reject.astype(bool)
    subject = dict(video_id=video, split=record['split'], sampled_frames=len(sel_frames),
                   rejected_frames=int(rejected.sum()), rejected_episodes=len(sj.episodes(rejected)),
                   frames_r1_measurable=int(np.isfinite(subjects.head_shoulder_ratio).sum()) if len(subjects) else 0,
                   median_head_shoulder_ratio=float(np.nanmedian(subjects.head_shoulder_ratio)) if len(subjects) and np.isfinite(subjects.head_shoulder_ratio).any() else np.nan,
                   median_arm_torso_ratio=float(np.nanmedian(subjects.arm_torso_ratio)) if len(subjects) and np.isfinite(subjects.arm_torso_ratio).any() else np.nan,
                   adult_frames=int((subjects.frame_decision == 'ADULT').sum()) if len(subjects) else 0,
                   unknown_frames=int((subjects.frame_decision == 'UNKNOWN').sum()) if len(subjects) else 0)
    return per_video, window_rows, subject, time.perf_counter() - started


def mean_sd(values, digits):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 'n/a'
    return f'{values.mean():.{digits}f} ± {values.std(ddof=1) if len(values) > 1 else 0.:.{digits}f}'


def summary(per_video, order, metrics):
    rows = []
    for c in order:
        part = per_video[per_video.config == c]
        row = dict(config=c, videos=int(part.video_id.nunique()))
        for m in metrics:
            if m not in part:
                continue
            _, low, high = bootstrap_ci(part[m])
            row[m] = mean_sd(part[m], DIGITS[m])
            row[f'{m}_ci95'] = f'[{low:.{DIGITS[m]}f}, {high:.{DIGITS[m]}f}]' if math.isfinite(low) else 'n/a'
        rows.append(row)
    return pd.DataFrame(rows)


def paired(per_video, order, metrics, reference):
    rows = []
    for m in metrics:
        wide = per_video.pivot_table(index='video_id', columns='config', values=m)
        block = [dict(metric=m, baseline=reference, candidate=c, **paired_test(wide[reference], wide[c]))
                 for c in order if c != reference and c in wide and reference in wide]
        for row, adjusted in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = adjusted
        rows += block
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 v2 Phase 1 evaluation.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--v1-report', required=True, help='v1 run_paper_evaluation output dir for the same run')
    parser.add_argument('--global-motion', default=CFG['phase1']['global_motion'])
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    run = (ROOT/args.run).resolve()
    v1 = (ROOT/args.v1_report).resolve()
    work = V2/'runs'/run.name
    target = V2/'reports'/run.name/'phase1'
    target.mkdir(parents=True, exist_ok=True)
    out = run/'outputs'
    log = json.loads((work/'tuning_log.json').read_text())
    configs = configurations(log['chosen'])
    meta = json.loads((v1/'evaluation_metadata.json').read_text())
    tau = meta['thresholds']['informative_snr_db']

    manifest = pd.read_csv(run/'manifest.csv')
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False, usecols=CAND_COLUMNS)
    features = pd.read_csv(work/'features'/f'roi_features_{args.global_motion}.csv')
    cand = cand.merge(features[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                                'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    subjects = pd.read_csv(work/'features/subject_frames.csv')
    frames0 = pd.read_csv(out/'confidence/roi_confidence.csv', low_memory=False, usecols=['video_id', 'frame_index'])
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = []
    for r in manifest.itertuples():
        c = cand[cand.video_id == r.video_id]
        frames = sorted(set(frames0[frames0.video_id == r.video_id].frame_index) | set(c.frame_index))
        jobs.append((r.video_id, r._asdict(), str(out), c, frames, recovery[recovery.video_id == r.video_id],
                     sam2[sam2.video_id == r.video_id], subjects[subjects.video_id == r.video_id], configs))
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate_video, jobs))
    v2 = pd.DataFrame([row for r in results for row in r[0]])
    windows = pd.DataFrame([row for r in results for row in r[1]])
    subject_table = pd.DataFrame([r[2] for r in results])
    for cid in configs:
        v2.loc[v2.config == cid, 'informative_window_percent'] = v2[v2.config == cid].video_id.map(
            windows.groupby('video_id')[f'snr_{cid}'].apply(lambda s: 100. * (s >= tau).mean()))

    v1_rows = pd.read_csv(v1/'per_video_metrics.csv')
    v1_rows = v1_rows[v1_rows.config.isin(V1_CONFIGS)]
    # Regression: D0 (v2 code path, v1 flags) must equal v1 Full on every metric.
    check = v1_rows[v1_rows.config == 'Full'].set_index('video_id')
    d0 = v2[v2.config == 'D0'].set_index('video_id').loc[check.index]
    regression = {m: float(np.nanmax(np.abs(d0[m].to_numpy(float) - check[m].to_numpy(float))))
                  if np.isfinite(check[m].to_numpy(float)).any() else 0. for m in METRICS}
    per_video = pd.concat([v1_rows, v2], ignore_index=True)
    order = list(V1_CONFIGS) + list(configs)
    per_video.to_csv(target/'per_video_metrics.csv', index=False)
    windows.to_csv(target/'window_metrics_v2.csv', index=False)
    subject_table.to_csv(target/'subject_verification.csv', index=False)
    summary(per_video, order, METRICS + ('retention',)).to_csv(target/'table_phase1_main.csv', index=False)
    for split in ('train', 'validation', 'test'):
        part = per_video[per_video.split == split]
        if len(part):
            summary(part, order, METRICS).to_csv(target/f'table_phase1_{split}.csv', index=False)
    paired(per_video, order, METRICS, 'Full').to_csv(target/'paired_vs_Full.csv', index=False)
    paired(per_video, order, METRICS, 'B0').to_csv(target/'paired_vs_B0.csv', index=False)

    # 1b: condition-term correlation audit on TRAINING ROI-frames only.
    train = cand[(cand.split == 'train') & (cand.roi_status == 'AVAILABLE')]
    pearson, spearman, flags = terms.correlation_audit(train)
    pearson.to_csv(target/'term_correlation_pearson_train.csv')
    spearman.to_csv(target/'term_correlation_spearman_train.csv')
    flags.to_csv(target/'term_correlation_flags_train.csv', index=False)

    (target/'phase1_metadata.json').write_text(json.dumps(dict(
        run=str(run), v1_report=str(v1), tau_db=tau, tau_source='v1 evaluation: median training ROI-window SNR',
        configs=configs, tuning=log['chosen'], training_videos=log['training_videos'],
        regression_D0_vs_Full_max_abs_diff=regression, global_motion=args.global_motion,
        wall_time_sec=time.perf_counter() - started, per_video_sec={j[0]: r[3] for j, r in zip(jobs, results)}),
        indent=2, default=str))
    print('D0 vs v1 Full max |diff|:', regression)
    print(pd.read_csv(target/'table_phase1_main.csv')[['config', 'videos', 'switches_per_sec', 'temporal_iou',
                                                       'mask_support', 'pulse_snr_db', 'informative_window_percent']].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
