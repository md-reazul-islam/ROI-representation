"""Revision items 3 and 4 (reviewer requests). Writes ONLY to module1_v2/results/revision_items1-4/.

Item 3  background negative control 'BG': a fixed box of the B0 size, placed in the B0 frame (first frame with an
        infant box) inside the largest connected region that lies outside (SAM 2 mask dilated by 15 px) and outside
        (infant box enlarged by 10%). Same pixel rule as B0 (unmasked image, mean_rgb), same POS pulse, windows, SNR,
        informative threshold tau, split-half agreement and block-shuffle pulse-present test as Phase 2.
Item 4  block-shuffle pulse-present test with 199 surrogates for every Table 6 configuration + BG. Surrogate s uses
        exactly the Phase 2 generator (seed 42, video CRC, s, shuffle), so surrogates 0-18 are the Phase 2 ones and the
        19-surrogate rate is reproduced as a check. v1 Full is re-tested with a second, independent set of 199
        surrogates (seed 43) to measure the Monte-Carlo uncertainty of its rate.
Nothing is tuned; tau, windows, weights and all settings are the frozen Phase 2 ones.

Usage: python module1_v2/scripts/revision_items.py --run runs/final34_paper --v1-report reports/paper_final34_paper
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import dataclasses

import cv2
import numpy as np
import pandas as pd

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2/'scripts'))
import run_phase2 as p2                                                       # noqa: E402  (read-only reuse)
from run_phase2 import (EXTRACT, HR, P2, CONFIGS, SIDES, bl, boxes_of, load_mask, mean_rgb, pe, pulse, sel,  # noqa: E402
                        window_hr, window_snr, finite, analysis_windows, configurations, CAND_COLUMNS)

OUT = V2/'results'/'revision_items1-4'
N_SUR = 199
SEED2 = 43
DILATE_PX = 15
BOX_ENLARGE = 1.10
# BG = identical to B0 (adaptive skin gate); BGn / B0n = same boxes without the skin gate (all non-dark,
# non-saturated pixels), because background rarely contains skin-coloured pixels
# BGm / BGmn = fallback placement that excludes only the dilated SAM 2 mask (not the enlarged infant box)
EXTRA = ('BG', 'BGn', 'BGm', 'BGmn', 'B0n')
NOSKIN = dataclasses.replace(EXTRACT, skin_mode='none')


class UnitsBG(p2.Units):
    def __init__(self, names, tiles):
        super().__init__(names, tiles)
        self.keys += [('fix', f, s) for f in EXTRA for s in SIDES]
        self.index = {k: i for i, k in enumerate(self.keys)}


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    return image if ok else None


def place_background(record, out, boxes, sam2_rows, b0, exclude_box=True):
    """Background box of the B0 size; returns (polygon or None, info dict, overlay image)."""
    f0 = next((f for f in sorted(boxes) if boxes[f] is not None), None)
    info = dict(video_id=record['video_id'], frame=f0)
    if f0 is None or b0 is None:
        return None, dict(info, status='no_infant_box'), None
    image = read_frame(record['video_path'], f0)
    h, w = image.shape[:2]
    b0 = np.asarray(b0, float)
    bw, bh = int(round(np.ptp(b0[:, 0]))), int(round(np.ptp(b0[:, 1])))
    # SAM 2 mask of the B0 frame, else of the nearest frame with a reliable mask
    mask, mask_frame = None, None
    rows = sam2_rows[sam2_rows.mask_reliable.astype(bool)] if len(sam2_rows) else sam2_rows
    if len(rows):
        order = rows.iloc[np.argsort(np.abs(rows.frame_index.to_numpy() - f0), kind='stable')]
        for _, row in order.iterrows():
            m = load_mask(Path(out)/'sam2', row)
            if m is not None and m.shape == (h, w):
                mask, mask_frame = m, int(row.frame_index)
                break
    forbidden = np.zeros((h, w), bool)
    if mask is not None:
        k = 2 * DILATE_PX + 1
        forbidden |= cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    x1, y1, x2, y2 = boxes[f0]
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * BOX_ENLARGE / 2, (y2 - y1) * BOX_ENLARGE / 2
    ex = [int(np.floor(max(cx - hw, 0))), int(np.floor(max(cy - hh, 0))), int(np.ceil(min(cx + hw, w))),
          int(np.ceil(min(cy + hh, h)))]
    if exclude_box:
        forbidden[ex[1]:ex[3], ex[0]:ex[2]] = True
    # letterbox / pillarbox bars (rows or columns that are black in the frame) are not image content
    bright = image.max(axis=2)
    bars = np.zeros((h, w), bool)
    bars[:, np.percentile(bright, 99, axis=0) <= EXTRACT.dark_level] = True
    bars[np.percentile(bright, 99, axis=1) <= EXTRACT.dark_level, :] = True
    allowed = (~forbidden & ~bars).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(allowed, connectivity=4)
    comps = sorted(range(1, count), key=lambda c: -stats[c, cv2.CC_STAT_AREA])
    info.update(mask_frame=mask_frame, b0_w=bw, b0_h=bh, bar_fraction=float(bars.mean()),
                largest_region_px=int(stats[comps[0], cv2.CC_STAT_AREA]) if comps else 0)
    polygon, scale, used = None, 1.0, None
    attempts = ([(rank, 1.0) for rank in range(len(comps))] + [(0, s) for s in np.arange(.95, .19, -.05)]) if comps else []
    for rank, s in attempts:
        region = (labels == comps[rank]).astype(np.uint8)
        ww, hh2 = max(2, int(bw * s)), max(2, int(bh * s))
        if ww >= w or hh2 >= h or stats[comps[rank], cv2.CC_STAT_AREA] < ww * hh2:
            continue
        integral = cv2.integral(1 - region)              # count of non-region pixels
        S = integral[hh2:, ww:] - integral[:-hh2, ww:] - integral[hh2:, :-ww] + integral[:-hh2, :-ww]
        fits = S == 0                                    # top-left (y, x) where the box lies fully in the region
        if fits.any():
            dist = cv2.distanceTransform(region, cv2.DIST_L2, 5)
            ys, xs = np.nonzero(fits)
            j = int(np.argmax(dist[ys + hh2 // 2, xs + ww // 2]))
            X, Y = float(xs[j]), float(ys[j])
            polygon = [[X, Y], [X + ww, Y], [X + ww, Y + hh2], [X, Y + hh2]]
            scale, used = float(s), rank
            break
    info.update(status='not_placed' if polygon is None else ('placed_shrunk' if scale < 1 else
                ('placed' if used == 0 else 'placed_smaller_region')),
                scale=scale, region_rank=used, bg_polygon=json.dumps(polygon) if polygon else None)
    overlay = image.copy()
    overlay[forbidden] = (overlay[forbidden] * .5).astype(np.uint8)
    cv2.polylines(overlay, [np.int32(b0)], True, (0, 165, 255), 4)
    if polygon:
        cv2.polylines(overlay, [np.int32(polygon)], True, (255, 120, 40), 4)
    return polygon, info, overlay


def bg_rgb(record, n, boxes_by_name, units, rgb):
    """Fill the BG / BGn / B0n columns: unmasked image, every native frame (B0 pixel rule; *n = no skin gate)."""
    jobs = []
    for name, (polygon, config) in boxes_by_name.items():
        if polygon is None:
            continue
        for side, poly in {'': polygon, 'L': bl.clip_half(polygon, 'L'), 'R': bl.clip_half(polygon, 'R')}.items():
            if poly is not None and len(poly) >= 2:
                jobs.append((units[('fix', name, side)], np.asarray(poly, float), config))
    cap = cv2.VideoCapture(str(record['video_path']))
    for i in range(n):
        ok, image = cap.read()
        if not ok:
            break
        for col, poly, config in jobs:
            value, _ = mean_rgb(image, poly, config)
            if value is not None:
                rgb[i, col] = value
    cap.release()


def fixed_windows(trace, windows, fps, halves=None):
    rows = []
    for a, b in windows:
        row = [window_snr(trace[a:b], fps, HR)]
        if halves is not None:
            row += [window_hr(trace[a:b], fps), window_hr(halves['L'][a:b], fps), window_hr(halves['R'][a:b], fps)]
        rows.append(row)
    return rows


def evaluate_video(job):
    video, record, out, cand, cand0, frames, recovery, sam2_rows, configs, tau, seed = job
    started = time.perf_counter()
    out = Path(out)
    fps, duration = float(record['fps']), float(record['duration_sec'])
    fps_sampled = len(frames) / duration if duration > 0 else np.nan
    names = sel.NAMES
    units = UnitsBG(names, P2['grid_rows'] * P2['grid_cols'])
    rgb, hold, tile_column = p2.extract(record, out, cand, cand0, frames, recovery, sam2_rows, units, names)
    n = len(rgb)
    boxes = boxes_of(recovery)
    b0 = pe.fixed_rois(cand0, boxes)[0]
    polygon, info, overlay = place_background(record, out, boxes, sam2_rows, b0)
    poly_m, info_m, overlay_m = place_background(record, out, boxes, sam2_rows, b0, exclude_box=False)
    info = dict(info, **{f'maskonly_{k}': v for k, v in info_m.items() if k not in ('video_id', 'frame')})
    overlay = (overlay, overlay_m)
    bg_rgb(record, n, {'BG': (polygon, EXTRACT), 'BGn': (polygon, NOSKIN), 'BGm': (poly_m, EXTRACT),
                       'BGmn': (poly_m, NOSKIN), 'B0n': (b0, NOSKIN)}, units, rgb)
    tile_left = {t: c < P2['grid_cols'] / 2 for t, c in tile_column.items()}
    for t in range(P2['grid_rows'] * P2['grid_cols']):
        tile_left.setdefault(t, t % P2['grid_cols'] < P2['grid_cols'] / 2)
    pulses = np.column_stack([pulse(rgb[:, j], fps) for j in range(rgb.shape[1])])
    windows = analysis_windows(n, fps, HR)
    lag = max(1, int(round(HR.estimation_window_sec / HR.estimation_step_sec)))
    sel_frames = sorted(cand.frame_index.unique())
    weights = {}
    for cid in p2.WEIGHTED:
        flags = configs['D0' if cid == 'Full' else cid]
        _, fusion, _, _ = sel.select_video(cand, sel_frames, flags, fps_sampled, p2.CFG['phase1']['top_k'],
                                           p2.CFG['phase1']['temperature'], None)
        weights[cid] = pe.native_weights(fusion, names, hold, n)
    motion_tab = cand[cand.roi_status == 'AVAILABLE'].pivot_table(index='frame_index', columns='roi_name',
                                                                  values='motion_magnitude')
    motion = np.full((len(windows), len(names)), np.nan)
    for w, (a, b) in enumerate(windows):
        part = motion_tab[(motion_tab.index >= a) & (motion_tab.index < b)]
        if len(part):
            motion[w] = part.reindex(columns=names).mean().to_numpy(float)

    def run_config(cid, P, observed=True):
        if cid in EXTRA:
            halves = {s: P[:, units[('fix', cid, s)]] for s in ('L', 'R')} if observed else None
            return fixed_windows(P[:, units[('fix', cid, '')]], windows, fps, halves)
        return p2.config_windows(cid, P, units, names, windows, fps, weights, motion, lag, tile_left, observed=observed)

    all_cfg = CONFIGS + EXTRA
    rows = [dict(video_id=video, split=record['split'], window=w) for w in range(len(windows))]
    observed = {}
    for cid in all_cfg:
        result = run_config(cid, pulses)
        observed[cid] = np.array([r[0] if finite(r[0]) else np.nan for r in result], float)
        for w, (snr, hr, hl, hrr) in enumerate(result):
            rows[w][f'snr_{cid}'] = snr
            rows[w][f'hr_{cid}'] = hr
            rows[w][f'agree_{cid}'] = bl.agree(hl, hrr, P2['agreement_bpm'])
    block = max(1, int(round(P2['shuffle_block_sec'] * fps)))
    full_units = [i for k, i in units.index.items() if k[2] == '']
    null = {c: [] for c in all_cfg}
    for s in range(N_SUR):
        rng = np.random.default_rng([seed, zlib.crc32(video.encode()), s, False])   # = Phase 2 shuffle generator
        sur = np.full_like(pulses, np.nan)
        for j in full_units:
            sur[:, j] = pulse(bl.shuffle_surrogate(rgb[:, j], rng, block), fps)
        for cid in all_cfg:
            null[cid].append(np.array([r[0] if finite(r[0]) else np.nan for r in run_config(cid, sur, False)], float))
    # second independent draw for v1 Full only (Monte-Carlo uncertainty)
    full_cols = [units[('roiv1', r, '')] for r in names]
    null_full2 = []
    for s in range(N_SUR):
        rng = np.random.default_rng([SEED2, zlib.crc32(video.encode()), s, False])
        sur = np.full_like(pulses, np.nan)
        # draw for every unit in the same order as above so the generator stream is used identically
        for j in full_units:
            trace = bl.shuffle_surrogate(rgb[:, j], rng, block)
            if j in full_cols:
                sur[:, j] = pulse(trace, fps)
        null_full2.append(np.array([r[0] if finite(r[0]) else np.nan for r in run_config('Full', sur, False)], float))

    per_video = []
    for cid in all_cfg:
        snr = observed[cid]
        good = np.isfinite(snr)
        pres19 = bl.present(snr, np.concatenate(null[cid][:P2['surrogates_shuffle']]), P2['null_percentile'])
        pres199 = bl.present(snr, np.concatenate(null[cid]), P2['null_percentile'])
        agreement = [rows[w][f'agree_{cid}'] for w in range(len(windows))]
        testable = [a for a in agreement if a is not None]
        row = dict(video_id=video, split=record['split'], config=cid, windows=len(windows),
                   usable_window_percent=100. * good.mean() if len(snr) else np.nan,
                   pulse_snr_db=float(np.median(snr[good])) if good.any() else np.nan,
                   informative_window_percent=100. * np.mean(np.where(good, snr, -np.inf) >= tau) if len(snr) else np.nan,
                   present_shuffle19_percent=100. * pres19.mean() if len(windows) else np.nan,
                   present_shuffle199_percent=100. * pres199.mean() if len(windows) else np.nan,
                   agreement_percent=100. * np.mean(testable) if testable else np.nan,
                   agreement_testable_percent=100. * len(testable) / len(windows) if len(windows) else np.nan,
                   null95_db_199=float(np.nanpercentile(np.concatenate(null[cid]), 95))
                   if np.isfinite(np.concatenate(null[cid])).any() else np.nan)
        if cid == 'Full':
            pres2 = bl.present(snr, np.concatenate(null_full2), P2['null_percentile'])
            row['present_shuffle199_seed43_percent'] = 100. * pres2.mean() if len(windows) else np.nan
        per_video.append(row)
        for w in range(len(windows)):
            rows[w][f'present199_{cid}'] = bool(pres199[w])
    return per_video, rows, info, overlay, time.perf_counter() - started


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--v1-report', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', default='')
    args = ap.parse_args(argv)
    root = V2.parent
    run, v1 = (root/args.run).resolve(), (root/args.v1_report).resolve()
    work, out = V2/'runs'/run.name, run/'outputs'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'bg_placement').mkdir(exist_ok=True)
    configs = configurations(json.loads((work/'tuning_log.json').read_text())['chosen'])
    tau = json.loads((v1/'evaluation_metadata.json').read_text())['thresholds']['informative_snr_db']
    manifest = pd.read_csv(run/'manifest.csv')
    only = [v for v in args.only.split(',') if v]
    if only:
        manifest = manifest[manifest.video_id.isin(only)]
    cand = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=CAND_COLUMNS + ['motion_magnitude'])
    features = pd.read_csv(work/'features/roi_features_median.csv')
    cand = cand.merge(features[['video_id', 'frame_index', 'roi_name', 'visibility_keypoint', 'motion_nuisance',
                                'motion_total', 'motion_residual', 'respiratory_ratio']],
                      on=['video_id', 'frame_index', 'roi_name'], how='left')
    cand0 = pd.read_csv(out/'confidence/roi_confidence.csv', low_memory=False,
                        usecols=['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'anatomical_confidence'])
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    jobs = []
    for r in manifest.itertuples():
        c, c0 = cand[cand.video_id == r.video_id], cand0[cand0.video_id == r.video_id]
        frames = sorted(set(c0.frame_index) | set(c.frame_index))
        jobs.append((r.video_id, r._asdict(), str(out), c, c0, frames, recovery[recovery.video_id == r.video_id],
                     sam2[sam2.video_id == r.video_id], configs, tau, P2['seed']))
    started = time.perf_counter()
    per_video, windows, placement = [], [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for job, (pv, rows, info, overlay, sec) in zip(jobs, pool.map(evaluate_video, jobs)):
            per_video += pv
            windows += rows
            placement.append(dict(info, seconds=sec))
            for tag, ov in zip(('spec', 'maskonly'), overlay):
                if ov is not None:
                    small = cv2.resize(ov, None, fx=480 / ov.shape[1], fy=480 / ov.shape[1])
                    cv2.imwrite(str(OUT/'bg_placement'/f'{job[0]}_{tag}.jpg'), small)
            print(f'{job[0]} done in {sec / 60:.1f} min ({info.get("status")})', flush=True)
            pd.DataFrame(per_video).to_csv(OUT/'items3_4_per_video_raw.csv', index=False)   # checkpoint
    pd.DataFrame(per_video).to_csv(OUT/'items3_4_per_video_raw.csv', index=False)
    pd.DataFrame(windows).to_csv(OUT/'items3_4_window_metrics.csv', index=False)
    pd.DataFrame(placement).to_csv(OUT/'item3_background_placement.csv', index=False)
    (OUT/'items3_4_metadata.json').write_text(json.dumps(dict(
        run=str(run), tau_db=tau, surrogates=N_SUR, seed=P2['seed'], seed_full_redraw=SEED2,
        shuffle_block_sec=P2['shuffle_block_sec'], null_percentile=P2['null_percentile'], dilate_px=DILATE_PX,
        box_enlarge=BOX_ENLARGE, wall_time_sec=time.perf_counter() - started), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
