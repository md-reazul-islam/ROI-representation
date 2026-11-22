"""Phase 2: stronger baselines and label-free validity tests, on the same pixels, pulse extraction and windows
as Phase 1.

Configurations
  B0, B1, Full, D3, D4, D5   recomputed here (B0/Full/D3 must reproduce the Phase 1 window SNRs; checked)
  M0  SAM 2 mask n adaptive skin over the whole infant box; no anatomy, no scoring
  M1  anatomical ROIs (mask n skin pixels), top-3 by signal quality SQI = sigmoid(SNR) x (1 - motion) of the
      PREVIOUS non-overlapping 8 s window (causal; Villarroel-style skin + SQI selection)
  M2  4 x 4 grid tiles of the infant box (mask n skin), top-3 by SNR of the previous non-overlapping window
      (causal; Kiddle-style tile selection)
  M3  ORACLE upper bound: the single anatomical ROI with the best SNR in the SAME window (not a method)
  BG  NEGATIVE CONTROL (not a method): the four frame-corner patches (each 20 % of width x height), background
      pixels only (outside the SAM 2 mask dilated by 15 px, else outside the infant box enlarged by 10 %: the same
      background definition as the camera-motion estimate), no skin filter; same POS pulse, windows and surrogate
      test. A valid pulse-present test passes ~5 % (the null rate) of background windows. Left / right halves =
      left / right corners. Excluded from the paired-test families.

Tests (per analysis window, per configuration)
  pulse-present  SNR > 95th percentile of the SNRs of surrogate traces of the same pixels. Primary null:
                 independent per-channel Fourier phase randomization (keeps each channel's spectrum, destroys
                 the cross-channel pulse coherence POS relies on). Secondary null: 0.25 s block shuffling.
                 Colour-dependent selection (M1-M3) is re-run on every surrogate, so selection gets no free win.
  agreement      each selected region is split at its own vertical midline into two disjoint halves; the two
                 half-fusions must give heart rates within +-6 bpm.

Usage: python module1_v2/scripts/run_phase2.py --run runs/final34_paper --v1-report reports/paper_final34_paper
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
for path in (V2/'src', ROOT/'src', ROOT/'scripts', V2/'scripts'):
    sys.path.insert(0, str(path))

from module1 import paper_eval as pe                                         # noqa: E402  (v1, read-only)
from module2.ablation import bootstrap_ci, holm, paired_test                 # noqa: E402
from module2.extraction import hold_indices, mean_rgb                        # noqa: E402
from module2.rppg import _usable, analysis_windows, hr_fft, signal_quality, window_snr  # noqa: E402
from run_paper_evaluation import EXTRACT, HR, boxes_of, load_mask, pulse     # noqa: E402  (v1 helpers)
from run_v2_evaluation import CAND_COLUMNS, configurations                   # noqa: E402
from module1_v2 import baselines as bl, selection as sel                     # noqa: E402
from module1_v2.motion import background_mask                               # noqa: E402

CFG = yaml.safe_load((V2/'configs/module1_v2.yaml').read_text())
P2 = CFG['phase2']
WEIGHTED = ('Full', 'I-B', 'D3', 'D4', 'D5')
CONFIGS = ('B0', 'B1') + WEIGHTED + ('M0', 'M1', 'M2', 'M3', 'BG')
CORNERS = ('TL', 'TR', 'BL', 'BR')                 # BG patches; L = TL + BL, R = TR + BR
CORNER_FRACTION = .2                               # patch size as a fraction of frame width / height
BG_MIN_PIXELS = 200                                # background pixels needed in a patch (else NaN)
NOT_PAIRED = ('M3', 'BG')                          # oracle and negative control: not candidates in the paired tests
# ROI traces: 'roi' = v2 pixels (2-point boxes read as rectangles; used by I-B, D3-D5, M1, M3),
#             'roiv1' = v1 pixels (2-point boxes carry none; used by Full only, so Full stays = v1)
METRICS = dict(pulse_snr_db=2, informative_window_percent=1, present_phase_percent=1, present_shuffle_percent=1,
               video_p_phase=3, agreement_percent=1, agreement_testable_percent=1)
SIDES = ('', 'L', 'R')


def window_hr(x, fs):
    x = _usable(x, HR)
    return None if x is None else hr_fft(x, fs, HR)


def finite(value):
    return value is not None and math.isfinite(value)


class Units:
    """Index of every colour trace: (kind, name, side)."""

    def __init__(self, names, tiles):
        self.keys = ([('roi', r, s) for r in names for s in SIDES] + [('roiv1', r, s) for r in names for s in SIDES]
                     + [('fix', f, s) for f in ('B0', 'B1') for s in SIDES]
                     + [('m0', 'box', s) for s in SIDES] + [('tile', t, '') for t in range(tiles)]
                     + [('bg', c, '') for c in CORNERS])
        self.index = {k: i for i, k in enumerate(self.keys)}

    def __getitem__(self, key):
        return self.index[key]

    def columns(self, kind, side=''):
        return [i for (k, _, s), i in self.index.items() if k == kind and s == side]


def extract(record, out, cand, cand0, frames, recovery, sam2_rows, units, names):
    """Per-native-frame mean skin colour of every unit (same pixel rules as v1 'm1' / Phase 1)."""
    fps = float(record['fps'])
    n = int(record['frame_count'])
    boxes = boxes_of(recovery)
    sam2_rows = sam2_rows.set_index('frame_index') if len(sam2_rows) else sam2_rows
    reliable = {}
    for f in frames:
        if len(sam2_rows) and f in sam2_rows.index:
            mask = load_mask(out/'sam2', sam2_rows.loc[f])
            if mask is not None:
                reliable[f] = mask
    fixed = pe.fixed_rois(cand0, boxes)
    fixed_poly = {'B0': fixed[0], 'B1': fixed[1][1] if fixed[1] else None}
    hold = hold_indices(frames, n, fps, EXTRACT.max_hold_sec)
    by_frame = {f: rows for f, rows in cand[cand.roi_status == 'AVAILABLE'].groupby('frame_index')}
    rgb = np.full((n, len(units.keys), 3), np.nan)
    tile_column = {}

    def put(image, polygon, key, box_ok=False):
        # v1 skips 2-point (BBOX) candidates for ROI traces but not for the fixed B0 / B1 geometry
        if polygon is None or len(polygon) < (2 if box_ok else 3):
            return
        value, _ = mean_rgb(image, np.asarray(polygon, float), EXTRACT)
        if value is not None:
            rgb[i, units[key]] = value

    def put_halves(image, polygon, kind, name):
        put(image, polygon, (kind, name, ''), box_ok=kind == 'fix')
        if polygon is not None and len(polygon) >= 3:
            for side in ('L', 'R'):
                put(image, bl.clip_half(polygon, side), (kind, name, side))

    background = {}                                         # per sampled frame: background pixels (cached)

    def put_background(image, source):
        if source not in background:
            background[source] = background_mask(image.shape[:2], reliable.get(source), boxes.get(source),
                                                 CFG['phase1']['background_dilation_px'])
        bgm = background[source]
        if bgm is None:                                     # no infant localization: background unknown
            return
        h, w = image.shape[:2]
        ph, pw = int(round(h * CORNER_FRACTION)), int(round(w * CORNER_FRACTION))
        for name, (y0, x0) in zip(CORNERS, ((0, 0), (0, w - pw), (h - ph, 0), (h - ph, w - pw))):
            crop = image[y0:y0 + ph, x0:x0 + pw]
            brightest = crop.max(axis=2)
            ok_px = bgm[y0:y0 + ph, x0:x0 + pw] & (brightest < EXTRACT.saturation_level) & (brightest > EXTRACT.dark_level)
            if ok_px.sum() >= BG_MIN_PIXELS:
                b, g, r = crop[ok_px].astype(float).mean(axis=0)
                rgb[i, units[('bg', name, '')]] = (r, g, b)

    capture = cv2.VideoCapture(str(record['video_path']))
    for i in range(n):
        ok, image = capture.read()
        if not ok:
            break
        for name, polygon in fixed_poly.items():            # v1 B0 / B1 use the unmasked image
            put_halves(image, polygon, 'fix', name)
        source = int(hold[i])
        if source < 0:
            continue
        put_background(image, source)                        # unmasked image, background pixels only
        mask = reliable.get(source)
        masked = image
        if mask is not None and mask.shape == image.shape[:2]:
            masked = image.copy()
            masked[~mask] = 0                                # below dark_level: excluded by mean_rgb
        box = boxes.get(source)
        if box is not None:
            put_halves(masked, bl.box_polygon(box), 'm0', 'box')
            for t, (polygon, column) in enumerate(bl.grid_tiles(box, P2['grid_rows'], P2['grid_cols'])):
                put(masked, polygon, ('tile', t, ''))
                tile_column[t] = column
        rows = by_frame.get(source)
        if rows is None:
            continue
        for row in rows.itertuples():
            polygon = pe.parse(row.polygon)
            put_halves(masked, sel.pixel_polygon(polygon, 'rectangle'), 'roi', row.roi_name)
            if polygon and len(polygon) >= 3:          # v1 pixels: identical where the geometry is a real polygon
                for side in SIDES:
                    rgb[i, units[('roiv1', row.roi_name, side)]] = rgb[i, units[('roi', row.roi_name, side)]]
    capture.release()
    return rgb, hold, tile_column


def roi_kind(cid):
    return 'roiv1' if cid == 'Full' else 'roi'


def weighted_traces(pulses, units, names, weights, kind='roi'):
    """Fused trace of a weight-based configuration plus its two half-fusions (same weights)."""
    return {side: pe.fused_pulse(pulses[:, [units[(kind, r, side)] for r in names]], weights) for side in SIDES}


def config_windows(cid, pulses, units, names, windows, fps, weights, motion, lag, tile_left, observed=True):
    """Per-window (snr, hr, hr_L, hr_R) of one configuration; halves only when `observed`."""
    out = []
    if cid == 'BG':                                          # negative control: equal-weight mean of the corner patches
        cols = {'': [units[('bg', c, '')] for c in CORNERS], 'L': [units[('bg', c, '')] for c in ('TL', 'BL')],
                'R': [units[('bg', c, '')] for c in ('TR', 'BR')]}
        for a, b in windows:
            fused = bl.window_mean(pulses, a, b, cols[''])
            row = [window_snr(fused, fps, HR)]
            if observed:
                row += [window_hr(fused, fps), window_hr(bl.window_mean(pulses, a, b, cols['L']), fps),
                        window_hr(bl.window_mean(pulses, a, b, cols['R']), fps)]
            out.append(row)
        return out
    if cid in ('B0', 'B1', 'M0') or cid in WEIGHTED:
        if cid in WEIGHTED:
            traces = weighted_traces(pulses, units, names, weights[cid], roi_kind(cid)) if observed else \
                {'': pe.fused_pulse(pulses[:, [units[(roi_kind(cid), r, '')] for r in names]], weights[cid])}
        else:
            kind, name = ('m0', 'box') if cid == 'M0' else ('fix', cid)
            traces = {s: pulses[:, units[(kind, name, s)]] for s in (SIDES if observed else ('',))}
        for a, b in windows:
            row = [window_snr(traces[''][a:b], fps, HR)]
            if observed:
                row += [window_hr(traces[''][a:b], fps), window_hr(traces['L'][a:b], fps), window_hr(traces['R'][a:b], fps)]
            out.append(row)
        return out
    # window-selected configurations
    kind = 'tile' if cid == 'M2' else 'roi'
    cols = units.columns(kind)
    snr = np.array([[window_snr(pulses[a:b, j], fps, HR) or np.nan for j in cols] for a, b in windows], float)
    snr[~np.isfinite(snr)] = np.nan
    if cid == 'M1':
        score = np.array([[signal_quality(s, m, HR) if np.isfinite(s) else np.nan for s, m in zip(srow, mrow)]
                          for srow, mrow in zip(snr, motion)], float)
        choice = bl.causal_selection(score, lag, P2['top_k'])
    elif cid == 'M2':
        choice = bl.causal_selection(snr, lag, P2['top_k'])
    else:
        choice = bl.oracle_selection(snr)
    for (a, b), units_w in zip(windows, choice):
        picked = bl.units_or_all(units_w, cols)          # no earlier window: all units of THIS kind
        fused = bl.window_mean(pulses, a, b, picked)
        row = [window_snr(fused, fps, HR)]
        if observed:
            if kind == 'roi':
                names_w = [units.keys[j][1] for j in picked]
                half = {s: bl.window_mean(pulses, a, b, [units[('roi', r, s)] for r in names_w]) for s in ('L', 'R')}
            else:
                left = [j for j in picked if tile_left[units.keys[j][1]]]
                right = [j for j in picked if not tile_left[units.keys[j][1]]]
                half = {'L': bl.window_mean(pulses, a, b, left) if left else np.full(b - a, np.nan),
                        'R': bl.window_mean(pulses, a, b, right) if right else np.full(b - a, np.nan)}
            row += [window_hr(fused, fps), window_hr(half['L'], fps), window_hr(half['R'], fps)]
        out.append(row)
    return out


def evaluate_video(job):
    video, record, out, cand, cand0, frames, recovery, sam2_rows, configs, tau, seed = job
    started = time.perf_counter()
    out = Path(out)
    fps = float(record['fps'])
    duration = float(record['duration_sec'])
    fps_sampled = len(frames) / duration if duration > 0 else np.nan
    names = sel.NAMES
    units = Units(names, P2['grid_rows'] * P2['grid_cols'])
    rgb, hold, tile_column = extract(record, out, cand, cand0, frames, recovery, sam2_rows, units, names)
    tile_left = {t: c < P2['grid_cols'] / 2 for t, c in tile_column.items()}
    for t in range(P2['grid_rows'] * P2['grid_cols']):
        tile_left.setdefault(t, t % P2['grid_cols'] < P2['grid_cols'] / 2)
    n = len(rgb)
    pulses = np.column_stack([pulse(rgb[:, j], fps) for j in range(rgb.shape[1])])
    windows = analysis_windows(n, fps, HR)
    lag = max(1, int(round(HR.estimation_window_sec / HR.estimation_step_sec)))

    sel_frames = sorted(cand.frame_index.unique())
    weights = {}
    for cid in WEIGHTED:
        flags = configs['D0' if cid == 'Full' else cid]
        _, fusion, _, _ = sel.select_video(cand, sel_frames, flags, fps_sampled, CFG['phase1']['top_k'],
                                           CFG['phase1']['temperature'], None)
        weights[cid] = pe.native_weights(fusion, names, hold, n)
    # per-ROI motion per window (for the M1 SQI), from the sampled frames inside the window
    motion_tab = cand[cand.roi_status == 'AVAILABLE'].pivot_table(index='frame_index', columns='roi_name',
                                                                  values='motion_magnitude')
    motion = np.full((len(windows), len(names)), np.nan)
    for w, (a, b) in enumerate(windows):
        part = motion_tab[(motion_tab.index >= a) & (motion_tab.index < b)]
        if len(part):
            motion[w] = part.reindex(columns=names).mean().to_numpy(float)

    rows = [dict(video_id=video, split=record['split'], window=w) for w in range(len(windows))]
    observed = {}
    for cid in CONFIGS:
        result = config_windows(cid, pulses, units, names, windows, fps, weights, motion, lag, tile_left)
        observed[cid] = np.array([r[0] if finite(r[0]) else np.nan for r in result], float)
        for w, (snr, hr, hl, hrr) in enumerate(result):
            rows[w][f'snr_{cid}'] = snr
            rows[w][f'hr_{cid}'] = hr
            rows[w][f'agree_{cid}'] = bl.agree(hl, hrr, P2['agreement_bpm'])

    # surrogate nulls
    null = {kind: {c: [] for c in CONFIGS} for kind in ('phase', 'shuffle')}
    null_median = {c: [] for c in CONFIGS}
    full_units = [i for k, i in units.index.items() if k[2] == '']
    for kind, count in (('phase', P2['surrogates_phase']), ('shuffle', P2['surrogates_shuffle'])):
        for s in range(count):
            rng = np.random.default_rng([seed, zlib.crc32(video.encode()), s, kind == 'phase'])
            sur = np.full_like(pulses, np.nan)
            for j in full_units:
                trace = (bl.phase_surrogate(rgb[:, j], rng) if kind == 'phase' else
                         bl.shuffle_surrogate(rgb[:, j], rng, max(1, int(round(P2['shuffle_block_sec'] * fps)))))
                sur[:, j] = pulse(trace, fps)
            for cid in CONFIGS:
                vals = np.array([r[0] if finite(r[0]) else np.nan for r in
                                 config_windows(cid, sur, units, names, windows, fps, weights, motion, lag, tile_left,
                                                observed=False)], float)
                null[kind][cid].append(vals)
                if kind == 'phase':
                    null_median[cid].append(np.nanmedian(vals) if np.isfinite(vals).any() else np.nan)

    per_video = []
    for cid in CONFIGS:
        snr = observed[cid]
        good = np.isfinite(snr)
        present = {k: bl.present(snr, np.concatenate(null[k][cid]) if null[k][cid] else [], P2['null_percentile'])
                   for k in null}
        agreement = [rows[w][f'agree_{cid}'] for w in range(len(windows))]
        testable = [a for a in agreement if a is not None]
        median = float(np.median(snr[good])) if good.any() else np.nan
        for w in range(len(windows)):
            rows[w][f'present_phase_{cid}'] = bool(present['phase'][w])
            rows[w][f'present_shuffle_{cid}'] = bool(present['shuffle'][w])
        per_video.append(dict(
            video_id=video, split=record['split'], config=cid, windows=len(windows),
            pulse_snr_db=median,
            informative_window_percent=100. * np.mean(np.where(good, snr, -np.inf) >= tau) if len(snr) else np.nan,
            present_phase_percent=100. * present['phase'].mean() if len(windows) else np.nan,
            present_shuffle_percent=100. * present['shuffle'].mean() if len(windows) else np.nan,
            video_p_phase=bl.empirical_p(median, null_median[cid]),
            agreement_percent=100. * np.mean(testable) if testable else np.nan,
            agreement_testable_percent=100. * len(testable) / len(windows) if len(windows) else np.nan))
    return per_video, rows, time.perf_counter() - started


def mean_sd(values, digits):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 'n/a'
    return f'{values.mean():.{digits}f} ± {values.std(ddof=1) if len(values) > 1 else 0.:.{digits}f}'


def summary(per_video):
    rows = []
    for c in CONFIGS:
        part = per_video[per_video.config == c]
        row = dict(config=c, videos=int(part.video_id.nunique()))
        for m, d in METRICS.items():
            _, low, high = bootstrap_ci(part[m])
            row[m] = mean_sd(part[m], d)
            row[f'{m}_ci95'] = f'[{low:.{d}f}, {high:.{d}f}]' if math.isfinite(low) else 'n/a'
        row['videos_pulse_present_p05'] = int((part.video_p_phase < .05).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired(per_video, reference):
    rows = []
    for m in METRICS:
        if m == 'video_p_phase':
            continue
        wide = per_video.pivot_table(index='video_id', columns='config', values=m)
        block = [dict(metric=m, baseline=reference, candidate=c, **paired_test(wide[reference], wide[c]))
                 for c in CONFIGS if c != reference and c in wide and reference in wide and c not in NOT_PAIRED]
        for row, adjusted in zip(block, holm([r['p_value'] for r in block])):
            row['p_holm'] = adjusted
        rows += block
    return pd.DataFrame(rows)


def write_tables(per_video, target):
    per_video.to_csv(target/'per_video_metrics.csv', index=False)
    summary(per_video).to_csv(target/'table_phase2_main.csv', index=False)
    for split in ('train', 'validation', 'test'):
        part = per_video[per_video.split == split]
        if len(part):
            summary(part).to_csv(target/f'table_phase2_{split}.csv', index=False)
    paired(per_video, 'Full').to_csv(target/'paired_vs_Full.csv', index=False)
    paired(per_video, 'B0').to_csv(target/'paired_vs_B0.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 1 v2 Phase 2 evaluation.')
    parser.add_argument('--run', required=True)
    parser.add_argument('--v1-report', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--out-name', default='phase2')
    parser.add_argument('--resummarize', action='store_true', help='only rebuild tables from saved CSVs')
    parser.add_argument('--only', default='', help='comma-separated video IDs to recompute and merge into saved CSVs')
    args = parser.parse_args(argv)
    run = (ROOT/args.run).resolve()
    v1 = (ROOT/args.v1_report).resolve()
    work = V2/'runs'/run.name
    target = V2/'reports'/run.name/args.out_name
    target.mkdir(parents=True, exist_ok=True)
    out = run/'outputs'
    configs = configurations(json.loads((work/'tuning_log.json').read_text())['chosen'])
    tau = json.loads((v1/'evaluation_metadata.json').read_text())['thresholds']['informative_snr_db']
    manifest = pd.read_csv(run/'manifest.csv')
    if args.max_videos:
        manifest = manifest.head(args.max_videos)
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
    if args.resummarize:
        per_video = pd.read_csv(target/'per_video_metrics.csv')
        windows = pd.read_csv(target/'window_metrics.csv')
        for cid in CONFIGS:   # v1 definition: share of ALL windows with SNR >= tau (no SNR = not informative)
            rate = windows.groupby('video_id')[f'snr_{cid}'].apply(lambda x: 100. * (x >= tau).mean())
            rows = per_video.config == cid
            per_video.loc[rows, 'informative_window_percent'] = per_video.loc[rows, 'video_id'].map(rate)
        write_tables(per_video, target)
        print(pd.read_csv(target/'table_phase2_main.csv')[['config', 'videos'] + list(METRICS)].to_string(index=False))
        return 0
    jobs = []
    for r in manifest.itertuples():
        c, c0 = cand[cand.video_id == r.video_id], cand0[cand0.video_id == r.video_id]
        frames = sorted(set(c0.frame_index) | set(c.frame_index))
        jobs.append((r.video_id, r._asdict(), str(out), c, c0, frames, recovery[recovery.video_id == r.video_id],
                     sam2[sam2.video_id == r.video_id], configs, tau, P2['seed']))
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(evaluate_video, jobs))
    per_video = pd.DataFrame([row for r in results for row in r[0]])
    windows = pd.DataFrame([row for r in results for row in r[1]])
    if only:   # merge the recomputed videos into the saved full-run tables
        old_pv, old_w = pd.read_csv(target/'per_video_metrics.csv'), pd.read_csv(target/'window_metrics.csv')
        per_video = pd.concat([old_pv[~old_pv.video_id.isin(only)], per_video]).sort_values(['video_id']).reset_index(drop=True)
        windows = pd.concat([old_w[~old_w.video_id.isin(only)], windows]).sort_values(['video_id', 'window']).reset_index(drop=True)
        meta_old = json.loads((target/'phase2_metadata.json').read_text())
        meta_old.setdefault('recomputed', []).append(dict(videos=only, reason='fixed B1 geometry: 2-point boxes as in v1'))
    windows.to_csv(target/'window_metrics.csv', index=False)
    write_tables(per_video, target)

    # Regression against Phase 1 / v1: identical pixels and windows must give identical window SNRs.
    regression, missing = {}, {}
    p1 = V2/'reports'/run.name/'phase1/window_metrics_v2.csv'
    if p1.is_file():
        ref = pd.read_csv(p1).merge(windows, on=['video_id', 'window'], suffixes=('_p1', ''))
        for cid, col in (('Full', 'snr_D0'), ('I-B', 'snr_I-B'), ('D3', 'snr_D3'), ('D4', 'snr_D4'), ('D5', 'snr_D5')):
            a, b = ref[f'{col}_p1' if f'{col}_p1' in ref else col].to_numpy(float), ref[f'snr_{cid}'].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            regression[cid] = float(np.max(np.abs(a[both] - b[both]))) if both.any() else None
            missing[cid] = int((np.isfinite(a) != np.isfinite(b)).sum())   # windows present in only one of the two
    v1w = v1/'window_metrics.csv'
    if v1w.is_file():
        ref = pd.read_csv(v1w, usecols=['video_id', 'window', 'snr_B0', 'snr_B1']).merge(
            windows[['video_id', 'window', 'snr_B0', 'snr_B1']], on=['video_id', 'window'], suffixes=('_v1', ''))
        for cid in ('B0', 'B1'):
            a, b = ref[f'snr_{cid}_v1'].to_numpy(float), ref[f'snr_{cid}'].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            regression[cid] = float(np.max(np.abs(a[both] - b[both]))) if both.any() else None
            missing[cid] = int((np.isfinite(a) != np.isfinite(b)).sum())   # windows present in only one of the two
    meta = dict(run=str(run), v1_report=str(v1), tau_db=tau, phase2=P2, configs=CONFIGS,
                regression_max_abs_window_snr_diff=regression,
                regression_window_presence_mismatch=missing, wall_time_sec=time.perf_counter() - started,
                per_video_sec={j[0]: r[2] for j, r in zip(jobs, results)})
    if only:
        meta = dict(meta_old, regression_max_abs_window_snr_diff=regression, regression_window_presence_mismatch=missing,
                    per_video_sec={**meta_old.get('per_video_sec', {}), **meta['per_video_sec']})
    (target/'phase2_metadata.json').write_text(json.dumps(meta, indent=2, default=str))
    print('Regression (max |window SNR diff|):', regression, '| windows present in only one:', missing)
    print(pd.read_csv(target/'table_phase2_main.csv')[['config', 'videos'] + [m for m in METRICS]].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
