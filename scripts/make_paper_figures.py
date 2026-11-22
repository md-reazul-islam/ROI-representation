"""Paper figures and the expert-annotation kit from run_paper_evaluation.py outputs.

Usage (from the Module1 folder):
    python scripts/make_paper_figures.py --run runs/pilot24_paper --report-dir reports/paper

Figures (reports/paper/figures):
  fig2_qualitative.png        frame | fused box | SAM 2 mask | pose | candidate ROIs | pixel weight map | final ROIs
                              The pixel weight map is the actual per-pixel contribution of the Full representation:
                              w(p) = sum_r beta_r 1[p in R_r n M n skin] / |R_r n M n skin| (beta_r = fusion weight).
  fig3_ablation_metrics.png   per-configuration mean and 95 % bootstrap CI over videos
  fig4_q_reliability.png      reliability diagram of Q (raw and train-fitted Platt) on held-out ROI windows
  fig5_motion_strata.png      pulse-informative window rate by motion tertile
Annotation kit (reports/paper/annotation_kit): stratified held-out frames with numbered candidate ROIs and a
label template (video_id, frame_index, roi_name, valid, best) for expert-label evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from module1 import paper_eval as pe
from module2.extraction import ExtractionConfig, shrink, skin_pixels
from run_paper_evaluation import CONFIGS, NAMES, bootstrap_ci, platt

INK, MUTED, GRID, ACCENT = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
BASE = ['#d4d3cf', '#c3c2bc', '#b1b0a9', '#9f9e96', '#8d8c83', '#7b7a70']   # neutral ramp for baselines
COLORS = dict(zip(CONFIGS, BASE + [ACCENT]))
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200})


def read_frame(path, index):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    capture.release()
    return image if ok else None


def polygon_mask(shape, polygon):
    canvas = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(canvas, [np.rint(np.asarray(polygon)).astype(np.int32)], 1)
    return canvas.astype(bool)


def qualitative(run, report, figures):
    out = run/'outputs'
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    frames = pd.read_csv(report/'sampled_frame_descriptors.csv')
    candidates = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False)
    recovery = pd.read_csv(out/'recovery/recovery_results.csv', low_memory=False)
    sam2 = pd.read_csv(out/'sam2/sam2_masks.csv', low_memory=False)
    pose = pd.read_csv(out/'pose_sam2/pose_results.csv', low_memory=False)
    held = frames[(frames.split != 'train') & frames.selected_full]
    if not len(held):   # e.g. a smoke run on training videos only
        held = frames[frames.selected_full]
    picks = []
    if len(held):
        test = held[held.split == 'test']
        source = test if len(test) else held
        middle = source.iloc[len(source)//2]
        picks.append(('Typical (held-out)', middle))
        for column, label, ascending in (('motion', 'High motion', False), ('occlusion', 'High torso occlusion', False),
                                         ('luminance', 'Low light', True)):
            ranked = frames[frames.selected_full].dropna(subset=[column]).sort_values(column, ascending=ascending)
            if len(ranked):
                picks.append((label, ranked.iloc[0]))
    config = ExtractionConfig()
    titles = ['Frame', 'Fused box / track', 'SAM 2 mask', 'Pose (308 kp)', 'Candidate ROIs', 'Pixel weight map',
              'Final dynamic ROI']
    fig, axes = plt.subplots(len(picks), 7, figsize=(10.5, 2.3 * len(picks)), squeeze=False)
    for row, (label, pick) in enumerate(picks):
        video, frame = pick.video_id, int(pick.frame_index)
        image = read_frame(manifest.loc[video, 'video_path'], frame)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        box_row = recovery[(recovery.video_id == video) & (recovery.frame_index == frame)].iloc[0]
        box = pe.parse(box_row.final_infant_bbox) if isinstance(box_row.final_infant_bbox, str) else None
        mask_row = sam2[(sam2.video_id == video) & (sam2.frame_index == frame)]
        mask = None
        if len(mask_row) and isinstance(mask_row.iloc[0].mask_path, str):
            loaded = cv2.imread(str(out/'sam2'/mask_row.iloc[0].mask_path), cv2.IMREAD_GRAYSCALE)
            mask = None if loaded is None else loaded > 0
        rows = candidates[(candidates.video_id == video) & (candidates.frame_index == frame) & (candidates.roi_status == 'AVAILABLE')]
        selected = pe.selections(rows, 'Full')[frame]
        panels = [rgb.copy() for _ in range(7)]
        if box:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(panels[1], (x1, y1), (x2, y2), (42, 120, 214), 3)
        if mask is not None:
            overlay = panels[2].copy()
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * np.array([42, 120, 214])).astype(np.uint8)
            panels[2] = overlay
        pose_row = pose[(pose.video_id == video) & (pose.frame_index == frame)]
        if len(pose_row):
            path = out/'pose_sam2'/pose_row.iloc[0].keypoints_path
            if path.is_file():
                for point in json.loads(path.read_text()).get('keypoints', []):
                    if point.get('x') is not None and (point.get('confidence') or point.get('score') or 0) >= .5:
                        cv2.circle(panels[3], (int(point['x']), int(point['y'])), 2, (235, 104, 52), -1)
        for r in rows.itertuples():
            polygon = pe.parse(r.polygon)
            if polygon and len(polygon) >= 3:
                cv2.polylines(panels[4], [np.rint(np.asarray(polygon)).astype(np.int32)], True, (255, 255, 255), 1)
        weight = np.zeros(rgb.shape[:2])
        for name, beta, polygon, _ in selected:
            if polygon is None:
                continue
            region = polygon_mask(rgb.shape, shrink(np.asarray(polygon, float), config.roi_shrink))
            if mask is not None:
                region &= mask
            brightest = image.max(axis=2)
            usable = region & (brightest < config.saturation_level) & (brightest > config.dark_level)
            skin = skin_pixels(image, usable, config)
            if skin.sum():
                weight[skin] += beta / skin.sum()
        heat = plt.cm.viridis(weight / weight.max() if weight.max() > 0 else weight)[..., :3]
        dimmed = (0.45 * rgb).astype(np.uint8)
        dimmed[weight > 0] = (0.25 * rgb[weight > 0] + 0.75 * 255 * heat[weight > 0]).astype(np.uint8)
        panels[5] = dimmed
        for rank, (name, beta, polygon, _) in enumerate(selected):
            if polygon is not None:
                cv2.polylines(panels[6], [np.rint(np.asarray(polygon)).astype(np.int32)], True,
                              (42, 120, 214) if rank == 0 else (255, 255, 255), 2 if rank == 0 else 1)
        for column, panel in enumerate(panels):
            ax = axes[row, column]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[column], fontsize=7, color=INK)
        axes[row, 0].set_ylabel(f'{label}\n{video} f{frame}', fontsize=6.5, color=INK)
        axes[row, 6].set_xlabel(', '.join(f'{n} {b:.2f}' for n, b, _, _ in selected), fontsize=5.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(figures/'fig2_qualitative.png', bbox_inches='tight')
    plt.close(fig)


def ablation_bars(report, figures):
    per_video = pd.read_csv(report/'per_video_metrics.csv')
    metrics = [('temporal_iou', 'Temporal IoU ↑'), ('centroid_jitter', 'Centroid jitter ↓'),
               ('mask_support', 'Mask support ↑'), ('pulse_snr_db', 'Pulse SNR (dB) ↑'),
               ('informative_window_percent', 'Pulse-informative windows (%) ↑'), ('switches_per_sec', 'Top-1 switches / s ↓')]
    fig, axes = plt.subplots(2, 3, figsize=(9, 4.6))
    for ax, (metric, title) in zip(axes.flat, metrics):
        for i, c in enumerate(CONFIGS):
            estimate, low, high = bootstrap_ci(per_video[per_video.config == c][metric])
            if not np.isfinite(estimate):
                continue
            ax.bar(i, estimate, width=.7, color=COLORS[c], edgecolor='white', linewidth=1)
            if np.isfinite(low):
                ax.plot([i, i], [low, high], color=INK, linewidth=1)
        ax.set_xticks(range(len(CONFIGS)), CONFIGS)
        ax.set_title(title, fontsize=8, color=INK, loc='left')
        ax.grid(axis='y', color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figures/'fig3_ablation_metrics.png', bbox_inches='tight')
    plt.close(fig)


def reliability(report, figures):
    roi = pd.read_csv(report/'roi_window_q_snr.csv')
    meta = json.loads((report/'evaluation_metadata.json').read_text())
    tau = meta['thresholds']['informative_snr_db']
    part = roi[roi.config == 'B5']
    train, held = part[part.split == 'train'], part[part.split != 'train']
    if not len(train) or not len(held):
        return
    a, b = platt(train.q01, train.snr >= tau)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=.8, linestyle='--')
    for probability, label, color in ((held.q01.clip(0, 1), 'Q (raw, rescaled)', '#9f9e96'),
                                      (1 / (1 + np.exp(-(a * held.q01 + b))), 'Q + Platt (train-fitted)', ACCENT)):
        ece, brier, table = pe.calibration(probability, held.snr >= tau)
        if len(table):
            ax.plot(table.mean_confidence, table.observed_rate, marker='o', markersize=4, linewidth=2, color=color,
                    label=f'{label}: ECE {ece:.3f}')
    ax.set_xlabel('Predicted reliability'); ax.set_ylabel('Observed pulse-informative rate')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.grid(color=GRID, linewidth=.6)
    fig.tight_layout()
    fig.savefig(figures/'fig4_q_reliability.png', bbox_inches='tight')
    plt.close(fig)


def motion_plot(report, figures):
    windows = pd.read_csv(report/'window_metrics.csv')
    meta = json.loads((report/'evaluation_metadata.json').read_text())
    tau, edges = meta['thresholds']['informative_snr_db'], meta['thresholds']['motion_tertiles']
    level = pd.cut(windows.motion, [-np.inf, *edges, np.inf], labels=['low', 'medium', 'high'])
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for c in ('B0', 'B2', 'B4', 'Full'):
        values = [windows[level == name].groupby('video_id')[f'snr_{c}'].apply(lambda s: 100 * (s >= tau).mean()).mean()
                  for name in ('low', 'medium', 'high')]
        ax.plot(range(3), values, marker='o', markersize=4, linewidth=2, color=COLORS[c] if c != 'B0' else '#7b7a70',
                linestyle='-' if c == 'Full' else ('--' if c == 'B0' else ':'), label=NAMES[c])
    ax.set_xticks(range(3), ['Low', 'Medium', 'High'])
    ax.set_xlabel('Motion tertile (thresholds from training videos)')
    ax.set_ylabel('Pulse-informative windows (%)')
    ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.22), ncol=2)
    ax.grid(axis='y', color=GRID, linewidth=.6)
    fig.tight_layout()
    fig.savefig(figures/'fig5_motion_strata.png', bbox_inches='tight')
    plt.close(fig)


def annotation_kit(run, report, per_video_frames=6, seed=0):
    out = run/'outputs'
    kit = report/'annotation_kit'
    (kit/'frames').mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(run/'manifest.csv').set_index('video_id')
    frames = pd.read_csv(report/'sampled_frame_descriptors.csv')
    candidates = pd.read_csv(out/'confidence_sam2/roi_confidence.csv', low_memory=False)
    rng = np.random.default_rng(seed)
    held = frames[frames.split != 'train'].copy()
    if not len(held):
        held = frames.copy()
    held['motion_level'] = pd.qcut(held.motion.rank(method='first'), 3, labels=['low', 'medium', 'high'])
    rows = []
    for video, g in held.groupby('video_id'):
        picks = []
        for _, s in g.groupby('motion_level', observed=True):
            picks += list(s.frame_index.to_numpy()[rng.permutation(len(s))[:per_video_frames // 3]])
        for frame in sorted(picks):
            image = read_frame(manifest.loc[video, 'video_path'], int(frame))
            if image is None:
                continue
            available = candidates[(candidates.video_id == video) & (candidates.frame_index == frame) &
                                   (candidates.roi_status == 'AVAILABLE')]
            for number, r in enumerate(available.itertuples(), 1):
                polygon = pe.parse(r.polygon)
                if polygon and len(polygon) >= 3:
                    points = np.rint(np.asarray(polygon)).astype(np.int32)
                    cv2.polylines(image, [points], True, (255, 255, 255), 1)
                    x, y = points.mean(axis=0)
                    cv2.putText(image, str(number), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                rows.append(dict(video_id=video, frame_index=int(frame), roi_number=number, roi_name=r.roi_name,
                                 valid='', best=''))
            cv2.imwrite(str(kit/'frames'/f'{video}_f{int(frame):06d}.png'), image)
    pd.DataFrame(rows).to_csv(kit/'labels_template.csv', index=False)
    (kit/'README.txt').write_text(
        'Expert ROI validity labels for the held-out (validation + test) videos.\n'
        'For every numbered candidate ROI in frames/<video>_f<frame>.png fill labels_template.csv:\n'
        '  valid = 1 if the region shows usable, unoccluded infant skin/torso for physiological sensing, else 0\n'
        '  best  = 1 for the single most suitable ROI of the frame (at most one per frame)\n'
        'Label without looking at model scores. Two annotators are recommended (report Cohen kappa).\n'
        'Then: python scripts/run_paper_evaluation.py --expert-labels labels.csv\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Paper figures and annotation kit for Module 1.')
    parser.add_argument('--run', default=str(ROOT/'runs/pilot24_paper'))
    parser.add_argument('--report-dir', default=str(ROOT/'reports/paper'))
    parser.add_argument('--skip-kit', action='store_true')
    args = parser.parse_args(argv)
    run, report = Path(args.run), Path(args.report_dir)
    figures = report/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    ablation_bars(report, figures)
    reliability(report, figures)
    motion_plot(report, figures)
    qualitative(run, report, figures)
    if not args.skip_kit:
        annotation_kit(run, report)
    print(f'Wrote {figures}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
