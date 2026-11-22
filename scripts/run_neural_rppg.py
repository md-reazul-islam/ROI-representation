"""Module 2 (HR branch, v3): Module 1-guided unsupervised neural rPPG with per-neonate test-time adaptation.

  --mode pretrain  extract ROI colour grids of all selected videos and train the source model without labels
                   (e.g. on the 24-video pilot); saves --save-model (default models/neural_rppg_source.pt).
                   --exclude-manifest drops every source video whose SHA-256 is in that manifest (the target
                   videos), so the source model never sees the infant it is later adapted to.
  --mode adapt     for every selected video: NEURAL_SRC = source model as is (if --pretrained), NEURAL_TTA =
                   source adapted to this video with Module 1 guidance, NEURAL_TTA_UNGUIDED = adapted without
                   guidance (skip with --no-unguided). --variants picks any of module2.neural.TTA_VARIANTS, e.g. the
                   guidance-component ablation NEURAL_TTA_NO_QSAMPLE / _NO_QWEIGHT / _NO_SITES. Pulses are written per video for
                   run_hr_rppg.py --neural-dir, which then treats them as extra pulse methods.
No reference HR is used anywhere (see module2.neural for the losses and the guidance).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.video import VideoReader
from module1.roi import ROI_NAMES
from module2.extraction import (ExtractionConfig, INFANT_BOX, box_polygon, hold_indices, number, parse_json,
                                roi_polygon)
from module2.neural import (TTA_VARIANTS, NeuralConfig, VideoClips, adapted_copy, build_model, finalize, infer, mean_rgb_grid,
                            train)
from module2.rppg import HRConfig


def build_config(section, cls, overrides=None):
    values = {field.name: section[field.name] for field in fields(cls) if field.name in section}
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def safe_name(video):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', str(video))


def extract_grids(path, rows, recovery_rows, ex, grid):
    """Per native frame: colour grids (n, m, G, G, 3) of the ROI pool + INFANT_BOX, Q (n, m), fps."""
    meta = VideoReader(path).metadata()
    if not meta.opened_successfully or meta.native_fps <= 0:
        raise ValueError(f'Cannot decode video: {path}')
    fps = float(meta.native_fps)
    pool = list(ex.roi_pool)
    by_roi = {name: {} for name in pool}
    for row in rows:
        if row['roi_name'] in by_roi:
            by_roi[row['roi_name']][int(row['frame_index'])] = row
    processed = sorted({int(row['frame_index']) for row in rows})
    hold = hold_indices(processed, max(meta.frame_count, processed[-1] + 1 if processed else 0), fps, ex.max_hold_sec)
    boxes = {int(row['frame_index']): parse_json(row.get('final_infant_bbox')) for row in recovery_rows}
    names = pool + [INFANT_BOX]
    grids, q = [], []
    capture = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            source = int(hold[index]) if index < len(hold) else -1
            frame_grids, frame_q = [], []
            for name in names:
                if name == INFANT_BOX:
                    box = boxes.get(source) if source >= 0 else None
                    polygon, value = (box_polygon(box) if box else None), None
                else:
                    row = by_roi[name].get(source, {}) if source >= 0 else {}
                    polygon, value = (roi_polygon(row) if row else None), (number(row.get('q')) if row else None)
                cells = mean_rgb_grid(image, polygon, ex, grid) if polygon is not None else None
                frame_grids.append(cells if cells is not None else np.full((grid, grid, 3), np.nan))
                frame_q.append(np.nan if value is None else value)
            grids.append(frame_grids)
            q.append(frame_q)
            index += 1
    finally:
        capture.release()
    return np.asarray(grids, np.float32), np.asarray(q, np.float32), fps, names


def load_videos(records, manifest, root, by_video, recovery_by_video, ex, nc, cache_dir):
    videos = []
    for _, record in records.iterrows():
        video = str(record.video_id)
        if video not in by_video:
            print(f'{video}: no Module 1 ROIs, skipped', flush=True)
            continue
        cache = cache_dir/f'{safe_name(video)}_grids.npz'
        if cache.is_file():
            data = np.load(cache, allow_pickle=False)
            grids, q, fps, names = data['grids'], data['q'], float(data['fps']), list(data['names'])
        else:
            start = time.time()
            grids, q, fps, names = extract_grids(manifest.resolve_video_path(root, record), by_video[video],
                                                 recovery_by_video.get(video, []), ex, nc.grid)
            np.savez_compressed(cache, grids=grids, q=q, fps=fps, names=np.array(names))
            print(f'{video}: {len(grids)} frames, colour grids extracted in {time.time()-start:.0f} s', flush=True)
        videos.append((video, str(record.split), VideoClips(video, grids, q, names, fps, nc)))
    return videos


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 2 v3: Module 1-guided unsupervised neural rPPG with '
                                                 'per-neonate test-time adaptation.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--confidence-dir', help='Module 1 Phase 10 output (default: confidence.output_dir)')
    parser.add_argument('--recovery', help='Phase 6 recovery_results.csv (default: recovery.output_dir)')
    parser.add_argument('--output-dir', default='outputs/neural_rppg')
    parser.add_argument('--mode', choices=('adapt', 'pretrain'), default='adapt')
    parser.add_argument('--pretrained', help='adapt: source checkpoint (without it, adaptation starts from random '
                                             'weights and NEURAL_SRC is not produced)')
    parser.add_argument('--save-model', default='models/neural_rppg_source.pt', help='pretrain: checkpoint path')
    parser.add_argument('--exclude-manifest', help='pretrain: leave out source videos whose sha256 appears in this '
                                                   'manifest (the target videos)')
    parser.add_argument('--steps', type=int, help='pretrain steps (default 3000) or adaptation steps (default 300)')
    parser.add_argument('--no-unguided', action='store_true', help='adapt: skip the unguided ablation variant')
    parser.add_argument('--variants', nargs='+', choices=tuple(TTA_VARIANTS),
                        help='adapt: adaptation variants (default NEURAL_TTA NEURAL_TTA_UNGUIDED)')
    parser.add_argument('--cache-dir', help='colour-grid cache shared between runs on the same Module 1 output '
                                            '(default <output-dir>/cache)')
    parser.add_argument('--split', choices=('train', 'validation', 'test', 'all'), default='all')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)

    import torch
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    cfg = load_config(args.config)
    root = Path(cfg['_project_root'])
    hr = build_config(cfg.get('hr', {}), HRConfig)
    ex = build_config(cfg.get('hr', {}), ExtractionConfig, dict(roi_pool=tuple(ROI_NAMES)))
    overrides = {}
    if args.steps:
        overrides['pretrain_steps' if args.mode == 'pretrain' else 'tta_steps'] = args.steps
    nc = build_config(cfg.get('neural_rppg', {}), NeuralConfig, overrides)
    confidence_dir = project_path(cfg, args.confidence_dir or cfg.get('confidence', {}).get('output_dir', 'outputs/confidence'))
    recovery_path = project_path(cfg, args.recovery or
                                 str(Path(cfg.get('recovery', {}).get('output_dir', 'outputs/recovery'))/'recovery_results.csv'))
    output = project_path(cfg, args.output_dir)
    for path in (confidence_dir/'roi_confidence.csv', recovery_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Run Module 1 (Phases 6-10) first.')
    if args.mode == 'adapt' and output.exists() and any(output.glob('*_neural_pulses.npz')):
        raise FileExistsError(f'{output} already has neural pulses; use --output-dir for a new experiment')

    manifest = DatasetManifest.load(project_path(cfg, args.manifest))
    records = manifest.df
    if args.videos:
        records = records[records.video_id.isin(args.videos)]
    elif args.split != 'all':
        records = records[records.split == args.split]
    excluded = []
    if args.exclude_manifest:
        targets = set(pd.read_csv(project_path(cfg, args.exclude_manifest)).sha256.dropna().astype(str))
        excluded = sorted(records.video_id[records.sha256.astype(str).isin(targets)].astype(str))
        records = records[~records.sha256.astype(str).isin(targets)]
        print(f'Left out (target videos): {", ".join(excluded) or "none"}', flush=True)
    columns = ['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'bbox', 'q']
    confidence = pd.read_csv(confidence_dir/'roi_confidence.csv', usecols=lambda c: c in columns, low_memory=False)
    recovery = pd.read_csv(recovery_path, usecols=lambda c: c in ('video_id', 'frame_index', 'final_infant_bbox'), low_memory=False)
    by_video = {str(v): g.to_dict('records') for v, g in confidence.groupby('video_id')}
    recovery_by_video = {str(v): g.to_dict('records') for v, g in recovery.groupby('video_id')}
    cache_dir = project_path(cfg, args.cache_dir) if args.cache_dir else output/'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    videos = load_videos(records, manifest, root, by_video, recovery_by_video, ex, nc, cache_dir)
    if not videos:
        raise ValueError('No videos with Module 1 ROIs selected')
    band = hr.band_hz
    base_meta = dict(neural_config=asdict(nc), band_hz=list(band), device=device, torch_version=torch.__version__,
                     python_version=platform.python_version(), timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     inputs=dict(confidence=str(confidence_dir/'roi_confidence.csv'), recovery=str(recovery_path)),
                     reference_hr_used=False)

    if args.mode == 'pretrain':
        start = time.time()
        model = build_model(nc)
        history = train(model, [v for _, _, v in videos], nc, band, nc.pretrain_steps, nc.pretrain_lr, True, device)
        target = project_path(cfg, args.save_model)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), neural_config=asdict(nc), band_hz=list(band),
                        source_videos=[v for v, _, _ in videos], excluded_target_videos=excluded), target)
        pd.DataFrame(history).to_csv(output/'pretrain_history.csv', index=False)
        (output/'pretrain_metadata.json').write_text(json.dumps(dict(base_meta, phase='MODULE_2_NEURAL_PRETRAIN',
                                                                     checkpoint=str(target), videos=[v for v, _, _ in videos],
                                                                     excluded_target_videos=excluded,
                                                                     minutes=(time.time()-start)/60), indent=2))
        print(f'Source model: {len(videos)} videos, {nc.pretrain_steps} steps, final loss {history[-1]["loss"]:.3f}, '
              f'{(time.time()-start)/60:.1f} min -> {target}')
        return 0

    source = build_model(nc)
    pretrained = None
    if args.pretrained:
        pretrained = project_path(cfg, args.pretrained)
        state = torch.load(pretrained, map_location='cpu', weights_only=False)
        if state['neural_config']['grid'] != nc.grid or state['neural_config']['width'] != nc.width:
            raise ValueError('Checkpoint grid/width differ from the neural_rppg configuration')
        source.load_state_dict(state['state_dict'])
    histories, infos = [], []
    for video, split, clips in videos:
        start = time.time()
        pulses = {}
        if pretrained is not None:
            pulses['NEURAL_SRC'] = finalize(infer(source, clips, nc, device), clips.native_fps, hr)
        names = args.variants or ['NEURAL_TTA'] + ([] if args.no_unguided else ['NEURAL_TTA_UNGUIDED'])
        try:
            for name in names:
                guided, overrides = TTA_VARIANTS[name]
                variant_config = replace(nc, **overrides)
                model, history = adapted_copy(source, clips, variant_config, band, guided, device)
                pulses[name] = finalize(infer(model, clips, variant_config, device), clips.native_fps, hr)
                histories += [dict(row, video_id=video, variant=name) for row in history]
        except ValueError as error:
            print(f'{video}: adaptation skipped ({error})', flush=True)
        np.savez_compressed(output/f'{safe_name(video)}_neural_pulses.npz', names=np.array(clips.names),
                            fps=clips.native_fps, methods=np.array(list(pulses)),
                            **{name: values.astype(np.float32) for name, values in pulses.items()})
        infos.append(dict(video_id=video, split=split, frames=clips.native_frames, fps=clips.native_fps,
                          methods=list(pulses), minutes=(time.time()-start)/60))
        print(f'{video}: {", ".join(pulses) or "no pulses"} in {(time.time()-start)/60:.1f} min', flush=True)
    pd.DataFrame(histories).to_csv(output/'adaptation_history.csv', index=False)
    (output/'experiment_metadata.json').write_text(json.dumps(dict(
        base_meta, phase='MODULE_2_NEURAL_ADAPT', run_status='COMPLETE',
        pretrained=None if pretrained is None else dict(path=str(pretrained), sha256=sha256_file(pretrained)),
        methods=sorted({m for info in infos for m in info['methods']}), video_info=infos,
        guidance='Q-softmax clip sampling, Q-weighted bandwidth+sparsity loss, cross-site PSD consistency',
        losses='SiNC (Speth et al., CVPR 2023): bandwidth, sparsity, variance; speed augmentation'), indent=2, default=str))
    print(f'Done: {len(infos)} video(s) -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
