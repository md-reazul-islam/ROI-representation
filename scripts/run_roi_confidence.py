from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import cv2
import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.confidence import ConfidenceConfig, ConfidenceMechanism
from module1.confidence.inputs import load_quality_table
from module1.data.manifest import DatasetManifest, sha256_file
from module1.data.sampler import FrameSampler
from module1.data.video import VideoReader
from module1.logging_utils import setup_logging
from module1.roi import ROI_NAMES


STRATEGIES = ('top1', 'top2', 'topk', 'confidence_weighted_fusion')


def parse_json(value, default=None):
    if value is None or value == '':
        return default
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


BEST_COLOR = (0, 0, 255)  # BGR red: reserved for the highest-confidence (top-1) ROI
SELECTED_COLOR = (40, 220, 70)
OTHER_COLOR = (200, 170, 120)


def put_label(canvas, text, origin, scale, color, thickness, background=None):
    """Outlined (or boxed) text kept fully inside the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(min(max(origin[0], 3), canvas.shape[1] - width - 3))
    y = int(min(max(origin[1], height + 3), canvas.shape[0] - baseline - 3))
    if background is not None:
        cv2.rectangle(canvas, (x-3, y-height-4), (x+width+3, y+baseline+2), background, -1)
    else:
        cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), thickness+3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_confidence(image, rows, selected):
    canvas = image.copy()
    size = max(canvas.shape[:2]) / 640  # text and lines grow with the frame resolution
    text, thick = .55 * size, max(1, int(round(1.5 * size)))
    scored = sorted((row for row in rows if row['roi_status'] == 'AVAILABLE' and row.get('q') is not None),
                    key=lambda row: (-row['q'], row['roi_name']))
    best = scored[0]['roi_name'] if scored else None
    rank = {row['roi_name']: index for index, row in enumerate(scored, 1)}
    # Draw non-selected first, then selected, then the best ROI on top.
    order = sorted((row for row in rows if row['roi_status'] == 'AVAILABLE' and row.get('bbox')),
                   key=lambda row: (row['roi_name'] == best, row['roi_name'] in selected))
    for row in order:
        name, q, alpha = row['roi_name'], row.get('q'), row.get('alpha')
        polygon = parse_json(row.get('polygon'), [])
        points = np.rint(np.asarray(polygon, dtype=float)).astype(np.int32)
        if len(points) < 3:
            x1, y1, x2, y2 = parse_json(row.get('bbox'))
            points = np.rint([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).astype(np.int32)
        q_text = 'NA' if q is None else f'{q:.2f}'
        a_text = 'NA' if alpha is None else f'{alpha:.2f}'
        centroid = parse_json(row.get('centroid')) or points.mean(axis=0).tolist()
        origin = (centroid[0] - 40 * size, centroid[1])
        if name == best:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [points], BEST_COLOR)
            canvas[:] = cv2.addWeighted(overlay, .3, canvas, .7, 0)
            cv2.polylines(canvas, [points], True, BEST_COLOR, thick + 2, cv2.LINE_AA)
            put_label(canvas, f'BEST #1 {name} Q={q_text} a={a_text}', origin, text * 1.15, (255, 255, 255),
                      thick, background=BEST_COLOR)
        elif name in selected:
            cv2.polylines(canvas, [points], True, SELECTED_COLOR, thick + 1, cv2.LINE_AA)
            put_label(canvas, f'#{rank.get(name, "-")} {name} Q={q_text} a={a_text}', origin, text, SELECTED_COLOR, thick)
        else:
            cv2.polylines(canvas, [points], True, OTHER_COLOR, 1, cv2.LINE_AA)
            put_label(canvas, f'{name} Q={q_text}', origin, text * .75, OTHER_COLOR, max(1, thick - 1))
    # Legend at the top-left corner.
    y = int(28 * size)
    put_label(canvas, 'RED = highest confidence (top-1 dynamic ROI)', (8, y), text, BEST_COLOR, thick)
    put_label(canvas, 'GREEN = other selected ROIs   GREY = candidates', (8, y + int(26 * size)), text * .9,
              (255, 255, 255), thick)
    return canvas


def fusion_rows(rows, strategy, config):
    available = sorted((row for row in rows if row['q'] is not None),
                       key=lambda row: (-row['q'], row['roi_name']))
    if strategy == 'top1':
        selected = available[:1]
    elif strategy == 'top2':
        selected = available[:2]
    elif strategy == 'topk':
        selected = available[:config.top_k]
    else:
        selected = available
    alpha_sum = sum(float(row['alpha']) for row in selected)
    weights = [float(row['alpha'])/alpha_sum for row in selected] if alpha_sum > 0 else []
    vectors = [row['_representation'] for row in selected]
    if vectors:
        vector = np.average(np.asarray(vectors, dtype=float), axis=0, weights=weights).tolist()
    else:
        vector = []
    return selected, weights, vector


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compute ROI confidence and dynamic visual representation selection.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--manifest', default='dataset_manifest.csv')
    parser.add_argument('--quality-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--regime', choices=('equal_fixed', 'validation_tuned', 'weakly_supervised'))
    parser.add_argument('--top-k', type=int)
    mask_mode = parser.add_mutually_exclusive_group()
    mask_mode.add_argument('--enable-sam2-confidence', action='store_true')
    mask_mode.add_argument('--disable-sam2-confidence', action='store_true')
    parser.add_argument('--sam2-weight', type=float, help='Validation-selected nonnegative mask contribution')
    parser.add_argument('--no-visualizations', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    settings = cfg.get('confidence', {})
    root = Path(cfg['_project_root'])
    logger = setup_logging(project_path(cfg, cfg['logging']['log_dir']), cfg['logging']['level'])
    quality_dir = project_path(cfg, args.quality_dir or settings.get('quality_dir', 'outputs/roi_quality'))
    output = project_path(cfg, args.output_dir or settings.get('output_dir', 'outputs/confidence'))
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError('Confidence output directory is not empty; use --output-dir for a new experiment')
    records, quality_metadata = load_quality_table(quality_dir)
    config_values = {key: settings[key] for key in ConfidenceConfig.__dataclass_fields__ if key in settings}
    sam2_conf = cfg.get('sam2', {}).get('roi_confidence', {})
    config_values['sam2_enabled'] = bool(sam2_conf.get('enabled', False))
    config_values['sam2_weight'] = float(sam2_conf.get('weight', 0.))
    if args.regime is not None:
        config_values['regime'] = args.regime
    if args.top_k is not None:
        config_values['top_k'] = args.top_k
    if args.enable_sam2_confidence or args.disable_sam2_confidence:
        config_values['sam2_enabled'] = args.enable_sam2_confidence
    if args.sam2_weight is not None:
        config_values['sam2_weight'] = args.sam2_weight
    confidence_config = ConfidenceConfig(**config_values)
    mechanism = ConfidenceMechanism(confidence_config)
    mechanism.fit(records)
    manifest_path = project_path(cfg, args.manifest)
    manifest = DatasetManifest.load(manifest_path)
    if set(manifest.df.video_id) != {row['video_id'] for row in records}:
        raise ValueError('Quality/manifest video set mismatch')
    sources = {'manifest': manifest_path, 'quality_features': quality_dir/'roi_quality_features.csv',
               'quality_metadata': quality_dir/'experiment_metadata.json'}
    hashes = {key: sha256_file(path) for key, path in sources.items()}
    by_frame = {}
    for row in records:
        by_frame.setdefault((row['video_id'], row['frame_index']), []).append(row)
    for key in by_frame:
        by_frame[key].sort(key=lambda row: ROI_NAMES.index(row['roi_name']))
    confidence_rows, dynamic_rows, fused_rows = [], [], []
    frame_selections = {}
    for frame_key, frame_rows in by_frame.items():
        scored = []
        for row in frame_rows:
            q, terms, imputed = mechanism.score(row) if row['roi_status'] == 'AVAILABLE' else (None, None, [])
            output_row = dict(row, q=q, alpha=0., selected=False, selection_rank=None,
                              weight_regime=confidence_config.regime,
                              weight_pose=mechanism.weights.pose, weight_visibility=mechanism.weights.visibility,
                              weight_skin=mechanism.weights.skin, weight_temporal=mechanism.weights.temporal,
                              weight_motion=mechanism.weights.motion,
                              weight_sam2=confidence_config.sam2_weight if confidence_config.sam2_enabled else 0.,
                              sam2_term=mechanism.sam2_term(row),
                              pose_term=terms[0] if terms else None, visibility_term=terms[1] if terms else None,
                              skin_term=terms[2] if terms else None, temporal_term=terms[3] if terms else None,
                              motion_term=terms[4] if terms else None, imputed_terms=json.dumps(imputed),
                              _representation=mechanism.representation(row)[0],
                              representation_missing=json.dumps(mechanism.representation(row)[1]))
            scored.append(output_row)
        scores = [row['q'] for row in scored]
        alphas = mechanism.softmax(scores)
        ranked = mechanism.rank(scores)
        order = sorted((index for index, score in enumerate(scores) if score is not None),
                       key=lambda index: (-scores[index], index))
        for index, row in enumerate(scored):
            row['alpha'], row['selection_rank'] = alphas[index], ranked[index]
            row['selected'] = ranked[index] is not None and ranked[index] <= confidence_config.top_k
        frame_selections[frame_key] = scored
        for row in scored:
            public = {key: value for key, value in row.items() if not key.startswith('_')}
            confidence_rows.append(public)
        for strategy in STRATEGIES:
            selected, weights, vector = fusion_rows(scored, strategy, confidence_config)
            names = [row['roi_name'] for row in selected]
            payload = dict(video_id=frame_key[0], frame_index=frame_key[1], timestamp_sec=frame_rows[0]['timestamp_sec'],
                           split=frame_rows[0]['split'], selection_strategy=strategy,
                           selected_roi_names=json.dumps(names), selected_alpha=json.dumps([row['alpha'] for row in selected]),
                           fusion_weights=json.dumps(weights), selection_count=len(selected),
                           fusion_weight_sum=sum(weights), fused_representation=json.dumps(vector),
                           confidence_scoring_implemented=False)
            dynamic_rows.append(payload)
            fused_rows.append(dict(payload, representation_dimension=len(vector),
                                   representation=json.dumps(vector), roi_weights=json.dumps(weights)))
    if any(sha256_file(path) != digest for key, digest in hashes.items() for path in [sources[key]]):
        raise ValueError('Confidence inputs changed during execution')
    (output/'visualizations').mkdir(parents=True)
    visualization_count = 0
    if not args.no_visualizations and settings.get('save_annotated_frames', True):
        for _, record in manifest.df.iterrows():
            video = str(record.video_id)
            path = manifest.resolve_video_path(root, record)
            meta = VideoReader(path).metadata()
            frames = sorted(frame for current_video, frame in by_frame if current_video == video)
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f'Cannot decode video: {path}')
            current_index = 0
            try:
                for frame in frames:
                    while current_index <= frame:
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError(f'{video}: early decoding failure at frame {frame}')
                        read_index = current_index
                        current_index += 1
                        if read_index == frame:
                            scored = frame_selections[(video, frame)]
                            selected = {row['roi_name'] for row in scored if row['selected']}
                            target = output/'visualizations'/f'{video}_frame_{frame:08d}.jpg'
                            if not cv2.imwrite(str(target), draw_confidence(image, scored, selected)):
                                raise OSError(f'Cannot write {target}')
                            visualization_count += 1
                            break
            finally:
                capture.release()
    pd.DataFrame(confidence_rows).to_csv(output/'roi_confidence.csv', index=False)
    pd.DataFrame(dynamic_rows).to_csv(output/'dynamic_roi.csv', index=False)
    pd.DataFrame(fused_rows).to_csv(output/'fused_representation.csv', index=False)
    metadata = dict(phase='PHASE_10_ROI_CONFIDENCE_DYNAMIC_SELECTION', run_status='COMPLETE',
                    total_frames=len(by_frame), confidence_records=len(confidence_rows),
                    dynamic_records=len(dynamic_rows), fused_records=len(fused_rows), visualization_count=visualization_count,
                    configuration=asdict(confidence_config), selected_regime=confidence_config.regime,
                    weights=mechanism.weights.as_dict(), fit_metadata=mechanism.fit_metadata,
                    softmax_temperature=confidence_config.temperature, selection_strategies=list(STRATEGIES),
                    source_hashes={key: dict(path=str(sources[key]), sha256=hashes[key]) for key in sources},
                    confidence_formula='Q = w1*P + w2*V + w3*S + w4*C - w5*M'
                        + (' + w_sam2*(SAM2_mask_support*SAM2_mask_quality)' if confidence_config.sam2_enabled else ''),
                    sam2_weight_policy='Zero by default; choose nonzero value on validation data and report it',
                    alpha_formula='exp(Q/temperature) / sum(exp(Q/temperature)) over available ROIs per frame',
                    representation_definition='normalized geometry and Phase 9 visual/temporal features; no HR or RR',
                    final_confidence_score_implemented=False, hr_calculated=False, rr_calculated=False,
                    weak_supervision_warning='Validation-tuned and learned regimes fit transparent pseudo-reliability targets from quality metrics, not ground-truth reliability labels.',
                    ground_truth_status='UNAVAILABLE', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    python_version=platform.python_version(), opencv_version=cv2.__version__,
                    quality_input_run_status=quality_metadata.get('run_status'))
    (output/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    logger.info('Confidence complete: %d frames, %d ROI records, regime=%s, weights=%s',
                len(by_frame), len(confidence_rows), confidence_config.regime, mechanism.weights.as_dict())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
