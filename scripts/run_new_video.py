"""Run the Module 1 pipeline on new video(s) and render annotated videos, with and/or without SAM 2.

Usage (from the Module1 folder):
    python scripts/run_new_video.py annotate --branch baseline   # every video in the annotate/ folder, no SAM 2
    python scripts/run_new_video.py annotate --branch sam2       # same videos, with SAM 2
    python scripts/run_new_video.py 111.mp4                      # one video file, no SAM 2

Everything is written to runs/<name>/ (name = folder or file name), so the 24- and 105-video
results are never touched. Phases 2-6 are shared by both branches. A step whose output already
exists is skipped, so a failed or second run resumes instead of starting over. The same models and
settings as the main runs are used; nothing is trained.

Final videos: runs/<name>/annotated_without_sam2/ and runs/<name>/annotated_with_sam2/
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from module1.data.video import VideoReader

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv'}
SKIP_DIRS = {'runs', 'outputs', 'external', 'models', '.venv', '.git', '__pycache__', '.ipynb_checkpoints'}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(value):
    """Resolve the video path; otherwise search the project's parent folder by name (case-insensitive)."""
    given = Path(value)
    for candidate in ((given,) if given.is_absolute() else (Path.cwd()/given, ROOT/given)):
        if candidate.is_file():
            return candidate.resolve()
    search_root = ROOT.parent
    wanted = given.name.lower()
    videos = []
    for folder, subfolders, files in os.walk(search_root):
        subfolders[:] = [name for name in subfolders if name not in SKIP_DIRS and not name.startswith('.')]
        videos += [Path(folder)/name for name in files if Path(name).suffix.lower() in VIDEO_EXTENSIONS]
    matches = [path for path in videos if path.name.lower() == wanted or path.stem.lower() == given.stem.lower()]
    if len(matches) == 1:
        print(f'Found video: {matches[0]}')
        return matches[0].resolve()
    if len(matches) > 1:
        raise FileNotFoundError(f'Several videos match {value}; pass the full path:\n' + '\n'.join(map(str, matches)))
    others = [str(path) for path in videos if 'data/Dataset' not in path.as_posix()]
    raise FileNotFoundError(f'Video not found: {value} (searched {search_root}).\n'
                            'Upload it into the Module1 folder and wait for the upload to finish.'
                            + ('\nOther videos found:\n' + '\n'.join(others) if others else ''))


def find_videos(value):
    """A folder gives all its videos (one run named after the folder); a file gives one video."""
    given = Path(value)
    for candidate in ((given,) if given.is_absolute() else (Path.cwd()/given, ROOT/given)):
        if candidate.is_dir():
            videos = sorted(p.resolve() for p in candidate.rglob('*')
                            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and '.ipynb_checkpoints' not in p.parts)
            if not videos:
                raise FileNotFoundError(f'No videos in {candidate}; upload them and wait for the upload to finish')
            return candidate.resolve().name, videos
    video = find_video(value)
    return video.stem, [video]


def video_id_for(video):
    return 'NEW_' + re.sub(r'[^A-Za-z0-9]+', '_', video.stem).strip('_')


def write_manifest(videos, target):
    rows = []
    for video in videos:
        meta = VideoReader(video).metadata()
        if not meta.opened_successfully or meta.native_fps <= 0 or meta.frame_count <= 0:
            raise ValueError(f'OpenCV cannot read {video}')
        rows.append(dict(source_dataset='user_supplied_video', source_type='user_supplied_video',
                         video_id=video_id_for(video), infant_id='UNAVAILABLE', session_id=video_id_for(video),
                         filename=video.name, video_path=str(video), fps=meta.native_fps,
                         frame_count=meta.frame_count, duration_sec=meta.duration_sec, width=meta.width,
                         height=meta.height, camera_viewpoint='UNKNOWN', ground_truth_hr_available='UNKNOWN',
                         ground_truth_rr_available='UNKNOWN', roi_annotation_available='UNKNOWN', split='test',
                         split_unit='video', subject_independent_verified=False, sha256=sha256(video),
                         notes='New video run; no ground truth.'))
        print(f'{video.name}: {meta.width}x{meta.height}, {meta.native_fps:.2f} fps, {meta.frame_count} frames')
    df = pd.DataFrame(rows)
    if df.video_id.duplicated().any():
        raise ValueError('Two videos map to the same ID; rename them: ' + ', '.join(df.filename[df.video_id.duplicated(keep=False)]))
    df.to_csv(target, index=False)


def write_config(run_dir, keep_phase_images):
    cfg = yaml.safe_load((ROOT/'configs/module1.yaml').read_text(encoding='utf-8'))
    # Phase outputs stay relative (they land in run_dir); shared models point back to Module1.
    absolute = lambda value: str((ROOT/value).resolve())
    for name in ('yolo', 'rtdetr'):
        cfg['detection'][name]['model_path'] = absolute(cfg['detection'][name]['model_path'])
    for key in ('source_root', 'model_config', 'checkpoint'):
        cfg['pose'][key] = absolute(cfg['pose'][key])
    cfg['sam2']['checkpoint'] = absolute(cfg['sam2']['checkpoint'])  # model_config is a SAM 2 package name
    if not keep_phase_images:
        for section in (cfg['detection']['yolo'], cfg['detection']['rtdetr'], cfg['fusion'], cfg['tracking'],
                        cfg['recovery'], cfg['pose'], cfg['roi'], cfg['roi_quality'], cfg['confidence'], cfg['sam2']):
            section['save_annotated_frames'] = False
    (run_dir/'configs').mkdir(parents=True, exist_ok=True)
    path = run_dir/'configs/module1.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    return path, cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Module 1 on new video(s) and render annotated videos.')
    parser.add_argument('video', help='A folder of videos (e.g. annotate) or one video file (e.g. 111.mp4)')
    parser.add_argument('--branch', choices=('baseline', 'sam2', 'both'), default='baseline',
                        help='baseline = without SAM 2 (default); sam2 = with SAM 2; both = both branches')
    parser.add_argument('--name', help='Run name (default: folder name or file name without extension)')
    parser.add_argument('--strategy', default='topk', choices=('top1', 'top2', 'topk', 'confidence_weighted_fusion'))
    parser.add_argument('--keep-phase-images', action='store_true', help='Also save per-phase annotated frames (slow, large)')
    args = parser.parse_args(argv)

    default_name, videos = find_videos(args.video)
    name = args.name or default_name
    run_dir = ROOT/'runs'/name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir/'manifest.csv'
    if manifest.is_file():
        previous = pd.read_csv(manifest)
        if set(zip(previous.filename, previous.sha256)) != {(video.name, sha256(video)) for video in videos}:
            raise ValueError(f'The videos changed since {run_dir} was created. Use a new --name '
                             f'(or delete {run_dir}) to process the new set of videos.')
    else:
        write_manifest(videos, manifest)
    config, cfg = write_config(run_dir, args.keep_phase_images)
    out = run_dir/'outputs'
    common = ['--config', str(config), '--manifest', str(manifest)]
    no_vis = [] if args.keep_phase_images else ['--no-visualizations']
    steps = [
        ('Phase 2 YOLO', 'run_yolo_baseline.py', [], out/'yolo/yolo_detections.csv'),
        ('Phase 3 RT-DETR', 'run_rtdetr_baseline.py', [], out/'rtdetr/rtdetr_detections.csv'),
        ('Phase 4 fusion', 'run_detection_fusion.py', no_vis, out/'fusion/detection_fusion.csv'),
        ('Phase 5 tracking', 'run_tracking.py', ['--tracker', 'bytetrack'] + no_vis, out/'tracking/bytetrack/tracking_results.csv'),
        ('Phase 6 recovery', 'run_temporal_recovery.py', no_vis, out/'recovery/recovery_results.csv'),
    ]
    branches = []
    if args.branch in ('baseline', 'both'):
        steps += [
            ('Phase 7 pose (without SAM 2)', 'run_sapiens2_pose.py', ['--without-sam2'] + no_vis, out/'pose/pose_results.csv'),
            ('Phase 8 ROIs (without SAM 2)', 'run_anatomical_rois.py', no_vis, out/'roi/candidate_rois.csv'),
            ('Phase 9 ROI quality (without SAM 2)', 'run_roi_quality.py', no_vis, out/'roi_quality/roi_quality_features.csv'),
            ('Phase 10 confidence (without SAM 2)', 'run_roi_confidence.py', ['--disable-sam2-confidence'] + no_vis,
             out/'confidence/dynamic_roi.csv'),
        ]
        branches.append(('outputs/confidence', 'annotated_videos', 'annotated_without_sam2', 'without_sam2'))
    if args.branch in ('sam2', 'both'):
        steps += [
            ('Phase 12 SAM 2', 'run_sam2_segmentation.py', ['--enable', '--require-model'] + no_vis, out/'sam2/sam2_masks.csv'),
            ('Phase 7 pose (with SAM 2)', 'run_sapiens2_pose.py',
             ['--sam2-dir', 'outputs/sam2', '--output-dir', 'outputs/pose_sam2'] + no_vis, out/'pose_sam2/pose_results.csv'),
            ('Phase 8 ROIs (with SAM 2)', 'run_anatomical_rois.py',
             ['--pose-dir', 'outputs/pose_sam2', '--sam2-dir', 'outputs/sam2', '--output-dir', 'outputs/roi_sam2'] + no_vis,
             out/'roi_sam2/candidate_rois.csv'),
            ('Phase 9 ROI quality (with SAM 2)', 'run_roi_quality.py',
             ['--roi-dir', 'outputs/roi_sam2', '--pose-dir', 'outputs/pose_sam2', '--output-dir', 'outputs/roi_quality_sam2'] + no_vis,
             out/'roi_quality_sam2/roi_quality_features.csv'),
            ('Phase 10 confidence (with SAM 2)', 'run_roi_confidence.py',
             ['--quality-dir', 'outputs/roi_quality_sam2', '--output-dir', 'outputs/confidence_sam2', '--enable-sam2-confidence',
              '--sam2-weight', str(cfg['sam2']['roi_confidence'].get('weight', 0.))] + no_vis,
             out/'confidence_sam2/dynamic_roi.csv'),
        ]
        branches.append(('outputs/confidence_sam2', 'annotated_videos_sam2', 'annotated_with_sam2', 'with_sam2'))
    for title, script, extra, done in steps:
        if done.is_file():
            print(f'== {title}: already done, skipped')
            continue
        # Remove a partial output folder left by a failed earlier attempt.
        partial = done.parent
        if partial.exists() and partial != out:
            shutil.rmtree(partial)
        print(f'== {title} ...', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts'/script)] + common + extra, check=True)

    finals = []
    for confidence_dir, rendered_dir, final_dir, label in branches:
        print(f'== Annotating videos ({label}) ...', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/video_annotate.py')] + common +
                       ['--confidence-dir', confidence_dir, '--split', 'test', '--strategy', args.strategy,
                        '--output-dir', f'outputs/{rendered_dir}'], check=True)
        (run_dir/final_dir).mkdir(exist_ok=True)
        for video in videos:
            rendered = out/rendered_dir/f'{video_id_for(video)}_test_{args.strategy}.mp4'
            final = run_dir/final_dir/f'{video.stem}_{label}_{args.strategy}.mp4'
            shutil.copyfile(rendered, final)
            finals.append(final)
    print('\nFINAL ANNOTATED VIDEOS:')
    for final in finals:
        print(f'  {final}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
