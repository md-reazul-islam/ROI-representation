"""Run the complete Module 1 pipeline (both branches) for the paper evaluation.

Usage (from the Module1 folder):
    python scripts/run_paper_pilot.py                      # runs/pilot24_paper, both branches
    python scripts/run_paper_pilot.py --max-videos 2       # smoke run on the first two videos
    python scripts/run_paper_pilot.py --name dataset105_paper   # any run whose runs/<name>/manifest.csv already
                                                                # exists (e.g. built by RUN_MODULE1_PAPER.ipynb)

Unlike run_new_video.py, the video IDs (VID_xxx) and the deterministic 18/3/3 video-level split of
dataset_manifest.csv are kept, so the paper evaluation (run_paper_evaluation.py) can report per split.
Video paths point at data/pilot24_links/<video_id>.mp4; every file's SHA-256 is checked against the
manifest first. Phases, models and settings are those of run_new_video.py; a phase whose output
exists is skipped, so an interrupted run resumes. After the shared Phases 2-6, the baseline and SAM 2
branches run concurrently (independent outputs; same models and settings). Per-phase wall time is
written to phase_timings.json and GPU memory to gpu_monitor.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

from run_new_video import sha256, write_config


def write_manifest(source, links, target, max_videos=None):
    manifest = pd.read_csv(source)
    if max_videos:
        manifest = manifest.head(max_videos)
    paths = [links/f'{video_id}.mp4' for video_id in manifest.video_id]
    for path, expected in zip(paths, manifest.sha256):
        if sha256(path) != expected:
            raise ValueError(f'{path} does not match the manifest SHA-256')
    manifest['video_path'] = [str(path.resolve()) for path in paths]
    manifest.to_csv(target, index=False)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run Module 1 (both branches) on the 24-video pilot for the paper.')
    parser.add_argument('--manifest', default=str(ROOT/'dataset_manifest.csv'))
    parser.add_argument('--links', default=str(ROOT/'data/pilot24_links'))
    parser.add_argument('--name', default='pilot24_paper')
    parser.add_argument('--max-videos', type=int, help='Only the first N manifest videos (smoke run)')
    args = parser.parse_args(argv)

    run_dir = ROOT/'runs'/args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir/'manifest.csv'
    if not manifest.is_file():
        write_manifest(Path(args.manifest), Path(args.links), manifest, args.max_videos)
    config, cfg = write_config(run_dir, keep_phase_images=False)
    out = run_dir/'outputs'
    common = ['--config', str(config), '--manifest', str(manifest)]
    no_vis = ['--no-visualizations']
    steps = [
        ('yolo', 'run_yolo_baseline.py', [], out/'yolo/yolo_detections.csv'),
        ('rtdetr', 'run_rtdetr_baseline.py', [], out/'rtdetr/rtdetr_detections.csv'),
        ('fusion', 'run_detection_fusion.py', no_vis, out/'fusion/detection_fusion.csv'),
        ('tracking', 'run_tracking.py', ['--tracker', 'bytetrack'] + no_vis, out/'tracking/bytetrack/tracking_results.csv'),
        ('recovery', 'run_temporal_recovery.py', no_vis, out/'recovery/recovery_results.csv'),
        ('pose', 'run_sapiens2_pose.py', ['--without-sam2'] + no_vis, out/'pose/pose_results.csv'),
        ('roi', 'run_anatomical_rois.py', no_vis, out/'roi/candidate_rois.csv'),
        ('roi_quality', 'run_roi_quality.py', no_vis, out/'roi_quality/roi_quality_features.csv'),
        ('confidence', 'run_roi_confidence.py', ['--disable-sam2-confidence'] + no_vis, out/'confidence/dynamic_roi.csv'),
        ('sam2', 'run_sam2_segmentation.py', ['--enable', '--require-model'] + no_vis, out/'sam2/sam2_masks.csv'),
        ('pose_sam2', 'run_sapiens2_pose.py', ['--sam2-dir', 'outputs/sam2', '--output-dir', 'outputs/pose_sam2'] + no_vis,
         out/'pose_sam2/pose_results.csv'),
        ('roi_sam2', 'run_anatomical_rois.py',
         ['--pose-dir', 'outputs/pose_sam2', '--sam2-dir', 'outputs/sam2', '--output-dir', 'outputs/roi_sam2'] + no_vis,
         out/'roi_sam2/candidate_rois.csv'),
        ('roi_quality_sam2', 'run_roi_quality.py',
         ['--roi-dir', 'outputs/roi_sam2', '--pose-dir', 'outputs/pose_sam2', '--output-dir', 'outputs/roi_quality_sam2'] + no_vis,
         out/'roi_quality_sam2/roi_quality_features.csv'),
        ('confidence_sam2', 'run_roi_confidence.py',
         ['--quality-dir', 'outputs/roi_quality_sam2', '--output-dir', 'outputs/confidence_sam2', '--enable-sam2-confidence',
          '--sam2-weight', str(cfg['sam2']['roi_confidence'].get('weight', 0.))] + no_vis,
         out/'confidence_sam2/dynamic_roi.csv'),
    ]
    timings_path = run_dir/'phase_timings.json'
    timings = json.loads(timings_path.read_text()) if timings_path.is_file() else {}
    lock = threading.Lock()

    def run_steps(sequence):
        for name, script, extra, done in sequence:
            if done.is_file():
                print(f'== {name}: already done, skipped', flush=True)
                continue
            partial = done.parent
            if partial.exists() and partial != out:
                shutil.rmtree(partial)
            print(f'== {name} ...', flush=True)
            start = time.perf_counter()
            subprocess.run([sys.executable, str(ROOT/'scripts'/script)] + common + extra, check=True, cwd=run_dir)
            with lock:
                timings[name] = time.perf_counter() - start
                timings_path.write_text(json.dumps(timings, indent=2))

    # GPU memory log for the compute table (read by run_paper_evaluation.py --gpu-log).
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu', '--format=csv,noheader',
                                '-lms', '2000'], stdout=open(run_dir/'gpu_monitor.csv', 'a'), stderr=subprocess.DEVNULL)
    try:
        run_pipeline(run_steps, steps)
    finally:
        monitor.terminate()
    print(f'Done: {run_dir}')
    return 0


def run_pipeline(run_steps, steps):
    run_steps(steps[:5])
    errors = []

    def branch(sequence):
        try:
            run_steps(sequence)
        except Exception as error:   # noqa: BLE001 - re-raised after both branches finish
            errors.append(error)
    threads = [threading.Thread(target=branch, args=(part,)) for part in (steps[5:9], steps[9:])]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


if __name__ == '__main__':
    raise SystemExit(main())
