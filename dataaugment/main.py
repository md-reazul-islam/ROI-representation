#!/usr/bin/env python
"""DataAugment: controlled motion / occlusion / lighting augmentation of UBFC-rPPG videos.

    python dataaugment/main.py                      # full run (first 3 UBFC videos, 3 severities each)
    python dataaugment/main.py --max-frames 120     # quick test on the first 120 frames

Each source video is read ONCE; the three severity outputs are generated in the same pass and written as
lossless FFV1/MKV. Every output is decoded again and compared frame-by-frame (MD5) with what was written.
Source files are opened read-only and their SHA-256 is checked before and after the run.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from src import lighting, motion, motion_fast, occlusion  # noqa: E402
from src.video_utils import (FFV1Writer, FrameDiffTracker, detect_subject_box, find_videos, mux_audio,  # noqa: E402
                             open_video, sha256, spectral_check, step_stats, verify_output)

CATEGORIES = ['motion', 'occlusion', 'lighting', 'motion_fast']   # append only: index = seed stream
SEVERITIES = ['small', 'medium', 'large']   # default; a config may set run_severities (e.g. [xlarge])
MODULES = {'motion': (motion.make_units, motion.MotionAugmenter),
           'occlusion': (occlusion.make_units, occlusion.OcclusionAugmenter),
           'lighting': (lighting.make_units, lighting.LightingAugmenter),
           'motion_fast': (motion_fast.make_units, motion_fast.FastMotionAugmenter)}
NOTES = {
    'motion': 'Each output frame is resampled once from the original frame with a sub-pixel affine warp '
              '(Lanczos-4). Resampling is a spatially local weighted average of neighbouring pixels of the same '
              'frame: it slightly smooths fine spatial detail and can overshoot by a few grey levels at sharp '
              'edges (clipped to 0-255), but it adds no temporal component. Uncovered borders are filled per '
              'border_mode (see border_fraction).',
    'motion_fast': 'Same warp as motion (one Lanczos-4 affine resampling per frame from the original pixels, no '
                   'temporal component); large quick shifts expose wider black borders (see border_fraction).',
    'occlusion': 'Visible (unoccluded) pixels are unchanged except inside the soft hand shadow. Occluders are '
                 'SIMULATED (procedural hand-shaped mask and cloth texture composited with soft alpha); they are '
                 'not real swaddling and carry no pulse. Occluder layers are resampled with bilinear '
                 'interpolation; the video pixels themselves are not resampled.',
    'lighting': 'No geometric resampling. Gains are applied in approximately linear light (gamma 2.2) and '
                're-quantised to 8 bit, which changes values by at most 1 grey level beyond the intended gain; '
                'pixels pushed beyond 0/255 clip (see saturated_frac_out).',
}


def subject_name(p):
    p = Path(p)
    return p.parent.name if p.stem.lower() == 'vid' else p.stem


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def preview_sheet(snaps, labels, path, scale=0.5):
    tiles = []
    for key in snaps:
        row = []
        for fr, idx in zip(snaps[key], labels):
            t = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.putText(t, f'{key} | frame {idx}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(t, f'{key} | frame {idx}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row.append(t)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(tiles))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(HERE / 'config/augmentation_config.yaml'))
    ap.add_argument('--max-frames', type=int, default=None, help='process only the first N frames (testing)')
    ap.add_argument('--skip-hash', action='store_true', help='skip SHA-256 integrity check of the source videos')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    in_dir, out_dir = (ROOT / cfg['input_dir']).resolve(), (ROOT / cfg['output_dir']).resolve()
    if out_dir == in_dir or in_dir in out_dir.parents or out_dir in in_dir.parents:
        sys.exit(f'output_dir {out_dir} must not overlap input_dir {in_dir}')
    assignment = cfg['assignment']
    vids = find_videos(in_dir, cfg['n_videos'])
    if len(vids) < len(assignment):
        sys.exit(f'need {len(assignment)} videos in {in_dir}, found {len(vids)}')
    bands = {k: tuple(v) for k, v in cfg['bands_hz'].items()}
    pthr, rthr = cfg['narrowband_power_threshold'], cfg['narrowband_peak_ratio']
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'DataAugment | input {in_dir} | output {out_dir}')
    hashes_before = {} if args.skip_hash else {str(v): sha256(v) for v in vids}
    run = {'created': dt.datetime.now().isoformat(timespec='seconds'), 'config_file': str(args.config),
           'config': cfg, 'max_frames': args.max_frames,
           'environment': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
                           'platform': platform.platform()},
           'videos': []}
    manifest = []

    for vid, cat in zip(vids, assignment):
        reader = open_video(vid)
        n = reader.n_frames if args.max_frames is None else min(reader.n_frames, args.max_frames)
        fps, W, H = reader.fps, reader.width, reader.height
        meta = {'width': W, 'height': H, 'fps': fps}
        name = subject_name(vid)
        ccfg = cfg[cat]
        seed_seq = np.random.SeedSequence([cfg['seed'], CATEGORIES.index(cat)])
        rng = np.random.default_rng(seed_seq)
        subject = detect_subject_box(reader)
        make_units, Cls = MODULES[cat]
        units = make_units(n, fps, ccfg, rng)
        SEVERITIES = cfg.get('run_severities', globals()['SEVERITIES'])
        augs = {s: Cls(ccfg, ccfg['severities'][s], units, meta, subject) for s in SEVERITIES}
        paths = {s: out_dir / cat / s / f'{name}_{cat}_{s}.mkv' for s in SEVERITIES}
        writers = {s: FFV1Writer(paths[s], fps, (W, H)) for s in SEVERITIES}
        trackers = {k: FrameDiffTracker() for k in ['original'] + SEVERITIES}
        prev_idx = sorted({int(round(p * (n - 1))) for p in cfg['preview_frames']})
        snaps = {k: [] for k in ['original'] + SEVERITIES}
        first = reader.read(0)
        identity = (motion.identity_check(first, ccfg) if cat in ('motion', 'motion_fast')
                    else lighting.identity_check(first) if cat == 'lighting' else None)

        print(f'\n[{cat}] {vid}  ({W}x{H}, {fps:.6f} fps, {n} frames, {reader.codec}); subject box: '
              f"{subject['method']}")
        t0, count = time.time(), 0
        for i, frame in enumerate(reader.frames(n)):
            trackers['original'].add(frame)
            if i in prev_idx:
                snaps['original'].append(frame.copy())
            for s in SEVERITIES:
                out = augs[s].apply(frame, i)
                writers[s].write(out)
                trackers[s].add(out)
                if i in prev_idx:
                    snaps[s].append(out)
            count += 1
            if count % 300 == 0:
                print(f'  {count}/{n} frames  ({count / (time.time() - t0):.1f} frames/s)', flush=True)
        for w in writers.values():
            w.close()
        if count != n:
            sys.exit(f'{vid}: read {count} of {n} frames')
        elapsed = time.time() - t0
        preview_path = out_dir / cat / f'preview_{name}.png'
        preview_sheet(snaps, prev_idx, preview_path)

        gt = vid.parent / 'ground_truth.txt'
        for s in SEVERITIES:
            p = paths[s]
            q = verify_output(p, writers[s].md5)
            if reader.has_audio:
                audio = mux_audio(p, vid)
            else:
                audio = {'source_has_audio': False if reader.has_audio is False else None,
                         'audio_preserved': 'not applicable', 'audio_sync': 'not applicable (no audio stream)'}
            dur_in, dur_out = n / fps, n / q['fps_container'] if q['fps_container'] else float('nan')
            quality = {
                'resolution_in': [W, H], 'resolution_out': [q['width'], q['height']],
                'fps_in_exact': f'{reader.fps_fraction.numerator}/{reader.fps_fraction.denominator}',
                'fps_in': round(fps, 6), 'fps_out_container': q['fps_container'],
                'fps_abs_diff': round(abs(fps - q['fps_container']), 6),
                'frame_count_in': n, 'frame_count_out_decoded': q['frame_count_decoded'],
                'frame_count_out_header': q['frame_count_header'],
                'duration_in_s': round(dur_in, 4), 'duration_out_s': round(dur_out, 4),
                'duration_diff_ms': round(1000 * abs(dur_out - dur_in), 3),
                'lossless_bit_exact': q['lossless_bit_exact'], 'frames_mismatched': q['frames_mismatched'],
                'codec': 'FFV1 (lossless) in Matroska via OpenCV/FFmpeg', 'file_size_mb': q['file_size_mb'],
                'audio': audio}
            sig = augs[s].control_signals()
            spectra = {k: spectral_check(v, fps, bands, pthr, rthr) for k, v in sig.items()}
            flagged = [f'{k}:{b}' for k, d in spectra.items() for b, r in d.items()
                       if isinstance(r, dict) and r.get('narrowband_flag')]
            temporal = {'control_signal_spectra': spectra, 'narrowband_flags': flagged,
                        'control_signal_steps': {k: step_stats(v) for k, v in sig.items()},
                        'frame_difference_original': trackers['original'].summary(),
                        'frame_difference_augmented': trackers[s].summary()}
            summ = augs[s].summary()
            checks = {'resolution_preserved': quality['resolution_out'] == quality['resolution_in'],
                      'fps_preserved_within_0.001': quality['fps_abs_diff'] < 1e-3,
                      'frame_count_preserved': q['frame_count_decoded'] == n,
                      'duration_within_5ms': quality['duration_diff_ms'] < 5,
                      'lossless_bit_exact': q['lossless_bit_exact'],
                      'no_narrowband_component_in_rr_hr_bands': not flagged}
            if identity is not None:
                checks['identity_transform_bit_exact'] = identity
            if cat == 'occlusion':
                checks['coverage_within_0.01_of_target'] = abs(summ['coverage_mean'] - summ['target_area_fraction']) < 0.01
            rec = {'category': cat, 'severity': s, 'source': {'path': str(vid), 'subject': name, 'codec': reader.codec,
                                                              'sha256': hashes_before.get(str(vid))},
                   'output': str(p), 'seed': {'base_seed': cfg['seed'], 'category_index': CATEGORIES.index(cat),
                                              'seed_sequence_entropy': seed_seq.entropy,
                                              'note': 'same unit trajectories for all three severities'},
                   'subject_box': subject, 'parameters': {'category_config': {k: v for k, v in ccfg.items()
                                                                              if k != 'severities'},
                                                          'severity': ccfg['severities'][s]},
                   'augmentation_summary': summ, 'quality': quality, 'temporal_consistency': temporal,
                   'interpolation_note': NOTES[cat], 'checks': checks, 'all_checks_passed': all(checks.values()),
                   'per_frame_parameters': str(p.with_name(p.stem + '_frames.csv')),
                   'processing_seconds_all_severities': round(elapsed, 1)}
            write_csv(p.with_name(p.stem + '_frames.csv'), augs[s].rows)
            if cfg.get('copy_ground_truth') and gt.exists():
                shutil.copy2(gt, p.with_name(f'{name}_ground_truth.txt'))
                rec['ground_truth_copy'] = str(p.with_name(f'{name}_ground_truth.txt'))
            json.dump(rec, open(p.with_suffix('.json'), 'w'), indent=2, default=str)
            run['videos'].append(rec)
            manifest.append({'category': cat, 'severity': s, 'source': str(vid.relative_to(ROOT)),
                             'output': str(p.relative_to(ROOT)), 'frames': q['frame_count_decoded'],
                             'fps_in': round(fps, 6), 'fps_out': q['fps_container'],
                             'resolution': f"{q['width']}x{q['height']}", 'size_mb': q['file_size_mb'],
                             'lossless': q['lossless_bit_exact'], 'narrowband_flags': len(flagged),
                             'all_checks_passed': rec['all_checks_passed']})
            print(f"  {s:6s} -> {p.relative_to(ROOT)} | bit-exact {q['lossless_bit_exact']} | "
                  f"{q['frame_count_decoded']} frames | checks {'PASS' if rec['all_checks_passed'] else 'FAIL'}")

    hashes_after = {} if args.skip_hash else {str(v): sha256(v) for v in vids}
    run['source_integrity'] = {'checked': not args.skip_hash, 'sha256_before': hashes_before,
                               'sha256_after': hashes_after,
                               'originals_unchanged': hashes_before == hashes_after if hashes_before else None}
    run['all_outputs_passed'] = all(v['all_checks_passed'] for v in run['videos'])
    json.dump(run, open(out_dir / 'validation_report.json', 'w'), indent=2, default=str)
    write_csv(out_dir / 'manifest.csv', manifest)
    print(f"\nOriginals unchanged: {run['source_integrity']['originals_unchanged']} | all outputs passed: "
          f"{run['all_outputs_passed']}\nReport: {out_dir / 'validation_report.json'}")


if __name__ == '__main__':
    main()
