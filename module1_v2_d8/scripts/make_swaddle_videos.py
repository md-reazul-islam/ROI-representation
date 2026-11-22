"""Swaddling augmentation (pre-registered in ../PREREGISTRATION.md), written as new test videos.

    python module1_v2_d8/scripts/make_swaddle_videos.py

UBFC subject1; a cloth layer rises from the bottom of the frame to a top edge at
  small  = chin + 0.6 face heights (chest and shoulders covered)
  medium = chin                    (neck covered)
  large  = chin - 0.25 face heights (chin and mouth covered)
with gentle edge waves (6 px, wavelength 120 px, slowly drifting phase) and an aperiodic vertical drift of +-0.05 face
heights (low-pass 0.05 Hz). Face box = DataAugment's Haar detection (median); chin = bottom of the face box.
Uses DataAugment's reader, cloth texture, noise, writer and checks (imported read only; dataaugment/ is not changed).
Output: module1_v2_d8/data_swaddle/subject1_swaddle_<severity>.mkv (+ .json), lossless FFV1, verified bit-exact.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

D8 = Path(__file__).resolve().parents[1]
ROOT = D8.parent
sys.path.insert(0, str(ROOT/'dataaugment'))
from src.occlusion import render_cloth  # noqa: E402
from src.video_utils import (FFV1Writer, detect_subject_box, open_video, sha256, smooth_noise, spectral_check,  # noqa: E402
                             verify_output)

SOURCE = ROOT/'data/ubfc/subject1/vid.avi'
OUT = D8/'data_swaddle'
SEVERITIES = {'small': 0.6, 'medium': 0.0, 'large': -0.25}      # top edge = chin + value x face height
BLANKET_BGR = (214, 196, 172)                                   # same pale-blue cloth as DataAugment's blanket
WAVE_AMP, WAVE_LEN, FEATHER = 6.0, 120.0, 4.0
DRIFT_FACE, DRIFT_HZ = 0.05, 0.05
SEED = 20261115


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reader = open_video(SOURCE)
    n, fps, W, H = reader.n_frames, reader.fps, reader.width, reader.height
    face = detect_subject_box(reader)
    chin, fh = face['y'] + face['h'], face['h']
    rng = np.random.default_rng(SEED)
    drift = DRIFT_FACE * fh * smooth_noise(n, fps, DRIFT_HZ, rng)
    phase = 1.5 * smooth_noise(n, fps, DRIFT_HZ, rng)
    cloth = render_cloth(H, W, BLANKET_BGR, rng)
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)[:, None]
    before = sha256(SOURCE)
    writers = {s: FFV1Writer(OUT/f'subject1_swaddle_{s}.mkv', fps, (W, H)) for s in SEVERITIES}
    coverage = {s: [] for s in SEVERITIES}
    for i, frame in enumerate(reader.frames(n)):
        for s, off in SEVERITIES.items():
            top = chin + off * fh + drift[i] + WAVE_AMP * np.sin(2 * np.pi * xs / WAVE_LEN + phase[i])
            alpha = np.clip((ys - top[None, :]) / FEATHER + .5, 0, 1)[:, :, None]
            out = np.clip(alpha * cloth + (1 - alpha) * frame.astype(np.float32), 0, 255).round().astype(np.uint8)
            writers[s].write(out)
            coverage[s].append(float(alpha.mean()))
    for w in writers.values():
        w.close()
    after = sha256(SOURCE)
    bands = {'rr': (0.15, 1.5), 'hr': (0.7, 4.0)}
    for s, off in SEVERITIES.items():
        p = OUT/f'subject1_swaddle_{s}.mkv'
        q = verify_output(p, writers[s].md5)
        spectra = {'drift_px': spectral_check(drift, fps, bands), 'phase_rad': spectral_check(phase, fps, bands)}
        flags = [f'{k}:{b}' for k, d in spectra.items() for b, r in d.items() if isinstance(r, dict) and r.get('narrowband_flag')]
        rec = dict(source=str(SOURCE), severity=s, top_edge_rule=f'chin + {off} x face height', face_box=face,
                   chin_y=chin, face_height=fh, seed=SEED, coverage_mean=round(float(np.mean(coverage[s])), 4),
                   frames=q['frame_count_decoded'], lossless_bit_exact=q['lossless_bit_exact'],
                   fps_in=fps, fps_out=q['fps_container'], narrowband_flags=flags,
                   original_unchanged=before == after,
                   checks_passed=bool(q['lossless_bit_exact'] and q['frame_count_decoded'] == n and not flags and before == after))
        p.with_suffix('.json').write_text(json.dumps(rec, indent=2, default=str))
        print(s, 'coverage', rec['coverage_mean'], 'bit-exact', rec['lossless_bit_exact'], 'frames', rec['frames'],
              'checks', 'PASS' if rec['checks_passed'] else 'FAIL')
    # preview: one frame per severity
    frame = reader.read(int(n * .5))
    tiles = []
    for s, off in SEVERITIES.items():
        top = chin + off * fh + WAVE_AMP * np.sin(2 * np.pi * xs / WAVE_LEN)
        alpha = np.clip((ys - top[None, :]) / FEATHER + .5, 0, 1)[:, :, None]
        t = np.clip(alpha * cloth + (1 - alpha) * frame, 0, 255).astype(np.uint8)
        cv2.putText(t, s, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        tiles.append(cv2.resize(t, None, fx=.5, fy=.5))
    cv2.imwrite(str(OUT/'preview_subject1_swaddle.png'), np.hstack(tiles))


if __name__ == '__main__':
    main()
