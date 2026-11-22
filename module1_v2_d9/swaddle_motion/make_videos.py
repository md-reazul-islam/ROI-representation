"""Swaddling + fast motion test videos (PREREGISTRATION.md in this folder).

    python module1_v2_d9/swaddle_motion/make_videos.py

UBFC subjects 1, 3, 4 x swaddle {medium, large}; cloth composited first, then DataAugment's motion_fast (severity
large) warps the whole frame. Output: module1_v2_d9/swaddle_motion/data/subject<k>_swmo_<level>.mkv (+ .json, preview).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'dataaugment'))
from src.motion_fast import FastMotionAugmenter, make_units  # noqa: E402
from src.occlusion import render_cloth  # noqa: E402
from src.video_utils import FFV1Writer, detect_subject_box, open_video, sha256, smooth_noise, verify_output  # noqa: E402

OUT = HERE/'data'
SUBJECTS = (1, 3, 4)
LEVELS = {'medium': 0.0, 'large': -0.25}
BLANKET_BGR = (214, 196, 172)
WAVE_AMP, WAVE_LEN, FEATHER = 6.0, 120.0, 4.0
DRIFT_FACE, DRIFT_HZ = 0.05, 0.05
SEED = 20261117


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mcfg = yaml.safe_load((ROOT/'dataaugment/config/motion_fast_config.yaml').read_text())['motion_fast']
    sev = mcfg['severities']['large']
    tiles = []
    for subj in SUBJECTS:
        src = ROOT/f'data/ubfc/subject{subj}/vid.avi'
        reader = open_video(src)
        n, fps, W, H = reader.n_frames, reader.fps, reader.width, reader.height
        face = detect_subject_box(reader)
        chin, fh = face['y'] + face['h'], face['h']
        rng = np.random.default_rng([SEED, subj])
        drift = DRIFT_FACE * fh * smooth_noise(n, fps, DRIFT_HZ, rng)
        phase = 1.5 * smooth_noise(n, fps, DRIFT_HZ, rng)
        cloth = render_cloth(H, W, BLANKET_BGR, rng)
        units = make_units(n, fps, mcfg, np.random.default_rng([SEED, subj, 1]))
        meta = {'width': W, 'height': H, 'fps': fps}
        augs = {lv: FastMotionAugmenter(mcfg, sev, units, meta, face) for lv in LEVELS}
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)[:, None]
        before = sha256(src)
        writers = {lv: FFV1Writer(OUT/f'subject{subj}_swmo_{lv}.mkv', fps, (W, H)) for lv in LEVELS}
        snap = int(n * .45)
        for i, frame in enumerate(reader.frames(n)):
            for lv, off in LEVELS.items():
                top = chin + off * fh + drift[i] + WAVE_AMP * np.sin(2 * np.pi * xs / WAVE_LEN + phase[i])
                alpha = np.clip((ys - top[None, :]) / FEATHER + .5, 0, 1)[:, :, None]
                sw = np.clip(alpha * cloth + (1 - alpha) * frame.astype(np.float32), 0, 255).round().astype(np.uint8)
                out = augs[lv].apply(sw, i)
                writers[lv].write(out)
                if i == snap:
                    t = out.copy()
                    cv2.putText(t, f's{subj} swaddle {lv} + fast motion', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
                    tiles.append(cv2.resize(t, None, fx=.5, fy=.5))
        for w in writers.values():
            w.close()
        after = sha256(src)
        for lv, off in LEVELS.items():
            p = OUT/f'subject{subj}_swmo_{lv}.mkv'
            q = verify_output(p, writers[lv].md5)
            rec = dict(source=str(src), subject=subj, swaddle_level=lv, top_edge_rule=f'chin + {off} x face height',
                       motion='motion_fast large', motion_summary=augs[lv].summary(), face_box=face, seed=SEED,
                       frames=q['frame_count_decoded'], lossless_bit_exact=q['lossless_bit_exact'], original_unchanged=before == after,
                       checks_passed=bool(q['lossless_bit_exact'] and q['frame_count_decoded'] == n and before == after))
            p.with_suffix('.json').write_text(json.dumps(rec, indent=2, default=str))
            print(f'subject{subj}', lv, 'peak shift', rec['motion_summary']['peak_abs_dx_px'], 'checks',
                  'PASS' if rec['checks_passed'] else 'FAIL', flush=True)
    cv2.imwrite(str(OUT/'preview_swmo.png'), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))


if __name__ == '__main__':
    main()
