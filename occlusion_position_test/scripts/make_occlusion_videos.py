"""Write the pre-registered occlusion-position test videos (../PREREGISTRATION.md).

    python occlusion_position_test/scripts/make_occlusion_videos.py

UBFC subjects 1, 3, 4 x 5 hand positions x 2 hand sizes = 30 lossless FFV1 videos in occlusion_position_test/data/:
subject<k>_occpos_<position>_<size>.mkv (+ _frames.csv with per-frame hand placement and coverage, + .json checks).
The hand is absent for the first 10 s, then present in aperiodic episodes (10-14 s on, 4-8 s off). All positions and
sizes of one subject share the same schedule and jitter (common random numbers), so they differ only in where the hand is.
Fixed-box coverage is measured against the first-frame person box of the ORIGINAL video (Module 1 run
ubfc_flat_38d8f553); the report re-measures it against the box of the new run.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import occluder as oc  # noqa: E402
from src.video_utils import (FFV1Writer, detect_subject_box, open_video, sha256, spectral_check,  # noqa: E402
                             verify_output)

ROOT = oc.ROOT
OUT = HERE.parent/'data'
B0_SOURCE_RUN = ROOT/'runs/ubfc_flat_38d8f553'


def b0_box(video_id, fraction=.5):
    sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
    from run_paper_evaluation import boxes_of
    rec = pd.read_csv(B0_SOURCE_RUN/'outputs/recovery/recovery_results.csv', low_memory=False)
    boxes = boxes_of(rec[rec.video_id == video_id])
    x1, y1, x2, y2 = next(boxes[f] for f in sorted(boxes) if boxes[f] is not None)
    dx, dy = (x2 - x1) * (1 - fraction) / 2, (y2 - y1) * (1 - fraction) / 2
    return (x1 + dx, y1 + dy, x2 - dx, y2 - dy)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bands = {'rr': (0.15, 1.5), 'hr': (0.7, 4.0)}
    summary = []
    for subj in oc.SUBJECTS:
        src = ROOT/f'data/ubfc/subject{subj}/vid.avi'
        reader = open_video(src)
        n, fps, W, H = reader.n_frames, reader.fps, reader.width, reader.height
        face = detect_subject_box(reader)
        regions = oc.face_regions(face)
        regions['fixed_box_est'] = b0_box(f'NEW_subject{subj}')
        vis, episodes = oc.schedule(n, fps, subj)
        jit = oc.jitter(n, fps, subj)
        before = sha256(src)
        combos = [(p, s) for p in oc.POSITIONS for s in oc.SIZES]
        hands = {c: oc.Hand(face, c[1], c[0], W, H, subj) for c in combos}
        name = {c: f'subject{subj}_occpos_{c[0]}_{c[1]}' for c in combos}
        writers = {c: FFV1Writer(OUT/f'{name[c]}.mkv', fps, (W, H)) for c in combos}
        rows = {c: [] for c in combos}
        for i, frame in enumerate(reader.frames(n)):
            for c in combos:
                out, a = hands[c].apply(frame, i, vis[i], jit)
                writers[c].write(out)
                x, y, ang = hands[c].placement(i, jit)
                r = dict(frame=i, time_sec=round(i / fps, 4), visibility=round(float(vis[i]), 4),
                         hand_x=round(x, 2), hand_y=round(y, 2), hand_angle_deg=round(ang, 2))
                for k, box in regions.items():
                    r[f'cov_{k}'] = 0.0 if a is None else round(oc.covered(a, box), 4)
                rows[c].append(r)
            if i % 300 == 0:
                print(f'subject{subj} frame {i}/{n}', flush=True)
        for w in writers.values():
            w.close()
        after = sha256(src)
        spectra = {'visibility': spectral_check(vis, fps, bands),
                   **{f'jitter_{k}': spectral_check(v, fps, bands) for k, v in jit.items()}}
        flags = {f'{k}:{b}' for k, d in spectra.items() for b, r in d.items() if isinstance(r, dict) and r.get('narrowband_flag')}
        for c in combos:
            p = OUT/f'{name[c]}.mkv'
            q = verify_output(p, writers[c].md5)
            fr = pd.DataFrame(rows[c])
            fr.to_csv(OUT/f'{name[c]}_frames.csv', index=False)
            on = fr[fr.visibility >= .99]
            rec = dict(source=str(src), subject=subj, position=c[0], size=c[1], palm_width_face=oc.SIZES[c[1]],
                       position_rule=oc.POSITIONS[c[0]], face_box=face, regions=regions, episodes_sec=episodes,
                       hand_present_pct=round(100 * float((vis > 0).mean()), 1), seed=oc.SEED,
                       coverage_while_present={k[4:]: round(float(on[k].mean()), 3) for k in fr.columns if k.startswith('cov_')},
                       frames=q['frame_count_decoded'], lossless_bit_exact=q['lossless_bit_exact'], fps_in=fps,
                       fps_out=q['fps_container'], hr_band_narrowband_flags=sorted(f for f in flags if f.endswith(':hr')),
                       rr_band_narrowband_flags=sorted(f for f in flags if f.endswith(':rr')),
                       original_unchanged=before == after)
            rec['checks_passed'] = bool(q['lossless_bit_exact'] and q['frame_count_decoded'] == n and before == after
                                        and not rec['hr_band_narrowband_flags'])
            p.with_suffix('.json').write_text(json.dumps(rec, indent=2, default=str))
            summary.append(dict(video=name[c], subject=subj, position=c[0], size=c[1], frames=rec['frames'],
                                hand_present_pct=rec['hand_present_pct'],
                                **{f'cov_{k}': v for k, v in rec['coverage_while_present'].items()},
                                checks='PASS' if rec['checks_passed'] else 'FAIL'))
        # preview: one occluded frame per position (large) and per size (centre)
        mid = next(int((s + e) / 2 * fps) for s, e in episodes)
        frame = reader.read(mid)
        tiles = []
        for c in [(p, 'large') for p in oc.POSITIONS] + [('centre', 'small')]:
            t, _ = hands[c].apply(frame, mid, 1.0, jit)
            t = t.copy()
            x1, y1, x2, y2 = (int(v) for v in regions['fixed_box_est'])
            cv2.rectangle(t, (x1, y1), (x2, y2), (255, 120, 0), 2)
            cv2.putText(t, f'{c[0]} {c[1]}', (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            tiles.append(cv2.resize(t, None, fx=.5, fy=.5))
        cv2.imwrite(str(OUT/f'preview_subject{subj}.png'), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    s = pd.DataFrame(summary)
    s.to_csv(OUT/'videos_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(s.to_string(index=False))


if __name__ == '__main__':
    main()
