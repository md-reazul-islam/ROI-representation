"""Demonstration video (NOT part of the pre-registered test; see ../PREREGISTRATION.md, Deviations / additions).

    python occlusion_position_test/demo_centre/make_demo_video.py

UBFC subject 1. One large hand reaching in sideways from the image right (fingers pointing to image left) covers the
nose, the centre of the face and the mid-cheeks, and leaves the forehead free. Palm width 0.6 face widths, palm centre
at face centre + (0.10 w, 0.15 h). Geometry chosen BEFORE any method ran on it, from coverage only: centre 0.94,
mid-cheeks 0.83-0.91, forehead 0.00, estimated fixed box 0.34. Same schedule, jitter, shadow and checks as the
pre-registered subject-1 videos. Output: demo_centre/subject1_democentre.mkv (+ .json, _frames.csv, preview).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'scripts'))
import occluder as oc  # noqa: E402
from make_occlusion_videos import b0_box  # noqa: E402
from src.video_utils import FFV1Writer, detect_subject_box, open_video, sha256, spectral_check, verify_output  # noqa: E402

PALM, DX, DY, ANGLE = 0.60, 0.10, 0.15, 90
SUBJ = 1


def main():
    oc.SIZES['demo'] = PALM
    oc.POSITIONS['demo'] = (DX, DY, ANGLE)
    src = oc.ROOT/f'data/ubfc/subject{SUBJ}/vid.avi'
    reader = open_video(src)
    n, fps, W, H = reader.n_frames, reader.fps, reader.width, reader.height
    face = detect_subject_box(reader)
    regions = oc.face_regions(face)
    regions['fixed_box_est'] = b0_box(f'NEW_subject{SUBJ}')
    vis, episodes = oc.schedule(n, fps, SUBJ)
    jit = oc.jitter(n, fps, SUBJ)
    hand = oc.Hand(face, 'demo', 'demo', W, H, SUBJ)
    name = f'subject{SUBJ}_democentre'
    before = sha256(src)
    writer = FFV1Writer(HERE/f'{name}.mkv', fps, (W, H))
    rows = []
    for i, frame in enumerate(reader.frames(n)):
        out, a = hand.apply(frame, i, vis[i], jit)
        writer.write(out)
        x, y, ang = hand.placement(i, jit)
        r = dict(frame=i, time_sec=round(i / fps, 4), visibility=round(float(vis[i]), 4), hand_x=round(x, 2),
                 hand_y=round(y, 2), hand_angle_deg=round(ang, 2))
        for k, box in regions.items():
            r[f'cov_{k}'] = 0.0 if a is None else round(oc.covered(a, box), 4)
        rows.append(r)
    writer.close()
    after = sha256(src)
    q = verify_output(HERE/f'{name}.mkv', writer.md5)
    fr = pd.DataFrame(rows)
    fr.to_csv(HERE/f'{name}_frames.csv', index=False)
    bands = {'rr': (0.15, 1.5), 'hr': (0.7, 4.0)}
    sp = {'visibility': spectral_check(vis, fps, bands), **{f'jitter_{k}': spectral_check(v, fps, bands) for k, v in jit.items()}}
    flags = sorted(f'{k}:{b}' for k, d in sp.items() for b, r in d.items() if isinstance(r, dict) and r.get('narrowband_flag'))
    on = fr[fr.visibility >= .99]
    rec = dict(source=str(src), purpose='demonstration, not pre-registered', palm_width_face=PALM, position_rule=(DX, DY, ANGLE),
               face_box=face, regions=regions, episodes_sec=episodes, hand_present_pct=round(100 * float((vis > 0).mean()), 1),
               coverage_while_present={k[4:]: round(float(on[k].mean()), 3) for k in fr.columns if k.startswith('cov_')},
               frames=q['frame_count_decoded'], lossless_bit_exact=q['lossless_bit_exact'], narrowband_flags=flags,
               original_unchanged=before == after)
    rec['checks_passed'] = bool(q['lossless_bit_exact'] and q['frame_count_decoded'] == n and before == after
                                and not [f for f in flags if f.endswith(':hr')])
    (HERE/f'{name}.json').write_text(json.dumps(rec, indent=2, default=str))
    mid = int((episodes[0][0] + episodes[0][1]) / 2 * fps)
    t, _ = hand.apply(reader.read(mid), mid, 1.0, jit)
    t = t.copy()
    x1, y1, x2, y2 = (int(v) for v in regions['fixed_box_est'])
    cv2.rectangle(t, (x1, y1), (x2, y2), (255, 120, 0), 2)
    cv2.imwrite(str(HERE/f'preview_{name}.png'), np.hstack([reader.read(mid), t]))
    print(json.dumps({k: rec[k] for k in ('coverage_while_present', 'hand_present_pct', 'narrowband_flags', 'checks_passed')}, indent=1))


if __name__ == '__main__':
    main()
