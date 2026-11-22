"""HR-specific region set: the Module 1 'selected' region (chest) carries no pulse, the face regions do."""
import json

import cv2
import numpy as np
import pandas as pd

from module2.extraction import ExtractionConfig
from module2.rppg import HRConfig
from run_paper_evaluation import pulse
from hr_rr_m1v2 import hr

BOXES = {                       # (y1, y2, x1, x2)
    'FACE': (20, 110, 40, 150),
    'FOREHEAD': (20, 50, 40, 150),
    'LEFT_CHEEK': (60, 100, 40, 90),
    'RIGHT_CHEEK': (60, 100, 100, 150),
    'UPPER_CHEST': (130, 230, 60, 280),
}
CFG = dict(surrogates=19, null_percentile=95., agreement_bpm=6., null='shuffle', shuffle_block_sec=.25,
           hr_specific=dict(candidates=['FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE', 'UPPER_CHEST'],
                            k=3, min_skin_px=150, min_skin_fraction=.3, min_available_fraction=.9, max_overlap=.3,
                            history_sec=20, evidence_scale_db=2., region_surrogates=9, hysteresis_delta=.1,
                            hysteresis_windows=3, min_relative_score=.25))


def video(path, bpm, seconds=30, fps=30., face_pulse=True, seed=0):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (320, 240))
    base = np.array([120., 150., 200.])
    y1, y2, x1, x2 = BOXES['FACE']
    c1, c2, d1, d2 = BOXES['UPPER_CHEST']
    for i in range(int(seconds * fps)):
        s = np.sin(2 * np.pi * bpm / 60 * i / fps) if face_pulse else 0.
        frame = np.full((240, 320, 3), 40, np.uint8)
        face = base * (1 + np.array([.004, .012, .006]) * s) + rng.normal(0, 2, (y2 - y1, x2 - x1, 3))
        chest = base + rng.normal(0, 2, (c2 - c1, d2 - d1, 3))
        frame[y1:y2, x1:x2] = np.clip(face, 0, 255).astype(np.uint8)
        frame[c1:c2, d1:d2] = np.clip(chest, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return int(seconds * fps)


def rows_for(n):
    out = []
    for f in range(n):
        for name, (y1, y2, x1, x2) in BOXES.items():
            out.append(dict(frame_index=f, roi_name=name, available=True, selected=name == 'UPPER_CHEST',
                            weight=1. if name == 'UPPER_CHEST' else 0., q_v2=.9 if name == 'UPPER_CHEST' else .7,
                            motion_magnitude=0., pixel_polygon=json.dumps([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])))
    return pd.DataFrame(out)


def run(tmp_path, mode, face_pulse=True, bpm=130):
    path = tmp_path / f'{mode}_{face_pulse}.avi'
    n = video(path, bpm, face_pulse=face_pulse)
    record = dict(video_id='VID_950', fps=30., frame_count=n, video_path=str(path))
    cfg = dict(CFG, region_set=mode)
    return pd.DataFrame(hr.analyse(record, rows_for(n), {}, cfg, HRConfig(), ExtractionConfig(skin_mode='none'), pulse, 0))


def test_module1_mode_on_a_pulseless_chest_does_not_validate(tmp_path):
    w = run(tmp_path, 'module1')
    assert w.label.eq('validated').mean() < .2


def test_hr_specific_moves_to_the_pulsing_face_and_recovers_hr(tmp_path):
    w = run(tmp_path, 'hr_specific')
    late = w[w.selection_source == 'history']
    assert len(late) > 5
    chosen = [set(json.loads(x)) for x in late.hr_regions]
    assert all('UPPER_CHEST' not in c for c in chosen[3:])
    assert all(not ('FACE' in c and ({'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK'} & c)) for c in chosen)   # no overlap
    v = late[late.label == 'validated']
    assert len(v) >= .6 * len(late)
    assert abs(v.hr_bpm.median() - 130) < 3


def test_hr_specific_does_not_validate_when_nothing_pulses(tmp_path):
    w = run(tmp_path, 'hr_specific', face_pulse=False)
    assert w.label.eq('validated').mean() < .2
