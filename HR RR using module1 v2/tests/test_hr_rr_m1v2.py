import json

import numpy as np
import pandas as pd

from module2.respiration import RRConfig
from hr_rr_m1v2 import rr, validity as va

CFG = dict(rois=['UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN', 'THORACO_ABDOMINAL'], chest_group=['UPPER_CHEST', 'LOWER_CHEST'],
           abdomen_group=['ABDOMEN'], weight_temperature=.1, surrogates=19, shuffle_block_sec=.25, null_percentile=95.,
           agreement_brpm=5., reference_flow_width=320)


def test_labels():
    assert va.label(False, True) == 'not_assessable'
    assert va.label(True, True) == 'validated'
    assert va.label(True, None) == 'present' and va.label(True, False) == 'present'


def synthetic(rate_brpm, seconds=30, fs=20., noise=.02, breathing=True, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    rows = []
    for roi in CFG['rois']:
        amp = 1.5 if breathing else 0.
        disp = amp * np.sin(2 * np.pi * rate_brpm / 60 * t + rng.uniform(0, 1))
        vel = np.diff(disp, prepend=disp[0]) + noise * rng.standard_normal(len(t))
        for f, v in enumerate(vel):
            rows.append(dict(frame_index=f, roi_name=roi, available=True, residual_dx=0.2 * v, residual_dy=v, q_v2=.5,
                             infant_box=json.dumps([0, 0, 300, 400])))
    return pd.DataFrame(rows), dict(video_id='VID_900', duration_sec=seconds, width=320)


def test_rr_recovers_and_validates_a_true_breathing_rhythm():
    rows, record = synthetic(45)
    out = rr.analyse(record, rows, CFG, RRConfig(), 0)
    assert out
    rates = [r['rr_brpm'] for r in out if r['rr_brpm'] is not None]
    assert abs(np.median(rates) - 45) < 2
    assert np.mean([r['label'] == 'validated' for r in out]) > .8


def test_rr_does_not_validate_noise():
    rows, record = synthetic(45, breathing=False, noise=.5)
    out = rr.analyse(record, rows, CFG, RRConfig(), 0)
    assert np.mean([r['label'] == 'validated' for r in out]) < .2


def _pulse_video(path, bpm, seconds=16, fps=30., size=(240, 320), pulsing=True, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    h, w = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    base = np.array([120., 150., 200.])            # BGR skin tone
    for i in range(int(seconds * fps)):
        s = np.sin(2 * np.pi * bpm / 60 * i / fps) if pulsing else 0.
        colour = base * (1 + np.array([.004, .012, .006]) * s)
        frame = np.full((h, w, 3), 40, np.uint8)
        patch = colour + rng.normal(0, 2, (160, 200, 3))
        frame[40:200, 60:260] = np.clip(patch, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return int(seconds * fps)


def _hr_rows(frames):
    poly = json.dumps([[70, 50], [250, 50], [250, 190], [70, 190]])
    return pd.DataFrame([dict(frame_index=f, roi_name='FACE', available=True, selected=True, weight=1., pixel_polygon=poly,
                              motion_magnitude=0.) for f in range(frames)])


def test_hr_recovers_and_validates_a_true_pulse(tmp_path):
    from module2.extraction import ExtractionConfig
    from module2.rppg import HRConfig
    from run_paper_evaluation import pulse
    from hr_rr_m1v2 import hr
    path = tmp_path / 'pulse.avi'
    n = _pulse_video(path, 120)
    record = dict(video_id='VID_901', fps=30., frame_count=n, video_path=str(path))
    cfg = dict(surrogates=19, null_percentile=95., agreement_bpm=6., null='shuffle', shuffle_block_sec=.25)
    out = hr.analyse(record, _hr_rows(n), {}, cfg, HRConfig(), ExtractionConfig(skin_mode='none'), pulse, 0)
    rates = [r['hr_bpm'] for r in out if r['hr_bpm'] is not None]
    assert abs(np.median(rates) - 120) < 3
    assert np.mean([r['label'] == 'validated' for r in out]) > .8


def test_hr_does_not_validate_a_static_patch(tmp_path):
    from module2.extraction import ExtractionConfig
    from module2.rppg import HRConfig
    from run_paper_evaluation import pulse
    from hr_rr_m1v2 import hr
    path = tmp_path / 'static.avi'
    n = _pulse_video(path, 120, pulsing=False)
    record = dict(video_id='VID_902', fps=30., frame_count=n, video_path=str(path))
    cfg = dict(surrogates=19, null_percentile=95., agreement_bpm=6., null='shuffle', shuffle_block_sec=.25)
    out = hr.analyse(record, _hr_rows(n), {}, cfg, HRConfig(), ExtractionConfig(skin_mode='none'), pulse, 0)
    assert np.mean([r['label'] == 'validated' for r in out]) < .2
