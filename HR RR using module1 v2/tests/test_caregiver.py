"""Caregiver contact add-on: episode building, SAM 2 mask-growth flag, window labels (no GPU / models needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


det = load('detect_caregiver_contact')
app = load('apply_caregiver_labels')


def frames_from(contact, fps=10.):
    return pd.DataFrame(dict(video_id='V', timestamp_sec=np.arange(len(contact)) / fps, contact_raw=contact,
                             adult_in_view=contact))


def test_episode_smoothing_merge_and_padding():
    on = np.zeros(300, bool)
    on[100:150] = True                       # 10.0-15.0 s
    on[155:170] = True                       # gap 0.5 s -> merged
    on[250] = True                           # single frame -> removed by the majority vote
    segs = det.segments(frames_from(on), 10., 30.)
    assert len(segs) == 1
    s = segs[0]
    assert abs(s['start_sec'] - 9.95) < 1e-6 and abs(s['end_sec'] - 16.95) < 1e-6
    assert abs(s['pad_start_sec'] - 8.95) < 1e-6 and abs(s['pad_end_sec'] - 17.95) < 1e-6


def test_padding_is_clipped_to_the_video():
    on = np.zeros(100, bool)
    on[:20] = True
    on[90:] = True
    segs = det.segments(frames_from(on), 10., 10.)
    assert segs[0]['pad_start_sec'] == 0. and segs[-1]['pad_end_sec'] == 10.


def test_mask_growth_flags_contact_but_shrink_does_not():
    area = np.full(200, 1000.)
    area[100:140] = 1800.                    # caregiver absorbed into the mask: x1.8
    area[160:] = 500.                        # SAM 2 self-correction / occlusion: x0.5
    rows = pd.DataFrame(dict(frame_index=np.arange(200) * 3, mask_area=area, mask_reliable=True))
    ratio, changed, shrink, ref = det.mask_change_flags(rows, 30.)
    assert changed[100:140].all() and not changed[:100].any() and not changed[140:].any()
    assert ref[120] == 99                     # the last clean mask is the reference during the change
    assert shrink[160] and not changed[160:].any()


def test_infant_and_adult_split():
    region = dict(box=np.array([400., 300., 1400., 900.]), mask=np.zeros((1080, 1920), bool))
    region['mask'][300:900, 400:1400] = True
    assert det.outside_fraction(np.array([450., 320., 1380., 880.]), region['box']) == 0.   # the infant itself
    assert det.outside_fraction(np.array([0., 0., 700., 600.]), region['box']) > .5           # caregiver from the corner


def test_window_labels_and_summary():
    windows = pd.DataFrame(dict(video_id='V', window_start_sec=[0., 5., 10., 20.], window_end_sec=[8., 13., 18., 28.],
                                window_center_sec=[4., 9., 14., 24.], label=['validated', 'validated', 'present', 'validated'],
                                hr_bpm=[120., 125., 130., 140.]))
    segs = pd.DataFrame(dict(video_id=['V'], start_sec=[14.], end_sec=[16.], pad_start_sec=[13.], pad_end_sec=[17.]))
    frames = frames_from(np.zeros(300, bool))
    t = app.label_windows(windows, segs, frames)
    assert t.caregiver_intervention.tolist() == [False, False, True, False]   # 5-13 s touches 13.0 only at the edge
    assert t.label_with_caregiver.tolist() == ['validated', 'validated', 'caregiver_intervention', 'validated']
    assert t.label.tolist() == windows.label.tolist()                          # original labels / values unchanged
    s = app.summarize(t, 'HR', 'hr_bpm').iloc[0]
    assert s.intervention_windows == 1 and s.validated_windows_excl_intervention == 3
    assert s.hr_bpm_median_validated_excl_intervention == 125.


def test_evaluation_excludes_intervention(tmp_path):
    windows = pd.DataFrame(dict(video_id='V', window_start_sec=[0., 10.], window_end_sec=[8., 18.],
                                window_center_sec=[4., 14.], label=['validated', 'validated'], hr_bpm=[120., 180.]))
    ref = tmp_path/'ref.csv'
    pd.DataFrame(dict(video_id='V', timestamp_sec=[4., 14.], hr_bpm=[121., 125.])).to_csv(ref, index=False)
    segs = pd.DataFrame(dict(video_id=['V'], start_sec=[12.], end_sec=[15.], pad_start_sec=[11.], pad_end_sec=[16.]))
    t = app.label_windows(windows, segs, frames_from(np.zeros(200, bool)))
    all_w = app.evaluate(t, ref, 'hr_bpm', 'hr_bpm', 5.)
    excl = app.evaluate(t[~t.caregiver_intervention], ref, 'hr_bpm', 'hr_bpm', 5.)
    assert all_w.mae.iloc[1] == 28. and excl.mae.iloc[1] == 1.
