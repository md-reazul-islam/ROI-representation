import json

import numpy as np
import pandas as pd
import pytest

from module2.events import (EventConfig, cluster_events, detect_events, match_events, rolling_median, runs,
                            summarize_clusters, timeline)
from test_rr import load_script, vitals_run  # noqa: F401  (fixture)


FS = 30.
NAMES = ['FOREHEAD', 'LEFT_CHEEK', 'UPPER_CHEST', 'LEFT_ARM']


def synthetic_infant(seconds=120., seed=0):
    """Pulse 150 bpm with a bradycardia (85 bpm, 38-50 s); breathing 40 /min with a 22 s apnoea (30-52 s) and a
    12 s pause (90-102 s) after paradoxical breathing (74-88 s); red pulsatility x1.6 (desaturation, 44-62 s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    hr = np.where((t >= 38) & (t < 50), 85., 150.)
    phase = 2 * np.pi * np.cumsum(hr / 60.) / FS
    direction = np.tile([.33, .77, .53], (len(t), 1))
    direction[(t >= 44) & (t < 62), 0] *= 1.6
    base = np.array([200., 140., 110.])
    rgb = np.stack([base * (1 + .004 * np.sin(phase)[:, None] * direction + .0005 * rng.standard_normal((len(t), 3)))
                    for _ in NAMES], axis=1)
    weights = np.full((len(t), len(NAMES)), 1. / len(NAMES))
    paused = ((t >= 30) & (t < 52)) | ((t >= 90) & (t < 102))
    waveform = np.where(paused, .02, 1.) * np.sin(2 * np.pi * 40 / 60 * t) + .01 * rng.standard_normal(len(t))
    gross = np.full(len(t), .01)
    centres = np.arange(4., seconds - 3.)
    rr_paused = ((centres >= 30) & (centres < 52)) | ((centres >= 90) & (centres < 102))
    paradox = (centres >= 74) & (centres < 88)
    rr = pd.DataFrame(dict(window_center_sec=centres, rr_brpm=np.where(rr_paused, np.nan, 40.),
                           taa_phase_deg=np.where(rr_paused, np.nan, np.where(paradox, 170., 20.)),
                           taa_coherence=np.where(rr_paused, np.nan, .9)))
    return rgb, weights, waveform, gross, rr


@pytest.fixture(scope='module')
def detected():
    rgb, weights, waveform, gross, rr = synthetic_infant()
    cfg = EventConfig()
    line = timeline(rgb, weights, NAMES, FS, waveform, gross, rr, cfg)
    return line, detect_events(line, cfg), cfg


def test_runs_merge_short_gaps_and_drop_short_runs():
    flag = [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
    assert runs(flag, 1.) == [(1, 3), (4, 7), (10, 11)]
    assert runs(flag, 1., merge_gap_sec=1.) == [(1, 7), (10, 11)]
    assert runs(flag, 1., merge_gap_sec=1., min_sec=3.) == [(1, 7)]


def test_rolling_median_falls_back_to_the_whole_series():
    values = np.array([1., 100., 100., 100., 4., 5., 6.])
    valid = np.array([True, False, False, False, True, True, True])
    # Span 2 s at 1 s steps = 3 points; 2 valid points needed, else the median of all valid points (4.5).
    assert np.allclose(rolling_median(values, valid, 1., 2., 2.), [4.5, 4.5, 4.5, 4.5, 4.5, 5., 5.5])
    assert np.isnan(rolling_median(values, valid & False, 1., 2., 2.)).all()


def test_timeline_tracks_heart_rate_breathing_and_spo2_proxy(detected):
    line, _, _ = detected
    normal = line[(line.time_sec > 10) & (line.time_sec < 25)]
    assert normal.hr_assessable.all() and (normal.hr_bpm - 150).abs().max() < 3.
    assert (normal.hr_sites_agreeing == 3).all() and not normal.bradycardia.any()   # head, trunk, arms
    low = line[(line.time_sec > 41) & (line.time_sec < 47)]
    assert (low.hr_bpm - 85).abs().max() < 3. and low.bradycardia.all()
    assert (line.hr_baseline_bpm[line.hr_assessable] - 150).abs().max() < 3.
    assert line.breathing_pause[(line.time_sec > 33) & (line.time_sec < 49)].all()
    assert not line.breathing_pause[(line.time_sec > 5) & (line.time_sec < 27)].any()
    assert line.ror_ratio[(line.time_sec > 50) & (line.time_sec < 57)].min() > 1.4
    assert (line.ror_ratio[(line.time_sec > 70) & (line.time_sec < 115)].dropna() - 1).abs().max() < .1


def test_apnoea_with_bradycardia_and_desaturation_is_one_abd_event(detected):
    _, events, cfg = detected
    by_type = {}
    for event in events:
        by_type.setdefault(event['event_type'], []).append(event)
    long, short = sorted(by_type['apnoea'], key=lambda e: e['start_sec'])
    assert abs(long['start_sec'] - 30) <= 2 and abs(long['end_sec'] - 52) <= 2
    assert long['prolonged'] and long['apnoea_type'] == 'central' and long['clinically_significant']
    [brady] = by_type['bradycardia']
    assert 36 <= brady['start_sec'] <= 40 and 48 <= brady['end_sec'] <= 52 and brady['hr_min_bpm'] < 90
    [desat] = by_type['desaturation']
    assert 40 <= desat['start_sec'] <= 46 and 60 <= desat['end_sec'] <= 66
    assert long['cluster_id'] == brady['cluster_id'] == desat['cluster_id'] and long['cluster_label'] == 'ABD'
    # The 12 s pause has no bradycardia / desaturation: below the 20 s AAP limit it is not a clinical apnoea
    # (and 28 s after the desaturation it is a new event); the paradoxical breathing just before it makes it
    # 'mixed', whereas the long apnoea, 22 s before that paradoxical episode, stays 'central'.
    assert abs(short['duration_sec'] - 12) <= 2 and not short['clinically_significant']
    assert short['apnoea_type'] == 'mixed' and short['cluster_label'] == 'A' and short['cluster_id'] != long['cluster_id']
    [paradox] = by_type['paradoxical_breathing']
    assert abs(paradox['start_sec'] - 74) <= 2 and paradox['taa_phase_median_deg'] > 150
    clusters = summarize_clusters(events)
    assert [c['label'] for c in clusters if c['clinically_significant']] == ['ABD']


def test_low_heart_rate_without_plausible_baseline_is_not_assessable():
    """A coherent low-frequency artefact over the whole clip must not become a bradycardia alarm."""
    rgb, weights, waveform, gross, rr = synthetic_infant(seconds=30.)
    t = np.arange(len(rgb)) / FS
    rgb = np.stack([np.array([200., 140., 110.]) * (1 + .004 * np.outer(np.sin(2*np.pi*65/60*t), [.33, .77, .53]))
                    for _ in NAMES], axis=1)
    line = timeline(rgb, weights, NAMES, FS, waveform, gross, rr.assign(rr_brpm=40.), EventConfig())
    assert line.hr_implausible_baseline.any() and not line.hr_assessable.any() and not line.bradycardia.any()


def test_gross_motion_is_not_an_apnoea():
    rgb, weights, waveform, gross, rr = synthetic_infant(seconds=60.)
    t = np.arange(len(waveform)) / FS
    still = (t >= 30) & (t < 52)
    gross = np.where(still, 1., gross)  # the 'pause' is a period of gross motion (e.g. handling)
    line = timeline(rgb, weights, NAMES, FS, waveform, gross, rr, EventConfig())
    events = detect_events(line, EventConfig())
    assert not any(e['event_type'] == 'apnoea' for e in events)
    assert not line.breath_assessable[(line.time_sec > 33) & (line.time_sec < 49)].any()


def test_clusters_join_events_within_the_association_window():
    cfg = EventConfig(assoc_sec=10.)
    events = [dict(event_type='apnoea', start_sec=0., end_sec=12., duration_sec=12., prolonged=False),
              dict(event_type='bradycardia', start_sec=18., end_sec=24., duration_sec=6.),
              dict(event_type='apnoea', start_sec=30., end_sec=42., duration_sec=12., prolonged=False),
              dict(event_type='desaturation', start_sec=60., end_sec=75., duration_sec=15.)]
    clustered = cluster_events(events, cfg)
    assert [e['cluster_id'] for e in clustered] == [1, 1, 2, 3]
    assert [e['cluster_label'] for e in clustered] == ['AB', 'AB', 'A', 'D']
    assert [e['clinically_significant'] for e in clustered] == [True, True, False, True]


def test_event_matching():
    detected = [dict(video_id='v', event_type='apnoea', start_sec=10., end_sec=30.),
                dict(video_id='v', event_type='apnoea', start_sec=100., end_sec=120.)]
    reference = [dict(video_id='v', event_type='apnoea', start_sec=12., end_sec=28.),
                 dict(video_id='v', event_type='apnoea', start_sec=200., end_sec=220.)]
    result = match_events(detected, reference, 'apnoea')
    assert result['sensitivity'] == .5 and result['ppv'] == .5


def test_end_to_end_events_cli(vitals_run):
    root, common, hr_bpm, rr_brpm = vitals_run
    assert load_script('run_rr').main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/rr',
                                                '--no-plots']) == 0
    assert load_script('run_hr_rppg').main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir',
                                                     'outputs/hr', '--no-plots']) == 0
    pd.DataFrame([dict(video_id='v', event_type='apnoea', start_sec=5., end_sec=25.)]).to_csv(root/'ref.csv', index=False)
    run = load_script('run_events')
    assert run.main(['--config', common[1], '--hr-dir', 'outputs/hr', '--rr-dir', 'outputs/rr',
                     '--output-dir', 'outputs/events', '--reference-events', 'ref.csv']) == 0
    output = root/'outputs/events'
    line = pd.read_csv(output/'event_timeline.csv')
    assert len(line) == 40 and line.breath_assessable.mean() > .9
    # The synthetic infant breathes throughout, paradoxically (chest and abdomen in anti-phase): no apnoea, one
    # paradoxical-breathing episode. Its pulse is on the face only, so no second site can confirm a low HR and
    # the bradycardia channel stays not assessable rather than silently 'normal'.
    assert not line.breathing_pause.any() and not line.bradycardia.any()
    assert not line.hr_assessable.any() and (line.hr_sites_agreeing <= 1).all()
    events = pd.read_csv(output/'events.csv')
    assert list(events.event_type) == ['paradoxical_breathing'] and events.taa_phase_median_deg.iloc[0] > 150
    summary = pd.read_csv(output/'event_summary.csv')
    assert summary.apnoea_events.iloc[0] == 0 and summary.breath_assessable_sec.iloc[0] > 18
    assert summary.hr_assessable_sec.iloc[0] == 0 and summary.paradoxical_breathing_events.iloc[0] == 1
    evaluation = pd.read_csv(output/'event_evaluation.csv')
    assert evaluation.set_index('event_type').sensitivity['apnoea'] == 0
    metadata = json.loads((output/'experiment_metadata.json').read_text())
    assert metadata['reference_used_for_detection'] is False
    assert (output/'plots/v_events.png').is_file()
    with pytest.raises(FileExistsError):
        run.main(['--config', common[1], '--hr-dir', 'outputs/hr', '--rr-dir', 'outputs/rr', '--output-dir', 'outputs/events'])
