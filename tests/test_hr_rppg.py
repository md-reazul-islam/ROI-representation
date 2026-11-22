import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from module2.evaluation import agreement, icc_2_1
from module2.extraction import ExtractionConfig, fusion_weights, hold_indices, mean_rgb
from module2.rppg import (ESTIMATORS, METHODS, HRConfig, analysis_windows, detrend_smoothness_priors, estimate_hr,
                          extract_pulse, kalman_smooth, signal_quality)


FS = 30.


def synthetic_rgb(hr_bpm=144., seconds=12., fs=FS, seed=0, noise=.002):
    """Skin RGB trace with a pulse along the blood-volume direction plus illumination drift and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    base = np.array([200., 140., 110.])
    pulse = np.outer(np.sin(2*np.pi*hr_bpm/60*t), [.33, .77, .53]) * .004
    drift = (.03*np.sin(2*np.pi*.15*t) + .01*np.sin(2*np.pi*.4*t))[:, None]  # common-mode illumination
    return base * (1 + pulse + drift + noise*rng.standard_normal((len(t), 3)))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('estimator', ESTIMATORS)
@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('hr_bpm', [100., 144., 175., 230.])
def test_methods_recover_known_heart_rate(method, hr_bpm, estimator):
    config = HRConfig()
    pulse = extract_pulse(synthetic_rgb(hr_bpm, noise=.001), FS, method, config)
    estimate, snr = estimate_hr(pulse[:int(8*FS)], FS, config, estimator=estimator)
    assert abs(estimate - hr_bpm) < 2.
    assert snr > config.snr_threshold_db


def test_ar_best_is_not_limited_by_fft_bin_spacing():
    config = HRConfig()
    t = np.arange(int(8*FS)) / FS
    x = np.sin(2*np.pi*143.3/60*t) + .2*np.random.default_rng(3).standard_normal(len(t))
    assert abs(estimate_hr(x, FS, config, estimator='ar_best')[0] - 143.3) < 1.


def test_smoothness_priors_detrending_removes_slow_drift_but_keeps_pulse():
    t = np.arange(600) / FS
    pulse = np.sin(2*np.pi*2.4*t)
    detrended = detrend_smoothness_priors(pulse + 5*t + 3*np.sin(2*np.pi*.05*t), 100.)
    assert np.corrcoef(detrended, pulse)[0, 1] > .99


def test_sqi_combines_snr_and_motion():
    config = HRConfig()
    assert signal_quality(10., 0., config) > .95
    assert signal_quality(10., .8, config) < .25          # strong infant motion
    assert signal_quality(-10., 0., config) < .05         # noise-dominated spectrum
    assert signal_quality(None, 0., config) == 0.


def test_kalman_follows_good_measurements_and_ignores_bad_ones():
    config = HRConfig()
    times = np.arange(10.)
    hr = [140.]*5 + [200.] + [141.]*4
    sqi = [1.]*5 + [.05] + [1.]*4
    smoothed = kalman_smooth(times, hr, sqi, config)
    assert abs(smoothed[5] - 140.) < 5.                  # an outlier with low SQI barely moves the estimate
    assert abs(smoothed[-1] - 141.) < 1.
    assert kalman_smooth([0., 1.], [None, 150.], [0., 1.], config) == [None, 150.]
    gap = kalman_smooth(np.arange(8.), [150., None, None, None, None, None, 140., None], [1.]*8, config)
    assert gap[:5] == [150.]*5 and gap[5] is None and gap[6] == 140.   # no flat line through a long signal loss


def test_long_gap_stays_missing_and_short_gap_is_filled():
    config = HRConfig()
    rgb = synthetic_rgb(seconds=20.)
    rgb[100:103] = np.nan          # 0.1 s: interpolated
    rgb[300:330] = np.nan          # 1.0 s: kept missing
    pulse = extract_pulse(rgb, FS, 'POS', config)
    assert np.isfinite(pulse[100:103]).all()
    assert np.isnan(pulse[300:330]).all()
    assert np.isfinite(pulse[:300]).mean() > .99 and np.isfinite(pulse[330:]).mean() > .99


def test_noise_only_is_flagged_unreliable():
    config = HRConfig()
    rng = np.random.default_rng(1)
    rgb = np.array([200., 140., 110.]) * (1 + .01*rng.standard_normal((360, 3)))
    _, snr = estimate_hr(extract_pulse(rgb, FS, 'POS', config), FS, config)
    assert signal_quality(snr, 0., config) < config.sqi_threshold


def test_analysis_windows_cover_short_and_long_clips():
    config = HRConfig()
    assert analysis_windows(int(4*FS), FS, config) == []
    assert analysis_windows(int(6*FS), FS, config) == [(0, 180)]
    windows = analysis_windows(int(12*FS), FS, config)
    assert windows[0] == (0, 240) and windows[-1][1] <= 360 and len(windows) == 5


def test_hold_indices_reuse_latest_processed_frame_within_limit():
    hold = hold_indices([0, 2, 3, 20], 25, FS, .25)
    assert hold[:5].tolist() == [0, 0, 2, 3, 3]
    assert hold[10] == 3 and hold[11] == -1          # 8 frames = 0.27 s > 0.25 s
    assert hold[24] == 20


def test_fusion_weights_follow_confidence_and_ignore_unusable_rois():
    config = ExtractionConfig(weight_smoothing_sec=0.)
    scores = np.array([[.8, .6, .2], [.8, np.nan, .2], [.1, .1, .1]])
    usable = np.array([[True, True, True], [True, True, True], [False, False, False]])
    weights = fusion_weights(scores, usable, config, FS)
    assert weights[0, 0] > weights[0, 1] > weights[0, 2]
    assert weights[1, 1] == 0 and weights[1].sum() == pytest.approx(1.)
    assert weights[2].sum() == 0


def test_mean_rgb_uses_skin_pixels_and_rejects_non_skin():
    image = np.zeros((100, 100, 3), np.uint8)
    image[:, :50] = (110, 140, 200)       # skin-like (BGR)
    image[:, 50:] = (200, 60, 30)         # blue blanket
    square = np.array([[10., 10.], [90., 10.], [90., 90.], [10., 90.]])
    value, count = mean_rgb(image, square, ExtractionConfig())
    assert value == pytest.approx((200., 140., 110.)) and count > 100
    blanket = np.array([[60., 10.], [95., 10.], [95., 90.], [60., 90.]])
    assert mean_rgb(image, blanket, ExtractionConfig())[0] is None


def test_adaptive_skin_mask_rejects_off_tone_pixels_inside_the_broad_skin_range():
    image = np.full((100, 100, 3), (110, 140, 200), np.uint8)   # infant skin
    image[40:60, :] = (95, 120, 215)                              # reddish tape: still in the broad YCrCb range
    square = np.array([[5., 5.], [95., 5.], [95., 95.], [5., 95.]])
    fixed, _ = mean_rgb(image, square, ExtractionConfig(skin_mode='fixed'))
    adaptive, _ = mean_rgb(image, square, ExtractionConfig(skin_mode='adaptive'))
    assert adaptive == pytest.approx((200., 140., 110.))
    assert fixed != pytest.approx((200., 140., 110.))


def test_dynamic_fusion_uses_signal_evidence_when_the_most_confident_roi_is_noisy():
    run = load_script('run_hr_rppg')
    config = HRConfig()
    t = np.arange(int(8*FS)) / FS
    rng = np.random.default_rng(5)
    noisy = rng.standard_normal(len(t))
    clean = np.sin(2*np.pi*150/60*t) + .3*rng.standard_normal(len(t))
    pulses = np.column_stack([noisy, clean])
    prior = np.array([.7, .3])                                    # Module 1 trusts the noisy ROI more
    goodness = np.array([10 ** (run.window_snr(pulses[:, r], FS, config) / 10) for r in range(2)])
    weights = run.window_weights(prior, goodness, np.array([.5, .5]), np.ones(2))
    assert weights['dynamic_fusion'][1] > weights['dynamic_fusion'][0]
    fused, _ = estimate_hr(run.weighted_pulse(pulses, weights['dynamic_fusion']), FS, config)
    assert abs(fused - 150.) < 2.


def test_agreement_metrics():
    result = agreement([150, 142, 160, 131], [148, 145, 158, 130])
    assert result['n'] == 4 and result['mae_bpm'] == pytest.approx(2.)
    assert result['bland_altman_bias_bpm'] == pytest.approx(.5)
    assert icc_2_1([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.)


@pytest.fixture
def module1_run(tmp_path):
    """A 12 s synthetic infant video (pulse 150 bpm on the face) with matching Module 1 outputs."""
    hr_bpm, seconds = 150., 12.
    (tmp_path/'configs').mkdir()
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'))))
    (tmp_path/'data').mkdir()
    video = tmp_path/'data/v.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), FS, (160, 120))
    t = np.arange(int(seconds*FS)) / FS
    for value in np.sin(2*np.pi*hr_bpm/60*t):
        frame = np.full((120, 160, 3), (200, 60, 30), np.uint8)
        face = np.array([110, 140, 200], float) * (1 + .03*value*np.array([.53, .77, .33]))
        frame[20:100, 40:120] = np.clip(np.rint(face), 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    pd.DataFrame([dict(video_id='v', filename='v.avi', split='test', fps=FS)]).to_csv(tmp_path/'manifest.csv', index=False)
    processed = sorted({int(round(k/20*FS)) for k in range(int(seconds*20))})
    rows, recovery = [], []
    regions = dict(FOREHEAD=[[50, 25], [110, 25], [110, 45], [50, 45]], LEFT_CHEEK=[[50, 55], [75, 55], [75, 90], [50, 90]],
                   RIGHT_CHEEK=[[85, 55], [110, 55], [110, 90], [85, 90]], UPPER_FACE=[[45, 22], [115, 22], [115, 60], [45, 60]],
                   FACE=[[42, 22], [118, 22], [118, 98], [42, 98]])
    for frame in processed:
        for index, (name, polygon) in enumerate(regions.items()):
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            rows.append(dict(video_id='v', split='test', frame_index=frame, timestamp_sec=frame/FS, roi_name=name,
                             roi_status='AVAILABLE', polygon=json.dumps(polygon),
                             bbox=json.dumps([min(xs), min(ys), max(xs), max(ys)]), q=.7 - .05*index, alpha=.2,
                             visible_skin_proportion=.8))
        recovery.append(dict(video_id='v', frame_index=frame, final_infant_bbox=json.dumps([30, 10, 130, 110])))
    (tmp_path/'outputs/confidence_sam2').mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path/'outputs/confidence_sam2/roi_confidence.csv', index=False)
    (tmp_path/'outputs/recovery').mkdir(parents=True)
    pd.DataFrame(recovery).to_csv(tmp_path/'outputs/recovery/recovery_results.csv', index=False)
    return tmp_path, ['--config', str(config), '--manifest', 'manifest.csv'], hr_bpm


def test_end_to_end_hr_monitor_and_evaluation(module1_run):
    root, common, hr_bpm = module1_run
    run = load_script('run_hr_rppg')
    assert run.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr_rppg_sam2']) == 0
    output = root/'outputs/hr_rppg_sam2'
    windows = pd.read_csv(output/'hr_windows.csv')
    assert set(windows.strategy) == {'dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box',
                                     'consensus_fusion'}
    primary = windows[(windows.method == 'POS') & (windows.strategy == 'dynamic_fusion')]
    assert len(primary) == 5 and primary.reliable.all()
    assert (primary.hr_bpm - hr_bpm).abs().max() < 2. and (primary.hr_kalman_bpm - hr_bpm).abs().max() < 2.
    top1 = windows[(windows.method == 'POS') & (windows.strategy == 'top1')]
    assert set(top1.top_roi) == {'FOREHEAD'}              # highest Module 1 confidence
    signals = pd.read_csv(output/'pulse_signals.csv')
    weights = signals.filter(like='weight_').to_numpy()
    assert len(signals) == 360 and np.allclose(weights.sum(axis=1), 1.)
    assert (output/'plots/v_hr.png').is_file()
    with pytest.raises(FileExistsError):
        run.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr_rppg_sam2'])

    render = load_script('render_hr_monitor')
    assert render.main(common + ['--hr-dir', 'outputs/hr_rppg_sam2', '--scale', '1']) == 0
    movie = output/'monitor_videos/v_hr_monitor_POS_dynamic_fusion.mp4'
    capture = cv2.VideoCapture(str(movie))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 360
    capture.release()

    reference = pd.DataFrame(dict(video_id='v', timestamp_sec=np.arange(13.), hr_bpm=hr_bpm))
    reference.to_csv(root/'reference.csv', index=False)
    evaluate = load_script('evaluate_hr')
    assert evaluate.main(['--config', common[1], '--hr-dir', 'outputs/hr_rppg_sam2', '--reference', 'reference.csv']) == 0
    metrics = pd.read_csv(output/'evaluation/hr_metrics.csv')
    best = metrics[(metrics.method == 'POS') & (metrics.strategy == 'dynamic_fusion') & (metrics.estimate == 'hr_kalman_bpm') &
                   (metrics.level == 'window') & (metrics.subset == 'all_windows')].iloc[0]
    assert best.mae_bpm < 2.
