import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from module2.respiration import (RRConfig, background_flow, estimate_rr, fuse_spectra, respiratory_signal, rr_windows,
                                 signal_quality, window_spectrum)
from module2.rppg import detrend_smoothness_priors


FS = 30.


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def breathing_velocity(rr_brpm, seconds=20., amplitude=.5, noise=.02, seed=0, phase=0.):
    """Per-frame ROI velocity (px/frame) of a chest wall moving mostly vertically at rr_brpm."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    w = 2 * np.pi * rr_brpm / 60
    displacement = amplitude * np.sin(w * t + phase)
    velocity = np.diff(np.r_[0., displacement])
    return np.column_stack([.3 * velocity, velocity]) + noise * rng.standard_normal((len(t), 2))


def rate(signal, config, seconds):
    frequencies, power = window_spectrum(signal, FS, config)
    return estimate_rr(frequencies, power, seconds, config)


@pytest.mark.parametrize('rr_brpm', [25., 40., 60., 85.])
def test_flow_signal_recovers_known_respiration_rate(rr_brpm):
    config = RRConfig()
    signal = respiratory_signal(breathing_velocity(rr_brpm), FS, 'FLOW', config)
    for start, stop in rr_windows(len(signal), FS, config):
        estimate, snr = rate(signal[start:stop], config, (stop - start) / FS)
        assert abs(estimate - rr_brpm) < 1.5
        assert signal_quality(snr, 0., config) > config.sqi_threshold


def test_intensity_signal_recovers_known_respiration_rate():
    config = RRConfig()
    t = np.arange(int(15 * FS)) / FS
    grey = 120 * (1 + .01 * np.sin(2*np.pi*45/60*t) + .02 * t / t[-1]) + .3 * np.random.default_rng(2).standard_normal(len(t))
    estimate, _ = rate(respiratory_signal(grey, FS, 'INTENSITY', config), config, 15.)
    assert abs(estimate - 45.) < 1.5


def test_detrending_keeps_slow_neonatal_breathing():
    config = RRConfig()
    t = np.arange(int(20 * FS)) / FS
    breath = np.sin(2*np.pi*config.rr_min_bpm/60*t)
    kept = detrend_smoothness_priors(breath, config.detrend_lambda_30fps)
    assert np.std(kept) / np.std(breath) > .85


def test_noise_only_is_mostly_flagged_unreliable():
    config = RRConfig()
    reliable = []
    for seed in range(40):
        noise = .05 * np.random.default_rng(seed).standard_normal((int(15 * FS), 2))
        _, snr = rate(respiratory_signal(noise, FS, 'FLOW', config), config, 15.)
        reliable.append(signal_quality(snr, 0., config) >= config.sqi_threshold)
    assert np.mean(reliable) <= .1


def test_gross_motion_lowers_quality():
    config = RRConfig()
    assert signal_quality(20., 0., config) > .95
    assert signal_quality(20., .8, config) < .25
    assert signal_quality(None, 0., config) == 0.


def test_windows_grow_then_slide_and_short_clips_are_skipped():
    config = RRConfig()
    assert rr_windows(int(7 * FS), FS, config) == []
    windows = rr_windows(int(20 * FS), FS, config)
    assert windows[0] == (0, 240)                                  # first estimate after 8 s
    assert windows[7] == (0, 450)                                  # full 15 s window
    assert windows[-1] == (150, 600) and len(windows) == 13
    assert all(b - a <= 450 for a, b in windows)
    assert rr_windows(int(12.5 * FS), FS, config)[-1] == (0, 375)  # a 12.5 s clip ends with a whole-clip window


def test_spectral_fusion_survives_paradoxical_breathing():
    """Chest and abdomen in anti-phase cancel in a waveform average but not in the spectral fusion."""
    config = RRConfig()
    chest = respiratory_signal(breathing_velocity(50., seconds=15., seed=1), FS, 'FLOW', config)
    abdomen = -chest + .05 * np.random.default_rng(4).standard_normal(len(chest))
    average = (chest + abdomen) / 2
    spectra = [window_spectrum(chest, FS, config), window_spectrum(abdomen, FS, config)]
    fused_rate, _ = estimate_rr(*fuse_spectra(spectra, np.array([.5, .5])), 15., config)
    assert abs(fused_rate - 50.) < 1.5
    assert np.std(average) < .1 * np.std(chest)


def test_background_flow_removes_camera_motion():
    config = RRConfig()
    flow = np.zeros((60, 80, 2), np.float32)
    flow[...] = (.8, -.5)                         # camera shake moves everything
    flow[20:40, 30:50, 1] += .3                   # breathing inside the infant box
    box = np.zeros((60, 80), bool)
    box[15:45, 25:55] = True
    assert background_flow(flow, box, config) == pytest.approx([.8, -.5])
    assert background_flow(flow, np.ones((60, 80), bool), config) == pytest.approx([0., 0.])   # no background visible


@pytest.fixture
def vitals_run(tmp_path):
    """20 s synthetic infant: face pulse at 150 bpm, chest and abdomen breathing in anti-phase at 40 /min,
    a static neck, and camera shake at 70 /min (which the background compensation must remove)."""
    hr_bpm, rr_brpm, seconds, width, height = 150., 40., 20., 320, 240
    (tmp_path/'configs').mkdir()
    config = tmp_path/'configs/module1.yaml'
    config.write_text(yaml.safe_dump(dict(processing=dict(target_fps=20), logging=dict(log_dir='logs', level='INFO'))))
    (tmp_path/'data').mkdir()
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (height + 40, width + 40)).astype(np.float32), (0, 0), 2.)
    texture = (texture - texture.mean()) / texture.std()
    background = np.clip(np.stack([150 + 25*texture, 90 + 25*texture, 60 + 25*texture], -1), 0, 255)
    skin = np.clip(np.stack([110 + 12*texture, 140 + 12*texture, 200 + 12*texture], -1), 0, 255)

    def shifted(image, dx, dy):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    writer = cv2.VideoWriter(str(tmp_path/'data/v.avi'), cv2.VideoWriter_fourcc(*'MJPG'), FS, (width, height))
    for t in np.arange(int(seconds*FS)) / FS:
        breath = 1.5 * np.sin(2*np.pi*rr_brpm/60*t)
        frame = background.copy()
        frame[120:170, 100:220] = shifted(skin, 0, breath)[120:170, 100:220]       # chest
        frame[170:215, 100:220] = shifted(skin, 0, -breath)[170:215, 100:220]      # abdomen (paradoxical)
        frame[85:110, 140:180] = skin[85:110, 140:180]                              # neck, static
        pulse = 1 + .03*np.sin(2*np.pi*hr_bpm/60*t)*np.array([.53, .77, .33])
        frame[20:80, 120:200] = np.array([110., 140., 200.]) * pulse                # face
        shake = 1.0 * np.sin(2*np.pi*70/60*t)
        frame = shifted(frame, shake, .5*shake)[20:20+height, 20:20+width]
        writer.write(np.clip(np.rint(frame), 0, 255).astype(np.uint8))
    writer.release()
    pd.DataFrame([dict(video_id='v', filename='v.avi', split='test', fps=FS)]).to_csv(tmp_path/'manifest.csv', index=False)

    def box(x1, y1, x2, y2):
        return [[x1-20, y1-20], [x2-20, y1-20], [x2-20, y2-20], [x1-20, y2-20]]   # frame coordinates after the crop
    regions = dict(FOREHEAD=(box(130, 22, 190, 40), .7), LEFT_CHEEK=(box(125, 45, 155, 75), .65),
                   RIGHT_CHEEK=(box(165, 45, 195, 75), .6), UPPER_FACE=(box(122, 22, 198, 55), .55),
                   FACE=(box(120, 20, 200, 80), .5), NECK=(box(142, 87, 178, 108), .75),
                   UPPER_CHEST=(box(105, 125, 215, 165), .6), ABDOMEN=(box(105, 175, 215, 210), .55))
    processed = sorted({int(round(k/20*FS)) for k in range(int(seconds*20))})
    rows, recovery = [], []
    for frame in processed:
        for name, (polygon, q) in regions.items():
            xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
            rows.append(dict(video_id='v', split='test', frame_index=frame, timestamp_sec=frame/FS, roi_name=name,
                             roi_status='AVAILABLE', polygon=json.dumps(polygon),
                             bbox=json.dumps([min(xs), min(ys), max(xs), max(ys)]), q=q, alpha=.1,
                             visible_skin_proportion=.8))
        recovery.append(dict(video_id='v', frame_index=frame, final_infant_bbox=json.dumps([80, 0, 200, 200])))
    (tmp_path/'outputs/confidence_sam2').mkdir(parents=True)
    pd.DataFrame(rows).to_csv(tmp_path/'outputs/confidence_sam2/roi_confidence.csv', index=False)
    (tmp_path/'outputs/recovery').mkdir(parents=True)
    pd.DataFrame(recovery).to_csv(tmp_path/'outputs/recovery/recovery_results.csv', index=False)
    return tmp_path, ['--config', str(config), '--manifest', 'manifest.csv'], hr_bpm, rr_brpm


def test_end_to_end_rr_joint_monitor_and_evaluation(vitals_run):
    root, common, hr_bpm, rr_brpm = vitals_run
    run = load_script('run_rr')
    assert run.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/rr_sam2']) == 0
    output = root/'outputs/rr_sam2'
    windows = pd.read_csv(output/'rr_windows.csv')
    assert set(windows.strategy) == {'dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box'}
    primary = windows[(windows.method == 'FLOW') & (windows.strategy == 'dynamic_fusion')]
    assert len(primary) == 13 and primary.reliable.mean() > .8
    assert (primary.rr_brpm - rr_brpm).abs().max() < 2. and (primary.rr_kalman_brpm - rr_brpm).abs().max() < 2.
    # The static neck has the highest Module 1 confidence but does not move (only residual camera shake),
    # so the amplitude gate excludes it and every strategy follows the breathing chest / abdomen.
    assert set(primary.top_roi) <= {'UPPER_CHEST', 'ABDOMEN'}
    assert not any('NECK' in json.loads(w) for w in primary.roi_weights)
    top1 = windows[(windows.method == 'FLOW') & (windows.strategy == 'top1')]
    assert set(top1.top_roi) == {'UPPER_CHEST'}
    assert (top1.rr_brpm - rr_brpm).abs().max() < 2.
    signals = pd.read_csv(output/'resp_signals.csv')
    assert len(signals) == 600 and np.allclose(signals.filter(like='weight_').sum(axis=1), 1.)
    assert (output/'plots/v_rr.png').is_file()
    with pytest.raises(FileExistsError):
        run.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/rr_sam2'])

    hr = load_script('run_hr_rppg')
    assert hr.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr_rppg_sam2',
                             '--roi-pool', 'face']) == 0
    render = load_script('render_vitals_monitor')
    assert render.main(common + ['--hr-dir', 'outputs/hr_rppg_sam2', '--rr-dir', 'outputs/rr_sam2', '--scale', '1']) == 0
    movie = output/'vitals_monitor_videos/v_vitals_monitor.mp4'
    capture = cv2.VideoCapture(str(movie))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 600
    capture.release()

    reference = pd.DataFrame(dict(video_id='v', timestamp_sec=np.arange(21.), rr_brpm=rr_brpm))
    reference.to_csv(root/'reference_rr.csv', index=False)
    evaluate = load_script('evaluate_rr')
    assert evaluate.main(['--config', common[1], '--rr-dir', 'outputs/rr_sam2', '--reference', 'reference_rr.csv']) == 0
    metrics = pd.read_csv(output/'evaluation/rr_metrics.csv')
    best = metrics[(metrics.method == 'FLOW') & (metrics.strategy == 'dynamic_fusion') & (metrics.estimate == 'rr_kalman_brpm') &
                   (metrics.level == 'window') & (metrics.subset == 'all_windows')].iloc[0]
    assert best.mae_brpm < 2.


def test_end_to_end_thoraco_abdominal_phase_and_cardiorespiratory_hr(vitals_run):
    """The synthetic chest and abdomen breathe in anti-phase (paradoxical): the phase must be near 180 deg.
    HR with --rr-windows adds the multi-site consensus and the respiratory-harmonic-aware strategies."""
    root, common, hr_bpm, rr_brpm = vitals_run
    assert load_script('run_rr').main(common + ['--confidence-dir', 'outputs/confidence_sam2',
                                                '--output-dir', 'outputs/rr', '--no-plots']) == 0
    windows = pd.read_csv(root/'outputs/rr/rr_windows.csv')
    primary = windows[(windows.method == 'FLOW') & (windows.strategy == 'dynamic_fusion')]
    assert set(primary.taa_chest_roi) == {'UPPER_CHEST'} and set(primary.taa_abdomen_roi) == {'ABDOMEN'}
    assert (primary.taa_phase_deg > 150).all() and primary.taa_coherence.median() > .8
    summary = pd.read_csv(root/'outputs/rr/rr_summary.csv')
    assert (summary.taa_phase_median_deg > 150).all()

    hr = load_script('run_hr_rppg')
    assert hr.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr',
                             '--rr-windows', 'outputs/rr/rr_windows.csv', '--no-plots']) == 0
    hr_windows = pd.read_csv(root/'outputs/hr/hr_windows.csv')
    assert {'consensus_fusion', 'cardiorespiratory_fusion'} <= set(hr_windows.strategy)
    proposed = hr_windows[(hr_windows.method == 'POS') & (hr_windows.strategy == 'cardiorespiratory_fusion')]
    assert len(proposed) == 13 and (proposed.rr_reference_brpm - rr_brpm).abs().max() < 2.
    assert (proposed.hr_bpm - hr_bpm).abs().max() < 3. and not proposed.harmonic_ambiguous.any()
    metadata = json.loads((root/'outputs/hr/experiment_metadata.json').read_text())
    assert metadata['rr_reference']['source'] == 'FLOW / dynamic_fusion windows'
    # Without --rr-windows only the consensus strategy is added.
    assert hr.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr_plain',
                             '--no-plots']) == 0
    plain = pd.read_csv(root/'outputs/hr_plain/hr_windows.csv')
    assert 'consensus_fusion' in set(plain.strategy) and 'cardiorespiratory_fusion' not in set(plain.strategy)
