import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from module2.neural import (NEURAL_METHODS, NeuralConfig, Sampler, VideoClips, adapted_copy, build_model, finalize,
                            infer, normalize_clips, resample, sinc_losses, train)
from module2.rppg import HRConfig, estimate_hr
from test_rr import load_script, vitals_run  # noqa: F401  (fixture)


FS = 30.
NAMES = ['FOREHEAD', 'LEFT_CHEEK', 'UPPER_CHEST', 'LEFT_ARM', 'INFANT_BOX']
CFG = NeuralConfig(batch_size=16, site_pairs=4)


def clips(hr_bpm, seconds=20., seed=0, q=(.8, .7, .5, .4)):
    """Colour grids with a pulse at hr_bpm (weaker on the trunk and arm), illumination drift and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    pulse = np.sin(2*np.pi*hr_bpm/60*t)[:, None] * np.array([.33, .77, .53]) * .004
    drift = (.03*np.sin(2*np.pi*.15*t))[:, None]
    grids = np.empty((len(t), len(NAMES), 4, 4, 3))
    for r in range(len(NAMES)):
        rgb = np.array([200., 140., 110.]) * (1 + (1. if r < 2 else .5)*pulse + drift + .002*rng.standard_normal((len(t), 3)))
        grids[:, r] = rgb[:, None, None, :] * (1 + .01*rng.standard_normal((1, 4, 4, 3)))
    quality = np.c_[np.tile(q, (len(t), 1)), np.full(len(t), np.nan)]
    return VideoClips('v', grids, quality, NAMES, FS, CFG)


def test_resample_keeps_a_sine_and_its_length():
    t = np.arange(300) / 25.
    x = np.sin(2*np.pi*2.*t)
    y = resample(x, 25., 30.)
    assert len(y) == int((300 - 1) / 25. * 30.) + 1
    assert np.allclose(y, np.sin(2*np.pi*2.*np.arange(len(y)) / 30.), atol=.03)
    assert len(resample(x, 25., 30., count=300)) == 300


def test_sinc_losses_prefer_a_clean_in_band_pulse():
    t = torch.arange(180) / FS
    band = HRConfig().band_hz
    clean = torch.stack([torch.sin(2*np.pi*f*t) for f in (2., 2.5, 3., 3.5)])
    noisy = torch.randn(4, 180, generator=torch.Generator().manual_seed(0))
    slow = torch.stack([torch.sin(2*np.pi*.5*t)] * 4)                  # breathing-like, outside the HR band
    main_clean, variance_clean, _ = sinc_losses(clean, FS, band, CFG)
    main_noisy, _, _ = sinc_losses(noisy, FS, band, CFG)
    main_slow, _, _ = sinc_losses(slow, FS, band, CFG)
    assert main_clean < .25 and main_noisy > .5 and main_slow > .9      # rectangular window: some leakage
    _, variance_same, _ = sinc_losses(torch.stack([torch.sin(2*np.pi*2.*t)] * 4), FS, band, CFG)
    assert variance_clean < variance_same                               # spread of frequencies across the batch


def test_guided_sampling_prefers_confident_rois_and_pairs_different_sites():
    video = clips(150.)
    guided = Sampler([video], CFG, True, np.random.default_rng(0))
    uniform = Sampler([video], CFG, False, np.random.default_rng(0))
    rois = np.array([c[1] for c in guided.candidates])
    assert np.isclose(uniform.probabilities[rois == 0].sum(), .25)
    assert guided.probabilities[rois == 0].sum() > .6                   # FOREHEAD has the highest Module 1 Q
    assert 4 not in set(rois)                                            # INFANT_BOX has no Q: never a training clip
    main, q, pairs_a, pairs_b = guided.batch()
    assert main.shape == (16, 3, int(CFG.clip_sec * FS), 4, 4) and q.shape == (16,)
    assert pairs_a is not None and pairs_a.shape == pairs_b.shape
    assert uniform.batch()[2] is None                                    # no cross-site term without guidance


def test_normalize_clips_removes_the_mean_colour():
    x = np.random.default_rng(0).uniform(100, 200, (2, 10, 4, 4, 3))
    y = normalize_clips(x)
    assert y.shape == (2, 3, 10, 4, 4) and np.allclose(y.mean(axis=2), 0, atol=1e-6)


def test_short_video_cannot_be_adapted():
    with pytest.raises(ValueError, match='too short'):
        Sampler([clips(150., seconds=5.)], CFG, True, np.random.default_rng(0))


def test_adaptation_recovers_the_heart_rate_of_an_unseen_infant():
    hr = HRConfig()
    cfg = NeuralConfig(batch_size=16, site_pairs=4, tta_steps=60)
    model = build_model(cfg)
    history = train(model, [clips(b, seed=i) for i, b in enumerate((120., 170., 200.))], cfg, hr.band_hz, 150,
                    cfg.pretrain_lr, True, 'cpu')
    assert history[-1]['loss'] < history[0]['loss']
    target = clips(140., seed=9)
    adapted, adaptation = adapted_copy(model, target, cfg, hr.band_hz, True, 'cpu')
    assert len(adaptation) == 60 and adaptation[-1]['site'] >= 0
    before = [p.detach().clone() for p in model.parameters()]
    pulse = finalize(infer(adapted, target, cfg, 'cpu'), FS, hr)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))   # the source model is untouched
    assert pulse.shape == (len(target.grids), len(NAMES)) and np.isfinite(pulse).all()
    estimates = [estimate_hr(pulse[s:s+240, 0], FS, hr)[0] for s in range(0, 360, 60)]
    assert max(abs(e - 140.) for e in estimates) < 3.


def test_end_to_end_pretrain_adapt_and_hr_with_neural_methods(vitals_run):
    root, common, hr_bpm, rr_brpm = vitals_run
    config = common[1]
    sources = pd.read_csv(root/'manifest.csv').assign(sha256='aaa')
    sources.to_csv(root/'manifest.csv', index=False)
    pd.DataFrame([dict(video_id='target', sha256='aaa')]).to_csv(root/'target_manifest.csv', index=False)
    run = load_script('run_neural_rppg')
    base = common + ['--confidence-dir', 'outputs/confidence_sam2', '--device', 'cpu']
    with pytest.raises(ValueError, match='No videos'):          # the only source video is the target: left out
        run.main(base + ['--mode', 'pretrain', '--steps', '5', '--output-dir', 'outputs/neural_pretrain',
                         '--exclude-manifest', 'target_manifest.csv'])
    assert run.main(base + ['--mode', 'pretrain', '--steps', '20', '--output-dir', 'outputs/neural_pretrain',
                            '--save-model', 'models/source.pt']) == 0
    assert (root/'models/source.pt').is_file()
    assert json.loads((root/'outputs/neural_pretrain/pretrain_metadata.json').read_text())['excluded_target_videos'] == []
    assert run.main(base + ['--mode', 'adapt', '--steps', '10', '--pretrained', 'models/source.pt',
                            '--output-dir', 'outputs/neural']) == 0
    data = np.load(root/'outputs/neural/v_neural_pulses.npz')
    assert list(data['methods']) == list(NEURAL_METHODS) and data['NEURAL_TTA'].shape[0] == 600
    history = pd.read_csv(root/'outputs/neural/adaptation_history.csv')
    assert set(history.variant) == {'NEURAL_TTA', 'NEURAL_TTA_UNGUIDED'} and len(history) == 20
    with pytest.raises(FileExistsError):
        run.main(base + ['--mode', 'adapt', '--steps', '10', '--output-dir', 'outputs/neural'])

    hr = load_script('run_hr_rppg')
    assert hr.main(common + ['--confidence-dir', 'outputs/confidence_sam2', '--output-dir', 'outputs/hr',
                             '--neural-dir', 'outputs/neural', '--no-plots']) == 0
    windows = pd.read_csv(root/'outputs/hr/hr_windows.csv')
    assert set(NEURAL_METHODS) <= set(windows.method)
    neural = windows[(windows.method == 'NEURAL_TTA') & (windows.strategy == 'dynamic_fusion')]
    assert len(neural) == 13 and neural.hr_bpm.notna().all()
    metadata = json.loads((root/'outputs/hr/experiment_metadata.json').read_text())
    assert metadata['neural_dir'].endswith('outputs/neural')
