"""Module 1-guided unsupervised neural rPPG with per-neonate test-time adaptation (Module 2, HR branch).

Input per ROI and frame: a grid x grid map of mean skin RGB inside the Module 1 ROI (cells without skin take
the ROI mean), resampled to model_fs. A small fully convolutional network (colour/temporal 3-D convolutions,
spatial attention over the grid, dilated temporal convolutions) maps a clip to a pulse waveform.

Training uses no reference HR. The losses are those of SiNC (Speth et al., CVPR 2023), restricted to the
neonatal HR band: bandwidth (power outside the band), sparsity (in-band power away from the spectral peak)
and variance (batch-mean normalized PSD vs a uniform distribution, squared EMD), with speed augmentation
(temporal resampling by a random factor) so that a batch spans many frequencies even for one infant.

Module 1 guidance (switched off for the unguided ablation):
  1. sampling: clips are drawn with probability softmax(mean Q / q_temperature) over (ROI, time) candidates;
  2. weighting: each clip's bandwidth + sparsity loss is weighted by its mean Q (normalized in the batch);
  3. anatomy: pairs of simultaneous clips from different anatomical sites (head, trunk, arms, legs) must have
     the same in-band PSD (1 - cosine similarity), because the cardiac pulse is systemic.
Each guidance component can be switched off on its own (guide_sampling, guide_weighting, guide_sites) for the
component ablation. Domain adaptation: a source model is pre-trained on unlabeled videos of other infants (pretrain), then adapted
to every target video on its own frames before inference (test-time adaptation). Per video three pulse sets
are produced for run_hr_rppg.py --neural-dir: NEURAL_SRC (source model, no adaptation), NEURAL_TTA (guided
adaptation) and NEURAL_TTA_UNGUIDED (adaptation without Module 1 guidance).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import cv2
import numpy as np

from module2.cardiorespiratory import site_of
from module2.extraction import shrink, skin_pixels
from module2.rppg import bandpass, detrend_smoothness_priors, interpolate_gaps, segments


NEURAL_METHODS = ('NEURAL_SRC', 'NEURAL_TTA', 'NEURAL_TTA_UNGUIDED')
# Adaptation variants: name -> (guided, NeuralConfig overrides). The last three remove one guidance component.
TTA_VARIANTS = dict(NEURAL_TTA=(True, {}), NEURAL_TTA_UNGUIDED=(False, {}),
                    NEURAL_TTA_NO_QSAMPLE=(True, dict(guide_sampling=False)),
                    NEURAL_TTA_NO_QWEIGHT=(True, dict(guide_weighting=False)),
                    NEURAL_TTA_NO_SITES=(True, dict(guide_sites=False)))


@dataclass(frozen=True)
class NeuralConfig:
    grid: int = 4
    model_fs: float = 30.
    clip_sec: float = 6.
    # Speed augmentation: a clip of clip_sec x factor seconds is resampled to clip_sec, multiplying every
    # frequency by the factor (150 bpm -> 90-240 bpm).
    speed_range: tuple = (.6, 1.6)
    width: int = 16
    batch_size: int = 32
    site_pairs: int = 8
    pretrain_steps: int = 3000
    pretrain_lr: float = 1e-3
    tta_steps: int = 300
    tta_lr: float = 5e-4
    sparsity_halfwidth_hz: float = .1
    variance_weight: float = 1.
    site_weight: float = .5
    q_temperature: float = .1
    clip_stride_sec: float = .5
    noise_std: float = .001
    seed: int = 42
    # Module 1 guidance components (used only when guided).
    guide_sampling: bool = True
    guide_weighting: bool = True
    guide_sites: bool = True


def mean_rgb_grid(image, polygon, ex, grid):
    """(grid, grid, 3) mean RGB of skin pixels per cell of the shrunken ROI's bounding box, or None.

    Pixel selection is that of extraction.mean_rgb (shrink, saturation/dark limits, skin mask); cells
    without skin take the ROI's mean colour, so they carry no pulse but no artificial edge either.
    """
    height, width = image.shape[:2]
    polygon = shrink(polygon, ex.roi_shrink)
    x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
    x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
    if x2 - x1 < grid or y2 - y1 < grid:
        return None
    crop = image[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon - [x1, y1]).astype(np.int32)], 1)
    inside = mask.astype(bool)
    brightest = crop.max(axis=2)
    selected = skin_pixels(crop, inside & (brightest < ex.saturation_level) & (brightest > ex.dark_level), ex)
    if selected.sum() < max(ex.min_pixels, ex.min_skin_fraction * inside.sum()):
        return None
    mean = crop[selected].astype(float).mean(axis=0)
    ys = np.linspace(0, crop.shape[0], grid + 1).astype(int)
    xs = np.linspace(0, crop.shape[1], grid + 1).astype(int)
    out = np.empty((grid, grid, 3))
    for i in range(grid):
        for j in range(grid):
            cell = selected[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            out[i, j] = crop[ys[i]:ys[i+1], xs[j]:xs[j+1]][cell].astype(float).mean(axis=0) if cell.any() else mean
    return out[..., ::-1]  # BGR -> RGB


def resample(values, fps, target_fs, count=None):
    """Linear resampling along axis 0 from fps to target_fs (NaN propagates); count = output length."""
    n = len(values)
    count = count if count is not None else int(math.floor((n - 1) / fps * target_fs)) + 1
    position = np.clip(np.arange(count) * fps / target_fs, 0, n - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, n - 1)
    fraction = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[low] * (1 - fraction) + values[high] * fraction


class VideoClips:
    """One video at model_fs: grids (n, m, G, G, 3) with NaN gaps, Q (n, m) and the site of every column."""

    def __init__(self, name, grids, q, names, fps, cfg):
        self.name, self.names, self.native_fps, self.native_frames = name, list(names), fps, len(grids)
        self.grids = resample(grids.astype(np.float32), fps, cfg.model_fs)
        self.q = resample(q.astype(np.float32), fps, cfg.model_fs)
        self.sites = [site_of(name) for name in self.names]
        # observed: usable for inference (any column, incl. INFANT_BOX); valid: also has a Module 1 Q (training).
        self.observed = np.isfinite(self.grids).all(axis=(2, 3, 4))
        self.valid = self.observed & np.isfinite(self.q)


def _torch():
    import torch
    return torch


def build_model(cfg):
    torch = _torch()
    nn = torch.nn

    class PulseNet(nn.Module):
        def __init__(self):
            super().__init__()
            w = cfg.width
            self.colour = nn.Sequential(nn.Conv3d(3, w, (5, 1, 1), padding=(2, 0, 0)), nn.ELU(),
                                        nn.Conv3d(w, w, (5, 1, 1), padding=(2, 0, 0)), nn.ELU())
            self.attention = nn.Conv3d(w, 1, 1)
            self.temporal = nn.Sequential(nn.Conv1d(w, 32, 5, padding=2), nn.ELU(),
                                          nn.Conv1d(32, 32, 5, padding=4, dilation=2), nn.ELU(),
                                          nn.Conv1d(32, 32, 5, padding=8, dilation=4), nn.ELU(),
                                          nn.Conv1d(32, 1, 1))

        def forward(self, x):                       # x: (B, 3, T, G, G), normalized colour
            h = self.colour(x)
            a = torch.softmax(self.attention(h).flatten(3), dim=-1).view_as(h[:, :1])
            y = self.temporal((h * a).sum(dim=(3, 4))).squeeze(1)
            return y - y.mean(dim=-1, keepdim=True)

    return PulseNet()


def normalize_clips(x):
    """x: (B, T, G, G, 3) raw colour -> (B, 3, T, G, G) relative colour change x / mean_t - 1."""
    mean = x.mean(axis=1, keepdims=True)
    x = x / np.where(mean > 0, mean, 1.) - 1.
    return np.ascontiguousarray(np.transpose(x, (0, 4, 1, 2, 3)), dtype=np.float32)


def spectra(y, fs, band_hz):
    """Normalized in-band PSD (B, F), full PSD terms and frequencies for the SiNC losses."""
    torch = _torch()
    nfft = 1 << int(math.ceil(math.log2(4 * y.shape[-1])))
    power = torch.fft.rfft(y - y.mean(dim=-1, keepdim=True), n=nfft).abs() ** 2
    frequencies = torch.fft.rfftfreq(nfft, 1. / fs).to(y.device)
    inside = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    return power, frequencies, inside


def sinc_losses(y, fs, band_hz, cfg, weights=None):
    """(weighted bandwidth + sparsity, variance, normalized in-band PSD) of a batch of pulse clips."""
    torch = _torch()
    power, frequencies, inside = spectra(y, fs, band_hz)
    eps = 1e-8
    bandwidth = power[:, ~inside].sum(-1) / (power.sum(-1) + eps)
    p_in, f_in = power[:, inside], frequencies[inside]
    peak = f_in[p_in.argmax(-1)]
    near = (f_in[None, :] - peak[:, None]).abs() <= cfg.sparsity_halfwidth_hz
    sparsity = (p_in * ~near).sum(-1) / (p_in.sum(-1) + eps)
    normalized = p_in / (p_in.sum(-1, keepdim=True) + eps)
    uniform = torch.full_like(normalized[0], 1. / normalized.shape[-1])
    variance = ((torch.cumsum(normalized.mean(0), 0) - torch.cumsum(uniform, 0)) ** 2).mean() * normalized.shape[-1]
    w = torch.ones_like(bandwidth) if weights is None else weights / weights.mean()
    return (w * (bandwidth + sparsity)).mean(), variance, normalized


class Sampler:
    """Training clips of one or more videos, with Module 1-guided or uniform sampling."""

    def __init__(self, videos, cfg, guided, rng):
        self.videos, self.cfg, self.guided, self.rng = videos, cfg, guided, rng
        self.length = int(round(cfg.clip_sec * cfg.model_fs))
        self.span = int(math.ceil(self.length * cfg.speed_range[1]))
        stride = max(1, int(round(cfg.clip_stride_sec * cfg.model_fs)))
        candidates, by_time = [], {}
        for v, video in enumerate(videos):
            for r, site in enumerate(video.sites):
                if site is None:
                    continue
                valid = video.valid[:, r]
                cumulative = np.r_[0, np.cumsum(~valid)]
                for start in range(0, len(valid) - self.span + 1, stride):
                    if cumulative[start + self.span] - cumulative[start] == 0:
                        candidates.append((v, r, start, float(np.mean(video.q[start:start+self.span, r]))))
                        by_time.setdefault((v, start), []).append(r)
        if not candidates:
            raise ValueError(f'No ROI is observed for {self.span / cfg.model_fs:.1f} s without gaps; video too short '
                             'or no usable skin ROI')
        self.candidates, self.by_time = candidates, by_time
        q = np.array([c[3] for c in candidates])
        if guided and cfg.guide_sampling:
            logits = (q - q.max()) / cfg.q_temperature
            self.probabilities = np.exp(logits) / np.exp(logits).sum()
        else:
            self.probabilities = np.full(len(candidates), 1. / len(candidates))

    def _clip(self, v, r, start, factor, flip):
        source = max(2, int(round(self.length * factor)))
        raw = self.videos[v].grids[start:start + source, r]
        position = np.linspace(0, source - 1, self.length)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, source - 1)
        fraction = (position - low)[:, None, None, None]
        clip = raw[low] * (1 - fraction) + raw[high] * fraction
        return clip[:, :, ::-1] if flip else clip

    def batch(self):
        """(main clips, main Q, site-pair clips A, site-pair clips B) as normalized float32 arrays."""
        cfg, rng = self.cfg, self.rng
        chosen = rng.choice(len(self.candidates), cfg.batch_size, p=self.probabilities)
        clips, q = [], []
        for index in chosen:
            v, r, start, mean_q = self.candidates[index]
            clips.append(self._clip(v, r, start, rng.uniform(*cfg.speed_range), rng.random() < .5))
            q.append(mean_q)
        pairs_a, pairs_b = [], []
        if self.guided and cfg.guide_sites and cfg.site_pairs:
            for index in rng.choice(len(self.candidates), cfg.site_pairs * 4, p=self.probabilities):
                v, r, start, _ = self.candidates[index]
                others = [o for o in self.by_time[(v, start)] if self.videos[v].sites[o] != self.videos[v].sites[r]]
                if not others:
                    continue
                factor = rng.uniform(*cfg.speed_range)
                pairs_a.append(self._clip(v, r, start, factor, False))
                pairs_b.append(self._clip(v, others[rng.integers(len(others))], start, factor, False))
                if len(pairs_a) == cfg.site_pairs:
                    break
        noise = lambda x: normalize_clips(np.stack(x)) + cfg.noise_std * rng.standard_normal((len(x), 3, self.length,
                                                                                               cfg.grid, cfg.grid))
        return (noise(clips).astype(np.float32), np.asarray(q, np.float32),
                noise(pairs_a).astype(np.float32) if pairs_a else None,
                noise(pairs_b).astype(np.float32) if pairs_b else None)


def train(model, videos, cfg, band_hz, steps, lr, guided, device):
    """Unsupervised training / adaptation in place; returns the per-step loss history."""
    torch = _torch()
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    sampler = Sampler(videos, cfg, guided, rng)
    q_range = np.ptp([c[3] for c in sampler.candidates]) or 1.
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for step in range(steps):
        clips, q, pairs_a, pairs_b = sampler.batch()
        y = model(torch.from_numpy(clips).to(device))
        # Q weights shifted to stay positive: the best clip counts about twice the worst one.
        weights = torch.from_numpy(1. + (q - q.min()) / q_range).to(device) if guided and cfg.guide_weighting else None
        main, variance, _ = sinc_losses(y, cfg.model_fs, band_hz, cfg, weights)
        loss = main + cfg.variance_weight * variance
        site = torch.zeros((), device=device)
        if pairs_a is not None:
            _, _, pa = sinc_losses(model(torch.from_numpy(pairs_a).to(device)), cfg.model_fs, band_hz, cfg)
            _, _, pb = sinc_losses(model(torch.from_numpy(pairs_b).to(device)), cfg.model_fs, band_hz, cfg)
            site = 1. - torch.nn.functional.cosine_similarity(pa, pb, dim=-1).mean()
            loss = loss + cfg.site_weight * site
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(dict(step=step, loss=loss.item(), bandwidth_sparsity=main.item(), variance=variance.item(),
                            site=site.item()))
    model.eval()
    return history


def adapted_copy(model, video, cfg, band_hz, guided, device):
    """Test-time adaptation of a copy of the source model on one video's own frames."""
    adapted = copy.deepcopy(model)
    return adapted, train(adapted, [video], cfg, band_hz, cfg.tta_steps, cfg.tta_lr, guided, device)


def infer(model, video, cfg, device):
    """Raw pulse (native frames, m) of every column: overlap-added Hann windows on gap-free segments."""
    torch = _torch()
    length = int(round(cfg.clip_sec * cfg.model_fs))
    step = length // 2
    n, m = video.observed.shape
    out = np.full((n, m), np.nan)
    model.to(device).eval()
    with torch.no_grad():
        for r in range(m):
            for start, stop in segments(video.observed[:, r], int(2 * cfg.model_fs)):
                size = min(length, stop - start)
                starts = list(range(start, stop - size + 1, max(1, step if size == length else size)))
                if starts[-1] != stop - size:
                    starts.append(stop - size)
                clips = normalize_clips(np.stack([video.grids[s:s+size, r] for s in starts]))
                y = model(torch.from_numpy(clips).to(device)).cpu().numpy()
                total, weight = np.zeros(stop - start), np.zeros(stop - start)
                window = np.hanning(size) + 1e-3
                for s, values in zip(starts, y):
                    total[s-start:s-start+size] += window * (values / (np.std(values) + 1e-8))
                    weight[s-start:s-start+size] += window
                out[start:stop, r] = total / weight
    return resample(out, cfg.model_fs, video.native_fps, video.native_frames)


def finalize(raw, fps, hr_config):
    """Detrended, band-passed, z-scored pulse per column; NaN gaps kept (as rppg.extract_pulse)."""
    raw = interpolate_gaps(raw, fps, hr_config.max_gap_sec)
    out = np.full(raw.shape, np.nan)
    minimum = max(int(math.ceil(2 * hr_config.pos_window_sec * fps)), 3 * (2*hr_config.filter_order + 1) + 1)
    for r in range(raw.shape[1]):
        for start, stop in segments(np.isfinite(raw[:, r]), minimum):
            filtered = bandpass(detrend_smoothness_priors(raw[start:stop, r], hr_config.detrend_lambda), fps, hr_config)
            std = np.std(filtered)
            out[start:stop, r] = (filtered - filtered.mean()) / std if std > 0 else 0.
    return out
