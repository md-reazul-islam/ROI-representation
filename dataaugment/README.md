# DataAugment: controlled robustness augmentation of UBFC-rPPG videos

DataAugment generates **motion**, **occlusion** and **lighting** variants of UBFC-rPPG videos at three severity
levels. The variants are used to test how contact-free HR/RR methods (e.g. fixed ROI vs. dynamic ROI) degrade under
controlled stress. The UBFC reference PPG stays valid because the pulse remains in the pixels and the timeline is
unchanged frame-for-frame.

The module is independent: it imports nothing from `src/module1`, `module1_v2`, `module2*` or the HR/RR notebooks,
and it writes only inside `dataaugment/`.

> **Important.** UBFC-rPPG shows **adult** volunteers sitting still. The occlusions are **simulated**: a procedural
> hand-shaped mask and a cloth texture composited over the video. They are **not** real hands, real blankets or
> **real swaddling**. Synthetic occluders carry no pulse and do not deform with the body. Report results on these
> videos as controlled stress tests, not as neonatal or clinical conditions.

## Installation

Nothing needs to be installed in the project environment. `numpy`, `opencv-python` (with its bundled FFmpeg and
FFV1 encoder), `scipy` and `PyYAML` are already present (see `requirements.txt`). For a new environment:

```bash
pip install -r dataaugment/requirements.txt
```

## Execution

Run from the project root (`Module1/`). In this project, `python` is not on PATH; use the conda interpreter:

```bash
python dataaugment/main.py
# or, with the env activated:
python dataaugment/main.py

python dataaugment/main.py --max-frames 120   # quick test (spectral checks need >= 30 s, so they are skipped)
python dataaugment/main.py --skip-hash        # skip the SHA-256 integrity check of the source videos
```

The program does the following:

1. It finds the first three videos under `data/ubfc` in natural order (subject1, subject3, subject4).
2. It assigns motion, occlusion and lighting respectively.
3. It reads each source **once** and generates small, medium and large in the same pass.
4. It decodes every output again to verify it.
5. It writes per-video JSON reports, per-frame CSVs, preview sheets and `data/validation_report.json`.

A full run takes about 13 minutes (2 CPU cores) and writes 3.5 GB.

## Structure

```text
dataaugment/
├── main.py                         entry point
├── config/augmentation_config.yaml all parameters, seeds, severities, physiological bands
├── src/video_utils.py              raw-AVI reader, FFV1 writer + verification, smooth/aperiodic signals, checks
├── src/motion.py                   hand-held camera motion
├── src/occlusion.py                simulated hand + blanket occlusion
├── src/lighting.py                 exposure, gradual fluctuation, flicker, moving shadow
├── data/
│   ├── validation_report.json      run-level report (config, environment, source SHA-256, all outputs)
│   ├── manifest.csv                one row per output
│   └── <category>/
│       ├── preview_<subject>.png   original + 3 severities at 4 time points (PNG, preview only)
│       └── <severity>/
│           ├── <subject>_<category>_<severity>.mkv         FFV1 lossless video
│           ├── <subject>_<category>_<severity>.json        parameters, seeds, quality + temporal checks
│           ├── <subject>_<category>_<severity>_frames.csv  per-frame transformation parameters
│           └── <subject>_ground_truth.txt                  unchanged copy of the UBFC reference PPG/HR
└── run_log.txt                     console log of the last full run
```

## Video quality preservation

| Requirement | How it is met |
|---|---|
| Original never modified | Sources are opened read-only. SHA-256 is computed before and after the run and compared (`source_integrity`). The output folder may not overlap the input folder. |
| No decode loss on input | UBFC files are uncompressed BGR24 AVI. `RawAviReader` copies the stored bytes directly; no codec is involved. OpenCV's decoder crashes on these files, which is why the custom reader exists. |
| Resolution, FPS, frame count, duration | Frames are never resized. The writer gets the source FPS; frame count is checked by decoding the output. |
| Lossless output | FFV1 in MKV. Every written frame's MD5 is compared with the decoded output frame (`lossless_bit_exact`). |
| No unnecessary re-encoding | One read per source, one encode per output, no intermediate files. |
| Audio | The UBFC AVIs have a single video stream and no audio, so audio checks are "not applicable". If a source had audio, `mux_audio` would stream-copy it, but only if an `ffmpeg` binary is available, and that path is untested. |
| Interpolation | See the next section. |

**FPS note.** OpenCV gives FFmpeg the frame rate rounded to 0.001 fps; for example, 29.264106 is stored as
29.264. Over a 52-s video this shifts the nominal duration by less than 1 ms (`duration_diff_ms` ≤ 0.995). Frames
are never dropped or duplicated, so the copied `ground_truth.txt` (one PPG sample and timestamp per frame) stays
aligned frame-for-frame.

### Unavoidable pixel changes

Lossless encoding stores the augmented frames exactly; it cannot undo the changes the augmentation makes.

- **Motion:** each output frame is resampled once from the original frame (no accumulated resampling). The warp is
  a sub-pixel affine transform with Lanczos-4 interpolation.
  - Resampling is a local weighted average of neighbouring pixels in the **same** frame. It slightly smooths fine
    spatial detail and can overshoot by a few grey levels at sharp edges (clipped to 0–255).
  - It adds no temporal component.
  - Uncovered borders are filled with black (`border_mode: constant`; `reflect` and `replicate` are also
    available), and the per-frame share is logged as `border_fraction`.
  - A zero-amplitude warp is verified to be bit-exact (`identity_transform_bit_exact`).
- **Occlusion:** visible pixels are unchanged except inside the soft hand shadow. Only the synthetic layers are
  resampled (bilinear).
- **Lighting:** there is no geometric resampling.
  - Gains are applied in approximately linear light (gamma 2.2) and re-quantised to 8 bit.
  - Pixels pushed past 0 or 255 clip; `saturated_frac_out` is logged per frame, with a maximum of 1.7% at large.
  - Unit gain is verified to be bit-exact.

## Augmentations

All severities of one category use the **same random trajectories**, scaled in amplitude. Small, medium and large
therefore differ only in strength, which gives a clean dose-response comparison. Seeds come from
`SeedSequence([seed, category_index])`, with `seed: 20261114`.

### A. Motion: subject1, simulated hand-held camera

- **Drift:** translation, rotation and zoom follow aperiodic, zero-phase low-pass filtered Gaussian noise
  (cut-off 0.08 Hz). The motion is smooth and continuous, never random frame-to-frame jumps.
- **Tremor:** a small jitter band-limited to 6–10 Hz, above the heart-rate band.

| Severity | Peak translation | Peak rotation | Zoom | Tremor RMS | Measured peak shift / border share (mean, max) |
|---|---|---|---|---|---|
| small | ±8 px | ±1° | ±2% | 0.25 px | 8.6 px / 1.4%, 3.5% |
| medium | ±20 px | ±3° | ±5% | 0.5 px | 21.1 px / 3.5%, 8.7% |
| large | ±40 px | ±6° | ±10% | 1.0 px | 42.2 px / 7.0%, 16.9% |

### B. Occlusion: subject3, simulated hand + blanket

- **Hand:** a procedural hand-shaped mask (palm, fingers, thumb, fading wrist) with fixed shading and a soft
  shadow. It wanders around the detected face (Haar cascade) along slow, aperiodic paths (cut-off 0.08 Hz).
- **Blanket:** a cloth texture (weave, folds) rising from the bottom with a slowly undulating edge.
- **Area control:** severity is the **share of the whole image covered** (union of soft alpha masks). The hand has
  a fixed share, and the blanket level is solved per frame so the union matches the target.

| Severity | Target | Measured coverage (min to max) | Hand / blanket share | Hand palm width |
|---|---|---|---|---|
| small | 10% | 10.0%, all frames | 5.0% / 5.0% | 92 px |
| medium | 25% | 25.0%, all frames | 8.7% / 16.3% | 121 px |
| large | 40% | 40.0%, all frames | 11.9% / 28.4% | 142 px |

### C. Lighting: subject4, simulated illumination changes

Each pixel's gain combines exposure (slow), flicker (frame-level), colour (slow), a spatial gradient (slow) and a
moving shadow:

- **Exposure:** slow aperiodic drift, cut-off 0.05 Hz.
- **Gradual fluctuation:** a slowly rotating linear gradient, like a side lamp, plus a slow warm/cool colour shift.
- **Flicker:** aperiodic. It has small frame-wise gain noise plus sparse random dips (lamp instability, 0.1-s
  recovery). The dip events are nested across severities.
- **Moving shadow:** a soft, flat-topped elliptical shadow drifting across the face and body.

| Severity | EV drift | Gradient | Colour | Flicker σ | Dip depth | Shadow | Measured gain range |
|---|---|---|---|---|---|---|---|
| small | ±0.15 | 6% | 1.5% | 0.4% | 4% | 20% | 0.89 to 1.09 |
| medium | ±0.35 | 12% | 3% | 0.8% | 8% | 35% | 0.77 to 1.23 |
| large | ±0.60 | 20% | 6% | 1.5% | 15% | 50% | 0.63 to 1.42 |

**Mains flicker is off by default.** Real 100/120 Hz lamp flicker seen by a ~29.3 fps camera aliases to a narrow
line at a few Hz. That line sits inside the heart-rate band and could be mistaken for a pulse. It can be enabled
explicitly with `lighting.mains_alias.enabled: true`.

## Physiological-signal safety

The augmentations must not create respiration-like or pulse-like periodic patterns. For this reason:

- No sinusoids are used for any movement or illumination drift; all are filtered noise.
- Drift cut-offs (0.05–0.08 Hz) lie below the respiratory band, which is 0.15–1.5 Hz (9–90 breaths/min) and covers
  adult and neonatal rates.
- Tremor (6–10 Hz) lies above the heart-rate band, which is 0.7–4 Hz (42–240 bpm).
- Flicker is aperiodic (broadband), so it adds noise but no spectral line.
- **Automatic check:** every control signal (camera shifts, occluder positions, blanket edge, gains, shadow
  positions) is tested for a narrowband line in each band. The check is a Welch spectrum with a running-median
  local prominence; it flags a band holding > 1% of the power **and** having a line at least 10× the local median.
  - It was validated on 40 seeds × 2 lengths with 0 false flags.
  - It detects injected sinusoids at 0.3, 0.8, 1.2 and 2.9 Hz.
  - Known blind spot: a line between 0.15 and 0.25 Hz sitting next to strong drift may be missed.

## Validation summary (full run)

The run is in `data/validation_report.json` and `run_log.txt`.

| Category / subject | Frames | FPS in, out | Resolution | Bit-exact | Δ duration | Narrowband flags | All checks |
|---|---|---|---|---|---|---|---|
| motion / subject1 (S, M, L) | 1547 | 29.264106, 29.264 | 640×480 | yes | 0.19 ms | 0 | PASS |
| occlusion / subject3 (S, M, L) | 1801 | 29.548271, 29.548 | 640×480 | yes | 0.56 ms | 0 | PASS |
| lighting / subject4 (S, M, L) | 1368 | 29.609362, 29.610 | 640×480 | yes | 0.99 ms | 0 | PASS |

- **Source integrity:** the SHA-256 of all three UBFC videos was identical before and after the run.
- **Temporal consistency:** consecutive-frame differences show no discontinuities. The max/median spike ratio is
  2.4–2.5 for motion vs 1.6 in the original, and 1.7 for occlusion vs 1.8 in the original.
- **Lighting spikes are intended.** For lighting the ratio is 4.1, 6.5 and 9.9 (small, medium, large) vs 1.3 in the
  original. These spikes are the flicker dips, sudden lamp drops with 0.1-s recovery.
- **Largest local prominence** of any control signal in the RR or HR band is 3.6, against a flag threshold of 10.

## Using the outputs for HR evaluation

- Read the `.mkv` files with `cv2.VideoCapture`.
- Use the copied `<subject>_ground_truth.txt`. Its rows are the PPG signal, HR and timestamps, one value per frame.
- Compare fixed-ROI, v1 and dynamic-ROI pipelines per severity, with the same HR estimator and windows for all.
- Per-frame parameters (`*_frames.csv`) allow error to be analysed against instantaneous motion, coverage or gain.
- Do not tune methods on these videos.

## Limitations

- Only one subject per category. For statistics, extend `n_videos`/`assignment`, or apply all three categories to
  more subjects.
- Camera motion is simulated as a global 2-D affine warp. There is no parallax, no subject motion, no motion blur
  and no rolling shutter.
- Occluders are flat, synthetic and pulse-free. They have an approximate shadow and no interaction with the body.
  They are not swaddling.
- Lighting uses a gamma-2.2 approximation of the unknown camera response.
- The subject box comes from an OpenCV Haar face detector on the first frames (adult faces). Infant footage may need
  a different detector.

## Fast, large motion (`motion_fast`, added 2026-11-15)

The slow-drift `motion` category never moves the face out of a box fixed on the first frame (at least 57 % of the
face stays inside it in every frame, even at *large*), so it cannot separate a fixed ROI from a tracked one.
`motion_fast` (module `src/motion_fast.py`, config `config/motion_fast_config.yaml`) does:

```bash
python dataaugment/main.py --config dataaugment/config/motion_fast_config.yaml
```

* Source: subject1 (same as `motion`). Output: `dataaugment/data_motion_fast/motion_fast/{small,medium,large}/`
  (own folder, so the original `validation_report.json` / `manifest.csv` are not overwritten).
* Quick repositioning moves (minimum-jerk, 0.25-0.6 s) at random Poisson times (mean 3 s, min gap 1 s; first 2 s
  unchanged), held in between; plus small slow drift and 6-10 Hz tremor. Same trajectories for all severities.
* Severities: max shift 50 / 100 / 160 px (0.31 / 0.62 / 1.0 face widths), rotation 4 / 8 / 12 deg, zoom 3 / 6 / 10 %.
* Face inside the fixed box (centre 50 % of the first-frame person box): mean 83 / 71 / 54 %; frames with less than
  half of the face inside: 0 / 7 / 46 %.
* All validation checks pass (bit-exact, fps, frame count, no narrowband line in the RR/HR bands).
* HR evaluation: `hr_eval/run_hr_eval_fast.sh` (run `ubfc_aug_fast`); the report gains a "Fast, large motion" panel.
