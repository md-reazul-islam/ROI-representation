# ROI representation (NeoSTAR)

Code, released results and a step-by-step notebook for the paper

> **NeoSTAR: A Stable, Auditable Dynamic ROI Representation for Neonatal Video**

NeoSTAR is a learning-free, auditable dynamic region-of-interest (ROI) representation for newborn video. It adds
temporal stabilization (confidence smoothing, hysteresis and Viterbi decoding) and an independent keypoint-visibility
term to an anatomy- and mask-guided pipeline (detection, SAM 2 infant mask, Sapiens2 pose, 17 anatomical candidate
regions). No model is trained or fine-tuned; only four stabilizer settings are selected on training videos.

## Quick start

```bash
conda create -n neostar python=3.12 -y && conda activate neostar
pip install -r environment/requirements-paper.txt
jupyter lab ROI_representation.ipynb        # run the cells top to bottom
```

Level A of the notebook rebuilds every table, figure and the full manuscript from the released result tables in a few
minutes on a CPU, and checks that the text is identical to `paper/NeoSTAR_manuscript.docx`.

## What the notebook reproduces

| Level | Reproduces | Needs | Time |
|---|---|---|---|
| A | all tables, figures, statistics and the manuscript, from the released results | Python (CPU) | minutes |
| C | the neonatal results (stability, label-free validity tests, internal consistency, cost) from raw video | the 34 neonatal videos, a GPU | ~2 days |
| B | the heart-rate robustness study on UBFC-rPPG with controlled perturbations, from raw video | UBFC-rPPG, a GPU | ~1-2 days |

Dataset locations are set in **Step 1** of the notebook (or in `config/paths.yaml`); `tools/setup_paths.py` links them
to the locations the code expects, so no code needs editing. After Levels B/C, `tools/verify_results.py` compares every
re-computed table with the untouched copy in `released/`.

## Data

* **Neonatal videos (34).** Short, publicly accessible web videos of newborns. They are third-party content and are not
  redistributed. `runs/final34_paper/manifest.csv` lists every file with its SHA-256 hash, frame rate, duration,
  resolution and split; Step 2 verifies your files against it.
* **UBFC-rPPG** (Bobbia et al., Pattern Recognition Letters 2019), subjects 1, 3 and 4 of DATASET_2, available from its
  authors. Perturbed videos (motion, lighting, occlusion, swaddling) are generated from it with logged seeds.
* **Model weights** (downloaded by `tools/download_models.py`, SHA-256 checked): YOLO11s and RT-DETR-L (Ultralytics),
  SAM 2.1 Hiera-T (Meta), Sapiens2-0.4B pose (Meta, Hugging Face `facebook/sapiens2-pose-0.4b`).

## Layout

| Folder | Content |
|---|---|
| `ROI_representation.ipynb` | the step-by-step notebook |
| `config/paths.yaml` | dataset and model locations |
| `tools/` | path setup, UBFC preparation, model download, result verification |
| `src/`, `scripts/`, `configs/` | base pipeline: detection, tracking, recovery, SAM 2, pose, candidate ROIs, condition terms |
| `module1_v2/` | NeoSTAR: code (`src/module1_v2`), evaluation scripts, configuration, tests, manuscript builder |
| `HR RR using module1 v2/` | heart-rate chain used to probe ROI quality (POS, validity gates) |
| `dataaugment/` | motion, fast-motion, lighting and occlusion generators; HR evaluation |
| `module1_v2_d8/`, `module1_v2_d9/` | swaddling videos and tests, incl. the pre-registrations (`PREREGISTRATION.md`) |
| `occlusion_position_test/` | hand-over-the-face-centre case (supplement S4) and its pre-registration |
| `face_track_baseline/` | tracked face-box baseline |
| `released/` | untouched copies of all released result tables (for verification) |
| `paper/` | the submitted package: manuscript, highlights, cover letter, supplement, Data S1, figures |

Folder names follow the project in which the experiments were run, so every script runs unchanged. "PF-ROI" in the
paper is the configuration `Full`/`D0` (internally "v1"); "NeoSTAR" is `D5` (internally "Module 1 v2").

## Notes on reproducibility

* All splits, seeds and stabilizer settings are fixed in files: `runs/final34_paper/manifest.csv` (seed 42),
  `module1_v2/runs/final34_paper/tuning_log.json`, the generator configs (seeds 20261114-20261117) and the bootstrap
  seed 42. The headline configuration is recorded in `module1_v2/reports/final34_paper/headline.json`.
* GPU inference is not bit-exact across hardware and library versions; after a full re-run, small numeric differences
  are reported by `tools/verify_results.py`.
* `phase1_prefix/`, `consistency_prefix/` and `phase2_s19/` are historical results (before the two-point-box fix and
  with 19 surrogates) that the paper quotes for comparison; the current code does not regenerate them.
* The pre-registered 30-video occlusion-position test in `occlusion_position_test/PREREGISTRATION.md` was not run; the
  paper reports only the single designed case, labelled as an illustration.
* Figure 1 is a fixed image with all infant faces blurred (`module1_v2/manuscript/assets/`).

## Citation and licence

Please cite the paper (reference to be added on publication).

The code in this repository is released under the [MIT License](LICENSE). Third-party components keep their own
licences and are not included: SAM 2 and Sapiens2 (installed by `tools/install_external.sh`), the model weights
(downloaded by `tools/download_models.py`) and the datasets (UBFC-rPPG is available from its authors under its own
terms; the neonatal videos are not redistributed).
