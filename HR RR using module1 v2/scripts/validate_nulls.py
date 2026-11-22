"""Check the two surrogate nulls on synthetic video with a known answer (true 120 bpm pulse vs static skin patch).
A valid pulse-present null must pass most windows of a true pulse and about 5% (or fewer) of a static patch.
Usage: python "HR RR using module1 v2/scripts/validate_nulls.py --out module1_v2/reports/final34_paper/phase2/null_validation.json
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', M2/'tests', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts'):
    sys.path.insert(0, str(p))
from test_hr_rr_m1v2 import _pulse_video                   # noqa: E402
from module1_v2 import baselines as bl                     # noqa: E402
from module2.extraction import ExtractionConfig, mean_rgb  # noqa: E402
from module2.rppg import HRConfig, analysis_windows, estimate_hr  # noqa: E402
from run_paper_evaluation import pulse                     # noqa: E402

POLYGON = np.array([[70, 50], [250, 50], [250, 190], [70, 190]], float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--surrogates', type=int, default=49)
    a = ap.parse_args(argv)
    hr, ex, fps = HRConfig(), ExtractionConfig(skin_mode='none'), 30.
    result = {}
    for name, pulsing in (('true_pulse_120bpm', True), ('static_patch', False)):
        path = Path(tempfile.mkdtemp())/'v.avi'
        n = _pulse_video(path, 120, pulsing=pulsing)
        cap = cv2.VideoCapture(str(path))
        rgb = np.array([mean_rgb(cap.read()[1], POLYGON, ex)[0] for _ in range(n)], float)
        windows = analysis_windows(n, fps, hr)
        observed = np.array([estimate_hr(pulse(rgb, fps)[a_:b], fps, hr)[1] for a_, b in windows])
        rng = np.random.default_rng(0)
        null = {'phase': [], 'shuffle': []}
        for _ in range(a.surrogates):
            for kind in null:
                trace = bl.phase_surrogate(rgb, rng) if kind == 'phase' else bl.shuffle_surrogate(rgb, rng, int(.25 * fps))
                s = pulse(trace, fps)
                null[kind] += [estimate_hr(s[a_:b], fps, hr)[1] for a_, b in windows]
        result[name] = {'windows': len(windows), 'observed_snr_median_db': float(np.median(observed))}
        for kind, values in null.items():
            thr = float(np.percentile(values, 95))
            result[name][f'{kind}_null_p95_db'] = thr
            result[name][f'{kind}_null_pass_percent'] = float(100 * np.mean(observed > thr))
    Path(a.out).write_text(json.dumps(dict(surrogates=a.surrogates, shuffle_block_sec=.25, **result), indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
