"""Compare re-computed results with the released ones (released/ holds an untouched copy of every result file).

    python tools/verify_results.py paper      # rebuilt manuscript text == paper/NeoSTAR_manuscript.docx
    python tools/verify_results.py neonatal   # Level C result tables
    python tools/verify_results.py ubfc       # Level B result tables
Numbers are compared with a tolerance (--rtol, default 1e-6): GPU inference is not bit-exact across hardware and
library versions, so small differences after a full re-run are expected and are reported, not hidden.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from common import ROOT

GROUPS = {
    'neonatal': ['module1_v2/reports/final34_paper/phase1', 'module1_v2/reports/final34_paper/phase2',
                 'module1_v2/reports/final34_paper/phase5', 'module1_v2/reports/final34_paper/consistency',
                 'module1_v2/reports/final34_paper/fusion_test', 'module1_v2/runs/final34_paper/tuning_log.json'],
    'ubfc': ['dataaugment/hr_eval/report', 'module1_v2_d8/report', 'module1_v2_d9/report',
             'module1_v2_d9/swaddle_motion/report', 'occlusion_position_test/demo_centre/report',
             'face_track_baseline/report', 'HR RR using module1 v2/runs'],
}


def compare_csv(a, b, rtol):
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return False, f'shape/columns differ ({x.shape} vs {y.shape})'
    worst, msg = 0.0, ''
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            uu, vv = u.to_numpy(float), v.to_numpy(float)
            if not np.array_equal(np.isnan(uu), np.isnan(vv)):
                return False, f'missing values differ in {c}'
            ok = ~np.isnan(uu)
            if ok.any():
                rel = np.abs(uu[ok] - vv[ok]) / np.maximum(1e-12, np.abs(vv[ok]))
                if rel.max() > worst:
                    worst, msg = float(rel.max()), c
        elif not (u.astype(str) == v.astype(str)).all():
            return False, f'text column {c} differs'
    return worst <= rtol, (f'max relative difference {worst:.2e} ({msg})' if worst else 'identical')


def paper():
    import docx
    new = sorted(ROOT.glob('module1_v2/reports/final34_paper/submission_*/NeoSTAR_manuscript.docx'))
    ref = ROOT/'paper/NeoSTAR_manuscript.docx'
    if not new:
        sys.exit('no rebuilt manuscript found (run Step A1)')
    text = lambda f: [p.text for p in docx.Document(f).paragraphs] + \
        [c.text for t in docx.Document(f).tables for r in t.rows for c in r.cells]  # noqa: E731
    a, b = text(new[0]), text(ref)
    same = a == b
    print(f'manuscript text: {"IDENTICAL to" if same else "DIFFERS from"} paper/NeoSTAR_manuscript.docx '
          f'({len(a)} paragraphs and table cells)')
    if not same:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print('first difference:\n  rebuilt :', x[:300], '\n  released:', y[:300])
                break
    return same


def group(name, rtol):
    ok_all, n = True, 0
    for part in GROUPS[name]:
        rel = ROOT/'released'/part
        files = [rel] if rel.is_file() else sorted(p for p in rel.rglob('*') if p.suffix in ('.csv', '.json'))
        for r in files:
            cur = ROOT/r.relative_to(ROOT/'released')
            n += 1
            if not cur.is_file():
                print(f'MISSING  {cur.relative_to(ROOT)}')
                ok_all = False
                continue
            if r.suffix == '.csv':
                ok, msg = compare_csv(cur, r, rtol)
            else:
                ok, msg = cur.read_bytes() == r.read_bytes(), 'identical' if cur.read_bytes() == r.read_bytes() else 'differs'
            ok_all &= ok
            if not ok or msg != 'identical':
                print(f'{"OK  " if ok else "DIFF"}     {cur.relative_to(ROOT)}: {msg}')
    print(f'{name}: {n} files compared, {"all within tolerance" if ok_all else "differences found (see above)"}')
    return ok_all


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['paper', 'neonatal', 'ubfc'])
    ap.add_argument('--rtol', type=float, default=1e-6)
    a = ap.parse_args()
    sys.exit(0 if (paper() if a.what == 'paper' else group(a.what, a.rtol)) else 1)
