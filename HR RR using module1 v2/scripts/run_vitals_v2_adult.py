"""UBFC / adult wrapper around run_vitals_v2.py (which stays unchanged, infant band 90-270 bpm).

Runs exactly the same HR pipeline with an adult HR search band (default 45-180 bpm), also applied to the POS
band-pass of the Module 1 pulse helper, and records the band in metadata.json. Used only by the UBFC notebooks.

Usage: python "HR RR using module1 v2/scripts/run_vitals_v2_adult.py" --hr-band 45,180 <all run_vitals_v2.py arguments>
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_vitals_v2 as rv                                    # noqa: E402  (unchanged infant script)
import run_paper_evaluation                                   # noqa: E402  (already on sys.path via rv)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--hr-band', default='45,180')
    known, rest = pre.parse_known_args(argv)
    low, high = (float(v) for v in known.hr_band.split(','))
    rv.HR = dataclasses.replace(rv.HR, hr_min_bpm=low, hr_max_bpm=high)      # HR search band of the HR branch
    run_paper_evaluation.HR = rv.HR                                           # POS band-pass in pulse()
    summarize = rv.summarize
    rv.summarize = lambda t, value, lo, hi: summarize(t, value, low, high) if value == 'hr_bpm' else summarize(t, value, lo, hi)
    multiprocessing.set_start_method('fork', force=True)     # workers inherit the patched band
    code = rv.main(rest)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--run'); p.add_argument('--module1-config', default=rv.CFG['input']['module1_config'])
    p.add_argument('--out-name'); p.add_argument('--hr-region-set')
    a, _ = p.parse_known_args(rest)
    suffix = '_hr_specific' if (a.hr_region_set or rv.CFG['hr'].get('region_set')) == 'hr_specific' else ''
    meta_path = rv.M2/'runs'/Path(a.run).name/(a.out_name or a.module1_config + suffix)/'metadata.json'
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        meta['hr_band_bpm'] = [low, high]
        meta['runner'] = 'run_vitals_v2_adult.py'
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
