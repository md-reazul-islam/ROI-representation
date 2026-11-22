"""Phase 1a tuning objective (TRAINING videos only).

OBJECTIVE (documented here, used by scripts/tune_phase1.py):
  Among all candidate settings whose mean (over training videos) quality retention is >= `retention_min`
  (default 0.98), choose the one with the fewest top-1 region switches per second (mean over training
  videos); ties are broken by higher retention. If no setting reaches the retention floor, the setting
  with the highest retention is chosen and this is logged.

  quality retention = mean over frames of sum(rescaled Q of the selected set) / sum(rescaled Q of the
  per-frame top-k), i.e. how much instantaneous confidence the stabilized selection gives up.

Why not a pulse-SNR objective: the objective must be computable on the training split. In the 5-video
dataset all three training videos are monochrome infrared recordings, on which colour-based pulse SNR
and skin likelihood are not meaningful; the objective therefore uses only Q and selection dynamics.
The video is the averaging unit, so long videos do not dominate.
"""
from __future__ import annotations

import math


def choose(results, retention_min=.98):
    """results: list of dicts with 'params', 'switches_per_sec', 'retention' (means over training videos)."""
    finite = [r for r in results if math.isfinite(r['switches_per_sec']) and math.isfinite(r['retention'])]
    if not finite:
        raise ValueError('no finite tuning result')
    feasible = [r for r in finite if r['retention'] >= retention_min]
    if not feasible:
        best = max(finite, key=lambda r: r['retention'])
        return dict(best, feasible=False)
    best = min(feasible, key=lambda r: (r['switches_per_sec'], -r['retention']))
    return dict(best, feasible=True)
