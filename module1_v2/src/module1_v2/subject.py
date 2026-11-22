"""Phase 1d: infant-vs-adult subject verification from body proportions.

The person-class detector and the adult-trained pose model can lock onto a caregiver. Two scale-free
proportion cues separate infants from adults:
  r1 = ear-to-ear distance / shoulder width   (head breadth relative to shoulders; approximate
       anthropometry: newborn ~0.7, adult ~0.4)
  r2 = mean arm length (shoulder-elbow + elbow-wrist) / torso length (shoulder midpoint to hip midpoint)
       (newborn ~0.8, adult ~1.15)
Thresholds are priors from these approximate values (midpoints), NOT fitted: no infant/adult labels exist.
A frame is ADULT if r1 < r1_min (or, when r1 is not measurable, r2 > r2_max), INFANT if the cue passes,
and UNKNOWN if neither cue is measurable. Per track, the decision is the majority of known per-frame
decisions in a trailing window; UNKNOWN never causes rejection (the check can only abstain on evidence).
"""
from __future__ import annotations

import math

import numpy as np

PARTS = ('left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
         'left_wrist', 'right_wrist', 'left_hip', 'right_hip')


def part_indices(names):
    index = {name: i for i, name in enumerate(names)}
    missing = [p for p in PARTS if p not in index]
    if missing:
        raise ValueError(f'keypoint names lack {missing}')
    return {p: index[p] for p in PARTS}


def _point(points, index, threshold):
    p = points.get(index)
    return None if p is None or p[2] < threshold else np.array(p[:2], float)


def proportions(points, parts, threshold=.5):
    """(r1, r2) of one frame; NaN where the cue's keypoints are not confident."""
    get = lambda name: _point(points, parts[name], threshold)
    le, re_, ls, rs = get('left_ear'), get('right_ear'), get('left_shoulder'), get('right_shoulder')
    r1 = np.nan
    if all(p is not None for p in (le, re_, ls, rs)):
        shoulders = np.linalg.norm(ls - rs)
        if shoulders > 1:
            r1 = float(np.linalg.norm(le - re_) / shoulders)
    r2 = np.nan
    lh, rh = get('left_hip'), get('right_hip')
    if all(p is not None for p in (ls, rs, lh, rh)):
        torso = np.linalg.norm((ls + rs) / 2 - (lh + rh) / 2)
        arms = []
        for side, shoulder in (('left', ls), ('right', rs)):
            elbow, wrist = get(f'{side}_elbow'), get(f'{side}_wrist')
            if elbow is not None and wrist is not None:
                arms.append(np.linalg.norm(shoulder - elbow) + np.linalg.norm(elbow - wrist))
        if arms and torso > 1:
            r2 = float(np.mean(arms) / torso)
    return r1, r2


def frame_decision(r1, r2, r1_min=.55, r2_max=1.):
    if math.isfinite(r1):
        return 'INFANT' if r1 >= r1_min else 'ADULT'
    if math.isfinite(r2):
        return 'INFANT' if r2 <= r2_max else 'ADULT'
    return 'UNKNOWN'


def track_decisions(frame_decisions, track_ids, times, window_sec=2.):
    """Accept/reject per frame: reject when, among known decisions of the same track in the trailing
    window, ADULT is the majority. Memory resets when the track ID changes."""
    out, history, current = [], [], object()
    for decision, track, time in zip(frame_decisions, track_ids, times):
        if track != current:
            history, current = [], track
        history.append((time, decision))
        history = [(t, d) for t, d in history if time - t <= window_sec]
        known = [d for _, d in history if d != 'UNKNOWN']
        adult = sum(d == 'ADULT' for d in known)
        out.append('REJECT' if known and adult > len(known) / 2 else 'ACCEPT')
    return out


def episodes(flags):
    """(start, stop) index runs where flags is True."""
    runs, start = [], None
    for i, flag in enumerate(list(flags) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    return runs
