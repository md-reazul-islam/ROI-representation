"""HR-specific region set (hr.region_set: hr_specific): Module 2 chooses its own HR regions from ALL Module 1 v2
candidates instead of using the Module 1 v2 top 3 (which is ranked for ROI quality, not for pulse).

Per 8 s window (1 s step):
  eligible   candidate regions (face, forehead, cheeks, upper face, upper chest) with enough skin pixels, a large
             enough skin fraction of the polygon (covered regions drop out) and Module 1 availability;
  evidence   pulse strength of each region measured ONLY on past windows (ending before this window starts, within
             history_sec): mean excess SNR over the region's own block-shuffle null threshold;
  score      prior (Module 1 v2 quality q x log skin pixels, no pulse information) x sigmoid(evidence / scale);
             without history (warm-up) the prior alone;
  selection  greedy top-k by score, never two regions whose polygons overlap by more than max_overlap
             (e.g. face + cheek), with hysteresis (a new set must beat the current one by delta for h windows);
  fusion     score-weighted mean of the selected regions' pulses; split-half agreement on their left / right halves.
Validity: the whole select-then-measure procedure is re-run on block-shuffled surrogates of all candidates (same
priors, same region thresholds), so selection on noise cannot validate itself: a window is 'present' only when the
fused SNR beats that null, 'validated' when the two halves also agree.

Gate 2 (agreement), cfg['gate2_mode']:
  strict    (default, original) left vs right halves of the selected regions; a missing half counts as 'present'.
  fallback  a half is AVAILABLE only if it alone passes eligibility (skin pixels, skin fraction, availability), has
            enough valid samples and an HR peak inside the band (not at its edge). The FIRST pair with both halves
            available is used: left_right, face_chest, top_bottom (non-overlapping pixels). |HR_a - HR_b| <= agreement_bpm
            -> 'validated', otherwise 'present' (no further pair is tried). No pair available -> gate 1 only:
            'validated_one_side'. Gate 1 (the surrogate null) is unchanged.
"""
import json
import math

import cv2
import numpy as np

from module1 import paper_eval as pe
from module1_v2 import baselines as bl
from module1_v2.selection import NAMES
from module2.extraction import hold_indices
from module2.rppg import analysis_windows, estimate_hr, kalman_smooth, signal_quality
from hr_rr_m1v2 import validity as va

SIDES = ('', 'L', 'R')
SIDES_TB = ('', 'L', 'R', 'T', 'B')
FACE_GROUP = ('FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE')


def half_areas(rows, names, sides):
    """{(frame, region, side): polygon area} of the half polygons (same clipping as the extraction)."""
    from hr_rr_m1v2.hr import half_polygon
    out = {}
    for r in rows[rows.available.astype(bool) & rows.roi_name.isin(names)].itertuples():
        if not isinstance(r.pixel_polygon, str):
            continue
        poly = json.loads(r.pixel_polygon)
        if not poly or len(poly) < 3:
            continue
        for side in sides:
            h = half_polygon(poly, side)
            if h is not None and len(h) >= 3:
                out[(int(r.frame_index), r.roi_name, side)] = polygon_area(h)
    return out


def half_hr(units, a, b, ctx):
    """(HR or None, reason) of one gate-2 half = units [(region, side, weight)] fused with the weights.
    Available only if the half alone passes eligibility (min_skin_px, min_skin_fraction, min_available_fraction),
    has >= min_valid_fraction valid samples (estimate_hr) and its HR peak lies inside the band, not at its edge."""
    spec, hr_config = ctx['spec'], ctx['hr_config']
    if not units:
        return None, 'none'
    cs = [ctx['index'][(n, s)] for n, s, _ in units]
    c = ctx['counts'][a:b][:, cs]
    tot = np.where(np.isfinite(c) & (c > 0), c, 0.).sum(axis=1)
    present = tot > 0
    if present.mean() < spec['min_available_fraction']:
        return None, 'not present'
    px = float(np.median(tot[present]))
    if px < spec['min_skin_px']:
        return None, f'{px:.0f} px'
    hold = ctx['hold']
    src = sorted(set(int(hold[i]) for i in range(a, b) if hold[i] >= 0))
    areas = [sum(ctx['areas'].get((f, n, s), 0.) for n, s, _ in units) for f in src]
    areas = [x for x in areas if x > 0]
    frac = px / np.median(areas) if areas else 0.
    if frac < spec['min_skin_fraction']:
        return None, f'skin {frac:.2f}'
    P = ctx['pulses'][a:b][:, cs]
    f = pe.fused_pulse(P, np.tile(np.asarray([x for _, _, x in units], float), (b - a, 1)))
    hr, _ = estimate_hr(f, ctx['fps'], hr_config)
    if hr is None:
        return None, 'samples'
    if not (hr_config.hr_min_bpm + 1. < hr < hr_config.hr_max_bpm - 1.):
        return None, f'band edge {hr:.0f}'
    return hr, 'ok'


def gate2_pairs(sel, wts, w, names, eligible, prior, spec):
    """Ordered (name, half_a units, half_b units) for gate2_mode 'fallback'."""
    face_group = spec.get('gate2_face_regions', FACE_GROUP)
    chest = spec.get('gate2_chest_region', 'UPPER_CHEST')
    face = [(names[j], '', float(x)) for j, x in zip(sel, wts) if names[j] in face_group]
    if not face:                                    # no face region selected: the best eligible face region (prior only)
        cand = [j for j, n in enumerate(names) if n in face_group and eligible[w, j]]
        if cand:
            face = [(names[max(cand, key=lambda j: prior[w, j])], '', 1.)]
    pairs = {'left_right': ([(names[j], 'L', float(x)) for j, x in zip(sel, wts)], [(names[j], 'R', float(x)) for j, x in zip(sel, wts)]),
             'face_chest': (face, [(chest, '', 1.)] if chest in names else []),
             'top_bottom': ([(names[j], 'T', float(x)) for j, x in zip(sel, wts)], [(names[j], 'B', float(x)) for j, x in zip(sel, wts)])}
    return [(p, *pairs[p]) for p in spec.get('gate2_pairs', ['left_right', 'face_chest', 'top_bottom'])]


def polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.
    x, y = p[:, 0], p[:, 1]
    return .5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def overlap_matrix(rows, names, step=10, scale=.25):
    """Median over sampled frames of intersection / smaller area for every pair of candidate polygons."""
    frames = sorted(rows.frame_index.unique())[::step]
    k = len(names)
    values = [[[] for _ in range(k)] for _ in range(k)]
    for f in frames:
        part = rows[(rows.frame_index == f) & rows.available.astype(bool)]
        masks = {}
        for r in part.itertuples():
            if r.roi_name in names and isinstance(r.pixel_polygon, str):
                poly = json.loads(r.pixel_polygon)
                if poly and len(poly) >= 3:
                    p = np.rint(np.asarray(poly, float) * scale).astype(np.int32)
                    h, w = int(p[:, 1].max()) + 2, int(p[:, 0].max()) + 2
                    masks[r.roi_name] = (p, h, w)
        if not masks:
            continue
        H = max(m[1] for m in masks.values())
        Wd = max(m[2] for m in masks.values())
        raster = {}
        for name, (p, _, _) in masks.items():
            m = np.zeros((H, Wd), np.uint8)
            cv2.fillPoly(m, [p], 1)
            raster[name] = m.astype(bool)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i < j and a in raster and b in raster:
                    small = min(raster[a].sum(), raster[b].sum())
                    if small > 0:
                        values[i][j].append((raster[a] & raster[b]).sum() / small)
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = np.median(values[i][j]) if values[i][j] else 0.
    return out


def window_priors(rows, counts, names, index, windows, hold, fps, spec):
    """(W, k) prior score and eligibility from non-pulse information only (skin pixels, skin fraction, q_v2)."""
    k = len(names)
    area = {}
    qv = {}
    for r in rows[rows.available.astype(bool) & rows.roi_name.isin(names)].itertuples():
        if isinstance(r.pixel_polygon, str):
            area[(int(r.frame_index), r.roi_name)] = polygon_area(json.loads(r.pixel_polygon))
        qv[(int(r.frame_index), r.roi_name)] = float(r.q_v2) if r.q_v2 == r.q_v2 else np.nan
    prior = np.zeros((len(windows), k))
    eligible = np.zeros((len(windows), k), bool)
    pixels = np.full((len(windows), k), np.nan)
    for w, (a, b) in enumerate(windows):
        src = sorted(set(int(hold[i]) for i in range(a, b) if hold[i] >= 0))
        for j, name in enumerate(names):
            c = counts[a:b, index[(name, '')]]
            finite = np.isfinite(c) & (c > 0)
            if finite.mean() < spec['min_available_fraction']:
                continue
            px = float(np.median(c[finite]))
            areas = [area[(f, name)] for f in src if (f, name) in area]
            frac = px / np.median(areas) if areas and np.median(areas) > 0 else 0.
            q = np.nanmean([qv.get((f, name), np.nan) for f in src]) if src else np.nan
            q = q if np.isfinite(q) else .5
            pixels[w, j] = px
            if px >= spec['min_skin_px'] and frac >= spec['min_skin_fraction']:
                eligible[w, j] = True
                prior[w, j] = max(q, 0.) * math.log1p(px)
    if prior.max() > 0:
        prior = prior / prior.max()
    return prior, eligible, pixels


def select(excess, prior, eligible, windows, overlap, fps, spec):
    """Per window: (selected region indices, weights, source). Evidence only from windows ending before the start."""
    W, k = excess.shape
    hist = int(round(spec['history_sec'] * fps))
    current, streak, out = [], 0, []
    for w, (a, b) in enumerate(windows):
        past = [j for j, (c, d) in enumerate(windows[:w]) if d <= a and d > a - hist]
        if past:
            ev = np.nanmean(excess[past], axis=0)
            gain = 1. / (1. + np.exp(-np.nan_to_num(ev, nan=-10.) / spec['evidence_scale_db']))
            source = 'history'
        else:
            gain = np.full(k, .5)
            source = 'prior'
        score = np.where(eligible[w], prior[w] * gain, 0.)
        if past:                                                 # once there is evidence: only regions that beat their
            beats = np.nan_to_num(ev, nan=-10.) > 0              # own null are added (if any does)
            if (beats & eligible[w]).any():
                score = np.where(beats, score, 0.)
        floor = spec.get('min_relative_score', .25) * score.max()
        proposed = []
        for j in np.argsort(-score):
            if score[j] <= 0 or score[j] < floor or len(proposed) >= spec['k']:
                break
            if all(overlap[j, i] <= spec['max_overlap'] for i in proposed):
                proposed.append(int(j))
        if not current or any(not eligible[w, j] or score[j] <= 0 for j in current):   # a member lost its evidence
            current, streak = proposed, 0
        elif set(proposed) != set(current):
            if score[proposed].sum() > (1 + spec['hysteresis_delta']) * score[current].sum():
                streak += 1
                if streak >= spec['hysteresis_windows']:
                    current, streak = proposed, 0
            else:
                streak = 0
        else:
            streak = 0
        s = score[current] if current else np.array([])
        weights = s / s.sum() if len(s) and s.sum() > 0 else (np.full(len(current), 1. / len(current)) if current else s)
        out.append((list(current), weights, source))
    return out


def region_snr(pulses, cols, windows, fps, hr_config):
    snr = np.full((len(windows), len(cols)), np.nan)
    for w, (a, b) in enumerate(windows):
        for j, c in enumerate(cols):
            _, s = estimate_hr(pulses[a:b, c], fps, hr_config)
            snr[w, j] = s if s is not None else np.nan
    return snr


def fuse(pulses, cols, chosen, weights, a, b):
    if not chosen:
        return None
    P = pulses[a:b][:, [cols[j] for j in chosen]]
    Wm = np.tile(np.asarray(weights, float), (b - a, 1))
    return pe.fused_pulse(P, Wm)


def analyse(record, rows, masks, cfg, hr_config, extract_config, pulse, seed, extract, unit_index, waves=None):
    spec = cfg['hr_specific']
    fps, n = float(record['fps']), int(record['frame_count'])
    frames = sorted(rows.frame_index.unique())
    hold = hold_indices(frames, n, fps, extract_config.max_hold_sec)
    mode = cfg.get('gate2_mode', 'strict')
    assert mode in ('strict', 'fallback'), f'gate2_mode must be strict or fallback, not {mode}'
    sides = SIDES_TB if mode == 'fallback' else SIDES
    index = unit_index(sides) if mode == 'fallback' else unit_index()
    counts = np.full((n, len(index)), np.nan)
    rgb = extract(record, rows, masks, extract_config, hold, counts=counts, sides=sides) if mode == 'fallback' \
        else extract(record, rows, masks, extract_config, hold, counts=counts)
    names = [r for r in spec['candidates'] if r in NAMES]
    cols = {s: [index[(r, s)] for r in names] for s in SIDES}
    windows = analysis_windows(n, fps, hr_config)
    pulses = np.full((n, len(index)), np.nan)
    for r in names:
        for s in sides:
            pulses[:, index[(r, s)]] = pulse(rgb[:, index[(r, s)]], fps)
    block = max(1, int(round(cfg.get('shuffle_block_sec', .25) * fps)))
    vid = int(record['video_id'][-3:]) if record['video_id'][-3:].isdigit() else 0

    # per-region null thresholds (block shuffle of that region's own pixels)
    thr = np.full(len(names), np.nan)
    for j, r in enumerate(names):
        null = []
        for s in range(spec['region_surrogates']):
            rng = np.random.default_rng([seed, vid, 1000 + j, s])
            y = pulse(bl.shuffle_surrogate(rgb[:, index[(r, '')]], rng, block), fps)
            for a, b in windows:
                _, v = estimate_hr(y[a:b], fps, hr_config)
                null.append(v if v is not None else np.nan)
        null = np.asarray(null, float)
        thr[j] = np.nanpercentile(null, cfg['null_percentile']) if np.isfinite(null).any() else np.nan
    snr = region_snr(pulses, cols[''], windows, fps, hr_config)
    excess = snr - thr[None, :]
    prior, eligible, pixels = window_priors(rows, counts, names, index, windows, hold, fps, spec)
    overlap = overlap_matrix(rows, names)
    chosen = select(excess, prior, eligible, windows, overlap, fps, spec)

    # whole-procedure null: surrogates of ALL candidates -> same selection rule -> fused SNR
    null = []
    for s in range(cfg['surrogates']):
        sp = np.full((n, len(index)), np.nan)
        for j, r in enumerate(names):
            rng = np.random.default_rng([seed, vid, s, 7])
            sp[:, index[(r, '')]] = pulse(bl.shuffle_surrogate(rgb[:, index[(r, '')]], rng, block), fps)
        ex_s = region_snr(sp, cols[''], windows, fps, hr_config) - thr[None, :]
        for w, ((a, b), (sel, wts, _)) in enumerate(zip(windows, select(ex_s, prior, eligible, windows, overlap, fps, spec))):
            f = fuse(sp, cols[''], sel, wts, a, b)
            if f is not None:
                _, v = estimate_hr(f, fps, hr_config)
                null.append(v if v is not None else np.nan)
    null = np.asarray(null, float)
    threshold = np.nanpercentile(null, cfg['null_percentile']) if np.isfinite(null).any() else np.nan

    motion = rows[rows.roi_name.isin(names)].groupby(['frame_index', 'roi_name']).motion_magnitude.mean() \
        if 'motion_magnitude' in rows else None
    out, previous = [], None
    native = np.zeros((n, len(names)))
    ctx = dict(spec=spec, hr_config=hr_config, index=index, counts=counts, pulses=pulses, hold=hold, fps=fps,
               areas=half_areas(rows, names, ('L', 'R', 'T', 'B')) if mode == 'fallback' else {})
    for w, ((a, b), (sel, wts, source)) in enumerate(zip(windows, chosen)):
        f = fuse(pulses, cols[''], sel, wts, a, b)
        hr, s_db = estimate_hr(f, fps, hr_config) if f is not None else (None, None)
        fl = fuse(pulses, cols['L'], sel, wts, a, b)
        fr = fuse(pulses, cols['R'], sel, wts, a, b)
        hl = estimate_hr(fl, fps, hr_config)[0] if fl is not None else None
        hrr = estimate_hr(fr, fps, hr_config)[0] if fr is not None else None
        mot = None
        if motion is not None and sel:
            src = sorted(set(int(hold[i]) for i in range(a, b) if hold[i] >= 0))
            vals = [motion.get((f_, names[j]), np.nan) for f_ in src for j in sel]
            vals = [v for v in vals if v == v]
            mot = float(np.mean(vals)) if vals else None
        sqi = signal_quality(s_db, mot, hr_config) if s_db is not None else 0.
        is_present = s_db is not None and math.isfinite(threshold) and s_db > threshold
        agreement = bl.agree(hl, hrr, cfg['agreement_bpm'])
        if mode == 'fallback':
            pair_used, ha, hb, notes, agree2 = 'none', None, None, [], None
            for pname, ua, ub in gate2_pairs(sel, wts, w, names, eligible, prior, spec):
                ha_, why_a = half_hr(ua, a, b, ctx)
                hb_, why_b = half_hr(ub, a, b, ctx)
                if ha_ is not None and hb_ is not None:
                    pair_used, ha, hb = pname, ha_, hb_
                    agree2 = bool(abs(ha - hb) <= cfg['agreement_bpm'])
                    break                                   # first available pair decides; no further pair is tried
                notes.append(f'{pname}: ' + '; '.join(f'{k} {v}' for k, v in (('a', why_a), ('b', why_b)) if v != 'ok'))
            label = va.label_fallback(is_present, agree2)
        else:
            pair_used, ha, hb, notes = ('left_right' if agreement is not None else 'none'), hl, hrr, []
            label = va.label(is_present, agreement)
        regions = {names[j]: round(float(x), 4) for j, x in zip(sel, wts)}
        out.append(dict(window=w, window_start_sec=a / fps, window_end_sec=b / fps, window_center_sec=(a + b) / 2 / fps,
                        hr_bpm=hr, snr_db=s_db, sqi=sqi, motion=mot, null_threshold_db=threshold, pulse_present=is_present,
                        hr_left_bpm=hl, hr_right_bpm=hrr, halves_agree=agreement, label=label,
                        hr_regions=json.dumps(regions), selection_source=source,
                        region_switch=bool(previous is not None and set(previous) != set(sel)),
                        region_snr_db=json.dumps({r: (None if not np.isfinite(v) else round(float(v), 2)) for r, v in zip(names, snr[w])}),
                        region_null_db=json.dumps({r: (None if not np.isfinite(v) else round(float(v), 2)) for r, v in zip(names, thr)}),
                        gate2_mode=mode, pair_used=pair_used, hr_a=ha, hr_b=hb,
                        diff=abs(ha - hb) if ha is not None and hb is not None else None,
                        one_side_only=label == 'validated_one_side', gate2_unavailable='; '.join(notes)))
        previous = sel
        c0 = int(round((a + b) / 2 - fps / 2)) if w else 0
        native[c0:, :] = 0.
        for j, x in zip(sel, wts):
            native[c0:, j] = x
    kal = va.smoothed([r['window_end_sec'] for r in out], [r['hr_bpm'] for r in out], [r['label'] for r in out],
                      [r['sqi'] for r in out], hr_config, kalman_smooth,
                      cfg.get('one_side_noise_factor', 2.) if mode == 'fallback' else None)
    for r, k in zip(out, kal):
        r['hr_kalman_bpm'] = k
    if waves is not None:
        waves.update(pulse=pe.fused_pulse(pulses[:, cols['']], native), pulse_fps=fps)
    return out
