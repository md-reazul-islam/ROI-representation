"""Plain-language HTML report of a fixed vs dynamic ROI comparison (compare_roi_fixed_vs_dynamic.py output).

Usage: python "HR RR using module1 v2/scripts/build_roi_comparison_report.py" --run runs/<run> [--out-name roi_comparison]
Writes HR RR using module1 v2/runs/<run>/<out-name>/ROI_comparison_report.html (self-contained: figures embedded).

Verdict rule per condition (fixed in advance, not tuned on the result)
  with a reference HR : dynamic (D5_HR) vs the best fixed ROI on MAE of reported windows; 'better' needs >= 1 bpm lower
                        MAE AND coverage no more than 10 points lower, or the same MAE (< 1 bpm apart) with >= 10 points
                        more coverage; 'worse' is the mirror image; otherwise 'tie'.
  without a reference : pulse evidence = gate-1 pass % (windows whose pulse beats the shuffled-signal null). 'more
                        pulse evidence' needs >= 10 points above the best fixed ROI; mirror for 'less'; otherwise 'tie'.
                        If no ROI passes in more than 15 % of the windows of a condition (chance is ~5 %), the condition
                        is 'no usable pulse': it cannot decide anything.
  conditions with fewer than 5 windows are not judged.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', ROOT/'module1_v2/src', ROOT/'src', ROOT/'scripts', M2/'scripts'):
    sys.path.insert(0, str(p))
import run_paper_evaluation as rpe                                  # noqa: E402
from module1 import paper_eval as pe                                # noqa: E402
from module2.extraction import hold_indices                         # noqa: E402
from module2.rppg import HRConfig, analysis_windows                 # noqa: E402
from compare_roi_fixed_vs_dynamic import bounding_rect, window_boxes  # noqa: E402

# reference palette (dataviz skill): first three categorical slots (validated all-pairs), grey for the control,
# ordinal blue ramp for window labels (light step >= 250 so every step clears 2:1 on the surface)
FAMILY = {'fixed': '#2a78d6', 'tracked baseline': '#eb6834', 'dynamic (ours)': '#1baf7a', 'control': '#52514e'}
METHOD = {  # method: (family, marker, plain name)
    'B0': ('fixed', 'o', 'Fixed box (first frame, whole video)'),
    'B0_WIN': ('fixed', 's', 'Fixed box re-placed every 8 s'),
    'B1': ('fixed', '^', 'Fixed anatomical region (first frame)'),
    'FACE_TRACK': ('tracked baseline', 'o', 'Tracked face box'),
    'M0': ('tracked baseline', 's', 'Whole visible baby skin'),
    'D5': ('dynamic (ours)', 'o', 'Dynamic ROI, Module 1 v2 (may switch inside a window)'),
    'D5_WIN': ('dynamic (ours)', 's', 'Dynamic ROI, choice frozen per 8 s window'),
    'D5_HR': ('dynamic (ours)', 'D', 'Dynamic ROI + HR-specific regions (our full method)'),
    'BG': ('control', 'x', 'Background corners (negative control, chance)'),
}
LABEL_COLOR = {'not_assessable': '#e8e7e3', 'present': '#86b6ef', 'validated_one_side': '#3987e5', 'validated': '#1c5cab'}
LABEL_TEXT = {'not_assessable': 'no pulse found', 'present': 'pulse found, halves disagree / untestable',
              'validated_one_side': 'validated (one side only)', 'validated': 'validated HR'}
COND_TEXT = {'moving': 'Baby moving', 'camera_motion': 'Camera moving', 'posture_change': 'Posture change / turning',
             'covered_low_skin': 'Covered (blanket / swaddle)', 'occluded_few_regions': 'Occluded (few regions visible)',
             'illumination_change': 'Lighting changing', 'dark': 'Dark', 'blurry': 'Blurry', 'caregiver': 'Caregiver in view',
             'clean': 'Clean (none of the above)', 'all': 'All windows'}
CLAIM = ['moving', 'posture_change', 'covered_low_skin', 'occluded_few_regions', 'illumination_change', 'dark']
FIXED, DYN = ('B0', 'B0_WIN', 'B1'), 'D5_HR'
MIN_WINDOWS, PULSE_FLOOR, PP, BPM, COVER = 5, 15., 10., 1., 10.
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e8e7e3'


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#fcfcfb')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor('#fcfcfb')


def verdicts(s, has_ref, methods):
    rows = []
    fixed = [m for m in FIXED if m in methods]
    for g in ['all', 'clean'] + list(COND_TEXT):
        if g in ('all', 'clean') and rows and any(r['group'] == g for r in rows):
            continue
        p = s[s.group == g].set_index('method')
        if not len(p) or DYN not in p.index:
            continue
        n = int(p.windows.iloc[0]) if pd.notna(p.windows.iloc[0]) else 0
        row = dict(group=g, windows=n)
        if n < MIN_WINDOWS:
            rows.append(dict(row, verdict='not judged', why=f'only {n} windows (< {MIN_WINDOWS})'))
            continue
        cov_dyn = p.at[DYN, 'reported_percent']
        if has_ref and 'mae_reported' in p:
            best = min(fixed, key=lambda m: p.at[m, 'mae_reported'] if pd.notna(p.at[m, 'mae_reported']) else np.inf)
            md, mf = p.at[DYN, 'mae_reported'], p.at[best, 'mae_reported']
            cd, cf = cov_dyn, p.at[best, 'reported_percent']
            if not (pd.notna(md) and pd.notna(mf)):
                v, why = 'not judged', 'one side reports no HR in this condition'
            elif md <= mf - BPM and cd >= cf - COVER:
                v, why = 'dynamic better', f'MAE {md:.1f} vs {mf:.1f} bpm ({best}), coverage {cd:.0f} % vs {cf:.0f} %'
            elif md >= mf + BPM and cd <= cf + COVER:
                v, why = 'fixed better', f'MAE {md:.1f} vs {mf:.1f} bpm ({best}), coverage {cd:.0f} % vs {cf:.0f} %'
            elif abs(md - mf) < BPM and cd >= cf + COVER:
                v, why = 'dynamic better', f'same accuracy (MAE {md:.1f} vs {mf:.1f} bpm, {best}) but more coverage {cd:.0f} % vs {cf:.0f} %'
            elif abs(md - mf) < BPM and cd <= cf - COVER:
                v, why = 'fixed better', f'same accuracy (MAE {md:.1f} vs {mf:.1f} bpm, {best}) but more coverage {cf:.0f} % vs {cd:.0f} %'
            else:
                v, why = 'tie', f'MAE {md:.1f} vs {mf:.1f} bpm ({best}), coverage {cd:.0f} % vs {cf:.0f} %'
            row.update(best_fixed=best, dyn_metric=md, fixed_metric=mf)
        else:
            gd = p.at[DYN, 'gate1_percent']
            best = max(fixed, key=lambda m: p.at[m, 'gate1_percent'])
            gf = p.at[best, 'gate1_percent']
            top = p.drop(index=[m for m in ('BG',) if m in p.index]).gate1_percent.max()
            if top <= PULSE_FLOOR:
                v, why = 'no usable pulse', f'best ROI passes the pulse test in {top:.0f} % of windows (chance ~5 %)'
            elif gd >= gf + PP:
                v, why = 'dynamic: more pulse evidence', f'pulse test passed {gd:.0f} % vs {gf:.0f} % ({best})'
            elif gd <= gf - PP:
                v, why = 'fixed: more pulse evidence', f'pulse test passed {gd:.0f} % vs {gf:.0f} % ({best})'
            else:
                v, why = 'tie', f'pulse test passed {gd:.0f} % vs {gf:.0f} % ({best})'
            row.update(best_fixed=best, dyn_metric=gd, fixed_metric=gf)
        rows.append(dict(row, verdict=v, why=why))
    return pd.DataFrame(rows)


def timeline(w, methods):
    conds = [c for c in COND_TEXT if f'cond_{c}' in w and c not in ('clean', 'all')]
    t = w.window_center_sec.to_numpy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 1.9 + .28 * (len(conds) + len(methods))), sharex=True,
                                 gridspec_kw=dict(height_ratios=[len(conds), len(methods)], hspace=.15))
    for k, c in enumerate(conds):
        on = w[f'cond_{c}'].astype(bool).to_numpy()
        a1.barh(np.full(on.sum(), k), np.ones(on.sum()), left=t[on] - .5, height=.6, color=INK2, alpha=.55, lw=0)
    a1.set_yticks(range(len(conds)), [COND_TEXT[c] for c in conds], fontsize=8, color=INK)
    a1.invert_yaxis()
    a1.set_title('Video conditions per window (dark = condition present)', fontsize=9, color=INK, loc='left')
    for k, m in enumerate(methods):
        lab = w[f'label_{m}'].fillna('not_assessable')
        a2.barh(np.full(len(t), k), np.ones(len(t)), left=t - .5, height=.6, lw=0,
                color=[LABEL_COLOR.get(x, LABEL_COLOR['not_assessable']) for x in lab])
    a2.set_yticks(range(len(methods)), [f'{m}' for m in methods], fontsize=8, color=INK)
    a2.invert_yaxis()
    a2.set_title('Result of each ROI per window', fontsize=9, color=INK, loc='left')
    a2.set_xlabel('time in video (s, window centre)', fontsize=8, color=INK2)
    for a in (a1, a2):
        style(a)
        a.tick_params(axis='y', length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, lw=0) for c in LABEL_COLOR.values()]
    fig.legend(handles, LABEL_TEXT.values(), loc='upper center', bbox_to_anchor=(.5, .02), ncol=4, fontsize=7.5,
               frameon=False)
    return png(fig)


def dotplot(s, metric, xlabel, methods, groups):
    p = s[(s.group.isin(groups)) & s.method.isin(methods)]
    fig, ax = plt.subplots(figsize=(8.5, .55 * len(groups) + 1.4))
    offs = np.linspace(-.3, .3, len(methods))
    for k, g in enumerate(groups):
        if k % 2 == 0:
            ax.axhspan(k - .45, k + .45, color='#f0efec', lw=0, zorder=0)
        for o, m in zip(offs, methods):
            v = p[(p.group == g) & (p.method == m)][metric]
            if len(v) and pd.notna(v.iloc[0]):
                fam, mk, _ = METHOD[m]
                ax.scatter(v.iloc[0], k + o, s=34 if mk != 'x' else 30, marker=mk, color=FAMILY[fam],
                           edgecolors='#fcfcfb' if mk != 'x' else None, linewidths=.8, zorder=3)
    ax.set_yticks(range(len(groups)), [f"{COND_TEXT.get(g, g)}  (n={int(np.nan_to_num(s[(s.group == g)].windows.iloc[0]))})"
                                       for g in groups], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.grid(axis='x', color=GRID, lw=.6)
    ax.set_axisbelow(True)
    style(ax)
    ax.tick_params(axis='y', length=0)
    handles = [plt.Line2D([], [], ls='', marker=METHOD[m][1], color=FAMILY[METHOD[m][0]], markersize=6) for m in methods]
    fig.legend(handles, methods, loc='upper center', bbox_to_anchor=(.5, .01), ncol=len(methods), fontsize=7.5,
               frameon=False, handletextpad=.2, columnspacing=1.)
    return png(fig)


REPORTED = ('validated', 'validated_one_side')
HARD = ('moving', 'posture_change', 'covered_low_skin', 'occluded_few_regions', 'illumination_change')


def hr_timeline(w, methods, band):
    """Small multiples: one panel per ROI, same axes. Grey dots = HR of every window (not validated);
    coloured dots = validated windows; line = Kalman track; black = reference HR (if any); shading = difficult windows."""
    ms = [m for m in methods if m != 'BG']
    t = w.window_center_sec.to_numpy()
    hard = np.zeros(len(w), bool)
    for c in HARD:
        if f'cond_{c}' in w:
            hard |= w[f'cond_{c}'].astype(bool).to_numpy()
    vals = np.concatenate([pd.to_numeric(w[f'hr_bpm_{m}'], errors='coerce').to_numpy(float) for m in ms])
    if 'hr_ref' in w:
        vals = np.concatenate([vals, w.hr_ref.to_numpy(float)])
    lo, hi = np.nanpercentile(vals, 1) - 5, np.nanpercentile(vals, 99) + 5
    lo, hi = max(lo, band[0] - 5), min(hi, band[1] + 5)
    fig, axes = plt.subplots(len(ms), 1, figsize=(10, 1.25 * len(ms) + .8), sharex=True, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), ms):
        color = FAMILY[METHOD[m][0]]
        for x, h in zip(t, hard):
            if h:
                ax.axvspan(x - .5, x + .5, color='#f0efec', lw=0, zorder=0)
        hr = pd.to_numeric(w[f'hr_bpm_{m}'], errors='coerce').to_numpy(float)
        rep_ = w[f'label_{m}'].isin(REPORTED).to_numpy()
        ax.scatter(t[~rep_], hr[~rep_], s=7, color='#c3c2b7', lw=0, zorder=2)
        ax.scatter(t[rep_], hr[rep_], s=14, color=color, lw=0, zorder=3)
        k = pd.to_numeric(w[f'hr_kalman_bpm_{m}'], errors='coerce').to_numpy(float)
        ax.plot(t, k, color=color, lw=1.6, zorder=4)
        if 'hr_ref' in w and w.hr_ref.notna().any():
            ax.plot(t, w.hr_ref.to_numpy(float), color=INK, lw=1.2, ls='--', zorder=5)
        ax.set_ylim(lo, hi)
        ax.text(1.005, .5, m, transform=ax.transAxes, fontsize=8.5, color=INK, va='center')
        ax.text(1.005, .15, f'{100 * rep_.mean():.0f} % valid.', transform=ax.transAxes, fontsize=7, color=INK2, va='center')
        ax.grid(axis='y', color=GRID, lw=.6)
        style(ax)
    np.atleast_1d(axes)[-1].set_xlabel('time in video (s, window centre)', fontsize=8, color=INK2)
    fig.text(.07, .5, 'HR (bpm)', rotation=90, va='center', fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], ls='', marker='o', color='#c3c2b7', markersize=4),
               plt.Line2D([], [], ls='', marker='o', color=INK2, markersize=5),
               plt.Line2D([], [], color=INK2, lw=1.6), plt.Rectangle((0, 0), 1, 1, color='#f0efec')]
    names = ['window HR, not validated', 'window HR, validated (colour = ROI family)', 'Kalman HR track',
             'difficult window (moving / posture / covered / occluded / lighting)']
    if 'hr_ref' in w and w.hr_ref.notna().any():
        handles.append(plt.Line2D([], [], color=INK, lw=1.2, ls='--'))
        names.append('reference HR')
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .01), ncol=3, fontsize=7.5, frameon=False)
    fig.subplots_adjust(hspace=.12)
    return png(fig)


def hr_table(w, s, methods, groups, has_ref, tol=5.):
    """Per condition and ROI: validated windows, median validated HR, median jump between consecutive validated
    windows, agreement with the dynamic ROI (D5_HR) where both report, and (with reference) MAE."""
    masks = {'all': np.ones(len(w), bool), 'clean': w.cond_clean.astype(bool).to_numpy()}
    for g in groups:
        if f'cond_{g}' in w:
            masks[g] = w[f'cond_{g}'].astype(bool).to_numpy()
    rows = []
    dyn_hr = pd.to_numeric(w[f'hr_bpm_{DYN}'], errors='coerce').to_numpy(float)
    dyn_rep = w[f'label_{DYN}'].isin(REPORTED).to_numpy()
    for g in groups:
        keep = masks.get(g)
        if keep is None:
            continue
        for m in [x for x in methods if x != 'BG']:
            hr = pd.to_numeric(w[f'hr_bpm_{m}'], errors='coerce').to_numpy(float)
            rep_ = w[f'label_{m}'].isin(REPORTED).to_numpy()
            sel = keep & rep_
            pv = s[(s.group == g) & (s.method == m)]
            row = {'condition': COND_TEXT.get(g, g), 'ROI': m, 'windows': int(keep.sum()), 'validated': int(sel.sum()),
                   'median HR (bpm)': float(np.nanmedian(hr[sel])) if sel.any() else np.nan,
                   'HR jump (bpm)': float(pv.hr_jump_median_bpm.iloc[0]) if len(pv) else np.nan}
            if m != DYN:
                both = keep & rep_ & dyn_rep & np.isfinite(hr) & np.isfinite(dyn_hr)
                row[f'agrees with {DYN} ±{tol:g} (%)'] = 100 * float(np.mean(np.abs(hr[both] - dyn_hr[both]) <= tol)) if both.any() else np.nan
                row['both validated'] = int(both.sum())
            else:
                row[f'agrees with {DYN} ±{tol:g} (%)'], row['both validated'] = np.nan, np.nan
            if has_ref and len(pv) and 'mae_reported' in pv:
                row['MAE (bpm)'] = float(pv.mae_reported.iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def stills(run, w, methods, picks):
    """Frames at chosen windows with the fixed box, the re-placed box, the face box and the dynamic regions drawn."""
    record = pd.read_csv(ROOT/run/'manifest.csv')
    video = w.video_id.iloc[0]
    record = record[record.video_id == video].iloc[0].to_dict()
    rows = pd.read_csv(ROOT/'module1_v2/runs'/Path(run).name/'dynamic_roi_v2_D5'/'dynamic_roi_v2.csv', low_memory=False)
    rows = rows[rows.video_id == video]
    cand = pd.read_csv(ROOT/run/'outputs/confidence_sam2/roi_confidence.csv', low_memory=False,
                       usecols=['video_id', 'frame_index', 'roi_name', 'roi_status', 'polygon', 'anatomical_confidence'])
    cand = cand[cand.video_id == video]
    rec = pd.read_csv(ROOT/run/'outputs/recovery/recovery_results.csv', low_memory=False)
    boxes = rpe.boxes_of(rec[rec.video_id == video])
    b0, _ = pe.fixed_rois(cand, boxes)
    fps, n = float(record['fps']), int(record['frame_count'])
    hold = hold_indices(sorted(rows.frame_index.unique()), n, fps, rpe.EXTRACT.max_hold_sec)
    windows = analysis_windows(n, fps, HRConfig())
    wboxes = window_boxes(boxes, hold, windows)
    cap = cv2.VideoCapture(str(record['video_path']))
    out = []
    bgr = lambda h: tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))
    for title, wi in picks:
        r = w[w.window == wi].iloc[0]
        f = int(round(r.window_center_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        src = int(hold[f]) if hold[f] >= 0 else int(rows.frame_index.iloc[(rows.frame_index - f).abs().argmin()])
        at = rows[(rows.frame_index == src) & rows.available]
        th = max(2, img.shape[1] // 400)

        def draw(poly, color, dash=False):
            if poly is None:
                return
            p = np.rint(np.asarray(poly, float)).astype(np.int32)
            cv2.polylines(img, [p], True, (255, 255, 255), th + 3, cv2.LINE_AA)
            cv2.polylines(img, [p], True, bgr(color), th, cv2.LINE_AA)
        draw(b0, FAMILY['fixed'])
        draw(wboxes[wi], '#86b6ef')
        face = at[at.roi_name == 'FACE']
        if len(face):
            draw(bounding_rect(json.loads(face.pixel_polygon.iloc[0])), FAMILY['tracked baseline'])
        regions = json.loads(r.regions_D5_HR) if isinstance(r.get('regions_D5_HR'), str) else {}
        for name in regions:
            q = at[at.roi_name == name]
            if len(q) and isinstance(q.pixel_polygon.iloc[0], str):
                draw(json.loads(q.pixel_polygon.iloc[0]), FAMILY['dynamic (ours)'])
        scale = 900 / img.shape[1]
        img = cv2.resize(img, (900, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        res = ', '.join(f"{m}: {LABEL_TEXT.get(r.get(f'label_{m}'), '?')}" for m in ('B0', 'B0_WIN', 'FACE_TRACK', 'D5_HR') if m in methods)
        out.append(dict(title=title, t=r.window_center_sec, img=base64.b64encode(buf).decode(),
                        regions=', '.join(regions) or 'none selected', result=res))
    cap.release()
    return out


def pick_windows(w):
    picks, used = [], set()
    for c, title in (('clean', 'Clean moment'), ('moving', 'Baby moving'), ('posture_change', 'Posture change'),
                     ('covered_low_skin', 'Covered / low skin'), ('illumination_change', 'Lighting changing')):
        col = f'cond_{c}'
        if col not in w:
            continue
        cand = w[w[col].astype(bool) & ~w.window.isin(used)]
        if len(cand):
            wi = int(cand.window.iloc[len(cand) // 2])
            picks.append((title, wi))
            used.update(range(wi - 8, wi + 9))
    return picks


def table_html(df, fmt=None):
    fmt = fmt or {}
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    body = ''
    for _, r in df.iterrows():
        cells = ''
        for c in df.columns:
            v = r[c]
            if isinstance(v, float):
                v = '' if not math.isfinite(v) else (fmt.get(c, '{:.1f}').format(v))
            cells += f'<td>{html.escape(str(v))}</td>'
        body += f'<tr>{cells}</tr>'
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out-name', default='roi_comparison')
    a = ap.parse_args(argv)
    run = a.run
    target = M2/'runs'/Path(run).name/a.out_name
    meta = json.loads((target/'metadata.json').read_text())
    s = pd.read_csv(target/'summary_by_condition.csv')
    w = pd.read_csv(target/'window_results_with_conditions.csv')
    eff = pd.read_csv(target/'effects_by_condition.csv') if (target/'effects_by_condition.csv').is_file() else pd.DataFrame()
    methods = [m for m in METHOD if m in meta['methods']]
    has_ref = bool(meta.get('reference_hr'))
    v = verdicts(s, has_ref, methods)
    claim = v[v.group.isin(CLAIM)]
    judged = claim[~claim.verdict.isin(['not judged', 'no usable pulse'])]
    win = judged.verdict.str.startswith('dynamic').sum()
    lose = judged.verdict.str.startswith('fixed').sum()
    if not len(judged):
        head_cls, headline = 'na', ('Cannot be judged on this video: in the difficult conditions no ROI found a usable pulse '
                                    '(or too few windows).')
    elif win and not lose:
        head_cls, headline = 'yes', (f'Supports the claim on this video: the dynamic ROI was ahead in {win} of {len(judged)} '
                                     'difficult conditions that could be judged, and behind in none.')
    elif lose and not win:
        head_cls, headline = 'no', (f'Does not support the claim on this video: a fixed ROI was ahead in {lose} of '
                                    f'{len(judged)} difficult conditions that could be judged.')
    else:
        head_cls, headline = 'mixed', (f'Mixed / not shown on this video: dynamic ahead in {win}, fixed ahead in {lose}, tie in '
                                       f'{len(judged) - win - lose} of {len(judged)} difficult conditions.')
    scope = ('HR accuracy against the reference' if has_ref else
             'pulse evidence only (no reference HR: this shows which ROI carries a detectable pulse, NOT whose HR is correct)')
    groups = ['all', 'clean'] + [g for g in CLAIM + ['camera_motion', 'blurry', 'caregiver']
                                 if g in set(s.group) and s[s.group == g].windows.iloc[0] >= MIN_WINDOWS]
    fig_t = timeline(w, methods)
    fig_g = dotplot(s, 'gate1_percent', 'windows where the pulse test passed (%)  -  background control = chance level',
                    methods, groups)
    fig_r = dotplot(s, 'reported_percent', 'windows with a reported (validated) HR (%)', methods, groups)
    fig_m = dotplot(s, 'mae_reported', 'HR error on reported windows, MAE (bpm)  -  lower is better', methods, groups) \
        if has_ref and 'mae_reported' in s else None
    shots = stills(run, w, methods, pick_windows(w))
    fig_hr = hr_timeline(w, methods, meta['hr_band_bpm'])
    hrt = hr_table(w, s, methods, groups, has_ref)
    hrt.to_csv(target/'hr_by_roi_and_condition.csv', index=False)
    hrt_html = table_html(hrt, {'both validated': '{:.0f}'})
    thr = pd.DataFrame([dict(condition=COND_TEXT.get(k, k), measure=t['measure'], **{'hard when': t['hard']},
                             threshold=t['threshold'], source=t['source'],
                             windows=int(np.nan_to_num(s[(s.group == k)].windows.iloc[0])) if (s.group == k).any() else 0)
                        for k, t in meta['condition_thresholds'].items()])
    cols = ['reported_percent', 'gate1_percent', 'snr_median_db', 'agreement_percent'] + \
        (['mae_reported', 'within5_reported_percent', 'mae_all'] if has_ref else [])
    num = s[s.group.isin(groups) & s.method.isin(methods)][['group', 'method', 'windows'] + [c for c in cols if c in s]].copy()
    num['group'] = num.group.map(lambda g: COND_TEXT.get(g, g))
    eff_html = ''
    if len(eff):
        metric = 'mae_reported' if has_ref else 'gate1_above_BG_pp'
        e = eff[(eff.metric == metric) & eff.group.isin(['all', 'clean'] + CLAIM)].copy()
        if len(e):
            e['group'] = e.group.map(lambda g: COND_TEXT.get(g, g))
            e['contrast'] = e.effect + ': ' + e.method + ' − ' + e.minus
            tab = e.pivot_table(index='group', columns='contrast', values='mean_diff', aggfunc='first').reset_index()
            eff_html = table_html(tab, {c: '{:+.1f}' for c in tab.columns})
    vt = v.copy()
    vt['condition'] = vt.group.map(lambda g: COND_TEXT.get(g, g))
    vt = vt[['condition', 'windows', 'verdict', 'why']]
    vrows = ''.join(f"<tr class='{('d' if r.verdict.startswith('dynamic') else 'f' if r.verdict.startswith('fixed') else 'n')}'>"
                    f"<td>{html.escape(r.condition)}</td><td>{r.windows}</td><td><b>{html.escape(r.verdict)}</b></td>"
                    f"<td>{html.escape(r.why)}</td></tr>" for r in vt.itertuples())
    mrows = ''.join(f"<tr><td><span class='sw' style='background:{FAMILY[METHOD[m][0]]}'></span>{m}</td>"
                    f"<td>{html.escape(METHOD[m][2])}</td><td>{METHOD[m][0]}</td></tr>" for m in methods)
    shots_html = ''.join(f"<figure><img src='data:image/jpeg;base64,{x['img']}' alt='{html.escape(x['title'])}'>"
                         f"<figcaption><b>{html.escape(x['title'])}</b> · t = {x['t']:.0f} s · dynamic regions: "
                         f"{html.escape(x['regions'])}<br><span class='mut'>{html.escape(x['result'])}</span></figcaption></figure>"
                         for x in shots)
    videos = ', '.join(sorted(w.video_id.unique()))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fixed vs Dynamic ROI</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e8e7e3;--band:#f0efec;--yes:#1baf7a;--no:#e34948;--mix:#eda100;--na:#8a8984}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0}}
main{{max-width:1000px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}}
.mut{{color:var(--ink2);font-size:13px}} .head{{border-left:6px solid var(--na);background:var(--band);padding:14px 16px;margin:18px 0;border-radius:4px}}
.head.yes{{border-color:var(--yes)}} .head.no{{border-color:var(--no)}} .head.mixed{{border-color:var(--mix)}}
.head b{{font-size:17px}} .tw{{overflow-x:auto}} table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--ink2);font-weight:600}}
tr.d td:nth-child(3){{color:#0d7a54}} tr.f td:nth-child(3){{color:#b42f2e}} img{{max-width:100%;height:auto;display:block}}
figure{{margin:16px 0}} figcaption{{font-size:13px;margin-top:6px}} .sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
ul{{padding-left:20px}} code{{font-size:12px}}
</style></head><body><main>
<h1>Fixed vs dynamic ROI</h1>
<div class="mut">Video: {html.escape(videos)} · {len(w)} windows of 8 s (1 s step) · HR band {meta['hr_band_bpm'][0]:.0f}–{meta['hr_band_bpm'][1]:.0f} bpm · reference HR: {'yes' if has_ref else 'none'}</div>
<div class="head {head_cls}"><b>{html.escape(headline)}</b><br><span class="mut">What this measures: {html.escape(scope)}.
One video, overlapping windows: this is a case study, not proof. The claim needs many infants with a reference HR.</span></div>

<h2>1. Verdict per condition</h2>
<p class="mut">Rule fixed before looking at the results: the full dynamic method (D5_HR) is compared with the best of the three fixed ROIs.
{('“Better” needs ≥ 1 bpm lower MAE with coverage no more than 10 points lower, or the same MAE (within 1 bpm) with ≥ 10 points more coverage.' if has_ref else '“More pulse evidence” needs ≥ 10 percentage points more windows passing the pulse test. If no ROI passes in more than 15 % of windows (chance ≈ 5 %), the condition cannot decide anything.')}
Conditions with fewer than {MIN_WINDOWS} windows are not judged.</p>
<div class="tw"><table><thead><tr><th>condition</th><th>windows</th><th>verdict</th><th>why</th></tr></thead><tbody>{vrows}</tbody></table></div>

<h2>2. What was compared</h2>
<p>Every ROI goes through exactly the same chain: same frames, skin-pixel rule, POS pulse, 8 s windows, pulse test against shuffled signals (gate 1),
agreement of two halves (gate 2) and Kalman. <b>Only the pixels used differ.</b></p>
<div class="tw"><table><thead><tr><th>method</th><th>ROI</th><th>family</th></tr></thead><tbody>{mrows}</tbody></table></div>

<h2>3. Conditions in this video and what each ROI found</h2>
<p class="mut">Top: which conditions were present in each window (measured automatically from the video). Bottom: the result of every ROI in the same window.
Read down a column to compare ROIs at the same moment.</p>
<img src="data:image/png;base64,{fig_t}" alt="timeline of conditions and per-ROI window results">

<h2>4. Heart rate measured by each ROI</h2>
<p class="mut">Every ROI measures its own HR in every 8 s window. One panel per ROI, same axes. Grey dots = window HR that did not pass validation
(not reported); coloured dots = validated HR; line = the Kalman HR track that would be reported; shaded = difficult windows.
{'Dashed black = reference HR: the closer, the better.' if has_ref else '<b>No reference HR</b>: this shows how steady and how often each ROI reports an HR, not which HR is correct. Where the ROIs agree, the HR is probably real; where one jumps away from the others, it is probably wrong.'}</p>
<img src="data:image/png;base64,{fig_hr}" alt="HR over time per ROI">
<h3 style="font-size:15px">HR per ROI and condition</h3>
<p class="mut"><b>validated</b> = windows with a reported HR · <b>HR jump</b> = median change between consecutive validated windows (lower = steadier; a real infant HR rarely changes by more than a few bpm per second)
· <b>agrees with D5_HR</b> = share of windows where both report and their HR is within 5 bpm{' · <b>MAE</b> = error against the reference' if has_ref else ''}.</p>
{hrt_html}

<h2>5. Pulse evidence by condition</h2>
<p class="mut">Share of windows in which the ROI's signal beat its shuffled-signal null. Grey × = background corners: that is what chance looks like.
Blue = fixed, orange = tracked baselines, green = dynamic (ours).</p>
<img src="data:image/png;base64,{fig_g}" alt="pulse test pass rate by condition and ROI">
<h2>6. Reported HR (coverage) by condition</h2>
<p class="mut">Share of windows with a validated HR (pulse test passed and the two halves agree within 6 bpm).</p>
<img src="data:image/png;base64,{fig_r}" alt="validated HR coverage by condition and ROI">
{f'<h2>6b. HR error against the reference</h2><img src="data:image/png;base64,{fig_m}" alt="MAE by condition and ROI">' if fig_m else ''}

<h2>7. Where each ROI looked</h2>
<p class="mut"><span class="sw" style="background:#2a78d6"></span>fixed box (first frame) ·
<span class="sw" style="background:#86b6ef"></span>fixed box re-placed for this window ·
<span class="sw" style="background:#eb6834"></span>tracked face box ·
<span class="sw" style="background:#1baf7a"></span>dynamic HR regions (D5_HR)</p>
{shots_html or '<p class="mut">No frames could be drawn.</p>'}

<h2>8. Which part of “dynamic” matters</h2>
<p class="mut">Difference between two methods ({'MAE, bpm: negative = first method better' if has_ref else 'pulse-test pass rate above background, percentage points: positive = first method better'}).
switching = freezing the choice per window; tracking = following the baby; selection = choosing HR-specific regions.</p>
{eff_html or '<p class="mut">not available</p>'}

<h2>9. All numbers</h2>
{table_html(num)}
<h3 style="font-size:15px">Condition thresholds</h3>
<p class="mut">“Tertile” = the worst third of this video's windows. For a frozen study, set fixed thresholds in configs/roi_comparison.yaml.</p>
{table_html(thr, {'threshold': '{:.4g}'})}

<h2>10. How far this result goes</h2>
<ul>
<li>{'There is a reference HR, so MAE shows accuracy.' if has_ref else '<b>No reference HR</b>: the report shows which ROI carries a detectable pulse, not whose heart rate is correct. To claim “better HR” you need ECG / pulse-oximeter HR recorded with the video.'}</li>
<li><b>One video</b>: windows overlap (8 s, 1 s step), so neighbouring windows are not independent. Effects need many infants and per-infant statistics.</li>
<li>Conditions are measured automatically and ranked within this video (worst third). A “moving” window here is relative to this video.</li>
<li>Gate 2 needs ≥ 10 % skin in each half of a ROI; large boxes over hair or bedding often cannot be tested and stay “pulse found”.</li>
</ul>
<p class="mut">Files: {html.escape(str(target))}</p>
</main></body></html>"""
    out = target/'ROI_comparison_report.html'
    out.write_text(page)
    v.to_csv(target/'verdicts_by_condition.csv', index=False)
    print(headline)
    print(v[['group', 'windows', 'verdict', 'why']].to_string(index=False))
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
