"""Reference-based robustness study (UBFC-rPPG + controlled perturbations) for the UPDATED v2 manuscript.

Read-only: every number is read from the existing report CSVs listed in SOURCES; nothing is re-run or re-tuned.
Used only by build_manuscript_v2_updated.py (the original builder and manuscript are unchanged).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'aug_paired': ROOT/'dataaugment/hr_eval/report/paired_differences.csv',      # D5 - fixed: motion, fast motion, lighting, occlusion
    'aug_metrics': ROOT/'module1_v2_d8/report/metrics.csv',                    # MAE + no-HR of fixed / v1 / D5 for the same videos
    'aug_windows': ROOT/'dataaugment/hr_eval/report/windows_merged.csv',
    'sw_metrics': ROOT/'module1_v2_d9/report/metrics.csv',                     # swaddling alone, subjects 3 + 4 (pre-registered)
    'sw_paired': ROOT/'module1_v2_d9/report/paired.csv',
    'swmo_metrics': ROOT/'module1_v2_d9/swaddle_motion/report/metrics.csv',    # swaddling + fast motion, subjects 1, 3, 4 (pre-registered)
    'swmo_paired': ROOT/'module1_v2_d9/swaddle_motion/report/paired.csv',
    'swmo_windows': ROOT/'module1_v2_d9/swaddle_motion/report/windows.csv',
    'occ_metrics': ROOT/'occlusion_position_test/demo_centre/report/metrics.csv',  # hand over the face centre (designed case)
    'occ_paired': ROOT/'occlusion_position_test/demo_centre/report/paired.csv',
    'occ_windows': ROOT/'occlusion_position_test/demo_centre/report/windows.csv',
    'ft_metrics': ROOT/'face_track_baseline/report/ubfc_metrics.csv',          # tracked face box, same videos and windows
    'ft_paired': ROOT/'face_track_baseline/report/ubfc_paired.csv',
    'ft_geo': ROOT/'face_track_baseline/report/final34_geometry.csv',
    'ft_f34': ROOT/'HR RR using module1 v2/runs/final34_paper/facetrack/window_results.csv',
    'f34_split': ROOT/'module1_v2/reports/final34_paper/phase1/per_video_metrics.csv',
}
LEVEL = {'none': 'none', 'small': 'small', 'medium': 'medium', 'large': 'large', 'xlarge': 'extra-large'}
INK, INK2, GRID, MUTED, BLUE, ORANGE, SHADE = '#0b0b0b', '#52514e', '#e4e3df', '#9a998f', '#2a78d6', '#eb6834', '#f3e3d3'


def _verdict(lo, hi):
    if hi < 0:
        return 'NeoSTAR better'
    if lo > 0:
        return 'fixed box better'
    return 'no clear difference'


def load():
    s = {k: pd.read_csv(v) for k, v in SOURCES.items()}
    rows = []

    def add(cond, level, subjects, design, mae, nohr, diff, windows):
        d = dict(condition=cond, level=level, subjects=subjects, design=design, windows=windows,
                 diff=diff[0], lo=diff[1], hi=diff[2], verdict=_verdict(diff[1], diff[2]))
        for k in ('fixed', 'v1', 'D5'):
            d[f'mae_{k}'], d[f'nohr_{k}'] = mae[k], nohr[k]
        rows.append(d)

    # undisturbed: originals of subjects 1, 3, 4 (from the pre-registered swaddle+motion report, level 'none')
    sm, sp = s['swmo_metrics'], s['swmo_paired']
    def swmo(level):
        m = sm[sm.level == level].set_index('method')
        p = sp[(sp.level == level) & (sp.contrast == 'D5 - fixed')].iloc[0]
        return ({k: m.at[k, 'mae'] for k in ('fixed', 'v1', 'D5')}, {k: int(m.at[k, 'no_hr']) for k in ('fixed', 'v1', 'D5')},
                (p.mean_diff_bpm, p.ci95_low, p.ci95_high), int(m.at['fixed', 'windows']))
    mae, nohr, diff, w = swmo('none')
    add('Undisturbed', 'none', '1, 3, 4', 'original videos', mae, nohr, diff, w)

    am, ap = s['aug_metrics'], s['aug_paired']
    for cat, name, subj in (('motion', 'Slow camera motion', '1'), ('motion_fast', 'Fast, large motion', '1'),
                            ('lighting', 'Lighting change', '4'), ('occlusion', 'Hand + blanket occlusion', '3')):
        for lev in ('small', 'medium', 'large', 'xlarge'):
            m = am[(am.category == cat) & (am.severity == lev)].set_index('method_key')
            if m.empty:
                continue
            p = ap[(ap.category == cat) & (ap.severity == lev) & (ap.contrast == 'D5 - Fixed box')].iloc[0]
            add(name, LEVEL[lev], subj, 'fixed protocol', {k: m.at[k, 'mae_all'] for k in ('fixed', 'v1', 'D5')},
                {k: int(m.at[k, 'no_hr']) for k in ('fixed', 'v1', 'D5')},
                (p.mean_abs_error_difference_bpm, p.ci95_low, p.ci95_high), int(m.at['fixed', 'windows']))

    wm, wp = s['sw_metrics'], s['sw_paired']
    for lev in ('small', 'medium', 'large'):
        m = wm[wm.severity == lev].set_index('method')
        p = wp[(wp.severity == lev) & (wp.contrast == 'D5 - fixed')].iloc[0]
        add('Swaddling only', lev, '3, 4', 'pre-registered', {k: m.at[k, 'mae'] for k in ('fixed', 'v1', 'D5')},
            {k: int(m.at[k, 'no_hr']) for k in ('fixed', 'v1', 'D5')}, (p.mean_diff_bpm, p.ci95_low, p.ci95_high),
            int(m.at['fixed', 'windows']))

    for lev, lab in (('medium', 'to neck'), ('large', 'to mouth')):
        mae, nohr, diff, w = swmo(lev)
        add('Swaddling + fast motion', lab, '1, 3, 4', 'pre-registered', mae, nohr, diff, w)

    om, op = s['occ_metrics'], s['occ_paired']
    m = om[om.windows_set == 'occluded'].set_index('method')
    p = op[(op.windows_set == 'occluded') & (op.contrast == 'D5 - fixed')].iloc[0]
    names = {'fixed': 'Fixed box', 'v1': 'Module 1 v1', 'D5': 'Module 1 v2 D5'}
    add('Hand over face centre', 'hand present', '1', 'designed case', {k: m.at[names[k], 'mae'] for k in names},
        {k: int(m.at[names[k], 'no_hr']) for k in names}, (p.mean_diff_bpm, p.ci95_low, p.ci95_high), int(m.at['Fixed box', 'windows']))

    T = pd.DataFrame(rows)
    T = _add_face(T, s)
    claims = pd.read_csv(ROOT/'module1_v2_d9/swaddle_motion/report/claims.csv')
    return dict(T=T, src=s, claims=claims, F34=_face_neonatal(s))


def _add_face(T, s):
    """Tracked face box (FACE_TRACK): MAE / no-HR and paired contrasts, from the same videos and windows as T."""
    fm = s['ft_metrics']
    m = T.merge(fm[['condition', 'level', 'windows', 'mae_fixed', 'mae_D5', 'mae_FACE', 'nohr_FACE']],
                on=['condition', 'level'], how='left', suffixes=('', '_ft'), validate='one_to_one')
    # the face-box job re-ran the fixed box and read the same D5 windows: both must agree with T exactly
    assert (m.windows == m.windows_ft).all(), 'face-box windows differ from Table 11'
    assert np.allclose(m.mae_fixed, m.mae_fixed_ft, atol=0.01) and np.allclose(m.mae_D5, m.mae_D5_ft, atol=0.01), \
        'face-box run inconsistent'   # observed max difference 0.0015 bpm (reference-matching rounding)
    m = m.drop(columns=['windows_ft', 'mae_fixed_ft', 'mae_D5_ft'])
    fp = s['ft_paired']
    for con, pre in (('D5 - FACE', 'face'), ('FACE - fixed', 'ff')):
        p = fp[fp.contrast == con][['condition', 'level', 'windows', 'mean_diff_bpm', 'ci95_low', 'ci95_high']]
        p = p.rename(columns={'windows': f'{pre}_windows', 'mean_diff_bpm': f'{pre}_diff', 'ci95_low': f'{pre}_lo',
                              'ci95_high': f'{pre}_hi'})
        m = m.merge(p, on=['condition', 'level'], how='left', validate='one_to_one')
    m['face_verdict'] = [('NeoSTAR better' if hi < 0 else 'face box better' if lo > 0 else 'no clear difference')
                         for lo, hi in zip(m.face_lo, m.face_hi)]
    m['ff_verdict'] = [('face box better' if hi < 0 else 'fixed box better' if lo > 0 else 'no clear difference')
                       for lo, hi in zip(m.ff_lo, m.ff_hi)]
    return m


def _face_neonatal(s):
    """Geometry of the tracked face box on the 34 neonatal videos, with the paper's own metric (paper_eval.geometry_metrics),
    next to the fixed box, v1 (Full) and NeoSTAR (D5) from the Phase 1 per-video table. Pulse measures of the face box are
    NOT reported here: they come from a different chain than Section 5.5 and would not be comparable."""
    geo = s['ft_geo'].set_index('method')
    rows = []
    for key, name in (('B0', 'Fixed box (B0)'), ('Full', 'v1 (Full)'), ('FACE_TRACK (face box)', 'Tracked face box'),
                      ('D5', 'NeoSTAR (D5)')):
        g = geo.loc[key]
        rows.append(dict(method=name, videos=int(g.videos), coverage=float(g.coverage_percent),
                         temporal_iou=float(g.temporal_iou), centroid_jitter=float(g.centroid_jitter)))
    return pd.DataFrame(rows)



def row(R, cond, level):
    T = R['T']
    return T[(T.condition == cond) & (T.level == level)].iloc[0]


# ------------------------------------------------------------------ figures
def _style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def forest(R, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = R['T'].copy()
    n = len(T)
    H = 0.27 * n + 2.2
    fig, axes = plt.subplots(1, 2, figsize=(7.6, H), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(n)[::-1]
    panels = ((axes[0], ('diff', 'lo', 'hi', 'verdict'), 'fixed box better', '(a) NeoSTAR − fixed box'),
              (axes[1], ('face_diff', 'face_lo', 'face_hi', 'face_verdict'), 'face box better', '(b) NeoSTAR − tracked face box'))
    lo_all = min(T.lo.min(), T.face_lo.min()) - 2
    hi_all = max(T.hi.max(), T.face_hi.max()) + 2
    for ax, (d, lo, hi, ver), worse, title in panels:
        col = {'NeoSTAR better': BLUE, worse: ORANGE, 'no clear difference': MUTED}
        ax.axvline(0, color=INK2, lw=0.9)
        for yi, (_, r) in zip(y, T.iterrows()):
            c = col[r[ver]]
            ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, solid_capstyle='round')
            mk = 's' if r.design == 'pre-registered' else ('D' if r.design == 'designed case' else 'o')
            ax.plot(r[d], yi, mk, color=c, ms=6, mec='white', mew=0.8, zorder=3)
        prev = None
        for yi, (_, r) in zip(y, T.iterrows()):
            if prev is not None and r.condition != prev:
                ax.axhline(yi + 0.5, color=GRID, lw=0.8)
            prev = r.condition
        ax.set_xlim(min(-40, lo_all), max(25, hi_all))
        ax.set_ylim(-0.7, n - 0.3)
        _style(ax)
        ax.set_title(title, fontsize=8.5, color=INK, loc='left')
        ax.set_xlabel('Δ absolute HR error (bpm)\n← NeoSTAR more accurate', fontsize=8, color=INK2)
    labels = ['Undisturbed (original videos)' if r.condition == 'Undisturbed' else f'{r.condition}: {r.level}' for _, r in T.iterrows()]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.6, color=INK)
    axes[1].tick_params(axis='y', length=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, lw=2, label='NeoSTAR better (CI < 0)'),
         Line2D([], [], color=ORANGE, lw=2, label='comparator better (CI > 0)'),
         Line2D([], [], color=MUTED, lw=2, label='no clear difference'),
         Line2D([], [], marker='s', ls='', color=INK2, label='pre-registered'),
         Line2D([], [], marker='o', ls='', color=INK2, label='fixed protocol / original videos'),
         Line2D([], [], marker='D', ls='', color=INK2, label='designed case, 1 video')]
    fig.subplots_adjust(left=0.29, right=0.985, top=1 - 0.35 / H, bottom=1.25 / H)
    fig.legend(handles=h, fontsize=7.2, frameon=False, loc='lower center', bbox_to_anchor=(0.62, 0.0), ncol=3)
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def timelines(R, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = R['src']
    aw = s['aug_windows']
    a = aw[(aw.category == 'motion_fast') & (aw.severity == 'xlarge')]
    a = a.rename(columns={'label_fixed': 'lab_fixed', 'label_D5': 'lab_D5', 'label_v1': 'lab_v1'})
    b = s['swmo_windows'][s['swmo_windows'].video_id == 'NEW_subject4_swmo_medium']
    c = s['occ_windows'].copy()
    runs = ROOT/'HR RR using module1 v2/runs/ubfc_occdemo'
    for cfg, keys in (('D5', (('fixed', 'B0'), ('D5', 'D5'))), ('D0', (('v1', 'D5'),))):
        w = pd.read_csv(runs/f'aug_eval_{cfg}'/'window_results_with_conditions.csv').set_index('window')
        for key, col in keys:
            c[f'lab_{key}'] = c.window.map(w[f'label_{col}'])
    panels = [
        (a, f"(a) Fast, large motion (up to 230 px), UBFC subject 1", row(R, 'Fast, large motion', 'extra-large'), None),
        (b, '(b) Swaddled to the neck + fast motion, UBFC subject 4 (pre-registered)', None, None),
        (c, '(c) Hand over the face centre, UBFC subject 1 (designed case)', row(R, 'Hand over face centre', 'hand present'),
         c),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    for ax, (d, title, r, occ) in zip(axes, panels):
        d = d.sort_values('window_center_sec')
        t = d.window_center_sec.to_numpy()
        if occ is not None:
            on = occ.set_index('window_center_sec').hand_visibility.reindex(t).to_numpy() >= 0.5
            for k in range(len(t)):
                if on[k]:
                    ax.axvspan(t[k] - 0.5, t[k] + 0.5, color=SHADE, lw=0)
        ax.plot(t, d.hr_ref, color=INK, lw=2, label='PPG reference')
        for key, colr, ls, lab in (('fixed', INK2, '--', 'Fixed box'), ('v1', ORANGE, ':', 'v1'), ('D5', BLUE, '-', 'NeoSTAR (D5)')):
            hr = d[f'hr_{key}'].to_numpy()
            ax.plot(t, hr, color=colr, ls=ls, lw=1.1, alpha=0.9, label=lab)
            val = (d[f'lab_{key}'].isin(['validated', 'validated_one_side']) & np.isfinite(hr)).to_numpy()
            ax.plot(t[val], hr[val], 'o', color=colr, ms=3.2)
            if key == 'fixed':
                miss = ~np.isfinite(hr)
                lo = 45
                ax.plot(t[miss], np.full(miss.sum(), lo + 1.5), '|', color=INK2, ms=7)
        mae = {k: np.nanmean(np.abs(d[f'hr_{k}'] - d.hr_ref)) for k in ('fixed', 'v1', 'D5')}
        if occ is not None:
            sel = on
            mae = {k: np.nanmean(np.abs(d[f'hr_{k}'] - d.hr_ref).to_numpy()[sel]) for k in ('fixed', 'v1', 'D5')}
        tag = ' (hand-present windows)' if occ is not None else ''
        ax.set_title(f"{title}\nMAE{tag}: fixed box {mae['fixed']:.1f}, v1 {mae['v1']:.1f}, NeoSTAR {mae['D5']:.1f} bpm",
                     fontsize=8.5, color=INK, loc='left')
        ax.set_ylim(45, 145)
        ax.set_ylabel('HR (bpm)', fontsize=8, color=INK2)
        _style(ax)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[-1].set_xlabel('time (s), centre of each 8-s window', fontsize=8, color=INK2)
    axes[0].legend(fontsize=7.2, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def make_figures(R, figs):
    figs.mkdir(parents=True, exist_ok=True)
    R['fig_forest'], R['fig_time'] = figs/'fig8_robustness_forest.png', figs/'fig9_robustness_timelines.png'
    forest(R, R['fig_forest'])  # legend below the axes
    timelines(R, R['fig_time'])


# ------------------------------------------------------------------ text
def ci(r):
    return f"{r['diff']:+.1f} bpm [{r.lo:+.1f}, {r.hi:+.1f}]"


def abstract_sentence(R):
    u, fx = row(R, 'Undisturbed', 'none'), row(R, 'Fast, large motion', 'extra-large')
    sm = row(R, 'Swaddling + fast motion', 'to neck')
    return ('In a reference-based robustness study on UBFC-rPPG videos with fingertip PPG and controlled, simulated '
            'perturbations, using the frozen neonatal settings, NeoSTAR matched a fixed box on undisturbed video (mean '
            f"absolute HR error {u.mae_D5:.1f} vs. {u.mae_fixed:.1f} bpm) and under lighting changes, and prevented its "
            f"failure under fast, large motion ({fx.mae_D5:.1f} vs. {fx.mae_fixed:.1f} bpm) and under swaddling combined "
            f"with motion ({sm.mae_D5:.1f} vs. {sm.mae_fixed:.1f} bpm; pre-registered, three subjects); with swaddling "
            'alone or generic hand-and-blanket occlusion the fixed box was as accurate or more accurate. A tracked face box '
            f'was never more accurate than NeoSTAR and was less accurate in {int((R["T"].face_verdict == "NeoSTAR better").sum())} '
            f'of {len(R["T"])} conditions, showing that the gain comes from region selection and stabilization, not from '
            'tracking alone.')


def contribution_bullet(R):
    return ('A pre-registered, reference-based robustness study on UBFC-rPPG with fingertip-PPG heart rate and controlled '
            'perturbations (fast motion, swaddling, swaddling with motion, occlusion and lighting), run with the frozen '
            'neonatal settings and compared with both a fixed box and a tracked face box, which identifies when a dynamic '
            'ROI is more accurate and when it is not.')


def setup_section(b, R):
    T = R['T']
    b.h2('4.6 Reference-Based Robustness Evaluation')
    b.para('The neonatal videos have no physiological reference, so heart-rate (HR) accuracy is evaluated separately on '
           'UBFC-rPPG [47], which provides fingertip photoplethysmography (PPG) for adult volunteers sitting still in front '
           'of a camera (640×480 px, about 30 FPS). Subjects 1, 3 and 4 were used. Because these recordings contain almost '
           'no disturbance, we add controlled, simulated perturbations to them; every perturbed video is written lossless '
           '(FFV1), checked bit-exact against its writer, and the original file is verified unchanged by its SHA-256 hash. '
           'Each generator logs its seeds and per-frame parameters.')
    b.bullets([
        '**Slow camera motion** (subject 1): smooth hand-held camera drift with peak translation ±8, ±20 and ±40 px '
        '(rotation up to 6°, zoom up to 10%).',
        '**Fast, large motion** (subject 1): Poisson-timed repositioning (every 3 s on average, each move 0.25-0.6 s with a '
        'minimum-jerk profile) with peak translation 50, 100, 160 and 230 px (about 0.35-1.6 face widths), rotation '
        '4-15° and zoom 3-12%.',
        '**Lighting change** (subject 4): exposure drift, spatial gradient, colour drift, aperiodic flicker with sparse dips '
        'and a moving shadow at three severities.',
        '**Hand + blanket occlusion** (subject 3): a procedural hand and a rising cloth covering 10%, 25% and 40% of the '
        'image.',
        '**Swaddling only** (subjects 3 and 4): a pale cloth whose top edge rises to the chest, to the chin or over the '
        'mouth, with a waving edge and slow drift.',
        '**Swaddling + fast motion** (subjects 1, 3 and 4): cloth to the neck or to the mouth, then the whole frame '
        'moved by the fast-motion generator (160 px), so subject and cloth move together.',
        '**Hand over the face centre** (subject 1): a skin-coloured hand entering from the side and covering the nose and '
        'mid-cheeks in three intervals.',
    ])
    b.para('**Methods and HR chain.** The fixed box (B0, the central half of the first-frame person box), v1 (D0 = Full) '
           'and NeoSTAR (D5) are compared with identical pixel rules and an identical HR chain: adaptive YCrCb skin pixels, '
           'POS [3], 8 s windows with a 1 s step, HR at the in-band spectral peak. The only change from the neonatal '
           'evaluation is the HR band, 45-180 bpm for adults instead of 90-270 bpm. NeoSTAR uses the stabilizer settings '
           'tuned on the 21 neonatal training videos; nothing was tuned on UBFC or on any perturbed video. A window is '
           '*validated* when it passes the pulse-present test (Section 4.4) and split-half agreement within 6 bpm.')
    b.para('**Tracked face box.** To separate the benefit of following the subject from that of NeoSTAR\'s region '
           'selection and stabilization, a fourth ROI is added that represents common rPPG practice: the bounding '
           'rectangle of Module 1\'s FACE region, computed on every sampled frame from the same pose estimate and held '
           'between samples. It follows the face as NeoSTAR does but uses one rectangle, with no region selection, no '
           'segmentation-mask constraint and no temporal stabilization, and it passes through the same skin filter and HR '
           'chain. The same videos and windows are used; the fixed-box results of this run were checked against the '
           'earlier run and are identical.')
    b.para('**Reference and statistics.** The reference HR is the UBFC PPG HR at the nearest window centre within 1 s. '
           'The error is |HR − reference| on every window in which a method gives an HR; windows without an HR are '
           'counted separately, because a method can look accurate by reporting less. Paired differences (NeoSTAR − fixed '
           'box) use windows where both methods give an HR, with 95% confidence intervals from a moving-block bootstrap '
           '[49] (blocks of 8 windows, resampled within video, 5000 resamples, seed 42). A difference is called a win only '
           'when its interval excludes zero. To check the alignment of camera and reference windows, the reference was '
           'shifted by -6 to +6 s: the error of the fixed box and NeoSTAR was lowest within ±1 s of zero lag for every '
           'subject and condition group (one window step), so no offset was applied.')
    b.para('**Pre-registration.** The swaddling-only and the swaddling + motion tests were specified in writing (videos, '
           'seeds, metrics and claim rules) before any method was run on them; their claims are applied automatically. The '
           'motion, lighting and hand + blanket tests follow one fixed protocol on one subject each. The hand-over-the-face-'
           'centre video was designed after the hand + blanket result to show the case in which a fixed box should fail; it '
           'is an illustration, not a test.')


def results_section(b, R, fig_no=8, table_no=11):
    T = R['T']
    u = row(R, 'Undisturbed', 'none')
    fx, fl = row(R, 'Fast, large motion', 'extra-large'), row(R, 'Fast, large motion', 'large')
    smm, sml = row(R, 'Swaddling + fast motion', 'to neck'), row(R, 'Swaddling + fast motion', 'to mouth')
    oc = row(R, 'Hand over face centre', 'hand present')
    occ = T[T.condition == 'Hand + blanket occlusion']
    sw = T[T.condition == 'Swaddling only']
    li, mo = T[T.condition == 'Lighting change'], T[T.condition == 'Slow camera motion']
    claims = R['claims'].set_index('claim').result
    b.h2('5.9 Heart-Rate Accuracy Under Controlled Perturbations')
    b.para(f'Table {table_no} and Figure {fig_no} summarize the reference-based study. **Undisturbed video.** On the '
           f'original recordings of subjects 1, 3 and 4 the three methods were equally accurate (mean absolute error: fixed '
           f'box {u.mae_fixed:.2f}, v1 {u.mae_v1:.2f}, NeoSTAR {u.mae_D5:.2f} bpm; NeoSTAR − fixed box {ci(u)}). '
           f'**Slow camera motion and lighting.** Neither changed the ranking: all methods stayed at '
           f'{min(mo.mae_fixed.min(), mo.mae_D5.min(), li.mae_fixed.min(), li.mae_D5.min()):.1f}-'
           f'{max(mo.mae_fixed.max(), mo.mae_D5.max(), li.mae_fixed.max(), li.mae_D5.max()):.1f} bpm and no difference '
           'was significant. NeoSTAR is therefore robust to these conditions, but not better than a fixed box under them.')
    b.para(f'**Fast, large motion.** The fixed box failed as the motion grew: its error rose from {row(R, "Fast, large motion", "small").mae_fixed:.1f} '
           f'to {fx.mae_fixed:.1f} bpm and it gave no HR in {fl.nohr_fixed} and {fx.nohr_fixed} of {fx.windows} windows at '
           f'the two largest levels, because the face left the box. NeoSTAR, which tracks the face regions, stayed at '
           f"{T[T.condition == 'Fast, large motion'].mae_D5.min():.1f}-{T[T.condition == 'Fast, large motion'].mae_D5.max():.1f} bpm with an HR in every window "
           f'(NeoSTAR − fixed box at the largest level {ci(fx)}; at smaller levels the difference favoured NeoSTAR but its '
           'interval included zero). **Swaddling + fast motion** (pre-registered, primary claim '
           f"{claims.get('M1 (primary)', 'n/a').lower()}). With the cloth to the neck the fixed box reached "
           f'{smm.mae_fixed:.1f} bpm and gave no HR in {smm.nohr_fixed} of {smm.windows} windows, whereas NeoSTAR reached '
           f'{smm.mae_D5:.1f} bpm with no missing windows ({ci(smm)}; better in all three subjects). With the cloth over the '
           f'mouth the result held ({sml.mae_D5:.1f} vs. {sml.mae_fixed:.1f} bpm, {ci(sml)}), but NeoSTAR gave no HR on '
           f'subject 1 ({sml.nohr_D5} windows), where the segmentation lost the face under the cloth.')
    b.para(f'**Swaddling only** (pre-registered). Without motion the cloth stays below a still face and the fixed box '
           f'keeps face skin; the fixed box was as accurate or slightly more accurate than NeoSTAR at every level '
           f'(differences {sw["diff"].min():+.1f} to {sw["diff"].max():+.1f} bpm; fixed box better at '
           f"{', '.join(sw[sw.verdict == 'fixed box better'].level)}). **Occlusion.** The result depended on where the "
           f'occluder was. Under the generic hand + blanket occlusion the fixed box was more accurate (differences '
           f'{occ["diff"].min():+.1f} to {occ["diff"].max():+.1f} bpm; fixed box better at '
           f"{', '.join(occ[occ.verdict == 'fixed box better'].level)}), because NeoSTAR's selected regions were partly "
           'covered while the larger fixed box kept enough clean skin. When a skin-coloured hand covered the centre of the '
           f'face, which lies inside the fixed box and passes its skin filter, the fixed box failed ({oc.mae_fixed:.1f} '
           f'bpm while the hand was present) and NeoSTAR stayed accurate ({oc.mae_D5:.1f} bpm; {ci(oc)}). This is one '
           'designed video and illustrates a mechanism; it is not a test.')
    b.para(f'**Stabilization versus dynamic ROI.** v1 and NeoSTAR performed alike in every condition (e.g. '
           f'{fx.mae_v1:.1f} vs. {fx.mae_D5:.1f} bpm under the largest motion, {smm.mae_v1:.1f} vs. {smm.mae_D5:.1f} bpm '
           'under swaddling to the neck with motion). The robustness to motion therefore comes from tracking anatomical '
           'regions, which both versions do; in this study the stabilization contributed stability (Section 5.2), not '
           'HR accuracy.')
    face_paragraphs(b, R, fig_no, table_no)
    import results_v2 as RV
    RV.figure(b, R['fig_forest'], 6.5,
              f'Figure {fig_no}. Heart-rate error of NeoSTAR (D5) minus (a) the fixed box and (b) the tracked face box '
              'under each condition (mean and 95% moving-block bootstrap CI over paired windows; UBFC-rPPG with '
              'fingertip-PPG reference). Blue: NeoSTAR more accurate; orange: the comparator more accurate; grey: interval '
              'includes zero. Squares: pre-registered tests; circles: fixed protocol on one subject; diamond: designed case.')
    RV.figure(b, R['fig_time'], 6.3,
              f'Figure {fig_no + 1}. HR over time for three conditions. Black: PPG reference; grey dashed: fixed box; '
              'orange dotted: v1; blue: NeoSTAR (D5); filled markers: validated windows; ticks at the bottom: windows in '
              'which the fixed box gave no HR; shading in (c): hand over the face centre.')
    b.caption(f'Table {table_no}. HR accuracy under controlled perturbations (UBFC-rPPG, fingertip-PPG reference). MAE in '
              'bpm over windows with an HR; in brackets: windows without an HR. Δ: NeoSTAR − comparator on paired windows, '
              'mean and 95% CI; * interval excludes zero.')
    rows = []
    for _, r in T.iterrows():
        f = lambda k: ('–' if not np.isfinite(r[f'mae_{k}']) else f"{r[f'mae_{k}']:.1f}") + (f" ({int(r[f'nohr_{k}'])})" if r[f'nohr_{k}'] else '')  # noqa: E731
        g = lambda d, lo, hi: f"{r[d]:+.1f} [{r[lo]:+.1f}, {r[hi]:+.1f}]" + ('*' if r[lo] > 0 or r[hi] < 0 else '')  # noqa: E731
        rows.append([r.condition, r.level, r.subjects, r.windows, f('fixed'), f('v1'), f('FACE'), f('D5'),
                     g('diff', 'lo', 'hi'), g('face_diff', 'face_lo', 'face_hi'), r.design])
    b.table(['Condition', 'Level', 'Subj.', 'Win.', 'Fixed box', 'v1', 'Face box', 'NeoSTAR', 'Δ vs fixed box',
             'Δ vs face box', 'Design'],
            rows, size=6.5, widths=[1.05, 0.6, 0.38, 0.33, 0.55, 0.45, 0.5, 0.55, 0.9, 0.9, 0.65])
    F = R['F34']
    b.caption(f'Table {table_no + 1}. Geometry of the tracked face box on the 34 neonatal videos, measured as in Section 5.2 '
              '(mean over videos). Coverage: share of sampled frames with an ROI; temporal IoU: overlap of the ROI bounding '
              'boxes in consecutive sampled frames; centroid jitter: frame-to-frame centroid displacement divided by the '
              'infant-box diagonal.')
    b.table(['ROI', 'Coverage (%)', 'Temporal IoU', 'Centroid jitter'],
            [[r.method, f"{r.coverage:.1f}", f"{r.temporal_iou:.3f}", f"{r.centroid_jitter:.3f}"] for _, r in F.iterrows()],
            size=8, widths=[1.8, 1.1, 1.1, 1.1])


def face_paragraphs(b, R, fig_no, table_no):
    T, F = R['T'], R['F34'].set_index('method')
    better = T[T.face_verdict == 'NeoSTAR better']
    worse = T[T.face_verdict == 'face box better']
    u = row(R, 'Undisturbed', 'none')
    sl = row(R, 'Slow camera motion', 'large')
    fx = row(R, 'Fast, large motion', 'extra-large')
    smm = row(R, 'Swaddling + fast motion', 'to neck')
    oc = row(R, 'Hand over face centre', 'hand present')
    fm = T[T.condition == 'Fast, large motion']
    occ = T[T.condition == 'Hand + blanket occlusion']
    assert worse.empty, 'text below assumes the face box never beats NeoSTAR'
    b.para(f'**Tracked face box.** Following the face is not enough by itself (Figure {fig_no}b, Table {table_no}). '
           f'NeoSTAR was more accurate than the tracked face box in {len(better)} of {len(T)} conditions and the face box '
           'was never more accurate. The face box already lost accuracy on undisturbed video '
           f'({u.mae_FACE:.1f} vs. {u.mae_D5:.1f} bpm; NeoSTAR − face box {u.face_diff:+.1f} [{u.face_lo:+.1f}, '
           f'{u.face_hi:+.1f}]) and under slow camera motion (up to {sl.mae_FACE:.1f} bpm, where NeoSTAR and the fixed box '
           f'stayed below {max(sl.mae_D5, sl.mae_fixed) + 0.1:.1f} bpm), and under fast motion it reached '
           f'{fm.mae_FACE.min():.1f}-{fm.mae_FACE.max():.1f} bpm against {fm.mae_D5.min():.1f}-{fm.mae_D5.max():.1f} bpm for '
           f'NeoSTAR (better at all four levels). With a hand over the face centre it reached {oc.mae_FACE:.1f} bpm against '
           f'{oc.mae_D5:.1f} bpm. Against the fixed box, tracking helped only when the subject moved far: the face box was '
           f'more accurate under the largest motion ({fx.ff_diff:+.1f} bpm [{fx.ff_lo:+.1f}, {fx.ff_hi:+.1f}]) and under '
           f'swaddling with motion ({smm.ff_diff:+.1f} bpm [{smm.ff_lo:+.1f}, {smm.ff_hi:+.1f}] to the neck), and less '
           f'accurate on still and slowly moving video. Under swaddling with motion, NeoSTAR and the face box did not '
           f'differ ({smm.mae_D5:.1f} vs. {smm.mae_FACE:.1f} bpm), and under generic hand + blanket occlusion the fixed box '
           f'remained the most accurate ROI (face box {occ.mae_FACE.min():.1f}-{occ.mae_FACE.max():.1f} bpm). A rectangle around '
           'the face includes eyes, eyebrows, hair and background at its edges, and its position follows the pose estimate '
           'from frame to frame; NeoSTAR\'s gain over it therefore comes from region selection, the mask constraint and '
           'stabilization, not from tracking alone.')
    fc, d5, f1 = F.loc['Tracked face box'], F.loc['NeoSTAR (D5)'], F.loc['v1 (Full)']
    b.para(f'On the 34 neonatal videos (Table {table_no + 1}) the tracked face box was available in '
           f'{fc.coverage:.1f}% of sampled frames, but its position was much less stable than NeoSTAR\'s: centroid jitter '
           f'{fc.centroid_jitter:.3f} vs. {d5.centroid_jitter:.3f} (about {fc.centroid_jitter / d5.centroid_jitter:.0f} times '
           f'larger) and temporal IoU {fc.temporal_iou:.3f} vs. {d5.temporal_iou:.3f}, with more jitter even than the '
           f'unstabilized v1 ({f1.centroid_jitter:.3f}). Infant faces are small and often turned or partly covered, so the '
           'pose-derived face rectangle moves with every keypoint error; this is the instability NeoSTAR is designed to remove.')


def discussion_paragraph(R):
    u, fx = row(R, 'Undisturbed', 'none'), row(R, 'Fast, large motion', 'extra-large')
    smm = row(R, 'Swaddling + fast motion', 'to neck')
    return ('**When does a dynamic ROI pay off?** A large fixed box averages many pixels and never switches, which helps '
            'a periodicity proxy, and on still video it is as accurate as NeoSTAR against a PPG reference '
            f'({u.mae_fixed:.1f} vs. {u.mae_D5:.1f} bpm). Its weakness is that it does not follow the subject: under fast, '
            f'large motion and under swaddling with motion it lost the face and its error rose to {fx.mae_fixed:.0f}-'
            f'{smm.mae_fixed:.0f} bpm with many windows without HR, while NeoSTAR stayed near 2-3 bpm. Conversely, when '
            'the subject is still and partly covered (swaddling alone, a hand and blanket at the image edge), the larger '
            'box keeps more clean skin and is at least as accurate. A dynamic ROI is therefore worth its cost when the '
            'subject or camera moves, which is common in neonatal care, and adds little for a still, uncovered subject. '
            'The robustness comes from following anatomical regions, which v1 already did; stabilization makes the '
            'representation stable and inspectable but did not change HR accuracy. Tracking alone is not the answer '
            'either: a tracked face box, the usual rPPG input, was never more accurate than NeoSTAR and was less accurate '
            'on still video, under slow and fast motion and with a hand over the face, because a rectangle around a moving '
            'face mixes skin with non-skin and moves with the pose estimate. On the neonatal web videos themselves '
            'the pulse ranking depended on the split (the fixed box led on the validation videos and not on the test '
            'videos, Section 5.6), so no pulse-quality advantage of either representation can be claimed from those data.')


LIMITATION_REPLACEMENT = [
    'HR accuracy was measured only on adult UBFC-rPPG recordings (three subjects) with simulated perturbations; the '
    'motion is a whole-frame warp and the cloth, hand and blanket are composited layers that carry no pulse and do not '
    'deform with the body. Most conditions were tested on one subject, and only the swaddling tests were pre-registered. '
    'Real infant recordings with reference heart rate (and datasets with real motion such as PURE [48] or MMPD [16]) '
    'remain necessary before clinical claims; respiratory-rate accuracy was not evaluated.',
    'NeoSTAR did not improve HR over a fixed box under lighting changes, swaddling alone or generic hand-and-blanket '
    'occlusion, and it lost the face when the cloth covered the mouth of one subject; its benefit is limited to conditions '
    'in which the subject moves. The hand-over-the-face-centre case is a single designed video.',
]


def conclusion_sentence(R):
    fx, smm = row(R, 'Fast, large motion', 'extra-large'), row(R, 'Swaddling + fast motion', 'to neck')
    return ('Against a PPG reference on UBFC-rPPG with controlled perturbations, the dynamic representation matched a fixed '
            f'box on still video and prevented its failure when the subject moved ({fx.mae_D5:.1f} vs. {fx.mae_fixed:.1f} '
            f'bpm under fast motion; {smm.mae_D5:.1f} vs. {smm.mae_fixed:.1f} bpm under swaddling with motion), whereas '
            'swaddling alone and generic occlusion did not favour it. The representation, the validity tests, the '
            'perturbation generators and the pre-registrations are released so that these benefits can be tested on '
            'neonatal recordings with a reference.')


NEW_REFERENCE = ('[49] H. R. Künsch, "The jackknife and the bootstrap for general stationary observations," Annals of '
                 'Statistics, vol. 17, no. 3, pp. 1217-1241, 1989, doi: 10.1214/aos/1176347265.')
