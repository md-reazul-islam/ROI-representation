"""Robustness study text for the SUBMISSION build (build_submission.py).

Same data and numbers as robustness.py (imported read-only), with the editorial changes made for submission:
  * the UBFC study is framed as a controlled pilot; single-subject conditions are labelled illustrative, and the
    status of each pre-registration is stated exactly;
  * the hand-over-the-face-centre video (a designed case, one video) is moved to the Supplementary Material, with one
    sentence left in the main text;
  * per-subject results of the pre-registered tests and the time-lag check are computed here for the supplement;
  * limitations are expanded and a future-work paragraph is added.
The updated manuscript (build_manuscript_v2_updated.py) keeps using robustness.py unchanged.
"""
import numpy as np
import pandas as pd

import robustness as RB
from robustness import ROOT, row, ci, INK, INK2, GRID, MUTED, BLUE, ORANGE, SHADE, _style  # noqa: F401

DEMO = 'Hand over face centre'
DESIGN_LABEL = {'original videos': 'original', 'fixed protocol': 'illustrative', 'pre-registered': None}
NEW_REFERENCE = RB.NEW_REFERENCE
contribution_bullet = RB.contribution_bullet


def load():
    """robustness.load() split into the main-text conditions and the designed case (supplement)."""
    R = RB.load()
    T = R['T']
    R['T_all'] = T
    R['demo'] = T[T.condition == DEMO].iloc[0]
    T = T[T.condition != DEMO].copy().reset_index(drop=True)
    # exact status of every condition (see setup_section)
    T['design'] = [('pre-reg. claim' if c == 'Swaddling + fast motion' else 'pre-reg. design') if d == 'pre-registered'
                   else DESIGN_LABEL[d] for c, d in zip(T.condition, T.design)]
    R['T'] = T
    R['per_subject'] = per_subject()
    R['lag'] = lag_check()
    return R


# ------------------------------------------------------------------ supplement numbers
def _windows():
    return pd.read_csv(ROOT/'face_track_baseline/report/ubfc_windows.csv')


def per_subject():
    """Per-subject MAE of the pre-registered conditions from the face-box window table (same windows as Table 11),
    checked against the pre-registered swaddle+motion report."""
    w = _windows()
    rows = []
    for cond in ('Swaddling only', 'Swaddling + fast motion'):
        g = w[w.condition == cond]
        for (lev, subj), s in g.groupby(['level', 'subject'], sort=False):
            r = dict(condition=cond, level=lev, subject=subj.replace('subject', ''), windows=len(s))
            for k, col in (('fixed', 'hr_fixed'), ('FACE', 'hr_FACE'), ('D5', 'hr_D5')):
                e = (s[col] - s.hr_ref).abs()
                r[f'mae_{k}'], r[f'nohr_{k}'] = float(e.mean()), int(s[col].isna().sum())
            rows.append(r)
    P = pd.DataFrame(rows)
    ref = pd.read_csv(ROOT/'module1_v2_d9/swaddle_motion/report/per_subject.csv')
    lv = {'to neck': 'medium', 'to mouth': 'large'}
    for _, r in P[P.condition == 'Swaddling + fast motion'].iterrows():
        for k in ('fixed', 'D5'):
            q = ref[(ref.level == lv[r.level]) & (ref.subject == f'subject{r.subject}') & (ref.method == k)].iloc[0]
            assert int(q.no_hr) == r[f'nohr_{k}'], (r.level, r.subject, k)
            assert (np.isnan(q.mae) and np.isnan(r[f'mae_{k}'])) or abs(q.mae - r[f'mae_{k}']) < 0.01, (r.level, r.subject, k)
    return P


def lag_check():
    """Error of the fixed box and NeoSTAR (pooled) when the reference is shifted by -6..+6 s, per condition group."""
    import sys
    sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
    from refmatch import nearest_reference
    ref = ROOT/'face_track_baseline/reference_hr_ppg.csv'
    w = _windows()
    w = w[w.condition != DEMO]
    lags = np.arange(-6, 6.5, 1.0)
    rows = []
    for cond, g in w.groupby('condition', sort=False):
        err = []
        for L in lags:
            r = nearest_reference(g.video_id, g.window_center_sec + L, ref)
            e = np.concatenate([np.abs(g[k] - r).dropna().to_numpy() for k in ('hr_fixed', 'hr_D5')])
            err.append(e.mean())
        err = np.array(err)
        rows.append(dict(condition=cond, mae_lag0=err[lags == 0][0], best_lag=lags[err.argmin()], mae_best=err.min()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def forest(R, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    T = R['T']
    n = len(T)
    H = 0.27 * n + 2.2
    fig, axes = plt.subplots(1, 2, figsize=(7.6, H), sharey=True, gridspec_kw=dict(wspace=0.08))
    y = np.arange(n)[::-1]
    panels = ((axes[0], ('diff', 'lo', 'hi', 'verdict'), 'fixed box better', '(a) NeoSTAR − fixed box'),
              (axes[1], ('face_diff', 'face_lo', 'face_hi', 'face_verdict'), 'face box better',
               '(b) NeoSTAR − tracked face box'))
    lo_all, hi_all = min(T.lo.min(), T.face_lo.min()) - 2, max(T.hi.max(), T.face_hi.max()) + 2
    for ax, (d, lo, hi, ver), worse, title in panels:
        col = {'NeoSTAR better': BLUE, worse: ORANGE, 'no clear difference': MUTED}
        ax.axvline(0, color=INK2, lw=0.9)
        for yi, (_, r) in zip(y, T.iterrows()):
            c = col[r[ver]]
            ax.plot([r[lo], r[hi]], [yi, yi], color=c, lw=2, solid_capstyle='round')
            ax.plot(r[d], yi, 's' if r.design.startswith('pre-reg') else 'o', color=c, ms=6, mec='white', mew=0.8, zorder=3)
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
    labels = ['Undisturbed (original videos)' if r.condition == 'Undisturbed' else f'{r.condition}: {r.level}'
              for _, r in T.iterrows()]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.6, color=INK)
    axes[1].tick_params(axis='y', length=0)
    h = [Line2D([], [], color=BLUE, lw=2, label='NeoSTAR better (CI < 0)'),
         Line2D([], [], color=ORANGE, lw=2, label='comparator better (CI > 0)'),
         Line2D([], [], color=MUTED, lw=2, label='no clear difference'),
         Line2D([], [], marker='s', ls='', color=INK2, label='pre-registered (2-3 subjects)'),
         Line2D([], [], marker='o', ls='', color=INK2, label='original / illustrative (1 subject)')]
    fig.subplots_adjust(left=0.29, right=0.985, top=1 - 0.35 / H, bottom=1.25 / H)
    fig.legend(handles=h, fontsize=7.0, frameon=False, loc='lower center', bbox_to_anchor=(0.6, 0.0), ncol=3,
               columnspacing=1.2, handlelength=1.8)
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def _demo_frame(R):
    s = R['src']
    c = s['occ_windows'].copy()
    runs = ROOT/'HR RR using module1 v2/runs/ubfc_occdemo'
    for cfg, keys in (('D5', (('fixed', 'B0'), ('D5', 'D5'))), ('D0', (('v1', 'D5'),))):
        w = pd.read_csv(runs/f'aug_eval_{cfg}'/'window_results_with_conditions.csv').set_index('window')
        for key, col in keys:
            c[f'lab_{key}'] = c.window.map(w[f'label_{col}'])
    return c


def _timeline_panels(axes, panels):
    for ax, (d, title, occ) in zip(axes, panels):
        d = d.sort_values('window_center_sec')
        t = d.window_center_sec.to_numpy()
        on = None
        if occ is not None:
            on = occ.set_index('window_center_sec').hand_visibility.reindex(t).to_numpy() >= 0.5
            for k in range(len(t)):
                if on[k]:
                    ax.axvspan(t[k] - 0.5, t[k] + 0.5, color=SHADE, lw=0)
        ax.plot(t, d.hr_ref, color=INK, lw=2, label='PPG reference')
        for key, colr, ls, lab in (('fixed', INK2, '--', 'Fixed box'), ('v1', ORANGE, ':', 'PF-ROI'),
                                   ('D5', BLUE, '-', 'NeoSTAR (D5)')):
            hr = d[f'hr_{key}'].to_numpy()
            ax.plot(t, hr, color=colr, ls=ls, lw=1.1, alpha=0.9, label=lab)
            val = (d[f'lab_{key}'].isin(['validated', 'validated_one_side']) & np.isfinite(hr)).to_numpy()
            ax.plot(t[val], hr[val], 'o', color=colr, ms=3.2)
            if key == 'fixed':
                miss = ~np.isfinite(hr)
                ax.plot(t[miss], np.full(miss.sum(), 46.5), '|', color=INK2, ms=7)
        err = {k: np.abs(d[f'hr_{k}'] - d.hr_ref).to_numpy() for k in ('fixed', 'v1', 'D5')}
        sel = on if on is not None else np.ones(len(t), bool)
        mae = {k: np.nanmean(v[sel]) for k, v in err.items()}
        tag = ' (hand-present windows)' if on is not None else ''
        ax.set_title(f"{title}\nMAE{tag}: fixed box {mae['fixed']:.1f}, PF-ROI {mae['v1']:.1f}, NeoSTAR {mae['D5']:.1f} bpm",
                     fontsize=8.5, color=INK, loc='left')
        ax.set_ylim(45, 145)
        ax.set_ylabel('HR (bpm)', fontsize=8, color=INK2)
        _style(ax)
        ax.grid(axis='y', color=GRID, lw=0.6)
    axes[-1].set_xlabel('time (s), centre of each 8-s window', fontsize=8, color=INK2)
    axes[0].legend(fontsize=7.2, frameon=False, ncol=4, loc='upper left', bbox_to_anchor=(0, 1.0))


def timelines(R, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = R['src']
    a = s['aug_windows']
    a = a[(a.category == 'motion_fast') & (a.severity == 'xlarge')]
    a = a.rename(columns={'label_fixed': 'lab_fixed', 'label_D5': 'lab_D5', 'label_v1': 'lab_v1'})
    b = s['swmo_windows'][s['swmo_windows'].video_id == 'NEW_subject4_swmo_medium']
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.1))
    _timeline_panels(axes, [(a, '(a) Fast, large motion (up to 230 px), UBFC subject 1 (illustrative)', None),
                            (b, '(b) Swaddled to the neck + fast motion, UBFC subject 4 (pre-registered)', None)])
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def demo_timeline(R, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = _demo_frame(R)
    fig, axes = plt.subplots(1, 1, figsize=(7.2, 2.7))
    _timeline_panels([axes], [(c, 'Hand over the face centre, UBFC subject 1 (designed case, one video)', c)])
    fig.tight_layout()
    fig.savefig(path, dpi=300, facecolor='white')
    plt.close(fig)


def make_figures(R, figs):
    figs.mkdir(parents=True, exist_ok=True)
    R['fig_forest'], R['fig_time'] = figs/'fig8_robustness_forest.png', figs/'fig9_robustness_timelines.png'
    R['fig_demo'] = figs/'figS1_hand_over_face_demo.png'
    forest(R, R['fig_forest'])
    timelines(R, R['fig_time'])
    demo_timeline(R, R['fig_demo'])


# ------------------------------------------------------------------ main text
def _subjects_better(R, cond, level):
    P = R['per_subject']
    p = P[(P.condition == cond) & (P.level == level)]
    # the pre-registered rule: NeoSTAR's MAE lower than the fixed box's (a subject without any NeoSTAR HR does not count)
    better = sum((not np.isnan(r.mae_D5)) and (np.isnan(r.mae_fixed) or r.mae_D5 < r.mae_fixed) for _, r in p.iterrows())
    claim = {'to neck': 'M1 (primary)', 'to mouth': 'M2'}.get(level)
    if cond == 'Swaddling + fast motion' and claim:
        detail = R['claims'].set_index('claim').detail[claim]
        assert f'subjects {better}/{len(p)}' in detail, (detail, better)
    return better, len(p)


def setup_section(b, R):
    b.h2('4.6 Reference-Based Robustness Evaluation')
    b.para('The neonatal videos have no physiological reference, so heart-rate (HR) accuracy is evaluated separately on '
           'UBFC-rPPG [47], which provides fingertip photoplethysmography (PPG) for adult volunteers sitting still in front '
           'of a camera (640×480 px, about 30 FPS). This is a controlled pilot study with three subjects (1, 3 and 4), '
           'designed to show under which conditions a dynamic ROI changes HR accuracy, not to estimate accuracy for a '
           'population. Because these recordings contain almost no disturbance, we add controlled, simulated perturbations '
           'to them; every perturbed video is written lossless (FFV1), checked bit-exact against its writer, and the '
           'original file is verified unchanged by its SHA-256 hash. Each generator logs its seeds and per-frame parameters.')
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
    ])
    b.para('**Methods and HR chain.** The fixed box (B0, the central half of the first-frame person box), v1 (D0 = Full) '
           'and NeoSTAR (D5) are compared with identical pixel rules and an identical HR chain: adaptive YCrCb skin pixels, '
           'POS [3], 8 s windows with a 1 s step, HR at the in-band spectral peak. The only change from the neonatal '
           'evaluation is the HR band, 45-180 bpm for adults instead of 90-270 bpm. NeoSTAR uses the stabilizer settings '
           'tuned on the 21 neonatal training videos; nothing was tuned on UBFC or on any perturbed video. A window is '
           '*validated* when it passes the pulse-present test (Section 4.4) and split-half agreement within 6 bpm.')
    b.para('**Tracked face box.** To separate the benefit of following the subject from that of NeoSTAR\'s region '
           'selection and stabilization, a fourth ROI is added that represents common rPPG practice: the bounding '
           'rectangle of the pipeline\'s FACE region, computed on every sampled frame from the same pose estimate and held '
           'between samples. It follows the face as NeoSTAR does but uses one rectangle, with no region selection, no '
           'segmentation-mask constraint and no temporal stabilization, and it passes through the same skin filter and HR '
           'chain. The same videos and windows are used; the fixed-box results of this run were checked against the '
           'earlier run and are identical.')
    b.para('**Reference and statistics.** The reference HR is the UBFC PPG HR at the nearest window centre within 1 s. '
           'The error is |HR − reference| on every window in which a method gives an HR; windows without an HR are '
           'counted separately, because a method can look accurate by reporting less. Paired differences (NeoSTAR − '
           'comparator) use windows where both methods give an HR, with 95% confidence intervals from a moving-block '
           'bootstrap [49] (blocks of 8 windows, resampled within video, 5000 resamples, seed 42). A difference is called a '
           'win only when its interval excludes zero. To check the alignment of camera and reference windows, the '
           'reference was shifted by -6 to +6 s: for every condition group the pooled error of the fixed box and NeoSTAR '
           'was lowest within ±1 s of zero lag (one window step; Supplementary Table S3), so no offset was applied.')
    b.para('**Study status.** Only the swaddling + fast motion test carries a pre-registered claim about NeoSTAR versus '
           'the fixed box: its videos, seeds, metrics and claim rules (pooled confidence interval below zero and a lower '
           'error in at least two of the three subjects) were fixed in writing before any method ran on the videos. The '
           'swaddling-only videos, their seeds and the analysis were also fixed in advance, as part of a pre-registration '
           'whose claims concerned a mask-repair variant that is not part of NeoSTAR; the NeoSTAR − fixed box contrast on '
           'them is reported with that fixed analysis. The motion, lighting and hand + blanket conditions follow one '
           'fixed protocol on one subject each and are illustrative. The pre-registrations, their deviations (none) and '
           'per-subject results are summarized in the Supplementary Material (Sections S2-S3).')


def results_section(b, R, fig_no=8, table_no=11):
    T = R['T']
    u = row(R, 'Undisturbed', 'none')
    fx, fl = row(R, 'Fast, large motion', 'extra-large'), row(R, 'Fast, large motion', 'large')
    smm, sml = row(R, 'Swaddling + fast motion', 'to neck'), row(R, 'Swaddling + fast motion', 'to mouth')
    oc = R['demo']
    occ = T[T.condition == 'Hand + blanket occlusion']
    sw = T[T.condition == 'Swaddling only']
    li, mo = T[T.condition == 'Lighting change'], T[T.condition == 'Slow camera motion']
    claims = R['claims'].set_index('claim').result
    bn, nn = _subjects_better(R, 'Swaddling + fast motion', 'to neck')
    bm, nm = _subjects_better(R, 'Swaddling + fast motion', 'to mouth')
    P = R['per_subject']
    swp = P[P.condition == 'Swaddling only']
    sw_fixed_lower = int((swp.mae_fixed <= swp.mae_D5).sum())
    b.h2('5.9 Heart-Rate Accuracy Under Controlled Perturbations')
    b.para(f'Table {table_no} and Figure {fig_no} summarize the pilot. **Undisturbed video.** On the '
           f'original recordings of subjects 1, 3 and 4 the three methods were equally accurate (mean absolute error: fixed '
           f'box {u.mae_fixed:.2f}, v1 {u.mae_v1:.2f}, NeoSTAR {u.mae_D5:.2f} bpm; NeoSTAR − fixed box {ci(u)}). '
           f'**Slow camera motion and lighting** (illustrative, one subject each). Neither changed the ranking: all methods '
           f'stayed at {min(mo.mae_fixed.min(), mo.mae_D5.min(), li.mae_fixed.min(), li.mae_D5.min()):.1f}-'
           f'{max(mo.mae_fixed.max(), mo.mae_D5.max(), li.mae_fixed.max(), li.mae_D5.max()):.1f} bpm and no difference '
           'was significant.')
    b.para(f'**Fast, large motion** (illustrative, subject 1). The fixed box failed as the motion grew: its error rose from '
           f'{row(R, "Fast, large motion", "small").mae_fixed:.1f} to {fx.mae_fixed:.1f} bpm and it gave no HR in '
           f'{fl.nohr_fixed} and {fx.nohr_fixed} of {fx.windows} windows at the two largest levels, because the face left '
           f'the box. NeoSTAR, which tracks the face regions, stayed at '
           f"{T[T.condition == 'Fast, large motion'].mae_D5.min():.1f}-{T[T.condition == 'Fast, large motion'].mae_D5.max():.1f} "
           f'bpm with an HR in every window (NeoSTAR − fixed box at the largest level {ci(fx)}; at smaller levels the '
           'difference favoured NeoSTAR but its interval included zero). **Swaddling + fast motion** (pre-registered, '
           f"three subjects; primary claim {claims.get('M1 (primary)', 'n/a').lower()}). With the cloth to the neck the fixed "
           f'box reached {smm.mae_fixed:.1f} bpm and gave no HR in {smm.nohr_fixed} of {smm.windows} windows, whereas '
           f'NeoSTAR reached {smm.mae_D5:.1f} bpm with no missing windows ({ci(smm)}; better in {bn} of {nn} subjects). '
           f'With the cloth over the mouth the pooled result held ({sml.mae_D5:.1f} vs. {sml.mae_fixed:.1f} bpm, {ci(sml)}; '
           f"claim {claims.get('M2', 'n/a').lower()}, better in {bm} of {nm} subjects), but NeoSTAR gave no HR on subject 1 "
           f'({sml.nohr_D5} windows), where the segmentation lost the face under the cloth.')
    b.para(f'**Swaddling only** (pre-registered videos, subjects 3 and 4). Without motion the cloth stays below a still '
           f'face and the fixed box keeps face skin; the fixed box was as accurate or slightly more accurate than NeoSTAR '
           f'at every level (differences {sw["diff"].min():+.1f} to {sw["diff"].max():+.1f} bpm; fixed box better at '
           f"{', '.join(sw[sw.verdict == 'fixed box better'].level)}; the fixed box had the lower error in "
           f'{sw_fixed_lower} of {len(swp)} subject-level comparisons, Supplementary Table S2). **Occlusion** (illustrative, subject 3). Under the generic hand + blanket occlusion the '
           f'fixed box was more accurate (differences {occ["diff"].min():+.1f} to {occ["diff"].max():+.1f} bpm; fixed box '
           f"better at {', '.join(occ[occ.verdict == 'fixed box better'].level)}), because NeoSTAR's selected regions were "
           'partly covered while the larger fixed box kept enough clean skin. In a single designed video in which a '
           'skin-coloured hand covers the centre of the face, inside the fixed box, the opposite occurred (fixed box '
           f'{oc.mae_fixed:.1f} vs. NeoSTAR {oc.mae_D5:.1f} bpm while the hand was present; Supplementary Section S4); this '
           'illustrates a mechanism and is not a test.')
    b.para(f'**Stabilization versus dynamic ROI.** v1 and NeoSTAR performed alike in every condition (e.g. '
           f'{fx.mae_v1:.1f} vs. {fx.mae_D5:.1f} bpm under the largest motion, {smm.mae_v1:.1f} vs. {smm.mae_D5:.1f} bpm '
           'under swaddling to the neck with motion). The robustness to motion therefore comes from tracking anatomical '
           'regions, which both versions do; in this pilot the stabilization contributed stability (Section 5.2), not '
           'HR accuracy.')
    face_paragraphs(b, R, fig_no, table_no)
    import results_v2 as RV
    RV.figure(b, R['fig_forest'], 6.5,
              f'Figure {fig_no}. Heart-rate error of NeoSTAR (D5) minus (a) the fixed box and (b) the tracked face box '
              'under each condition (mean and 95% moving-block bootstrap CI over paired windows; UBFC-rPPG with '
              'fingertip-PPG reference). Blue: NeoSTAR more accurate; orange: the comparator more accurate; grey: interval '
              'includes zero. Squares: pre-registered videos (swaddling, two or three subjects); circles: original videos '
              'and illustrative single-subject conditions.')
    RV.figure(b, R['fig_time'], 6.3,
              f'Figure {fig_no + 1}. HR over time for two conditions. Black: PPG reference; grey dashed: fixed box; '
              'orange dotted: v1; blue: NeoSTAR (D5); filled markers: validated windows; ticks at the bottom: windows in '
              'which the fixed box gave no HR.')
    b.caption(f'Table {table_no}. HR accuracy under controlled perturbations (UBFC-rPPG pilot, fingertip-PPG reference). '
              'MAE in bpm over windows with an HR; in brackets: windows without an HR. Δ: NeoSTAR − comparator on paired '
              'windows, mean and 95% CI; * interval excludes zero. Status: original = unperturbed videos; illustrative = '
              'one subject, fixed protocol; pre-reg. design = videos and analysis fixed in advance; pre-reg. claim = '
              'pre-registered claim about NeoSTAR versus the fixed box.')
    rows = []
    for _, r in T.iterrows():
        f = lambda k: ('–' if not np.isfinite(r[f'mae_{k}']) else f"{r[f'mae_{k}']:.1f}") + (f" ({int(r[f'nohr_{k}'])})" if r[f'nohr_{k}'] else '')  # noqa: E731
        g = lambda d, lo, hi: f"{r[d]:+.1f} [{r[lo]:+.1f}, {r[hi]:+.1f}]" + ('*' if r[lo] > 0 or r[hi] < 0 else '')  # noqa: E731
        rows.append([r.condition, r.level, r.subjects, r.windows, f('fixed'), f('v1'), f('FACE'), f('D5'),
                     g('diff', 'lo', 'hi'), g('face_diff', 'face_lo', 'face_hi'), r.design])
    b.table(['Condition', 'Level', 'Subj.', 'Win.', 'Fixed box', 'v1', 'Face box', 'NeoSTAR', 'Δ vs fixed box',
             'Δ vs face box', 'Status'],
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
    assert T[T.face_verdict == 'face box better'].empty, 'text below assumes the face box never beats NeoSTAR'
    u = row(R, 'Undisturbed', 'none')
    sl = row(R, 'Slow camera motion', 'large')
    fx = row(R, 'Fast, large motion', 'extra-large')
    smm = row(R, 'Swaddling + fast motion', 'to neck')
    fm = T[T.condition == 'Fast, large motion']
    occ = T[T.condition == 'Hand + blanket occlusion']
    b.para(f'**Tracked face box.** Following the face is not enough by itself (Figure {fig_no}b, Table {table_no}). '
           f'NeoSTAR was more accurate than the tracked face box in {len(better)} of {len(T)} conditions and the face box '
           'was never more accurate. The face box already lost accuracy on undisturbed video '
           f'({u.mae_FACE:.1f} vs. {u.mae_D5:.1f} bpm; NeoSTAR − face box {u.face_diff:+.1f} [{u.face_lo:+.1f}, '
           f'{u.face_hi:+.1f}]) and under slow camera motion (up to {sl.mae_FACE:.1f} bpm, where NeoSTAR and the fixed box '
           f'stayed below {max(sl.mae_D5, sl.mae_fixed) + 0.1:.1f} bpm), and under fast motion it reached '
           f'{fm.mae_FACE.min():.1f}-{fm.mae_FACE.max():.1f} bpm against {fm.mae_D5.min():.1f}-{fm.mae_D5.max():.1f} bpm for '
           f'NeoSTAR (better at all four levels). Against the fixed box, tracking helped only when the subject moved far: '
           f'the face box was more accurate under the largest motion ({fx.ff_diff:+.1f} bpm [{fx.ff_lo:+.1f}, '
           f'{fx.ff_hi:+.1f}]) and under swaddling with motion ({smm.ff_diff:+.1f} bpm [{smm.ff_lo:+.1f}, {smm.ff_hi:+.1f}] '
           'to the neck), and less accurate on still and slowly moving video. Under swaddling with motion, NeoSTAR and the '
           f'face box did not differ ({smm.mae_D5:.1f} vs. {smm.mae_FACE:.1f} bpm), and under generic hand + blanket '
           f'occlusion the fixed box remained the most accurate ROI (face box {occ.mae_FACE.min():.1f}-'
           f'{occ.mae_FACE.max():.1f} bpm). A rectangle around the face includes eyes, eyebrows, hair and background at its '
           'edges, and its position follows the pose estimate from frame to frame; NeoSTAR\'s gain over it therefore comes '
           'from region selection, the mask constraint and stabilization, not from tracking alone.')
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
            'on still video and under slow and fast motion, because a rectangle around a moving face mixes skin with '
            'non-skin and moves with the pose estimate. These conclusions rest on a three-subject pilot with simulated '
            'perturbations and indicate where a dynamic ROI matters; they do not establish its accuracy on infants. On '
            'the neonatal web videos themselves the pulse ranking depended on the split (the fixed box led on the '
            'validation videos and not on the test videos, Section 5.6), so no pulse-quality advantage of either '
            'representation can be claimed from those data.')


LIMITATION_REPLACEMENT = [
    '**Pilot scale of the reference-based study.** HR accuracy was measured on three adult UBFC-rPPG subjects; most '
    'conditions were applied to one subject and are illustrative, and only the swaddling + motion test carries a '
    'pre-registered claim about NeoSTAR. The confidence intervals describe window-to-window variation within these '
    'videos, not variation across people, cameras or skin tones.',
    '**Simulated perturbations.** Motion is a whole-frame warp, so face and background move together; the cloth, hand '
    'and blanket are composited layers that carry no pulse, do not deform with the body and do not change the lighting '
    'of the skin beneath them. Real infant motion (limb movement, head turning inside a wrap, handling by caregivers) '
    'may affect a dynamic ROI differently.',
    '**No infant reference.** No HR or respiratory-rate reference exists for the neonatal videos, so the representation '
    'was validated on infants only for stability, internal consistency and label-free pulse evidence; respiratory-rate '
    'accuracy was not evaluated at all.',
    '**Limited benefit.** NeoSTAR did not improve HR over a fixed box under lighting changes, swaddling alone or generic '
    'hand-and-blanket occlusion, and it lost the face when the cloth covered the mouth of one subject; its benefit is '
    'limited to conditions in which the subject moves.',
]

FUTURE_WORK = (
    '**Future work.** Four steps follow directly from these limitations. (i) Evaluate the frozen representation on '
    'neonatal recordings with synchronized ECG or pulse-oximetry and respiratory references, such as the NBHR newborn '
    'database [59], under a prospective protocol with ethics approval. (ii) Repeat the robustness '
    'study on adult datasets with real, not simulated, motion and more subjects (e.g. PURE [48] and MMPD [16]), so that '
    'between-subject variation can be estimated. (iii) Use the chest and abdominal regions of the representation for '
    'respiratory-rate estimation, and compare classical and learned signal extractors on the same ROIs, which is the '
    'subject of our follow-up work on heart and respiratory rate. (iv) Replace the rule-based region confidence by a '
    'calibrated reliability model once labelled neonatal data exist, and evaluate the online decoder (D3L) and a lighter '
    'pose model for real-time bedside use.')


def conclusion_sentence(R):
    fx, smm = row(R, 'Fast, large motion', 'extra-large'), row(R, 'Swaddling + fast motion', 'to neck')
    return ('In a controlled pilot against a PPG reference on UBFC-rPPG, the dynamic representation matched a fixed '
            f'box on still video and prevented its failure when the subject moved ({fx.mae_D5:.1f} vs. {fx.mae_fixed:.1f} '
            f'bpm under fast motion; {smm.mae_D5:.1f} vs. {smm.mae_fixed:.1f} bpm under swaddling with motion), whereas '
            'swaddling alone and generic occlusion did not favour it. The representation, the validity tests, the '
            'perturbation generators and the pre-registrations are released so that these benefits can be tested on '
            'neonatal recordings with a reference.')


# ------------------------------------------------------------------ supplement sections
def supplement(b, R, D):
    """Sections S2-S5 of the Supplementary Material (S1 is the video manifest, written by the caller)."""
    T, P, L, oc = R['T'], R['per_subject'], R['lag'], R['demo']
    claims = R['claims']
    b.h1('S2. Pre-registrations')
    b.para('Two pre-registrations were written before any result on their test videos was seen; nothing in '
           'them was changed after results were seen, and no deviation was recorded.')
    b.bullets([
        '**Swaddling + fast motion (claims about NeoSTAR).** Videos: UBFC subjects 1, 3 and 4, cloth to the neck '
        '(medium) or over the mouth (large), whole frame moved by the fast-motion generator at 160 px; seeds fixed per '
        'subject, both levels of a subject sharing the same motion. Methods frozen: fixed box (B0), PF-ROI (D0), NeoSTAR '
        '(D5). Error |HR − reference| on all windows; no-HR windows counted; paired moving-block bootstrap (block 8 '
        'windows, 5000 resamples, seed 42), pooled over subjects. Claim rule: NeoSTAR − fixed box CI upper bound < 0 '
        'and NeoSTAR MAE lower than the fixed box in at least 2 of 3 subjects. Results: '
        + '; '.join(f"{r.claim.split(' ')[0]} {r.result.lower()} ({r.detail.split('  ')[0]})"
                    for _, r in claims.iterrows() if r.claim.split(' ')[0] in ('M1', 'M2')) + '.',
        '**Swaddling only (videos and analysis fixed in advance).** Videos: UBFC subjects 3 and 4, cloth top edge at chin '
        '+0.6, 0 and −0.25 face heights (small, medium, large), seed fixed; same reference matching and bootstrap. The '
        'claims of this pre-registration concerned an experimental mask-repair variant that is not part of NeoSTAR and is '
        'not reported in the paper (none of its primary claims was supported); the NeoSTAR − fixed box contrast reported '
        'in Table 11 of the paper uses the pre-registered videos and analysis unchanged.',
        'The full pre-registration documents, generator code and seeds are released with the code.',
    ])
    b.h1('S3. Per-Subject Results and Time-Lag Check')
    b.caption('Table S2. Per-subject heart-rate error in the pre-registered conditions (MAE in bpm over windows with an HR; '
              'in brackets: windows without an HR).')
    f = lambda m, n: ('–' if np.isnan(m) else f'{m:.1f}') + (f' ({n})' if n else '')  # noqa: E731
    b.table(['Condition', 'Level', 'Subject', 'Windows', 'Fixed box', 'Face box', 'NeoSTAR'],
            [[r.condition, r.level, r.subject, r.windows, f(r.mae_fixed, r.nohr_fixed), f(r.mae_FACE, r.nohr_FACE),
              f(r.mae_D5, r.nohr_D5)] for _, r in P.iterrows()], size=8,
            widths=[1.6, 0.8, 0.7, 0.8, 0.9, 0.9, 0.9])
    b.caption('Table S3. Time-lag check. Mean absolute error of the fixed box and NeoSTAR (pooled) when the reference is '
              'shifted by −6 to +6 s in 1-s steps; best lag = shift with the lowest error.')
    b.table(['Condition group', 'MAE at 0 s (bpm)', 'Best lag (s)', 'MAE at best lag (bpm)'],
            [[r.condition, f'{r.mae_lag0:.2f}', f'{r.best_lag:+.0f}', f'{r.mae_best:.2f}'] for _, r in L.iterrows()],
            size=8, widths=[2.2, 1.3, 1.0, 1.5])
    assert (L.best_lag.abs() <= 1).all(), 'lag check: a condition group is best outside ±1 s'
    b.h1('S4. Designed Case: Hand Over the Face Centre')
    b.para('A skin-coloured procedural hand enters from the side and covers the nose and mid-cheeks of UBFC subject 1 in '
           'three intervals. The case was designed after the hand + blanket result to show the situation in which a fixed '
           'box is expected to fail: the occluder lies inside the box and passes its skin filter. It is one video and '
           'illustrates a mechanism; it is not a test.')
    g = lambda d, lo, hi: f"{oc[d]:+.1f} [{oc[lo]:+.1f}, {oc[hi]:+.1f}]"  # noqa: E731
    b.caption('Table S4. Hand over the face centre: windows in which the hand is present (MAE in bpm; Δ: NeoSTAR − '
              'comparator, mean and 95% moving-block bootstrap CI).')
    b.table(['Windows', 'Fixed box', 'PF-ROI', 'Face box', 'NeoSTAR', 'Δ vs fixed box', 'Δ vs face box'],
            [[int(oc.windows), f'{oc.mae_fixed:.1f}', f'{oc.mae_v1:.1f}', f'{oc.mae_FACE:.1f}', f'{oc.mae_D5:.1f}',
              g('diff', 'lo', 'hi'), g('face_diff', 'face_lo', 'face_hi')]], size=8,
            widths=[0.7, 0.8, 0.8, 0.8, 0.8, 1.3, 1.3])
    import results_v2 as RV
    RV.figure(b, R['fig_demo'], 6.3, 'Figure S1. HR over time with a hand over the face centre. Black: PPG reference; grey '
              'dashed: fixed box; orange dotted: PF-ROI; blue: NeoSTAR (D5); filled markers: validated windows; shading: '
              'hand present.')
