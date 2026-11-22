"""D9 report with the pre-registered claim rules applied automatically (../PREREGISTRATION.md).

    python module1_v2_d9/scripts/d9_report.py

Test (subjects 3, 4 swaddling): fixed box + D5 from runs/ubfc_swaddle_test (aug_eval_D5), v1 (aug_eval_D0), D9 from the
mirror run ubfc_swaddle_test_d9 (aug_eval_D9). Level 'none' = the original videos of subjects 3, 4.
No harm: D5 from every existing UBFC run (aug_eval_D5) vs D9 from its mirror (<run>_d9/aug_eval_D9); final34 label-free.
Writes module1_v2_d9/report/: claims.csv, metrics.csv, paired.csv, no_harm.csv, final34_check.csv, triggers.csv,
windows.csv, fig_swaddle_d9_paper.png, D9_report.html.
"""
import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

D9 = Path(__file__).resolve().parents[1]
ROOT = D9.parent
sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
from refmatch import nearest_reference  # noqa: E402

HRR = ROOT/'HR RR using module1 v2/runs'
REF = D9/'reference_hr_ppg.csv'
OUT = D9/'report'
OK = {'validated', 'validated_one_side'}
BLOCK, N_BOOT, MARGIN = 8, 5000, 2.0
SEVS = ['none', 'small', 'medium', 'large']
INK, INK2, GRID, SURFACE, MISS = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#c62828'
COL = {'fixed': ('Fixed box', '#2a78d6', 's', '--'), 'D5': ('Module 1 v2 D5', '#9aa0a6', '^', ':'),
       'D9': ('Module 1 v2 D9', '#1baf7a', 'o', '-')}
NOHARM_RUNS = ['ubfc_flat_38d8f553', 'ubfc_aug', 'ubfc_aug_fast', 'ubfc_aug_fast_xl', 'ubfc_occdemo']


def wr(run, folder):
    p = HRR/run/folder/'window_results_with_conditions.csv'
    return pd.read_csv(p) if p.is_file() else None


def cols(df, slot, key):
    return df[['video_id', 'window', f'hr_bpm_{slot}', f'label_{slot}']].rename(
        columns={f'hr_bpm_{slot}': f'hr_{key}', f'label_{slot}': f'lab_{key}'})


def tag(v):
    s = v.replace('NEW_', '').split('_')
    if len(s) == 1:
        return s[0], 'original', 'none'
    if 'democentre' in v:
        return s[0], 'occlusion_demo', 'centre'
    return s[0], '_'.join(s[1:-1]), s[-1]


def test_windows():
    a = wr('ubfc_swaddle_test', 'aug_eval_D5')
    w = a[['video_id', 'window', 'window_center_sec']].merge(cols(a, 'B0', 'fixed'), on=['video_id', 'window']) \
        .merge(cols(a, 'D5', 'D5'), on=['video_id', 'window'])
    w = w.merge(cols(wr('ubfc_swaddle_test', 'aug_eval_D0'), 'D5', 'v1'), on=['video_id', 'window'], how='left')
    m = wr('ubfc_swaddle_test_d9', 'aug_eval_D9')
    w = w.merge(cols(m, 'D5', 'D9'), on=['video_id', 'window'], how='left')
    w = w.merge(cols(m, 'B0', 'fixed_mirror'), on=['video_id', 'window'], how='left')
    o = wr('ubfc_flat_38d8f553', 'aug_eval_D5')
    o = o[o.video_id.isin(['NEW_subject3', 'NEW_subject4'])]
    on = o[['video_id', 'window', 'window_center_sec']].merge(cols(o, 'B0', 'fixed'), on=['video_id', 'window']) \
        .merge(cols(o, 'D5', 'D5'), on=['video_id', 'window'])
    on = on.merge(cols(wr('ubfc_flat_38d8f553_d9', 'aug_eval_D9'), 'D5', 'D9'), on=['video_id', 'window'], how='left')
    w = pd.concat([w, on], ignore_index=True)
    w['hr_ref'] = nearest_reference(w.video_id, w.window_center_sec, REF)
    t = w.video_id.map(tag)
    w['subject'], w['severity'] = t.str[0], t.str[2]
    return w


def strat_ci(parts, rng):
    parts = [p for p in parts if len(p)]
    if sum(len(p) for p in parts) < 2 * BLOCK:
        return np.nan, np.nan
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        tot = cnt = 0
        for d in parts:
            n = len(d)
            idx = rng.integers(0, n, n) if n <= BLOCK else \
                (rng.integers(0, n - BLOCK + 1, int(np.ceil(n / BLOCK)))[:, None] + np.arange(BLOCK)).ravel()[:n]
            tot += d[idx].sum()
            cnt += n
        means[b] = tot / cnt
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def contrast(g, a, b, rng):
    h = g[g.hr_ref.notna() & g[f'hr_{a}'].notna() & g[f'hr_{b}'].notna()].sort_values(['video_id', 'window'])
    parts = [((x[f'hr_{a}'] - x.hr_ref).abs() - (x[f'hr_{b}'] - x.hr_ref).abs()).to_numpy(float) for _, x in h.groupby('video_id')]
    d = np.concatenate(parts) if parts else np.array([])
    lo, hi = strat_ci(parts, rng)
    return dict(windows=len(d), mean_diff_bpm=d.mean() if len(d) else np.nan, ci95_low=lo, ci95_high=hi,
                verdict=f'{a} better' if hi < 0 else f'{b} better' if lo > 0 else 'no clear difference')


def metrics(w, keys):
    rows = []
    for sev, g in w.groupby('severity'):
        g = g[g.hr_ref.notna()]
        for k in keys:
            e = (g[f'hr_{k}'] - g.hr_ref).abs()
            rows.append(dict(severity=sev, method=k, windows=len(g), no_hr=int(e.isna().sum()), mae=e.mean(),
                             validated_pct=100 * (g[f'lab_{k}'].isin(OK) & e.notna()).mean(), within5_pct=100 * (e <= 5).mean()))
    return pd.DataFrame(rows)


def no_harm(rng):
    rows = []
    for run in NOHARM_RUNS + ['ubfc_swaddle']:
        a, m = wr(run, 'aug_eval_D5'), wr(f'{run}_d9', 'aug_eval_D9')
        if a is None or m is None:
            rows.append(dict(run=run, category='MISSING'))
            continue
        w = a[['video_id', 'window', 'window_center_sec']].merge(cols(a, 'D5', 'D5'), on=['video_id', 'window']) \
            .merge(cols(m, 'D5', 'D9'), on=['video_id', 'window'])
        w['hr_ref'] = nearest_reference(w.video_id, w.window_center_sec, REF)
        for v, g in w.groupby('video_id'):
            subj, cat, sev = tag(v)
            r = contrast(g, 'D9', 'D5', rng)
            e5, e9 = (g.hr_D5 - g.hr_ref).abs(), (g.hr_D9 - g.hr_ref).abs()
            rows.append(dict(run=run, video_id=v, category=cat + (' (development)' if run == 'ubfc_swaddle' else ''),
                             severity=sev, mae_D5=e5.mean(), mae_D9=e9.mean(), nohr_D5=int(e5.isna().sum()),
                             nohr_D9=int(e9.isna().sum()), **r))
    return pd.DataFrame(rows)


def final34():
    out = {}
    for key, run, folder in (('D5', 'final34_paper', 'd8check_D5'), ('D9', 'final34_paper_d9', 'd9check_D9')):
        p = HRR/run/folder/'summary_by_condition.csv'
        if not p.is_file():
            return None
        s = pd.read_csv(p)
        r = s[(s.group == 'all') & (s.method == 'D5')].iloc[0]
        out[key] = dict(validated_pct=r.reported_percent, gate1_above_BG_pp=r.gate1_above_BG_pp, snr_median_db=r.snr_median_db,
                        windows=r.windows)
    t = pd.DataFrame(out).T
    t.loc['D9 - D5'] = t.loc['D9'] - t.loc['D5']
    ok = t.loc['D9 - D5', 'validated_pct'] >= -5 and t.loc['D9 - D5', 'gate1_above_BG_pp'] >= -5
    t['no_harm'] = ['', '', 'PASS' if ok else 'FAIL']
    return t


def triggers():
    rows = []
    for p in sorted((D9/'runs').glob('*/D9/metadata.json')):
        for r in json.loads(p.read_text())['per_video']:
            rows.append(dict(run=p.parts[-3], **r))
    return pd.DataFrame(rows)


def figure(w, met, pair):
    fig = plt.figure(figsize=(11, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], hspace=.6, wspace=.22, top=.87)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    x = np.arange(len(SEVS))
    for k, (name, c, mk, ls) in COL.items():
        m = met.set_index(['severity', 'method'])
        mae = [m.loc[(s, k), 'mae'] if (s, k) in m.index else np.nan for s in SEVS]
        val = [m.loc[(s, k), 'validated_pct'] if (s, k) in m.index else np.nan for s in SEVS]
        kw = dict(color=c, lw=2.2, marker=mk, ms=8, mec=SURFACE, mew=1.2, ls=ls, label=name)
        a.plot(x, mae, **kw)
        b.plot(x, val, **kw)
        for xi, v in zip(x, mae):
            if np.isfinite(v) and k != 'D5':
                a.annotate(f'{v:.1f}', (xi, v), xytext=(0, 9 if k == 'fixed' else -15), textcoords='offset points',
                           ha='center', fontsize=8.5, color=INK2)
    r = pair[(pair.severity == 'large') & (pair.contrast == 'D9 - D5')]
    if len(r):
        r = r.iloc[0]
        a.text(.03, .62, f'large: D9 − D5 = {r.mean_diff_bpm:+.1f} bpm\n95 % CI [{r.ci95_low:+.1f}, {r.ci95_high:+.1f}]',
               transform=a.transAxes, fontsize=8, color=INK, va='top')
    labels = ['original', 'small\n(chest)', 'medium\n(neck)', 'large\n(chin + mouth)']
    for ax, yl, t in ((a, 'MAE vs PPG (bpm)', 'HR error (lower is better)'), (b, 'validated windows (%)', 'HR the system reports (higher is better)')):
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', color=GRID, lw=.8)
        ax.tick_params(labelsize=8.5, colors=INK2)
        ax.set_xticks(x, labels, fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5, color=INK2)
        ax.set_title(t, loc='left', fontsize=10, color=INK)
    a.set_ylim(0, None)
    b.set_ylim(-3, 105)
    a.legend(fontsize=8.5, frameon=False, loc='upper left')
    for j, subj in enumerate(('subject3', 'subject4')):
        ax = fig.add_subplot(gs[1 + j, :])
        d = w[(w.video_id == f'NEW_{subj}_swaddle_large')].sort_values('window_center_sec')
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', color=GRID, lw=.8)
        t = d.window_center_sec
        ax.plot(t, d.hr_ref, color=INK, lw=2.4, zorder=4)
        txt = []
        for k, (name, c, mk, ls) in COL.items():
            hr = d[f'hr_{k}']
            y = hr.clip(46, 129)
            valid = d[f'lab_{k}'].isin(OK) & hr.notna()
            ax.plot(t, y, color=c, lw=1.1, alpha=.5, ls=ls)
            ax.scatter(t[valid], y[valid], marker=mk, s=34, color=c, edgecolor=SURFACE, lw=.8, zorder=5)
            ax.scatter(t[~valid & hr.notna()], y[~valid & hr.notna()], marker=mk, s=30, facecolor='none', edgecolor=c, lw=1.3, zorder=5)
            miss = hr.isna()
            if miss.any():
                ax.scatter(t[miss], np.full(miss.sum(), 47.5), marker='|', s=70, color=c, lw=1.6, zorder=6)
            txt.append(f'{name.replace("Module 1 v2 ", "")} {(hr - d.hr_ref).abs().mean():.1f}')
        ax.set_ylim(45, 130)
        ax.set_ylabel('HR (bpm)', fontsize=8.5, color=INK2)
        ax.set_title(f'UBFC {subj}, swaddling large (test)   |   MAE (bpm): ' + ',  '.join(txt), loc='left', fontsize=9.5, color=INK)
        if j == 1:
            ax.set_xlabel('time (s), centre of each 8-s window', fontsize=8.5, color=INK2)
    h = [Line2D([], [], color=INK, lw=2.4, label='PPG reference')] + \
        [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.1, ms=6, label=n) for n, c, mk, ls in COL.values()] + \
        [Line2D([], [], color=INK2, marker='o', ls='', ms=6, label='filled = validated'),
         Line2D([], [], color=INK2, marker='o', ls='', ms=6, mfc='none', label='hollow = not validated'),
         Line2D([], [], color=INK2, marker='|', ls='', ms=10, mew=1.6, label='no HR')]
    fig.legend(handles=h, loc='upper center', ncol=7, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, .955))
    fig.suptitle('Swaddling: Module 1 v2 D9 vs D5 vs fixed box (UBFC subjects 3, 4: pre-registered test videos)',
                 x=.01, ha='left', y=.985, fontsize=12.5, color=INK)
    p = OUT/'fig_swaddle_d9_paper.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=SURFACE)
    plt.close(fig)
    return p


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    w = test_windows()
    w.to_csv(OUT/'windows.csv', index=False)
    met = metrics(w, ['fixed', 'v1', 'D5', 'D9'])
    pair = pd.DataFrame([dict(severity=s, contrast=f'{a} - {b}', **contrast(w[w.severity == s], a, b, rng))
                         for s in SEVS for a, b in (('D9', 'D5'), ('D9', 'fixed'), ('D5', 'fixed'), ('D9', 'v1'))])
    nh = no_harm(rng)
    f34 = final34()
    trg = triggers()
    mirror_check = float((w.hr_fixed_mirror - w.hr_fixed).abs().max()) if 'hr_fixed_mirror' in w else np.nan

    def ci(s, c):
        r = pair[(pair.severity == s) & (pair.contrast == c)]
        return r.iloc[0] if len(r) else None
    cl = []
    r = ci('large', 'D9 - D5')
    cl.append(dict(claim='S1 (primary)', statement='D9 recovers HR under heavy swaddling where D5 fails',
                   rule='large: D9 - D5 CI upper < 0', result='SUPPORTED' if r is not None and r.ci95_high < 0 else 'NOT supported',
                   detail=f'{r.mean_diff_bpm:+.2f} [{r.ci95_low:.2f}, {r.ci95_high:.2f}], n={r.windows}' if r is not None else ''))
    lv = {s: ci(s, 'D9 - fixed') for s in ('small', 'medium', 'large')}
    ok = all(x is not None and x.ci95_high <= MARGIN for x in lv.values())
    cl.append(dict(claim='S2', statement='D9 comparable to the fixed box under swaddling (margin 2 bpm)',
                   rule='every level: D9 - fixed CI upper <= +2', result='SUPPORTED' if ok else 'NOT supported',
                   detail='; '.join(f'{s} [{x.ci95_low:.2f}, {x.ci95_high:.2f}]' for s, x in lv.items() if x is not None)))
    hits = [s for s, x in lv.items() if x is not None and x.ci95_high < 0]
    cl.append(dict(claim='S3', statement='D9 beats the fixed box under swaddling', rule='D9 - fixed CI upper < 0 at some level',
                   result='SUPPORTED at ' + ', '.join(hits) if hits else 'NOT supported', detail=''))
    ex = nh[~nh.category.str.contains('development', na=False)]
    worse = ex[ex.ci95_low > 0]
    f34ok = f34 is not None and f34.loc['D9 - D5', 'no_harm'] == 'PASS'
    cl.append(dict(claim='No harm', statement='D9 never clearly worse than D5 on existing conditions; final34 within 5 pp',
                   rule='D9 - D5 CI lower <= 0 for every existing video; final34 PASS',
                   result='SUPPORTED' if worse.empty and f34ok and 'MISSING' not in set(nh.category) else 'NOT supported',
                   detail=('worse at: ' + ', '.join(worse.video_id) if len(worse) else 'never clearly worse') +
                   f'; final34 {"PASS" if f34ok else "FAIL/missing"}'))
    cl = pd.DataFrame(cl)
    for n, t in (('claims', cl), ('metrics', met), ('paired', pair), ('no_harm', nh), ('triggers', trg)):
        t.to_csv(OUT/f'{n}.csv', index=False)
    if f34 is not None:
        f34.to_csv(OUT/'final34_check.csv')
    p = figure(w, met, pair)
    pd.set_option('display.width', 250)
    print(cl.to_string(index=False))
    print(met.pivot(index='severity', columns='method', values='mae').reindex(SEVS).round(2).to_string())
    print(met.pivot(index='severity', columns='method', values='no_hr').reindex(SEVS).to_string())
    print(pair.round(2).to_string(index=False))
    print(nh[['category', 'severity', 'video_id', 'mae_D5', 'mae_D9', 'nohr_D5', 'nohr_D9', 'mean_diff_bpm', 'ci95_low', 'ci95_high', 'verdict']].round(2).to_string(index=False))
    print(f34)
    print(trg.to_string(index=False))
    print('fixed box mirror vs original max abs diff (must be 0):', mirror_check)
    img = base64.b64encode(p.read_bytes()).decode()
    (OUT/'D9_report.html').write_text(f"""<!doctype html><meta charset="utf-8"><title>D9 swaddling report</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:3px 6px}}</style>
<h1>D9 swaddling evaluation</h1><p>Pre-registration: {D9/'PREREGISTRATION.md'}. Fixed-box mirror check (max |diff|): {mirror_check}</p>
<h2>Pre-registered claims</h2>{cl.to_html(index=False)}<p><img style="max-width:100%" src="data:image/png;base64,{img}"></p>
<h2>Test metrics</h2>{met.round(2).to_html(index=False)}<h2>Paired</h2>{pair.round(2).to_html(index=False)}
<h2>No harm</h2>{nh.round(2).to_html(index=False)}<h2>final34</h2>{f34.round(2).to_html() if f34 is not None else 'missing'}
<h2>Rule triggers</h2>{trg.to_html(index=False)}""")
    print('report:', OUT/'D9_report.html')


if __name__ == '__main__':
    main()
