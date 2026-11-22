"""D8 evaluation report with the pre-registered claim rules applied automatically (../PREREGISTRATION.md).

    python module1_v2_d8/scripts/d8_report.py

Reads (all produced by run_d8_eval.sh with the unchanged comparison tool, strict gate 2, band 45-180 bpm):
  HR RR using module1 v2/runs/<run>/aug_eval_D5   -> fixed box (B0), D5
  HR RR using module1 v2/runs/<run>/aug_eval_D0   -> v1 (method slot "D5" carries D0)
  HR RR using module1 v2/runs/<run>/aug_eval_D8   -> D8 (method slot "D5" carries D8)
  final34: HR RR using module1 v2/runs/final34_paper/d8check_D5 and d8check_D8 (label-free, infant band)
Reference: fingertip-PPG HR per 8-s window, nearest window centre within 1 s (module1_v2_d8/reference_hr_ppg.csv).
Writes module1_v2_d8/report/: D8_report.html, metrics.csv, paired.csv, claims.csv, final34_check.csv, windows.csv, figures.
"""
import base64
import datetime as dt
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D8 = Path(__file__).resolve().parents[1]
ROOT = D8.parent
sys.path.insert(0, str(ROOT/'dataaugment/hr_eval'))
from refmatch import nearest_reference  # noqa: E402

M2RUNS = ROOT/'HR RR using module1 v2/runs'
REF = D8/'reference_hr_ppg.csv'
OUT = D8/'report'
RUNS = ['ubfc_flat_38d8f553', 'ubfc_aug', 'ubfc_aug_fast', 'ubfc_aug_fast_xl', 'ubfc_swaddle']
ORIGINAL = {'NEW_subject1': ['motion', 'motion_fast', 'swaddle'], 'NEW_subject3': ['occlusion'], 'NEW_subject4': ['lighting']}
CATS = ['motion', 'motion_fast', 'occlusion', 'lighting', 'swaddle']
LABEL = {'motion': 'Motion (slow drift)', 'motion_fast': 'Fast, large motion', 'occlusion': 'Occlusion (hand + blanket)',
         'lighting': 'Lighting', 'swaddle': 'Swaddling (cloth up to chest / chin / mouth)'}
SEVS = ['none', 'small', 'medium', 'large', 'xlarge']
SEVN = {'none': 'original', 'xlarge': 'extra-large'}
METHODS = {'fixed': ('Fixed box', '#2a78d6', 's', '--'), 'v1': ('Module 1 v1', '#eb6834', '^', ':'),
           'D5': ('Module 1 v2 D5', '#1baf7a', 'o', '-'), 'D8': ('Module 1 v2 D8', '#7a4fd6', 'D', '-')}
REPORTED = ('validated', 'validated_one_side')
BLOCK, N_BOOT = 8, 5000


def load():
    parts = []
    for run in RUNS:
        d = {k: M2RUNS/run/f'aug_eval_{k}'/'window_results_with_conditions.csv' for k in ('D5', 'D0', 'D8')}
        if not all(p.is_file() for p in d.values()):
            print('skipping', run, '(missing', [k for k, p in d.items() if not p.is_file()], ')')
            continue
        a, b, c = (pd.read_csv(d[k]) for k in ('D5', 'D0', 'D8'))
        w = a[['video_id', 'window', 'window_center_sec', 'hr_bpm_B0', 'label_B0', 'hr_bpm_D5', 'label_D5']].rename(
            columns={'hr_bpm_B0': 'hr_fixed', 'label_B0': 'label_fixed', 'hr_bpm_D5': 'hr_D5', 'label_D5': 'label_D5'})
        for src, key in ((b, 'v1'), (c, 'D8')):
            w = w.merge(src[['video_id', 'window', 'hr_bpm_D5', 'label_D5']].rename(
                columns={'hr_bpm_D5': f'hr_{key}', 'label_D5': f'label_{key}'}), on=['video_id', 'window'], how='left')
        parts.append(w)
    w = pd.concat(parts, ignore_index=True)
    w['hr_ref'] = nearest_reference(w.video_id, w.window_center_sec, REF)
    rows = []
    for v, g in w.groupby('video_id', sort=False):
        if v in ORIGINAL:
            for cat in ORIGINAL[v]:
                rows.append(g.assign(category=cat, severity='none'))
        else:
            s = v.replace('NEW_', '').split('_')
            rows.append(g.assign(category='_'.join(s[1:-1]), severity=s[-1]))
    return pd.concat(rows, ignore_index=True)


def block_ci(d, rng):
    n = len(d)
    if n < 2 * BLOCK:
        return np.nan, np.nan
    starts, k = np.arange(n - BLOCK + 1), int(np.ceil(n / BLOCK))
    means = [d[(rng.choice(starts, k)[:, None] + np.arange(BLOCK)).ravel()[:n]].mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def metrics(w):
    rows = []
    for (cat, sev), g in w.groupby(['category', 'severity']):
        g = g[g.hr_ref.notna()]
        for m, (name, *_) in METHODS.items():
            err = (g[f'hr_{m}'] - g.hr_ref).abs()
            rep = g[f'label_{m}'].isin(REPORTED) & err.notna()
            rows.append(dict(category=cat, severity=sev, method=name, method_key=m, windows=len(g),
                             no_hr=int(err.isna().sum()), mae_all=err.mean(), validated_pct=100 * rep.mean(),
                             mae_validated=err[rep].mean(), within5_all_pct=100 * (err <= 5).mean()))
    return pd.DataFrame(rows)


def paired(w):
    rng = np.random.default_rng(42)
    rows = []
    for cat in CATS:
        for sev in SEVS:
            g = w[(w.category == cat) & (w.severity == sev) & w.hr_ref.notna() & w.hr_D8.notna()].sort_values('window')
            if g.empty:
                continue
            for other in ('fixed', 'D5', 'v1'):
                h = g[g[f'hr_{other}'].notna()]
                d = ((h.hr_D8 - h.hr_ref).abs() - (h[f'hr_{other}'] - h.hr_ref).abs()).to_numpy(float)
                lo, hi = block_ci(d, rng)
                rows.append(dict(category=cat, severity=sev, contrast=f'D8 - {METHODS[other][0]}', windows=len(d),
                                 mean_diff_bpm=d.mean() if len(d) else np.nan, ci95_low=lo, ci95_high=hi,
                                 verdict='D8 better' if hi < 0 else f'{METHODS[other][0]} better' if lo > 0 else 'no clear difference'))
    return pd.DataFrame(rows)


def final34_check():
    rows = {}
    for key, folder, method in (('D5', 'd8check_D5', 'D5'), ('D8', 'd8check_D8', 'D5')):
        f = M2RUNS/'final34_paper'/folder/'summary_by_condition.csv'
        if not f.is_file():
            return None
        s = pd.read_csv(f)
        r = s[(s.group == 'all') & (s.method == method)].iloc[0]
        rows[key] = dict(validated_pct=r.reported_percent, gate1_above_BG_pp=r.gate1_above_BG_pp, snr_median_db=r.snr_median_db,
                         windows=r.windows)
    t = pd.DataFrame(rows).T
    t.loc['D8 - D5'] = t.loc['D8'] - t.loc['D5']
    t['no_harm'] = ''
    ok = (t.loc['D8 - D5', 'validated_pct'] >= -5) and (t.loc['D8 - D5', 'gate1_above_BG_pp'] >= -5)
    t.loc['D8 - D5', 'no_harm'] = 'PASS' if ok else 'FAIL'
    return t


def claims(pair, f34):
    def ci(cat, sev, contrast):
        r = pair[(pair.category == cat) & (pair.severity == sev) & (pair.contrast == contrast)]
        return (r.ci95_low.iloc[0], r.ci95_high.iloc[0]) if len(r) else (np.nan, np.nan)
    out = []
    for cat, claim in (('occlusion', 'D8 handles occlusion'), ('swaddle', 'D8 handles swaddling')):
        hits = [s for s in ('medium', 'large') if ci(cat, s, 'D8 - Fixed box')[1] < 0]
        out.append(dict(claim=claim, rule='D8 - fixed box CI upper < 0 at medium or large',
                        result='SUPPORTED' if hits else 'NOT supported', detail='levels: ' + (', '.join(hits) or 'none')))
    f34_ok = f34 is not None and f34.loc['D8 - D5', 'no_harm'] == 'PASS'
    for cat in CATS:
        sevs = [s for s in SEVS if not pair[(pair.category == cat) & (pair.severity == s)].empty]
        worse = [s for s in sevs if ci(cat, s, 'D8 - Module 1 v2 D5')[0] > 0]
        out.append(dict(claim=f'No harm vs D5: {LABEL[cat]}', rule='D8 - D5 CI lower <= 0 at every level; final34 within 5 pp',
                        result='SUPPORTED' if not worse and f34_ok else 'NOT supported',
                        detail=('D8 clearly worse at: ' + ', '.join(worse) if worse else 'never clearly worse than D5') +
                               ('' if f34_ok else '; final34 check ' + ('missing' if f34 is None else 'FAILED'))))
        best = [s for s in sevs if s != 'none' and ci(cat, s, 'D8 - Fixed box')[1] < 0 and ci(cat, s, 'D8 - Module 1 v2 D5')[1] <= 0]
        out.append(dict(claim=f'Best: {LABEL[cat]}', rule='D8 - fixed CI < 0 and D8 - D5 CI upper <= 0',
                        result='SUPPORTED at ' + ', '.join(SEVN.get(s, s) for s in best) if best else 'NOT supported', detail=''))
    return pd.DataFrame(out)


def fig_severity(met, path):
    cats = [c for c in CATS if (met.category == c).any()]
    fig, axes = plt.subplots(1, len(cats), figsize=(3.6 * len(cats), 3.5), sharey=True)
    for ax, cat in zip(np.atleast_1d(axes), cats):
        r = met[met.category == cat]
        sevs = [s for s in SEVS if (r.severity == s).any()]
        for m, (name, color, mk, ls) in METHODS.items():
            q = r[r.method_key == m].set_index('severity').reindex(sevs)
            ax.plot(range(len(sevs)), q.mae_all, color=color, marker=mk, ls=ls, lw=2, label=name)
        ax.set_xticks(range(len(sevs)), [SEVN.get(s, s) for s in sevs], fontsize=8)
        ax.set_title(LABEL[cat].split(' (')[0], loc='left', fontsize=10)
        ax.grid(axis='y', alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel('MAE vs PPG, all windows (bpm)')
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle('HR error by severity: fixed box vs Module 1 v1 vs v2 D5 vs v2 D8 (lower is better)', x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_timeline(w, cat, path):
    sub = w[w.category == cat]
    sevs = [s for s in SEVS if (sub.severity == s).any()]
    fig, axes = plt.subplots(len(sevs), 1, figsize=(11, 2.4 * len(sevs) + .6), sharex=True, squeeze=False)
    for ax, sev in zip(axes[:, 0], sevs):
        g = sub[sub.severity == sev].sort_values('window_center_sec')
        ax.plot(g.window_center_sec, g.hr_ref, color='#0b0b0b', lw=2.2, label='PPG reference', zorder=5)
        txt = []
        for m in ('fixed', 'D5', 'D8'):
            name, color, mk, ls = METHODS[m]
            ok = g[f'label_{m}'].isin(REPORTED)
            ax.plot(g.window_center_sec, g[f'hr_{m}'], color=color, ls=ls, lw=1, alpha=.6)
            ax.scatter(g.window_center_sec[ok], g[f'hr_{m}'][ok], color=color, marker=mk, s=24, zorder=6, label=name)
            ax.scatter(g.window_center_sec[~ok], g[f'hr_{m}'][~ok], facecolor='none', edgecolor=color, marker=mk, s=22, zorder=6)
            txt.append(f'{name.replace("Module 1 v2 ", "")} {(g[f"hr_{m}"] - g.hr_ref).abs().mean():.1f}')
        ax.set_ylim(45, 135)
        ax.grid(alpha=.3)
        ax.set_title(f'{SEVN.get(sev, sev)}   |   MAE (bpm): ' + ', '.join(txt), loc='left', fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False, ncol=4)
    axes[-1, 0].set_xlabel('time (s); filled = validated, hollow = not validated')
    fig.suptitle(f'{LABEL[cat]}: reference vs fixed box, D5, D8', x=.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def b64(p):
    return 'data:image/png;base64,' + base64.b64encode(Path(p).read_bytes()).decode()


def tbl(df):
    return df.to_html(index=False, float_format=lambda x: f'{x:.2f}', na_rep='—', border=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    w = load()
    w.to_csv(OUT/'windows.csv', index=False)
    met = metrics(w)
    met.to_csv(OUT/'metrics.csv', index=False)
    pair = paired(w)
    pair.to_csv(OUT/'paired.csv', index=False)
    f34 = final34_check()
    if f34 is not None:
        f34.to_csv(OUT/'final34_check.csv')
    cl = claims(pair, f34)
    cl.to_csv(OUT/'claims.csv', index=False)
    fig_severity(met, OUT/'fig_mae_by_severity.png')
    figs = []
    for cat in ('occlusion', 'swaddle', 'motion_fast', 'lighting'):
        if (w.category == cat).any():
            fig_timeline(w, cat, OUT/f'fig_timeline_{cat}.png')
            figs.append(cat)
    piv = met.pivot_table(index=['category', 'severity'], columns='method', values='mae_all', aggfunc='first').reset_index()
    flags = {}
    for run in RUNS:
        m = D8/'runs'/run/'dynamic_roi_v2_D8'/'metadata.json'
        if m.is_file():
            for v in json.loads(m.read_text())['per_video']:
                flags[v['video_id']] = v
    flag_tbl = pd.DataFrame(flags.values()) if flags else pd.DataFrame()
    css = ('body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:auto;padding:20px 16px;'
           'background:#fcfcfb;color:#0b0b0b}table{border-collapse:collapse;font-size:12.5px;margin:8px 0}'
           'td,th{border-bottom:1px solid #e4e3df;padding:3px 8px;text-align:right}img{max-width:100%}'
           '.box{border-left:3px solid #7a4fd6;padding:6px 12px;background:#f4f2fa}')
    page = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Module 1 v2 D8 Evaluation</title><style>{css}</style></head><body>',
            '<h1>Module 1 v2 D8 (D5 + occlusion / coverage rules): pre-registered evaluation</h1>',
            f'<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Rules, thresholds and claim rules were fixed before any D8 result: '
            f'<code>{html.escape(str(D8/"PREREGISTRATION.md"))}</code>. D8 was motivated by D5\'s occlusion failure on these same '
            'videos; no other subjects with a reference HR were available. One adult subject per condition, synthetic '
            'disturbances: a controlled pilot.</p>',
            '<div class="box"><h2>Claim checks (pre-registered rules)</h2>' + tbl(cl) + '</div>',
            '<h2>HR error by severity</h2><img src="' + b64(OUT/'fig_mae_by_severity.png') + '">',
            '<h2>MAE, all windows (bpm)</h2>' + tbl(piv),
            '<h2>Paired differences (D8 minus comparator; negative = D8 better; 95 % moving-block bootstrap CI)</h2>' + tbl(pair),
            '<h2>Infant data (final34, no reference): D8 vs D5 label-free</h2>' +
            (f34.reset_index().rename(columns={'index': 'config'}).to_html(index=False, float_format=lambda x: f'{x:.2f}', border=0)
             if f34 is not None else '<p>not available</p>'),
            '<h2>How often the D8 rules fired</h2>' + (tbl(flag_tbl) if len(flag_tbl) else '<p>—</p>')]
    for cat in figs:
        page.append(f'<h2>{LABEL[cat]}</h2><img src="{b64(OUT/f"fig_timeline_{cat}.png")}">')
    page.append('<h2>Full metrics</h2>' + tbl(met) + '</body></html>')
    (OUT/'D8_report.html').write_text(''.join(page), encoding='utf-8')
    pd.set_option('display.width', 220)
    print(cl.to_string(index=False))
    print(piv.round(2).to_string(index=False))
    if f34 is not None:
        print(f34.round(2).to_string())
    print('report:', OUT/'D8_report.html')


if __name__ == '__main__':
    main()
