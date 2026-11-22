"""Swaddling + fast motion report with the pre-registered claim rules (PREREGISTRATION.md in this folder).

    python module1_v2_d9/swaddle_motion/swmo_report.py

Fixed box + D5 from runs/ubfc_swaddle_motion (aug_eval_D5), v1 (aug_eval_D0), D9 from the mirror run
ubfc_swaddle_motion_d9 (aug_eval_D9). Level 'none' = the original videos of subjects 1, 3, 4 (ubfc_flat_38d8f553).
Writes swaddle_motion/report/: claims.csv, metrics.csv, per_subject.csv, paired.csv, windows.csv, fig_swaddle_motion_paper.png,
swmo_report.html.
"""
import base64
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'scripts'))
import d9_report as dr  # noqa: E402  (shared helpers: wr, cols, contrast, styling constants)
from refmatch import nearest_reference  # noqa: E402

REF = HERE/'reference_hr_ppg.csv'
OUT = HERE/'report'
LV = ['none', 'medium', 'large']
LABEL = {'none': 'original\n(no cloth, still)', 'medium': 'swaddle to neck\n+ fast motion', 'large': 'swaddle to mouth\n+ fast motion'}
COL = {'fixed': ('Fixed box', '#2a78d6', 's', '--'), 'D5': ('Module 1 v2 D5', '#1baf7a', 'o', '-'),
       'D9': ('Module 1 v2 D9', '#7a4fd6', 'D', '-')}
OK = dr.OK


def windows():
    a = dr.wr('ubfc_swaddle_motion', 'aug_eval_D5')
    w = a[['video_id', 'window', 'window_center_sec']].merge(dr.cols(a, 'B0', 'fixed'), on=['video_id', 'window']) \
        .merge(dr.cols(a, 'D5', 'D5'), on=['video_id', 'window'])
    for run, folder, key in (('ubfc_swaddle_motion', 'aug_eval_D0', 'v1'), ('ubfc_swaddle_motion_d9', 'aug_eval_D9', 'D9')):
        x = dr.wr(run, folder)
        if x is not None:
            w = w.merge(dr.cols(x, 'D5', key), on=['video_id', 'window'], how='left')
        else:
            w[f'hr_{key}'], w[f'lab_{key}'] = np.nan, None
    o = dr.wr('ubfc_flat_38d8f553', 'aug_eval_D5')
    o = o[o.video_id.isin(['NEW_subject1', 'NEW_subject3', 'NEW_subject4'])]
    on = o[['video_id', 'window', 'window_center_sec']].merge(dr.cols(o, 'B0', 'fixed'), on=['video_id', 'window']) \
        .merge(dr.cols(o, 'D5', 'D5'), on=['video_id', 'window'])
    for run, folder, key in (('ubfc_flat_38d8f553', 'aug_eval_D0', 'v1'), ('ubfc_flat_38d8f553_d9', 'aug_eval_D9', 'D9')):
        x = dr.wr(run, folder)
        if x is not None:
            on = on.merge(dr.cols(x, 'D5', key), on=['video_id', 'window'], how='left')
    w = pd.concat([w, on], ignore_index=True)
    w['hr_ref'] = nearest_reference(w.video_id, w.window_center_sec, REF)
    s = w.video_id.str.replace('NEW_', '', regex=False).str.split('_')
    w['subject'] = s.str[0]
    w['level'] = s.apply(lambda x: x[-1] if len(x) > 1 else 'none')
    return w


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    w = windows()
    w.to_csv(OUT/'windows.csv', index=False)
    rows, subj = [], []
    for lv, g in w.groupby('level'):
        g = g[g.hr_ref.notna()]
        for k in ('fixed', 'v1', 'D5', 'D9'):
            e = (g[f'hr_{k}'] - g.hr_ref).abs()
            rows.append(dict(level=lv, method=k, windows=len(g), no_hr=int(e.isna().sum()), mae=e.mean(),
                             validated_pct=100 * (g[f'lab_{k}'].isin(OK) & e.notna()).mean()))
            for s, gg in g.groupby('subject'):
                subj.append(dict(level=lv, subject=s, method=k, mae=(gg[f'hr_{k}'] - gg.hr_ref).abs().mean(),
                                 no_hr=int(gg[f'hr_{k}'].isna().sum())))
    met, subj = pd.DataFrame(rows), pd.DataFrame(subj)
    pair = pd.DataFrame([dict(level=lv, contrast=f'{a} - {b}', **dr.contrast(w[w.level == lv], a, b, rng))
                         for lv in LV for a, b in (('D5', 'fixed'), ('D9', 'fixed'), ('D9', 'D5'), ('v1', 'fixed'))])

    def won(lv, k):
        t = subj[subj.level == lv].pivot(index='subject', columns='method', values='mae')
        return int((t[k] < t['fixed']).sum()) if k in t else 0
    cl = []
    for cid, k, lv in (('M1 (primary)', 'D5', 'medium'), ('M2', 'D5', 'large'), ('M3', 'D9', 'medium'), ('M4', 'D9', 'large')):
        r = pair[(pair.level == lv) & (pair.contrast == f'{k} - fixed')].iloc[0]
        ok = r.ci95_high < 0 and won(lv, k) >= 2
        cl.append(dict(claim=cid, statement=f'{k} more accurate than the fixed box: swaddle {lv} + fast motion',
                       rule=f'{k} - fixed CI upper < 0 (pooled) and {k} better in >= 2 of 3 subjects',
                       result='SUPPORTED' if ok else 'NOT supported',
                       detail=f'{r.mean_diff_bpm:+.2f} [{r.ci95_low:.2f}, {r.ci95_high:.2f}], n={r.windows}, subjects {won(lv, k)}/3'
                       + ('  (D9: usable only if D9 no-harm passes)' if k == 'D9' else '')))
    cl = pd.DataFrame(cl)
    for n, t in (('claims', cl), ('metrics', met), ('per_subject', subj), ('paired', pair)):
        t.to_csv(OUT/f'{n}.csv', index=False)

    # figure
    fig = plt.figure(figsize=(11, 10.5), facecolor=dr.SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1], hspace=.62, wspace=.22, top=.87)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    x = np.arange(len(LV))
    m = met.set_index(['level', 'method'])
    for k, (name, c, mk, ls) in COL.items():
        mae = [m.loc[(lv, k), 'mae'] if (lv, k) in m.index else np.nan for lv in LV]
        val = [m.loc[(lv, k), 'validated_pct'] if (lv, k) in m.index else np.nan for lv in LV]
        kw = dict(color=c, lw=2.2, marker=mk, ms=8, mec=dr.SURFACE, mew=1.2, ls=ls, label=name)
        a.plot(x, mae, **kw)
        b.plot(x, val, **kw)
        for xi, v in zip(x, mae):
            if np.isfinite(v):
                a.annotate(f'{v:.1f}', (xi, v), xytext=(0, 9) if k == 'fixed' else (14 if k == 'D9' else -14, -14),
                           textcoords='offset points', ha='center', fontsize=8, color=c)
    r = pair[(pair.level == 'medium') & (pair.contrast == 'D5 - fixed')].iloc[0]
    a.text(.03, .62, f'swaddle to neck + motion: D5 − fixed = {r.mean_diff_bpm:+.1f} bpm\n95 % CI [{r.ci95_low:+.1f}, {r.ci95_high:+.1f}] (block bootstrap, 3 subjects)',
           transform=a.transAxes, fontsize=8, color=dr.INK, va='top')
    for ax, yl, t in ((a, 'MAE vs PPG (bpm)', 'HR error (lower is better)'), (b, 'validated windows (%)', 'HR the system reports (higher is better)')):
        ax.set_facecolor(dr.SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', color=dr.GRID, lw=.8)
        ax.tick_params(labelsize=8.5, colors=dr.INK2)
        ax.set_xticks(x, [LABEL[lv] for lv in LV], fontsize=8.5)
        ax.set_ylabel(yl, fontsize=8.5, color=dr.INK2)
        ax.set_title(t, loc='left', fontsize=10, color=dr.INK)
        ax.set_xlim(-.4, len(LV) - .6)
    a.set_ylim(0, None)
    b.set_ylim(-3, 105)
    a.legend(fontsize=8.5, frameon=False, loc='upper left')
    for j, s in enumerate(('subject3', 'subject4')):
        ax = fig.add_subplot(gs[1 + j, :])
        d = w[w.video_id == f'NEW_{s}_swmo_medium'].sort_values('window_center_sec')
        ax.set_facecolor(dr.SURFACE)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(axis='y', color=dr.GRID, lw=.8)
        t = d.window_center_sec
        ax.plot(t, d.hr_ref, color=dr.INK, lw=2.4, zorder=4)
        txt = []
        for k, (name, c, mk, ls) in COL.items():
            hr = d[f'hr_{k}']
            y = hr.clip(46, 139)
            valid = d[f'lab_{k}'].isin(OK) & hr.notna()
            ax.plot(t, y, color=c, lw=1.1, alpha=.5, ls=ls)
            ax.scatter(t[valid], y[valid], marker=mk, s=34, color=c, edgecolor=dr.SURFACE, lw=.8, zorder=5)
            ax.scatter(t[~valid & hr.notna()], y[~valid & hr.notna()], marker=mk, s=30, facecolor='none', edgecolor=c, lw=1.3, zorder=5)
            miss = hr.isna()
            if miss.any():
                ax.scatter(t[miss], np.full(miss.sum(), 47.5), marker='|', s=70, color=c, lw=1.6, zorder=6)
            txt.append(f'{name.replace("Module 1 v2 ", "")} {(hr - d.hr_ref).abs().mean():.1f}')
        ax.set_ylim(45, 140)
        ax.set_ylabel('HR (bpm)', fontsize=8.5, color=dr.INK2)
        ax.set_title(f'UBFC {s}, swaddled to the neck + fast motion   |   MAE (bpm): ' + ',  '.join(txt), loc='left', fontsize=9.5, color=dr.INK)
        if j == 1:
            ax.set_xlabel('time (s), centre of each 8-s window', fontsize=8.5, color=dr.INK2)
    h = [Line2D([], [], color=dr.INK, lw=2.4, label='PPG reference')] + \
        [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.1, ms=6, label=n) for n, c, mk, ls in COL.values()] + \
        [Line2D([], [], color=dr.INK2, marker='o', ls='', ms=6, label='filled = validated'),
         Line2D([], [], color=dr.INK2, marker='o', ls='', ms=6, mfc='none', label='hollow = not validated'),
         Line2D([], [], color=dr.INK2, marker='|', ls='', ms=10, mew=1.6, label='no HR')]
    fig.legend(handles=h, loc='upper center', ncol=7, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, .955))
    fig.suptitle('Swaddled + fast motion: dynamic ROI vs fixed box (UBFC subjects 1, 3, 4; pre-registered)',
                 x=.01, ha='left', y=.985, fontsize=12.5, color=dr.INK)
    p = OUT/'fig_swaddle_motion_paper.png'
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor=dr.SURFACE)
    plt.close(fig)
    pd.set_option('display.width', 250)
    print(cl.to_string(index=False))
    print(met.pivot(index='level', columns='method', values='mae').reindex(LV).round(2).to_string())
    print(met.pivot(index='level', columns='method', values='no_hr').reindex(LV).to_string())
    print(subj.pivot_table(index=['level', 'subject'], columns='method', values='mae').round(2).to_string())
    print(pair.round(2).to_string(index=False))
    img = base64.b64encode(p.read_bytes()).decode()
    (OUT/'swmo_report.html').write_text(f"""<!doctype html><meta charset="utf-8"><title>Swaddle + motion report</title>
<style>body{{font:14px system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:3px 6px}}</style>
<h1>Swaddling + fast motion</h1><h2>Pre-registered claims</h2>{cl.to_html(index=False)}
<p><img style="max-width:100%" src="data:image/png;base64,{img}"></p><h2>Metrics</h2>{met.round(2).to_html(index=False)}
<h2>Per subject</h2>{subj.round(2).to_html(index=False)}<h2>Paired</h2>{pair.round(2).to_html(index=False)}""")
    print('report:', OUT/'swmo_report.html')


if __name__ == '__main__':
    main()
