"""Build the v1 / v2 / v3 ablation report (tables, statistics, figures, REPORT.md / REPORT.html, LaTeX tables).

Input: --datasets JSON written by 'Run v123 ablation study.ipynb', a list of
  {name, branch, manifest, hr_dir, rr_dir, events_dir, neural_dirs: [...], pretrain_dirs: [...], run_dir?}
in priority order: a video (SHA-256) present in several datasets is analysed once, from the first dataset.
Optional: --synthetic-dir (benchmark_v123_synthetic.py), --timings JSON {stage: minutes}, reference CSVs
(--reference-hr: dataset?, video_id, timestamp_sec, hr_bpm; --reference-rr: ... rr_brpm; --reference-events:
dataset?, video_id, event_type, start_sec, end_sec). Statistics: see module2.ablation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd

from module2.ablation import (HR_COMPONENTS, HR_RANGE, HR_VERSIONS, RR_RANGE, accuracy, bootstrap_ci, compare,
                              coverage_accuracy, fmt_ci, fmt_p, friedman, holm, pair_reference, paired_test,
                              per_video_quality, summarize)
from module1.reporting import Report, plt_setup
from module2.events import match_events

COLORS = {'v1': '#4c72b0', 'v2': '#dd8452', 'v3': '#c44e52'}
METRICS = ['reliable_percent', 'sqi_median', 'snr_median_db', 'coverage_percent', 'jump_median', 'plausible_percent']
METRIC_NAMES = dict(reliable_percent='Reliable windows (%)', sqi_median='SQI (median)', snr_median_db='SNR (dB, median)',
                    coverage_percent='Coverage (%)', jump_median='Window-to-window change (median)',
                    plausible_percent='Physiological range (%)', mae='MAE', rmse='RMSE')
METHOD_ORDER = ['GREEN', 'CHROM', 'POS', 'INTENSITY', 'FLOW', 'NEURAL_SRC', 'NEURAL_TTA_UNGUIDED', 'NEURAL_TTA_NO_QSAMPLE',
                'NEURAL_TTA_NO_QWEIGHT', 'NEURAL_TTA_NO_SITES', 'NEURAL_TTA']
STRATEGY_ORDER = ['infant_box', 'pool_mean', 'top1', 'confidence_fusion', 'snr_fusion', 'dynamic_fusion', 'consensus_fusion',
                  'cardiorespiratory_fusion']


def ordered(grid):
    rows = [m for m in METHOD_ORDER if m in grid.index] + [m for m in grid.index if m not in METHOD_ORDER]
    columns = [c for c in STRATEGY_ORDER if c in grid.columns] + [c for c in grid.columns if c not in STRATEGY_ORDER]
    return grid.loc[rows, columns]


HIGHER_IS_BETTER = dict(reliable_percent=True, sqi_median=True, snr_median_db=True, coverage_percent=True,
                        jump_median=False, plausible_percent=True)


def read_csv(path, **kwargs):
    path = Path(path)
    return pd.read_csv(path, low_memory=False, **kwargs) if path.is_file() else pd.DataFrame()


# --------------------------------------------------------------------------------------------- loading
def load(datasets):
    videos, hr, rr, events, event_summary, histories, neural_meta, pretrain_meta = ([] for _ in range(8))
    seen = set()
    for d in datasets:
        manifest = pd.read_csv(d['manifest'])
        manifest['sha256'] = manifest.get('sha256', manifest.video_id).astype(str)
        for row in manifest.itertuples():
            duplicate = row.sha256 in seen
            seen.add(row.sha256)
            videos.append(dict(dataset=d['name'], branch=d['branch'], video_id=str(row.video_id), filename=row.filename,
                               duration_sec=row.duration_sec, fps=row.fps, width=row.width, height=row.height,
                               sha256=row.sha256, video_key=row.sha256[:16], analysed=not duplicate))
        tag = lambda frame: frame.assign(dataset=d['name'], video_id=frame.video_id.astype(str)) if len(frame) else frame
        hr.append(tag(read_csv(Path(d['hr_dir'])/'hr_windows.csv')))
        rr.append(tag(read_csv(Path(d['rr_dir'])/'rr_windows.csv')))
        events.append(tag(read_csv(Path(d['events_dir'])/'events.csv')))
        event_summary.append(tag(read_csv(Path(d['events_dir'])/'event_summary.csv')))
        for folder in d.get('neural_dirs', []):
            history = read_csv(Path(folder)/'adaptation_history.csv')
            if len(history):
                histories.append(history.assign(dataset=d['name'], video_id=history.video_id.astype(str)))
            meta = Path(folder)/'experiment_metadata.json'
            if meta.is_file():
                for info in json.loads(meta.read_text())['video_info']:
                    neural_meta.append(dict(info, dataset=d['name']))
        for folder in d.get('pretrain_dirs', []):
            meta = Path(folder)/'pretrain_metadata.json'
            if meta.is_file():
                info = json.loads(meta.read_text())
                pretrain_meta.append(dict(dataset=d['name'], fold=Path(folder).name, source_videos=len(info['videos']),
                                          excluded_target_videos=len(info.get('excluded_target_videos', [])),
                                          minutes=info.get('minutes')))
    videos = pd.DataFrame(videos)
    keep = videos[videos.analysed][['dataset', 'video_id', 'video_key']]
    restrict = lambda frames: (pd.concat([f for f in frames if len(f)], ignore_index=True).merge(keep, on=['dataset', 'video_id'])
                               if any(len(f) for f in frames) else pd.DataFrame(columns=['dataset', 'video_id', 'video_key']))
    return dict(videos=videos, hr=restrict(hr), rr=restrict(rr), events=restrict(events),
                event_summary=restrict(event_summary), histories=restrict(histories),
                neural_meta=pd.DataFrame(neural_meta), pretrain_meta=pd.DataFrame(pretrain_meta))


def config_rows(quality, definitions):
    """Per-video rows of the named configurations: definitions = [(label, method, strategy)]."""
    parts = []
    for label, method, strategy in definitions:
        part = quality[(quality.method == method) & (quality.strategy == strategy)]
        parts.append(part.assign(version=label))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mean_ci_table(per_video, group, metrics, digits=1):
    summary = summarize(per_video, group, metrics)
    out = summary[[group, 'videos']].copy()
    for metric in metrics:
        out[METRIC_NAMES.get(metric, metric)] = [fmt_ci(r[metric], r[f'{metric}_ci_low'], r[f'{metric}_ci_high'], digits)
                                                 for _, r in summary.iterrows()]
    return summary, out


def stats_table(comparison):
    if comparison.empty:
        return comparison
    out = comparison.copy()
    out['metric'] = out.metric.map(lambda m: METRIC_NAMES.get(m, m))
    out['p (Wilcoxon)'] = out.p_value.map(fmt_p)
    out['p (Holm)'] = out.p_holm.map(fmt_p)
    return out[['metric', 'baseline', 'candidate', 'n', 'mean_difference', 'median_difference', 'rank_biserial',
                'p (Wilcoxon)', 'p (Holm)']].round(3)


# --------------------------------------------------------------------------------------------- sections
def section_data(report, data):
    videos = data['videos']
    report.heading('1. Data')
    rows = []
    for name, part in videos.groupby('dataset', sort=False):
        analysed = part[part.analysed]
        rows.append(dict(dataset=name, module1_branch=part.branch.iloc[0], videos=len(part), analysed=len(analysed),
                         duplicates_removed=len(part) - len(analysed), total_duration_s=round(analysed.duration_sec.sum(), 1),
                         median_duration_s=round(analysed.duration_sec.median(), 1) if len(analysed) else None,
                         fps_range=f'{analysed.fps.min():.1f}-{analysed.fps.max():.1f}' if len(analysed) else '',
                         resolution=', '.join(sorted({f'{w}x{h}' for w, h in zip(analysed.width, analysed.height)}))[:60]))
    table = pd.DataFrame(rows)
    total = videos[videos.analysed]
    report.text(f'{len(total)} unique videos ({total.duration_sec.sum() / 60:.1f} min) from {len(table)} dataset(s). '
                'Videos are matched by SHA-256; a video present in several datasets is analysed once (first dataset '
                'listed). The video is the statistical unit throughout.')
    report.table(table, 'T1_datasets', 'Datasets (after removing duplicate videos)')
    videos.to_csv(report.output/'tables'/'A1_all_videos.csv', index=False)


def section_hr(report, data, args):
    plt = plt_setup()
    hr = data['hr']
    quality = per_video_quality(hr, 'hr_bpm', HR_RANGE)
    quality = quality.merge(hr[['dataset', 'video_id', 'video_key']].drop_duplicates(), on=['dataset', 'video_id'])
    quality.to_csv(report.output/'tables'/'A2_hr_quality_per_video_all_configurations.csv', index=False)
    report.heading('2. Heart rate: v1 vs v2 vs v3')
    report.text('All versions are computed in one run per dataset (same windows of 8 s with 1 s step, same Module 1 '
                'output, same code); a version is a pulse method and a fusion strategy. Values are the mean over '
                'videos with a 95 % bootstrap confidence interval. Without a reference HR these metrics describe '
                'signal quality, stability and plausibility, not accuracy (Section 7).')
    versions = config_rows(quality, [(k, m, s) for k, _, m, s, _ in HR_VERSIONS])
    available = [k for k, *_ in HR_VERSIONS if k in set(versions.version)]
    summary, table = mean_ci_table(versions, 'version', METRICS)
    table.insert(1, 'configuration', [next(f'{m} + {s}' for k, _, m, s, _ in HR_VERSIONS if k == v) for v in table.version])
    report.table(table, 'T2_hr_main_ablation', 'HR: main ablation v1 / v2 / v3 (mean [95 % CI] over videos)')
    rows = []
    for metric in METRICS:
        wide = versions.pivot_table(index='video_key', columns='version', values=metric, aggfunc='first')
        result = friedman(wide[[v for v in available]]) if len(available) == 3 else dict(n=0, statistic=np.nan, p_value=np.nan)
        rows.append(dict(metric=METRIC_NAMES[metric], test='Friedman v1/v2/v3', n=result['n'],
                         statistic=round(result['statistic'], 3) if np.isfinite(result['statistic']) else None,
                         p=fmt_p(result['p_value'])))
    pairwise = pd.concat([compare(versions, 'v1', ['v2', 'v3'], METRICS), compare(versions, 'v2', ['v3'], METRICS)])
    report.table(stats_table(pairwise), 'T3_hr_main_statistics',
                 'HR: paired Wilcoxon signed-rank tests over videos (difference = candidate - baseline; rank-biserial '
                 'effect size; Holm correction per metric and baseline)')
    report.table(pd.DataFrame(rows), 'T3b_hr_friedman', 'HR: Friedman test across v1, v2 and v3')

    # Figure: main metrics with CI + paired per-video lines.
    figure, axes = plt.subplots(1, 4, figsize=(11, 2.8))
    for axis, metric in zip(axes, ('reliable_percent', 'sqi_median', 'snr_median_db', 'jump_median')):
        for i, version in enumerate(available):
            row = summary[summary.version == version].iloc[0]
            axis.errorbar(i, row[metric], yerr=[[row[metric] - row[f'{metric}_ci_low']], [row[f'{metric}_ci_high'] - row[metric]]],
                          fmt='o', color=COLORS[version], capsize=3, ms=6, zorder=3)
        wide = versions.pivot_table(index='video_key', columns='version', values=metric, aggfunc='first')[available]
        for _, values in wide.iterrows():
            axis.plot(range(len(available)), values.values, color='#999', alpha=.25, lw=.7, zorder=1)
        axis.set_xticks(range(len(available)), available)
        axis.set_title(METRIC_NAMES[metric], fontsize=9)
        axis.grid(axis='y', alpha=.3)
    figure.tight_layout()
    report.figure(figure, 'F1_hr_main_ablation', 'HR main ablation: mean and 95 % CI over videos (dots), each grey '
                  'line one video. Lower is better only for the window-to-window change.')

    # Component ablation.
    report.heading('3. Heart rate: component ablation (one component removed at a time)', 2)
    components = config_rows(quality, [(label, m, s) for label, m, s, _, _ in HR_COMPONENTS])
    references = {label: ref for label, _, _, ref, _ in HR_COMPONENTS}
    removed = {label: what for label, _, _, _, what in HR_COMPONENTS}
    both = pd.concat([versions, components], ignore_index=True)
    rows, comps = [], []
    for label, reference in references.items():
        if label not in set(components.version) or reference not in set(versions.version):
            continue
        comparison = compare(both, reference, [label], METRICS)
        comps.append(comparison)
        for metric in ('reliable_percent', 'sqi_median', 'snr_median_db', 'jump_median'):
            r = comparison[comparison.metric == metric]
            if len(r):
                r = r.iloc[0]
                rows.append(dict(configuration=label, removed=removed[label], reference=reference, metric=METRIC_NAMES[metric],
                                 n=r.n, delta_mean=round(r.mean_difference, 3), rank_biserial=round(r.rank_biserial, 3),
                                 p=fmt_p(r.p_value)))
    component_table = pd.DataFrame(rows)
    if len(comps):
        all_comps = pd.concat(comps, ignore_index=True)
        all_comps['p_holm_all'] = holm(all_comps.p_value)
        all_comps.to_csv(report.output/'tables'/'A3_hr_component_statistics.csv', index=False)
        component_table = component_table.merge(
            all_comps.assign(metric=all_comps.metric.map(METRIC_NAMES))[['candidate', 'metric', 'p_holm_all']]
            .rename(columns=dict(candidate='configuration')), on=['configuration', 'metric'], how='left')
        component_table['p (Holm, all)'] = component_table.pop('p_holm_all').map(fmt_p)
    report.table(component_table, 'T4_hr_component_ablation',
                 'HR component ablation: change when one component is removed (delta = without - with; negative '
                 'reliability / SQI deltas mean the component helps; Wilcoxon over videos, Holm over all tests)')
    comp_summary = summarize(both, 'version', ['reliable_percent', 'sqi_median'])
    if len(component_table):
        figure, axes = plt.subplots(1, 2, figsize=(10, .32 * len(references) + 1.2), sharey=True)
        labels = [l for l in references if l in set(comp_summary.version)]
        for axis, metric in zip(axes, ('reliable_percent', 'sqi_median')):
            for y, label in enumerate(labels):
                wide = both.pivot_table(index='video_key', columns='version', values=metric, aggfunc='first')
                delta = (wide[label] - wide[references[label]]).dropna()
                estimate, low, high = bootstrap_ci(delta)
                axis.errorbar(estimate, y, xerr=[[estimate - low if np.isfinite(low) else 0], [high - estimate if np.isfinite(high) else 0]],
                              fmt='o', color=COLORS.get(references[label], '#444'), capsize=2)
            axis.axvline(0, color='k', lw=.8)
            axis.set_title(f'delta {METRIC_NAMES[metric]}', fontsize=9)
            axis.grid(axis='x', alpha=.3)
        axes[0].set_yticks(range(len(labels)), labels, fontsize=8)
        axes[0].invert_yaxis()
        figure.tight_layout()
        report.figure(figure, 'F2_hr_component_forest', 'HR component ablation: mean change (95 % bootstrap CI over '
                      'videos) when the component is removed; colour = the full version it is removed from.')

    # Factorial method x strategy.
    report.heading('4. Heart rate: full factorial (pulse method x fusion strategy) and estimator', 2)
    factorial = (quality.groupby(['method', 'strategy']).agg(videos=('video_key', 'nunique'),
                                                             reliable_percent=('reliable_percent', 'mean'),
                                                             sqi_median=('sqi_median', 'mean'),
                                                             snr_median_db=('snr_median_db', 'mean'),
                                                             jump_median=('jump_median', 'mean'),
                                                             plausible_percent=('plausible_percent', 'mean'))
                 .round(2).reset_index())
    report.table(factorial, 'T5_hr_factorial', 'HR: every pulse method x fusion strategy (mean over videos)')
    for metric in ('reliable_percent', 'sqi_median'):
        grid = ordered(factorial.pivot(index='method', columns='strategy', values=metric))
        figure, axis = plt.subplots(figsize=(1.1 * grid.shape[1] + 2, .42 * grid.shape[0] + 1.2))
        image = axis.imshow(grid.values, cmap='viridis', aspect='auto')
        axis.set_xticks(range(grid.shape[1]), grid.columns, rotation=35, ha='right')
        axis.set_yticks(range(grid.shape[0]), grid.index)
        for (i, j), value in np.ndenumerate(grid.values):
            if np.isfinite(value):
                axis.text(j, i, f'{value:.2g}', ha='center', va='center', fontsize=7,
                          color='w' if value < np.nanmean(grid.values) else 'k')
        figure.colorbar(image, ax=axis, label=METRIC_NAMES[metric])
        figure.tight_layout()
        report.figure(figure, f'F3_hr_factorial_{metric}', f'HR factorial ablation: {METRIC_NAMES[metric]} '
                      '(mean over videos) for every pulse method and fusion strategy.')
    rows = []
    for key, _, method, strategy, _ in HR_VERSIONS:
        part = hr[(hr.method == method) & (hr.strategy == strategy)].sort_values(['dataset', 'video_id', 'window_end_sec'])
        if part.empty:
            continue
        row = dict(version=key, configuration=f'{method} + {strategy}')
        for column, name in (('hr_fft_bpm', 'fft'), ('hr_ar_bpm', 'ar_best'), ('hr_kalman_bpm', 'kalman')):
            values = pd.to_numeric(part[column], errors='coerce')
            jumps = part.assign(v=values).groupby(['dataset', 'video_id']).v.apply(lambda s: s.diff().abs().median())
            row[f'{name}_coverage_percent'] = round(100 * values.notna().mean(), 1)
            row[f'{name}_jump_median_bpm'] = round(jumps.mean(), 2) if jumps.notna().any() else None
        difference = (pd.to_numeric(part.hr_fft_bpm, errors='coerce') - pd.to_numeric(part.hr_ar_bpm, errors='coerce')).abs()
        row['fft_vs_ar_median_abs_bpm'] = round(difference.median(), 2) if difference.notna().any() else None
        rows.append(row)
    report.table(pd.DataFrame(rows), 'T6_hr_estimator_ablation',
                 'HR estimator ablation: FFT peak vs AR best model vs SQI-weighted Kalman (multi-site strategies '
                 'report only the FFT consensus, so AR is not available there)')

    # Inter-version agreement.
    rows = []
    key = ['dataset', 'video_id', 'window_end_sec']
    tables = {k: hr[(hr.method == m) & (hr.strategy == s)][key + ['hr_bpm', 'reliable']].rename(
        columns=dict(hr_bpm=f'hr_{k}', reliable=f'rel_{k}')) for k, _, m, s, _ in HR_VERSIONS}
    figure, axes = plt.subplots(1, 3, figsize=(11, 3))
    for axis, (a, b) in zip(axes, (('v1', 'v2'), ('v2', 'v3'), ('v1', 'v3'))):
        if tables[a].empty or tables[b].empty:
            continue
        merged = tables[a].merge(tables[b], on=key).dropna(subset=[f'hr_{a}', f'hr_{b}'])
        difference, mean = merged[f'hr_{b}'] - merged[f'hr_{a}'], (merged[f'hr_{a}'] + merged[f'hr_{b}']) / 2
        both_reliable = merged[f'rel_{a}'].astype(bool) & merged[f'rel_{b}'].astype(bool)
        sd = difference.std(ddof=1) if len(difference) > 1 else np.nan
        rows.append(dict(pair=f'{b} vs {a}', windows=len(merged), bias_bpm=round(difference.mean(), 2),
                         loa_low_bpm=round(difference.mean() - 1.96 * sd, 2), loa_high_bpm=round(difference.mean() + 1.96 * sd, 2),
                         within_5bpm_percent=round(100 * (difference.abs() <= 5).mean(), 1),
                         windows_both_reliable=int(both_reliable.sum()),
                         within_5bpm_both_reliable_percent=round(100 * (difference[both_reliable].abs() <= 5).mean(), 1)
                         if both_reliable.any() else None))
        axis.scatter(mean, difference, s=4, alpha=.4, color=COLORS[b])
        for value, style in ((difference.mean(), '-'), (difference.mean() - 1.96 * sd, '--'), (difference.mean() + 1.96 * sd, '--')):
            if np.isfinite(value):
                axis.axhline(value, color='k', ls=style, lw=.8)
        axis.set_xlabel(f'mean of {a} and {b} (bpm)')
        axis.set_ylabel(f'{b} - {a} (bpm)')
        axis.grid(alpha=.3)
    figure.tight_layout()
    report.figure(figure, 'F4_hr_version_agreement', 'Agreement between versions (Bland-Altman style, all windows): '
                  'bias and 95 % limits of agreement. This measures how much the versions differ, not accuracy.')
    report.table(pd.DataFrame(rows), 'T7_hr_version_agreement', 'HR agreement between versions (window level)')
    return quality, versions


def section_neural(report, data):
    plt = plt_setup()
    report.heading('5. Neural rPPG: pre-training, adaptation and cost', 2)
    histories, meta, pretrain = data['histories'], data['neural_meta'], data['pretrain_meta']
    from module2.neural import NeuralConfig, build_model
    model = build_model(NeuralConfig())
    parameters = sum(p.numel() for p in model.parameters())
    report.text(f'PulseNet has {parameters:,} parameters. The source model is pre-trained without labels on pilot '
                'videos of other infants; every target video is excluded from the source set of its own fold '
                '(SHA-256). Adaptation runs on the target video alone, without reference HR.')
    if len(pretrain):
        report.table(pretrain, 'T8_neural_pretraining', 'Neural pre-training folds (source videos, excluded targets, minutes)')
    if len(meta):
        cost = (meta.groupby('dataset').agg(videos=('video_id', 'nunique'), frames_total=('frames', 'sum'),
                                            minutes_total=('minutes', 'sum'), minutes_per_video=('minutes', 'mean'))
                .round(2).reset_index())
        report.table(cost, 'T9_neural_adaptation_cost', 'Neural adaptation cost (all variants of a video together)')
    if len(histories):
        figure, axis = plt.subplots(figsize=(6.5, 3))
        for variant, part in histories.groupby('variant'):
            curve = part.groupby('step').loss.agg(['mean', 'std'])
            smooth = curve['mean'].rolling(10, min_periods=1).mean()
            axis.plot(curve.index, smooth, label=variant, lw=1.3)
            axis.fill_between(curve.index, smooth - curve['std'], smooth + curve['std'], alpha=.08)
        axis.set_xlabel('adaptation step')
        axis.set_ylabel('unsupervised loss')
        axis.legend(fontsize=7)
        axis.grid(alpha=.3)
        figure.tight_layout()
        report.figure(figure, 'F5_neural_adaptation_curves', 'Test-time adaptation loss (mean over videos, smoothed; band '
                      '= 1 SD) for every adaptation variant.')


def section_rr(report, data):
    plt = plt_setup()
    rr = data['rr']
    report.heading('6. Respiration rate and thoraco-abdominal phase', 2)
    if rr.empty:
        report.text('No RR output.')
        return None
    quality = per_video_quality(rr, 'rr_brpm', RR_RANGE)
    quality = quality.merge(rr[['dataset', 'video_id', 'video_key']].drop_duplicates(), on=['dataset', 'video_id'])
    quality.to_csv(report.output/'tables'/'A4_rr_quality_per_video_all_configurations.csv', index=False)
    report.text('RR is identical in v2 and v3; v2 adds the thoraco-abdominal (TAA) phase to the v1 RR. The RR '
                'ablation therefore covers the breathing signal (FLOW = camera-compensated optical flow, INTENSITY = '
                'grey level) and the ROI fusion strategy.')
    factorial = (quality.groupby(['method', 'strategy']).agg(videos=('video_key', 'nunique'),
                                                             reliable_percent=('reliable_percent', 'mean'),
                                                             sqi_median=('sqi_median', 'mean'),
                                                             snr_median_db=('snr_median_db', 'mean'),
                                                             jump_median=('jump_median', 'mean'),
                                                             plausible_percent=('plausible_percent', 'mean'))
                 .round(2).reset_index())
    report.table(factorial, 'T10_rr_factorial', 'RR: breathing signal x fusion strategy (mean over videos)')
    per = quality.assign(version=quality.method + ' + ' + quality.strategy)
    baseline = 'FLOW + dynamic_fusion'
    others = [v for v in per.version.unique() if v != baseline]
    report.table(stats_table(compare(per, baseline, others, ['reliable_percent', 'sqi_median', 'jump_median'])),
                 'T11_rr_statistics', f'RR: paired Wilcoxon tests of every configuration against {baseline}')
    primary = rr[(rr.method == 'FLOW') & (rr.strategy == 'dynamic_fusion')]
    if 'taa_phase_deg' in primary:
        taa = primary.groupby(['dataset', 'video_id']).apply(lambda g: pd.Series(dict(
            windows=len(g), taa_measured_percent=100 * g.taa_phase_deg.notna().mean(),
            taa_trusted_percent=100 * (g.taa_coherence >= .5).mean(),
            taa_phase_median_deg=g.taa_phase_deg[g.taa_coherence >= .5].median()))).reset_index()
        taa['pattern'] = pd.cut(taa.taa_phase_median_deg, [-1, 45, 135, 181], labels=['synchronous', 'asynchronous', 'paradoxical'])
        report.table(taa.round(1), 'T12_rr_taa_per_video', 'Thoraco-abdominal phase per video (v2 / v3; trusted = '
                     'coherence >= 0.5)')
        summary = pd.DataFrame([dict(videos=len(taa), videos_with_trusted_phase=int((taa.taa_trusted_percent > 0).sum()),
                                     trusted_window_percent_mean=round(taa.taa_trusted_percent.mean(), 1),
                                     **{f'videos_{k}': int((taa.pattern == k).sum()) for k in ('synchronous', 'asynchronous', 'paradoxical')})])
        report.table(summary, 'T13_rr_taa_summary', 'Thoraco-abdominal phase coverage (what v2 adds to v1)')
    grid = ordered(factorial.pivot(index='method', columns='strategy', values='reliable_percent'))
    figure, axis = plt.subplots(figsize=(7, 2.2))
    image = axis.imshow(grid.values, cmap='viridis', aspect='auto')
    axis.set_xticks(range(grid.shape[1]), grid.columns, rotation=30, ha='right')
    axis.set_yticks(range(grid.shape[0]), grid.index)
    for (i, j), value in np.ndenumerate(grid.values):
        axis.text(j, i, f'{value:.2g}', ha='center', va='center', fontsize=7, color='w')
    figure.colorbar(image, ax=axis, label='reliable windows (%)')
    figure.tight_layout()
    report.figure(figure, 'F6_rr_factorial', 'RR ablation: reliable windows (%) for every breathing signal and fusion strategy.')
    return quality


def section_events(report, data):
    report.heading('7. Apnoea, bradycardia and desaturation events (v3)', 2)
    summary, events = data['event_summary'], data['events']
    if summary.empty:
        report.text('No event output.')
        return
    columns = ['duration_sec', 'hr_assessable_sec', 'breath_assessable_sec', 'taa_assessable_sec',
               'spo2_proxy_assessable_sec', 'apnoea_events', 'bradycardia_events', 'desaturation_events',
               'paradoxical_breathing_events', 'periodic_breathing_events', 'abd_clusters']
    columns = [c for c in columns if c in summary]
    per_dataset = summary.groupby('dataset')[columns].sum().reset_index()
    total = summary[columns].sum().to_frame().T.assign(dataset='all')
    table = pd.concat([per_dataset, total], ignore_index=True)
    for channel in ('hr', 'breath', 'taa', 'spo2_proxy'):
        if f'{channel}_assessable_sec' in table:
            table[f'{channel}_assessable_percent'] = (100 * table[f'{channel}_assessable_sec'] / table.duration_sec).round(1)
    report.text('Every channel is either assessable or not at each time point (0.5 s grid); events are only searched '
                'in assessable time. The desaturation channel is an uncalibrated red/blue ratio-of-ratios trend.')
    report.table(table.round(1), 'T14_events_real', 'Events on real video: assessable time per channel and event counts')
    if len(events):
        report.table(events.drop(columns=['video_key'], errors='ignore').round(2), 'A5_events_list',
                     'All detected events', latex=False)


def section_synthetic(report, synthetic):
    plt = plt_setup()
    report.heading('8. Controlled synthetic benchmarks (known ground truth)', 2)
    report.text('Each benchmark isolates failure modes with known ground truth (scripts/benchmark_v123_synthetic.py). '
                'They show whether a component handles that failure mode, not its accuracy on real NICU video.')
    if synthetic is None:
        report.text('Synthetic benchmarks were not run.')
        return
    path = synthetic/'S1_mechanisms.json'
    if path.is_file():
        rows = []
        for scenario, result in json.loads(path.read_text()).items():
            if all(isinstance(v, dict) for v in result.values()):
                for method, values in result.items():
                    rows.append(dict(scenario=scenario, method=method, **values))
            else:
                flat = {k: v for k, v in result.items() if not isinstance(v, dict)}
                for method, values in ((k, v) for k, v in result.items() if isinstance(v, dict)):
                    rows.append(dict(scenario=scenario, method=method, **values, **flat))
                if not any(isinstance(v, dict) for v in result.values()):
                    rows.append(dict(scenario=scenario, method='v2', **flat))
        report.table(pd.DataFrame(rows), 'T15_synthetic_S1_v2_mechanisms',
                     'S1: v2 mechanisms (300 trials per scenario; MAE in bpm or breaths/min, within5 = % within 5)')
    path = synthetic/'S2_neural_summary.csv'
    if path.is_file():
        summary = pd.read_csv(path).sort_values('mae_bpm').round(2)
        per_target = pd.read_csv(synthetic/'S2_neural_per_target.csv')
        wide = per_target.pivot(index='target', columns='method', values='mae_bpm')
        rows = []
        for method in wide:
            if method != 'NEURAL_TTA':
                result = paired_test(wide['NEURAL_TTA'], wide[method])
                rows.append(dict(method=method, n=result['n'], mae_minus_tta_bpm=round(result['mean_difference'], 2),
                                 rank_biserial=round(result['rank_biserial'], 3), p=fmt_p(result['p_value'])))
        report.table(summary, 'T16_synthetic_S2_neural', 'S2: HR error under domain shift (window level, all targets)')
        report.table(pd.DataFrame(rows), 'T17_synthetic_S2_neural_tests',
                     'S2: per-target MAE of each method minus NEURAL_TTA (positive = NEURAL_TTA better; Wilcoxon over targets)')
        figure, axis = plt.subplots(figsize=(7, 2.8))
        order = summary.method.tolist()
        axis.boxplot([wide[m].dropna() for m in order], showfliers=False)
        axis.set_xticks(range(1, len(order) + 1), order, rotation=30, ha='right', fontsize=7)
        axis.set_ylabel('MAE per target (bpm)')
        axis.grid(axis='y', alpha=.3)
        figure.tight_layout()
        report.figure(figure, 'F7_synthetic_neural', 'S2: per-target HR MAE under domain shift for every pulse method.')
    path = synthetic/'S3_events.csv'
    if path.is_file():
        events = pd.read_csv(path).round(3)
        report.table(events, 'T18_synthetic_S3_events', 'S3: event detection with one safeguard removed '
                     '(event-level sensitivity, PPV, F1, false alarms per hour)')
        figure, axes = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
        configs = events.configuration.unique()
        for axis, kind in zip(axes, ('apnoea', 'bradycardia', 'desaturation')):
            part = events[events.event_type == kind].set_index('configuration').reindex(configs)
            y = np.arange(len(configs))
            axis.barh(y - .2, part.sensitivity.fillna(0), .4, label='sensitivity', color='#4c72b0')
            axis.barh(y + .2, part.ppv.fillna(0), .4, label='PPV', color='#dd8452')
            axis.set_title(kind)
            axis.set_xlim(0, 1.05)
            axis.grid(axis='x', alpha=.3)
        axes[0].set_yticks(np.arange(len(configs)), configs, fontsize=7)
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=7)
        figure.tight_layout()
        report.figure(figure, 'F8_synthetic_events', 'S3: event detection sensitivity and PPV for the full detector and '
                      'with one safeguard removed.')


def section_accuracy(report, data, references):
    plt = plt_setup()
    report.heading('9. Accuracy against reference measurements', 2)
    if not any(references.values()):
        report.text('No reference HR, RR or events were given, so accuracy (MAE, RMSE, Bland-Altman, correlation, ICC) '
                    'is not reported. For publication, record ECG / pulse oximetry (HR, SpO2) and impedance '
                    'pneumography (RR, apnoea) synchronously with the video and rerun with REFERENCE_HR / REFERENCE_RR '
                    '/ REFERENCE_EVENTS; every accuracy table and figure of this section is then produced '
                    'automatically.')
        return
    for kind, column, windows, definitions in (
            ('HR', 'hr_bpm', data['hr'], [(k, m, s) for k, _, m, s, _ in HR_VERSIONS] + [(l, m, s) for l, m, s, _, _ in HR_COMPONENTS]),
            ('RR', 'rr_brpm', data['rr'], [('v1/v2/v3', 'FLOW', 'dynamic_fusion')] +
             [(f'{m} + {s}', m, s) for m in ('FLOW', 'INTENSITY') for s in ('dynamic_fusion', 'confidence_fusion', 'snr_fusion', 'top1', 'pool_mean', 'infant_box')])):
        if not references.get(kind) or windows.empty:
            continue
        reference = pd.read_csv(references[kind])
        reference['video_id'] = reference.video_id.astype(str)
        pairs = pair_reference(windows, reference, column)
        pairs = pairs[pairs.reference.notna()]
        if pairs.empty:
            report.text(f'{kind}: no window overlaps the reference (check dataset, video_id and timestamps).')
            continue
        pairs.to_csv(report.output/'tables'/f'A6_{kind.lower()}_reference_pairs.csv', index=False)
        estimate = column
        kalman = 'hr_kalman_bpm' if kind == 'HR' else 'rr_kalman_brpm'
        rows, per_video = [], []
        for label, method, strategy in definitions:
            part = pairs[(pairs.method == method) & (pairs.strategy == strategy)]
            if part.empty:
                continue
            for subset, chosen in (('all windows', part), ('reliable windows', part[part.reliable.astype(bool)])):
                for name in (estimate, kalman):
                    if name in chosen:
                        result = accuracy(chosen, name)
                        rows.append(dict(configuration=label, method=method, strategy=strategy, subset=subset, estimate=name,
                                         coverage_percent=round(100 * len(chosen) / len(part), 1),
                                         **{k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}))
            for video, group in part.groupby('video_key'):
                per_video.append(dict(version=label, video_key=video,
                                      mae=float((group[estimate] - group.reference).abs().mean())))
        table = pd.DataFrame(rows)
        report.table(table, f'T19_{kind.lower()}_accuracy', f'{kind} accuracy against the reference (window level)')
        per_video = pd.DataFrame(per_video)
        if kind == 'HR' and len(per_video):
            comparison = pd.concat([compare(per_video, 'v1', ['v2', 'v3'], ['mae']), compare(per_video, 'v2', ['v3'], ['mae']),
                                    *[compare(per_video, ref, [label], ['mae']) for label, _, _, ref, _ in HR_COMPONENTS]])
            report.table(stats_table(comparison), 'T20_hr_accuracy_statistics',
                         'HR: paired Wilcoxon tests of per-video MAE (negative difference = candidate more accurate)')
            figure, axes = plt.subplots(1, 4, figsize=(13, 3))
            for axis, (key, _, method, strategy, _) in zip(axes, HR_VERSIONS):
                part = pairs[(pairs.method == method) & (pairs.strategy == strategy)].dropna(subset=[estimate])
                if part.empty:
                    continue
                difference, mean = part[estimate] - part.reference, (part[estimate] + part.reference) / 2
                sd = difference.std(ddof=1)
                axis.scatter(mean, difference, s=4, alpha=.4, color=COLORS[key])
                for value, style in ((difference.mean(), '-'), (difference.mean() - 1.96 * sd, '--'), (difference.mean() + 1.96 * sd, '--')):
                    axis.axhline(value, color='k', ls=style, lw=.8)
                axis.set_title(f'{key}: bias {difference.mean():.1f}, LoA +-{1.96 * sd:.1f} bpm', fontsize=8)
                axis.set_xlabel('mean of camera and reference (bpm)')
                axis.grid(alpha=.3)
                curve = coverage_accuracy(part, estimate)
                axes[3].plot(curve.coverage_percent, curve.mae, 'o-', color=COLORS[key], ms=3, label=key)
            axes[0].set_ylabel('camera - reference (bpm)')
            axes[3].set_xlabel('coverage (%)')
            axes[3].set_ylabel('MAE (bpm)')
            axes[3].legend(fontsize=7)
            axes[3].set_title('reliability-accuracy trade-off (SQI threshold)', fontsize=8)
            figure.tight_layout()
            report.figure(figure, 'F9_hr_bland_altman_coverage', 'HR against the reference: Bland-Altman per version and '
                          'MAE vs coverage when windows below an SQI threshold are discarded.')
    if references.get('events') and not data['events'].empty:
        reference = pd.read_csv(references['events'])
        reference['video_id'] = reference.video_id.astype(str)
        keys = ['dataset', 'video_id'] if 'dataset' in reference else ['video_id']
        detected = data['events'][data['events'].get('clinically_significant', True).astype(bool)]
        tag = lambda frame: frame.assign(video_id=frame[keys].astype(str).agg('/'.join, axis=1)).to_dict('records')
        rows = [match_events(tag(detected), tag(reference), kind) for kind in ('apnoea', 'bradycardia', 'desaturation')]
        report.table(pd.DataFrame(rows), 'T21_events_accuracy', 'Event detection against the reference events')


def section_reproducibility(report, datasets, data):
    rows = []
    folders = (('v1 notebook', 'hr_rppg_sam2', 'POS', 'dynamic_fusion'), ('v2 notebook', 'hr_updated', 'POS', 'cardiorespiratory_fusion'),
               ('v3 notebook', 'hr_v3', 'NEURAL_TTA', 'cardiorespiratory_fusion'))
    hr = data['hr']
    for d in datasets:
        if not d.get('run_dir'):
            continue
        for name, folder, method, strategy in folders:
            path = Path(d['run_dir'])/'outputs'/folder/'hr_windows.csv'
            if not path.is_file():
                continue
            old = pd.read_csv(path, low_memory=False)
            old = old[(old.method == method) & (old.strategy == strategy)].assign(video_id=lambda f: f.video_id.astype(str))
            new = hr[(hr.dataset == d['name']) & (hr.method == method) & (hr.strategy == strategy)]
            merged = old.merge(new, on=['video_id', 'window_end_sec'], suffixes=('_notebook', '_ablation'))
            difference = (merged.hr_bpm_notebook - merged.hr_bpm_ablation).abs()
            rows.append(dict(dataset=d['name'], source=name, configuration=f'{method} + {strategy}', windows=len(merged),
                             median_abs_difference_bpm=round(difference.median(), 3) if len(merged) else None,
                             identical_percent=round(100 * (difference < .01).mean(), 1) if len(merged) else None))
    if rows:
        report.heading('10. Reproducibility of the individual notebooks', 2)
        report.text('The window HR of the stand-alone v1 / v2 / v3 notebooks compared with the same configuration in '
                    'this ablation run. Differences come from code changes after a notebook ran (v1) or from a '
                    'different source model (v3: each ablation fold pre-trains its own model).')
        report.table(pd.DataFrame(rows), 'T22_reproducibility', 'Stand-alone notebooks vs ablation run (same configuration)')


def example_figure(report, data):
    plt = plt_setup()
    hr, rr = data['hr'], data['rr']
    if hr.empty:
        return
    longest = hr.groupby(['dataset', 'video_id']).window_end_sec.max().idxmax()
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for key, _, method, strategy, _ in HR_VERSIONS:
        part = hr[(hr.dataset == longest[0]) & (hr.video_id == longest[1]) & (hr.method == method) & (hr.strategy == strategy)]
        top.plot(part.window_end_sec, part.hr_bpm, color=COLORS[key], lw=1.4, label=f'{key} ({method}, {strategy})')
        bad = part[~part.reliable.astype(bool)]
        top.plot(bad.window_end_sec, bad.hr_bpm, 'x', color=COLORS[key], ms=3)
    top.set_ylabel('HR (bpm)')
    top.legend(fontsize=7)
    top.grid(alpha=.3)
    part = rr[(rr.dataset == longest[0]) & (rr.video_id == longest[1]) & (rr.method == 'FLOW') & (rr.strategy == 'dynamic_fusion')]
    bottom.plot(part.window_end_sec, part.rr_brpm, color='#55a868', lw=1.4, label='RR (FLOW, dynamic fusion)')
    bottom.set_ylabel('RR (breaths/min)')
    bottom.set_xlabel('time (s, window end)')
    bottom.grid(alpha=.3)
    figure.tight_layout()
    report.figure(figure, 'F10_example_timeline', f'Example: longest video ({longest[0]} / {longest[1]}); x = window '
                  'below the reliability threshold.')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the v1/v2/v3 ablation report.')
    parser.add_argument('--datasets', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--synthetic-dir')
    parser.add_argument('--timings')
    parser.add_argument('--reference-hr')
    parser.add_argument('--reference-rr')
    parser.add_argument('--reference-events')
    args = parser.parse_args(argv)
    datasets = json.loads(Path(args.datasets).read_text())
    output = Path(args.output_dir)
    report = Report(output)
    data = load(datasets)
    references = dict(HR=args.reference_hr, RR=args.reference_rr, events=args.reference_events)

    report.text(f'Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by scripts/report_v123_ablation.py '
                f'(Python {platform.python_version()}). Versions: v1 = RUN_VITALS_NEW_VIDEOS(v1).ipynb, '
                'v2 = Updated HR RR(v2).ipynb, v3 = Updated HR RR(v3).ipynb.')
    report.heading('0. Versions compared')
    report.table(pd.DataFrame([dict(version=k, description=d, pulse_method=m, fusion_strategy=s) for k, d, m, s, _ in HR_VERSIONS] +
                              [dict(version='v2 / v3 RR', description='RR with thoraco-abdominal phase (v1: without)',
                                    pulse_method='FLOW', fusion_strategy='dynamic_fusion'),
                               dict(version='v3 events', description='apnoea / bradycardia / desaturation events',
                                    pulse_method='POS (extended band)', fusion_strategy='multi-site consensus')]),
                 'T0_versions', 'Pipeline versions in the ablation study')
    section_data(report, data)
    section_hr(report, data, args)
    section_neural(report, data)
    section_rr(report, data)
    section_events(report, data)
    section_synthetic(report, Path(args.synthetic_dir) if args.synthetic_dir else None)
    section_accuracy(report, data, references)
    section_reproducibility(report, datasets, data)
    example_figure(report, data)
    if args.timings and Path(args.timings).is_file():
        timings = json.loads(Path(args.timings).read_text())
        report.heading('11. Runtime', 2)
        report.table(pd.DataFrame([dict(stage=k, minutes=round(v, 2)) for k, v in timings.items()]), 'T23_runtime',
                     'Runtime per stage of this ablation run (minutes)')
    report.heading('12. Limitations', 2)
    report.text('1. No reference HR, RR, SpO2 or event annotations were available unless Section 9 shows them; '
                'reference-free metrics (SQI, SNR, reliability, stability) can favour a method that produces '
                'spectrally clean but wrong signals. In particular the neural models are trained with spectral '
                'concentration losses that the SQI also rewards, so a higher SQI of v3 is not evidence of higher '
                'accuracy.\n\n'
                '2. Infant identity is not recorded in the manifests; videos of the same infant may fall into the '
                'pre-training source and the target set, which can inflate the neural results.\n\n'
                '3. Many videos are short public clips with editing, overlays and variable lighting; they are not '
                'continuous NICU recordings, and clips shorter than 20 s cannot contain an apnoea of >= 20 s.\n\n'
                '4. The desaturation channel is an uncalibrated RGB ratio-of-ratios trend, not an SpO2 value.\n\n'
                '5. Synthetic benchmarks isolate failure modes under controlled conditions and do not replace a '
                'clinical validation.')
    report.write('Ablation study: camera heart rate and respiration v1 vs v2 vs v3')
    files = sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())
    (output/'report_files.json').write_text(json.dumps(files, indent=1))
    print(f'Report: {output/"REPORT.md"} and REPORT.html; {len(files)} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
