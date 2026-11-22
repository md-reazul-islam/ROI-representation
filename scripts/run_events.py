"""Module 2 (v3): combined apnoea, bradycardia and desaturation (ABD) event detection from the camera vitals.

Uses the outputs of run_hr_rppg.py (--hr-dir: rgb_traces.csv, pulse_signals.csv) and run_rr.py (--rr-dir:
resp_signals.csv, rr_windows.csv); see module2.events for the channels, definitions and limits. Outputs:
  event_timeline.csv   per video and grid point: extended-band HR, breathing envelope, phase, SpO2 proxy, flags
  events.csv           one row per event (apnoea, bradycardia, desaturation, paradoxical / periodic breathing)
  event_clusters.csv   ABD clusters (label A, B, D, AB, AD, BD, ABD)
  event_summary.csv    per video: assessable time per channel, event counts and rates per assessable hour
  plots/<video>_events.png
With --reference-events (CSV: video_id, event_type, start_sec, end_sec from bedside monitoring), event-level
sensitivity and PPV are written to event_evaluation.csv. No reference is used for detection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd

from module1.config import load_config, project_path
from module1.data.manifest import sha256_file
from module2.events import EVENT_TYPES, EventConfig, detect_events, match_events, summarize_clusters, timeline
from module2.rppg import HRConfig


def build_config(section, cls, overrides=None):
    values = {field.name: section[field.name] for field in fields(cls) if field.name in section}
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    for key, value in values.items():
        if isinstance(value, list):
            values[key] = tuple(value)
    return cls(**values)


def video_inputs(video, traces, pulses, resp, rr, rr_method):
    """(rgb (n, k, 3), weights (n, k), ROI names, fps, waveform, gross, rr windows) of one video."""
    t = traces[traces.video_id == video].sort_values('frame_index')
    p = pulses[pulses.video_id == video].sort_values('frame_index')
    pool = [c[len('weight_'):] for c in p.columns if c.startswith('weight_')]
    rgb = np.stack([t[[f'{c}_{name}' for c in 'rgb']].to_numpy(float) for name in pool], axis=1)
    weights = p[[f'weight_{name}' for name in pool]].to_numpy(float)
    times = t.timestamp_sec.to_numpy(float)
    fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 30.
    r = resp[resp.video_id == video].sort_values('frame_index')
    windows = rr[(rr.video_id == video) & (rr.method == rr_method) & (rr.strategy == 'dynamic_fusion')]
    return (rgb, weights, pool, fps, r[f'resp_{rr_method}_dynamic_fusion'].to_numpy(float),
            r.gross_motion_box_per_sec.to_numpy(float), windows.reset_index(drop=True))


def summarize(video, line, events, cfg):
    step = cfg.step_sec
    row = dict(video_id=video, duration_sec=len(line) * step,
               hr_assessable_sec=line.hr_assessable.sum() * step, breath_assessable_sec=line.breath_assessable.sum() * step,
               taa_assessable_sec=line.taa_assessable.sum() * step,
               spo2_proxy_assessable_sec=line.spo2_proxy_assessable.sum() * step,
               hr_implausible_baseline_sec=line.hr_implausible_baseline.sum() * step,
               hr_median_bpm=line.hr_bpm[line.hr_assessable].median(),
               taa_phase_median_deg=line.taa_phase_deg[line.taa_assessable].median())
    assessable = dict(apnoea='breath', bradycardia='hr', desaturation='spo2_proxy',
                      paradoxical_breathing='taa', periodic_breathing='breath')
    for kind, channel in assessable.items():
        found = [e for e in events if e['event_type'] == kind and e.get('clinically_significant', True)]
        hours = row[f'{channel}_assessable_sec'] / 3600.
        row[f'{kind}_events'] = len(found)
        row[f'{kind}_per_hour'] = len(found) / hours if hours > 0 else None
    row['abd_clusters'] = len({e['cluster_id'] for e in events if e.get('clinically_significant') and 'cluster_id' in e})
    row['min_assessable_sec_for_apnoea'] = cfg.apnoea_min_sec
    return row


def plot_video(target, video, line, events, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
    t = line.time_sec
    colors = dict(apnoea='#4c72b0', bradycardia='#c44e52', desaturation='#8172b3', paradoxical_breathing='#dd8452',
                  periodic_breathing='#55a868')
    rows = dict(bradycardia=0, apnoea=1, periodic_breathing=1, paradoxical_breathing=2, desaturation=3)
    axes[0].plot(t, line.hr_bpm.where(line.hr_assessable), color='#c44e52', lw=1.4, label='HR (assessable)')
    axes[0].plot(t, line.hr_bpm.where(~line.hr_assessable), color='#c44e52', lw=.8, alpha=.3, label='HR (not assessable)')
    axes[0].plot(t, line.hr_baseline_bpm, '--', color='k', lw=1, label='baseline')
    axes[0].axhline(cfg.brady_abs_bpm, color='k', ls=':', lw=1)
    axes[0].set_ylabel('HR (bpm)')
    axes[1].plot(t, line.breath_ratio.where(line.breath_assessable), color='#4c72b0', lw=1.4, label='envelope / baseline')
    axes[1].axhline(cfg.pause_ratio, color='k', ls=':', lw=1)
    axes[1].set_ylabel('breathing')
    axes[2].scatter(t, line.taa_phase_deg, c=line.taa_coherence, cmap='viridis', vmin=0, vmax=1, s=8)
    axes[2].axhline(cfg.paradox_deg, color='k', ls=':', lw=1)
    axes[2].set_ylim(0, 180)
    axes[2].set_ylabel('TAA phase (deg)')
    axes[3].plot(t, line.ror_ratio, color='#8172b3', lw=1.4, label='R / baseline')
    axes[3].axhline(1 + cfg.desat_rise, color='k', ls=':', lw=1)
    axes[3].set_ylabel('SpO2 proxy R')
    axes[3].set_xlabel('time (s)')
    for event in events:
        if event['event_type'] == 'apnoea' and not event['clinically_significant']:
            continue
        axes[rows[event['event_type']]].axvspan(event['start_sec'], event['end_sec'], color=colors[event['event_type']],
                                                alpha=.2, label=event['event_type'])
    for axis in axes:
        missing = ~line.hr_assessable if axis is axes[0] else ~line.breath_assessable if axis in axes[1:3] else \
            ~line.spo2_proxy_assessable
        axis.fill_between(t, 0, 1, where=missing, transform=axis.get_xaxis_transform(), color='#bbbbbb', alpha=.25,
                          lw=0)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize=7, loc='upper right')
        axis.grid(alpha=.3)
    clusters = summarize_clusters(events)
    labels = ', '.join(f'{c["label"]} at {c["start_sec"]:.0f} s' for c in clusters if c['clinically_significant'])
    axes[0].set_title(f'{video}: camera ABD events ({labels or "none"}); grey = not assessable')
    figure.tight_layout()
    figure.savefig(target, dpi=120)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Module 2 v3: apnoea, bradycardia and desaturation events from '
                                                 'the camera HR, RR and thoraco-abdominal phase.')
    parser.add_argument('--config', default=str(ROOT/'configs/module1.yaml'))
    parser.add_argument('--hr-dir', default='outputs/hr_rppg', help='run_hr_rppg.py output')
    parser.add_argument('--rr-dir', default='outputs/rr', help='run_rr.py output')
    parser.add_argument('--output-dir', default='outputs/events')
    parser.add_argument('--rr-method', choices=('FLOW', 'INTENSITY'), default='FLOW')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--reference-events', help='CSV video_id, event_type, start_sec, end_sec (for evaluation only)')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    hr_config = build_config(cfg.get('hr', {}), HRConfig)
    ec = build_config(cfg.get('events', {}), EventConfig)
    hr_dir, rr_dir, output = (project_path(cfg, p) for p in (args.hr_dir, args.rr_dir, args.output_dir))
    inputs = dict(rgb_traces=hr_dir/'rgb_traces.csv', pulse_signals=hr_dir/'pulse_signals.csv',
                  resp_signals=rr_dir/'resp_signals.csv', rr_windows=rr_dir/'rr_windows.csv')
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}. Run run_hr_rppg.py and run_rr.py first.')
    if output.exists() and any(output.rglob('*')):
        raise FileExistsError(f'{output} is not empty; use --output-dir for a new experiment')

    traces = pd.read_csv(inputs['rgb_traces'], low_memory=False)
    pulses = pd.read_csv(inputs['pulse_signals'], usecols=lambda c: c in ('video_id', 'frame_index') or
                         c.startswith('weight_'), low_memory=False)
    resp = pd.read_csv(inputs['resp_signals'], low_memory=False)
    rr = pd.read_csv(inputs['rr_windows'], low_memory=False)
    videos = args.videos or sorted(set(traces.video_id.astype(str)) & set(resp.video_id.astype(str)))
    if not videos:
        raise ValueError('No video has both HR and RR outputs')
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_plots:
        (output/'plots').mkdir()
    lines, events, clusters, summaries = [], [], [], []
    for video in videos:
        line = timeline(*video_inputs(video, traces, pulses, resp, rr, args.rr_method), ec, hr_config)
        found = [dict(e, video_id=video) for e in detect_events(line, ec)]
        lines.append(line.assign(video_id=video))
        events += found
        clusters += [dict(c, video_id=video) for c in summarize_clusters(found)]
        summaries.append(summarize(video, line, found, ec))
        if not args.no_plots:
            plot_video(output/'plots'/f'{video}_events.png', video, line, found, ec)
        significant = [e for e in found if e.get('clinically_significant', True)]
        print(f'{video}: {len(line) * ec.step_sec:.0f} s, assessable HR {summaries[-1]["hr_assessable_sec"]:.0f} s / '
              f'breathing {summaries[-1]["breath_assessable_sec"]:.0f} s, {len(significant)} event(s): '
              f'{", ".join(e["event_type"] for e in significant) or "none"}', flush=True)

    first = ['video_id', 'event_type', 'start_sec', 'end_sec', 'duration_sec', 'cluster_id', 'cluster_label',
             'clinically_significant']
    event_table = pd.DataFrame(events, columns=list(dict.fromkeys(first + [k for e in events for k in e])))
    pd.concat(lines, ignore_index=True).pipe(lambda d: d[['video_id'] + [c for c in d if c != 'video_id']]) \
        .to_csv(output/'event_timeline.csv', index=False)
    event_table.to_csv(output/'events.csv', index=False)
    pd.DataFrame(clusters, columns=list(dict.fromkeys(['video_id', 'cluster_id', 'label', 'start_sec', 'end_sec',
                                                       'clinically_significant'] + [k for c in clusters for k in c]))) \
        .to_csv(output/'event_clusters.csv', index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'event_summary.csv', index=False)

    evaluation = None
    if args.reference_events:
        reference_path = project_path(cfg, args.reference_events)
        reference = pd.read_csv(reference_path)
        reference = reference[reference.video_id.astype(str).isin(videos)].to_dict('records')
        significant = [e for e in events if e.get('clinically_significant', True)]
        evaluation = pd.DataFrame([match_events(significant, reference, kind) for kind in EVENT_TYPES])
        evaluation.to_csv(output/'event_evaluation.csv', index=False)
        print(evaluation.to_string(index=False))

    metadata = dict(phase='MODULE_2_EVENTS', run_status='COMPLETE', videos=videos, event_config=asdict(ec),
                    hr_config=asdict(hr_config), rr_method=args.rr_method,
                    inputs={name: dict(path=str(path), sha256=sha256_file(path)) for name, path in inputs.items()},
                    reference_events=args.reference_events, reference_used_for_detection=False,
                    definitions=dict(
                        bradycardia=f'extended-band ({ec.hr_min_bpm:g}+ bpm) {ec.hr_method} HR < {ec.brady_abs_bpm:g} '
                                    f'bpm or < {1 - ec.brady_rel_drop:.2f} x baseline for >= {ec.brady_min_sec:g} s; '
                                    f'needs >= {ec.min_agreeing_sites} agreeing sites, SQI >= {ec.hr_sqi_threshold:g}, '
                                    f'no RR harmonic and a baseline >= {ec.min_baseline_bpm:g} bpm',
                        apnoea=f'breathing envelope < {ec.pause_ratio:g} x baseline (or no moving ROI) for >= '
                               f'{ec.apnoea_prolonged_sec:g} s, or >= {ec.apnoea_min_sec:g} s with B or D (AAP 2003)',
                        desaturation=f'uncalibrated ratio-of-ratios (red/blue) >= {1 + ec.desat_rise:g} x baseline '
                                     f'for >= {ec.desat_min_sec:g} s',
                        clusters=f'A, B, D starting within {ec.assoc_sec:g} s of the previous one'),
                    limitations=['no reference monitoring: thresholds are literature defaults / heuristics',
                                 'RGB camera has no infrared channel: the SpO2 proxy is a relative trend only',
                                 'a camera sees breathing effort, not airflow: purely obstructive apnoea appears as '
                                 'paradoxical breathing, not as a pause'],
                    timestamp_utc=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version())
    (output/'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    print(summary.to_string(index=False))
    print(f'Done: {len(videos)} video(s) -> {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
