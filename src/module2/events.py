"""Combined apnoea, bradycardia and desaturation (ABD) event detection from the camera vitals (Module 2, v3).

Inputs are the outputs of run_hr_rppg.py (per-ROI skin RGB traces and the dynamic-fusion ROI weights) and
run_rr.py (fused breathing waveform, gross motion, RR windows with the thoraco-abdominal phase). Everything
is evaluated on one time grid (step_sec) and every channel is either assessable or not at every grid point:
time without a usable signal is reported as not assessable, never as "no event".

Channels
  bradycardia   HR re-estimated on an extended band (default 50-270 bpm; the HR branch stops at 90 bpm, below
                most bradycardia thresholds) in short windows with the multi-site consensus of the HR branch
                (module2.cardiorespiratory). Low frequencies collect motion and breathing leakage, so a window is
                assessable only if >= min_agreeing_sites anatomical sites agree, the SQI of the fused pulse is
                >= hr_sqi_threshold and the HR is not an ambiguous respiratory harmonic (k = 1, 2, 3 x RR;
                neonatal breathing reaches the extended band). Bradycardia: HR < brady_abs_bpm or below
                (1 - brady_rel_drop) x the rolling baseline HR, for at least brady_min_sec. Neonatal bradycardias
                are transient drops: where the baseline itself is below min_baseline_bpm, the low HR is far more
                likely a coherent artefact (illumination, camera motion) and the channel is not assessable.
  apnoea        breathing envelope = moving RMS of the fused breathing waveform; a pause is an envelope below
                pause_ratio x its rolling baseline, or an RR window in which no ROI moves at all (FLOW). Gross
                motion makes the breathing channel not assessable. Pause >= apnoea_prolonged_sec, or
                >= apnoea_min_sec with a bradycardia or desaturation (AAP definition), is a clinical apnoea.
                An apnoea is 'central' unless paradoxical chest-abdomen motion (thoraco-abdominal phase >=
                paradox_deg with coherence >= min_taa_coherence) occurs within mixed_window_sec of it ('mixed').
  paradoxical   phase >= paradox_deg (coherent) for >= paradox_min_sec: obstructive pattern suspected (a camera
                sees effort, not airflow, so a purely obstructive apnoea appears as paradoxical breathing).
  periodic      >= periodic_min_pauses pauses of >= pause_min_sec separated by <= periodic_max_gap_sec.
  desaturation  uncalibrated camera proxy: ratio of ratios R = (AC_red/DC_red) / (AC_blue/DC_blue) of the
                pulsatile skin colour (fused over ROIs, only where the pulse is reliable); SpO2 falls as R rises.
                Desaturation: R >= (1 + desat_rise) x its rolling baseline for >= desat_min_sec. RGB cameras
                lack the infrared channel of a pulse oximeter, so this is a relative trend, not an SpO2 value.
Clusters: every apnoea of >= apnoea_min_sec opens an event; a bradycardia or desaturation starting within
assoc_sec of the end of the current event joins it (an apnoea joins only an apnoea-free event it overlaps), so
two apnoeas are never merged. The cluster label lists its components (A, B, D, AB, AD, BD, ABD).
All thresholds are literature defaults or heuristics; none has been validated against reference monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import pandas as pd

from module2.cardiorespiratory import CardioRespConfig, consensus_hr, roi_spectrum
from module2.rppg import (HRConfig, _usable, bandpass, detrend_smoothness_priors, extract_pulse, segments,
                          signal_quality, snr_db)


EVENT_TYPES = ('apnoea', 'bradycardia', 'desaturation')
LETTERS = dict(apnoea='A', bradycardia='B', desaturation='D')


@dataclass(frozen=True)
class EventConfig:
    step_sec: float = .5
    # Extended-band HR for bradycardia.
    hr_min_bpm: float = 50.
    hr_window_sec: float = 5.
    hr_method: str = 'POS'
    hr_sqi_threshold: float = .5
    min_agreeing_sites: int = 2
    # Bradycardia: < 100 bpm (most NICU protocols) or a drop of >= 33 % below baseline, for >= 4 s.
    brady_abs_bpm: float = 100.
    brady_rel_drop: float = .33
    brady_min_sec: float = 4.
    min_baseline_bpm: float = 100.
    harmonic_halfwidth_bpm: float = 4.
    rr_harmonics: tuple = (1, 2, 3)
    # Rolling baselines (centred median over this span; whole-video median if less is assessable).
    baseline_sec: float = 120.
    baseline_min_sec: float = 5.
    # Breathing pauses and apnoea (AAP 2003: >= 20 s, or shorter with bradycardia / desaturation).
    envelope_sec: float = 3.
    pause_ratio: float = .25
    pause_min_sec: float = 3.
    apnoea_min_sec: float = 10.
    apnoea_prolonged_sec: float = 20.
    gross_motion_box_per_sec: float = .1
    # Periodic breathing (>= 3 pauses of >= 3 s separated by <= 20 s of breathing).
    periodic_min_pauses: int = 3
    periodic_max_gap_sec: float = 20.
    # Thoraco-abdominal phase.
    paradox_deg: float = 135.
    min_taa_coherence: float = .5
    paradox_min_sec: float = 10.
    mixed_window_sec: float = 10.
    # Desaturation proxy (uncalibrated).
    desat_window_sec: float = 8.
    desat_rise: float = .25
    desat_min_sec: float = 10.
    # B / D starting within this time of the end of an apnoea (or of each other) belong to one event.
    assoc_sec: float = 30.
    # Runs of a flag separated by at most this much time are merged.
    merge_gap_sec: float = 1.


def runs(flag, step, merge_gap_sec=0., min_sec=0.):
    """(start_index, stop_index) of True runs, merging runs separated by <= merge_gap_sec, keeping >= min_sec."""
    out = []
    for start, stop in segments(np.asarray(flag, bool), 1):
        if out and (start - out[-1][1]) * step <= merge_gap_sec + 1e-9:
            out[-1] = (out[-1][0], stop)
        else:
            out.append((start, stop))
    return [(a, b) for a, b in out if (b - a) * step >= min_sec - 1e-9]


def rolling_median(values, valid, step, span_sec, min_sec):
    """Centred rolling median of values[valid] over span_sec; whole-series median where too little is valid."""
    values = np.where(valid, values, np.nan)
    overall = float(np.nanmedian(values)) if valid.sum() * step >= min_sec else np.nan
    half = max(1, int(round(span_sec / 2 / step)))
    minimum = max(1, int(round(min_sec / step)))
    out = np.full(len(values), overall)
    for i in range(len(values)):
        part = values[max(0, i - half):i + half + 1]
        if np.isfinite(part).sum() >= minimum:
            out[i] = np.nanmedian(part)
    return out


def fused_pulse(pulses, weights):
    """Weighted mean of per-ROI pulses (n, k) with weights (n, k) or (k,); NaN where no ROI is usable."""
    weights = np.broadcast_to(weights, pulses.shape)
    finite = np.isfinite(pulses)
    w = np.where(finite, np.nan_to_num(weights, nan=0.), 0.)
    total = w.sum(axis=1)
    return np.divide((np.where(finite, pulses, 0.) * w).sum(axis=1), total, out=np.full(len(total), np.nan),
                     where=total > 0)


def window_bounds(t, fps, n, window_sec):
    """Sample bounds of a window of window_sec centred on t, or None if it leaves the clip."""
    start = int(round((t - window_sec / 2) * fps))
    stop = start + int(round(window_sec * fps))
    return (start, stop) if start >= 0 and stop <= n else None


def ratio_of_ratios(rgb, weights, fps, bounds, hr_config):
    """Weighted geometric mean over ROIs of (AC_r/DC_r)/(AC_b/DC_b) of the pulsatile colour in one window."""
    start, stop = bounds
    w = np.nanmean(weights[start:stop], axis=0)
    logs, used = [], []
    for r in range(rgb.shape[1]):
        part = rgb[start:stop, r]
        if not (w[r] > 0) or not np.isfinite(part).all():
            continue
        dc = part.mean(axis=0)
        if (dc <= 0).any():
            continue
        ac = [np.std(bandpass(detrend_smoothness_priors(part[:, c] / dc[c], hr_config.detrend_lambda), fps,
                              hr_config)) for c in (0, 2)]
        if min(ac) > 0:
            logs.append(math.log(ac[0] / ac[1]))
            used.append(w[r])
    return math.exp(np.average(logs, weights=used)) if logs else np.nan


def sample_windows(table, times, column, max_offset):
    """Value of `column` in the row whose window_center_sec is nearest to each time (NaN beyond max_offset)."""
    out = np.full(len(times), np.nan)
    if table is None or table.empty:
        return out
    centres = table.window_center_sec.to_numpy(float)
    values = table[column].to_numpy(float)
    for i, t in enumerate(times):
        j = int(np.argmin(np.abs(centres - t)))
        if abs(centres[j] - t) <= max_offset:
            out[i] = values[j]
    return out


def moving_rms(x, fps, window_sec):
    """Centred moving RMS of a signal with NaN gaps (NaN where fewer than half the samples are finite)."""
    length = max(1, int(round(window_sec * fps)))
    finite = np.isfinite(x)
    kernel = np.ones(length)
    energy = np.convolve(np.where(finite, x, 0.) ** 2, kernel, mode='same')
    count = np.convolve(finite.astype(float), kernel, mode='same')
    return np.where(count >= length / 2, np.sqrt(energy / np.maximum(count, 1)), np.nan)


def window_hr(pulses, weights, names, fps, bounds, rr_bpm, hr_config, cr):
    """(HR, SQI, agreeing sites, harmonic ambiguous) of one window by multi-site consensus."""
    start, stop = bounds
    w = np.nan_to_num(np.nanmean(weights[start:stop], axis=0), nan=0.)
    spectra = [roi_spectrum(pulses[start:stop, r], fps, hr_config) if w[r] > 0 else None for r in range(len(names))]
    result = consensus_hr(spectra, names, w, fps, hr_config, cr, None if math.isnan(rr_bpm) else rr_bpm)
    bpm = result['hr_bpm']
    clean = _usable(fused_pulse(pulses[start:stop], w), hr_config)
    if bpm is None or clean is None:
        return np.nan, np.nan, 0, False
    sqi = signal_quality(float(snr_db(clean, fps, bpm, hr_config)), None, hr_config)
    return bpm, sqi, result['consensus_support'], result['harmonic_ambiguous']


def timeline(rgb, weights, names, fps, waveform, gross, rr_windows, cfg, hr_config=None):
    """Per grid point: all channels, their assessability and the per-channel event flags.

    rgb (n, k, 3) skin RGB per ROI (names: the k ROI names) and weights (n, k) dynamic-fusion weights from
    run_hr_rppg.py;
    waveform (m,) fused breathing signal and gross (m,) gross motion (box heights / s) from run_rr.py;
    rr_windows: FLOW / dynamic_fusion rows of rr_windows.csv (or None).
    """
    hr_config = replace(hr_config or HRConfig(), hr_min_bpm=cfg.hr_min_bpm, min_window_sec=cfg.hr_window_sec)
    n = len(rgb)
    times = np.arange(0., n / fps, cfg.step_sec)
    step = cfg.step_sec
    table = pd.DataFrame(dict(time_sec=times))

    # Heart rate on the extended band, by multi-site consensus.
    cr = CardioRespConfig(rr_harmonics=tuple(cfg.rr_harmonics), harmonic_halfwidth_bpm=cfg.harmonic_halfwidth_bpm)
    pulses = np.stack([extract_pulse(rgb[:, r], fps, cfg.hr_method, hr_config) for r in range(rgb.shape[1])], axis=1)
    rr_bpm = sample_windows(rr_windows, times, 'rr_brpm', step)
    hr, sqi, ror = (np.full(len(times), np.nan) for _ in range(3))
    support, harmonic = np.zeros(len(times), int), np.zeros(len(times), bool)
    for i, t in enumerate(times):
        bounds = window_bounds(t, fps, n, cfg.hr_window_sec)
        if bounds is not None:
            hr[i], sqi[i], support[i], harmonic[i] = window_hr(pulses, weights, names, fps, bounds, rr_bpm[i],
                                                               hr_config, cr)
    hr_ok = np.isfinite(hr) & (sqi >= cfg.hr_sqi_threshold) & (support >= cfg.min_agreeing_sites) & ~harmonic
    hr_base = rolling_median(hr, hr_ok, step, cfg.baseline_sec, cfg.baseline_min_sec)
    implausible = hr_ok & ~(hr_base >= cfg.min_baseline_bpm)
    hr_ok &= ~implausible
    brady = hr_ok & ((hr < cfg.brady_abs_bpm) | (hr < (1 - cfg.brady_rel_drop) * hr_base))

    # Desaturation proxy, only where the pulse is reliable.
    for i, t in enumerate(times):
        bounds = window_bounds(t, fps, n, cfg.desat_window_sec)
        if bounds is not None and hr_ok[i]:
            ror[i] = ratio_of_ratios(rgb, weights, fps, bounds, hr_config)
    ror_ok = np.isfinite(ror)
    ror_base = rolling_median(ror, ror_ok, step, cfg.baseline_sec, cfg.baseline_min_sec)
    desat = ror_ok & (ror >= (1 + cfg.desat_rise) * ror_base)

    # Breathing.
    frames = np.clip(np.rint(times * fps).astype(int), 0, len(waveform) - 1)
    envelope = moving_rms(np.asarray(waveform, float), fps, cfg.envelope_sec)[frames]
    half = max(1, int(round(cfg.envelope_sec * fps / 2)))
    moving = np.array([np.nanmax(gross[max(0, f - half):f + half + 1]) > cfg.gross_motion_box_per_sec
                       if np.isfinite(gross[max(0, f - half):f + half + 1]).any() else False for f in frames])
    breath_ok = np.isfinite(envelope) & ~moving
    breath_base = rolling_median(envelope, breath_ok, step, cfg.baseline_sec, cfg.baseline_min_sec)
    # An RR window with a finite waveform but no moving ROI (run_rr.py amplitude gate) has no breathing at all.
    covered = sample_windows(rr_windows, times, 'window_center_sec', step)
    still = np.isfinite(covered) & ~np.isfinite(rr_bpm) & breath_ok
    pause = breath_ok & ((envelope < cfg.pause_ratio * breath_base) | still)

    phase = sample_windows(rr_windows, times, 'taa_phase_deg', step) if _has(rr_windows, 'taa_phase_deg') else times * np.nan
    coherence = (sample_windows(rr_windows, times, 'taa_coherence', step) if _has(rr_windows, 'taa_coherence')
                 else times * np.nan)
    taa_ok = np.isfinite(phase) & (coherence >= cfg.min_taa_coherence) & breath_ok
    paradoxical = taa_ok & (phase >= cfg.paradox_deg)

    table = table.assign(hr_bpm=hr, hr_sqi=sqi, hr_sites_agreeing=support, hr_harmonic_ambiguous=harmonic,
                         hr_implausible_baseline=implausible, hr_assessable=hr_ok,
                         hr_baseline_bpm=hr_base, bradycardia=brady, rr_brpm=rr_bpm, breath_envelope=envelope,
                         breath_baseline=breath_base, breath_ratio=envelope / breath_base, gross_motion=moving,
                         breath_assessable=breath_ok, breathing_pause=pause, taa_phase_deg=phase,
                         taa_coherence=coherence, taa_assessable=taa_ok, paradoxical=paradoxical,
                         ror=ror, ror_baseline=ror_base, ror_ratio=ror / ror_base, spo2_proxy_assessable=ror_ok,
                         desaturation=desat)
    return table


def _has(table, column):
    return table is not None and column in table and table[column].notna().any()


def _event(kind, times, start, stop, step, **extra):
    return dict(event_type=kind, start_sec=float(times[start]), end_sec=float(times[stop - 1] + step),
                duration_sec=float((stop - start) * step), **extra)


def detect_events(line, cfg):
    """Events of one video from its timeline (see timeline()); returns a list of dicts."""
    times, step = line.time_sec.to_numpy(), cfg.step_sec
    events = []
    for start, stop in runs(line.bradycardia, step, cfg.merge_gap_sec, cfg.brady_min_sec):
        part = line.iloc[start:stop]
        events.append(_event('bradycardia', times, start, stop, step, hr_min_bpm=float(part.hr_bpm.min()),
                             hr_baseline_bpm=float(part.hr_baseline_bpm.median())))
    for start, stop in runs(line.desaturation, step, cfg.merge_gap_sec, cfg.desat_min_sec):
        part = line.iloc[start:stop]
        events.append(_event('desaturation', times, start, stop, step, ror_ratio_max=float(part.ror_ratio.max())))
    paradox = runs(line.paradoxical, step, cfg.merge_gap_sec, cfg.paradox_min_sec)
    for start, stop in paradox:
        events.append(_event('paradoxical_breathing', times, start, stop, step,
                             taa_phase_median_deg=float(line.taa_phase_deg.iloc[start:stop].median())))
    # Pauses: a pause ends wherever breathing becomes not assessable (gross motion is not an apnoea).
    pauses = runs(line.breathing_pause, step, 0., cfg.pause_min_sec)
    for start, stop in pauses:
        duration = (stop - start) * step
        if duration < cfg.apnoea_min_sec:
            continue
        near = [(a, b) for a, b in paradox if times[a] <= times[stop - 1] + cfg.mixed_window_sec and
                times[b - 1] >= times[start] - cfg.mixed_window_sec]
        events.append(_event('apnoea', times, start, stop, step, apnoea_type='mixed' if near else 'central',
                             prolonged=bool(duration >= cfg.apnoea_prolonged_sec),
                             breath_ratio_median=float(line.breath_ratio.iloc[start:stop].median())))
    group = []
    for start, stop in pauses + [(None, None)]:
        if start is not None and group and (start - group[-1][1]) * step <= cfg.periodic_max_gap_sec:
            group.append((start, stop))
            continue
        if len(group) >= cfg.periodic_min_pauses:
            events.append(_event('periodic_breathing', times, group[0][0], group[-1][1], step, pauses=len(group)))
        group = [] if start is None else [(start, stop)]
    return cluster_events(sorted(events, key=lambda e: (e['start_sec'], e['event_type'])), cfg)


def cluster_events(events, cfg):
    """Group apnoeas, bradycardias and desaturations into ABD events and flag clinically significant ones."""
    current, number = None, 0
    for event in events:
        if event['event_type'] not in EVENT_TYPES:
            continue
        apnoea = event['event_type'] == 'apnoea'
        joins = current is not None and (
            event['start_sec'] < current['end'] and not current['apnoea'] if apnoea
            else event['start_sec'] <= current['end'] + cfg.assoc_sec)
        if not joins:
            number += 1
            current = dict(id=number, end=event['end_sec'], members=[], apnoea=False)
        current['apnoea'] |= apnoea
        current['end'] = max(current['end'], event['end_sec'])
        current['members'].append(event)
        event['cluster_id'] = number
        event['_cluster'] = current
    for event in events:
        cluster = event.pop('_cluster', None)
        if cluster is None:
            continue
        kinds = {member['event_type'] for member in cluster['members']}
        event['cluster_label'] = ''.join(LETTERS[k] for k in EVENT_TYPES if k in kinds)
        # AAP: an apnoea shorter than apnoea_prolonged_sec counts only with a bradycardia or desaturation.
        event['clinically_significant'] = bool(event['event_type'] != 'apnoea' or event['prolonged'] or len(kinds) > 1)
    return events


def summarize_clusters(events):
    """One row per ABD cluster."""
    rows = {}
    for event in events:
        if 'cluster_id' not in event:
            continue
        row = rows.setdefault(event['cluster_id'], dict(cluster_id=event['cluster_id'], label=event['cluster_label'],
                                                        start_sec=event['start_sec'], end_sec=event['end_sec'],
                                                        clinically_significant=False))
        row['start_sec'] = min(row['start_sec'], event['start_sec'])
        row['end_sec'] = max(row['end_sec'], event['end_sec'])
        row['clinically_significant'] |= event['clinically_significant']
        if event['event_type'] == 'bradycardia':
            row['hr_min_bpm'] = min(row.get('hr_min_bpm', np.inf), event['hr_min_bpm'])
        if event['event_type'] == 'apnoea':
            row['apnoea_sec'] = row.get('apnoea_sec', 0.) + event['duration_sec']
            row['apnoea_type'] = event['apnoea_type']
        if event['event_type'] == 'desaturation':
            row['ror_ratio_max'] = max(row.get('ror_ratio_max', 0.), event['ror_ratio_max'])
    return list(rows.values())


def match_events(detected, reference, event_type):
    """Event-level sensitivity / PPV: a detection matches a reference event of the same type if they overlap."""
    det = [e for e in detected if e['event_type'] == event_type]
    ref = [e for e in reference if e['event_type'] == event_type]
    overlap = lambda a, b: a['video_id'] == b['video_id'] and a['start_sec'] < b['end_sec'] and b['start_sec'] < a['end_sec']
    found = sum(any(overlap(r, d) for d in det) for r in ref)
    true = sum(any(overlap(d, r) for r in ref) for d in det)
    return dict(event_type=event_type, reference_events=len(ref), detected_events=len(det),
                sensitivity=found / len(ref) if ref else None, ppv=true / len(det) if det else None)
