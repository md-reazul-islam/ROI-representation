"""Multi-site and cross-vital extensions of Module 2 (training-free, neonatal-specific).

1. Anatomically diverse spectral consensus (HR). The cardiac pulse is systemic: it modulates skin
   colour at the same frequency at the head, trunk and limbs, whereas most NICU artefacts (a caregiver's
   hand, a moving tube, local glare, a twitching limb) are local. Per window, ROI pulse spectra are
   averaged inside each anatomical site (head, trunk, arms, legs) with the dynamic-fusion weights; each
   site votes with its own spectral peak ("one site, one vote"), the HR supported by the most sites wins,
   and ties are broken by the Module 1-weighted spectral power. A confident but artefact-dominated region
   is outvoted by agreeing sites, and no frequency between two disagreeing peaks is ever invented.
   consensus_support counts the sites that agree with the result.
2. Respiratory-harmonic-aware HR peak selection. Neonatal breathing (30-90 /min) puts its 2nd-5th
   harmonics inside the neonatal HR band (90-270 bpm); in adults the RR harmonics fall below the HR band.
   Breathing motion and shading leak into the pulse trace, so with the RR of the motion branch
   (run_rr.py), an HR peak lying on k x RR is replaced by the strongest non-harmonic peak when that
   competitor is comparably strong. A strong isolated peak on a harmonic is kept (HR = k x RR also occurs
   physiologically) and the window is flagged harmonic_ambiguous instead of being silently trusted.
   In the consensus, a body site whose own peak is such an ambiguous harmonic (typically a covered chest
   or abdomen that shows breathing but no pulse) abstains from the vote instead of distorting it.
3. Thoraco-abdominal phase (RR). Chest and abdominal breathing displacements are projected on one shared
   principal axis, so their signs are comparable, and the cross-spectral phase at the respiratory
   frequency gives the thoraco-abdominal asynchrony angle: about 0 deg synchronous, 180 deg paradoxical.
   Depth cameras have measured it in preterm infants (Ottaviani et al., IEEE JTEHM 2022); here it comes
   from an ordinary RGB camera through the Module 1 anatomical ROIs.
All three are heuristics without reference validation; evaluate them with evaluate_hr.py / evaluate_rr.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal as sps

from module2.respiration import _filtered
from module2.rppg import _usable, interpolate_gaps, spectrum


SITES = dict(HEAD=('HEAD', 'FACE', 'FOREHEAD', 'LEFT_CHEEK', 'RIGHT_CHEEK', 'UPPER_FACE'),
             TRUNK=('NECK', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'UPPER_CHEST', 'LOWER_CHEST', 'ABDOMEN',
                    'THORACO_ABDOMINAL'),
             ARMS=('LEFT_ARM', 'RIGHT_ARM'), LEGS=('LEFT_LEG', 'RIGHT_LEG'))
CHEST_ROIS = ('UPPER_CHEST', 'LOWER_CHEST')
ABDOMEN_ROIS = ('ABDOMEN',)


@dataclass(frozen=True)
class CardioRespConfig:
    # Consensus: a site agrees when its own peak is within this distance of the consensus HR.
    site_agreement_bpm: float = 6.
    # Harmonic-aware selection.
    rr_harmonics: tuple = (2, 3, 4, 5)
    # Half-width around the 2nd harmonic; it grows as k/2 for harmonic k, since an RR error e moves
    # the k-th harmonic by k x e.
    harmonic_halfwidth_bpm: float = 4.
    # A non-harmonic spectral peak (local maximum, not the flank of the harmonic's own main lobe)
    # replaces a harmonic peak if its power is at least this fraction of it.
    harmonic_competitor_ratio: float = .3
    # RR reference from run_rr.py: FLOW / dynamic_fusion windows with at least this SQI, centred within
    # max_rr_offset_sec of the HR window centre.
    min_rr_sqi: float = .3
    max_rr_offset_sec: float = 8.
    # Thoraco-abdominal phase.
    taa_min_valid_fraction: float = .9
    # Welch segments of this many breaths (>= 2 segments per window keep the coherence informative).
    taa_breaths_per_segment: float = 3.


def site_of(name):
    for site, members in SITES.items():
        if name in members:
            return site
    return None


def _hr_band(frequencies, fs, hr_config):
    low, high = hr_config.band_hz
    return (frequencies >= low) & (frequencies <= min(high, fs / 2))


def roi_spectrum(x, fs, hr_config):
    """(frequencies, power normalized to unit power in the HR band) of one ROI pulse window, or None."""
    x = _usable(x, hr_config)
    if x is None:
        return None
    frequencies, power = spectrum(x, fs, hr_config)
    total = power[_hr_band(frequencies, fs, hr_config)].sum()
    return (frequencies, power / total) if total > 0 else None


def interpolated_peak(frequencies, power, mask):
    """(bpm, power) of the parabolically interpolated log-power peak inside mask, or (None, None)."""
    indices = np.flatnonzero(mask)
    if indices.size == 0 or not np.any(power[indices] > 0):
        return None, None
    peak = indices[np.argmax(power[indices])]
    f0 = frequencies[peak]
    if 0 < peak < len(power) - 1:
        a, b, c = np.log(power[peak-1:peak+2] + 1e-20)
        denominator = a - 2*b + c
        if denominator != 0:
            f0 += .5 * (a - c) / denominator * (frequencies[1] - frequencies[0])
    return float(f0 * 60.), float(power[peak])


def near_harmonic(frequencies_bpm, rr_bpm, config):
    """Boolean mask (or bool) of frequencies within harmonic_halfwidth_bpm of k x RR."""
    values = np.asarray(frequencies_bpm, dtype=float)
    near = np.zeros(values.shape, bool)
    for k in config.rr_harmonics:
        near |= np.abs(values - k * rr_bpm) <= config.harmonic_halfwidth_bpm * k / 2
    return near if near.ndim else bool(near)


def site_spectra(spectra, names, weights):
    """(frequencies, weighted mean ROI spectrum per site, total ROI weight per site) over ROIs with a
    spectrum and positive weight."""
    sums, totals, frequencies = {}, {}, None
    for item, name, weight in zip(spectra, names, weights):
        site = site_of(name)
        if item is None or weight <= 0 or site is None:
            continue
        frequencies = item[0]
        sums[site] = sums.get(site, 0.) + weight * item[1]
        totals[site] = totals.get(site, 0.) + weight
    return frequencies, {site: sums[site] / totals[site] for site in sums}, totals


def select_hr_peak(frequencies, power, band, rr_bpm, config):
    """(hr_bpm, harmonic_ambiguous): the band peak, moved off a respiratory harmonic when a comparably
    strong non-harmonic spectral peak exists."""
    bpm, peak_power = interpolated_peak(frequencies, power, band)
    if bpm is None or rr_bpm is None or not near_harmonic(bpm, rr_bpm, config):
        return bpm, False
    local_maximum = np.zeros(len(power), bool)
    local_maximum[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    clear = band & local_maximum & ~near_harmonic(frequencies * 60., rr_bpm, config)
    alternative, alternative_power = interpolated_peak(frequencies, power, clear)
    if alternative is not None and alternative_power >= config.harmonic_competitor_ratio * peak_power:
        return alternative, False
    return bpm, True


def consensus_hr(spectra, names, weights, fs, hr_config, config, rr_bpm=None):
    """Multi-site vote for one window from per-ROI spectra (None = unusable ROI) and fusion weights.

    Every site votes with its own spectral peak; the candidate HR supported by the most sites (within
    site_agreement_bpm) wins, ties broken by sum_s W_s P_s(HR) with W_s the site's fusion weight, and the
    result is the W-weighted mean of the agreeing site peaks. With rr_bpm, a site whose own peak is an
    ambiguous respiratory harmonic abstains (if every site abstains, all vote and the HR is flagged).
    consensus_sites counts the voting sites, consensus_abstained the abstaining ones.
    """
    result = dict(hr_bpm=None, consensus_sites=0, consensus_abstained=0, consensus_support=0,
                  harmonic_ambiguous=False)
    frequencies, per_site, site_weight = site_spectra(spectra, names, weights)
    if not per_site:
        return result
    band = _hr_band(frequencies, fs, hr_config)
    own = {site: select_hr_peak(frequencies, power, band, rr_bpm, config) for site, power in per_site.items()}
    voters = [site for site in per_site if own[site][0] is not None and not own[site][1]]
    fallback = not voters
    voters = voters or [site for site in per_site if own[site][0] is not None]
    if not voters:
        return result

    def agreeing(candidate):
        return [site for site in voters if abs(own[site][0] - candidate) <= config.site_agreement_bpm]

    def power_at(candidate):
        index = int(np.argmin(np.abs(frequencies * 60. - candidate)))
        return sum(site_weight[site] * per_site[site][index] for site in voters)

    winner = max((own[site][0] for site in voters), key=lambda c: (len(agreeing(c)), power_at(c)))
    support = agreeing(winner)
    bpm = sum(site_weight[site] * own[site][0] for site in support) / sum(site_weight[site] for site in support)
    ambiguous = fallback or (rr_bpm is not None and near_harmonic(bpm, rr_bpm, config))
    return dict(hr_bpm=float(bpm), consensus_sites=len(voters), consensus_abstained=len(per_site) - len(voters),
                consensus_support=len(support), harmonic_ambiguous=bool(ambiguous))


def nearest_rr(rr_table, center_sec, config):
    """RR (breaths/min) of the reference window whose centre is closest to center_sec, or None."""
    if rr_table is None or rr_table.empty:
        return None
    offsets = (rr_table.window_center_sec - center_sec).abs()
    best = offsets.idxmin()
    return float(rr_table.rr_brpm[best]) if offsets[best] <= config.max_rr_offset_sec else None


def shared_axis_displacements(velocity_a, velocity_b, fs, rr_config, config):
    """Band-passed breathing displacements of two ROIs projected on their common principal axis.

    velocity_*: (n, 2) per-frame ROI flow (px/frame), possibly with NaN gaps. Returns two (n,) arrays,
    or None when either ROI is observed for less than taa_min_valid_fraction of the window.
    """
    traces = []
    for velocity in (velocity_a, velocity_b):
        velocity = interpolate_gaps(np.asarray(velocity, dtype=float), fs, rr_config.max_gap_sec)
        finite = np.isfinite(velocity).all(axis=1)
        if finite.mean() < config.taa_min_valid_fraction or len(velocity) < 3:
            return None
        velocity = np.where(finite[:, None], velocity, 0.)
        displacement = np.cumsum(velocity, axis=0)
        traces.append(np.column_stack([_filtered(displacement[:, c], fs, rr_config) for c in range(2)]))
    _, _, vt = np.linalg.svd(np.vstack(traces), full_matrices=False)
    return traces[0] @ vt[0], traces[1] @ vt[0]


def thoraco_abdominal_phase(chest, abdomen, fs, rr_bpm, config):
    """(phase_deg in [0, 180], magnitude-squared coherence) between chest and abdomen at the RR frequency."""
    if rr_bpm is None or not math.isfinite(rr_bpm) or rr_bpm <= 0 or np.std(chest) == 0 or np.std(abdomen) == 0:
        return None, None
    segment = int(min(len(chest), max(16, round(config.taa_breaths_per_segment * fs * 60. / rr_bpm))))
    frequencies, cross = sps.csd(chest, abdomen, fs=fs, nperseg=segment)
    _, coherence = sps.coherence(chest, abdomen, fs=fs, nperseg=segment)
    index = int(np.argmin(np.abs(frequencies - rr_bpm / 60.)))
    return float(abs(np.degrees(np.angle(cross[index])))), float(coherence[index])
