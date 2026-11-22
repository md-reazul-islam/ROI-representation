"""Build a self-contained HTML project report from the outputs of one run (no synthetic values).

Every number, table and data figure is read from this project's output files, so re-running the pipeline on other
videos and re-running this script regenerates the report from the new results. Text that depends on the results
(verdicts, comparisons) is written from the numbers. Literature values in the comparison tables are the published
numbers of the cited papers and are labelled as such.

Also writes the same report as a Word document (.docx) next to the HTML.

Usage (from the Module1 folder):
  python "HR RR using module1 v2/scripts/build_project_report.py" \
      --results "HR RR using module1 v2/runs/<RUN>/D5_hr_specific_adult" \
      [--reference-hr <csv: video_id,timestamp_sec,hr_bpm>] [--reference-name "PPG waveform"] \
      [--dataset "UBFC-rPPG Dataset 2"] [--module1-report module1_v2/reports/final34_paper] [--out report.html]
Inputs read: <results>/hr_windows.csv, metadata.json, waveforms/*.npz; module1_v2/runs/<RUN>/dynamic_roi_v2_<CFG>/;
runs/<RUN>/manifest.csv (videos); optional reference CSV; optional Module 1 v2 benchmark tables.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import html
import io
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402
import pandas as pd                                                     # noqa: E402

M2 = Path(__file__).resolve().parents[1]
ROOT = M2.parent
for p in (M2/'src', ROOT/'module1_v2/src', ROOT/'src'):
    sys.path.insert(0, str(p))
from module2.rppg import HRConfig, window_snr                          # noqa: E402

C_VAL, C_REJ, C_REF, C_PRES, C_ACC = '#2a78d6', '#9a9a9a', '#111111', '#e39b1b', '#c0392b'

# Published numbers used only in the comparison tables (verify against the original papers before citing).
PUBLISHED_UBFC = [
    ('CHROM (de Haan & Jeanne 2013)', 'classic, no training', 4.06, 'as tabulated in RhythmMamba [R15]'),
    ('POS (Wang et al. 2017)', 'classic, no training', 4.08, 'as tabulated in RhythmMamba [R15]'),
    ('PhysNet / TS-CAN, trained on PURE', 'deep learning', '0.98-1.30', 'rPPG-Toolbox [R13], cross-dataset'),
]

REFERENCES = [
    ('R1', 'Verkruysse W, Svaasand LO, Nelson JS. Remote plethysmographic imaging using ambient light. Opt. Express 2008.'),
    ('R2', 'Poh MZ, McDuff DJ, Picard RW. Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Opt. Express 2010.'),
    ('R3', 'de Haan G, Jeanne V. Robust pulse rate from chrominance-based rPPG (CHROM). IEEE Trans. Biomed. Eng. 2013.'),
    ('R4', 'Wang W, den Brinker AC, Stuijk S, de Haan G. Algorithmic principles of remote PPG (POS). IEEE Trans. Biomed. Eng. 2017.'),
    ('R5', 'Kumar M, Veeraraghavan A, Sabharwal A. DistancePPG: robust non-contact vital signs monitoring using a camera. Biomed. Opt. Express 2015.'),
    ('R6', 'Tulyakov S, et al. Self-adaptive matrix completion for heart rate estimation from face videos under realistic conditions. CVPR 2016.'),
    ('R7', 'Aarts LAM, et al. Non-contact heart rate monitoring utilizing camera photoplethysmography in the neonatal intensive care unit - a pilot study. Early Hum. Dev. 2013.'),
    ('R8', 'Villarroel M, et al. Non-contact physiological monitoring of preterm infants in the neonatal intensive care unit. npj Digit. Med. 2019.'),
    ('R9', 'Chaichulee S, et al. Multi-task convolutional neural network for patient detection and skin segmentation in continuous non-contact vital sign monitoring. IEEE FG 2017.'),
    ('R10', 'Wang W, Stuijk S, de Haan G. Living-skin classification via remote-PPG. IEEE Trans. Biomed. Eng. 2017.'),
    ('R11', 'Li Q, Clifford GD. Dynamic time warping and machine learning for signal quality assessment of pulsatile signals. Physiol. Meas. 2012.'),
    ('R12', 'Theiler J, et al. Testing for nonlinearity in time series: the method of surrogate data. Physica D 1992.'),
    ('R13', 'Liu X, et al. rPPG-Toolbox: Deep Remote PPG Toolbox. NeurIPS Datasets & Benchmarks 2023 (arXiv 2210.00716).'),
    ('R14', 'Bobbia S, et al. Unsupervised skin tissue segmentation for remote photoplethysmography (UBFC-rPPG). Pattern Recognit. Lett. 2019.'),
    ('R15', 'Zou B, et al. RhythmMamba: fast, lightweight, and accurate remote physiological measurement. arXiv 2404.06483.'),
    ('R16', 'Ravi N, et al. SAM 2: Segment Anything in images and videos. arXiv 2024.'),
    ('R17', 'Khirodkar R, et al. Sapiens: foundation for human vision models. ECCV 2024.'),
    ('R18', 'Yu Z, Li X, Zhao G. Remote photoplethysmograph signal measurement from facial videos using spatio-temporal networks (PhysNet). BMVC 2019.'),
    ('R19', 'Liu X, et al. Multi-task temporal shift attention networks for on-device contactless vitals measurement (TS-CAN). NeurIPS 2020.'),
    ('R20', 'Viterbi AJ. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Trans. Inf. Theory 1967.'),
]


# ------------------------------------------------------------------------------------------------ helpers
def fig_html(fig, caption, width=None):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    style = f' style="max-width:{width}px"' if width else ''
    return f'<figure><img src="data:image/png;base64,{data}"{style}><figcaption>{caption}</figcaption></figure>'


def img_html(bgr, caption):
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return (f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}">'
            f'<figcaption>{caption}</figcaption></figure>') if ok else ''


def table_html(df, fmt=None, cls='data'):
    fmt = fmt or {}
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = '–' if not np.isfinite(v) else (fmt.get(c, '{:.2f}').format(v))
            else:
                s = html.escape(str(v))
            cells.append(f'<td>{s}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="tw"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def natural(v):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(v))]


def parse_pm(s):
    """'1.23 ± 0.45' -> (1.23, 0.45)."""
    m = re.match(r'\s*([-+]?[\d.]+)\s*±\s*([\d.]+)', str(s))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def match_reference(hr, ref):
    rows = []
    for video, p in hr.groupby('video_id'):
        r = ref[ref.video_id == video].dropna(subset=['hr_bpm']).sort_values('timestamp_sec')
        if not len(r):
            continue
        idx = np.searchsorted(r.timestamp_sec.to_numpy(), p.window_center_sec.to_numpy()).clip(0, len(r) - 1)
        near = r.iloc[idx]
        ok = (np.abs(near.timestamp_sec.to_numpy() - p.window_center_sec.to_numpy()) <= 1.) & p.hr_bpm.notna().to_numpy()
        q = p[ok].copy()
        q['hr_reference'] = near.hr_bpm.to_numpy()[ok]
        rows.append(q)
    if not rows:
        return pd.DataFrame()
    e = pd.concat(rows, ignore_index=True)
    e['error'] = e.hr_bpm - e.hr_reference
    return e


def metrics(e):
    if not len(e):
        return dict(windows=0, mae=np.nan, rmse=np.nan, within5=np.nan, r=np.nan)
    a = e.error.abs()
    return dict(windows=len(e), mae=a.mean(), rmse=float(np.sqrt((e.error ** 2).mean())), within5=100 * (a <= 5).mean(),
                r=e.hr_bpm.corr(e.hr_reference) if len(e) > 2 else np.nan)


def reason(row):
    if row.label == 'validated':
        return 'validated'
    if not bool(row.pulse_present):
        return 'pulse not above null'
    if not bool(row.halves_agree):
        return 'halves disagree'
    return 'other'


# ------------------------------------------------------------------------------------------------ diagrams (static SVG)
SVG_PIPELINE = '''
<svg viewBox="0 0 1100 250" class="diagram" role="img" aria-label="End-to-end pipeline">
 <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrow"/></marker></defs>
 <g class="box in"><rect x="10" y="80" width="130" height="90" rx="10"/><text x="75" y="115">Input video</text><text x="75" y="137" class="s">RGB camera,</text><text x="75" y="153" class="s">any length / fps</text></g>
 <g class="box m1"><rect x="175" y="20" width="250" height="210" rx="10"/><text x="300" y="45">Module 1 (base)</text>
   <text x="300" y="75" class="s">detection: YOLO11s + RT-DETR</text><text x="300" y="95" class="s">tracking + recovery</text>
   <text x="300" y="115" class="s">SAM 2 infant mask</text><text x="300" y="135" class="s">Sapiens pose (308 keypoints)</text>
   <text x="300" y="155" class="s">anatomical ROI polygons</text><text x="300" y="175" class="s">per-ROI quality terms</text>
   <text x="300" y="200" class="s">(pose, visibility, skin,</text><text x="300" y="216" class="s">temporal, motion)</text></g>
 <g class="box m1v2"><rect x="460" y="20" width="230" height="210" rx="10"/><text x="575" y="45">Module 1 v2</text>
   <text x="575" y="75" class="s">corrected terms (keypoint V')</text><text x="575" y="95" class="s">camera-compensated motion</text>
   <text x="575" y="115" class="s">EMA-smoothed quality</text><text x="575" y="135" class="s">Viterbi + hysteresis</text>
   <text x="575" y="155" class="s">smoothed weights, crossfade</text><text x="575" y="185" class="b">→ dynamic ROI set</text>
   <text x="575" y="205" class="s">(polygons + weights / frame)</text></g>
 <g class="box hr"><rect x="725" y="20" width="200" height="210" rx="10"/><text x="825" y="45">Module 2 v2: HR</text>
   <text x="825" y="75" class="s">HR-specific skin regions</text><text x="825" y="95" class="s">(face, forehead, cheeks,</text>
   <text x="825" y="111" class="s">upper face, upper chest)</text><text x="825" y="135" class="s">mean RGB → POS pulse</text>
   <text x="825" y="155" class="s">8 s windows, 1 s step</text><text x="825" y="175" class="s">FFT peak → HR, SNR</text>
   <text x="825" y="205" class="b">window validity test</text></g>
 <g class="box out"><rect x="960" y="60" width="130" height="130" rx="10"/><text x="1025" y="95">Output</text>
   <text x="1025" y="120" class="s">validated HR</text><text x="1025" y="138" class="s">or "not assessable"</text>
   <text x="1025" y="160" class="s">+ Kalman track</text><text x="1025" y="176" class="s">+ accuracy (if ref.)</text></g>
 <line x1="140" y1="125" x2="172" y2="125" class="ln" marker-end="url(#ar)"/>
 <line x1="425" y1="125" x2="457" y2="125" class="ln" marker-end="url(#ar)"/>
 <line x1="690" y1="125" x2="722" y2="125" class="ln" marker-end="url(#ar)"/>
 <line x1="925" y1="125" x2="957" y2="125" class="ln" marker-end="url(#ar)"/>
</svg>'''

SVG_MODULE1V2 = '''
<svg viewBox="0 0 1100 300" class="diagram" role="img" aria-label="Module 1 v2 architecture">
 <defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrow"/></marker></defs>
 <g class="box m1"><rect x="10" y="20" width="200" height="260" rx="10"/><text x="110" y="45">Per frame (Module 1)</text>
   <text x="110" y="75" class="s">ROI candidates r = 1..R</text><text x="110" y="95" class="s">(head, face, forehead,</text><text x="110" y="111" class="s">cheeks, chest, abdomen,</text><text x="110" y="127" class="s">limbs ...)</text>
   <text x="110" y="155" class="s">polygon ∩ SAM 2 mask</text><text x="110" y="175" class="s">skin pixels</text><text x="110" y="205" class="s">5 quality terms per ROI</text><text x="110" y="225" class="s">P, V, S, T, M</text></g>
 <g class="box m1v2"><rect x="245" y="20" width="200" height="120" rx="10"/><text x="345" y="45">1b/1c corrected terms</text>
   <text x="345" y="72" class="s">V' = share of ROI keypoints</text><text x="345" y="88" class="s">with confidence ≥ 0.5</text><text x="345" y="112" class="s">M: residual flow after</text><text x="345" y="128" class="s">camera motion</text></g>
 <g class="box m1v2"><rect x="245" y="160" width="200" height="120" rx="10"/><text x="345" y="185">Quality score</text>
   <text x="345" y="212" class="s">Q = 0.2·(P + V' + S + T + M)</text><text x="345" y="236" class="s">equal weights, no training</text><text x="345" y="258" class="s">EMA over time (β = 0.95)</text></g>
 <g class="box m1v2"><rect x="480" y="20" width="250" height="260" rx="10"/><text x="605" y="45">1a temporal stabilizer</text>
   <text x="605" y="75" class="s">Viterbi path over ROI states:</text><text x="605" y="93" class="s">maximise Σ Q − 0.2 · #switches</text>
   <text x="605" y="120" class="s">hysteresis: new ROI must be</text><text x="605" y="136" class="s">5 % better for 8 frames</text>
   <text x="605" y="165" class="s">top-3 set, softmax weights</text><text x="605" y="183" class="s">weights smoothed (α = 0.5),</text><text x="605" y="199" class="s">1 s crossfade</text>
   <text x="605" y="230" class="s">settings tuned once on 21</text><text x="605" y="246" class="s">training videos, then frozen</text></g>
 <g class="box out"><rect x="765" y="70" width="320" height="160" rx="10"/><text x="925" y="100">dynamic_roi_v2.csv</text>
   <text x="925" y="128" class="s">every frame × every ROI:</text><text x="925" y="146" class="s">pixel polygon, q_v2, selected, rank, weight,</text>
   <text x="925" y="164" class="s">skin proportion, motion (total / residual),</text><text x="925" y="182" class="s">camera motion, infant box</text>
   <text x="925" y="208" class="b">consumed by HR (and RR)</text></g>
 <line x1="210" y1="80" x2="242" y2="80" class="ln" marker-end="url(#ar2)"/>
 <line x1="210" y1="220" x2="242" y2="220" class="ln" marker-end="url(#ar2)"/>
 <line x1="345" y1="140" x2="345" y2="157" class="ln" marker-end="url(#ar2)"/>
 <line x1="445" y1="220" x2="477" y2="220" class="ln" marker-end="url(#ar2)"/>
 <line x1="730" y1="150" x2="762" y2="150" class="ln" marker-end="url(#ar2)"/>
</svg>'''

SVG_HR = '''
<svg viewBox="0 0 1100 280" class="diagram" role="img" aria-label="HR measurement architecture">
 <defs><marker id="ar3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrow"/></marker></defs>
 <g class="box m1v2"><rect x="10" y="40" width="180" height="200" rx="10"/><text x="100" y="65">All Module 1 v2</text><text x="100" y="83">candidates</text>
   <text x="100" y="112" class="s">face, forehead, L/R cheek,</text><text x="100" y="128" class="s">upper face, upper chest</text>
   <text x="100" y="155" class="s">eligible: ≥150 skin px,</text><text x="100" y="171" class="s">skin ≥ 30 % of polygon,</text><text x="100" y="187" class="s">present ≥ 90 % of window</text></g>
 <g class="box hr"><rect x="225" y="40" width="200" height="200" rx="10"/><text x="325" y="65">Per region</text>
   <text x="325" y="95" class="s">mean RGB of skin pixels</text><text x="325" y="115" class="s">POS projection (1.6 s)</text>
   <text x="325" y="135" class="s">detrend + band-pass</text><text x="325" y="155" class="s">(HR band)</text>
   <text x="325" y="185" class="s">SNR vs its own</text><text x="325" y="201" class="s">block-shuffle null</text></g>
 <g class="box hr"><rect x="460" y="40" width="210" height="200" rx="10"/><text x="565" y="65">Region selection</text>
   <text x="565" y="95" class="s">score = prior × sigmoid(</text><text x="565" y="111" class="s">past excess SNR / 2 dB)</text>
   <text x="565" y="137" class="s">past windows only (≤ 20 s)</text><text x="565" y="160" class="s">k ≤ 3, overlap ≤ 30 %</text>
   <text x="565" y="183" class="s">hysteresis: +10 % for</text><text x="565" y="199" class="s">3 windows to switch</text></g>
 <g class="box hr"><rect x="705" y="40" width="190" height="200" rx="10"/><text x="800" y="65">Per 8 s window</text>
   <text x="800" y="95" class="s">score-weighted fusion</text><text x="800" y="115" class="s">FFT peak → HR</text>
   <text x="800" y="135" class="s">SNR (de Haan)</text><text x="800" y="163" class="s">left-half and right-half</text><text x="800" y="179" class="s">pulses → HR_L, HR_R</text>
   <text x="800" y="210" class="b">→ validity test</text></g>
 <g class="box out"><rect x="930" y="70" width="160" height="140" rx="10"/><text x="1010" y="100">Window label</text>
   <text x="1010" y="128" class="s v">validated</text><text x="1010" y="150" class="s p">present</text><text x="1010" y="172" class="s n">not assessable</text>
   <text x="1010" y="196" class="s">Kalman on validated</text></g>
 <line x1="190" y1="140" x2="222" y2="140" class="ln" marker-end="url(#ar3)"/>
 <line x1="425" y1="140" x2="457" y2="140" class="ln" marker-end="url(#ar3)"/>
 <line x1="670" y1="140" x2="702" y2="140" class="ln" marker-end="url(#ar3)"/>
 <line x1="895" y1="140" x2="927" y2="140" class="ln" marker-end="url(#ar3)"/>
</svg>'''

SVG_VALIDITY = '''
<svg viewBox="0 0 1100 330" class="diagram" role="img" aria-label="Window validation decision">
 <defs><marker id="ar4" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" class="arrow"/></marker></defs>
 <g class="box hr"><rect x="10" y="110" width="170" height="110" rx="10"/><text x="95" y="145">8 s window</text><text x="95" y="170" class="s">fused pulse, HR, SNR</text><text x="95" y="188" class="s">HR_L, HR_R</text></g>
 <g class="box gate"><rect x="225" y="20" width="270" height="140" rx="10"/><text x="360" y="45">Gate 1: is a pulse present?</text>
   <text x="360" y="72" class="s">repeat the whole procedure on 49</text><text x="360" y="88" class="s">surrogates: same pixels, 0.25 s blocks</text>
   <text x="360" y="104" class="s">shuffled in time (rhythm destroyed)</text><text x="360" y="130" class="b">SNR &gt; 95th percentile of surrogate SNRs?</text></g>
 <g class="box gate"><rect x="225" y="185" width="270" height="130" rx="10"/><text x="360" y="210">Gate 2: do two body parts agree?</text>
   <text x="360" y="237" class="s">left half and right half of the</text><text x="360" y="253" class="s">selected regions measured separately</text>
   <text x="360" y="285" class="b">|HR_L − HR_R| ≤ 6 bpm?</text></g>
 <g class="box lv"><rect x="560" y="20" width="230" height="70" rx="10"/><text x="675" y="50">VALIDATED</text><text x="675" y="72" class="s">gate 1 yes and gate 2 yes</text></g>
 <g class="box lp"><rect x="560" y="125" width="230" height="70" rx="10"/><text x="675" y="155">PRESENT (not reported)</text><text x="675" y="177" class="s">gate 1 yes, gate 2 no</text></g>
 <g class="box ln_"><rect x="560" y="230" width="230" height="70" rx="10"/><text x="675" y="260">NOT ASSESSABLE</text><text x="675" y="282" class="s">gate 1 no</text></g>
 <g class="box out"><rect x="840" y="60" width="250" height="210" rx="10"/><text x="965" y="90">Reported to the clinician</text>
   <text x="965" y="120" class="s">only VALIDATED windows give an HR</text><text x="965" y="140" class="s">and update the Kalman track;</text>
   <text x="965" y="168" class="s">all others show "--" with the reason</text><text x="965" y="196" class="s">(never silently filled in)</text>
   <text x="965" y="228" class="s">false-alarm rate on pure noise</text><text x="965" y="244" class="s">≤ 5 % by construction (gate 1)</text></g>
 <line x1="180" y1="150" x2="222" y2="100" class="ln" marker-end="url(#ar4)"/>
 <line x1="180" y1="185" x2="222" y2="240" class="ln" marker-end="url(#ar4)"/>
 <line x1="495" y1="80" x2="557" y2="55" class="ln" marker-end="url(#ar4)"/>
 <line x1="495" y1="250" x2="557" y2="160" class="ln" marker-end="url(#ar4)"/>
 <line x1="495" y1="130" x2="557" y2="262" class="ln" marker-end="url(#ar4)"/>
 <line x1="790" y1="55" x2="837" y2="120" class="ln" marker-end="url(#ar4)"/>
</svg>'''

CSS = '''
:root{--bg:#ffffff;--fg:#1d1d1f;--muted:#5f6368;--line:#d9dce1;--card:#f6f7f9;--accent:#2a78d6;--m1:#e8f0fb;--m1v2:#e6f4ec;--hr:#fdf1e3;--gate:#f3ecfa;
 --lv:#dff0e3;--lp:#fdf0d8;--ln:#ececec;--out:#eef1f4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#16181b;--fg:#e8e8ea;--muted:#a3a7ad;--line:#33373d;--card:#1f2226;--accent:#6aa7ef;
 --m1:#1d2a3b;--m1v2:#1c3026;--hr:#3a2c1b;--gate:#2c2238;--lv:#1d3324;--lp:#3a3019;--ln:#2a2c2f;--out:#23272c}}
:root[data-theme="dark"]{--bg:#16181b;--fg:#e8e8ea;--muted:#a3a7ad;--line:#33373d;--card:#1f2226;--accent:#6aa7ef;--m1:#1d2a3b;--m1v2:#1c3026;--hr:#3a2c1b;--gate:#2c2238;--lv:#1d3324;--lp:#3a3019;--ln:#2a2c2f;--out:#23272c}
*{box-sizing:border-box} body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0}
main{max-width:1080px;margin:0 auto;padding:24px 16px 80px} h1{font-size:28px;line-height:1.25;margin:8px 0 4px} h2{font-size:21px;margin:40px 0 8px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:17px;margin:24px 0 6px} p,li{max-width:900px} .muted{color:var(--muted)} .small{font-size:13px}
nav.toc{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 16px;margin:16px 0} nav.toc a{color:var(--accent);text-decoration:none} nav.toc ol{margin:4px 0;padding-left:22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px} .card .v{font-size:24px;font-weight:650} .card .k{font-size:13px;color:var(--muted)}
.tw{overflow-x:auto;margin:10px 0} table.data{border-collapse:collapse;font-size:13.5px;min-width:60%} table.data th,table.data td{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left;white-space:nowrap}
table.data th{background:var(--card);font-weight:600} figure{margin:14px 0} figure img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
figcaption{font-size:13px;color:var(--muted);margin-top:4px;max-width:900px} .note{background:var(--card);border-left:4px solid var(--accent);padding:8px 14px;border-radius:6px;margin:12px 0}
.warn{border-left-color:#d08a00} svg.diagram{width:100%;height:auto;margin:10px 0} svg text{fill:var(--fg);font-size:14px;text-anchor:middle;font-weight:600}
svg text.s{font-size:12px;font-weight:400} svg text.b{font-size:12.5px;font-weight:700} svg .ln{stroke:var(--muted);stroke-width:2} svg .arrow{fill:var(--muted)}
svg .box rect{stroke:var(--line);stroke-width:1.5} .in rect,.out rect{fill:var(--out)} .m1 rect{fill:var(--m1)} .m1v2 rect{fill:var(--m1v2)} .hr rect{fill:var(--hr)} .gate rect{fill:var(--gate)}
.lv rect{fill:var(--lv)} .lp rect{fill:var(--lp)} .ln_ rect{fill:var(--ln)} svg text.v{fill:#1f8a3a} svg text.p{fill:#b37400} svg text.n{fill:var(--muted)}
code{background:var(--card);padding:1px 5px;border-radius:4px;font-size:13px}
@media print{h2{break-before:auto} figure,table{break-inside:avoid}}
'''


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--reference-hr')
    ap.add_argument('--reference-name', default='reference')
    ap.add_argument('--dataset', default='')
    ap.add_argument('--module1-report', default='module1_v2/reports/final34_paper')
    ap.add_argument('--title', default='Camera-based heart-rate monitoring with dynamic ROI selection and window validation')
    ap.add_argument('--out')
    a = ap.parse_args(argv)

    res = (ROOT/a.results).resolve() if not Path(a.results).is_absolute() else Path(a.results)
    meta = json.loads((res/'metadata.json').read_text())
    run = Path(meta['run']); run_name = run.name
    cfg_name = meta.get('module1_config', 'D5')
    band = meta.get('hr_band_bpm', [90., 270.])
    hcfg = meta['config']['hr']
    hr = pd.read_csv(res/'hr_windows.csv')
    hr['reason'] = hr.apply(reason, axis=1)
    videos = sorted(hr.video_id.unique(), key=natural)
    manifest = pd.read_csv(run/'manifest.csv') if (run/'manifest.csv').is_file() else pd.DataFrame()
    rep = (ROOT/a.module1_report) if a.module1_report else None
    ref = pd.read_csv(a.reference_hr) if a.reference_hr else None
    err = match_reference(hr, ref) if ref is not None else pd.DataFrame()
    has_ref = len(err) > 0
    HR = dataclasses.replace(HRConfig(), hr_min_bpm=float(band[0]), hr_max_bpm=float(band[1]))
    S = []                                                         # html sections

    # ------------------------------------------------------------- headline numbers
    n_win = len(hr); n_val = int((hr.label == 'validated').sum())
    val_pct = 100 * n_val / n_win if n_win else np.nan
    m_val = metrics(err[err.label == 'validated']) if has_ref else None
    m_all = metrics(err) if has_ref else None
    m_rej = metrics(err[err.label != 'validated']) if has_ref else None
    dur = manifest.set_index('video_id').duration_sec.reindex(videos).sum() if len(manifest) else np.nan

    cards = [('videos', f'{len(videos)}'), ('total duration', f'{dur:.0f} s' if np.isfinite(dur) else '–'),
             ('HR windows (8 s)', f'{n_win}'), ('validated windows', f'{n_val} ({val_pct:.0f} %)')]
    if has_ref:
        cards += [('MAE, validated', f'{m_val["mae"]:.2f} bpm'), ('MAE, all windows', f'{m_all["mae"]:.2f} bpm'),
                  ('within ±5 bpm (validated)', f'{m_val["within5"]:.0f} %'), ('Pearson r (validated)', f'{m_val["r"]:.2f}')]
    cards_html = '<div class="cards">' + ''.join(f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in cards) + '</div>'

    # ------------------------------------------------------------- Section 1: architecture
    s = ['<h2 id="s1">1. System architecture: from input video to validated heart rate</h2>',
         '<p>The system has two modules. <b>Module 1</b> finds the infant (or subject), its body parts and its skin, and '
         '<b>Module 1 v2</b> turns these into a stable, weighted set of regions of interest (ROIs) for every frame. '
         '<b>Module 2 v2</b> measures the heart rate (HR) from the skin pixels of HR-specific regions in 8 s windows and '
         'reports an HR only for windows that pass a two-gate validity test.</p>', SVG_PIPELINE,
         '<p class="small muted">Figure 1. End-to-end pipeline. Colours: blue = Module 1 (perception), green = Module 1 v2 '
         '(dynamic ROI), orange = Module 2 v2 (HR), grey = input / output.</p>',
         '<h3>1.1 Module 1 v2: dynamic ROI selection</h3>', SVG_MODULE1V2,
         '<p class="small muted">Figure 2. Module 1 v2. Each ROI gets an equal-weight quality score from five terms; v2 replaces '
         'the visibility term by keypoint visibility V\' (in v1 visibility was identical to skin likelihood), and stabilizes '
         'the choice over time with an EMA, a Viterbi path with a switch penalty and hysteresis. '
         f'Configuration used in this run: <code>{cfg_name}</code>.</p>',
         '<h3>1.2 Module 2 v2: heart-rate measurement</h3>', SVG_HR,
         '<p class="small muted">Figure 3. HR branch (<code>hr_specific</code>). HR picks its own skin regions from all Module 1 v2 '
         'candidates, ranked by their pulse evidence in <i>past</i> windows only, then fuses their POS pulses per 8 s window.</p>']
    # real frame with ROIs from this run
    frame_fig = roi_frame_figure(run, run_name, cfg_name, hr, manifest, videos)
    if frame_fig:
        s.append(frame_fig)
    S.append(''.join(s))

    # ------------------------------------------------------------- Section 2: existing work
    S.append('''<h2 id="s2">2. Existing work</h2>
<div class="tw"><table class="data"><thead><tr><th>Line of work</th><th>Representative papers</th><th>What it does</th><th>What it does not do</th></tr></thead><tbody>
<tr><td>Classic rPPG signal extraction</td><td>Green [R1], ICA [R2], CHROM [R3], POS [R4]</td><td>pulse from mean skin colour of a face box</td><td>no ROI reasoning, always outputs an HR</td></tr>
<tr><td>Region weighting / selection</td><td>DistancePPG [R5], matrix completion [R6]</td><td>weights or selects face regions by signal quality</td><td>face only, adults, no temporal stability control, no statistical pulse test</td></tr>
<tr><td>Deep rPPG</td><td>PhysNet [R18], TS-CAN [R19], rPPG-Toolbox [R13]</td><td>learned end-to-end pulse; ~1 bpm MAE on UBFC</td><td>needs labelled training data, adult faces, gives no "do not trust" output</td></tr>
<tr><td>Neonatal camera monitoring</td><td>Aarts 2013 [R7], Villarroel 2019 [R8], Chaichulee 2017 [R9]</td><td>NICU feasibility, skin segmentation, quality-based exclusion</td><td>fixed or skin-segmented regions, no anatomy-specific dynamic ROI</td></tr>
<tr><td>Signal quality / living skin</td><td>SQI for PPG [R11], living-skin classification [R10]</td><td>scores pulse quality or detects living skin</td><td>thresholds learned or fixed, no per-window false-alarm guarantee</td></tr>
<tr><td>Surrogate-data testing</td><td>Theiler 1992 [R12]</td><td>tests if structure in a signal is beyond chance</td><td>(general statistics; not applied per window in rPPG to our knowledge)</td></tr>
</tbody></table></div>''')

    # ------------------------------------------------------------- Section 3/4
    S.append('''<h2 id="s3">3. Research gap</h2><ol>
<li><b>Infants are not adult faces.</b> Most rPPG methods use a face box. A newborn's face is small, often covered (tubes, CPAP, hands,
blankets) and turned away; the useful skin moves between face, forehead, cheeks and chest.</li>
<li><b>Dynamic ROI methods are unstable or face-only.</b> Per-frame "best region" selection switches region many times per second, which
injects artefacts into the pulse; existing region weighting works on adult faces.</li>
<li><b>Methods always output a number.</b> Classic and deep rPPG return an HR for every window, even when the video carries no pulse
(compression, motion, occlusion, caregiver handling). In a NICU a wrong but confident HR is worse than no HR.</li>
<li><b>Quality thresholds are ad hoc.</b> SNR or SQI cut-offs are chosen per dataset and give no guaranteed false-alarm rate.</li>
</ol>
<h2 id="s4">4. Our contribution</h2><ol>
<li><b>Anatomy-aware dynamic ROI (Module 1 v2):</b> pose + SAM 2 mask + skin give labelled body-part ROIs per frame; quality terms are
corrected (independent keypoint visibility, camera-compensated motion) and the selection is stabilized (EMA, Viterbi with a switch
penalty, hysteresis). Stabilizer settings are tuned once on training videos and frozen.</li>
<li><b>HR-specific region selection:</b> HR chooses its own skin regions (face, forehead, cheeks, upper face, upper chest) from all candidates,
ranked on pulse evidence from past windows only (causal, no look-ahead), with overlap exclusion and hysteresis.</li>
<li><b>Per-window validity test with a controlled false-alarm rate:</b> a block-shuffle surrogate null of the <i>same</i> pixels (whole
select-then-measure procedure repeated on surrogates) plus left/right split-half agreement. Only validated windows are reported.</li>
<li><b>Honest output:</b> every window is labelled validated / present / not assessable (and caregiver intervention in the infant
notebook), so downstream users know when not to trust the camera.</li></ol>''')

    # ------------------------------------------------------------- Section 5 methods
    S.append(f'''<h2 id="s5">5. Methods</h2>
<h3>5.1 Dynamic ROI (Module 1 + Module 1 v2)</h3><ol>
<li><b>Subject localisation:</b> YOLO11s and RT-DETR person detections are fused, tracked and recovered across frames; SAM 2 [R16]
produces the subject mask.</li>
<li><b>Anatomy:</b> Sapiens pose [R17] (308 keypoints) gives body landmarks; each ROI (head, face, forehead, cheeks, upper face, chest, abdomen,
limbs, ...) is a polygon built from its template keypoints and intersected with the SAM 2 mask and skin pixels.</li>
<li><b>Quality:</b> Q = 0.2·(P + V' + S + T + M): pose confidence P, keypoint visibility V' (share of the ROI's keypoints with confidence ≥ 0.5),
skin likelihood S, temporal consistency T, and a motion term M (low motion = high quality). Equal weights; nothing is trained.</li>
<li><b>Stabilizer:</b> EMA of Q (β = 0.95), a Viterbi path [R20] over ROI states that maximises ΣQ − 0.2 × (number of switches), hysteresis
(a new ROI must be 5 % better for 8 frames), top-3 set with softmax weights, weight smoothing (α = 0.5) and a 1 s crossfade.
Settings were chosen on 21 training videos (objective: fewest switches with ≥ 98 % of the per-frame quality retained) and frozen.</li></ol>
<h3>5.2 HR measurement (Module 2 v2, <code>hr_specific</code>)</h3><ol>
<li><b>Eligible regions</b> per window: ≥ 150 skin pixels, skin ≥ 30 % of the polygon, present in ≥ 90 % of the window.</li>
<li><b>Pulse per region:</b> mean RGB of skin pixels → POS [R4] (1.6 s) → detrending and band-pass to the HR band
({band[0]:.0f}–{band[1]:.0f} bpm in this run).</li>
<li><b>Region score:</b> prior (Module 1 v2 quality × log skin pixels) × sigmoid(mean excess SNR over the region's own null in windows that ended before
this window, within 20 s / 2 dB). Up to 3 non-overlapping regions (overlap ≤ 30 %); a new set must score 10 % higher for 3 windows.</li>
<li><b>HR per window:</b> score-weighted fused pulse, 8 s window, 1 s step; HR = interpolated FFT peak; SNR as in de Haan & Jeanne [R3].</li>
<li><b>Validity:</b> see Section 8. A Kalman filter tracks HR over validated windows only.</li></ol>''')

    # ------------------------------------------------------------- Section 6: Module 1 v2 vs alternatives
    S.append(section_roi(rep, run, run_name, cfg_name, hr))

    # ------------------------------------------------------------- Section 7: HR vs existing
    s = ['<h2 id="s7">7. Is our HR measurement better than existing work?</h2>']
    if has_ref:
        s.append(f'<p>Measured on this run ({html.escape(a.dataset or run_name)}, {len(videos)} videos) against the '
                 f'{html.escape(a.reference_name)} reference:</p>')
        t = pd.DataFrame([dict(method='This project, all windows (comparable to papers)', windows=m_all['windows'],
                               **{'MAE (bpm)': m_all['mae'], 'RMSE (bpm)': m_all['rmse'], '±5 bpm %': m_all['within5'], 'r': m_all['r']}),
                          dict(method=f'This project, validated windows only ({val_pct:.0f} % of windows)', windows=m_val['windows'],
                               **{'MAE (bpm)': m_val['mae'], 'RMSE (bpm)': m_val['rmse'], '±5 bpm %': m_val['within5'], 'r': m_val['r']}),
                          dict(method='This project, rejected windows', windows=m_rej['windows'],
                               **{'MAE (bpm)': m_rej['mae'], 'RMSE (bpm)': m_rej['rmse'], '±5 bpm %': m_rej['within5'], 'r': m_rej['r']})])
        s.append(table_html(t, {'MAE (bpm)': '{:.2f}', 'RMSE (bpm)': '{:.2f}', '±5 bpm %': '{:.1f}', 'r': '{:.2f}'}))
        if 'ubfc' in (a.dataset + run_name).lower():
            pub = pd.DataFrame(PUBLISHED_UBFC, columns=['published method (UBFC-rPPG)', 'type', 'MAE (bpm)', 'source'])
            s.append('<p>Published results on UBFC-rPPG (full dataset, protocols differ; values as reported in the cited work):</p>')
            s.append(table_html(pub))
            classic = 4.06
            verdict = ('comparable to' if abs(m_all['mae'] - classic) <= 1.0 else ('better than' if m_all['mae'] < classic else 'worse than'))
            s.append(f'<div class="note"><b>Verdict (generated from the numbers).</b> On all windows our MAE is {m_all["mae"]:.2f} bpm, '
                     f'{verdict} the classic CHROM/POS methods (~4.1 bpm) and {"worse than" if m_all["mae"] > 1.3 else "comparable to"} '
                     f'trained deep models (~1.0–1.3 bpm). Our pulse extractor is POS, so similar all-window accuracy is expected. '
                     f'The difference is the validity test: on the {m_val["windows"]} validated windows the MAE drops to '
                     f'{m_val["mae"]:.2f} bpm ({m_val["within5"]:.0f} % within ±5 bpm), while the rejected windows have '
                     f'{m_rej["mae"]:.2f} bpm. The validated-only MAE must not be compared directly with the published full-video numbers, '
                     f'because it covers only {val_pct:.0f} % of the windows. Our evaluation is per 8 s window on '
                     f'{len(videos)} subjects; many papers use one HR per video on all subjects.</div>')
        else:
            s.append('<div class="note">No published benchmark is configured for this dataset; the table compares validated, all and rejected '
                     'windows of this project.</div>')
    else:
        s.append('<div class="note warn">No reference HR was given for this run, so accuracy cannot be compared with existing work. '
                 'Only the validation rate and HR values are reported (Section 9). Use a dataset with reference HR (e.g. UBFC-rPPG, '
                 'or NICU recordings with monitor readings) for this section.</div>')
    S.append(''.join(s))

    # ------------------------------------------------------------- Section 8: validation
    S.append(section_validation(hr, err, res, HR, hcfg, videos, has_ref))

    # ------------------------------------------------------------- Section 9: results
    S.append(section_results(hr, err, videos, manifest, has_ref, a.reference_name, cards_html))

    # ------------------------------------------------------------- Section 10: limitations
    lim = []
    lim.append(f'<li><b>Sample size:</b> {len(videos)} video(s), {n_win} windows. 8 s windows with a 1 s step overlap, so windows are not '
               'independent; confidence intervals over windows would be too narrow.</li>')
    if 'ubfc' in (a.dataset + run_name).lower():
        lim.append('<li><b>Adults, not infants:</b> UBFC-rPPG shows adult faces at rest in front of a webcam. It checks the signal chain; it is '
                   'not an infant or NICU result. The HR band was set to the adult range for this run.</li>')
    if has_ref:
        lim.append(f'<li><b>Coverage:</b> only {val_pct:.0f} % of windows are validated. '
                   f'{(100 * (err[err.label != "validated"].error.abs() <= 5).mean()):.0f} % of the rejected windows were in fact within ±5 bpm, '
                   'so the test is conservative and gives up some correct windows.</li>')
    lim += ['<li><b>Infant evaluation without reference:</b> the 34 infant web videos used to develop Module 1 v2 have no reference HR; '
            'their ROI metrics are label-free proxies. Web / social-media clips are re-compressed, which removes most of the pulse.</li>',
            '<li><b>Module 1 v2 is not better on pulse SNR than a fixed box</b> on the infant benchmark (Section 6); its gains are stability '
            'and anatomical labelling.</li>',
            '<li><b>Offline Viterbi:</b> configuration D5 uses the whole-video Viterbi path; an online fixed-lag variant (D3L) exists but is not '
            'the default.</li>',
            '<li><b>Speed:</b> the Sapiens pose model runs at about 1.5 frames/s on an A100 for 640×480 video; real-time use needs a lighter pose '
            'model or frame striding (Module 1 v2 Phase 5).</li>',
            '<li><b>Caregiver contact</b> (infant notebook) misses gloved hands when no adult body is visible.</li>',
            '<li><b>Fast HR changes</b> within an 8 s window spread the spectral peak and lower SNR, so such windows are often not validated.</li>']
    S.append('<h2 id="s10">10. Limitations</h2><ul>' + ''.join(lim) + '</ul>')
    S.append('<h2 id="refs">References</h2><ol class="small">' +
             ''.join(f'<li id="{k}">[{k}] {html.escape(t)}</li>' for k, t in REFERENCES) + '</ol>'
             '<p class="small muted">Literature values are copied from the cited papers and are the only numbers in this report not produced '
             'by this project. Check them against the originals before citing.</p>')

    toc = ''.join(f'<li><a href="#s{i}">{t}</a></li>' for i, t in enumerate(
        ['Architecture', 'Existing work', 'Research gap', 'Our contribution', 'Methods', 'Dynamic ROI vs existing',
         'HR measurement vs existing', 'Window validation', 'Results', 'Limitations'], 1))
    generated = dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    head = (f'<h1>{html.escape(a.title)}</h1><p class="muted">Project report generated {generated} from run '
            f'<code>{html.escape(run_name)}</code> ({html.escape(a.dataset) if a.dataset else "dataset not named"}); '
            f'results folder <code>{html.escape(str(res.relative_to(ROOT)) if str(res).startswith(str(ROOT)) else str(res))}</code>. '
            'All numbers and data figures come from this project\'s outputs.</p>' + cards_html +
            f'<nav class="toc"><b>Contents</b><ol>{toc}</ol></nav>')
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>HR Project Report</title><style>{CSS}</style></head><body><main>{head}{"".join(S)}</main></body></html>')
    out = Path(a.out) if a.out else res/'report'/'Project_Report.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding='utf-8')
    print(f'report: {out} ({out.stat().st_size / 1e6:.1f} MB)')
    from report_html_to_docx import convert                     # same content as a Word document
    docx_path = convert(out, out.with_suffix('.docx'))
    print(f'report (Word): {docx_path} ({docx_path.stat().st_size / 1e6:.1f} MB)')
    return 0


# ------------------------------------------------------------------------------------------------ section builders
def roi_frame_figure(run, run_name, cfg_name, hr, manifest, videos):
    """A real frame of this run with all ROI candidates (thin), the Module 1 v2 selection (coloured) and the HR regions."""
    src = ROOT/'module1_v2/runs'/run_name/f'dynamic_roi_v2_{cfg_name}'/'dynamic_roi_v2.csv'
    if not src.is_file() or not len(manifest):
        return ''
    video = videos[0]
    usecols = ['video_id', 'frame_index', 'timestamp_sec', 'roi_name', 'available', 'pixel_polygon', 'selected', 'weight']
    rows = []
    for chunk in pd.read_csv(src, usecols=usecols, chunksize=50000):
        rows.append(chunk[chunk.video_id == video])
    d = pd.concat(rows)
    h = hr[(hr.video_id == video) & (hr.label == 'validated')]
    t = float(h.window_center_sec.iloc[len(h) // 2]) if len(h) else float(d.timestamp_sec.median())
    frame_idx = int(d.iloc[(d.timestamp_sec - t).abs().argsort().iloc[0]].frame_index)
    f = d[d.frame_index == frame_idx]
    path = manifest.set_index('video_id').video_path.get(video)
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok, img = cap.read(); cap.release()
    if not ok:
        return ''
    hr_regions = {}
    if len(h):
        row = h.iloc[(h.window_center_sec - t).abs().argsort().iloc[0]]
        hr_regions = json.loads(row.hr_regions) if isinstance(row.hr_regions, str) else {}
    over = img.copy()
    scale = max(1., 900 / img.shape[1])
    for r in f.itertuples():
        if not r.available or not isinstance(r.pixel_polygon, str):
            continue
        pts = np.asarray(json.loads(r.pixel_polygon), float)
        if pts.ndim != 2 or len(pts) < 2:
            continue
        if len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        p = pts.astype(np.int32).reshape(-1, 1, 2)
        if r.roi_name in hr_regions:
            cv2.fillPoly(over, [p], (60, 60, 230))
        cv2.polylines(img, [p], True, (210, 210, 210), 1, cv2.LINE_AA)
        if bool(r.selected):
            cv2.polylines(img, [p], True, (60, 200, 60), 2, cv2.LINE_AA)
            c = pts.mean(0).astype(int)
            cv2.putText(img, f'{r.roi_name} {r.weight:.2f}', (int(c[0]) - 30, int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, .38, (40, 160, 40), 1, cv2.LINE_AA)
    img = cv2.addWeighted(over, .35, img, .65, 0)
    for name in hr_regions:
        r = f[f.roi_name == name]
        if len(r) and isinstance(r.iloc[0].pixel_polygon, str):
            pts = np.asarray(json.loads(r.iloc[0].pixel_polygon), float)
            if pts.ndim == 2 and len(pts) >= 3:
                cv2.polylines(img, [pts.astype(np.int32).reshape(-1, 1, 2)], True, (40, 40, 220), 2, cv2.LINE_AA)
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img_html(img, f'Figure 4. Real frame from this run ({video}, t = {t:.1f} s, frame {frame_idx}). Grey: all Module 1 ROI candidates; '
                         f'green: Module 1 v2 selected ROIs with weights; red fill: the HR regions used in this window '
                         f'({", ".join(hr_regions) or "none"}).')


def section_roi(rep, run, run_name, cfg_name, hr):
    s = ['<h2 id="s6">6. Is our dynamic ROI selection better than existing approaches?</h2>']
    # (a) on this run: v1 per-frame top-1 vs v2 selection switches
    src = ROOT/'module1_v2/runs'/run_name/f'dynamic_roi_v2_{cfg_name}'/'dynamic_roi_v2.csv'
    if src.is_file():
        d = pd.read_csv(src, usecols=['video_id', 'frame_index', 'timestamp_sec', 'roi_name', 'available', 'q_v1', 'selected', 'rank'])
        rows = []
        for v, g in d.groupby('video_id'):
            g = g[g.available]
            span = g.timestamp_sec.max() - g.timestamp_sec.min()
            top_v1 = g.sort_values('q_v1', ascending=False).groupby('frame_index').roi_name.first().sort_index()
            top_v2 = g[g.selected & (g['rank'] == 1)].groupby('frame_index').roi_name.first().sort_index()
            sw1 = (top_v1 != top_v1.shift()).iloc[1:].sum() / span if span > 0 else np.nan
            sw2 = (top_v2 != top_v2.shift()).iloc[1:].sum() / span if span > 0 else np.nan
            h = hr[hr.video_id == v]
            hsw = h.region_switch.astype(bool).sum() / max(1., h.window_center_sec.max() - h.window_center_sec.min()) if 'region_switch' in h else np.nan
            rows.append({'video': v, 'v1 per-frame top-1 switches/s': sw1, f'v2 ({cfg_name}) top-1 switches/s': sw2,
                         'HR region-set switches/s': hsw})
        t = pd.DataFrame(rows)
        s.append('<h3>6.1 On this run</h3><p>Top-1 ROI changes per second, computed from this run\'s Module 1 output: v1 picks the best-quality ROI '
                 'independently in each frame; v2 uses the stabilized selection. Fewer switches = fewer artificial jumps in the pixel trace.</p>')
        s.append(table_html(t))
        fig, ax = plt.subplots(figsize=(7, 2.8))
        x = np.arange(len(t)); w = .38
        ax.bar(x - w / 2, t.iloc[:, 1], w, color=C_REJ, label='v1 per-frame top-1')
        ax.bar(x + w / 2, t.iloc[:, 2], w, color=C_VAL, label=f'v2 {cfg_name}')
        ax.set_xticks(x, [v.replace('NEW_', '') for v in t.video], fontsize=8); ax.set_ylabel('top-1 switches / s', fontsize=8)
        ax.grid(axis='y', alpha=.3); ax.legend(fontsize=8, frameon=False); ax.spines[['top', 'right']].set_visible(False)
        s.append(fig_html(fig, 'Figure 5. ROI switching on this run: v1 (per-frame) vs v2 (stabilized).', 640))
    # (b) infant benchmark
    if rep and (rep/'phase1/table_phase1_main.csv').is_file():
        p1 = pd.read_csv(rep/'phase1/table_phase1_main.csv'); p1 = p1.set_index(p1.columns[0])
        p2 = pd.read_csv(rep/'phase2/table_phase2_main.csv') if (rep/'phase2/table_phase2_main.csv').is_file() else None
        p2 = p2.set_index(p2.columns[0]) if p2 is not None else None
        names = {'B0': 'fixed box (classic)', 'B2': 'anatomy dynamic (no quality)', 'Full': 'Module 1 v1 (per-frame)', 'D3': 'v2 stabilizer (D3)',
                 'D5': 'v2 + keypoint visibility (D5)'}
        rows = []
        for c, n in names.items():
            if c not in p1.index:
                continue
            r = dict(config=c, description=n)
            for col, lab in (('switches_per_sec', 'switches/s'), ('temporal_iou', 'temporal IoU'), ('mask_support', 'mask support'),
                             ('pulse_snr_db', 'pulse SNR (dB)'), ('informative_window_percent', 'informative %')):
                r[lab] = parse_pm(p1.loc[c, col])[0] if col in p1.columns else np.nan
            if p2 is not None and c in p2.index:
                col = [k for k in p2.columns if 'present_shuffle' in k and 'ci95' not in k]
                r['pulse-present % (null)'] = parse_pm(p2.loc[c, col[0]])[0] if col else np.nan
            rows.append(r)
        t = pd.DataFrame(rows)
        n_vid = int(p1['videos'].iloc[0]) if 'videos' in p1 else ''
        s.append(f'<h3>6.2 Infant benchmark (Module 1 v2 development set, {n_vid} web videos, mean over videos)</h3>'
                 f'<p class="small muted">Read from <code>{html.escape(str(rep.relative_to(ROOT)))}</code>; it changes only when that benchmark is re-run. '
                 'No reference HR exists for these videos, so SNR and pulse-present % are label-free proxies.</p>')
        s.append(table_html(t, {'switches/s': '{:.2f}', 'temporal IoU': '{:.3f}', 'mask support': '{:.3f}', 'pulse SNR (dB)': '{:.2f}',
                                'informative %': '{:.1f}', 'pulse-present % (null)': '{:.1f}'}))
        fig, axes = plt.subplots(1, 3, figsize=(11, 3))
        for ax, col, lab in zip(axes, ['switches/s', 'pulse SNR (dB)', 'pulse-present % (null)'], ['ROI switches / s (lower = stabler)',
                                                                                                    'pulse SNR proxy (dB)', 'pulse-present % (chance 5 %)']):
            if col not in t:
                ax.axis('off'); continue
            colours = [C_VAL if c.startswith('D') else C_REJ for c in t.config]
            ax.bar(t.config, t[col], color=colours)
            ax.set_title(lab, fontsize=9, loc='left'); ax.tick_params(labelsize=8); ax.grid(axis='y', alpha=.3)
            ax.spines[['top', 'right']].set_visible(False)
            if 'present' in col:
                ax.axhline(5, color=C_ACC, ls='--', lw=1)
        fig.tight_layout()
        s.append(fig_html(fig, 'Figure 6. Infant benchmark: blue = Module 1 v2 configurations, grey = alternatives.'))
        def g(c, k):
            return t.set_index('config').get(k, pd.Series()).get(c, np.nan)
        b0, d5, full = 'B0', cfg_name if cfg_name in set(t.config) else 'D5', 'Full'
        s.append('<div class="note"><b>Verdict (generated from the numbers).</b> '
                 f'Compared with Module 1 v1 the stabilized selection cuts ROI switching from {g(full, "switches/s"):.2f} to '
                 f'{g(d5, "switches/s"):.2f} per second and raises temporal IoU from {g(full, "temporal IoU"):.3f} to {g(d5, "temporal IoU"):.3f}. '
                 f'Against a fixed box (the classic choice) the pulse SNR proxy is {g(d5, "pulse SNR (dB)"):.2f} vs {g(b0, "pulse SNR (dB)"):.2f} dB: '
                 + ('the dynamic ROI is <b>not</b> better on SNR. ' if g(d5, 'pulse SNR (dB)') <= g(b0, 'pulse SNR (dB)') else 'the dynamic ROI is higher on SNR. ')
                 + 'So the honest claim is: our dynamic ROI is the most <i>stable</i> anatomy-aware selection and it tells the HR and RR branches '
                 '<i>which body part</i> each pixel belongs to (face for HR, chest/abdomen for RR, exclusion of covered regions); it is not '
                 'claimed to maximise pulse SNR by itself.</div>')
    else:
        s.append('<div class="note warn">Module 1 v2 benchmark tables not found; only the run-level comparison is shown.</div>')
    s.append('<p><b>Compared with existing ROI approaches:</b> a fixed face box [R1–R4] cannot follow an infant who turns, is covered, or shows '
             'skin mainly on the chest; per-region weighting such as DistancePPG [R5] and matrix completion [R6] works on adult faces and '
             'has no temporal stability control; neonatal skin segmentation [R9] finds skin but not anatomy. Module 1 v2 combines anatomy, '
             'mask and skin with an explicit, tuned-once stability control.</p>')
    return ''.join(s)


def section_validation(hr, err, res, HR, hcfg, videos, has_ref):
    s = ['<h2 id="s8">8. Window validation: what it is, how it works, why the NICU needs it</h2>',
         '<p>Every 8 s window gets an HR estimate, but an FFT always has <i>some</i> highest peak, even on pure noise. The validity test asks '
         'two questions before an HR is shown:</p>', SVG_VALIDITY,
         '<p class="small muted">Figure 7. Two-gate decision per window. Gate 1 controls the false-alarm rate (≤ 5 % of noise windows pass); '
         'gate 2 requires two disjoint skin areas to report the same rate (a local artefact rarely appears on both sides).</p>',
         f'<p><b>Gate 1, pulse present.</b> The time series of the same pixels is cut into {hcfg.get("shuffle_block_sec", .25)} s blocks that are '
         f'shuffled in time. This keeps the colour statistics and short-term noise but destroys any heart rhythm. The complete measurement '
         f'(region selection, POS, FFT, SNR) is repeated on {hcfg.get("surrogates", 49)} such surrogates; the window passes if its SNR exceeds '
         f'the {hcfg.get("null_percentile", 95):.0f}th percentile of the surrogate SNRs. Because region selection is repeated on the surrogates too, '
         'picking the luckiest region on noise cannot validate itself.</p>'
         f'<p><b>Gate 2, agreement.</b> The selected regions are split into left and right halves; each half gives its own HR. They must agree within '
         f'{hcfg.get("agreement_bpm", 6):.0f} bpm.</p>']
    # example windows with recomputed spectra
    ex = example_windows(hr, res, HR, hcfg)
    if ex:
        s.append(ex)
    # timeline SNR vs null for each video
    fig, axes = plt.subplots(len(videos), 1, figsize=(9, 1.9 * len(videos) + .4), sharex=True, squeeze=False)
    for ax, v in zip(axes[:, 0], videos):
        p = hr[hr.video_id == v]
        ax.plot(p.window_center_sec, p.null_threshold_db, color=C_ACC, lw=1.2, ls='--', label='null threshold (95th pct)')
        for lab, c, m in (('validated', C_VAL, 'o'), ('present', C_PRES, 's'), ('not_assessable', C_REJ, 'o')):
            q = p[p.label == lab]
            ax.scatter(q.window_center_sec, q.snr_db, s=16, color=c, marker=m, label=lab.replace('_', ' '), zorder=3)
        ax.set_ylabel('SNR (dB)', fontsize=8); ax.set_title(v, fontsize=8, loc='left'); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=4, loc='lower right'); axes[-1, 0].set_xlabel('time (s)', fontsize=8)
    fig.tight_layout()
    s.append(fig_html(fig, 'Figure 9. Every window of this run: its pulse SNR (dots) against its own surrogate-null threshold (dashed). '
                           'Windows below the line fail gate 1.'))
    # reasons table
    rt = hr.groupby(['video_id', 'reason']).size().unstack(fill_value=0).reset_index()
    s.append('<h3>Why windows were not validated (this run)</h3>' + table_html(rt))
    if has_ref:
        fig, ax = plt.subplots(figsize=(7, 3))
        groups = [('validated', err[err.label == 'validated'].error.abs()), ('not validated', err[err.label != 'validated'].error.abs())]
        rng = np.random.default_rng(0)
        for i, (lab, e) in enumerate(groups):
            ax.scatter(i + rng.uniform(-.18, .18, len(e)), e, s=12, alpha=.6, color=C_VAL if i == 0 else C_REJ)
            if len(e):
                ax.hlines(e.median(), i - .3, i + .3, color=C_REF, lw=2)
        ax.axhline(5, color=C_ACC, ls='--', lw=1); ax.text(1.45, 5.5, '±5 bpm', fontsize=7, color=C_ACC, ha='right')
        ax.set_xticks([0, 1], [f'{l}\n(n = {len(e)}, MAE {e.mean():.1f})' for l, e in groups], fontsize=8)
        ax.set_ylabel('|camera − reference| (bpm)', fontsize=8); ax.grid(axis='y', alpha=.3); ax.spines[['top', 'right']].set_visible(False)
        s.append(fig_html(fig, 'Figure 10. Absolute error of validated vs not-validated windows in this run (black bar = median). '
                               'The validity test removes the large errors.', 620))
    s.append('''<h3>Why this matters in the NICU</h3><ul>
<li><b>A wrong HR is dangerous.</b> A false normal value can hide bradycardia; a false abnormal value causes alarm fatigue. "Not assessable"
is safe: staff fall back on the contact monitor.</li>
<li><b>NICU video is often not measurable:</b> handling, phototherapy light, incubator reflections, blankets, tubes and crying. The test
detects these windows without a model of each artefact.</li>
<li><b>Known false-alarm rate:</b> by construction at most about 5 % of windows without a pulse pass gate 1, independent of dataset, so the
threshold does not need re-tuning per unit or camera.</li>
<li><b>Auditable:</b> each window carries its SNR, null threshold, left/right HR and reason, so clinicians and reviewers can see why a value was
shown or withheld.</li></ul>
<h3>Has anything like this been proposed before?</h3>
<p>Signal-quality gating itself is common: SNR or SQI thresholds [R3, R11], quality-based exclusion in neonatal studies [R8] and living-skin
classification [R10]. Surrogate-data testing is a classic statistical tool [R12]. We did not find an rPPG method that validates each window
with a surrogate null of the same pixels that repeats the full region-selection procedure, combined with a split-half agreement test.
This should be stated as "to our knowledge" and checked with a systematic literature search before submission.</p>''')
    return ''.join(s)


def example_windows(hr, res, HR, hcfg):
    """Spectra of one validated, one not-assessable (and one disagreeing) window, recomputed from the saved fused pulse."""
    picks = []
    for lab, sel in (('validated', hr[hr.label == 'validated'].sort_values('snr_db', ascending=False)),
                     ('not assessable (gate 1 fails)', hr[hr.reason == 'pulse not above null'].sort_values('snr_db')),
                     ('present, halves disagree (gate 2 fails)', hr[hr.reason == 'halves disagree'])):
        if len(sel):
            picks.append((lab, sel.iloc[len(sel) // 3] if lab.startswith('validated') else sel.iloc[len(sel) // 2]))
    if not picks:
        return ''
    fig, axes = plt.subplots(2, len(picks), figsize=(4.2 * len(picks), 5.4), squeeze=False)
    for j, (lab, w) in enumerate(picks):
        z = np.load(res/'waveforms'/f'{w.video_id}.npz')
        x, fs = z['pulse'], float(z['pulse_fps'])
        a = int(round(w.window_start_sec * fs)); b = a + int(round(HR.estimation_window_sec * fs))
        seg = x[a:b]
        seg = seg[np.isfinite(seg)]
        ax = axes[0, j]
        if len(seg) < fs * 4:
            ax.axis('off'); axes[1, j].axis('off'); continue
        seg = seg - seg.mean()
        nfft = 1 << 14
        f = np.fft.rfftfreq(nfft, 1 / fs) * 60; pw = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft)) ** 2
        band = (f >= HR.hr_min_bpm) & (f <= HR.hr_max_bpm)
        ax.plot(f[band], pw[band] / pw[band].max(), color=C_VAL if lab.startswith('validated') else C_REJ, lw=1.4)
        ax.axvline(w.hr_bpm, color=C_REF, lw=1, ls=':')
        ax.set_title(f'{lab}\n{w.video_id}, t = {w.window_center_sec:.0f} s, HR {w.hr_bpm:.0f} bpm\n'
                     f'HR_L {w.hr_left_bpm:.0f} / HR_R {w.hr_right_bpm:.0f} bpm', fontsize=8, loc='left')
        ax.set_xlabel('frequency (bpm)', fontsize=8); ax.set_ylabel('power (norm.)', fontsize=8); ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        # the pipeline's own gate values for this window (logged in hr_windows.csv)
        ax2 = axes[1, j]
        thr, obs = float(w.null_threshold_db), float(w.snr_db)
        g1 = obs > thr
        diff = abs(float(w.hr_left_bpm) - float(w.hr_right_bpm)) if np.isfinite([w.hr_left_bpm, w.hr_right_bpm]).all() else np.nan
        lim = float(hcfg.get('agreement_bpm', 6))
        g2 = bool(np.isfinite(diff) and diff <= lim)
        ax2.barh([1], [obs], color=C_VAL if g1 else C_REJ, height=.5)
        ax2.axvline(thr, color=C_ACC, ls='--', lw=1.2)
        ax2.text(thr, 1.38, f' null {thr:.2f} dB', color=C_ACC, fontsize=7, va='bottom')
        ax2.text(obs, 1, f' {obs:.2f} dB', fontsize=7, va='center', ha='left' if obs >= 0 else 'right')
        lo = min(0, obs, thr) - 1.5; hi = max(0, obs, thr) + 2.5
        ax2.set_xlim(lo, hi); ax2.set_ylim(.3, 1.7); ax2.set_yticks([]); ax2.axvline(0, color='#bbb', lw=.6)
        ax2.set_xlabel('pulse SNR of the window (dB)', fontsize=8); ax2.tick_params(labelsize=7)
        ax2.spines[['top', 'right', 'left']].set_visible(False)
        ax2.set_title(f'gate 1: SNR {obs:.2f} {">" if g1 else "≤"} null {thr:.2f} dB → {"PASS" if g1 else "FAIL"}\n'
                      f'gate 2: |HR_L − HR_R| = {diff:.1f} {"≤" if g2 else ">"} {lim:.0f} bpm → {"PASS" if g2 else "FAIL"}',
                      fontsize=8, loc='left')
    fig.tight_layout()
    return fig_html(fig, 'Figure 8. Three real windows of this run. Top: spectrum of the fused pulse (dotted = estimated HR). Bottom: the '
                         'pipeline\'s own gate values for the window (from hr_windows.csv): the window SNR against the 95th percentile of its 49 '
                         'block-shuffle surrogates (dashed), and the left/right HR difference against the agreement limit.')


def section_results(hr, err, videos, manifest, has_ref, ref_name, cards_html):
    s = ['<h2 id="s9">9. Results</h2>', cards_html]
    rows = []
    for v in videos:
        p = hr[hr.video_id == v]
        r = {'video': v, 'windows': len(p), 'validated %': 100 * (p.label == 'validated').mean(),
             'present %': 100 * (p.label != 'not_assessable').mean(),
             'HR median, validated': p[p.label == 'validated'].hr_bpm.median(), 'SNR median (dB)': p.snr_db.median()}
        if has_ref:
            e = err[err.video_id == v]
            mv, ma = metrics(e[e.label == 'validated']), metrics(e)
            r.update({'reference HR median': e.hr_reference.median(), 'MAE validated': mv['mae'], 'MAE all': ma['mae'],
                      '±5 bpm % (val.)': mv['within5'], 'r (val.)': mv['r']})
        rows.append(r)
    t = pd.DataFrame(rows)
    s.append('<h3>9.1 Per video</h3>' + table_html(t, {'validated %': '{:.1f}', 'present %': '{:.1f}', '±5 bpm % (val.)': '{:.1f}'}))
    # per-video HR plots
    for v in videos:
        p = hr[hr.video_id == v]
        fig, ax = plt.subplots(figsize=(9, 2.7))
        if has_ref:
            e = err[err.video_id == v].sort_values('window_center_sec')
            ax.plot(e.window_center_sec, e.hr_reference, color=C_REF, lw=1.6, label=f'{ref_name}')
        o = p[p.label != 'validated']; q = p[p.label == 'validated']
        ax.scatter(o.window_center_sec, o.hr_bpm, s=18, facecolors='none', edgecolors=C_REJ, label='camera, not validated', zorder=3)
        ax.scatter(q.window_center_sec, q.hr_bpm, s=20, color=C_VAL, label='camera, validated', zorder=4)
        if len(q):
            ax.plot(p.window_center_sec, p.hr_kalman_bpm, color=C_VAL, lw=.8, alpha=.5)
        ax.set_title(f'{v}: validated {100 * (p.label == "validated").mean():.0f} % of windows', fontsize=9, loc='left')
        ax.set_xlabel('time (s)', fontsize=8); ax.set_ylabel('HR (bpm)', fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, frameon=False, ncol=3, loc='best'); ax.spines[['top', 'right']].set_visible(False)
        s.append(fig_html(fig, f'HR over time for {v}: filled = validated (reported), hollow = withheld, thin line = Kalman track.'))
    if has_ref:
        rows = []
        for name, e in (('validated windows', err[err.label == 'validated']), ('all windows', err), ('rejected windows', err[err.label != 'validated'])):
            m = metrics(e); m['subset'] = name; rows.append(m)
        mt = pd.DataFrame(rows)[['subset', 'windows', 'mae', 'rmse', 'within5', 'r']].rename(
            columns={'mae': 'MAE (bpm)', 'rmse': 'RMSE (bpm)', 'within5': '±5 bpm %', 'r': 'Pearson r'})
        s.append(f'<h3>9.2 Accuracy against the {html.escape(ref_name)} reference (all videos)</h3>' + table_html(mt, {'±5 bpm %': '{:.1f}'}))
        # Bland-Altman
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        cmap = plt.get_cmap('tab10'); col = {v: cmap(i % 10) for i, v in enumerate(videos)}
        ba = []
        for ax, (name, e) in zip(axes, (('validated windows', err[err.label == 'validated']), ('all windows', err))):
            mean = (e.hr_bpm + e.hr_reference) / 2; diff = e.error
            for v in videos:
                m = e.video_id == v
                ax.scatter(mean[m], diff[m], s=13, alpha=.75, color=col[v], label=v.replace('NEW_', ''), edgecolors='none')
            if len(e) > 1:
                bias, sd = diff.mean(), diff.std(ddof=1)
                for y, ls, lab in ((bias, '--', f'bias {bias:+.1f}'), (bias + 1.96 * sd, ':', f'+1.96 SD {bias + 1.96 * sd:+.1f}'),
                                   (bias - 1.96 * sd, ':', f'−1.96 SD {bias - 1.96 * sd:+.1f}')):
                    ax.axhline(y, color=C_REF, ls=ls, lw=1)
                    ax.annotate(lab, xy=(1, y), xycoords=('axes fraction', 'data'), xytext=(-4, 3), textcoords='offset points', ha='right',
                                fontsize=7, bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=.85))
                ba.append(dict(subset=name, windows=len(e), bias=bias, sd=sd, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd))
            ax.axhline(0, color='#888', lw=.6); ax.set_title(f'{name} (n = {len(e)})', fontsize=9, loc='left')
            ax.set_xlabel('mean of camera and reference HR (bpm)', fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        axes[0].set_ylabel('camera − reference (bpm)', fontsize=8)
        axes[1].legend(fontsize=7, frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        s.append(fig_html(fig, 'Figure 11. Bland–Altman agreement (dashed = bias, dotted = 95 % limits of agreement).'))
        s.append(table_html(pd.DataFrame(ba), {'bias': '{:+.2f}', 'sd': '{:.2f}', 'loa_low': '{:+.2f}', 'loa_high': '{:+.2f}'}))
        # scatter camera vs reference
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        o = err[err.label != 'validated']; q = err[err.label == 'validated']
        ax.scatter(o.hr_reference, o.hr_bpm, s=14, facecolors='none', edgecolors=C_REJ, label='not validated')
        ax.scatter(q.hr_reference, q.hr_bpm, s=14, color=C_VAL, label='validated')
        lo = np.nanmin([err.hr_reference.min(), err.hr_bpm.min()]) - 5; hi = np.nanmax([err.hr_reference.max(), err.hr_bpm.max()]) + 5
        ax.plot([lo, hi], [lo, hi], color=C_REF, lw=1); ax.fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5], color='#999', alpha=.12, lw=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel(f'{ref_name} HR (bpm)', fontsize=8); ax.set_ylabel('camera HR (bpm)', fontsize=8)
        ax.legend(fontsize=7, frameon=False); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        s.append(fig_html(fig, 'Figure 12. Camera vs reference HR per window (line = identity, band = ±5 bpm).', 460))
    # region usage
    use = {}
    for r in hr[hr.label == 'validated'].hr_regions.dropna():
        try:
            for k in json.loads(r):
                use[k] = use.get(k, 0) + 1
        except (ValueError, TypeError):
            pass
    if use:
        u = pd.DataFrame(sorted(use.items(), key=lambda kv: -kv[1]), columns=['region', 'validated windows using it'])
        s.append('<h3>9.3 Which skin regions gave the validated HR</h3>' + table_html(u))
    return ''.join(s)


if __name__ == '__main__':
    raise SystemExit(main())
