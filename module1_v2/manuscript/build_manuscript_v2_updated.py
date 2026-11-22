"""UPDATED Module 1 v2 manuscript: the original manuscript plus the reference-based robustness study (UBFC-rPPG).

Usage: python module1_v2/manuscript/build_manuscript_v2_updated.py --name final34_paper --v1-report reports/paper_final34_paper

Writes module1_v2/reports/<name>/Module1_v2_Manuscript_updated.docx and figures to manuscript_figures_updated/.
The original builder (build_manuscript_v2.py), its modules and Module1_v2_Manuscript.docx are NOT modified: this script
builds the same document and, through a Builder subclass, inserts Sections 4.6 and 5.9 and replaces the few paragraphs
listed in EDITS. Every edit must fire exactly once, otherwise the build stops.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'reports/paper/manuscript_build'))
from docbuild import Builder        # noqa: E402
import results_v2 as R              # noqa: E402
import sections as S                # noqa: E402
import robustness as RB             # noqa: E402

TITLE = ('Stable, Auditable Dynamic ROI Representation for Neonatal Video: Temporal Stabilization, Label-Free Validity '
         'Tests and Reference-Based Robustness to Motion, Swaddling and Occlusion')


# ---- naming: the internal "v1" / "v2" become the per-frame dynamic ROI (PF-ROI) / NeoSTAR in every text and figure
NAME_RULES = [
    ('(hereafter v1)', '(hereafter the per-frame dynamic ROI, PF-ROI)'),
    ('Stages marked v2 were changed', 'Stages marked "new" were changed'),
    ('(v2 = changed relative to the v1 baseline', '(new = changed relative to the PF-ROI baseline'),
    ('the v1 per-frame rule and the v2 stabilized selection', 'the PF-ROI per-frame rule and the NeoSTAR stabilized selection'),
    ('v1 per-frame top-k', 'PF-ROI (per-frame top-k)'),
    ('Module 1 v2 D5', 'NeoSTAR (D5)'),
    ('Module 1 v1', 'PF-ROI'),
    ("Module 1's FACE region", "the pipeline's FACE region"),
    ('v1 Full', 'PF-ROI (Full)'),
    ('old Module 1 (Full)', 'PF-ROI (Full)'),
]
NAME_RE = [(re.compile(r'(?<![\w./-])v1(?![\w.])'), 'PF-ROI'), (re.compile(r'(?<![\w./-])v2(?![\w.])'), 'NeoSTAR')]


def rename(text):
    if not isinstance(text, str):
        return text
    if text.strip() == 'v2':                       # Figure 1 stage badge: too small for "NeoSTAR"
        return text.replace('v2', 'new')
    for a, b in NAME_RULES:
        text = text.replace(a, b)
    for rx, b in NAME_RE:
        text = rx.sub(b, text)
    return text


def rename_figures():
    """Apply rename() to every matplotlib text drawn in this process (updated build only)."""
    import matplotlib.text as mtext
    if getattr(mtext.Text.set_text, '_renamed', False):
        return
    orig = mtext.Text.set_text

    def set_text(self, s):
        return orig(self, rename(s) if isinstance(s, str) else s)
    set_text._renamed = True
    mtext.Text.set_text = set_text


class RenamingBuilder(Builder):
    image_map = {}          # original figure file name -> redrawn file (Phase 1 figures, see phase1_figures_updated)

    def image(self, path, *a, **k):
        super().image(self.image_map.get(Path(path).name, path), *a, **k)

    def title(self, text):
        super().title(rename(text))

    def h1(self, text):
        super().h1(rename(text))

    def h2(self, text):
        super().h2(rename(text))

    def para(self, text, *a, **k):
        super().para(rename(text), *a, **k)

    def bullets(self, items):
        super().bullets([rename(x) for x in items])

    def caption(self, text):
        super().caption(rename(text))

    def table(self, header, rows, *a, **k):
        return super().table([rename(h) for h in header], [[rename(c) for c in r] for r in rows], *a, **k)


class UpdatedBuilder(RenamingBuilder):
    def __init__(self, template, rb, D):
        super().__init__(template)
        self.rb, self.D, self.fired = rb, D, set()
        self.rbm = RB          # robustness text module; the submission build substitutes its own

    def _hit(self, key):
        if key in self.fired:
            raise RuntimeError(f'edit fired twice: {key}')
        self.fired.add(key)

    def title(self, text):
        self._hit('title')
        super().title(TITLE)

    def h1(self, text):
        if text == '5. Results':
            self._hit('setup 4.6')
            self.rbm.setup_section(self, self.rb)
        if text == '6. Discussion':
            self._hit('results 5.9')
            self.rbm.results_section(self, self.rb)
        super().h1(text)

    def para(self, text, *a, **k):
        rb = self.rb
        if text.startswith('**Research question:**'):
            self._hit('research question')
            text = text.replace('more pulse evidence than fixed or alternative ROIs?',
                                'more pulse evidence than fixed or alternative ROIs, and, against a physiological '
                                'reference, under which conditions does it give more accurate heart rate than a fixed box?')
        elif text.startswith('**Abstract:**'):
            self._hit('abstract')
            text = '**Abstract:** ' + short_abstract(self.D, rb)
        elif text.startswith('**Keywords:**'):
            self._hit('keywords')
            text = text + ', motion robustness, swaddling, occlusion, UBFC-rPPG'
        elif text.startswith('This paper presents NeoSTAR'):
            self._hit('intro reference')
            old = 'no ground-truth physiological reference is used anywhere.'
            assert old in text
            text = text.replace(old, 'no ground-truth physiological reference is used for any design or tuning decision.')
            super().para(text, *a, **k)
            text = ('Label-free tests can show whether a pulse is present but not whether the heart rate is right. We '
                    'therefore add a reference-based robustness study: UBFC-rPPG recordings with fingertip-PPG heart rate, '
                    'perturbed by controlled, simulated fast motion, swaddling, swaddling with motion, occlusion and '
                    'lighting changes, and processed with the frozen neonatal settings. It asks the question a fixed crop '
                    'raises in neonatal care: when does a dynamic ROI give more accurate heart rate than a fixed box, and '
                    'when does it not?')
            super().para(text, *a, **k)
            self._hit('intro learning-free')
            text = INTRO_LEARNING_FREE
        elif text.startswith('Deep rPPG models learn spatial attention'):
            self._hit('related 2.2')
            text = DEEP_RELATED
        elif text.startswith('The pipeline has eight stages'):
            self._hit('method learning-free')
            text = text + ' ' + LEARNING_FREE
        elif text.startswith('**When does a dynamic ROI pay off?**'):
            self._hit('discussion deep')
            super().para(text, *a, **k)
            text = DEEP_DISCUSSION
        elif text.startswith('The neonatal videos have no physiological reference, so heart-rate'):
            self._hit('probe framing')
            text = text + ' ' + PROBE
        elif text.startswith('Prior neonatal studies establish clinical feasibility'):
            self._hit('research gap')
            text = text.replace('NeoSTAR targets exactly this gap.',
                                'Nor do they test, against a physiological reference, under which conditions a dynamic ROI '
                                'is worth its complexity compared with a fixed crop. NeoSTAR targets exactly this gap.')
        elif text.startswith('**Fixed boxes are hard to beat on short web video.**'):
            self._hit('discussion fixed box')
            super().para(self.rbm.discussion_paragraph(rb), *a, **k)
            self._hit('discussion deep')
            text = DEEP_DISCUSSION
        elif text.startswith('NeoSTAR turns an unstable'):
            self._hit('conclusion')
            i = text.index('The representation and the validity tests are released')
            text = text[:i] + self.rbm.conclusion_sentence(rb)
        elif text.startswith('The pipeline, the v2 evaluation scripts'):
            self._hit('data availability')
            text = DATA_AVAILABILITY
        elif text.startswith('The split is predeclared'):
            self._hit('ethics')
            i = text.index('**Ethics statement:**')
            text = text[:i] + ETHICS
        elif text.startswith('[48] R. Stricker'):
            self._hit('reference 49')
            super().para(text, *a, **k)
            text = self.rbm.NEW_REFERENCE
        super().para(text, *a, **k)

    def bullets(self, items):
        items = list(items)
        if items and items[-1].startswith('An efficiency study of the full pipeline'):
            self._hit('contributions')
            items.insert(len(items) - 1, self.rbm.contribution_bullet(self.rb))
        elif items and items[0].startswith('No physiological reference was available'):
            self._hit('limitations')
            items = self.rbm.LIMITATION_REPLACEMENT + items[1:] + [LIMIT_LEARNED]
        super().bullets(items)


EDITS = {'title', 'setup 4.6', 'results 5.9', 'research question', 'abstract', 'keywords', 'intro reference',
         'research gap', 'discussion fixed box', 'conclusion', 'data availability', 'reference 49', 'contributions',
         'limitations', 'intro learning-free', 'related 2.2', 'method learning-free', 'discussion deep', 'probe framing',
         'ethics'}
ABSTRACT_MAX_WORDS = 250


def short_abstract(D, rb):
    """Abstract of at most ABSTRACT_MAX_WORDS words; every number is computed, as in results_v2.abstract."""
    c, n, f1 = D['chosen'], len(D['manifest']), D['full1']
    m = lambda cid, col, ph=1: R.mean(D, cid, col, ph)  # noqa: E731
    pres = [m(x, 'present_shuffle_percent', 2) for x in ('B0', 'Full', c)]
    chance = 100 - D['p2']['null_percentile']
    b0_better = m('B0', 'pulse_snr_db') > m(c, 'pulse_snr_db')
    T = rb['T']
    row = lambda a, b: RB.row(rb, a, b)  # noqa: E731
    u, fx, sm = row('Undisturbed', 'none'), row('Fast, large motion', 'extra-large'), row('Swaddling + fast motion', 'to neck')
    text = (
        'Camera-based neonatal monitoring needs a region of interest (ROI) that follows the infant, yet dynamic ROI '
        'selection switches between near-equal regions from frame to frame. We present NeoSTAR, a learning-free, auditable '
        'dynamic ROI representation that adds temporal stabilization (confidence smoothing, hysteresis and Viterbi '
        'decoding; four settings chosen on training videos by a colour-free objective) and an independent keypoint-'
        f'visibility term to an anatomy- and mask-guided pipeline. On {n} unconstrained newborn videos with a video-level '
        f"split, stabilization reduced region switching from {m('Full', 'switches_per_sec'):.1f} to "
        f"{m(c, 'switches_per_sec'):.2f} switches/s ({m('D3L', 'switches_per_sec'):.2f} with a 1-s online lag) and raised "
        f"temporal IoU from {m('Full', 'temporal_iou'):.2f} to {m(c, 'temporal_iou'):.2f} (Holm-corrected p "
        f"{R.p(R.pair(f1, 'switches_per_sec', c).p_holm)}), while retaining at least "
        f"{100 * min(D['retention_split']):.0f}% of region confidence. Two label-free validity tests, a surrogate-null "
        'pulse-present test and split-half heart-rate agreement, showed that pulse evidence in these web videos is weak: '
        f'only {min(pres):.0f}-{max(pres):.0f}% of windows exceeded the surrogate null (chance {chance:.0f}%), and NeoSTAR '
        + ('did not exceed' if b0_better else 'matched') + ' a fixed box. Against a fingertip-PPG reference on UBFC-rPPG '
        'with controlled perturbations and frozen settings, NeoSTAR matched a fixed box on undisturbed video '
        f'({u.mae_D5:.1f} vs. {u.mae_fixed:.1f} bpm) and prevented its failure under fast, large motion ({fx.mae_D5:.1f} '
        f'vs. {fx.mae_fixed:.1f} bpm) and swaddling with motion ({sm.mae_D5:.1f} vs. {sm.mae_fixed:.1f} bpm; '
        'pre-registered); with swaddling alone or hand-and-blanket occlusion the fixed box was as accurate or better. A '
        f'tracked face box was never more accurate than NeoSTAR and was less accurate in '
        f'{int((T.face_verdict == "NeoSTAR better").sum())} of {len(T)} conditions. NeoSTAR thus provides a stable, '
        'inspectable infant ROI whose heart-rate benefit appears when the subject moves.')
    assert (T.face_verdict != 'face box better').all()
    words = len(text.split())
    if words > ABSTRACT_MAX_WORDS:
        raise SystemExit(f'abstract has {words} words (limit {ABSTRACT_MAX_WORDS})')
    return text


ETHICS = (
    '**Ethics statement:** The neonatal videos were publicly accessible at the time of collection and were posted by their '
    'uploaders; no recording was made for this study, no family or uploader was contacted, and no personal metadata '
    '(names, accounts, locations) were stored. Only derived numerical data (ROI geometry, colour traces and window '
    'statistics) were analysed, and the videos are not redistributed. UBFC-rPPG was used under its licence. [AUTHORS TO '
    'CONFIRM: the institutional review board determination for the use of public web videos (e.g. exempt or not human-'
    'subjects research, with protocol number), the platform terms of use under which the videos were accessed, and that '
    'every infant face shown in a figure (e.g. Figure 1) is blurred or shown with permission.]')
DATA_AVAILABILITY = (
    'The pipeline, the v2 evaluation scripts (stability, validity-test and efficiency analyses), the configuration files, '
    'the tuning logs, all per-video result tables, the perturbation generators with their seeds, the pre-registrations '
    'and the per-window heart-rate tables of the robustness study are released with the paper, together with the '
    'manifest of the neonatal videos with the SHA-256 hash of every file, so that the same files can be verified. The '
    'videos themselves are not redistributed; UBFC-rPPG is available from its authors [47]. [AUTHORS TO COMPLETE: '
    'repository URL, and the source link of each video in the manifest (the current manifest holds file names and '
    'hashes only).]')

# ---- "why not deep learning" defence (paper 1 is about the ROI representation; HR/RR extractors are paper 2)
INTRO_LEARNING_FREE = (
    'NeoSTAR is learning-free in a precise sense: its perception stages are pretrained deep models (detectors, a video '
    'segmenter and a pose model), but no model is trained or fine-tuned on neonatal data. This is a deliberate choice. '
    'No ROI ground truth and no physiological reference exist for the neonatal videos, so a model trained for this task '
    'would have nothing to learn from and nothing to be checked against; clinical use requires that every region choice '
    'can be inspected and explained; and an explicit, anatomically labelled representation can feed any downstream '
    'signal extractor, classical or learned, for heart rate from facial skin and for respiration from the chest and '
    'abdomen.')
DEEP_RELATED = (
    'Deep rPPG models learn spatial attention and motion robustness end to end [4]-[10], [13]-[18]. They do not remove '
    'the ROI problem; they move it upstream: each model expects a face crop from a separate detector or tracker, and the '
    'crop defines what the network can see. Their spatial attention is implicit, optimized for a downstream physiological '
    'estimate, not constrained to be stable over time and not open to inspection; it requires physiological labels for '
    'training; the models cover the face only, whereas respiration is read from the chest and abdomen; and they are '
    'trained almost exclusively on adult faces, with a documented adult-to-infant domain gap [24]-[26]. NeoSTAR is '
    'therefore complementary: it provides the explicit, temporally tracked, infant-specific input representation such '
    'models consume, without labels, and a learned extractor or reliability model can be placed on top of it once '
    'labelled neonatal data exist.')
LEARNING_FREE = ('Here "learning-free" means that only pretrained perception models are used, no model is trained or '
                 'fine-tuned on the target data, and the only data-dependent choices are the four stabilizer settings.')
DEEP_DISCUSSION = (
    '**Why not an end-to-end learned model?** NeoSTAR already relies on deep learning for perception; what it avoids is a '
    'task-specific model trained on neonatal video. Four reasons support this for an ROI representation. First, no '
    'neonatal ROI or physiological labels exist here, so such a model could be neither trained nor validated. Second, an '
    'end-to-end rPPG network still needs an upstream crop and fails when the subject leaves it, which is the failure the '
    'fixed box showed under motion (Section 5.9); the representation is a prerequisite for, not an alternative to, a '
    'learned extractor. Third, its attention is implicit and changes from frame to frame, whereas NeoSTAR states which '
    'anatomical region is used, why, and how stably, which is what a clinical audit needs. Fourth, it covers the face '
    'only, while neonatal respiration requires chest and abdominal regions. Comparing learned and classical signal '
    'extractors on top of the representation, and training a learned reliability model once labelled neonatal data '
    'exist, are the natural next steps.')
LIMIT_LEARNED = ('No learned ROI baseline was compared, because no neonatal ROI or physiological labels exist to train or '
                 'evaluate one; learned signal extractors are outside the scope of this representation study.')
PROBE = ('Heart rate is used here as a probe of ROI quality: the same pixel rules and the same classical extractor are '
         'applied to every ROI, so differences in accuracy reflect the ROI, not the extractor.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--v1-report', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--headline', default=None)
    args = ap.parse_args()
    v1 = Path(args.v1_report) if args.v1_report else ROOT/'reports'/f'paper_{args.name}'
    v1 = v1 if v1.is_absolute() else ROOT/v1
    D = R.load(args.name, v1, R.decided_headline(ROOT/'module1_v2/reports'/args.name, args.headline))
    out = Path(args.out) if args.out else D['rep']/'Module1_v2_Manuscript_updated.docx'
    out = out if out.is_absolute() else ROOT/out
    original = D['rep']/'Module1_v2_Manuscript.docx'
    if out.resolve() == original.resolve():
        raise SystemExit('refusing to overwrite the original manuscript')
    figs = D['rep']/'manuscript_figures_updated'
    rename_figures()
    R.make_figures(D, figs)
    import make_figure1
    D['fig1'] = figs/'fig1_architecture.png'
    D['fig1_source'] = make_figure1.make(args.name, D['fig1'])
    rb = RB.load()
    RB.make_figures(rb, figs)
    import phase1_figures_updated as P1U
    RenamingBuilder.image_map = P1U.make(args.name, D['chosen'], figs)

    b = UpdatedBuilder(ROOT/'reports/paper/manuscript_build/orig_docx/basepaper.docx', rb, D)
    b.title('')
    b.para('**Research question:** Can a learning-free, anatomy- and mask-guided dynamic ROI representation be made '
           'temporally stable and internally consistent, and do label-free tests support that its selected pixels carry '
           'more pulse evidence than fixed or alternative ROIs?')
    b.para('**Abstract:** ' + R.abstract(D))
    b.para('**Keywords:** neonatal video, dynamic ROI, temporal stabilization, Viterbi decoding, remote photoplethysmography, '
           'surrogate data, label-free evaluation, non-contact monitoring')
    pv = D['pv1']
    D['sw'] = {x: float(pv.loc[pv.config == x, 'switches_per_sec'].mean()) for x in ('Full', 'D3', 'D3L', D['chosen'])}
    D['ret_min'] = float(pv[pv.config.isin(['D3', 'D3L']) & pv.split.isin(['validation', 'test'])]
                         .groupby(['config', 'split']).retention.mean().min())
    S.introduction(b, D)
    S.related_work(b)
    S.method(b, D)
    S.setup(b, D)
    R.results(b, D, figs)
    R.discussion(b, D)
    b.h1('References')
    for ref in S.REFERENCES:
        b.para(ref, size=10)
    left = [t for t in [p.text for p in b.doc.paragraphs] + [c.text for t in b.doc.tables for r in t.rows for c in r.cells]
            if re.search(r'(?<![\w./-])v[12](?![\w.])', t)]
    if left:
        raise SystemExit(f'"v1"/"v2" left in the text: {[x[:80] for x in left]}')
    missing = EDITS - b.fired
    if missing:
        raise SystemExit(f'edits that did not fire: {sorted(missing)}')
    body = b.doc.element.xml
    for rid, rel in list(b.doc.part.rels.items()):
        if rel.reltype.endswith('/image') and f'"{rid}"' not in body:
            b.doc.part.drop_rel(rid)
    out.parent.mkdir(parents=True, exist_ok=True)
    b.save(out)
    print('Wrote', out, '| headline config:', D['chosen'], '| edits:', len(b.fired))


if __name__ == '__main__':
    main()
