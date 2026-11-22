"""Journal submission package of the NeoSTAR paper (paper 1: ROI representation).

Usage:
  python module1_v2/manuscript/build_submission.py --name final34_paper --v1-report reports/paper_final34_paper \
      [--journal results_in_engineering] [--info module1_v2/manuscript/submission_info.json]

Builds the UPDATED manuscript (build_manuscript_v2_updated.py, unchanged) through a Builder subclass and adds:
  * the chosen title and the author block (submission_info.json);
  * the expanded Related Work with a literature comparison table (related_work_sub.py);
  * the pilot framing of the robustness study, the designed hand-over-face case moved to the supplement, per-subject
    results, stronger limitations and future work (robustness_sub.py);
  * the journal's format (journals/<journal>.py): abstract and keyword limits, citation order and style, reference
    style, line numbers, line spacing, page numbers and declarations;
  * separate files: highlights, cover letter, supplementary material, video manifest (Data S1), figures, checklist.
To change journal, add a profile in journals/ and pass --journal; nothing else changes.
The original manuscript and the updated manuscript are not touched.
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'reports/paper/manuscript_build'))
import build_manuscript_v2_updated as U    # noqa: E402
from build_manuscript_v2_updated import UpdatedBuilder, RenamingBuilder, rename_figures, LIMIT_LEARNED  # noqa: E402,F401
from docbuild import Builder               # noqa: E402
import results_v2 as R                     # noqa: E402
import sections as S                       # noqa: E402
import robustness_sub as RS                # noqa: E402
import related_work_sub as RW              # noqa: E402
import references as REF                   # noqa: E402
import journals                            # noqa: E402

TITLE = 'NeoSTAR: A Stable, Auditable Dynamic ROI Representation for Neonatal Video'
KEYWORDS = ['neonatal video', 'dynamic region of interest', 'temporal stabilization', 'remote photoplethysmography',
            'label-free evaluation', 'motion robustness', 'non-contact monitoring', 'Viterbi decoding',
            'surrogate data', 'UBFC-rPPG']
ORIGINAL_MD5 = 'ae820cf0fb3c82bb3ce4f0e2a73932dc'
# edits of the updated builder that this build replaces by its own text
REPLACED = {'related 2.2', 'research gap', 'reference 49', 'data availability'}
TABLE_RE = re.compile(r'\bTable (\d+)\b')
FILL_RE = re.compile(r'FILL[^"\]]*')


def fill(v):
    """Visible marker for a value the authors still have to supply."""
    return f'[{v}]' if isinstance(v, str) and v.startswith('FILL') else v


class SubmissionBuilder(UpdatedBuilder):
    """Updated manuscript + submission edits. `cite_map` None = collection pass (records citation order)."""

    def __init__(self, template, rb, D, P, info, cite_map=None):
        super().__init__(template, rb, D)
        self.rbm = RS
        self.P, self.info, self.cite_map = P, info, cite_map
        self.order, self.images, self.raw = [], [], False

    # ---------------------------------------------------------------- text transform (every run and table cell)
    def tx(self, text):
        if not isinstance(text, str) or self.raw:     # raw: reference list entries keep their own numbers
            return text
        if self.cite_map is None:
            for g in REF.CITE_GROUP.finditer(text):
                for n in REF.numbers_in(g.group(0)):
                    if n not in self.order:
                        self.order.append(n)
        elif self.P['citations']['order'] == 'appearance' or self.P['citations']['style'] != 'ieee':
            text = REF.CITE_GROUP.sub(lambda g: REF.format_group([self.cite_map[n] for n in REF.numbers_in(g.group(0))],
                                                                  self.P['citations']['style']), text)
        return shift_tables(text)

    def _runs(self, paragraph, text, size=12, italic=False):
        return super()._runs(paragraph, self.tx(text), size, italic)

    def table(self, header, rows, *a, **k):
        return super().table([self.tx(h) for h in header], [[self.tx(str(c)) for c in r] for r in rows], *a, **k)

    def caption(self, text):
        if text.startswith('Figure 1.') and self.D.get('fig1_caption'):
            text = self.D['fig1_caption']
        return super().caption(text)

    def image(self, path, *a, **k):
        self.images.append(Path(RenamingBuilder.image_map.get(Path(path).name, path)))
        super().image(path, *a, **k)

    # ---------------------------------------------------------------- front matter
    def title(self, text):
        self._hit('title')
        RenamingBuilder.title(self, TITLE)
        authors, aff = self.info['authors'], self.info['affiliations']
        p = self.doc.add_paragraph(style='normal')
        p.alignment = 1
        for i, a in enumerate(authors):
            self._plain(p, ('' if i == 0 else ', ') + fill(a['name']), 12)
            self._plain(p, ','.join(a['affiliations']) + ('*' if a.get('corresponding') else ''), 12, sup=True)
        for k, v in aff.items():
            q = self.doc.add_paragraph(style='normal')
            q.alignment = 1
            self._plain(q, k, 10, sup=True)
            self._plain(q, ' ' + fill(v), 10)
        corr = [a for a in authors if a.get('corresponding')]
        if corr:
            q = self.doc.add_paragraph(style='normal')
            q.alignment = 1
            self._plain(q, f"* Corresponding author: {fill(corr[0]['name'])}, {fill(corr[0]['email'])}", 10)

    def _plain(self, p, text, size, sup=False):
        from docx.shared import Pt
        r = p.add_run(text)
        r.font.name, r.font.size, r.font.superscript = 'Times New Roman', Pt(size), sup
        return r

    def para(self, text, *a, **k):
        if text.startswith('**Research question:**') and not self.P['research_question_paragraph']:
            self._hit('research question')
            return
        if text.startswith('**Abstract:**'):
            self._hit('abstract')
            return RenamingBuilder.para(self, '**Abstract:** ' + abstract(self.D, self.rb, self.P), *a, **k)
        if text.startswith('**Keywords:**'):
            self._hit('keywords')
            return RenamingBuilder.para(self, '**Keywords:** ' + '; '.join(KEYWORDS[:self.P['keywords_max']]), *a, **k)
        if text.startswith('The split is predeclared'):
            self._hit('ethics')
            i = text.index('**Ethics statement:**')
            return RenamingBuilder.para(self, text[:i] + 'The ethics of using these videos are addressed in the Ethics '
                                        'statement at the end of the paper.', *a, **k)
        if text.startswith('The pipeline, the v2 evaluation scripts'):
            self._hit('data availability')
            return
        return super().para(text, *a, **k)

    def h1(self, text):
        if text == '7. Limitations':
            return RenamingBuilder.h1(self, '7. Limitations and Future Work')
        if text == '8. Conclusion':
            RenamingBuilder.para(self, RS.FUTURE_WORK)
        if text in ('Data and Code Availability', 'References'):
            return
        return super().h1(text)


def figure1_caption(D):
    """Caption of the fixed Figure 1 (assets/Figure1_blurred_named.png); its printed switch rates are checked."""
    m = lambda c: R.mean(D, c, 'switches_per_sec', 1)  # noqa: E731
    assert f"{m('Full'):.2f}" == '9.86' and f"{m('D3'):.2f}" == '0.24', 'Figure 1 switch rates no longer match the data'
    return ('Figure 1. NeoSTAR architecture, illustrated with frames of the neonatal videos (faces blurred): validated, '
            'timestamped frames sampled at up to 20 FPS (1); fused, tracked infant box (2); SAM 2 infant mask (3); '
            'mask-constrained pose and the anatomical mapping to 17 candidate regions (4); condition terms, with '
            'examples of the conditions they must handle (5); top-region switches over 10 s for the PF-ROI per-frame '
            'rule and for the NeoSTAR stabilizer alone (D3), with the mean switch rates over all videos; the headline '
            f"configuration {D['chosen']} reaches {m(D['chosen']):.2f} switches/s (6); refined pixels (ROI ∩ SAM 2 mask ∩ "
            'adaptive skin) with smoothed, cross-faded fusion weights (7); and the output representation: selected '
            'regions, fusion weights, refined pixels and condition metadata (8). Stages marked "new" were changed in '
            'this work. No stage contains a trained model; only four stabilizer settings (stage 6) are selected on '
            'training videos.')


def shift_tables(text):
    """The comparison table of Section 2 becomes Table 1; every earlier table number moves up by one."""
    text = TABLE_RE.sub(lambda m: f'Table {int(m.group(1)) + 1}', text)
    return text.replace('Table @', 'Table ')


def abstract(D, rb, P):
    text = U.short_abstract(D, rb)
    old = 'Against a fingertip-PPG reference on UBFC-rPPG with controlled perturbations and frozen settings'
    assert old in text, 'abstract wording changed'
    text = text.replace(old, 'In a controlled pilot against a fingertip-PPG reference (three UBFC-rPPG subjects, '
                             'simulated perturbations, frozen settings)')
    n = len(text.split())
    if n > P['abstract_max_words']:
        raise SystemExit(f"abstract has {n} words (limit {P['abstract_max_words']})")
    return text


# ------------------------------------------------------------------ composition (same order as the updated builder)
def compose(b, D, figs):
    b.title('')
    b.para('**Research question:** Can a learning-free, anatomy- and mask-guided dynamic ROI representation be made '
           'temporally stable and internally consistent, and do label-free tests support that its selected pixels carry '
           'more pulse evidence than fixed or alternative ROIs?')
    b.para('**Abstract:** ')
    b.para('**Keywords:** ')
    pv = D['pv1']
    D['sw'] = {x: float(pv.loc[pv.config == x, 'switches_per_sec'].mean()) for x in ('Full', 'D3', 'D3L', D['chosen'])}
    D['ret_min'] = float(pv[pv.config.isin(['D3', 'D3L']) & pv.split.isin(['validation', 'test'])]
                         .groupby(['config', 'split']).retention.mean().min())
    S.introduction(b, D)
    RW.related_work(b)
    S.method(b, D)
    S.setup(b, D)
    R.results(b, D, figs)
    R.discussion(b, D)
    declarations(b)
    return b


def declarations(b):
    info, dec = b.info, b.info['declarations']
    text = {
        'credit': '; '.join(f'**{fill(name)}:** {fill(roles)}' for name, roles in dec['credit'].items()) + '.',
        'competing': fill(dec['competing']),
        'funding': fill(dec['funding']),
        'ethics': ('The 34 neonatal videos are short clips that were publicly accessible on the web at the time of '
                   'collection; no recording was made for this study, no family or uploader was contacted, and no '
                   'names, account details or locations were stored. Only derived numerical data (ROI geometry, colour '
                   'traces and window statistics) were analysed, the videos are not redistributed, and the infant\'s face '
                   'in Figure 1 is blurred. UBFC-rPPG [47] is a public research dataset of adult volunteers and was used '
                   f"under its terms of use. {fill(dec['irb'])}"),
        'data': ('The code (pipeline, evaluation scripts, configuration files, tuning logs, perturbation generators with '
                 'their seeds and the pre-registrations) and all per-video and per-window result tables are available at '
                 f"{fill(dec['repository_url'])}. The manifest of the neonatal videos (identifier, split, frame rate, "
                 'duration, resolution and SHA-256 hash of every file) is provided as Supplementary Data S1, so that the '
                 'same files can be verified; the videos themselves are third-party content and are not redistributed. '
                 'UBFC-rPPG is available from its authors [47].'),
        'ai': (f"During the preparation of this work the authors used {dec['ai_tool']} {dec['ai_purpose']}. After using "
               'this tool, the authors reviewed and edited the content as needed and take full responsibility for the '
               'content of the publication.'),
        'acknowledgements': dec.get('acknowledgements', ''),
    }
    for key, heading in b.P['declarations']:
        if not text.get(key):
            continue
        b.h1(heading)
        b.para(text[key])


def references(b, order, style):
    RenamingBuilder.h1(b, 'References')
    render = REF.STYLES[style]
    b.raw = True
    for i, n in enumerate(order, 1):
        Builder.para(b, f'[{i}] ' + render(REF.DB[n]), size=10)
    b.raw = False


# ------------------------------------------------------------------ layout
def apply_layout(doc, P):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt
    L = P['layout']
    sect = doc.sections[0]
    if L.get('line_numbers'):
        ln = OxmlElement('w:lnNumType')
        ln.set(qn('w:countBy'), '1')
        ln.set(qn('w:restart'), 'continuous')
        sect._sectPr.append(ln)
    if L.get('page_numbers'):
        p = sect.footer.paragraphs[0] if sect.footer.paragraphs else sect.footer.add_paragraph()
        p.text = ''
        p.alignment = 1
        for kind, val in (('begin', None), (None, 'PAGE'), ('end', None)):
            r = p.add_run()
            if kind:
                e = OxmlElement('w:fldChar')
                e.set(qn('w:fldCharType'), kind)
            else:
                e = OxmlElement('w:instrText')
                e.set(qn('xml:space'), 'preserve')
                e.text = val
            r._r.append(e)
            r.font.size = Pt(10)
    in_table = {id(p._p) for t in doc.tables for row in t.rows for c in row.cells for p in c.paragraphs}
    body = list(doc.element.body)
    paras = doc.paragraphs
    for i, p in enumerate(paras):
        if id(p._p) in in_table:
            continue
        pf = p.paragraph_format
        if L.get('line_spacing') and p.style.name == 'normal' and not p._p.xpath('.//w:drawing'):
            pf.line_spacing = L['line_spacing']
        if p._p.xpath('.//w:drawing') or p.style.name.startswith('Heading'):
            pf.keep_with_next = True                          # figure stays with its caption, heading with its text
        nxt = p._p.getnext()
        if p.text.startswith(('Table ', 'Table S')) and nxt is not None and nxt.tag == qn('w:tbl'):
            pf.keep_with_next = True                          # table caption stays with its table
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.line_spacing = 1.0
                    p.paragraph_format.keep_with_next = True  # keeps short tables on one page
    del body


def layout_report(doc):
    """Structural check of what can be checked without rendering pages."""
    from docx.shared import Emu
    s = doc.sections[0]
    text_w = (s.page_width - s.left_margin - s.right_margin) / 914400
    issues = []
    for shp in doc.inline_shapes:
        w = Emu(shp.width).inches
        if w > text_w + 1e-3:
            issues.append(f'image {w:.2f} in wider than text {text_w:.2f} in')
    for k, t in enumerate(doc.tables, 1):
        widths = [c.width for c in t.rows[0].cells]
        if all(widths):
            w = sum(widths) / 914400
            if w > text_w + 0.01:
                issues.append(f'table {k} {w:.2f} in wider than text {text_w:.2f} in')
    return text_w, issues


# ------------------------------------------------------------------ separate files
def highlights(D, rb, P, out):
    import docx
    h = P['highlights']
    if not h:
        return None
    c, m = D['chosen'], lambda cid, col: R.mean(D, cid, col, 1)  # noqa: E731
    T = rb['T']
    items = [
        'NeoSTAR is a learning-free, auditable dynamic ROI representation for newborn video',
        f"Temporal stabilization cut region switching from {m('Full', 'switches_per_sec'):.1f} to "
        f"{m(c, 'switches_per_sec'):.2f} switches per second",
        'Surrogate-null and split-half tests check pulse evidence in the ROI without labels',
        'In a PPG-referenced pilot, NeoSTAR prevented fixed-box failure under fast motion',
        f"A tracked face box was never more accurate than NeoSTAR in {len(T)} test conditions",
    ]
    assert (T.face_verdict != 'face box better').all()
    for x in items:
        if len(x) > h['max_chars']:
            raise SystemExit(f'highlight longer than {h["max_chars"]} characters ({len(x)}): {x}')
    assert h['min'] <= len(items) <= h['max']
    d = docx.Document()
    d.add_heading('Highlights', level=1)
    for x in items:
        d.add_paragraph(x, style='List Bullet')
    d.save(out)
    return items


def cover_letter(D, rb, P, info, out, n_words):
    import docx
    from docx.shared import Pt
    c, m = D['chosen'], lambda cid, col: R.mean(D, cid, col, 1)  # noqa: E731
    T = rb['T']
    fx, smm = RS.row(rb, 'Fast, large motion', 'extra-large'), RS.row(rb, 'Swaddling + fast motion', 'to neck')
    corr = next((a for a in info['authors'] if a.get('corresponding')), info['authors'][0])
    aff = '; '.join(fill(info['affiliations'][k]) for k in corr['affiliations'])
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name, st.font.size = 'Times New Roman', Pt(11)
    paras = [
        date.today().strftime('%d %B %Y'),
        P['cover_letter']['salutation'],
        f"Please consider our manuscript \"{TITLE}\" for publication as a {P['article_type'].lower()} in {P['name']}.",
        'Camera-based monitoring could spare newborns the skin injuries caused by adhesive sensors, but it depends on a '
        'region of interest (ROI) that follows the infant. Dynamic ROI selection is unstable: neighbouring body regions '
        'receive almost equal scores, and the selected region switches several times per second. The manuscript presents '
        'NeoSTAR, a learning-free and auditable dynamic ROI representation for neonatal video, and evaluates it with '
        'tests that do not require labels and, separately, against a physiological reference.',
        'The main results are: (i) temporal stabilization reduces region switching from '
        f"{m('Full', 'switches_per_sec'):.1f} to {m(c, 'switches_per_sec'):.2f} switches per second on 34 newborn videos "
        'while retaining the region confidence; (ii) surrogate-null and split-half tests show that common label-free '
        'pulse proxies overstate pulse evidence in such videos, and we report this negative result openly; and (iii) in '
        'a controlled pilot with a fingertip-PPG reference, the representation matched a fixed box on still video and '
        f'prevented its failure under fast motion ({fx.mae_D5:.1f} vs. {fx.mae_fixed:.1f} bpm) and under swaddling with '
        f'motion ({smm.mae_D5:.1f} vs. {smm.mae_fixed:.1f} bpm, pre-registered), while a tracked face box was never more '
        f'accurate than NeoSTAR in {len(T)} conditions. We also report where the dynamic ROI does not help.',
        P['cover_letter']['scope'] + ' The work is engineering of a complete, inspectable vision pipeline with a strict, '
        'leakage-controlled evaluation; all settings were fixed on training videos, two tests were pre-registered, and '
        'the code, perturbation generators and result tables are released.',
        'This manuscript is original, has not been published previously and is not under consideration for publication '
        'elsewhere. All authors have approved the manuscript and agree with its submission. Competing interests, funding, '
        'ethics, data availability and the use of generative AI are declared in the manuscript.',
        'Thank you for considering our work.',
        'Sincerely,',
        fill(corr['name']),
        f"{aff}",
        fill(corr['email']),
    ]
    for t in paras:
        d.add_paragraph(t)
    d.save(out)


def supplement(D, rb, P, out, csv_out):
    import pandas as pd
    m = D['manifest']
    cols = ['video_id', 'split', 'fps', 'frame_count', 'duration_sec', 'width', 'height', 'sha256']
    man = m[cols].copy()
    man.to_csv(csv_out, index=False)
    b = SupplementBuilder(ROOT/'reports/paper/manuscript_build/orig_docx/basepaper.docx')
    b.title('Supplementary Material')
    b.para(f'**{TITLE}**', align=1)
    b.h1('S1. Neonatal Video Manifest')
    b.para(f'Table S1 lists the {len(man)} neonatal videos; Supplementary Data S1 ({csv_out.name}) gives the same rows with '
           'the full SHA-256 hash of every file, so that the files can be verified. The videos are not redistributed.')
    b.caption('Table S1. Neonatal videos (split, frame rate, duration, resolution and the first 12 characters of the '
              'SHA-256 hash).')
    b.table(['Video', 'Split', 'FPS', 'Frames', 'Duration (s)', 'Resolution', 'SHA-256 (first 12)'],
            [[r.video_id, r.split, f'{r.fps:.2f}', int(r.frame_count), f'{r.duration_sec:.1f}',
              f'{int(r.width)}×{int(r.height)}', r.sha256[:12]] for _, r in man.iterrows()], size=7.5,
            widths=[0.8, 0.9, 0.6, 0.7, 0.9, 0.9, 1.3])
    RS.supplement(b, rb, D)
    b.save(out)
    return pd.read_csv(csv_out)


class SupplementBuilder(RenamingBuilder):
    def _runs(self, paragraph, text, size=12, italic=False):
        return super()._runs(paragraph, shift_tables(text), size, italic)

    def table(self, header, rows, *a, **k):
        return super().table(header, [[shift_tables(str(c)) for c in r] for r in rows], *a, **k)


# ------------------------------------------------------------------ main
def build(template, rb, D, P, info, figs, cite_map):
    b = SubmissionBuilder(template, rb, D, P, info, cite_map)
    compose(b, D, figs)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--v1-report', default=None)
    ap.add_argument('--journal', default='results_in_engineering')
    ap.add_argument('--info', default=str(HERE/'submission_info.json'))
    ap.add_argument('--headline', default=None)
    ap.add_argument('--figure1', default=str(HERE/'assets/Figure1_blurred_named.png'),
                    help="fixed Figure 1 image (faces blurred); 'generate' redraws it from the videos (make_figure1.py)")
    args = ap.parse_args()
    P = journals.load(args.journal)
    info = json.load(open(args.info, encoding='utf-8'))
    v1 = Path(args.v1_report) if args.v1_report else ROOT/'reports'/f'paper_{args.name}'
    v1 = v1 if v1.is_absolute() else ROOT/v1
    D = R.load(args.name, v1, R.decided_headline(ROOT/'module1_v2/reports'/args.name, args.headline))
    outdir = D['rep']/P['output_dir']
    outdir.mkdir(parents=True, exist_ok=True)
    figs = D['rep']/'manuscript_figures_submission'
    rename_figures()
    R.make_figures(D, figs)
    if args.figure1 == 'generate':
        import make_figure1
        D['fig1'] = figs/'fig1_architecture.png'
        D['fig1_source'] = make_figure1.make(args.name, D['fig1'])
    else:
        D['fig1'] = Path(args.figure1)
        D['fig1_caption'] = figure1_caption(D)
    rb = RS.load()
    RS.make_figures(rb, figs)
    import phase1_figures_updated as P1U
    RenamingBuilder.image_map = P1U.make(args.name, D['chosen'], figs)
    template = ROOT/'reports/paper/manuscript_build/orig_docx/basepaper.docx'

    # pass 1: citation order of first appearance; pass 2: the manuscript
    b1 = build(template, rb, D, P, info, figs, None)
    order = b1.order if P['citations']['order'] == 'appearance' else sorted(b1.order)
    missing = [n for n in order if n not in REF.DB]
    if missing:
        raise SystemExit(f'cited but not in references.json: {missing}')
    cite_map = {n: i for i, n in enumerate(order, 1)}
    b = build(template, rb, D, P, info, figs, cite_map)
    references(b, order, P['reference_style'])
    uncited = sorted(set(REF.DB) - set(order))

    # checks
    paras = [p.text for p in b.doc.paragraphs] + [c.text for t in b.doc.tables for r in t.rows for c in r.cells]
    left = [t for t in paras if re.search(r'(?<![\w./-])v[12](?![\w.])', t)]
    if left:
        raise SystemExit(f'"v1"/"v2" left in the text: {[x[:80] for x in left]}')
    missing_edits = (U.EDITS - REPLACED) - b.fired
    if missing_edits:
        raise SystemExit(f'edits that did not fire: {sorted(missing_edits)}')
    if any('AUTHORS TO' in t for t in paras):
        raise SystemExit('an [AUTHORS TO ...] placeholder is left in the text')
    stale = [t[:60] for t in paras if re.search(r'Table @|\[\d+\]-\[\d+\]', t) and not t.startswith('[')]
    if P['citations']['style'] == 'elsevier' and stale:
        raise SystemExit(f'unconverted citations or table markers: {stale}')
    tables_cited = {int(x) for t in paras for x in re.findall(r'\bTable (\d+)\b', t)}
    n_tables = len([t for t in paras if re.match(r'Table \d+\.', t)])
    assert tables_cited == set(range(1, n_tables + 1)), (sorted(tables_cited), n_tables)
    body = b.doc.element.body.xml
    for rid, rel in list(b.doc.part.rels.items()):
        if rel.reltype.endswith('/image') and f'"{rid}"' not in body:
            b.doc.part.drop_rel(rid)
    apply_layout(b.doc, P)
    text_w, issues = layout_report(b.doc)
    pre = P['file_prefix']
    files = {k: outdir/f'{pre}_{v}' for k, v in (('ms', 'manuscript.docx'), ('hl', 'highlights.docx'),
                                                  ('cl', 'cover_letter.docx'), ('sm', 'supplementary_material.docx'),
                                                  ('csv', 'supplementary_data_S1_video_manifest.csv'),
                                                  ('chk', 'SUBMISSION_CHECKLIST.md'))}
    b.save(files['ms'])

    # figures in order of appearance, as separate files
    fdir = outdir/'figures'
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir()
    for i, src in enumerate(b.images, 1):
        shutil.copy(src, fdir/f'Figure_{i}{src.suffix}')
    words = sum(len(t.split()) for t in [p.text for p in b.doc.paragraphs])
    hl = highlights(D, rb, P, files['hl'])
    cover_letter(D, rb, P, info, files['cl'], words)
    supplement(D, rb, P, files['sm'], files['csv'])

    fills = sorted({m.group(0).strip() for t in paras for m in FILL_RE.finditer(t)})
    checklist(files, P, info, fills, uncited, issues, text_w, hl, len(b.images), n_tables, words, order)
    original = D['rep']/'Module1_v2_Manuscript.docx'      # not part of the shared repository
    if original.is_file():
        assert hashlib.md5(original.read_bytes()).hexdigest() == ORIGINAL_MD5, 'the original manuscript changed'
    print('Wrote', outdir)
    print(f'journal: {P["name"]} | references cited: {len(order)} | dropped (uncited): {uncited} | figures: '
          f'{len(b.images)} | tables: {n_tables} | layout issues: {issues or "none"} | FILL items: {len(fills)}')


def checklist(files, P, info, fills, uncited, issues, text_w, hl, n_fig, n_tab, words, order):
    L = [f"# Submission checklist: {P['name']}", '',
         f"Built {date.today().isoformat()} by module1_v2/manuscript/build_submission.py --journal {P['key']}.", '',
         '## Files', ''] + [f'- `{f.name}`' for f in files.values()] + ['- `figures/` (one file per figure, in order)', '',
         '## Still to fill in (shown as [FILL: ...] in the manuscript; edit submission_info.json and rebuild)', '']
    L += [f'- [ ] {x}' for x in fills] or ['- none']
    L += ['', '## Drafted for you: confirm before submitting', ''] + [f'- [ ] {x}' for x in info.get('to_confirm', [])]
    L += ['- [x] Figure 1: all infant faces blurred (module1_v2/manuscript/assets/Figure1_blurred_named.png)',
          '- [ ] Open the manuscript in Word: check page breaks and figure/table placement (could not be rendered here)',
          '', '## Journal requirements used', '',
          f"- Guide for authors: {P['guide_url']}",
          f"- Confirmed from the guide: {'; '.join(P['verified'])}",
          '- Assumed (Elsevier standard; check the guide): abstract ≤ '
          f"{P['abstract_max_words']} words, ≤ {P['keywords_max']} keywords, numbered citations in order of appearance, "
          'line numbers, declarations listed below', '',
          '## Build checks', '',
          f'- Text width {text_w:.2f} in; layout issues: {", ".join(issues) if issues else "none"}',
          f'- {n_fig} figures, {n_tab} tables, about {words:,} words including references',
          f'- {len(order)} references cited, renumbered in order of first appearance',
          f"- Highlights ({len(hl or [])}): " + ' | '.join(f'{x} ({len(x)} chars)' for x in (hl or [])),
          f"- References not cited by the submission text and therefore not listed: {uncited or 'none'}",
          '- No "v1"/"v2" left; every planned edit applied; original manuscript unchanged (MD5 checked)', '',
          '## Declarations included', ''] + [f'- {h}' for k, h in P['declarations']
                                              if k != 'acknowledgements' or info['declarations'].get(k)]
    files['chk'].write_text('\n'.join(L) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
