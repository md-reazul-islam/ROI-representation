"""Module 1 v2 manuscript (.docx) in the template of the v1 manuscript.

Usage: python module1_v2/manuscript/build_manuscript_v2.py --name final34_paper --v1-report reports/paper_final34_paper
       [--out module1_v2/reports/<name>/Module1_v2_Manuscript.docx]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/'reports/paper/manuscript_build'))   # v1 Builder + reference list, imported read-only
from docbuild import Builder        # noqa: E402
import results_v2 as R              # noqa: E402
import sections as S                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='final34_paper')
    ap.add_argument('--v1-report', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--headline', default=None, help='headline configuration (default: validation rule)')
    args = ap.parse_args()
    v1 = Path(args.v1_report) if args.v1_report else ROOT/'reports'/f'paper_{args.name}'
    v1 = v1 if v1.is_absolute() else ROOT/v1
    D = R.load(args.name, v1, R.decided_headline(ROOT/'module1_v2/reports'/args.name, args.headline))
    out = Path(args.out) if args.out else D['rep']/'Module1_v2_Manuscript.docx'
    out = out if out.is_absolute() else ROOT/out
    figs = D['rep']/'manuscript_figures'
    R.make_figures(D, figs)
    import make_figure1
    D['fig1'] = figs/'fig1_architecture.png'
    D['fig1_source'] = make_figure1.make(args.name, D['fig1'])

    b = Builder(ROOT/'reports/paper/manuscript_build/orig_docx/basepaper.docx')
    b.title('Stable, Auditable Dynamic ROI Representation for Neonatal Video: Temporal Stabilization, Corrected Quality '
            'Terms and Label-Free Validity Tests')
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
    # drop image parts no longer referenced by the body (the template's old Figure 1)
    body = b.doc.element.xml
    for rid, rel in list(b.doc.part.rels.items()):
        if rel.reltype.endswith('/image') and f'"{rid}"' not in body:
            b.doc.part.drop_rel(rid)
    out.parent.mkdir(parents=True, exist_ok=True)
    b.save(out)
    print('Wrote', out, '| headline config (validation):', D['chosen'])


if __name__ == '__main__':
    main()
