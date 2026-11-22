"""Paper-report writer shared by the Module 1 and Module 2 evaluations.

Report collects headings, paragraphs, tables (CSV + booktabs LaTeX) and figures (300-dpi PNG + PDF) and writes
REPORT.md and a self-contained REPORT.html (figures embedded) with numbered table / figure captions.
"""
from __future__ import annotations

import base64
import html

import numpy as np
import pandas as pd


def write_latex(table, path, caption, label):
    """Booktabs LaTeX table (escape special characters; no index)."""
    text = table.to_latex(index=False, escape=True, caption=caption, label=label, na_rep='--',
                          column_format='l' + 'r' * (table.shape[1] - 1), position='t')
    text = text.replace('\\begin{table}[t]', '\\begin{table}[t]\n\\centering\\small')
    path.write_text(text, encoding='utf-8')


class Report:
    """Collects blocks (headings, text, tables, figures) and writes Markdown, HTML and LaTeX."""

    def __init__(self, output):
        self.output, self.blocks = output, []
        for sub in ('tables', 'latex', 'figures'):
            (output/sub).mkdir(parents=True, exist_ok=True)

    def heading(self, text, level=2):
        self.blocks.append(('h', level, text))

    def text(self, text):
        self.blocks.append(('p', text.strip()))

    def table(self, frame, name, caption, latex=True, show=True):
        frame.to_csv(self.output/'tables'/f'{name}.csv', index=False)
        if latex and len(frame):
            write_latex(frame, self.output/'latex'/f'{name}.tex', caption, f'tab:{name}')
        if show:
            self.blocks.append(('t', frame, name, caption))

    def figure(self, figure, name, caption):
        for suffix, dpi in (('png', 300), ('pdf', None)):
            figure.savefig(self.output/'figures'/f'{name}.{suffix}', dpi=dpi, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(figure)
        self.blocks.append(('f', name, caption))

    def write(self, title):
        md, body = [f'# {title}\n'], [f'<h1>{html.escape(title)}</h1>']
        numbers = dict(t=0, f=0)
        for block in self.blocks:
            kind = block[0]
            if kind == 'h':
                md.append(f'\n{"#" * block[1]} {block[2]}\n')
                body.append(f'<h{block[1]}>{html.escape(block[2])}</h{block[1]}>')
            elif kind == 'p':
                md.append(block[1] + '\n')
                body.append(''.join(f'<p>{html.escape(p)}</p>' for p in block[1].split('\n\n')))
            elif kind == 't':
                numbers['t'] += 1
                frame, name, caption = block[1:]
                label = f'Table {numbers["t"]}. {caption} (tables/{name}.csv, latex/{name}.tex)'
                shown = frame.copy()
                for column in shown:
                    if shown[column].dtype.kind == 'f':
                        shown[column] = shown[column].map(lambda v: '' if pd.isna(v) else f'{v:.3g}')
                md.append(f'**{label}**\n\n' + markdown_table(shown) + '\n')
                body.append(f'<p class="caption">{html.escape(label)}</p>' + shown.to_html(index=False, border=0))
            elif kind == 'f':
                numbers['f'] += 1
                name, caption = block[1:]
                label = f'Figure {numbers["f"]}. {caption}'
                md.append(f'![{name}](figures/{name}.png)\n\n*{label}*\n')
                data = base64.b64encode((self.output/'figures'/f'{name}.png').read_bytes()).decode()
                body.append(f'<figure><img src="data:image/png;base64,{data}"><figcaption>{html.escape(label)}'
                            f'</figcaption></figure>')
        (self.output/'REPORT.md').write_text('\n'.join(md), encoding='utf-8')
        style = ('body{font-family:Georgia,serif;max-width:1100px;margin:2em auto;padding:0 16px;line-height:1.5;'
                 'color:#222;background:#fff}table{border-collapse:collapse;font-size:13px;margin:.5em 0 1.5em;'
                 'display:block;overflow-x:auto}th,td{padding:3px 8px;border-bottom:1px solid #ddd;text-align:right}'
                 'th{border-bottom:2px solid #444}td:first-child,th:first-child{text-align:left}img{max-width:100%}'
                 'figure{margin:1em 0 2em}figcaption,.caption{font-size:14px;font-style:italic;color:#444}')
        (self.output/'REPORT.html').write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}'
                                               f'</title><meta name="viewport" content="width=device-width,initial-scale=1">'
                                               f'<style>{style}</style></head><body>{"".join(body)}</body></html>',
                                               encoding='utf-8')


def markdown_table(frame):
    cell = lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).replace('|', '\\|')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |', '|' + '---|' * frame.shape[1]]
    lines += ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join(lines)


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100,
                         'font.family': 'serif'})
    return plt
