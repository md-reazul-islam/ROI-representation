"""Convert the project report HTML (build_project_report.py) into a Word document with the same content.

Headings, paragraphs (bold / italic / code), lists, tables, notes, data figures and the architecture diagrams
(inline SVG, redrawn as images) are carried over, so the .docx always matches the HTML built from the same run.

Usage: python "HR RR using module1 v2/scripts/report_html_to_docx.py" <Project_Report.html> [<out.docx>]
"""
from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                         # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch          # noqa: E402
from bs4 import BeautifulSoup, NavigableString, Tag                    # noqa: E402
from docx import Document                                               # noqa: E402
from docx.enum.section import WD_ORIENT                                 # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT                          # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH                           # noqa: E402
from docx.oxml import OxmlElement                                       # noqa: E402
from docx.oxml.ns import qn                                             # noqa: E402
from docx.shared import Inches, Pt, RGBColor                           # noqa: E402

FILL = {'in': '#eef1f4', 'out': '#eef1f4', 'm1': '#e8f0fb', 'm1v2': '#e6f4ec', 'hr': '#fdf1e3', 'gate': '#f3ecfa',
        'lv': '#dff0e3', 'lp': '#fdf0d8', 'ln_': '#ececec'}
TEXT = {'v': '#1f8a3a', 'p': '#b37400', 'n': '#6b6b6b'}
MUTED = RGBColor(0x5f, 0x63, 0x68)
PAGE_W_IN = 7.0                                                   # usable width (Letter, 0.75 in margins)


# ------------------------------------------------------------------------------------------------ SVG -> PNG
def svg_to_png(svg):
    vb = [float(v) for v in svg.get('viewbox', svg.get('viewBox', '0 0 1100 300')).split()]
    w, h = vb[2], vb[3]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    for g in svg.find_all('g', recursive=False):
        cls = [c for c in (g.get('class') or []) if c != 'box']
        fill = FILL.get(cls[0], '#f2f2f2') if cls else '#f2f2f2'
        for r in g.find_all('rect'):
            x, y, rw, rh = (float(r.get(k)) for k in ('x', 'y', 'width', 'height'))
            ax.add_patch(FancyBboxPatch((x, y), rw, rh, boxstyle='round,pad=0,rounding_size=10', fc=fill, ec='#c9ccd1', lw=1.2))
        for t in g.find_all('text'):
            tc = t.get('class') or []
            size, weight, colour = 14, 'bold', '#1d1d1f'
            if 's' in tc:
                size, weight = 12, 'normal'
            if 'b' in tc:
                size, weight = 12.5, 'bold'
            for k, c in TEXT.items():
                if k in tc:
                    colour, weight = c, 'bold'
            ax.text(float(t.get('x')), float(t.get('y')), t.get_text(), ha='center', va='baseline', fontsize=size * .72,
                    fontweight=weight, color=colour)
    for ln in svg.find_all('line'):
        x1, y1, x2, y2 = (float(ln.get(k)) for k in ('x1', 'y1', 'x2', 'y2'))
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, color='#6b6f75', lw=1.6))
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=200, facecolor='white'); plt.close(fig)
    return buf.getvalue(), w


# ------------------------------------------------------------------------------------------------ docx helpers
def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:val'), 'clear'); shd.set(qn('w:color'), 'auto'); shd.set(qn('w:fill'), hex_fill)
    tcPr.append(shd)


def add_runs(par, node, size=None, colour=None, bold=False, italic=False, mono=False):
    """Append the inline content of an HTML node (text, b/strong, i/em, code, br) as runs."""
    for child in node.children:
        if isinstance(child, NavigableString):
            text = re.sub(r'\s+', ' ', str(child))
            if not text:
                continue
            run = par.add_run(text)
            run.bold, run.italic = bold or None, italic or None
            if size:
                run.font.size = Pt(size)
            if colour:
                run.font.color.rgb = colour
            if mono:
                run.font.name = 'Consolas'; run._element.rPr.rFonts.set(qn('w:eastAsia'), 'Consolas')
        elif isinstance(child, Tag):
            if child.name == 'br':
                par.add_run().add_break()
            elif child.name in ('svg', 'img', 'figure', 'table', 'ul', 'ol'):
                continue
            else:
                add_runs(par, child, size, colour, bold or child.name in ('b', 'strong'), italic or child.name in ('i', 'em'),
                         mono or child.name == 'code')


def add_table(doc, table):
    rows = table.find_all('tr')
    if not rows:
        return
    ncol = max(len(r.find_all(['th', 'td'])) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = 'Table Grid'; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    size = 8 if ncol <= 6 else 7
    for i, r in enumerate(rows):
        for j, c in enumerate(r.find_all(['th', 'td'])):
            cell = t.cell(i, j); cell.text = ''
            p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            add_runs(p, c, size=size, bold=c.name == 'th')
            if c.name == 'th':
                shade(cell, 'EEF1F4')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_picture(doc, data, px_width, caption):
    width = min(PAGE_W_IN, max(3.0, px_width / 160 * 1.0)) if px_width else PAGE_W_IN
    doc.add_picture(io.BytesIO(data), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption is not None:
        p = doc.add_paragraph(); add_runs(p, caption, size=8.5, colour=MUTED, italic=True)


def img_bytes(img):
    m = re.match(r'data:image/\w+;base64,(.*)', img.get('src', ''), re.S)
    if not m:
        return None, 0
    data = base64.b64decode(m.group(1))
    try:
        from PIL import Image
        px = Image.open(io.BytesIO(data)).size[0]
    except Exception:                                              # noqa: BLE001
        px = 0
    style = img.get('style', '')
    mw = re.search(r'max-width:(\d+)px', style)
    return data, (int(mw.group(1)) * 1.6 if mw else px)


def restart_numbering(doc, paragraphs):
    """Give a numbered list its own numbering instance so it starts again at 1."""
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    style = doc.styles['List Number']._element
    num_id = int(style.pPr.numPr.numId.val)
    abstract = next(n for n in numbering.findall(qn('w:num')) if int(n.get(qn('w:numId'))) == num_id).find(qn('w:abstractNumId')).get(qn('w:val'))
    new_id = max(int(n.get(qn('w:numId'))) for n in numbering.findall(qn('w:num'))) + 1
    num = OxmlElement('w:num'); num.set(qn('w:numId'), str(new_id))
    a = OxmlElement('w:abstractNumId'); a.set(qn('w:val'), abstract); num.append(a)
    over = OxmlElement('w:lvlOverride'); over.set(qn('w:ilvl'), '0')
    start = OxmlElement('w:startOverride'); start.set(qn('w:val'), '1'); over.append(start); num.append(over)
    numbering.append(num)
    for par in paragraphs:
        pPr = par._p.get_or_add_pPr()
        numPr = OxmlElement('w:numPr')
        il = OxmlElement('w:ilvl'); il.set(qn('w:val'), '0'); ni = OxmlElement('w:numId'); ni.set(qn('w:val'), str(new_id))
        numPr.append(il); numPr.append(ni)
        style_el = pPr.find(qn('w:pStyle'))
        if style_el is not None:
            style_el.addnext(numPr)
        else:
            pPr.insert(0, numPr)


# ------------------------------------------------------------------------------------------------ conversion
def convert(html_path, out_path=None):
    html_path = Path(html_path)
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.orientation = WD_ORIENT.PORTRAIT
    for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
        setattr(sec, side, Inches(.75))
    st = doc.styles['Normal']; st.font.name = 'Calibri'; st.font.size = Pt(10.5)
    main = soup.find('main') or soup.body

    def walk(node):
        for el in node.children:
            if not isinstance(el, Tag):
                continue
            cls = el.get('class') or []
            if el.name == 'h1':
                doc.add_heading(el.get_text(' ', strip=True), level=0)
            elif el.name == 'h2':
                doc.add_heading(el.get_text(' ', strip=True), level=1)
            elif el.name == 'h3':
                doc.add_heading(el.get_text(' ', strip=True), level=2)
            elif el.name == 'nav':
                continue                                           # Word users navigate with the headings pane
            elif el.name == 'p':
                p = doc.add_paragraph()
                small = 'small' in cls or 'muted' in cls
                add_runs(p, el, size=8.5 if small else None, colour=MUTED if 'muted' in cls else None)
            elif el.name in ('ol', 'ul'):
                small = 'small' in cls
                if small:                                          # references already carry [R1] labels
                    for li in el.find_all('li', recursive=False):
                        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(2)
                        add_runs(p, li, size=8.5)
                    continue
                style = 'List Number' if el.name == 'ol' else 'List Bullet'
                pars = []
                for li in el.find_all('li', recursive=False):
                    p = doc.add_paragraph(style=style)
                    add_runs(p, li)
                    pars.append(p)
                if el.name == 'ol':
                    restart_numbering(doc, pars)
            elif el.name == 'svg':
                data, w = svg_to_png(el)
                add_picture(doc, data, None, None)
            elif el.name == 'figure':
                img = el.find('img')
                data, px = img_bytes(img) if img else (None, 0)
                if data:
                    add_picture(doc, data, px, el.find('figcaption'))
            elif el.name == 'div' and 'cards' in cls:
                cards = el.find_all('div', class_='card')
                ncol = 4
                t = doc.add_table(rows=(len(cards) + ncol - 1) // ncol, cols=ncol); t.style = 'Table Grid'
                for k, c in enumerate(cards):
                    cell = t.cell(k // ncol, k % ncol); cell.text = ''; shade(cell, 'F6F7F9')
                    p = cell.paragraphs[0]; r = p.add_run(c.find(class_='v').get_text(strip=True)); r.bold = True; r.font.size = Pt(13)
                    p2 = cell.add_paragraph(); r2 = p2.add_run(c.find(class_='k').get_text(strip=True)); r2.font.size = Pt(8); r2.font.color.rgb = MUTED
                doc.add_paragraph()
            elif el.name == 'div' and 'tw' in cls:
                tb = el.find('table')
                if tb:
                    add_table(doc, tb)
            elif el.name == 'div' and 'note' in cls:
                t = doc.add_table(rows=1, cols=1); t.style = 'Table Grid'
                cell = t.cell(0, 0); cell.text = ''; shade(cell, 'FFF7E6' if 'warn' in cls else 'EAF2FC')
                add_runs(cell.paragraphs[0], el, size=9.5)
                doc.add_paragraph()
            elif el.name in ('div', 'section', 'main'):
                walk(el)
    walk(main)
    zoom = doc.settings.element.find(qn('w:zoom'))               # python-docx template omits the required percent
    if zoom is not None and zoom.get(qn('w:percent')) is None:
        zoom.set(qn('w:percent'), '100')
    out = Path(out_path) if out_path else html_path.with_suffix('.docx')
    doc.save(out)
    return out


if __name__ == '__main__':
    print(convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
