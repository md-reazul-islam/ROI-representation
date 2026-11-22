"""Build the completed manuscript .docx from the original (styles, page setup, Figure 1 image kept)."""
import copy
import re

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt, Inches


class Builder:
    def __init__(self, template):
        self.doc = docx.Document(template)
        body = self.doc.element.body
        self.figure1 = None
        for p in self.doc.paragraphs:
            if p._p.xpath('.//w:drawing'):
                self.figure1 = copy.deepcopy(p._p)
                break
        for child in list(body):
            if child.tag != qn('w:sectPr'):
                body.remove(child)

    def _runs(self, paragraph, text, size=12, italic=False):
        # **bold** and *italic* inline markup
        for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*)', text):
            if not part:
                continue
            if part.startswith('**'):
                run = paragraph.add_run(part[2:-2]); run.bold = True
            elif part.startswith('*'):
                run = paragraph.add_run(part[1:-1]); run.italic = True
            else:
                run = paragraph.add_run(part)
            run.font.name = 'Times New Roman'
            run.font.size = Pt(size)
            if italic:
                run.italic = True
        return paragraph

    def title(self, text):
        p = self.doc.add_paragraph(style='Title')
        self._runs(p, text, 18)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def h1(self, text):
        p = self.doc.add_paragraph(style='Heading 1')
        self._runs(p, text, 14)

    def h2(self, text):
        p = self.doc.add_paragraph(style='Heading 2')
        self._runs(p, text, 12)

    def para(self, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=12, italic=False):
        p = self.doc.add_paragraph(style='normal')
        self._runs(p, text, size, italic)
        p.alignment = align
        return p

    def eq(self, text):
        p = self.para(text, WD_ALIGN_PARAGRAPH.CENTER, italic=True)
        return p

    def bullets(self, items):
        for item in items:
            p = self.doc.add_paragraph(style='normal')
            p.paragraph_format.left_indent = Inches(.3)
            p.paragraph_format.first_line_indent = Inches(-.2)
            self._runs(p, '– ' + item)

    def caption(self, text):
        p = self.para(text, WD_ALIGN_PARAGRAPH.CENTER, size=10)
        return p

    def figure1_original(self):
        if self.figure1 is not None:
            self.doc.element.body.insert(len(self.doc.element.body) - 1, copy.deepcopy(self.figure1))

    def image(self, path, width_in=6.5):
        p = self.doc.add_paragraph(style='normal')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(path), width=Inches(width_in))

    def table(self, header, rows, size=8.5, widths=None):
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.style = self.doc.styles['Table Grid'] if 'Table Grid' in [s.name for s in self.doc.styles] else None
        for r, values in enumerate([header] + rows):
            for c, value in enumerate(values):
                cell = t.cell(r, c)
                cell.text = ''
                p = cell.paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c else WD_ALIGN_PARAGRAPH.LEFT
                run = p.add_run(str(value))
                run.font.size = Pt(size)
                run.font.name = 'Times New Roman'
                run.bold = r == 0
                if widths:
                    cell.width = Inches(widths[c])
        if t.style is None:
            self._borders(t)
        self.doc.add_paragraph(style='normal')
        return t

    @staticmethod
    def _borders(table):
        properties = table._tbl.tblPr
        borders = OxmlElement('w:tblBorders')
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            element = OxmlElement(f'w:{edge}')
            element.set(qn('w:val'), 'single'); element.set(qn('w:sz'), '4'); element.set(qn('w:color'), '808080')
            borders.append(element)
        properties.append(borders)

    def save(self, path):
        self.doc.save(path)
