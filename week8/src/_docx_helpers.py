"""Shared python-docx helpers for the Week 8 documents."""
import os
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
BLUE = RGBColor(0x2a, 0x78, 0xd6); GREY = RGBColor(0x55, 0x55, 0x55)
pc = lambda x, d=1: f"{100 * x:.{d}f}%"

def new_doc(title, sub):
    d = Document(); s = d.sections[0]; s.left_margin = s.right_margin = Cm(2); s.top_margin = s.bottom_margin = Cm(1.9)
    d.styles["Normal"].font.name = "Calibri"; d.styles["Normal"].font.size = Pt(10.5)
    for h, sz in (("Heading 1", 16), ("Heading 2", 12.5), ("Heading 3", 11)):
        d.styles[h].font.name = "Calibri"; d.styles[h].font.size = Pt(sz); d.styles[h].font.color.rgb = BLUE
    p = d.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = BLUE
    p = d.add_paragraph(); r = p.add_run(sub); r.font.size = Pt(11); r.font.color.rgb = GREY
    p = d.add_paragraph(); r = p.add_run("HealthConnect Clinic Experience Lab · AnalystLab Africa Experience Lab, Week 8 · Data Science track · Final Integration, Presentation & Showcase"); r.italic = True; r.font.size = Pt(9)
    return d
def shade(cell, col):
    tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), col); tcPr.append(sh)
def para(d, t, bold=False, italic=False, size=None, after=6):
    p = d.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    if size: r.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(after); return p
def bullets(d, items):
    for t in items:
        p = d.add_paragraph(style="List Bullet")
        if isinstance(t, tuple): r = p.add_run(t[0]); r.bold = True; p.add_run(t[1])
        else: p.add_run(t)
        p.paragraph_format.space_after = Pt(2)
def box(d, text, fill="E8F1FC", bold_first=None):
    t = d.add_table(rows=1, cols=1); t.style = "Table Grid"; c = t.rows[0].cells[0]; shade(c, fill); c.text = ""
    p = c.paragraphs[0]
    if bold_first: r = p.add_run(bold_first + " "); r.bold = True
    p.add_run(text); d.add_paragraph().paragraph_format.space_after = Pt(2)
def table(d, df, widths=None, size=8.5, colorcol=None):
    t = d.add_table(rows=1, cols=len(df.columns)); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]; cell.text = ""; r = cell.paragraphs[0].add_run(str(c)); r.bold = True; r.font.size = Pt(size); r.font.color.rgb = RGBColor(255, 255, 255); shade(cell, "2A78D6")
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if isinstance(v, float): v = "" if pd.isna(v) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}" if abs(v) < 10 else f"{v:,.1f}")
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run("" if v is None else str(v)); r.font.size = Pt(size)
            if colorcol and df.columns[i] in colorcol:
                sv = str(v)
                if sv.startswith("PASS"): shade(cells[i], "D5F0E4")
                elif sv.startswith("FAIL"): shade(cells[i], "FADBDB")
    if widths:
        t.autofit = False
        for row in t.rows:
            for i, w in enumerate(widths): row.cells[i].width = Cm(w)
    d.add_paragraph().paragraph_format.space_after = Pt(2); return t
def fig(d, path, caption, width=16.5):
    d.add_heading(caption, 3); d.add_picture(path, width=Cm(width)); d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
