"""Small layout toolkit for the report: fonts, styles, headings with table of contents, figures, callouts, code listings taken from the real files."""
import os, re, textwrap
import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, Preformatted, NextPageTemplate, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS_APP = r'C:\Users\Timur\Documents\wetlandear-app\lib\fonts'
FONTS_MPL = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
pdfmetrics.registerFont(TTFont('DV', os.path.join(FONTS_APP, 'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-B', os.path.join(FONTS_APP, 'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFont(TTFont('DV-I', os.path.join(FONTS_MPL, 'DejaVuSans-Oblique.ttf')))
pdfmetrics.registerFont(TTFont('DV-BI', os.path.join(FONTS_MPL, 'DejaVuSans-BoldOblique.ttf')))
pdfmetrics.registerFont(TTFont('DVM', os.path.join(FONTS_MPL, 'DejaVuSansMono.ttf')))
pdfmetrics.registerFont(TTFont('DVM-B', os.path.join(FONTS_MPL, 'DejaVuSansMono-Bold.ttf')))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-BI')

BLUE, INK, SOFT, LINE, SKY = colors.HexColor('#1F5BFF'), colors.HexColor('#0B1B3A'), colors.HexColor('#55657F'), colors.HexColor('#DCE4F2'), colors.HexColor('#E8EFFE')
CODEBG, WARN = colors.HexColor('#F5F7FB'), colors.HexColor('#FFF4E0')
W, H = A4
LM = RM = 20 * mm
CW = W - LM - RM

S = {
    'body': ParagraphStyle('body', fontName='DV', fontSize=9.6, leading=14.2, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5.5),
    'h1': ParagraphStyle('h1', fontName='DV-B', fontSize=17, leading=21, textColor=BLUE, spaceBefore=4, spaceAfter=9, keepWithNext=1),
    'h2': ParagraphStyle('h2', fontName='DV-B', fontSize=12, leading=15.5, textColor=INK, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    'h3': ParagraphStyle('h3', fontName='DV-B', fontSize=10, leading=13, textColor=SOFT, spaceBefore=6, spaceAfter=3, keepWithNext=1),
    'cap': ParagraphStyle('cap', fontName='DV-I', fontSize=8.3, leading=11, textColor=SOFT, alignment=TA_LEFT, spaceBefore=3, spaceAfter=9),
    'bul': ParagraphStyle('bul', fontName='DV', fontSize=9.6, leading=14, textColor=INK, leftIndent=13, bulletIndent=2, spaceAfter=2.5, alignment=TA_LEFT),
    'cell': ParagraphStyle('cell', fontName='DV', fontSize=8.2, leading=10.6, textColor=INK, alignment=TA_LEFT),
    'cellb': ParagraphStyle('cellb', fontName='DV-B', fontSize=8.2, leading=10.6, textColor=INK, alignment=TA_LEFT),
    'cellh': ParagraphStyle('cellh', fontName='DV-B', fontSize=8, leading=10, textColor=colors.white, alignment=TA_LEFT),
    'call': ParagraphStyle('call', fontName='DV', fontSize=9.2, leading=13.4, textColor=INK, alignment=TA_LEFT),
    'code': ParagraphStyle('code', fontName='DVM', fontSize=6.6, leading=8.4, textColor=INK, alignment=TA_LEFT),
    'title': ParagraphStyle('title', fontName='DV-B', fontSize=27, leading=33, textColor=INK, alignment=TA_LEFT),
    'sub': ParagraphStyle('sub', fontName='DV', fontSize=13, leading=18, textColor=SOFT, alignment=TA_LEFT),
    'small': ParagraphStyle('small', fontName='DV', fontSize=8.4, leading=11.6, textColor=SOFT, alignment=TA_LEFT),
    'toc1': ParagraphStyle('toc1', fontName='DV-B', fontSize=10, leading=15, textColor=INK, leftIndent=0, firstLineIndent=0),
    'toc2': ParagraphStyle('toc2', fontName='DV', fontSize=9, leading=12.5, textColor=SOFT, leftIndent=12, firstLineIndent=0),
}


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Doc(BaseDocTemplate):
    def __init__(self, path, **kw):
        super().__init__(path, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=22 * mm, bottomMargin=20 * mm, **kw)
        main = Frame(LM, 20 * mm, CW, H - 42 * mm, id='main', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        cover = Frame(LM, 20 * mm, CW, H - 40 * mm, id='cover', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='cover', frames=[cover], onPage=self._cover), PageTemplate(id='main', frames=[main], onPage=self._page)])
        self.running = ''
        self._bm = 0

    def _cover(self, c, d):
        c.saveState(); c.setFillColor(BLUE); c.rect(0, H - 14 * mm, W, 14 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white); c.setFont('DV-B', 10); c.drawString(LM, H - 9 * mm, 'WetlandEar'); c.restoreState()

    def _page(self, c, d):
        c.saveState(); c.setStrokeColor(LINE); c.setLineWidth(.6); c.line(LM, H - 16 * mm, W - RM, H - 16 * mm)
        c.setFont('DV', 7.8); c.setFillColor(SOFT); c.drawString(LM, H - 13 * mm, 'WetlandEar · технический отчёт'); c.drawRightString(W - RM, H - 13 * mm, self.running)
        c.line(LM, 15 * mm, W - RM, 15 * mm); c.drawString(LM, 10.5 * mm, 'Все цифры взяты из журналов обучения и метрик проекта'); c.drawRightString(W - RM, 10.5 * mm, 'стр. %d' % d.page); c.restoreState()

    def beforeDocument(self):
        self._bm = 0   # bookmark keys must repeat on every pass or the table of contents never settles
        self.running = ''

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ('h1', 'h2'):
            txt = f.getPlainText(); lvl = 0 if f.style.name == 'h1' else 1
            key = 'bm%d' % self._bm; self._bm += 1
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(txt, key, level=lvl, closed=(lvl == 1))
            self.notify('TOCEntry', (lvl, txt, self.page, key))
            if lvl == 0: self.running = txt


def P(t, st='body'): return Paragraph(t, S[st])
def H1(t): return [CondPageBreak(60 * mm), Paragraph(t, S['h1'])]
def H2(t): return [CondPageBreak(55 * mm), Paragraph(t, S['h2'])]
def H3(t): return Paragraph(t, S['h3'])
def SP(h=4): return Spacer(1, h * mm)
def BUL(items): return [Paragraph(i, S['bul'], bulletText='•') for i in items]


def NUM(items):
    return [Paragraph(i, S['bul'], bulletText='%d.' % (k + 1)) for k, i in enumerate(items)]


def CALL(title, body, kind='info'):
    bg = SKY if kind == 'info' else WARN; bar = BLUE if kind == 'info' else colors.HexColor('#D98A1F')
    paras = [Paragraph('<b>%s</b>' % title, S['call'])] + [(b if not isinstance(b, str) else Paragraph(b, S['call'])) for b in (body if isinstance(body, list) else [body])]
    t = Table([[paras]], colWidths=[CW])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), bg), ('LINEBEFORE', (0, 0), (0, -1), 3, bar), ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 9), ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return [t, SP(3)]


def FIG(name, caption, width=1.0):
    path = os.path.join(HERE, 'figs', name + '.png')
    iw, ih = ImageReader(path).getSize(); w = CW * width; h = w * ih / iw
    if h > 205 * mm: h = 205 * mm; w = h * iw / ih
    return [KeepTogether([Image(path, width=w, height=h), Paragraph(caption, S['cap'])])]


def TBL(rows, widths, header=True, zebra=True, font=None, align_right=()):
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else Paragraph(c, S['cellh'] if (header and i == 0) else S['cell']) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 3.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5), ('LINEBELOW', (0, 0), (-1, -1), .4, LINE)]
    if header: st += [('BACKGROUND', (0, 0), (-1, 0), BLUE)]
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0: st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F7F9FD')))
    t.setStyle(TableStyle(st))
    return t


def excerpt(path, start, end=None, max_lines=60, strip_indent=False):
    """Real lines of a real file between two markers (regular expressions on single lines). The report quotes code, it never retypes it."""
    lines = open(path, encoding='utf-8').read().split('\n')
    i0 = next(i for i, l in enumerate(lines) if re.search(start, l))
    i1 = len(lines)
    if end:
        i1 = next((i for i in range(i0 + 1, len(lines)) if re.search(end, lines[i])), len(lines))
    out = lines[i0:i1]
    while out and not out[-1].strip(): out.pop()
    cut = ''
    if len(out) > max_lines: out = out[:max_lines]; cut = '...'
    return out, cut, i0 + 1


def CODE(path, start, end=None, caption='', max_lines=60, width=104, shown=None):
    lines, cut, first = excerpt(path, start, end, max_lines)
    rows = []
    for k, l in enumerate(lines):
        l = l.replace('\t', '    ')
        parts = textwrap.wrap(l, width, subsequent_indent='        ', replace_whitespace=False, drop_whitespace=False, break_long_words=True) or ['']
        for j, p in enumerate(parts):
            rows.append(('%3d  ' % (first + k) if j == 0 else '     ') + (p if j == 0 else '↳ ' + p.lstrip()))
    if cut: rows.append('     ...')
    txt = '\n'.join(rows)
    # the report is meant to be shown in public: replace the author's working folder in listings with a neutral name
    for a in (r'C:\Users\Timur\Documents\wetlandear-models', r'C:\Users\Timur\Documents\wetlandear-app', '/c/Users/Timur/Documents/wetlandear-models', '/c/Users/Timur/Documents/wetlandear-app'):
        txt = txt.replace(a, '<папка проекта>')
    pre = Preformatted(txt, S['code'])
    t = Table([[pre]], colWidths=[CW])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), CODEBG), ('BOX', (0, 0), (-1, -1), .5, LINE), ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 4), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    shown = shown or os.path.basename(path)
    cap = Paragraph('Файл <font name="DVM">%s</font>, строки %d–%d. %s' % (esc(shown), first, first + len(lines) - 1, caption), S['cap'])
    if len(rows) > 46:
        return [t, cap]
    return [KeepTogether([t, cap])]
