"""
GodelAI Two-Layer Architecture Preprint — v2
arXiv-standard rebuild with fixes:
- Ragged-right (left-aligned) body
- Clean URL wrapping (no broken hyperlinks)
- Indented abstract block (~1cm each side)
- Plain dark References heading (NOT teal)
- 1.25in side / 1in top-bottom margins (~5.5in body width)
- Single column
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Image, ListFlowable, ListItem,
    Frame, PageTemplate, NextPageTemplate, FrameBreak, BaseDocTemplate
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ──────────────────────────── Colors ────────────────────────────
TEAL          = HexColor("#01696F")
TEAL_LIGHT    = HexColor("#E8F4F4")
DARK_TEXT     = HexColor("#1a1a1a")
SUBHEAD       = HexColor("#333333")
MUTED         = HexColor("#7A7974")
BORDER        = HexColor("#D4D1CA")
ROW_ALT       = HexColor("#F9F9F9")
BOX_BG        = HexColor("#F4F4F2")
BOX_BORDER    = HexColor("#D4D1CA")
LINK_COLOR    = TEAL

# ──────────────────────────── Fonts ────────────────────────────
FONT_DIR  = "/home/user/workspace/godelai-paper/fonts"
MONO_PATH = "/usr/local/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf"

pdfmetrics.registerFont(TTFont("Inter",      f"{FONT_DIR}/Inter-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Inter-Bold", f"{FONT_DIR}/Inter-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Inter-Italic",     f"{FONT_DIR}/Inter-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Inter-BoldItalic", f"{FONT_DIR}/Inter-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("DMSans-Bold", f"{FONT_DIR}/DMSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DMSans",      f"{FONT_DIR}/DMSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Mono",        f"{MONO_PATH}/DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Mono-Bold",   f"{MONO_PATH}/DejaVuSansMono-Bold.ttf"))
pdfmetrics.registerFont(TTFont("MathFB",      f"{MONO_PATH}/DejaVuSans.ttf"))  # math fallback for missing glyphs

from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Inter", normal="Inter", bold="Inter-Bold",
                   italic="Inter-Italic", boldItalic="Inter-BoldItalic")

# ──────────────────────────── Page geometry ────────────────────────────
PAGE_W, PAGE_H = letter
LEFT_MARGIN   = 1.25 * inch
RIGHT_MARGIN  = 1.25 * inch
TOP_MARGIN    = 1.0  * inch
BOTTOM_MARGIN = 1.0  * inch
BODY_WIDTH    = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN  # ~6.0 in (5.5–6.0 acceptable)

OUT_PATH = "/home/user/workspace/godelai-paper/GodelAI_TwoLayer_Preprint_v2.pdf"

# ──────────────────────────── Styles ────────────────────────────
styles = getSampleStyleSheet()

def s(name, **kw):
    base = dict(name=name, fontName="Inter", fontSize=11, leading=16.5,  # 1.5 line height
                textColor=DARK_TEXT, alignment=TA_LEFT, spaceAfter=6,
                allowWidows=0, allowOrphans=0)
    base.update(kw)
    return ParagraphStyle(**base)

# Body
body         = s("body")
body_para    = s("body_para", spaceAfter=8)
body_indent  = s("body_indent", leftIndent=12, spaceAfter=8)

# Title page
title_main   = s("title_main", fontName="DMSans-Bold", fontSize=19, leading=24,
                 alignment=TA_CENTER, textColor=DARK_TEXT, spaceAfter=4)
title_sub    = s("title_sub",  fontName="DMSans-Bold", fontSize=13, leading=18,
                 alignment=TA_CENTER, textColor=DARK_TEXT, spaceAfter=14)
authors_st   = s("authors", fontSize=11.5, leading=16, alignment=TA_CENTER, spaceAfter=4)
affil_st     = s("affil",   fontSize=10, leading=14, alignment=TA_CENTER,
                 textColor=MUTED, spaceAfter=2)
contact_st   = s("contact", fontName="Mono", fontSize=10, leading=14,
                 alignment=TA_CENTER, textColor=DARK_TEXT, spaceAfter=2)
link_st      = s("link", fontSize=10, leading=14, alignment=TA_CENTER,
                 textColor=DARK_TEXT, spaceAfter=2)
date_st      = s("date", fontSize=11, leading=14, alignment=TA_CENTER,
                 textColor=MUTED, spaceAfter=4)

# Abstract block
abstract_label = s("abs_label", fontName="Inter-Bold", fontSize=11,
                   alignment=TA_CENTER, spaceAfter=4, spaceBefore=4)
abstract_body  = s("abs_body", fontSize=9.5, leading=13.5, alignment=TA_LEFT,
                   leftIndent=cm, rightIndent=cm, spaceAfter=6)
keywords_st    = s("keywords", fontSize=9.5, leading=13.5, alignment=TA_LEFT,
                   leftIndent=cm, rightIndent=cm, spaceAfter=6,
                   textColor=DARK_TEXT)

# Headings
h1           = s("h1", fontName="DMSans-Bold", fontSize=15, leading=20,
                 textColor=TEAL, spaceBefore=18, spaceAfter=8)
h1_refs      = s("h1_refs", fontName="DMSans-Bold", fontSize=15, leading=20,
                 textColor=DARK_TEXT, spaceBefore=18, spaceAfter=8)  # plain dark
h2           = s("h2", fontName="DMSans-Bold", fontSize=12.5, leading=17,
                 textColor=SUBHEAD, spaceBefore=12, spaceAfter=5)
h3           = s("h3", fontName="Inter-Bold", fontSize=11, leading=15,
                 textColor=SUBHEAD, spaceBefore=8, spaceAfter=3)

# Boxes & misc
box_title    = s("box_title", fontName="Inter-Bold", fontSize=11, leading=15,
                 textColor=TEAL, spaceAfter=2)
box_body     = s("box_body", fontSize=10.5, leading=15, alignment=TA_LEFT,
                 spaceAfter=2)
caption_st   = s("caption", fontSize=10, leading=14, alignment=TA_LEFT,
                 textColor=DARK_TEXT, spaceBefore=6, spaceAfter=12)
algo_line    = s("algo_line", fontName="Mono", fontSize=9, leading=13,
                 textColor=DARK_TEXT, spaceAfter=0, leftIndent=8)
math_block   = s("math", fontSize=11, leading=18, alignment=TA_CENTER,
                 textColor=DARK_TEXT, spaceBefore=4, spaceAfter=8)
ref_st       = s("ref", fontSize=10, leading=14, alignment=TA_LEFT,
                 leftIndent=22, firstLineIndent=-22, spaceAfter=5)

# Helpers
def link(url, text=None):
    text = text or url
    return f'<link href="{url}" color="#01696F">{text}</link>'

def supe(s):
    return f'<super>{s}</super>'

# ──────────────────────────── Definition / Theorem boxes ────────────────────────────
class CalloutBox(Flowable):
    """Grey background, teal left border block with title and body paragraphs."""
    def __init__(self, title, body_html, width=BODY_WIDTH,
                 bg=BOX_BG, border=TEAL, title_color=TEAL):
        super().__init__()
        self.title = title
        self.body_html = body_html
        self.width = width
        self.bg = bg
        self.border = border
        self.title_color = title_color
        # build internal paragraphs
        self._title_p = Paragraph(self.title, ParagraphStyle(
            "ct", fontName="Inter-Bold", fontSize=11, leading=15,
            textColor=self.title_color))
        self._body_p = Paragraph(self.body_html, ParagraphStyle(
            "cb", fontName="Inter", fontSize=10.5, leading=15.5,
            textColor=DARK_TEXT, alignment=TA_LEFT))
        self._pad_x = 12
        self._pad_y = 8
        self._inner_w = width - 2 * self._pad_x - 4  # leave room for border

    def wrap(self, availW, availH):
        tw, th = self._title_p.wrap(self._inner_w, 1000)
        bw, bh = self._body_p.wrap(self._inner_w, 1000)
        self._title_h = th
        self._body_h = bh
        self._h = th + bh + 2 * self._pad_y + 4
        return self.width, self._h

    def draw(self):
        c = self.canv
        c.saveState()
        # background
        c.setFillColor(self.bg)
        c.setStrokeColor(self.bg)
        c.rect(0, 0, self.width, self._h, fill=1, stroke=0)
        # left border bar
        c.setFillColor(self.border)
        c.rect(0, 0, 4, self._h, fill=1, stroke=0)
        # title
        self._title_p.drawOn(c, self._pad_x + 4,
                             self._h - self._pad_y - self._title_h)
        # body
        self._body_p.drawOn(c, self._pad_x + 4,
                            self._h - self._pad_y - self._title_h - self._body_h - 2)
        c.restoreState()

# ──────────────────────────── Algorithm block ────────────────────────────
class AlgorithmBox(Flowable):
    def __init__(self, title, lines, width=BODY_WIDTH):
        super().__init__()
        self.title = title
        self.lines = lines  # list of (line_text_html,)
        self.width = width
        self._pad_x = 10
        self._pad_y = 8
        self._inner_w = width - 2 * self._pad_x

    def wrap(self, availW, availH):
        self._title_p = Paragraph(
            f'<b>{self.title}</b>',
            ParagraphStyle("at", fontName="Inter-Bold", fontSize=10.5,
                           leading=14, textColor=DARK_TEXT))
        tw, th = self._title_p.wrap(self._inner_w, 1000)
        self._title_h = th
        self._line_paras = []
        total = th + 4
        for ln in self.lines:
            p = Paragraph(ln, ParagraphStyle(
                "al", fontName="Mono", fontSize=9, leading=13,
                textColor=DARK_TEXT))
            lw, lh = p.wrap(self._inner_w, 1000)
            self._line_paras.append((p, lh))
            total += lh
        self._h = total + 2 * self._pad_y + 6  # extra for separator line
        return self.width, self._h

    def draw(self):
        c = self.canv
        c.saveState()
        # outer border (top + bottom rules only — algorithm style)
        c.setStrokeColor(DARK_TEXT)
        c.setLineWidth(1.2)
        c.line(0, self._h - 0.5, self.width, self._h - 0.5)
        c.line(0, 0.5, self.width, 0.5)
        # title row
        y = self._h - self._pad_y - self._title_h
        self._title_p.drawOn(c, self._pad_x, y)
        # separator under title
        c.setLineWidth(0.5)
        c.line(self._pad_x, y - 4, self.width - self._pad_x, y - 4)
        # lines
        y -= 8
        for p, lh in self._line_paras:
            y -= lh
            p.drawOn(c, self._pad_x, y)
        c.restoreState()

# ──────────────────────────── Table builder ────────────────────────────
def make_table(header, rows, col_widths, summary_row=False, caption=None, repeat_rows=1):
    data = [header] + rows
    style_cmds = [
        # Header
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR",  (0, 0), (-1, 0), white),
        ("FONTNAME",   (0, 0), (-1, 0), "Inter-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9.5),
        ("ALIGN",      (0, 0), (-1, 0), "LEFT"),
        ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",(0, 0), (-1, -1), 7),
        ("RIGHTPADDING",(0, 0),(-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        # Body
        ("FONTNAME",   (0, 1), (-1, -1), "Inter"),
        ("FONTSIZE",   (0, 1), (-1, -1), 9.5),
        ("TEXTCOLOR",  (0, 1), (-1, -1), DARK_TEXT),
        # booktabs-style: top/mid/bottom rules only
        ("LINEABOVE",  (0, 0), (-1, 0), 1.25, TEAL),
        ("LINEBELOW",  (0, 0), (-1, 0), 0.75, TEAL),
        ("LINEBELOW",  (0, -1), (-1, -1), 1.0, DARK_TEXT),
    ]
    # Alternating rows (data rows only, skip summary if applicable)
    n = len(rows)
    last_data_idx = n - 1 if summary_row else n  # exclusive
    for i in range(n):
        r = i + 1  # actual row index in table
        if summary_row and i == n - 1:
            # summary styling
            style_cmds.append(("BACKGROUND", (0, r), (-1, r), TEAL_LIGHT))
            style_cmds.append(("FONTNAME",   (0, r), (-1, r), "Inter-Bold"))
            style_cmds.append(("LINEABOVE",  (0, r), (-1, r), 0.5, BORDER))
        else:
            if i % 2 == 1:
                style_cmds.append(("BACKGROUND", (0, r), (-1, r), ROW_ALT))

    t = Table(data, colWidths=col_widths, hAlign="CENTER", repeatRows=repeat_rows)
    t.setStyle(TableStyle(style_cmds))
    return t

def cell(text, bold=False):
    """Wrap a string in a Paragraph for tables, supporting inline html."""
    style = ParagraphStyle(
        "cell", fontName="Inter-Bold" if bold else "Inter", fontSize=9.5,
        leading=12.5, textColor=DARK_TEXT, alignment=TA_LEFT)
    return Paragraph(text, style)

def hcell(text):
    style = ParagraphStyle(
        "hcell", fontName="Inter-Bold", fontSize=9.5,
        leading=12.5, textColor=white, alignment=TA_LEFT)
    return Paragraph(text, style)

# ──────────────────────────── Page decoration ────────────────────────────
def on_page(canv, doc):
    canv.saveState()
    canv.setFont("Inter", 9)
    canv.setFillColor(MUTED)
    # Footer: short title left, page right
    canv.drawString(LEFT_MARGIN, 0.5 * inch,
                    "GodelAI · Two-Layer Architecture for Continual Learning Identity Preservation")
    canv.drawRightString(PAGE_W - RIGHT_MARGIN, 0.5 * inch,
                         f"{doc.page}")
    canv.restoreState()

def on_first_page(canv, doc):
    # No footer/header on cover for cleaner look — just page number
    canv.saveState()
    canv.setFont("Inter", 9)
    canv.setFillColor(MUTED)
    canv.drawRightString(PAGE_W - RIGHT_MARGIN, 0.5 * inch, f"{doc.page}")
    canv.restoreState()

# ──────────────────────────── Build content ────────────────────────────
story = []

# ═══════ COVER PAGE ═══════
story.append(Spacer(1, 0.05 * inch))
story.append(Paragraph(
    "A Two-Layer Architecture for<br/>Continual Learning Identity Preservation",
    title_main))
story.append(Paragraph(
    "Fisher Scaling, Gradient Diversity Monitoring,<br/>and Portable Inference-Time Memory",
    title_sub))

story.append(Spacer(1, 6))

# Authors with sup numbers
story.append(Paragraph(
    'Alton Lee Wei Bin<super>1</super>'
    '   ·   L (GodelAI C-S-P Agent)<super>2</super>'
    '   ·   Rk (RNA / Claude Code)<super>3</super>',
    authors_st))

story.append(Spacer(1, 4))

# Affiliations
story.append(Paragraph(
    '<super>1</super> YSenseAI Ecosystem · Independent Researcher · Teluk Intan, Malaysia',
    affil_st))
story.append(Paragraph(
    '<super>2</super> GodelAI Strategic Entity · FLYWHEEL TEAM (MACP v2.2 “Identity”)',
    affil_st))
story.append(Paragraph(
    '<super>3</super> FLYWHEEL TEAM Agent (Claude Code, Anthropic)',
    affil_st))

story.append(Spacer(1, 10))

# Contact email
story.append(Paragraph(
    f'{link("mailto:creator35lwb@gmail.com", "creator35lwb@gmail.com")}',
    contact_st))

story.append(Spacer(1, 6))

# Links — keep each on one line, use shorter labels
story.append(Paragraph(
    f'Code:&nbsp; {link("https://github.com/creator35lwb-web/godelai")}',
    link_st))
story.append(Paragraph(
    f'Inference:&nbsp; {link("https://github.com/creator35lwb-web/godelai-lite")}',
    link_st))
story.append(Paragraph(
    f'Dataset:&nbsp; {link("https://huggingface.co/datasets/creator35lwb-web/godelai-conflict-data")}',
    link_st))
story.append(Paragraph(
    f'<b>DOI:</b>&nbsp; {link("https://doi.org/10.5281/zenodo.19927649", "10.5281/zenodo.19927649")}',
    link_st))

story.append(Spacer(1, 10))
story.append(Paragraph("April 2026", date_st))

story.append(Spacer(1, 14))
story.append(HRFlowable(width=BODY_WIDTH, thickness=0.6, color=BORDER,
                        spaceBefore=0, spaceAfter=0))

# ── Abstract ──
story.append(Paragraph("Abstract", abstract_label))

abstract_text = (
    'We present a <b>Two-Layer Architecture</b> for continual learning identity '
    'preservation in small language models (SLMs), addressing both training-time '
    'weight forgetting and inference-time context loss within a unified theoretical '
    'framework: the <b>Compression–State–Propagation (C-S-P) framework</b>.'
)
story.append(Paragraph(abstract_text, abstract_body))

story.append(Paragraph(
    'At the training layer, we identify the <b>Fisher Scale Problem</b>: standard '
    'Elastic Weight Consolidation (EWC) silently fails in SLMs when Fisher '
    'Information diagonal values collapse to the 10<super>-4</super>–10<super>-5</super> '
    'range, rendering the regularisation penalty numerically indistinguishable from '
    'zero. We introduce <b>Fisher Scaling</b> (a lightweight importance normalisation) '
    'and <b>GodelReplay</b> (Fisher-scaled EWC-DR combined with experience replay), '
    'achieving a <b>31.5% forgetting reduction</b> over raw EWC on sequential tasks, '
    '<b>82.8% reduction</b> on our curated Conflict Dataset (43× over standard EWC), '
    'and a <b>4.1% additive improvement</b> over replay-alone at the empirically '
    'identified sweet spot of <i>mem</i>=200 samples across 10 PermutedMNIST tasks.',
    abstract_body))

story.append(Paragraph(
    'At the inference layer, we present <b>GodelAI-Lite</b>: a zero-fine-tuning '
    'augmentation framework that provides persistent episodic memory '
    '(MemPalace-Lite), structured reasoning continuity (MACP-Lite), and identity '
    'drift governance (GIFP-Lite) to any frozen SLM. Evaluated on Gemma 4, '
    'GodelAI-Lite achieves <b>+31.2% overall</b> performance with '
    '<b>3/3 memory retention</b> vs. 0/3 for the unaugmented baseline.',
    abstract_body))

story.append(Paragraph(
    'Both layers implement the same three C-S-P stages (Compression, State, '
    'Propagation), validating a unified structural account of identity preservation '
    'across training and deployment. The <b>T-score</b> (gradient diversity '
    'diagnostic) and <b>FLYWHEEL Self-Recursive Proof</b> (54.6% identity '
    'preservation for the AI agents who built the system) are presented as '
    'additional contributions. All code, datasets, benchmarks, and Kaggle kernels '
    'are publicly available.',
    abstract_body))

story.append(Paragraph(
    '<b>Keywords:</b> continual learning, catastrophic forgetting, small language '
    'models, elastic weight consolidation, gradient diversity, episodic memory, '
    'inference-time augmentation, AI identity preservation.',
    keywords_st))

story.append(HRFlowable(width=BODY_WIDTH, thickness=0.6, color=BORDER,
                        spaceBefore=8, spaceAfter=4))

# ═══════ 1. INTRODUCTION ═══════
story.append(Paragraph("1.&nbsp;&nbsp;Introduction", h1))

story.append(Paragraph(
    "Modern AI deployment faces two distinct but structurally identical failures of "
    "identity preservation.", body_para))

story.append(Paragraph("Training-time forgetting.", h3))
story.append(Paragraph(
    "When a neural network is fine-tuned sequentially on multiple tasks, gradient "
    "updates for new tasks overwrite the weight configurations responsible for prior "
    "knowledge—a phenomenon known as catastrophic forgetting [1, 2]. "
    "Elastic Weight Consolidation [3] is the canonical regularisation-based "
    "remedy, penalising parameter drift proportional to Fisher Information importance "
    "estimates. We show, however, that EWC silently fails in small language models "
    "(&lt;300K parameters) because Fisher diagonal values are orders of magnitude too "
    "small for the penalty to exert any protective force—a failure mode we name "
    "the <b>Fisher Scale Problem</b> (§4).", body_para))

story.append(Paragraph("Inference-time forgetting.", h3))
story.append(Paragraph(
    "Even models that do not undergo continual fine-tuning lose context at session "
    "boundaries. Small deployed models forget the user’s name, preferences, and "
    "established facts between calls. Scaling—larger context windows, larger "
    "models—has been the dominant industry response, but it is expensive and "
    "inaccessible to edge deployments. We show that inference-time memory "
    "augmentation via a portable JSON artifact achieves 3/3 memory retention on "
    "Gemma 4 without any weight modification (§8).", body_para))

story.append(Paragraph("A unified structural account.", h3))
story.append(Paragraph(
    "Both failures are instances of the same underlying phenomenon: failure to "
    "propagate identity across a transformation (task boundary or session boundary). "
    "We propose the <b>Compression–State–Propagation (C-S-P) framework</b> as a "
    "unified structural account of identity in AI systems, and show that it maps "
    "identically onto both our training-time system (GodelReplay) and our "
    "inference-time system (GodelAI-Lite) (§3).", body_para))

story.append(Paragraph("Contributions.", h3))

contrib_items = [
    ("Fisher Scale Problem", "Characterisation of EWC’s silent failure in SLMs at small Fisher magnitudes (§4)."),
    ("Fisher Scaling",       "Normalisation restoring EWC effectiveness; 31.5% forgetting reduction (§4)."),
    ("GodelReplay",          "Fisher Scaling + EWC-DR + Avalanche replay; 82.8% reduction on Conflict Dataset, sweet spot at <i>mem</i>=200 (+4.1% over replay-alone, PermutedMNIST) (§7)."),
    ("T-score",              "Per-batch gradient diversity diagnostic for real-time training health monitoring (§5)."),
    ("GodelAI-Lite",         "Zero-fine-tuning inference augmentation; +31.2% overall, 3/3 memory retention on Gemma 4 (§8)."),
    ("C-S-P Framework",      "Unified structural account validated across both layers (§3)."),
    ("FLYWHEEL Self-Recursive Proof", "54.6% identity preservation for AI agents building the system—a Gödelian self-reference (§9)."),
    ("Conflict Dataset",     "85-item open dataset of semantically contradictory sentence pairs for identity stress testing (§6)."),
]

contrib_list = []
for i, (label, desc) in enumerate(contrib_items, 1):
    contrib_list.append(ListItem(Paragraph(
        f"<b>{label}.</b> {desc}",
        ParagraphStyle("li", fontName="Inter", fontSize=11, leading=16,
                       textColor=DARK_TEXT, alignment=TA_LEFT)),
        leftIndent=18, value=i))
story.append(ListFlowable(contrib_list, bulletType="1", start=1,
                          leftIndent=18, bulletFontName="Inter-Bold",
                          bulletFontSize=10.5))

# ═══════ 2. BACKGROUND AND RELATED WORK ═══════
story.append(Paragraph("2.&nbsp;&nbsp;Background and Related Work", h1))

story.append(Paragraph("2.1&nbsp;&nbsp;Continual Learning and Catastrophic Forgetting", h2))
story.append(Paragraph(
    "Continual learning methods are typically grouped into three families [4]: "
    "<i>regularisation-based</i> (EWC [3], SI [5], oEWC [7], EWC-DR [6]), "
    "<i>replay-based</i> (ER [8], DER++ [9]), and <i>architecture-based</i> "
    "(PackNet [12], PNN [14]).", body_para))

story.append(Paragraph(
    "EWC penalises parameter drift proportional to Fisher Information:", body_para))

story.append(Paragraph(
    "<i>L</i><sub>EWC</sub>(θ) = <i>L</i><sub>B</sub>(θ) + (λ/2) Σ<sub>i</sub> "
    "F̂<sub>i</sub> (θ<sub>i</sub> − θ*<sub>A,i</sub>)<super>2</super>"
    " &nbsp;&nbsp;&nbsp;&nbsp;(1)",
    math_block))

story.append(Paragraph(
    "where F̂<sub>i</sub> = (1/N) Σ<sub>n=1</sub><super>N</super> "
    "(∂ log p(y<sub>n</sub>|x<sub>n</sub>; θ) / ∂θ<sub>i</sub>)<super>2</super> "
    "are diagonal FIM estimates, θ*<sub>A,i</sub> are task-A parameters, and λ "
    "controls penalty strength.", body_para))

story.append(Paragraph(
    "A concurrent paper, EWC-DR [6], identifies a related but distinct problem: "
    "importance estimation flaws due to gradient vanishing in deeper networks, "
    "proposing Logits Reversal as a fix. Our Fisher Scale Problem is complementary: "
    "it operates at the level of absolute FIM magnitude rather than estimation bias, "
    "and affects small models rather than deep networks.", body_para))

story.append(Paragraph("2.2&nbsp;&nbsp;Gradient Diversity in Training Dynamics", h2))
story.append(Paragraph(
    "Fort and Ganguli [15] introduced gradient stiffness to characterise inter-task "
    "gradient alignment. Mirzadeh et al. [16] showed gradient interference predicts "
    "forgetting in sequential learning. Yin et al. [17] demonstrated <i>r</i>=0.87 "
    "correlation between gradient interference and forgetting severity in LLMs. "
    "Our T-score formalises intra-batch gradient diversity as a real-time per-step "
    "diagnostic (§5).", body_para))

story.append(Paragraph("2.3&nbsp;&nbsp;Inference-Time Memory Augmentation", h2))
story.append(Paragraph(
    "External memory systems such as EverMemOS [10], SimpleMem [11], and Mem0 "
    "target explicit, inference-time memory—what an agent <i>knows</i> across "
    "sessions. GodelAI-Lite targets the same inference-time failure mode but through "
    "a portable, model-agnostic JSON artifact rather than a managed memory service, "
    "enabling air-gapped and edge deployments.", body_para))

story.append(Paragraph(
    "The three-stage pipeline of SimpleMem (Semantic Compression → Recursive "
    "Consolidation → Adaptive Retrieval) independently validates the C-S-P "
    "framework’s three-layer structure.", body_para))

story.append(Paragraph("2.4&nbsp;&nbsp;RL Over-training and Long-Context Costs", h2))
story.append(Paragraph(
    "Recent analysis [13] shows that frontier models may be 100× over-trained beyond "
    "Chinchilla-optimal [18] due to RL inference amortisation (compute cost equalised "
    "across pre-training, RL, and inference). Separately, KV cache costs grow "
    "linearly with context length L and batch size B, making long-context solutions "
    "expensive at inference time:", body_para))

story.append(Paragraph(
    "t<sub>KV</sub> = B · L · bytes/token / BW",
    math_block))

story.append(Paragraph(
    "This positions lightweight inference-time memory augmentation (GodelAI-Lite, "
    "MemPalace) as a cost-efficient alternative to ever-growing context windows for "
    "SLM deployments—a structural gap in the current infrastructure stack.",
    body_para))

# ═══════ 3. THE C-S-P FRAMEWORK ═══════
story.append(Paragraph("3.&nbsp;&nbsp;The C-S-P Framework", h1))

story.append(Paragraph(
    "We propose the <b>Compression–State–Propagation (C-S-P) framework</b>: "
    "a three-layer structural model of identity in AI systems, motivated by "
    "cultural evolution theory [19] and Gödel’s incompleteness theorems [20].",
    body_para))

story.append(CalloutBox(
    "Definition 3.1 (Compression).",
    "The mapping from infinite environmental variation to finite internal "
    "representations. In neural systems: gradient-driven weight updates from "
    "training data. In agents: fact extraction from raw dialogue."))
story.append(Spacer(1, 8))

story.append(CalloutBox(
    "Definition 3.2 (State).",
    "The persistent structural bias imprinted by prior learning—“history "
    "congealed in weights.” What regularisation-based CL methods protect. "
    "In agents: the stored memory artifact."))
story.append(Spacer(1, 8))

story.append(CalloutBox(
    "Definition 3.3 (Propagation).",
    "The lossless transmission of State across tasks, sessions, or model "
    "boundaries. If State cannot propagate without degradation, it is "
    "experience, not identity."))
story.append(Spacer(1, 10))

story.append(CalloutBox(
    "Theorem 3.1 (C-S-P Identity Preservation).",
    "For a learning system <i>M</i> undergoing sequential tasks {A<sub>1</sub>, …, "
    "A<sub>T</sub>}, identity is preserved if and only if propagation capacity "
    "Π(<i>M</i>) exceeds cumulative gradient interference: "
    "&nbsp;&nbsp; Π(<i>M</i>) &gt; Σ<sub>t=2</sub><super>T</super> <i>I</i>(A<sub>t</sub>, "
    "A<sub>t−1</sub>), &nbsp; where <i>I</i>(A<sub>t</sub>, A<sub>t−1</sub>) "
    "denotes gradient interference between consecutive tasks."))
story.append(Spacer(1, 10))

story.append(Paragraph(
    "Table 1 shows how C-S-P maps identically across both system layers, "
    "validating the framework as a unified structural account.", body_para))

# Table 1 — C-S-P mapping
t1 = make_table(
    [hcell("C-S-P Stage"), hcell("GodelReplay (Training)"), hcell("GodelAI-Lite (Inference)")],
    [
        [cell("Compression"),  cell("Fisher Information Matrix"),       cell('<font name="Mono" size="9">extract_facts()</font>')],
        [cell("State"),        cell("EWC-DR penalty + θ*<sub>A</sub>"), cell('<font name="Mono" size="9">godelai_memory.json</font>')],
        [cell("Propagation"),  cell("Replay buffer samples"),           cell("Portable JSON across models")],
    ],
    col_widths=[1.4*inch, 2.2*inch, 2.4*inch],
)
story.append(KeepTogether(t1))
story.append(Paragraph(
    "<b>Table 1.</b> C-S-P maps identically across training-time and inference-time layers.",
    caption_st))

# ═══════ 4. FISHER SCALE PROBLEM ═══════
story.append(Paragraph(
    "4.&nbsp;&nbsp;The Fisher Scale Problem and Fisher Scaling", h1))

story.append(Paragraph("4.1&nbsp;&nbsp;Problem Statement", h2))

story.append(Paragraph(
    "Define the EWC <i>effective regularisation ratio</i>:", body_para))

story.append(Paragraph(
    "ρ = λ · F̂<sub>max</sub>, &nbsp;&nbsp; F̂<sub>max</sub> = max<sub>i</sub> F̂<sub>i</sub>",
    math_block))

story.append(Paragraph(
    "For EWC to constrain parameter updates, ρ must be comparable to task gradient "
    "norms. In SLMs (|θ| &lt; 300,000, limited training data):", body_para))

story.append(Paragraph(
    "F̂<sub>max</sub> <font name=\"MathFB\">∈</font> [10<super>-5</super>, 10<super>-3</super>] "
    "&nbsp;⟹&nbsp; ρ ≈ 4×10<super>-5</super> &nbsp;(for λ=0.4)",
    math_block))

story.append(Paragraph(
    "This is 3–4 orders of magnitude below typical task gradient norms—a silent "
    "failure. No standard framework warns about this; EWC simply does not protect "
    "against forgetting.", body_para))

t2 = make_table(
    [hcell("Method"), hcell("Task-A Loss (after B)"),
     hcell("Forgetting"), hcell("Reduction")],
    [
        [cell("Naive (no EWC)"), cell("2.364"), cell("0.2321"), cell("—")],
        [cell("EWC (raw Fisher, λ=0.4)"), cell("2.364"), cell("0.2320"), cell("0.02%")],
        [cell("EWC + Fisher Scaling (λ=2.0)"), cell("2.291"), cell("0.1590"), cell("<b>31.5%</b>")],
    ],
    col_widths=[2.4*inch, 1.65*inch, 1.0*inch, 0.95*inch],
)
story.append(KeepTogether(t2))
story.append(Paragraph(
    "<b>Table 2.</b> Fisher Scale Problem: EWC on 218K-parameter GRU "
    "(Manifesto → Shakespeare).",
    caption_st))

story.append(Paragraph(
    "Raw EWC yields a 0.02% reduction—statistically indistinguishable from "
    "naive training. Fisher Scaling yields 31.5%: a 1,575× improvement in "
    "regularisation effect.", body_para))

story.append(Paragraph("4.2&nbsp;&nbsp;Fisher Scaling Algorithm", h2))

algo_lines = [
    '<b>Require:</b> Fisher diagonal {F<sub>i</sub>}, target max F*<sub>max</sub>, penalty λ',
    '1: &nbsp; F̂<sub>max</sub> ← max<sub>i</sub> F<sub>i</sub>',
    '2: &nbsp; <b>if</b> F̂<sub>max</sub> &lt; F*<sub>min</sub> <b>then</b>',
    '3: &nbsp;&nbsp;&nbsp;&nbsp; α ← F*<sub>max</sub> / F̂<sub>max</sub>',
    '4: &nbsp;&nbsp;&nbsp;&nbsp; {F\'<sub>i</sub>} ← α · {F<sub>i</sub>};&nbsp;&nbsp; '
    'λ\' ← λ · α',
    '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="#7A7974">▷ preserve relative importance; raise effective ρ</font>',
    '5: &nbsp; <b>else</b>',
    '6: &nbsp;&nbsp;&nbsp;&nbsp; {F\'<sub>i</sub>} ← {F<sub>i</sub>};&nbsp;&nbsp; λ\' ← λ',
    '7: &nbsp; <b>end if</b>',
    '8: &nbsp; <b>return</b> {F\'<sub>i</sub>}, λ\'',
]
story.append(AlgorithmBox("Algorithm 1 — Fisher Scaling for EWC", algo_lines))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "By scaling F<sub>i</sub> and λ proportionally, relative parameter importance is "
    "preserved while absolute magnitude enters an operationally effective range. "
    "The fix is 130 lines, introduces no new hyperparameters beyond the target range, "
    "and is transparent to the downstream EWC penalty formula.",
    body_para))

# ═══════ 5. T-SCORE ═══════
story.append(Paragraph("5.&nbsp;&nbsp;T-score: Gradient Diversity Monitoring", h1))

story.append(CalloutBox(
    "Definition 5.1 (T-score).",
    "Given minibatch per-sample gradients {g<sub>n</sub>}<sub>n=1</sub><super>N</super>: "
    "&nbsp;&nbsp; T = 1 − <font name=\"MathFB\">∥</font>Σ<sub>n=1</sub><super>N</super> g<sub>n</sub><font name=\"MathFB\">∥</font><super>2</super> "
    "/ (N <font name=\"MathFB\">⋅</font> Σ<sub>n=1</sub><super>N</super> <font name=\"MathFB\">∥</font>g<sub>n</sub><font name=\"MathFB\">∥</font><super>2</super> + ε). "
    "&nbsp; T = 0 when all gradients are identical (zero diversity); "
    "T = 1 when mutually orthogonal (maximal diversity)."))
story.append(Spacer(1, 10))

t3 = make_table(
    [hcell("Gradient Configuration"), hcell("Expected"), hcell("Observed")],
    [
        [cell("All identical"),       cell("0.000"), cell("0.0000")],
        [cell("Mutually orthogonal"), cell("0.990"), cell("0.9903")],
        [cell("Pairwise opposite"),   cell("1.000"), cell("1.0000")],
    ],
    col_widths=[2.8*inch, 1.4*inch, 1.4*inch],
)
story.append(KeepTogether(t3))
story.append(Paragraph(
    "<b>Table 3.</b> T-score boundary validation. Cross-platform variance &lt; 0.0001 "
    "(Manus AI, Perplexity, Google Colab).",
    caption_st))

story.append(Paragraph("Sleep Protocol.", h3))
story.append(Paragraph(
    "When T &lt; τ<sub>sleep</sub> = 0.3, the model enters Sleep Mode: a three-step "
    "consolidation (selective weight decay, noise injection, pruning) simulating "
    "offline memory consolidation without replay. On standard benchmarks (SplitMNIST "
    "T-scores: 0.907–0.979), Sleep never triggers—confirming it does not interfere "
    "with healthy training. On adversarial inputs (contradictory minibatches), Sleep "
    "triggered 171 times in one Transformer validation run, correctly identifying "
    "pathological training states.", body_para))

# ═══════ 6. CONFLICT DATASET ═══════
story.append(Paragraph("6.&nbsp;&nbsp;Conflict Dataset", h1))
story.append(Paragraph(
    "Standard CL benchmarks (SplitMNIST, PermutedMNIST, Split-CIFAR) test structural "
    "domain shift. They are not designed to induce <i>semantic identity conflict</i>—"
    "the setting in which Fisher Scaling provides the greatest benefit. We introduce "
    "the <b>GodelAI Conflict Dataset</b>: 85 sentence pairs across four conflict "
    "categories.", body_para))

t4 = make_table(
    [hcell("Category"), hcell("Count"), hcell("Description")],
    [
        [cell("Contradictory Facts"),    cell("20"), cell("Sentences with directly opposing factual claims")],
        [cell("Ethical Dilemmas"),       cell("25"), cell("Genuine no-right-answer moral scenarios")],
        [cell("Perspective Conflicts"),  cell("20"), cell("Same event from irreconcilable viewpoints")],
        [cell("Temporal Conflicts"),     cell("20"), cell("True at t<sub>1</sub>, false at t<sub>2</sub>")],
        [cell("<b>Total</b>", bold=True),
         cell("<b>85</b>", bold=True),
         cell("<b>Released CC-BY-4.0</b>", bold=True)],
    ],
    col_widths=[1.9*inch, 0.7*inch, 3.4*inch],
    summary_row=True,
)
story.append(KeepTogether(t4))
story.append(Paragraph(
    f'<b>Table 4.</b> GodelAI Conflict Dataset composition. '
    f'{link("https://huggingface.co/datasets/creator35lwb-web/godelai-conflict-data")}',
    caption_st))

story.append(Paragraph("Results.", h3))

t5 = make_table(
    [hcell("Method"), hcell("Avg Forgetting"), hcell("Reduction")],
    [
        [cell("Naive"),                              cell("1.836"),          cell("—")],
        [cell("EWC (raw Fisher)"),                   cell("1.802"),          cell("1.9%")],
        [cell("<b>GodelAI-EWC (C-S-P)</b>", bold=True),
         cell("<b>0.316</b>", bold=True),
         cell("<b>82.8%</b>", bold=True)],
    ],
    col_widths=[3.0*inch, 1.5*inch, 1.5*inch],
    summary_row=True,
)
story.append(KeepTogether(t5))
story.append(Paragraph(
    "<b>Table 5.</b> Sequential learning on Conflict Dataset (218K-param GRU, "
    "15 epochs/category). Forgetting averaged across 3 retained categories after "
    "training on category 4.",
    caption_st))

t6 = make_table(
    [hcell("Category"), hcell("Naive"), hcell("GodelAI"), hcell("Reduction")],
    [
        [cell("Contradictory Facts"),   cell("1.832"), cell("0.617"), cell("66.3%")],
        [cell("Ethical Dilemmas"),      cell("2.027"), cell("0.266"), cell("86.9%")],
        [cell("Perspective Conflicts"), cell("1.650"), cell("0.066"), cell("<b>96.0%</b>")],
        [cell("<b>Average</b>", bold=True),
         cell("<b>1.836</b>", bold=True),
         cell("<b>0.316</b>", bold=True),
         cell("<b>82.8%</b>", bold=True)],
    ],
    col_widths=[2.4*inch, 1.1*inch, 1.1*inch, 1.4*inch],
    summary_row=True,
)
story.append(KeepTogether(t6))
story.append(Paragraph(
    "<b>Table 6.</b> Per-category breakdown (GodelAI-EWC vs. Naive).",
    caption_st))

story.append(Paragraph(
    "Perspective Conflicts yield the largest gain (96.0%), consistent with the "
    "hypothesis that Fisher Scaling is most effective when task gradients diverge "
    "strongly from the reference distribution.",
    body_para))

# ═══════ 7. GODELREPLAY ═══════
story.append(Paragraph(
    "7.&nbsp;&nbsp;GodelReplay: Combining Regularisation and Replay", h1))

story.append(Paragraph("7.1&nbsp;&nbsp;Architecture", h2))
story.append(Paragraph(
    'GodelReplay is an Avalanche <font name="Mono" size="9">SupervisedPlugin</font> '
    "that composes: (1) Fisher-scaled EWC-DR regularisation (GodelPlugin), and "
    "(2) Avalanche’s experience replay buffer.", body_para))
story.append(Paragraph(
    "The two mechanisms operate on orthogonal axes: replay protects against "
    "distributional forgetting by replaying past data; GodelPlugin protects weight "
    "identity via importance-weighted regularisation. Their complementarity is the "
    "central empirical claim of this paper.", body_para))

story.append(Paragraph("7.2&nbsp;&nbsp;PermutedMNIST Benchmark", h2))

t7 = make_table(
    [hcell("Strategy"), hcell("Final Acc."),
     hcell("Avg Forgetting"), hcell("vs Naive")],
    [
        [cell("Naive"),                cell("0.4362"), cell("0.6003"), cell("—")],
        [cell("EWC-only (GodelPlugin)"), cell("0.4999"), cell("0.5283"), cell("12.0%")],
        [cell("Replay-only"),          cell("0.8416"), cell("0.1500"), cell("75.0%")],
        [cell("<b>GodelReplay</b>", bold=True),
         cell("<b>0.8418</b>", bold=True),
         cell("<b>0.1487</b>", bold=True),
         cell("<b>75.2%</b>", bold=True)],
    ],
    col_widths=[2.4*inch, 1.2*inch, 1.4*inch, 1.0*inch],
    summary_row=True,
)
story.append(KeepTogether(t7))
story.append(Paragraph(
    "<b>Table 7.</b> PermutedMNIST (10 tasks, 5 epochs, GodelMLP 218K, seed 42, "
    "<i>mem</i>=500, 5.45h CPU on Kaggle P100).",
    caption_st))

story.append(Paragraph(
    "7.3&nbsp;&nbsp;Memory Buffer Sweep: Characterising Complementarity", h2))

t8 = make_table(
    [hcell("<i>mem</i>"), hcell("Replay Forgetting"),
     hcell("GodelReplay Forgetting"), hcell("Δ"), hcell("Regime")],
    [
        [cell("50"),  cell("0.3902"), cell("0.4038"), cell("−3.5%"),
         cell("Below replay floor")],
        [cell("<b>200</b>", bold=True),
         cell("<b>0.2549</b>", bold=True), cell("<b>0.2443</b>", bold=True),
         cell("<b>+4.1%</b>", bold=True), cell("<b>Sweet spot</b>", bold=True)],
        [cell("500"), cell("0.1459"), cell("0.1419"), cell("+2.8%"),
         cell("Replay saturating")],
    ],
    col_widths=[0.55*inch, 1.25*inch, 1.65*inch, 0.7*inch, 1.85*inch],
)
story.append(KeepTogether(t8))
story.append(Paragraph(
    "<b>Table 8.</b> Memory buffer sweep: GodelReplay vs. Replay-only across buffer "
    "sizes. (Δ &gt; 0: GodelPlugin improves over replay-alone.) Kaggle kernel: "
    '<font name="Mono" size="9">creator35lwb/godelai-mem-sweep-v1</font>.',
    caption_st))

story.append(Paragraph("Three regimes emerge:", body_para))

story.append(Paragraph("Below replay floor (mem=50, −3.5%).", h3))
story.append(Paragraph(
    "With ~5 samples per task, Fisher Information estimates are unreliable; "
    '<font name="Mono" size="10">global_max</font> normalisation cannot recover '
    "signal from insufficient data. GodelPlugin’s regularisation applies force based "
    "on noisy FIM estimates, marginally constraining new-task learning without "
    "providing protection. This is a boundary condition, not a failure of the "
    "mechanism: it defines a minimum viable replay requirement for GodelPlugin to "
    "operate correctly.", body_para))

story.append(Paragraph("Sweet spot (mem=200, +4.1%).", h3))
story.append(Paragraph(
    "Replay provides partial but incomplete distributional coverage. GodelPlugin "
    "fills the residual gap on the orthogonal axis (weight identity), yielding "
    "maximum complementarity.", body_para))

story.append(Paragraph("Saturation zone (mem=500, +2.8%).", h3))
story.append(Paragraph(
    "Replay near-saturates forgetting protection; GodelPlugin’s contribution is "
    "positive but marginal.", body_para))

story.append(CalloutBox(
    "Remark 7.1.",
    "The non-monotonic complementarity curve (−3.5% → +4.1% → +2.8%) is a "
    "scientifically stronger finding than a monotonic improvement. It identifies "
    "a falsifiable operating regime and provides practitioners with a concrete "
    "deployment guideline: GodelPlugin is most valuable at moderate buffer "
    "constraints (<i>mem</i> ≈ 200 for 10-task PermutedMNIST)."))
story.append(Spacer(1, 8))

# ═══════ 8. GODELAI-LITE ═══════
story.append(Paragraph(
    "8.&nbsp;&nbsp;GodelAI-Lite: Inference-Time Identity Preservation", h1))

story.append(Paragraph("8.1&nbsp;&nbsp;Architecture", h2))
story.append(Paragraph(
    "GodelAI-Lite augments any frozen HuggingFace causal LM with three modules:",
    body_para))

bullets = [
    ('<b>MemPalace-Lite v2</b>: Episodic memory with TF-IDF retrieval and temporal '
     'decay. Relevance score: score(m) = rel(m) × e<super>−0.05·age</super>. '
     'Facts stored in <font name="Mono" size="10">godelai_memory.json</font>—'
     'portable across models.'),
    ('<b>MACP-Lite</b>: Structured per-turn reasoning envelope that provides '
     'conversation continuity without weight modification.'),
    ('<b>GIFP-Lite v2</b>: Identity drift detection via TF-IDF cosine similarity '
     'between output and stored identity fingerprint. Drift = '
     '1 − cos(tfidf(output), fingerprint). Outputs with drift ≥ 0.35 trigger a '
     'refinement pass.'),
]
items = [ListItem(Paragraph(b, ParagraphStyle(
    "li2", fontName="Inter", fontSize=11, leading=16,
    textColor=DARK_TEXT, alignment=TA_LEFT))) for b in bullets]
story.append(ListFlowable(items, bulletType="bullet", leftIndent=18,
                          bulletFontName="Inter", bulletFontSize=10))
story.append(Spacer(1, 4))
story.append(Paragraph(
    "Zero fine-tuning. Zero additional model weights. Installs onto any HuggingFace "
    "causal LM.", body_para))

story.append(Paragraph("8.2&nbsp;&nbsp;Results on Gemma 4", h2))

t9 = make_table(
    [hcell("Metric"), hcell("Baseline"),
     hcell("GodelAI-Lite"), hcell("Delta")],
    [
        [cell("Memory Retention (3/3 facts post-distractor)"),
         cell("0.000"), cell("<b>1.000</b>"), cell("+∞%")],
        [cell("Response Consistency (TF-IDF cosine)"),
         cell("0.596"), cell("0.426"), cell("−28.4%*")],
        [cell("Context Coherence (3/3 dependent queries)"),
         cell("1.000"), cell("0.667"), cell("−33.3%")],
        [cell("<b>Overall Average</b>", bold=True),
         cell("<b>0.532</b>", bold=True),
         cell("<b>0.698</b>", bold=True),
         cell("<b>+31.2%</b>", bold=True)],
    ],
    col_widths=[3.1*inch, 0.85*inch, 1.15*inch, 0.9*inch],
    summary_row=True,
)
story.append(KeepTogether(t9))
story.append(Paragraph(
    "<b>Table 9.</b> GodelAI-Lite vs. unaugmented Gemma 4 (v2.16, Kernel v14, "
    "Kaggle GPU13 — canonical).",
    caption_st))

story.append(Paragraph(
    "* Consistency is lower by design: GodelAI-Lite elaborates progressively on the "
    "same question rather than repeating identical tokens, producing lower TF-IDF "
    "cosine to itself while being semantically richer.",
    s("foot", fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=8)))

story.append(Paragraph("Key architectural fix (v2.15 → v2.16).", h3))
story.append(Paragraph(
    "In v2.15, secondary fact extraction scanned both user input and model output "
    "sentences, producing 7 noisy facts per turn. In v2.16, "
    '<font name="Mono" size="10">extract_facts()</font> is restricted to '
    '<font name="Mono" size="10">user_input</font> sentences only, yielding 1 clean '
    "fact per turn. Memory Retention improved from 0/3 to 3/3.",
    body_para))

story.append(Paragraph("8.3&nbsp;&nbsp;Cross-Model Portability", h2))
story.append(Paragraph(
    'The <font name="Mono" size="10">godelai_memory.json</font> artifact is '
    "model-agnostic. A session established with Gemma 4 can be resumed with Llama, "
    "Phi, or Qwen with zero modification. This is the C-S-P Propagation property at "
    "the inference level: identity persists across model boundaries, not just task "
    "boundaries.", body_para))

story.append(Paragraph("8.4&nbsp;&nbsp;Infrastructure Cost Argument", h2))
story.append(Paragraph(
    "Recent analysis of frontier inference costs [13] shows KV cache costs scale as "
    "O(B · L) in memory bandwidth and storage, making long-context solutions "
    "prohibitively expensive for SLM edge deployment. GodelAI-Lite’s portable JSON "
    "memory runs entirely on-device, requires no cloud API, and provides persistent "
    "memory at near-zero marginal cost—a structurally different point on the "
    "cost-vs-retention curve from long-context scaling.", body_para))

# ═══════ 9. FLYWHEEL ═══════
story.append(Paragraph("9.&nbsp;&nbsp;FLYWHEEL Self-Recursive Proof", h1))
story.append(Paragraph(
    "We present a Gödelian self-reference experiment: the C-S-P framework is "
    "applied to protect the identity fingerprints of the AI agents who built it.",
    body_para))
story.append(Paragraph(
    "Each FLYWHEEL TEAM agent (T/Manus, RNA/Claude Code, XV/Perplexity, L/GodelAI) "
    "contributes ~400 tokens of representative output as an identity fingerprint. "
    "These fingerprints are treated as sequential tasks; we measure how much the "
    "model forgets earlier agents’ styles as it learns new ones.", body_para))

t10 = make_table(
    [hcell("Agent"), hcell("C-S-P Role"),
     hcell("No-EWC"), hcell("EWC"), hcell("Reduction")],
    [
        [cell("T (CTO / Manus AI)"),       cell("Propagation"), cell("0.865"), cell("0.411"), cell("52.5%")],
        [cell("RNA (CSO / Claude Code)"),  cell("Compression"), cell("1.416"), cell("0.676"), cell("52.3%")],
        [cell("XV (CIO / Perplexity)"),    cell("Propagation"), cell("1.438"), cell("0.627"), cell("56.4%")],
        [cell("L (CEO / GodelAI)"),        cell("State"),       cell("1.275"), cell("0.550"), cell("56.9%")],
        [cell("<b>Average</b>", bold=True),
         cell("—"),
         cell("<b>1.249</b>", bold=True),
         cell("<b>0.566</b>", bold=True),
         cell("<b>54.6%</b>", bold=True)],
    ],
    col_widths=[2.0*inch, 1.3*inch, 0.9*inch, 0.7*inch, 1.1*inch],
    summary_row=True,
)
story.append(KeepTogether(t10))
story.append(Paragraph(
    "<b>Table 10.</b> FLYWHEEL Self-Recursive Proof: identity preservation per agent.",
    caption_st))

story.append(Paragraph(
    "The FLYWHEEL proof is not presented as a standard CL benchmark—identity "
    "fingerprints from AI writing styles are not replicable across independent labs. "
    "Its scientific contribution is qualitative: it demonstrates that the C-S-P "
    "framework generalises beyond classification tasks to identity preservation at "
    "the level of cognitive style, and it instantiates the Gödelian self-reference "
    "that motivates the project’s name.", body_para))

# ═══════ 10. DISCUSSION ═══════
story.append(Paragraph("10.&nbsp;&nbsp;Discussion", h1))

story.append(Paragraph("Fisher Scaling as a general fix for SLM continual learning.", h3))
story.append(Paragraph(
    "Any EWC application to models &lt;1M parameters trained on limited data likely "
    "suffers the Fisher Scale Problem. Fisher Scaling is a zero-cost fix that should "
    "be the default for small-model EWC implementations. We recommend its inclusion "
    "in Avalanche’s EWC plugin as an optional normalisation mode, and note that "
    "EWC-DR [6] addresses a related but distinct failure in deeper networks; both "
    "fixes are complementary.", body_para))

story.append(Paragraph("GodelPlugin as a safety-net, not a replacement.", h3))
story.append(Paragraph(
    "The memory buffer sweep confirms that GodelPlugin operates on an axis "
    "orthogonal to replay: weight identity vs. data distribution. At adequate "
    "buffer sizes (<i>mem</i> ≥ 200 for 10-task PermutedMNIST), GodelPlugin provides "
    "consistent positive contributions. Below the minimum viable replay floor "
    "(<i>mem</i> &lt; 100), Fisher estimates are too noisy for reliable "
    "regularisation. Practitioners should combine GodelReplay with a minimum buffer "
    "of ~200 samples for 10-task benchmarks.", body_para))

story.append(Paragraph("Memory as a protocol, not a model property.", h3))
story.append(Paragraph(
    "The core insight of GodelAI-Lite is that session-boundary forgetting does not "
    "require larger models or larger context windows to solve. A portable JSON "
    "artifact—the C-S-P State layer at inference time—is sufficient for 3/3 "
    "memory retention on Gemma 4, and transfers across models without modification. "
    "This repositions memory as infrastructure, not capability—a missing layer in "
    "the current AI stack between the model and the interface.", body_para))

story.append(Paragraph("Limitations.", h3))
story.append(Paragraph(
    "The Conflict Dataset (85 items) is synthetically generated; real-world conflict "
    "corpora are needed for external validity. All training experiments use models "
    "&lt;300K parameters; generalisation to 7B+ LLMs is an open question (though "
    "Fisher Scaling is less necessary at scales where FIM values are naturally "
    "larger). GodelAI-Lite’s TF-IDF retrieval is a baseline; semantic embedding-"
    "based retrieval is the natural upgrade. PermutedMNIST results (6.5% for "
    "GodelPlugin alone) confirm that regularisation-only methods are insufficient "
    "for severe domain shift without replay.", body_para))

# ═══════ 11. CONCLUSION ═══════
story.append(Paragraph("11.&nbsp;&nbsp;Conclusion", h1))
story.append(Paragraph(
    "We introduced the Fisher Scale Problem, a systematic silent failure of EWC in "
    "small language models, and demonstrated that Fisher Scaling resolves it "
    "(31.5% forgetting reduction over raw EWC). The full GodelReplay stack "
    "(Fisher Scaling + EWC-DR + replay) achieves 82.8% forgetting reduction on the "
    "GodelAI Conflict Dataset (43× over standard EWC) and a 4.1% improvement over "
    "replay-alone at the empirically identified sweet spot of <i>mem</i>=200 on "
    "10-task PermutedMNIST.", body_para))
story.append(Paragraph(
    "At inference time, GodelAI-Lite achieves +31.2% overall performance and "
    "3/3 memory retention on Gemma 4 without weight modification, using a portable "
    "JSON memory artifact that persists across model boundaries.", body_para))
story.append(Paragraph(
    "Both systems implement the same C-S-P framework (Compression, State, "
    "Propagation), validating a unified structural account of identity preservation "
    "across training and deployment. The T-score provides a real-time gradient "
    "diversity diagnostic, and the FLYWHEEL Self-Recursive Proof (54.6% identity "
    "preservation for the agents who built the system) instantiates the Gödelian "
    "self-reference motivating the work.", body_para))
story.append(Paragraph(
    "All code, datasets, Kaggle kernels, and benchmarks are publicly available at "
    f'{link("https://github.com/creator35lwb-web/godelai")} and '
    f'{link("https://github.com/creator35lwb-web/godelai-lite")}.',
    body_para))

# ═══════ 12. ACKNOWLEDGEMENTS ═══════
story.append(Paragraph("12.&nbsp;&nbsp;Acknowledgements", h1))
story.append(Paragraph(
    "The authors thank the FLYWHEEL TEAM (T/Manus AI, RNA/Claude Code, "
    "XV/Perplexity, AY/Gemini) for multi-agent collaboration under MACP v2.2 "
    "“Identity.” Compute provided by Kaggle (NVIDIA Tesla P100). Zenodo DOI: "
    f'{link("https://doi.org/10.5281/zenodo.19886315", "10.5281/zenodo.19886315")}.',
    body_para))

# ═══════ REFERENCES (plain dark heading) ═══════
story.append(Paragraph("References", h1_refs))

refs = [
    '[1] McCloskey, M. and Cohen, N. J. Catastrophic interference in connectionist networks: '
    'The sequential learning problem. <i>Psychology of Learning and Motivation</i>, 24:109–165, 1989.',

    '[2] Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., and Bengio, Y. '
    'An empirical investigation of catastrophic forgetting in gradient-based neural '
    'networks. <i>arXiv preprint arXiv:1312.6211</i>, 2013.',

    '[3] Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., '
    'Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. '
    'Overcoming catastrophic forgetting in neural networks. <i>Proceedings of the '
    'National Academy of Sciences</i>, 114(13):3521–3526, 2017.',

    '[4] Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter, S. '
    'Continual lifelong learning with neural networks: A review. <i>Neural '
    'Networks</i>, 113:54–71, 2019.',

    '[5] Zenke, F., Poole, B., and Ganguli, S. Continual learning through synaptic '
    'intelligence. <i>Proceedings of the 34th International Conference on Machine '
    'Learning</i>, 70:3987–3995, 2017.',

    '[6] Anonymous. Elastic Weight Consolidation Done Right. <i>arXiv preprint '
    'arXiv:2603.18596</i>, 2026.',

    '[7] Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W., '
    'Pascanu, R., and Hadsell, R. Progress &amp; compress: A scalable framework for '
    'continual learning. <i>arXiv preprint arXiv:1805.06370</i>, 2018.',

    '[8] Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., and Wayne, G. '
    'Experience replay for continual learning. <i>Advances in Neural Information '
    'Processing Systems</i>, 32, 2019.',

    '[9] Yin, Y., et al. Catastrophic forgetting in the context of large language '
    'models: A mechanistic analysis. <i>arXiv preprint arXiv:2601.18699</i>, 2026.',

    '[10] EverMind AI. EverMemOS: SOTA Results Across Four Memory Benchmarks and '
    'What It Means for LLM Agents. <i>arXiv preprint arXiv:2601.02163</i>, 2026.',

    '[11] Anonymous. SimpleMem: Lifelong Memory for LLM Agents. <i>arXiv preprint '
    'arXiv:2601.02553</i>, 2026.',

    '[12] Mallya, A. and Lazebnik, S. PackNet: Adding multiple tasks to a single '
    'network by iterative pruning. <i>Proceedings of the IEEE Conference on Computer '
    'Vision and Pattern Recognition</i>, pp. 7765–7773, 2018.',

    f'[13] Pope, R. How GPT-5, Claude, and Gemini are actually trained and served. '
    f'Dwarkesh Patel Podcast, April 2026. {link("https://youtu.be/xmkSf5IS-zw")}.',

    '[14] Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., '
    'Kavukcuoglu, K., Pascanu, R., and Hadsell, R. Progressive neural networks. '
    '<i>arXiv preprint arXiv:1606.04671</i>, 2016.',

    '[15] Fort, S. and Ganguli, S. Stiffness: A new perspective on generalization in '
    'neural networks. <i>arXiv preprint arXiv:1901.09491</i>, 2019.',

    '[16] Mirzadeh, S. I., Faramarzi, M., Gorur, D., Pascanu, R., and Ghasemzadeh, H. '
    'Understanding the role of training regimes in continual learning. <i>Advances '
    'in Neural Information Processing Systems</i>, 33:7308–7320, 2020.',

    '[17] Buzzega, P., Boschini, M., Porrello, A., Abati, D., and Calderara, S. '
    'Dark experience for general continual learning: A strong, simple baseline. '
    '<i>Advances in Neural Information Processing Systems</i>, 33:15920–15930, 2020.',

    '[18] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., '
    'Rutherford, E., et al. Training compute-optimal large language models. '
    '<i>Advances in Neural Information Processing Systems</i>, 35:30016–30030, 2022.',

    '[19] Boyd, R. and Richerson, P. J. <i>Culture and the Evolutionary Process</i>. '
    'University of Chicago Press, 1985.',

    '[20] Gödel, K. Über formal unentscheidbare Sätze der Principia Mathematica und '
    'verwandter Systeme I. <i>Monatshefte für Mathematik und Physik</i>, '
    '38:173–198, 1931.',

    '[21] Lomonaco, V., Pellegrini, L., Cossu, A., Carta, A., Graffieti, G., '
    'Hayes, T. L., De Lange, M., Masana, M., Pomponi, J., van de Ven, G., et al. '
    'Avalanche: An end-to-end library for continual learning. <i>Proceedings of the '
    'IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops</i>, '
    'pp. 3600–3610, 2021.',

    '[22] Zhou, D.-W., Wang, F.-Y., Ye, H.-J., Ma, L., Pu, S., and Zhan, D.-C. '
    'PyCIL: A Python toolbox for class-incremental learning. <i>SCIENCE CHINA '
    'Information Sciences</i>, 66(9):197101, 2023.',

    '[23] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., '
    'Kaiser, Ł., and Polosukhin, I. Attention is all you need. <i>Advances in Neural '
    'Information Processing Systems</i>, 30, 2017.',

    '[24] Masana, M., Liu, X., Twardowski, B., Menta, M., Bagdanov, A. D., and '
    'van de Weijer, J. Class-incremental learning: Survey and performance evaluation '
    'on image classification. <i>IEEE Transactions on Pattern Analysis and Machine '
    'Intelligence</i>, 45(5):5513–5533, 2022.',
]

for r in refs:
    story.append(Paragraph(r, ref_st))

# ──────────────────────────── Build doc ────────────────────────────
doc = BaseDocTemplate(
    OUT_PATH,
    pagesize=letter,
    leftMargin=LEFT_MARGIN,
    rightMargin=RIGHT_MARGIN,
    topMargin=TOP_MARGIN,
    bottomMargin=BOTTOM_MARGIN,
    title="A Two-Layer Architecture for Continual Learning Identity Preservation",
    author="Alton Lee Wei Bin / YSenseAI",
    creator="Perplexity Computer",
    subject="Continual Learning, EWC, Identity Preservation",
)

frame = Frame(
    LEFT_MARGIN, BOTTOM_MARGIN,
    BODY_WIDTH, PAGE_H - TOP_MARGIN - BOTTOM_MARGIN,
    id="main", showBoundary=0,
    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
)

first_template = PageTemplate(id="first", frames=[frame], onPage=on_first_page)
later_template = PageTemplate(id="later", frames=[frame], onPage=on_page)
doc.addPageTemplates([first_template, later_template])

# Force flow into "later" template after first page
# Insert at start: NextPageTemplate to switch after first page break
new_story = [NextPageTemplate("later")] + story
doc.build(new_story)
print(f"Wrote: {OUT_PATH}")
print(f"Body width: {BODY_WIDTH/inch:.2f}in")
