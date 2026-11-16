"""Build the Case Study 1 results presentation (.pptx).

Run with uv so python-pptx does not need a global install:
    uv run --with python-pptx python presentation/build_slides.py

Output: presentation/Case_Study_1_Presentation.pptx

Content comes only from the final report (docs/report/report.pdf). The visual
style is lifted verbatim from the Case Study 2 exposé deck so the two decks match:
SRH orange #E64415 accent, Lato font, srh_logo.jpg, clean white, charcoal text,
thin gray rules, minimalist.

Thirteen slides, matching the Case Study 2 deck's length.

Writing rules: no em dashes. En dash only in number/date ranges and in the term
"Metal-Organic Framework" (rendered with an en dash to match the exposé).
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# ---- SRH exposé palette (from docs/expose.html) ----------------------------
ORANGE = RGBColor(0xE6, 0x44, 0x15)   # SRH accent
INK = RGBColor(0x1A, 0x1A, 0x1A)      # headings / body
GRAY = RGBColor(0x55, 0x55, 0x55)     # secondary / captions
RULE = RGBColor(0xE3, 0xE3, 0xE3)     # thin section rules
RULE2 = RGBColor(0xBB, 0xBB, 0xBB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FAINT = RGBColor(0xFA, 0xF6, 0xF4)    # faint warm tint for callout box

FONT = "Lato"

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIG = ROOT / "case_study_1" / "figures"
LOGO = str(ROOT / "case_study_1" / "report" / "assets" / "srh_logo.jpg")
OUT = ROOT / "case_study_1" / "presentation" / "Case_Study_1_Presentation.pptx"
LOGO_RATIO = 457 / 591  # height / width

# en dash and >= as used in the exposé
ND = "–"
GE = "≥"

SW = Inches(13.333)
SH = Inches(7.5)

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]


# ---- helpers ---------------------------------------------------------------
def _no_shadow(shape):
    # Flat look: the default shadow comes from the shape's <p:style> effectRef.
    # Remove that style and add an explicit empty effect list.
    el = shape._element
    style = el.find(qn("p:style"))
    if style is not None:
        el.remove(style)
    spPr = el.spPr
    if spPr.find(qn("a:effectLst")) is None:
        spPr.append(
            parse_xml('<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>')
        )


def slide():
    return prs.slides.add_slide(BLANK)


def R(txt, size=14, color=INK, bold=False, italic=False):
    return (txt, size, color, bold, italic)


def text(s, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=6, line_spacing=1.12):
    tb = s.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        for (txt, size, color, bold, italic) in para:
            run = p.add_run()
            run.text = txt
            run.font.size = Pt(size)
            run.font.color.rgb = color
            run.font.bold = bold
            run.font.italic = italic
            run.font.name = FONT
    return tb


def rule(s, x, y, w, color=RULE, pt=1.0):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Pt(pt))
    r.fill.solid()
    r.fill.fore_color.rgb = color
    r.line.fill.background()
    r.shadow.inherit = False
    _no_shadow(r)
    return r


def box(s, x, y, w, h, fill=None, line=None, pt=1.0):
    b = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        b.fill.background()
    else:
        b.fill.solid()
        b.fill.fore_color.rgb = fill
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line
        b.line.width = Pt(pt)
    b.shadow.inherit = False
    _no_shadow(b)
    return b


def logo_small(s):
    w = Inches(0.95)
    s.shapes.add_picture(LOGO, Inches(0.6), Inches(0.46), width=w, height=Emu(int(w * LOGO_RATIO)))


def pagenum(s, n):
    text(s, Inches(12.5), Inches(7.04), Inches(0.7), Inches(0.3),
         [[R(str(n), 10, RULE2)]], align=PP_ALIGN.RIGHT)


def header(s, num, title, n):
    """Logo, orange number + charcoal title, thin rule. Returns body top (Inches)."""
    logo_small(s)
    runs = []
    if num:
        runs.append(R(num + "   ", 27, ORANGE, bold=True))
    runs.append(R(title, 27, INK, bold=True))
    text(s, Inches(0.62), Inches(1.42), Inches(12.1), Inches(0.7), [runs])
    rule(s, Inches(0.64), Inches(2.18), Inches(12.05), color=RULE, pt=1.2)
    pagenum(s, n)
    return Inches(2.5)


def bullets(s, x, y, w, items, size=15, gap=9, lead=1.16):
    """items: list of (text, level). level 0 = orange bullet, 1 = indented dash."""
    paras = []
    for txt, lvl in items:
        if lvl == 0:
            paras.append([R("•  ", size, ORANGE, bold=True), R(txt, size, INK)])
        else:
            paras.append([R("      –  ", size - 1, ORANGE), R(txt, size - 1, GRAY)])
    return text(s, x, y, w, Inches(4.4), paras, space_after=gap, line_spacing=lead)




def figure(s, path, y, h_in, caption=None, x=None, w_in=None):
    """Centre a figure under the header, optionally with a small caption."""
    from PIL import Image
    iw, ih = Image.open(path).size
    h = Inches(h_in)
    w = Emu(int(h * iw / ih)) if w_in is None else Inches(w_in)
    if w_in is not None:
        h = Emu(int(w * ih / iw))
    xx = Emu(int((SW - w) / 2)) if x is None else x
    s.shapes.add_picture(str(path), xx, y, width=w, height=h)
    if caption:
        text(s, Inches(0.7), Emu(int(y + h)) + Inches(0.12), Inches(11.9), Inches(0.5),
             [[R(caption, 11.5, GRAY, italic=True)]], align=PP_ALIGN.CENTER)


def stat_row(s, y, items, w_each=None):
    """Big-number row: (value, label) pairs in orange over gray."""
    n = len(items)
    total = Inches(11.9)
    w = Emu(int(total / n)) if w_each is None else Inches(w_each)
    x0 = Inches(0.72)
    for i, (val, lab) in enumerate(items):
        x = Emu(int(x0 + i * w))
        text(s, x, y, Emu(int(w - Inches(0.25))), Inches(0.6),
             [[R(val, 30, ORANGE, bold=True)]])
        text(s, x, Emu(int(y)) + Inches(0.56), Emu(int(w - Inches(0.25))), Inches(0.7),
             [[R(lab, 12, GRAY)]], line_spacing=1.12)


def table(s, x, y, cols, rows, widths, size=12.5, head_size=11,
          row_h=0.33, highlight=None):
    """Thin-ruled table in the report's style: no vertical lines, orange header."""
    cx = x
    for j, (c, wd) in enumerate(zip(cols, widths)):
        text(s, Emu(int(cx)), y, Inches(wd), Inches(0.3),
             [[R(c, head_size, GRAY, bold=True)]],
             align=PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT)
        cx += Inches(wd)
    rule(s, x, Emu(int(y)) + Inches(0.28), Emu(int(cx - x)), color=RULE2, pt=1.0)
    for i, row in enumerate(rows):
        ry = Emu(int(y)) + Inches(0.4 + i * row_h)
        cx = x
        bold = (highlight is not None and i == highlight)
        for j, (v, wd) in enumerate(zip(row, widths)):
            text(s, Emu(int(cx)), ry, Inches(wd), Inches(0.3),
                 [[R(v, size, ORANGE if bold else INK, bold=bold)]],
                 align=PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT)
            cx += Inches(wd)
    rule(s, x, Emu(int(y)) + Inches(0.4 + len(rows) * row_h), Emu(int(cx - x)),
         color=RULE, pt=1.0)


# ===========================================================================
# 1  Title
# ===========================================================================
s = slide()
lw = Inches(1.7)
s.shapes.add_picture(LOGO, Emu(int((SW - lw) / 2)), Inches(1.05), width=lw,
                     height=Emu(int(lw * LOGO_RATIO)))
text(s, Inches(1.0), Inches(2.72), Inches(11.33), Inches(0.4),
     [[R("C A S E   S T U D Y   1      ·      T O P I C   8", 14, GRAY, bold=True)]],
     align=PP_ALIGN.CENTER)
text(s, Inches(0.9), Inches(3.3), Inches(11.53), Inches(1.6),
     [[R("Probabilistic Day-Ahead Solar Forecasting and", 26, INK, bold=True)],
      [R("Risk-Aware Battery Dispatch under Concept Drift", 26, INK, bold=True)]],
     align=PP_ALIGN.CENTER, line_spacing=1.08)
rule(s, Emu(int((SW - Inches(1.4)) / 2)), Inches(4.85), Inches(1.4), color=ORANGE, pt=3.0)
text(s, Inches(1.0), Inches(5.35), Inches(11.33), Inches(1.5),
     [[R("Devendra Singh Dhakad", 16, INK, bold=True),
       R("      ·      Matriculation No. 100004684", 16, GRAY)],
      [R("M.Sc. Data Science and Artificial Intelligence", 14, GRAY)],
      [R("Supervisor: Prof. Dr. Mehrdad Jalali", 14, GRAY)],
      [R("SRH University of Applied Sciences Heidelberg", 14, GRAY)]],
     align=PP_ALIGN.CENTER, space_after=7)
text(s, Inches(1.0), Inches(6.85), Inches(11.33), Inches(0.35),
     [[R("September 2026", 12, RULE2)]], align=PP_ALIGN.CENTER)

# ===========================================================================
# 2  Why the mean is the wrong answer
# ===========================================================================
s = slide()
y = header(s, "1", "Why the mean is the wrong answer", 2)
bullets(s, Inches(0.7), y, Inches(11.9), [
    ("At 12:00 on day D, a site must commit an hourly grid schedule for all 24 hours "
     "of day D+1. Horizons are 12 to 35 hours, and every hour is decided on the same "
     "information.", 0),
    ("The schedule is settled against reality at asymmetric prices: being short costs "
     "300 EUR/MWh, being long recovers only 90 EUR/MWh.", 0),
    ("Once the two costs differ, the cost-minimising commitment is no longer the "
     "expected value of tomorrow's generation. It is a quantile of the predictive "
     "distribution.", 0),
])
b = box(s, Inches(0.72), Inches(5.05), Inches(11.86), Inches(1.15), fill=FAINT)
rule(s, Inches(0.72), Inches(5.05), Inches(0.06), color=ORANGE, pt=0)
bb = box(s, Inches(0.72), Inches(5.05), Inches(0.055), Inches(1.15), fill=ORANGE)
text(s, Inches(1.05), Inches(5.28), Inches(11.3), Inches(0.9),
     [[R("A point forecast cannot express that quantity at all. ", 15.5, INK, bold=True),
       R("It reports one number where the decision needs a shape. So this study scores "
         "forecasts twice: by proper scoring rules, and by the euros they save.", 15.5, INK)]],
     line_spacing=1.2)

# ===========================================================================
# 3  Research questions
# ===========================================================================
s = slide()
y = header(s, "2", "Five research questions", 3)
for i, (tag, q) in enumerate([
    ("RQ1", "How much day-ahead skill is available with no weather input, and does "
            "conformal calibration deliver valid coverage under seasonal drift?"),
    ("RQ2", "Does a better forecast score buy a better decision? What is the predictive "
            "distribution worth, in euros?"),
    ("RQ3", "At an equal retraining budget, does drift-triggered retraining beat a "
            "fixed calendar?"),
    ("RQ4", "What does the model actually depend on, and is SHAP attribution the same "
            "thing as necessity?"),
    ("RQ5", "A new site has days of history, not years. When does training locally beat "
            "borrowing a model from another country?"),
]):
    ry = Emu(int(y)) + Inches(i * 0.82)
    text(s, Inches(0.72), ry, Inches(1.0), Inches(0.4), [[R(tag, 17, ORANGE, bold=True)]])
    text(s, Inches(1.75), ry, Inches(10.8), Inches(0.75), [[R(q, 15, INK)]], line_spacing=1.16)

# ===========================================================================
# 4  Data and the day-ahead protocol
# ===========================================================================
s = slide()
y = header(s, "3", "Data and the day-ahead protocol", 4)
stat_row(s, y, [
    ("33,192", "hourly records, Open Power System Data"),
    ("1,383", "complete delivery days, 2016 to 2020"),
    ("0", "numerical weather prediction inputs used"),
    ("12" + ND + "35 h", "forecast horizon band"),
])
figure(s, FIG / "day_ahead_protocol.png", Inches(4.3), 1.55,
       caption="No feature may cross the gate. The 12 to 35 hour band follows from a noon "
               "closure, not the 1 to 24 an unstated protocol would imply.")
text(s, Inches(0.72), Inches(6.55), Inches(11.9), Inches(0.6),
     [[R("NWP-free by design: ", 14, INK, bold=True),
       R("a small operator cannot buy a forecast feed. The deterministic part of PV is "
         "astronomy, and astronomy is free, so solar position and clear-sky irradiance are "
         "computed analytically and the learner spends its capacity on the residual.", 14, GRAY)]],
     line_spacing=1.16)

# ===========================================================================
# 5  Pipeline and the leakage guard
# ===========================================================================
s = slide()
y = header(s, "4", "Pipeline, and the one safety property", 5)
figure(s, FIG / "pipeline_architecture.png", Inches(2.42), 3.55, x=Inches(0.72))
text(s, Inches(7.3), Inches(2.5), Inches(5.3), Inches(4.2),
     [[R("The leakage problem", 16, INK, bold=True)],
      [R("With a noon gate, the obvious y[t-24] feature is illegal for every delivery "
         "hour after midday: that observation does not exist when the schedule must be "
         "submitted.", 13.5, INK)],
      [R("So no feature uses a fixed lag. Each one steps back to the most recent "
         "same-hour value that genuinely exists at the origin.", 13.5, INK)],
      [R("How it is checked", 16, INK, bold=True)],
      [R("The test suite poisons the series after a chosen gate and asserts the design "
         "matrix is bit-for-bit unchanged for every earlier row.", 13.5, INK)],
      [R("That test caught two real bugs that code review had passed.", 13.5, ORANGE, bold=True)]],
     space_after=9, line_spacing=1.18)

# ===========================================================================
# 6  RQ1: skill, and a calibration failure
# ===========================================================================
s = slide()
y = header(s, "5", "RQ1: sharp intervals are not valid intervals", 6)
figure(s, FIG / "04_coverage_vs_width.png", Inches(2.42), 3.6, x=Inches(0.6))
text(s, Inches(7.15), Inches(2.5), Inches(5.5), Inches(4.3),
     [[R("A nominal 90% interval built from in-sample residuals covered", 14, INK)],
      [R("22.5%", 40, ORANGE, bold=True), R("   of outcomes.", 14, GRAY)],
      [R("It is simultaneously the sharpest configuration in the study and by far the "
         "worst. No metric except the interval score registers the failure.", 14, INK)],
      [R("Conformal calibration repairs it without any distributional assumption:", 14, INK)]],
     space_after=10, line_spacing=1.16)
table(s, Inches(7.15), Inches(5.52), ["", "PICP", "Winkler"],
      [["raw", "0.225", "0.806"], ["conformalised", "0.630", "0.349"], ["", "", "57% better"]],
      [2.5, 1.3, 1.5], size=13, highlight=2)

# ===========================================================================
# 7  RQ1: what that looks like in operation
# ===========================================================================
s = slide()
y = header(s, "6", "RQ1: the same failure, up close", 7)
figure(s, FIG / "05_fan_chart.png", Inches(2.4), 2.75)
text(s, Inches(0.72), Inches(5.5), Inches(11.9), Inches(1.5),
     [[R("Five delivery days with nested 50, 80 and 90 percent intervals. The realised "
         "output escapes the 90 percent band twice in five days: above it on 1 June, below "
         "it on 4 June after an unforecast cloudy afternoon.", 14.5, INK)],
      [R("Two exceedances in five days is visibly worse than the one-in-ten a 90 percent "
         "interval promises. It is also a fair picture of the NWP-free limit: the model "
         "knows the shape of tomorrow, but not tomorrow's weather.", 14.5, GRAY)]],
     space_after=9, line_spacing=1.16)

# ===========================================================================
# 8  RQ2: does a better forecast buy a better decision
# ===========================================================================
s = slide()
y = header(s, "7", "RQ2: forecast skill against decision value", 8)
figure(s, FIG / "08_forecast_vs_decision.png", Inches(2.42), 3.7, x=Inches(0.55))
text(s, Inches(6.95), Inches(2.5), Inches(5.65), Inches(4.3),
     [[R("Every delivery day is settled through a two-stage stochastic LP. Regret is cost "
         "minus a perfect-foresight oracle.", 14, INK)],
      [R("r = 0.884", 30, ORANGE, bold=True),
       R("   across 20 model variants", 13.5, GRAY)],
      [R("High enough to use a scoring rule during development. Loose enough that pairs "
         "invert: gbm_quantile_k[raw] scores 7% worse on CRPS than its calibrated twin and "
         "still costs less to act on.", 14, INK)],
      [R("Inflated, too, by Prophet sitting at the far end of both axes: a correlation "
         "over a model set the analyst chose is a property of that choice.",
         13.5, GRAY, italic=True)]],
     space_after=10, line_spacing=1.16)
table(s, Inches(6.95), Inches(5.42), ["Policy", "Regret EUR/day", "vs point"],
      [["stochastic", "159.82", "-3.0%"],
       ["point forecast", "164.71", "0.0%"],
       ["CVaR", "198.43", "+20.5%"],
       ["fixed quantile", "207.75", "+26.1%"]],
      [2.4, 1.7, 1.1], size=12.5, row_h=0.3, highlight=0)

# ===========================================================================
# 9  RQ2: when the predictive distribution pays
# ===========================================================================
s = slide()
y = header(s, "8", "RQ2: when the distribution is worth having", 9)
figure(s, FIG / "13_sensitivity.png", Inches(2.35), 2.85)
text(s, Inches(0.72), Inches(5.42), Inches(5.75), Inches(1.7),
     [[R("3% to 67%", 26, ORANGE, bold=True)],
      [R("The value of committing against the full distribution rather than a point "
         "forecast, depending on the imbalance penalty and the storage available. No single "
         "number for it is meaningful without stating both.", 13.5, INK)]],
     space_after=8, line_spacing=1.16)
text(s, Inches(7.0), Inches(5.42), Inches(5.6), Inches(1.7),
     [[R("A hypothesis the data refuted", 15, INK, bold=True)],
      [R("We expected storage to substitute for forecast quality. At high penalties the "
         "opposite holds: the advantage rises with battery size, from +17.8% to +35.2%. "
         "Recourse is what makes a scenario set valuable, so storage and probabilistic "
         "forecasting are complements.", 13.5, INK)]],
     space_after=8, line_spacing=1.16)

# ===========================================================================
# 10  RQ3: drift and adaptive conformal
# ===========================================================================
s = slide()
y = header(s, "9", "RQ3: drift, retraining, and conditional validity", 10)
text(s, Inches(0.72), y, Inches(5.7), Inches(0.4),
     [[R("Retraining at an equal budget of six refits", 15, INK, bold=True)]])
table(s, Inches(0.72), Inches(2.95), ["Policy", "MAE", "Refits"],
      [["static, never refit", "0.0439", "1"],
       ["periodic calendar", "0.0437", "7"],
       ["drift-triggered", "0.0434", "2"]],
      [2.9, 1.3, 1.1], size=13)
text(s, Inches(0.72), Inches(4.35), Inches(5.7), Inches(2.4),
     [[R("Best accuracy on every metric, for two refits against the calendar's seven.",
         14, INK)],
      [R("This required fixing the detectors first. ", 14, ORANGE, bold=True),
       R("ADWIN and Page-Hinkley threshold absolute deviations, and the error stream has "
         "magnitude 0.04, so every detector reported zero triggers. ADWIN detects a step "
         "from 0 to 10 but not from 0.02 to 0.06. Standardising the stream makes the "
         "thresholds scale-free.", 14, INK)]],
     space_after=10, line_spacing=1.16)
text(s, Inches(7.0), y, Inches(5.6), Inches(0.4),
     [[R("Marginal validity is not enough", 15, INK, bold=True)]])
table(s, Inches(7.0), Inches(2.95), ["Method", "Overall", "Worst month", "Width"],
      [["split conformal", "0.903", "0.828", "0.1696"],
       ["adaptive, g=0.05", "0.896", "0.879", "0.1606"]],
      [2.0, 1.1, 1.5, 1.0], size=12.5)
text(s, Inches(7.0), Inches(4.05), Inches(5.6), Inches(2.6),
     [[R("Split conformal met its guarantee exactly, 0.903 against a 0.90 target, and "
         "still dropped to 83 percent in its worst month.", 14, INK)],
      [R("Adaptive conformal recovers worst-month coverage to 88 percent and is "
         "simultaneously sharper. Validity is bought by putting the width where it is "
         "needed, not by adding width.", 14, INK)],
      [R("An operator does not experience the average of the year. They experience "
         "January, and then February.", 14, GRAY, italic=True)]],
     space_after=10, line_spacing=1.16)

# ===========================================================================
# 11  RQ4: attribution is not necessity
# ===========================================================================
s = slide()
y = header(s, "10", "RQ4: attribution is not necessity", 11)
text(s, Inches(0.72), y, Inches(11.9), Inches(0.4),
     [[R("TreeSHAP attribution set against leave-one-family-out ablation, same model, "
         "same test period", 14.5, GRAY)]])
table(s, Inches(0.72), Inches(3.05), ["Feature family", "Share of |SHAP|", "MAE if removed", "Change"],
      [["physical, solar geometry", "67.3%", "0.0432", "-1.6%"],
       ["autoregressive lags", "17.3%", "0.0441", "+0.5%"],
       ["recent state at origin", "8.5%", "0.0450", "+2.6%"],
       ["calendar", "3.7%", "0.0435", "-0.8%"],
       ["exogenous", "3.2%", "0.0449", "+2.3%"],
       ["full model", "", "0.0438", ""]],
      [4.6, 2.4, 2.4, 1.6], size=14, row_h=0.36, highlight=2)
text(s, Inches(0.72), Inches(5.72), Inches(11.9), Inches(1.3),
     [[R("The two orderings are close to reversed. ", 15, ORANGE, bold=True),
       R("Solar geometry takes two-thirds of all attribution and is entirely removable, "
         "because the same deterministic shape is recoverable from the calendar and "
         "autoregressive blocks. The cloud-state features carry 8.5 percent and nothing "
         "can stand in for them.", 15, INK)],
      [R("SHAP answers what a fitted model leaned on, not what the problem requires.",
         14, GRAY)]],
     space_after=9, line_spacing=1.17)

# ===========================================================================
# 12  RQ5: small data
# ===========================================================================
s = slide()
y = header(s, "11", "RQ5: a new site with days of history", 12)
figure(s, FIG / "14_transfer_curves.png", Inches(2.4), 2.6)
stat_row(s, Inches(5.12), [
    ("120" + ND + "240 d", "of local history before training locally beats borrowing"),
    ("7 days", "of pooled local data beats 480 days of local-only, at 2 of 3 sites"),
    ("9" + ND + "degree", "latitude span across Switzerland, Denmark, Great Britain"),
])
text(s, Inches(0.72), Inches(6.32), Inches(11.9), Inches(0.7),
     [[R("Zero-shot transfer works because the representation is portable by construction: "
         "the target is normalised and the clear-sky block is recomputed for the target's own "
         "coordinates, so what transfers is the atmospheric relationship, not the site.",
         14, INK)]],
     line_spacing=1.16)

# ===========================================================================
# 13  Conclusions
# ===========================================================================
s = slide()
y = header(s, "12", "What the study concludes", 13)
bullets(s, Inches(0.7), y, Inches(11.9), [
    ("Calibrate, then trust. Raw quantile intervals covered as little as 22.5 percent at "
     "a nominal 90. Conformal prediction repairs that cheaply, and PICP should never be "
     "reported without an interval score.", 0),
    ("Marginal validity is not conditional validity. Split conformal met its guarantee "
     "and still failed the worst month; adaptive conformal fixed it and was sharper.", 0),
    ("Evaluate the decision, not only the forecast. The predictive distribution was worth "
     "between 3 and 67 percent of regret, and the risk-averse policies lost money while "
     "succeeding at reducing imbalance volume.", 0),
    ("Borrowing beats collecting, for four to eight months. Storage and probabilistic "
     "forecasting are complements, not substitutes.", 0),
    ("Cheap baselines earn their place. Persistence bounded every claim here and exposed "
     "the study's most serious defect: an encoder silently fed zeros for 100 percent of "
     "test rows, detectable only because it lost to a naive method.", 0),
])
b2 = box(s, Inches(0.72), Inches(5.72), Inches(11.86), Inches(1.05), fill=FAINT)
box(s, Inches(0.72), Inches(5.72), Inches(0.055), Inches(1.05), fill=ORANGE)
text(s, Inches(1.05), Inches(5.92), Inches(11.3), Inches(0.8),
     [[R("Honest limits:  ", 14.5, ORANGE, bold=True),
       R("NWP-free is self-imposed and caps the skill. The target is a national aggregate, "
         "smoother than any single site. Load and price are treated as known. Three folds "
         "make the leaderboard a cluster, not a ranking. Six defects are documented in the "
         "report, with the methods that caught them.", 14.5, INK)]],
     line_spacing=1.18)

# ===========================================================================
# Speaker notes, from presentation/speaker_script.md
# ===========================================================================
# The script lives in one markdown file so the rehearsal copy and the Presenter
# View notes cannot drift apart: edit the markdown, rebuild, and both match.
def _load_script(path):
    import re
    md = Path(path).read_text()
    parts = re.split(r"^## Slide (\d+) · .+?$", md, flags=re.M)
    notes = {}
    for i in range(1, len(parts), 2):
        body = parts[i + 1].split("\n---")[0]
        paras = [ln.strip() for ln in body.strip().split("\n\n") if ln.strip()]
        notes[int(parts[i])] = "\n\n".join(paras)
    qa = md.split("## Likely questions, with short answers", 1)
    if len(qa) == 2:
        qa_text = qa[1].replace("**", "")
        qa_paras = [ln.strip() for ln in qa_text.strip().split("\n\n") if ln.strip()]
        notes["qa"] = "\n\n".join(qa_paras[1:])  # drop the intro line
    return notes


SCRIPT = ROOT / "case_study_1" / "presentation" / "speaker_script.md"
if SCRIPT.exists():
    notes = _load_script(SCRIPT)
    for idx, sl in enumerate(prs.slides, start=1):
        body = notes.get(idx)
        if not body:
            continue
        if idx == len(prs.slides) and "qa" in notes:
            body += "\n\n" + "=" * 40 + "\nLIKELY QUESTIONS\n" + "=" * 40 + "\n\n" + notes["qa"]
        sl.notes_slide.notes_text_frame.text = body

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
