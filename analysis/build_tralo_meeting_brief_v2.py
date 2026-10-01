"""Build the focused three-page meeting brief from audited, development-only results."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/tralo_meeting_brief_20261001_v2.pdf"
E = json.loads((ROOT / "experiments/tralo_pdf_evidence_20261001.json").read_text(encoding="utf-8"))
OUT.parent.mkdir(parents=True, exist_ok=True)

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Brief", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("BriefBold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFontFamily("Brief", normal="Brief", bold="BriefBold")
NAVY = colors.HexColor("#173047")
TEAL = colors.HexColor("#0A776F")
PALE = colors.HexColor("#F0F5F7")
RULE = colors.HexColor("#CFDEE5")

styles = getSampleStyleSheet()
for name, font, size, leading, color, before, after in [
    ("TitleB", "BriefBold", 22, 27, NAVY, 0, 10),
    ("H1B", "BriefBold", 14, 18, NAVY, 2, 7),
    ("H2B", "BriefBold", 10.2, 13.3, TEAL, 7, 4),
    ("BodyB", "Brief", 9.4, 13.6, NAVY, 0, 7),
    ("SmallB", "Brief", 8.05, 11.2, NAVY, 0, 5),
    ("CellB", "Brief", 7.9, 10.8, NAVY, 0, 0),
    ("HeadB", "BriefBold", 7.8, 10.6, colors.white, 0, 0),
    ("BoxB", "BriefBold", 10.1, 14.4, NAVY, 0, 0),
]:
    styles.add(ParagraphStyle(name=name, fontName=font, fontSize=size, leading=leading,
                              textColor=color, spaceBefore=before, spaceAfter=after))

story = []


def p(value, style="BodyB"):
    return Paragraph(str(value), styles[style])


def add(value, style="BodyB"):
    story.append(p(value, style))


def h(value):
    add(value, "H1B")
    story.append(HRFlowable(width="100%", thickness=.7, color=RULE, spaceAfter=7))


def sub(value):
    add(value, "H2B")


def box(value):
    t = Table([[p(value, "BoxB")]], colWidths=[176 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), .7, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([t, Spacer(1, 8)])


def table(headers, rows, widths, compact=False):
    cells = [[p(x, "HeadB") for x in headers]]
    cells.extend([[p(x, "CellB") for x in row] for row in rows])
    t = Table(cells, colWidths=[w * mm for w in widths], repeatRows=1, hAlign="LEFT")
    pad = 4 if compact else 5
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), .35, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
    ]))
    story.extend([t, Spacer(1, 7)])


def matrix_from_f1(f1_percent):
    # Every seed allocates 76 of 826 development cases to grade 3; 106 are truly grade 3.
    tp = (f1_percent / 100) * (76 + 106) / 2
    return tp, 76 - tp, 106 - tp, 826 - 76 - 106 + tp


add("TraLO: the promising result, in context", "TitleB")
add("Three-page meeting brief  |  1 October 2026  |  Development-set evidence", "SmallB")
box("Our best supported finding is modest: on knee X-rays with MobileNetV3-Large, TraLO places about <b>one more true grade-3 case</b> among 76 available slots than the same model with the same post-training Clipper. This is a 72-seed result. It is <b>not</b> evidence that TraLO works on every dataset or backbone.")
h("1. What are we trying to improve?")
add("Suppose a clinic can send only <b>76 of 826</b> knee X-rays to the grade-3 treatment pathway. First a network assigns a score to each image. Then a <b>Clipper</b> enforces the limit by reserving the 76 highest grade-3 scores. The clinic cares about <i>which patients</i> occupy those slots, not just whether the number is exactly 76.")
add("The headline measure is <b>constrained-class F1 (cc-F1)</b> for grade 3 after the Clipper. It balances missed true cases and incorrect referrals. We also track accuracy, macro-F1 across all five grades, and weighted-F1, which gives common grades more influence. A gain in cc-F1 accompanied by damage to all other grades would need careful judgment.")
h("2. Where is TraLO, and where is Kassif's method?")
table(["Step", "Kassif's PAO", "Our TraLO study"], [
    ("Supervised base", "Train an image network on labeled knee images.",
     "We adopted and tested his stronger training recipe: image augmentation, class-balanced sampling and early stopping."),
    ("Constraint update", "PAO repeatedly changes the cost-sensitive training loss, up-weighting labeled training errors that overfill a class, then retrains.",
     "TraLO keeps the supervised run fixed and takes a small gradient step on each saved model snapshot using predicted counts from <b>unlabeled</b> deployment images."),
    ("Final decision", "Apply the capacity-aware Clipper.",
     "Apply the <b>same</b> Clipper to TraLO and all controls; average snapshot probabilities before the cut."),
], [28, 73, 75], compact=True)
add("So we did <b>not</b> simply rename PAO. The architecture can be the same; the scientific difference is the constraint update. The stronger base training recipe improved our old pipeline substantially, even without TraLO. The experiment below isolates TraLO's additional contribution over that stronger baseline.")
story.append(PageBreak())

h("3. The positive comparison: knee X-rays, MobileNetV3-Large")
add("The 72 seeds use the same images, model training, saved snapshots and Clipper. Only the TraLO direction changes. Percentages are means across seeds; +/- is between-seed standard deviation (SD). 'pp' means percentage points.")
g = E["global_knee"]["mobilenetv3"]
table(["Measure", "Plain snapshots + Clipper", "TraLO snapshots + Clipper", "Paired TraLO gain, 95% CI"], [
    ("Grade-3 cc-F1", f"{g['ens_pto']['mean']:.2f} +/- {g['ens_pto']['seed_sd']:.2f}%",
     f"<b>{g['ens_tralo']['mean']:.2f} +/- {g['ens_tralo']['seed_sd']:.2f}%</b>", "<b>+1.18 pp [+0.84, +1.51]</b>"),
    ("Overall accuracy", "See paired contrast", "See paired contrast", "+0.87 pp [+0.63, +1.11]"),
    ("Macro-F1", "See paired contrast", "See paired contrast", "+0.93 pp [+0.67, +1.19]"),
    ("Weighted-F1", "See paired contrast", "See paired contrast", "+0.46 pp [+0.23, +0.69]"),
], [35, 46, 47, 48], compact=True)
add("The full score file reports paired differences for the three secondary metrics, not the two absolute arm averages, so the table gives the audited differences rather than inventing missing averages. The same-size <b>random sham step</b> adds essentially nothing to cc-F1; TraLO also beats that sham by +1.18 pp [ +0.85, +1.50 ]. The prespecified primary contrasts pass their adjusted significance test.", "SmallB")

sub("What does a 1.18-point improvement mean for patients?")
pto = matrix_from_f1(g["ens_pto"]["mean"])
tralo = matrix_from_f1(g["ens_tralo"]["mean"])
table(["Mean grade-3 decision counts", "True grade 3", "Not grade 3"], [
    ("Plain / selected", f"{pto[0]:.1f} correct", f"{pto[1]:.1f} incorrect"),
    ("Plain / not selected", f"{pto[2]:.1f} missed", f"{pto[3]:.1f} correctly excluded"),
    ("TraLO / selected", f"<b>{tralo[0]:.1f} correct</b>", f"{tralo[1]:.1f} incorrect"),
    ("TraLO / not selected", f"{tralo[2]:.1f} missed", f"{tralo[3]:.1f} correctly excluded"),
], [62, 55, 59], compact=True)
add("This is a <b>derived mean 2-by-2 matrix</b>, not one seed's raw confusion matrix: each run selects 76 cases, the development pool has 106 true grade-3 cases, and the mean cc-F1 determines mean true positives. The effect is roughly <b>one extra correct patient in the same 76 slots</b>.", "SmallB")

sub("A separate local-constraint result is not yet a win")
m = E["local_fmow2"]["mobilenetv3"]["caps"]["167"]["arms"]
table(["Satellite MobileNetV3, cap 167", "Plain + Clipper", "Local TraLO + Clipper"], [
    ("Class-1 cc-F1", f"{m['ens_pto']['cc_f1']['mean']*100:.2f}%", f"{m['ens_joint']['cc_f1']['mean']*100:.2f}%"),
    ("Accuracy / macro-F1 / weighted-F1",
     "/".join(f"{m['ens_pto'][k]['mean']*100:.2f}" for k in ["accuracy", "macro_f1", "weighted_f1"]),
     "/".join(f"{m['ens_joint'][k]['mean']*100:.2f}" for k in ["accuracy", "macro_f1", "weighted_f1"])),
], [63, 56, 57], compact=True)
add("The cc-F1 change is only +0.39 pp, with a 95% interval from -0.19 to +0.98; accuracy and the broader F1 scores fall. We therefore do not call this a positive transfer result.", "SmallB")
story.append(PageBreak())

h("4. Kassif's PAO beside TraLO: our matched re-run")
add("The table below is <b>our own 24-seed ResNet18 reproduction of Kassif's training pipeline</b> on the same knee development pool and grade-3 cap, with his PAO, plain predict-then-optimize (PTO), and our TraLO side-step. It is <b>not a numerical table copied from Kassif's paper</b>.")
table(["Method in our re-run", "cc-F1", "Accuracy", "Macro-F1", "Weighted-F1"], [
    ("Plain PTO + Clipper", "69.78%", "62.45%", "<b>62.83%</b>", "<b>61.25%</b>"),
    ("Kassif PAO + Clipper", "69.69%", "61.38%", "62.63%", "60.58%"),
    ("TraLO step + Clipper", "<b>70.51%</b>", "<b>62.61%</b>", "62.65%", "60.89%"),
], [54, 29, 29, 32, 32], compact=True)
add("Bold marks the highest <i>point estimate within this table</i>, not a proven overall winner. Across seeds, PAO minus plain PTO is -0.09 cc-F1 points [ -2.85, +2.66 ]; TraLO minus its same-size random sham is +0.73 [ +0.08, +1.38 ], but misses the prespecified multiple-comparison threshold (adjusted p = 0.088). The paired difference SDs are 6.52 and 1.54 points, respectively. This smaller-backbone result is suggestive, not decisive.", "SmallB")
sub("What did Kassif's article itself report?")
add("Kassif and Singer describe PAO as outperforming a predict-then-optimize pipeline on knee images and DermaMNIST, especially under tight capacity. The accessible publisher preview does <b>not</b> expose its numerical results tables, and its setup is not identical to our matched re-run. We therefore cannot honestly place claimed article percentages in the table above. The source is DOI 10.1016/j.engappai.2026.115989; the author's code is github.com/YuvalKassif/ConstrainedClassification.")
h("5. Is this enough for a paper?")
box("<b>A lead, not a finished claim.</b> TraLO is mathematically distinct from PAO and has a reproducible 72-seed knee MobileNetV3 gain over a strong Clipper and a random-step control. But the gain is small, the development set has been viewed repeatedly, and local satellite results are inconclusive. A general or clinical superiority claim would be premature.")
add("The first next experiment is a <b>matched full-training comparison</b> of TraLO, PAO/ALM, plain Clipper and a zero-update null on one frozen modern backbone and the same compute budget. That experiment is prepared but has no accepted result yet. Second, study the <b>ranking</b> of cases entering limited slots: lowering a predicted class count is less valuable when the Clipper already enforces the count. Third, freeze the method and evaluate once on untouched cases; the sealed test sets remain unopened.")
add("For the professor: our practical training pipeline improved mostly because of better image training and snapshot averaging borrowed and tested from Kassif's setup. TraLO adds a separate, small knee benefit on MobileNetV3. If the matched comparison and untouched evaluation fail to preserve that benefit, we should keep the stronger base classifier and stop pursuing this count-gradient version of TraLO.")
add("Audit sources: experiments/claude_stepens_additional_result_20260928.md; experiments/claude_yuval_pipeline_result_20260927.md; experiments/fmow_boundary_mnv3_result_20261001.md; experiments/tralo_pdf_evidence_20261001.json. No reserved-country or sealed knee labels were used.", "SmallB")


def footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 11 * mm, width, 11 * mm, stroke=0, fill=1)
    canvas.setFont("BriefBold", 7.6)
    canvas.setFillColor(colors.white)
    canvas.drawString(17 * mm, height - 7 * mm, "TraLO  /  MEETING BRIEF")
    canvas.drawRightString(width - 17 * mm, height - 7 * mm, "1 OCT 2026")
    canvas.setStrokeColor(RULE)
    canvas.line(17 * mm, 14 * mm, width - 17 * mm, 14 * mm)
    canvas.setFont("Brief", 7.4)
    canvas.setFillColor(NAVY)
    canvas.drawString(17 * mm, 9 * mm, "Development evidence; untouched tests remain sealed")
    canvas.drawRightString(width - 17 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm,
                        topMargin=20 * mm, bottomMargin=18 * mm,
                        title="TraLO: focused meeting brief", author="TraLO research audit")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
