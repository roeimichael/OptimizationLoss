"""Build a three-page, plain-language meeting brief from audited TraLO results."""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/tralo_meeting_brief_20261001.pdf"
EVIDENCE = json.loads((ROOT / "experiments/tralo_pdf_evidence_20261001.json").read_text(encoding="utf-8"))
OUT.parent.mkdir(parents=True, exist_ok=True)

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Brief", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("BriefBold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFontFamily("Brief", normal="Brief", bold="BriefBold")
NAVY = colors.HexColor("#173047")
TEAL = colors.HexColor("#0A776F")
RED = colors.HexColor("#A63D3D")
PALE = colors.HexColor("#F0F5F7")
RULE = colors.HexColor("#CFDEE5")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleB", fontName="BriefBold", fontSize=23, leading=28,
                          textColor=NAVY, spaceAfter=12))
styles.add(ParagraphStyle(name="H1B", fontName="BriefBold", fontSize=15, leading=19,
                          textColor=NAVY, spaceBefore=3, spaceAfter=9))
styles.add(ParagraphStyle(name="H2B", fontName="BriefBold", fontSize=10.5, leading=14,
                          textColor=TEAL, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="BodyB", fontName="Brief", fontSize=9.75, leading=14.5,
                          textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="SmallB", fontName="Brief", fontSize=8.4, leading=12,
                          textColor=NAVY, spaceAfter=6))
styles.add(ParagraphStyle(name="CellB", fontName="Brief", fontSize=8.05, leading=11.3,
                          textColor=NAVY))
styles.add(ParagraphStyle(name="HeadB", fontName="BriefBold", fontSize=8.0, leading=11,
                          textColor=colors.white))
styles.add(ParagraphStyle(name="BoxB", fontName="BriefBold", fontSize=10.8, leading=16,
                          textColor=NAVY))

story = []


def p(text, style="BodyB"):
    return Paragraph(text, styles[style])


def add(text, style="BodyB"):
    story.append(p(text, style))


def h(text):
    add(text, "H1B")
    story.append(HRFlowable(width="100%", thickness=.7, color=RULE, spaceAfter=8))


def sub(text):
    add(text, "H2B")


def box(text):
    t = Table([[p(text, "BoxB")]], colWidths=[176*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), PALE),
        ("BOX", (0,0), (-1,-1), .7, RULE),
        ("LEFTPADDING", (0,0), (-1,-1), 11),
        ("RIGHTPADDING", (0,0), (-1,-1), 11),
        ("TOPPADDING", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 9))


def table(headers, rows, widths):
    data = [[p(x, "HeadB") for x in headers]]
    data.extend([[p(str(x), "CellB") for x in row] for row in rows])
    t = Table(data, colWidths=[w*mm for w in widths], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE]),
        ("GRID", (0,0), (-1,-1), .35, RULE),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.extend([t, Spacer(1, 9)])


def mean_sd(stats, scale=100):
    return f"{stats['mean']*scale:.2f} +/- {stats['seed_sd']*scale:.2f}"


def local(backbone, cap, arm):
    return EVIDENCE["local_fmow2"][backbone]["caps"][str(cap)]["arms"][arm]["cc_f1"]


add("TraLO: what we can tell the professor today", "TitleB")
add("Meeting brief  |  1 October 2026  |  Three pages  |  TraLO is the project's name; 'Trello' refers to the same work.", "SmallB")
box("The short answer: We improved the <b>training pipeline</b> substantially. The TraLO <b>constraint step</b> then gives a small but repeatable benefit on one knee backbone, harms another, and has not shown a useful country-local benefit on satellite images. It is a research lead, not a proven better general model.")
h("1. What problem are we solving?")
add("Imagine a model that scores knee X-rays for disease severity, but a clinic can investigate only <b>76 of 826</b> development cases as grade 3. A normal classifier may call too many cases grade 3. A <b>Clipper</b> solves the capacity problem after training: it keeps the 76 highest-scoring cases and sends the rest to other classes. This automatically obeys the cap. The scientific question is whether changing the model during or after training can put <b>more truly grade-3 cases</b> into those same 76 slots.")
add("We measure that with <b>constrained-class F1 (cc-F1)</b>: a score that rewards both finding true cases and avoiding false ones, <i>after</i> the cap is enforced. We also report accuracy, macro-F1 (equal weight to classes), and weighted-F1 (more weight to common classes). A method that improves the capped class while damaging the rest is a tradeoff, not an overall win.")
h("2. What changed from old TraLO to now?")
table(["Part", "Old TraLO", "Current tested pipeline"], [
    ("Base model", "A short, simple ResNet18 run; final epoch used.",
     "Tested modern MobileNetV3-Large, EfficientNet-B5, and ViT-B/16 as separate backbones. TraLO is a weight update on each backbone, not a new image architecture."),
    ("Training recipe", "Little augmentation; uniform batches; final checkpoint.",
     "We tested Kassif's richer image augmentation, class-balanced sampling and early stopping. We use fixed snapshot averaging to stabilize scores."),
    ("Constraint handling", "A global count-gradient step followed by a cap-aware Clipper.",
     "We preserved the global idea and tested a smaller, checked step. Satellite work also adds country caps. Every arm still gets the same exact allocator."),
], [32,63,81])
add("Yuval Kassif's <b>PAO</b> is different: it repeatedly retrains with extra weight on training examples falsely predicted as the capped class. TraLO uses a gradient of the predicted count on <b>unlabeled</b> deployment images. We borrowed and tested his training recipe, but did <b>not</b> silently substitute PAO for TraLO. The large baseline improvement came mostly from augmentation, not from either constraint loss.")
story.append(PageBreak())

h("3. The numbers that matter")
add("All numbers below are <b>allocated cc-F1 percentages</b>. 'Mean +/- SD' shows variation among seeds; the interval on the difference is a paired 95% confidence interval. These are development-set results, not a final unseen-test result.")
global_m = EVIDENCE["global_knee"]["mobilenetv3"]
global_b = EVIDENCE["global_knee"]["efficientnet_b5"]
table(["Study (seeds)", "Clipper / PTO mean +/- SD", "TraLO mean +/- SD", "TraLO - PTO, 95% interval"], [
    ("Knee, MobileNetV3 (72)", mean_sd(global_m["ens_pto"], 1), mean_sd(global_m["ens_tralo"], 1), "+1.18 pp [+0.84, +1.51]"),
    ("Knee, EfficientNet-B5 (48)", mean_sd(global_b["ens_pto"], 1), mean_sd(global_b["ens_tralo"], 1), "-0.87 pp [-1.52, -0.22]"),
    ("Satellite, MobileNetV3, cap 167 (12)", mean_sd(local("mobilenetv3",167,"ens_pto")), mean_sd(local("mobilenetv3",167,"ens_joint")), "+0.39 pp [-0.19, +0.98]"),
    ("Satellite, ViT-B/16, cap 167 (12)", mean_sd(local("vit_b16",167,"ens_pto")), mean_sd(local("vit_b16",167,"ens_joint")), "-0.09 pp [-0.36, +0.19]"),
], [58,38,39,41])
add("At the tighter satellite cap of 83, MobileNetV3 changes from <b>39.46% to 39.74%</b> (+0.28 pp, interval [-0.18, +0.74]); ViT changes from <b>39.02% to 39.07%</b> (+0.06 pp, interval [-0.53, +0.64]). Both intervals include zero. The earlier fixed 0.1 local step was actively harmful: <b>-8.66</b> and <b>-6.97</b> points at caps 167 and 83. We preserved that failure.")
sub("Did the result come from Yuval's idea instead?")
add("Inside a matched version of Kassif's pipeline, ordinary training plus Clipper improved the older knee baseline by about <b>4.81 cc-F1 points</b> on ResNet18. That comparison is between studies, so it tells us the recipe matters but is not a clean causal estimate for every ingredient. A separate 24-seed factorial measured <b>+5.64 points</b> from image augmentation alone. His PAO loss itself did <b>not</b> reliably beat ordinary training plus Clipper: on EfficientNet-B5 it changed cc-F1 by +0.96 points with interval [-0.28, +2.20]; on ResNet18 it changed by -0.09 [-2.85, +2.66]. His method is not an established winner in our matched re-run either.")
sub("What about ALM, null, and other metrics?")
add("The zero-step <b>null/PTO</b> is the main fair control: same trained model and same allocator, without TraLO's step. A random step of the same size is a second check. MobileNetV3's knee gain survived both. The satellite local method did not pass the full metric bar: at cap 167, MobileNetV3 accuracy fell <b>54.12% to 52.52%</b>, macro-F1 <b>47.73% to 46.31%</b>, and weighted-F1 <b>55.85% to 54.41%</b>. ViT's primary result was near zero and its accuracy also fell. The older ALM comparison used frozen features and only four seeds; it is <b>not</b> a fair full-backbone ALM verdict. The matched full-training ALM comparison has not run yet.")
story.append(PageBreak())

h("4. What do the training logs tell us?")
add("The satellite steps were not inactive. In the calibrated MobileNetV3 study, all <b>72 of 72</b> planned joint snapshot steps were applied at each cap. The mean step was much smaller than the failed 0.1 version. In ViT, all <b>78 of 78</b> were applied. Yet the number of correct images added to the limited slots was almost matched by correct images pushed out: at ViT cap 167, about <b>4.58 correct entries</b> versus <b>4.75 correct exits</b> per seed. The method changed the ranking; it did not improve the ranking enough.")
add("Training also showed a warning sign. In ViT, average training loss fell from <b>0.835</b> at epoch 1 to <b>0.128</b> at epoch 6, while stopping-set loss rose from <b>1.532</b> to <b>2.236</b>. Eight of twelve seeds selected epoch 1 as best. MobileNetV3 showed the same pattern. This suggests that memorizing training images is part of the bottleneck, but the logs alone do not prove it caused the F1 result.")
h("5. Should we keep researching TraLO?")
box("<b>Yes, but as a narrow research question.</b> The knee MobileNetV3 signal is real within this development protocol: about one extra correct case among 76 slots. The larger B5 backbone reverses the effect, and the country-local satellite results do not demonstrate a practical gain. We should not call TraLO a generally improved model or use the satellite numbers as a headline win.")
add("The practical gain worth keeping right now is the <b>better supervised recipe and snapshot averaging</b>. That can help even if TraLO is removed. To justify TraLO as a method, we must show that its directed update adds value over this strong post-hoc baseline, not merely that the whole newer pipeline beats the older one.")
sub("Next three tests, in order")
add("<b>1. Finish the matched full-training comparison.</b> On the same modern backbone and fixed data, compare TraLO, full ALM, Clipper/PTO, and a zero-constraint null at equal compute. Its implementation is prepared but has not cleared all review and release gates, so there is no honest full-ALM performance number yet.")
add("<b>2. Attack the ranking problem.</b> TraLO reduces a predicted count, but the allocator already enforces counts. A useful next method must improve <i>which</i> images enter the scarce slots. Design a training-label-aware ranking signal with a same-dose null and stop it if it only changes counts or pushes out correct cases.")
add("<b>3. Confirm once on untouched data.</b> Freeze the method, cap, backbone, metrics and analysis before evaluating a genuinely new hospital or country. The repeatedly inspected development cohorts are useful for diagnosis but cannot certify generalization. The sealed knee test and reserved satellite countries have not been opened.")
add("<b>Meeting conclusion:</b> Kassif's training recipe improved our practical baseline more than either PAO or TraLO did. TraLO still has a small, credible knee lead on MobileNetV3, but its effect is fragile across backbones and absent locally. Continue only if the next matched test explains and reproduces an advantage over a strong Clipper/null; otherwise prioritize the supervised model and move beyond count-only constraint updates.")
add("Evidence: audited seed scores in experiments/claude_stepens_additional_result_20260928.md, fmow_boundary_mnv3_result_20261001.md, fmow_boundary_vit_v2_result_20261001.md; Kassif replication in claude_yuval_pipeline_result_20260927.md. Full PDF report v2 and failed scorer receipts are preserved. No sealed test labels were used.", "SmallB")


def footer(canvas, doc):
    canvas.saveState()
    w, height = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, height-11*mm, w, 11*mm, stroke=0, fill=1)
    canvas.setFont("BriefBold", 7.6)
    canvas.setFillColor(colors.white)
    canvas.drawString(17*mm, height-7*mm, "TraLO  /  MEETING BRIEF")
    canvas.drawRightString(w-17*mm, height-7*mm, "1 OCT 2026")
    canvas.setStrokeColor(RULE)
    canvas.line(17*mm, 14*mm, w-17*mm, 14*mm)
    canvas.setFont("Brief", 7.4)
    canvas.setFillColor(NAVY)
    canvas.drawString(17*mm, 9*mm, "Development evidence; untouched tests remain sealed")
    canvas.drawRightString(w-17*mm, 9*mm, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
                        topMargin=20*mm, bottomMargin=18*mm,
                        title="TraLO: meeting brief", author="TraLO research audit")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
