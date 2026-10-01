"""Build the dated, evidence-linked TraLO research briefing."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Circle, Drawing, Line, String
from reportlab.platypus import (
    HRFlowable, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "tralo_research_status_20261001_v2.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

FONT_ROOT = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("ArialT", str(FONT_ROOT / "arial.ttf")))
pdfmetrics.registerFont(TTFont("ArialTB", str(FONT_ROOT / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("ArialTI", str(FONT_ROOT / "ariali.ttf")))
pdfmetrics.registerFontFamily("ArialT", normal="ArialT", bold="ArialTB", italic="ArialTI", boldItalic="ArialTB")

INK = colors.HexColor("#173047")
NAVY = colors.HexColor("#10283F")
BLUE = colors.HexColor("#176A9C")
TEAL = colors.HexColor("#0A776F")
AMBER = colors.HexColor("#A65B12")
RED = colors.HexColor("#A63D3D")
PALE = colors.HexColor("#EFF5F8")
PALE2 = colors.HexColor("#F7F9FA")
RULE = colors.HexColor("#D7E3E9")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverKicker", fontName="ArialTB", fontSize=11, leading=15,
                          textColor=TEAL, spaceAfter=14, uppercase=True))
styles.add(ParagraphStyle(name="CoverTitle", fontName="ArialTB", fontSize=27, leading=32,
                          textColor=NAVY, spaceAfter=20))
styles.add(ParagraphStyle(name="CoverSub", fontName="ArialT", fontSize=12, leading=18,
                          textColor=INK, spaceAfter=17))
styles.add(ParagraphStyle(name="H1x", fontName="ArialTB", fontSize=18, leading=23,
                          textColor=NAVY, spaceBefore=7, spaceAfter=11, keepWithNext=True))
styles.add(ParagraphStyle(name="H2x", fontName="ArialTB", fontSize=11.2, leading=15,
                          textColor=BLUE, spaceBefore=12, spaceAfter=5, keepWithNext=True))
styles.add(ParagraphStyle(name="Bodyx", fontName="ArialT", fontSize=9.3, leading=14.2,
                          textColor=INK, spaceAfter=8))
styles.add(ParagraphStyle(name="Smallx", fontName="ArialT", fontSize=8.1, leading=11.5,
                          textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Tinyx", fontName="ArialT", fontSize=7.1, leading=10,
                          textColor=INK, spaceAfter=3))
styles.add(ParagraphStyle(name="TableHeadx", fontName="ArialTB", fontSize=7.5, leading=10,
                          textColor=colors.white))
styles.add(ParagraphStyle(name="TableCellx", fontName="ArialT", fontSize=7.55, leading=10.5,
                          textColor=INK))
styles.add(ParagraphStyle(name="Calloutx", fontName="ArialTB", fontSize=9.1, leading=13.7,
                          textColor=NAVY))
styles.add(ParagraphStyle(name="Sourcex", fontName="ArialT", fontSize=7.5, leading=10.8,
                          textColor=INK, spaceAfter=4))

story = []

def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])

def add(text, style="Bodyx"):
    story.append(P(text, style))

def title(num, text):
    story.append(P(f"{num}. {text}", "H1x"))
    story.append(HRFlowable(width="100%", thickness=1, color=RULE, spaceAfter=9))

def sub(text):
    add(text, "H2x")

def note(text, tint=PALE):
    box = Table([[P(text, "Calloutx")]], colWidths=[175*mm])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), tint),
        ("BOX", (0,0), (-1,-1), 0.6, RULE),
        ("LEFTPADDING", (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(box)
    story.append(Spacer(1, 9))

def bullet(text):
    add("<font color='#0A776F'><b>•</b></font>  " + text, "Bodyx")

def table(headers, rows, widths, font=7.55):
    data = [[P(h, "TableHeadx") for h in headers]]
    data += [[P(str(x), "TableCellx") for x in row] for row in rows]
    t = Table(data, colWidths=[w*mm for w in widths], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, PALE2]),
        ("GRID", (0,0), (-1,-1), 0.25, RULE),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 9))

def newpage():
    story.append(PageBreak())

def knee_effect_plot():
    """Show paired effect and CI on a common percentage-point axis."""
    draw = Drawing(490, 125)
    x0, x1 = 155, 470
    lo, hi = -1.8, 1.8
    xp = lambda v: x0 + (v-lo)/(hi-lo)*(x1-x0)
    draw.add(String(x0, 116, "Paired TraLO - PTO effect on grade-3 F1 (percentage points)",
                    fontName="ArialTB", fontSize=8.2, fillColor=NAVY))
    for tick in (-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5):
        x = xp(tick)
        draw.add(Line(x, 16, x, 100, strokeColor=RULE if tick else INK,
                      strokeWidth=0.5 if tick else 1.0))
        draw.add(String(x-8, 6, f"{tick:+.1f}", fontName="ArialT", fontSize=6.7,
                        fillColor=INK))
    for i, (name, est, low, high, tone) in enumerate([
        ("MobileNetV3", 1.18, .84, 1.51, TEAL),
        ("EfficientNet-B5", -.87, -1.52, -.22, RED),
        ("ResNet18", 1.11, .78, 1.45, BLUE),
        ("RegNetY-400MF", .93, .57, 1.29, BLUE),
    ]):
        y = 92-i*23
        draw.add(String(4, y-3, name, fontName="ArialT", fontSize=7.6,
                        fillColor=INK))
        draw.add(Line(xp(low), y, xp(high), y, strokeColor=tone, strokeWidth=2))
        draw.add(Line(xp(low), y-4, xp(low), y+4, strokeColor=tone, strokeWidth=1))
        draw.add(Line(xp(high), y-4, xp(high), y+4, strokeColor=tone, strokeWidth=1))
        draw.add(Circle(xp(est), y, 3.3, fillColor=tone, strokeColor=tone))
    return draw

add("RESEARCH BRIEFING  /  1 OCTOBER 2026", "CoverKicker")
add("TraLO after Kassif:<br/>what changed, what worked,<br/>and what still fails", "CoverTitle")
add("A numerical and methodological account of the knee and fmow2 studies, with MobileNetV3, EfficientNet-B5, and ViT-B/16 placed ahead of the small diagnostic backbones.", "CoverSub")
note("Current verdict: a reproducible, small <b>global</b> knee development signal exists on some backbones. A useful <b>country-local</b> advantage has not been established. The calibrated MobileNetV3 and ViT-B/16 local blocks are now both independently audited; neither shows a reliable primary gain, and both lose overall accuracy.")
add("Evidence cutoff: 1 October 2026, approximately 08:00 UTC. This is a progress report, not a submission-ready paper or held-out validation. The 1,656-image Chen knee test and five reserved fmow2 countries remain sealed. This v2 PDF preserves the earlier report and adds the completed ViT score.", "Smallx")
add("How to read the numbers: all reported intervals are seed-paired on already inspected development cohorts. A confidence interval crossing zero does not demonstrate a benefit. A positive development effect alone does not establish generalization or a deployable clinical/satellite system.", "Smallx")
story.append(Spacer(1, 16))
table(["The question", "Answer today"], [
    ("Did adopting Kassif's recipe help?", "Yes, for the plain knee baseline; the factorial attributes most of the gain to augmentation, not to the constraint loss."),
    ("Does TraLO improve modern backbones?", "Global knee step: +1.18 pp on MobileNetV3-Large; -0.87 pp on EfficientNet-B5. The effect is not architecture-invariant."),
    ("Does the local method work on fmow2?", "The original 0.1 step clearly harmed class-1 F1. A calibrated smaller step gained only +0.39/+0.28 pp at two caps, with intervals crossing zero and accuracy losses."),
    ("Is ViT a win?", "The audited 12-seed ViT block is -0.09/+0.06 cc-F1 percentage points at the two caps, both intervals spanning zero; accuracy also falls."),
], [53,122])
add("Terminology: the user has called the project 'Trello'. The method and repository call it <b>TraLO</b>. Kassif's loss is <b>PAO</b>; <b>PTO</b> means train normally, then allocate predictions under a capacity constraint.", "Smallx")
newpage()

title(1, "The decision problem and the metric")
add("A classifier predicts a probability for each class. A downstream service cannot act on unlimited positive predictions: at most K grade-3 knee cases can be selected, or at most K class-1 satellite images can be called under a pooled cap and country caps. The deployed decision is therefore the model's scores <b>plus</b> a capacity-aware allocator. Raw argmax predictions are useful diagnostics, but they are not the deployed policy.")
add("For the knee development pool, the allocator assigns exactly 76 grade-3 slots from 826 images. For fmow2, it uses 1,673 images from five development countries, pooled class-1 caps of 167 or 83, and additional country upper bounds derived from country membership without reading country labels. Allocation and evaluation are separate: the training/allocator path never receives development labels; an independent scorer opens those labels only after provenance and integrity gates pass.")
sub("Why a method can win one column and lose another")
add("The main constrained-class F1 (cc-F1) measures precision and recall for the capped class after allocation: F1 = 2TP/(2TP + FP + FN). With an exactly filled cap K and a fixed number of true positives in the pool, it is determined by how many correct items occupy the K slots. Accuracy counts all classes; macro-F1 averages class F1 equally; weighted-F1 weights by class support. A step may improve which images occupy scarce class-1 or grade-3 slots while damaging decisions among other classes. That is a real tradeoff, not an accounting contradiction.")
table(["Term", "What is actually evaluated", "Why it matters"], [
    ("Raw call count", "Number of argmax predictions for capped class before allocation.", "Can collapse even while allocator still fills the cap; exposes whether a step is too strong."),
    ("PTO / local null", "Same trained scores and exact allocator, no constraint side-step.", "Strong comparator: post-hoc allocation already enforces the hard quotas."),
    ("Sham", "Random direction matched to TraLO's parameter displacement, then same allocator.", "Separates constraint direction from a generic parameter perturbation."),
    ("Pooled-only", "Only pooled count gradient, without country terms, then country-aware allocation.", "Tests whether the local component adds beyond the pooled direction."),
    ("PHR/ALM", "Penalty or dual-weighted constraint direction; full ALM would alter training through time.", "A snapshot direction is not evidence about persistent full ALM training."),
], [31,72,72])
note("The same numerical percentage-point change is <b>not</b> interchangeable across tasks: knee grade-3 F1 and fmow2 class-1 F1 have different images, class supports, caps, backbones, and seed blocks.")
newpage()

title(2, "Before Kassif: the original TraLO recipe")
add("The earlier knee end-to-end TraLO v3 study trained an ImageNet-pretrained ResNet18 on labeled knee images with cross-entropy, uniform batches, ImageNet normalization, no online augmentation, a short 10-epoch budget, and the final-epoch state. Its exact post-hoc Clipper was the practical baseline. TraLO added a count-gradient step to a copy of that trained model; it did not add a new prediction head or replace the backbone architecture. The prediction score order, not merely the number of raw positive calls, determined the constrained F1 after the exact cut.")
add("That simple recipe had a diagnosed weakness. The training set became nearly memorized within a few epochs, so label-aware boundary information on the training set was sparse; meanwhile development raw counts swung sharply across epochs. A single final checkpoint was a noisy estimate of the ranking used at deployment. The original local constraint question was still unresolved: a pooled cap alone does not imply country-level feasibility.")
sub("What the published Kassif idea added")
add("Kassif and Singer's resource-constrained medical-image work proposed PAO, an adaptive training loss for an operational class cap. In the audited repository, PAO increases the loss weight of training examples predicted as the capped class when their <i>true</i> class is different. It repeatedly retrains from the same nominal starting point while a multiplier C grows until the model's raw positive count meets a target. It can therefore distinguish training false positives from true positives. TraLO's soft-count gradient is label-free on the unlabeled deployment pool: it cannot make that distinction directly. Both systems can be followed by the same exact post-hoc allocation.")
add("The paper's reported PAO advantage was not accepted as a clean baseline. The repository audit found one-seed reporting, a target cap and stopping condition informed by test-set information, no reseed between some retrains, and outcome filtering in a plotting path. Those are evaluation concerns, not proof that the published effect is false. Our paired re-run used label-free cap construction and held test labels back. The repository and the journal abstract are listed in the sources.")
sub("The important distinction")
note("We borrowed and tested Kassif's <b>training recipe</b>; we did not rename PAO as TraLO. PAO is a label-aware, repeatedly trained loss. TraLO is an additional constraint-direction step on a trained model or snapshot. The architecture is a separate choice. The allocator is a separate deployment choice.")
newpage()

title(3, "After Kassif: the training and deployment pipeline")
table(["Pipeline component", "Earlier TraLO v3", "Kassif-matched knee comparison / current direction"], [
    ("Backbone", "ImageNet ResNet18 diagnostic baseline.", "ResNet18 reproduction, then MobileNetV3-Large, EfficientNet-B5 and ViT-B/16 extensions; same class head per backbone."),
    ("Images", "RGB copy of gray knee X-ray, 224 px, ImageNet normalization.", "Kassif knee recipe: RGB 224 with random flip, small rotation, affine shift/scale and color jitter, dataset mean/std. fmow2 later uses full-frame RGB 224 and pinned ImageNet normalization."),
    ("Sampling", "Uniform shuffled batches, each training example once per epoch.", "Kassif knee recipe uses class-balanced sampling with replacement. The newer fmow2 studies retain a fixed balanced recipe."),
    ("Optimizer", "Cross-entropy with short 10-epoch training; final state used.", "Matched knee recipe uses Adam lr 1e-4, weight decay 1e-4, batch 32, decay x0.8 every five epochs. PAO additionally has its own C-dependent effective LR."),
    ("Stopping", "Final epoch, no train-carved early-stop set.", "A subject-grouped 10% carve from training supplies early-stop loss; best checkpoint restored. In current fmow2 work, a stop-country split serves this role."),
    ("Score stabilizing", "One checkpoint's probabilities.", "Fixed snapshot window averaged into one probability vector before allocation. This is ensemble averaging over epochs of one training run, not a changed network architecture."),
    ("Constraint action", "One count-direction step on a model copy; global cap.", "Global snapshot side-steps, then pooled-plus-country directions. Boundary policy accepts a smaller step only when label-free soft/hard checks pass; no extra supervised training for those side copies."),
    ("Deployment", "Exact capped-first cut.", "Same exact knee cut, or country-aware local capped-first on fmow2; all comparator arms share it."),
], [33,68,74])
add("These changes were not all improvements. The eight-cell, 24-seed factorial separated augmentation, balanced sampling, and early stopping. It found a large augmentation effect on the plain PTO baseline, no detectable mean cc-F1 gain from the sampler, and a small negative mean early-stop effect at the fixed knee cap. We therefore report recipe gains separately from losses or gains caused by the TraLO direction.")
newpage()

title(4, "How the algorithm looks now")
table(["Stage", "Inputs and action", "Invariant / evidence"], [
    ("1. Split and preflight", "Freeze train, stop, development, and sealed test identities; hash images, labels, configs, weights and code.", "No overlap or byte duplicates; declared cap does not read evaluation labels."),
    ("2. Ordinary training", "One supervised trajectory per seed/backbone. Save epoch events, stop loss, checkpoint hashes, PTO predictions.", "All comparison arms begin from the same trained state and snapshot window."),
    ("3. Side-copy methods", "At each eligible snapshot, copy parameters. Compute pooled and (if declared) country soft-count gradients on unlabeled pool. TraLO moves a fixed or accepted boundary radius; pooled-only omits country terms; sham uses matched dose; PHR uses dual/penalty direction.", "Main training optimizer, batch-normalization buffers, RNG and supervised batches remain unchanged; actual gradient, parameter dose and before/after counts are checked."),
    ("4. Ensemble", "Average each arm's fixed snapshot probabilities into a deployable score vector.", "Same snapshot policy across arms; failed/unused side-steps are explicit zeros, not silently omitted."),
    ("5. Allocation", "Apply the same exact hard-cap allocator to every arm; knee has one global cap, fmow2 has pooled and country caps.", "Independent scorer recounts capacity, selection, artifacts and source provenance before reading development labels."),
    ("6. Evaluation", "Compute absolute cc-F1, accuracy, macro/weighted F1 and paired seed effects; retain every seed and all failures.", "Development is exploratory when repeatedly viewed. Sealed cohorts remain unopened."),
], [30,92,53])
sub("What is and is not a 'new model'")
add("Changing ResNet18 to MobileNetV3-Large, EfficientNet-B5, or ViT-B/16 changes the feature extractor architecture and pretrained weights. Applying TraLO to a side copy does <b>not</b> train a separate new architecture; it changes weights of that same trained backbone by a small constraint-directed displacement before probabilities are averaged. The snapshot ensemble uses multiple temporal states from one training trajectory. PAO, by contrast, can require multiple full retrainings. The prospective persistent TraLO versus full ALM panel would change the actual training trajectory, but has not yet produced a matched full-backbone result.")
newpage()

title(5, "What the Kassif replication actually showed")
add("The paired knee study used 24 seeds on ResNet18 and another 24 on Kassif's EfficientNet-B5. Each seed compared plain PTO, PAO's final retrain, TraLO's targeted step, and a dose-matched sham under the same backbone recipe and an exact 76-slot allocator. The target cap came from training prevalence; the 826-image development pool supplied unlabeled images for counts, and labels only for offline scoring. Common initialization, sampler order, and augmentation draws were verified across retrains.")
table(["Study", "Plain PTO cc-F1", "PAO - PTO, 95% CI", "TraLO - sham, 95% CI", "Interpretation"], [
    ("ResNet18 (24 seeds)", "69.78%", "-0.09 pp [-2.85, +2.66]", "+0.73 pp [+0.08, +1.38]; Holm p=0.088", "No family-adjusted PAO or TraLO advantage."),
    ("EfficientNet-B5 (24 seeds)", "69.96%", "+0.96 pp [-0.28, +2.20]", "-0.50 pp [-1.35, +0.34]", "No reliable constraint-loss advantage; TraLO also hurt macro/weighted F1 by about 2 pp."),
], [33,25,40,45,32])
add("The plain Kassif-style PTO baseline was about 4.81 cc-F1 points above the older ResNet18 recipe's 64.97% baseline in an <b>unpaired</b> cross-study comparison. That is evidence of a practical recipe difference, not evidence for PAO or TraLO. An exploratory single-run snapshot ensemble added +2.43 points on ResNet18; a separately fixed B5 block confirmed +4.08 [+3.04, +5.11] points over B5's best checkpoint. A post-hoc B5 ensemble score of about 74.04% was greater than either PAO or TraLO in that matched pipeline. These ensemble comparisons do not make TraLO the cause of the gain.")
sub("Factorial explanation: which recipe lever mattered?")
table(["ResNet18 factorial, 24 paired seeds", "Mean cc-F1 effect (95% CI)", "Reading"], [
    ("Augmentation on vs off", "+5.64 pp [+4.60, +6.68]", "Large, multiplicity-adjusted positive plain-baseline effect."),
    ("Balanced sampling on vs off", "+0.72 pp [-0.24, +1.68]", "No detectable mean cc-F1 benefit; accuracy declined."),
    ("Early stopping on vs off", "-0.95 pp [-1.77, -0.13]", "Small mean loss at this cap; interacts with augmentation."),
    ("TraLO vs sham pooled over 8 recipes", "+0.72 pp [+0.46, +0.97]", "Small real direction signal on ResNet18, but below the snapshot ensemble gain in every cell's point estimate."),
], [61,48,66])
add("In logs, PAO repeatedly reduced the raw capped-class count but overshot the target and reacted to noisy epoch counts. In the B5 block it needed 2.71 retrains per seed on average; an equal-compute three-model plain ensemble had 73.35% cc-F1 versus PAO's 70.92% in the exploratory group comparison. No single favorable PAO seed is promoted over the complete paired block.")
newpage()

title(6, "Modern backbones: global knee snapshot study")
add("The later global step-ensemble study asked a narrower question: given a Kassif-style trained model and a fixed snapshot ensemble, does the constraint direction improve the deployed 76-slot grade-3 ranking? This is the most relevant completed modern-backbone knee evidence. Effects are paired TraLO minus the <b>same seed's</b> zero-step PTO ensemble, not differences between architectures trained on unrelated seeds.")
table(["Backbone", "Seeds", "PTO cc-F1", "TraLO cc-F1", "Paired effect, 95% CI"], [
    ("MobileNetV3-Large", "72", "73.28%", "74.45%", "+1.18 pp [+0.84, +1.51]"),
    ("EfficientNet-B5", "48", "74.31%", "73.44%", "-0.87 pp [-1.52, -0.22]"),
    ("ResNet18 (diagnostic)", "72", "72.61%", "73.72%", "+1.11 pp [+0.78, +1.45]"),
    ("RegNetY-400MF (diagnostic)", "72", "73.28%", "74.21%", "+0.93 pp [+0.57, +1.29]"),
], [47,16,27,28,57])
story.append(knee_effect_plot())
story.append(Spacer(1, 5))
add("MobileNetV3 also gained +0.87 pp in accuracy, +0.93 in macro-F1 and +0.46 in weighted-F1. EfficientNet-B5 lost -0.22, -2.00 and -2.06 points on those same secondaries. ResNet18 and RegNetY had positive constrained F1 but small weighted-F1 losses. Therefore the valid statement is <b>backbone-dependent knee development signal</b>, not universal improvement across models and metrics.")
add("The larger B5 backbone's higher plain PTO score does not rescue its negative TraLO effect. Architecture capacity, training trajectory and the orientation of the constraint step interact. A method can have a good base classifier but a bad side-step. The sealed Chen test was not used to select a backbone and is not part of these figures.")
sub("Transfer beyond knee")
add("A separate global fmow2 MobileNetV3 block (48 seeds, pooled cap 167) yielded only +0.24 percentage points of class-1 cc-F1 versus PTO, 95% CI [-0.09, +0.57], Holm p=0.308, while accuracy fell by 1.37 points [1.12, 1.63] in magnitude. The knee result therefore has not transferred as a robust satellite result. It remains possible that a different dataset, backbone, or constraint policy behaves differently; this block does not justify calling the method generally improved.")
newpage()

title(7, "Country-local fmow2: why the early studies lost")
add("The fmow2 task adds a pooled class-1 upper bound and five country upper bounds. The fair null is a country-aware exact allocator with <b>zero</b> side-step. Historical pooled-only Clipper is not a feasible local comparator: on saved PTO ensembles, it breached at least one country cap in 12/12 seeds at both pooled caps, with mean total country excess 79.42 at cap 167 and 47.17 at cap 83. That audit used no labels.")
table(["Fixed 0.1 MobileNetV3 study", "PTO cc-F1", "Joint TraLO cc-F1", "Joint - PTO, 95% CI"], [
    ("Pooled cap 167, 12 seeds", "0.4961", "0.4095", "-0.0866 [-0.0947, -0.0785]"),
    ("Pooled cap 83, 12 seeds", "0.4052", "0.3356", "-0.0697 [-0.0854, -0.0539]"),
], [60,30,39,46])
add("These are large losses in decimal F1 units: -8.66 and -6.97 percentage points. The 0.1 side-step was active, so the negative result cannot be dismissed as an implementation that never moved. The raw class-1 call count collapsed from roughly 233 to 15-17, while the exact allocator still filled its required slots. The failure was mainly <b>which images were ranked into scarce slots</b>, not a failure to use the budget. Across the separately seeded PHR-direction block at cap 167, joint TraLO moved 239 correct images in but pushed 402 correct PTO selections out. All seeds lost selected true positives to PTO at the fixed dose.")
sub("PHR snapshot comparison is also negative - with an important limit")
table(["Pooled cap, 12 seeds", "PTO", "Joint", "PHR snapshot", "PHR - PTO, 95% CI"], [
    ("167", "0.49260", "0.42167", "0.41471", "-0.07789 [-0.09604, -0.05975]"),
    ("83", "0.39967", "0.33445", "0.33445", "-0.06522 [-0.08113, -0.04930]"),
], [44,23,25,33,50])
add("The PHR direction was active at every saved snapshot and lost too. It was a side-copy direction with dual updates across snapshots, <b>not</b> a persistent full augmented-Lagrangian training run. It does not settle TraLO versus a properly matched full ALM. The historical four-seed ALM study used frozen features and a different update schedule; it is shown later as context only.")
newpage()

title(8, "Calibrated MobileNetV3: latest complete local block")
add("After preserving the 0.1 failure, the study fixed a <b>new</b> label-free boundary policy: accept a much smaller candidate displacement only if soft and hard count probes satisfy the declared safeguards. It was not fitted to development labels. The joint, pooled-only, dose-matched sham, PHR direction and zero-step PTO all share the same supervised MobileNetV3 trajectory and country-aware allocator within each seed. All 12 fresh seeds 6401-6412 exited 0, and the independent scorer passed the final source, data, artifact, quota, dose and FP32 dual-replay checks.")
table(["Cap", "PTO cc-F1", "Joint cc-F1", "Joint - PTO, 95% CI", "Accuracy PTO -> joint"], [
    ("167", "0.491732", "0.495648", "+0.003916 [-0.001938, +0.009771]", "0.541243 -> 0.525154"),
    ("83", "0.394649", "0.397436", "+0.002787 [-0.001818, +0.007392]", "0.537209 -> 0.529289"),
], [17,26,28,66,38])
add("The primary paired p values were 0.169 and 0.210; the six-test Holm value for each was about 0.997. The small positive point estimates are <b>not</b> reliable evidence of an improvement. At cap 167 macro-F1 fell from 0.477342 to 0.463106 and weighted-F1 from 0.558474 to 0.544105. At cap 83 they fell from 0.461807 to 0.454928 and 0.550888 to 0.545200. The registered lead rule failed at both caps, including the secondary-metric guardrail.")
add("At cap 167, pooled-only cc-F1 was 0.491297 and PHR was 0.494778; joint minus pooled-only was +0.004352 [0.002436, 0.006267] in a <b>descriptive, non-registered contrast</b>. At cap 83, pooled-only was 0.396321 and PHR 0.396878; joint minus pooled-only was +0.001115 [-0.001936, 0.004165]. These inspected-development comparisons cannot establish a unique country-local mechanism. A favorable unregistered contrast does not override the failed primary and secondary conditions.")
sub("What the logs add")
add("All 72/72 joint snapshot opportunities applied a step at each cap, with mean radius about 0.01305 at cap 167 and 0.00733 at cap 83. Mean raw calls fell from about 231.4 to 180.4 and 202.6, far less violently than under the old 0.1 dose. The smaller step exchanged 181 total 167-slot selections across seeds (81 correct entries, 72 correct exits) and 63 total 83-slot selections (44 correct entries, 39 exits). Yet all twelve seeds chose stop epoch 1; mean training loss fell 0.9296 to 0.0995 while mean stop loss rose 1.2222 to 2.0836. That divergence is a plausible generalization bottleneck, not proof of a causal explanation for this block's F1 tradeoff.")
newpage()

title(9, "ViT-B/16: complete audited transformer result")
add("A separately pinned pretrained ViT-B/16 study tested the same local boundary question with fresh seeds. The first smoke fixture failed before seed claim, and a subsequent seed-6500 run failed a no-grad versus gradient replay check. Both remain recorded. The v2 release corrected the attention execution path, passed a real-image FP32 memory and gradient preflight, then completed matched seed-6600 step/ref pilots with exact PTO trajectory parity.")
add("The single seed-6600 pilot had suggested +0.005222 cc-F1 at cap 167 and a tie at cap 83. As predeclared, that pilot did not select settings or stop the full block. All twelve fixed seeds 6601-6612 completed once by 1 October 04:12:47 UTC on dsisco02. A corrected, separately released scorer then passed the complete-block gates on those <b>same</b> saved runs. The following are absolute allocated class-1 cc-F1 means and paired joint-minus-PTO intervals; the development countries had previously been viewed.")
table(["Pooled cap", "PTO cc-F1", "Joint cc-F1", "PHR cc-F1", "Joint - PTO, 95% CI"], [
    ("167", "0.479112", "0.478242", "0.481288", "-0.000870 [-0.003640, +0.001900]"),
    ("83", "0.390190", "0.390747", "0.389632", "+0.000557 [-0.005303, +0.006418]"),
], [28,29,31,30,57])
add("Both primary intervals cross zero; the Holm-adjusted joint-versus-PTO p value is 1.0 at each cap. The sham has the same mean cc-F1 as PTO. At cap 167, joint accuracy fell from 0.514595 to 0.511556, macro-F1 from 0.465558 to 0.462468 and weighted-F1 from 0.530650 to 0.527267. At cap 83 the corresponding accuracy pair is 0.511656 to 0.510062. PHR's slightly higher cap-167 cc-F1 (0.481288) comes with worse secondary metrics; its contrast does not pass the six-test Holm family. The registered exploratory-lead condition is false at both caps.")
add("Training logs show 78/78 joint and 78/78 PHR opportunities applied at each cap: inactivity is not the cause. Mean training loss fell from 0.8347 at epoch 1 to 0.1253 at the final epoch, while stopping loss rose from 1.5321 to 2.3975; eight of twelve seeds selected epoch 1. The cap-167 joint allocator replaced an average 11 of 167 slots per seed, adding 4.58 correct items and removing 4.75. At cap 83, 7.33 slots changed, with 5.50 correct entries and 5.42 correct exits. The direction moved the ranking, but useful entries and exits nearly cancelled.")
note("The original scorer failures remain preserved: FP32 PHR dual continuity, then a direct-versus-country pooled reduction, then a scorer/runner release-path assumption. Source-backed scorer-only fixes were tested on both hosts without weakening tolerances, changing the ViT runner or repeating any seed. The final JSON passed all gates; it supports <b>no ViT local-method win</b> on this inspected development cohort.", colors.HexColor("#FFF5E8"))
newpage()

title(10, "Clipper, null, and ALM: what comparison is fair?")
add("'Clipper' can mean different things in these records. The knee `capped_first` method allocates the top constrained-class scores to a fixed number of slots after training. On fmow2, a pooled-only cut is not country-feasible; the fair local PTO/null is the country-aware allocator with zero side-step. A sham is a separate random-dose control, not the same as zero-step PTO. These distinctions matter because merely meeting a hard quota is a property of the allocator, not proof that training learned a better ranking.")
table(["Older frozen-feature study", "Clipper", "Shared null", "TraLO 0.0001", "Inexact PHR-ALM"], [
    ("Knee, 826 dev images; 4 seeds", "39.649%", "41.078%", "42.191%", "33.022%"),
    ("CIFAR-100 2,000-image dev subset; 4 seeds", "52.057%", "51.390%", "51.764%", "50.661%"),
], [68,25,27,29,26])
add("For knee, TraLO minus the shared null was +1.113 percentage points, 95% CI [-0.737, +2.964]; for CIFAR it was +0.374 [-0.331, +1.078]. Both cross zero. ALM minus null was -8.056 [-11.618, -4.494] on knee and -0.729 [-3.081, +1.622] on CIFAR. The frozen ResNet18 feature extractor, four seeds, different pass counts and update schedules prevent this table from deciding which local full-backbone training algorithm is better.")
add("A prospective seven-epoch, 24 GPU-hour ceiling design now specifies one matched MobileNetV3 training comparison: supervised CE/PTO, shared zero-step null, historical Clipper analogue, persistent TraLO at both caps, and persistent PHR-ALM at both caps. It requires independent cost accounting, source/data/gradient checks, immutable release, a gated pilot, and authorization of the distinct scientific question and budget. It has <b>no GPU result yet</b>. The most recent code review still found a cost-accounting edge case; publishing a full-ALM performance number now would be fabrication.")
newpage()

title(11, "What we can say, what we cannot, and the next experiment")
table(["Claim", "Present evidence", "Decision"], [
    ("A better practical knee pipeline exists than the original v3 recipe.", "The plain baseline gained about 4.81 pp unpaired; an eight-cell factorial gives augmentation +5.64 [+4.60, +6.68] pp.", "Supported for the inspected knee development design; attribute mostly to augmentation, not TraLO."),
    ("Global TraLO improves constrained selection.", "MobileNetV3 +1.18 pp and small-backbone replications positive; EfficientNet-B5 -0.87 pp.", "Backbone-dependent lead on knee development, not universal or held-out."),
    ("Local TraLO beats a country-aware post-hoc allocator.", "Fixed 0.1 loses 6.97-8.66 pp; smaller boundary policy +0.28-0.39 pp with intervals crossing zero and accuracy losses.", "Not established on fmow2. The inspected development countries cannot validate another tuned policy."),
    ("TraLO beats full ALM on a modern backbone.", "Only frozen-feature historical results and an active PHR snapshot-direction comparison exist.", "Unmeasured. Finish matched full-training panel before this claim."),
    ("ViT proves the newer method.", "Audited 12-seed joint-PTO cc-F1 is -0.00087/+0.00056 at the two caps; both intervals cross zero, accuracy falls.", "No ViT local-method lead on this inspected development cohort."),
], [54,78,43])
sub("Recommended scientific path")
bullet("Preserve the audited negative and null ViT result alongside MobileNetV3 and the old fixed-dose failure. Do not optimize the boundary policy on these repeatedly viewed fmow2 development countries.")
bullet("Treat MobileNetV3's calibrated local result as a tradeoff, not a winner. The direction may be worth studying mechanistically because dose calibration prevented collapse, but repeatedly viewed fmow2 development labels cannot provide independent confirmation.")
bullet("Complete the matched persistent TraLO-versus-full-ALM/Clipper/null implementation and cost gates before GPU dispatch. Lock the question, data boundary, seven-epoch budget, metrics and interpretation in an immutable protocol.")
bullet("For a paper claim, require a prospective independent cohort or truly untouched domain/backbone under a fixed policy. Keep Chen and reserved fmow2 cohorts sealed until an explicit final evaluation decision; do not recycle them for tuning.")
note("The strongest current practical result may be the <b>training and snapshot pipeline</b> rather than the constraint loss. The strongest method lead is a small global knee effect on MobileNetV3 that reverses on B5. No current evidence supports saying the pooled-plus-country local variant is an improved model overall.")
newpage()

title(12, "Evidence map, provenance and limits")
add("The following paths are the preserved local research record in this workspace. The `experiments/README.md` index groups them without moving or deleting referenced artifacts. Every completed fixed block retains per-seed outputs, runner and scorer release hashes, completion receipts and failures. Both modern-backbone boundary blocks were independently recomputed after scorer-only numerical/provenance corrections; the original failed gates remain on disk.")
for s in [
    "Kassif and Singer, <i>Adaptive resource-constrained neural networks for multi-class medical image classification</i>, Engineering Applications of Artificial Intelligence 182, article 115989 (2026), DOI 10.1016/j.engappai.2026.115989. Public abstract: https://cris.iucc.ac.il/en/publications/adaptive-resource-constrained-neural-networks-for-multi-class-med/",
    "Audited Kassif code repository: https://github.com/YuvalKassif/ConstrainedClassification ; local audit `experiments/claude_yuval_repo_audit_20260927.md`, repository commit 413d96c.",
    "Kassif pipeline preregistration and paired result: `experiments/claude_yuval_pipeline_prereg_20260927.md`, `experiments/claude_yuval_pipeline_result_20260927.md`; recipe factorial `experiments/claude_recipe_factorial_result_20260927.md`.",
    "Global knee step-ensemble: `experiments/claude_stepens_result_20260928.md` and `experiments/claude_stepens_additional_result_20260928.md`.",
    "fmow2 local fixed dose, PHR direction and allocator feasibility: `experiments/fmow_local_fixed_dose_result_20260930.md`, `experiments/fmow_local_alm_direction_result_20260930.md`, `experiments/fmow_global_vs_local_clipper_diagnostic_20260930.md`.",
    "Calibrated MobileNetV3 protocol and result: `experiments/fmow_boundary_mnv3_protocol_20260930.md`, `experiments/fmow_boundary_mnv3_result_20261001.md`. Audited per-seed score: `C:/Users/roeym/.codex/rebuild-audit-20260922/fmow_boundary_full_score_31d50d05_20261001.json` (SHA256 a2211d6e974a7075938ba49a3a80fcbd4bd19a01cc0129943c6d90ce7986c32e). Runner 4334855f; scorer 31d50d05. First failed scorer receipt: `fmow_boundary_full_score_gate_failure_20260930T2226Z.json`.",
    "ViT protocol and audited result: `experiments/fmow_boundary_vit_preflight_20260930.md`, `experiments/fmow_boundary_vit_v2_replay_protocol_20261001.md`, `experiments/fmow_boundary_vit_v2_result_20261001.md`. Runner e7acc6d9; scorer 24a78b8e. Full per-seed JSON `C:/Users/roeym/.codex/rebuild-audit-20260922/fmow_vit_full_score_24a78b8e_20261001.json` (SHA256 b26bb18c112f8ed26a817a0bd159c9b57b75b14339f546396181c7d52ce263d5).",
    "Historical ALM and prospective design: `experiments/alm_two_dataset_result_20260924.md`, `experiments/fmow_full_alm_matched_protocol_draft_20261001.md`.",
]:
    add("<font color='#176A9C'><b>•</b></font> " + s, "Sourcex")
add("Limits: seed intervals quantify training variability conditional on fixed development images; they do not quantify new-hospital/new-country uncertainty. Several reported contrasts are exploratory or descriptive, and inspected fmow2 countries have been viewed many times. Absolute metrics across different tasks and cap definitions are not directly comparable. No sealed test result appears in this report.", "Smallx")

def page_canvas(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h-12*mm, w, 12*mm, fill=1, stroke=0)
    canvas.setFont("ArialTB", 7)
    canvas.setFillColor(colors.white)
    canvas.drawString(17*mm, h-7.5*mm, "TraLO  /  EVIDENCE BRIEFING")
    canvas.drawRightString(w-17*mm, h-7.5*mm, "1 OCT 2026")
    canvas.setStrokeColor(RULE)
    canvas.line(17*mm, 15*mm, w-17*mm, 15*mm)
    canvas.setFont("ArialT", 7.4)
    canvas.setFillColor(INK)
    canvas.drawString(17*mm, 10*mm, "Development evidence; sealed test cohorts not opened")
    canvas.drawRightString(w-17*mm, 10*mm, f"{doc.page}")
    canvas.restoreState()

doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=17*mm, leftMargin=17*mm,
                        topMargin=22*mm, bottomMargin=20*mm, title="TraLO after Kassif: research status",
                        author="TraLO research audit", subject="Method changes, numerical results and evidence limits")
doc.build(story, onFirstPage=page_canvas, onLaterPages=page_canvas)
print(OUT)
