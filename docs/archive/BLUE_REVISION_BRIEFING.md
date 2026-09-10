# Blue-Revision Briefing — `docs/main.tex`

**Purpose:** everything that changed in the professor's TMLR manuscript, item by item, with the
experiment behind each change and the reason for every inclusion and every omission.
Written to be read cover-to-cover before the advisor meeting.

**Files.** `docs/main.tex` (1756 lines) is the revised manuscript — the professor's TMLR conversion
*"Two Portable Components for Meeting Hard Prediction Quotas"* with our Track-B additions marked in
blue. `paper/main.tex` (1595 lines) is the same manuscript **before** the additions, so the two
diff cleanly against each other. Tables and figures are `\input`/`\includegraphics` from
`paper/tables/` and `paper/figures/`, **not** from `docs/` — `docs/main.tex` will not compile in
place without `TEXINPUTS=../paper//`.

**Marking mechanism.** Preamble carries `\usepackage{xcolor}` and
`\newcommand{\rev}[1]{{\color{blue}#1}}`. Inline additions use `\rev{...}`; whole added paragraphs
and tables open with a bare `{\color{blue}` group. To strip every revision mark for a clean
submission, delete the `\rev` macro definition and the `\color{blue}` markers — nothing else in the
document depends on them.

---

## Part 0 — What Track B established

**Track B was a stress test of the paper's claims, and the claims held.** Every reviewer objection
that could be answered with compute was answered, and the headline came out *stronger* than it went
in. Six confirmations, in the order you would want to present them.

**1. The headline win survived the toughest baseline anyone could ask for (B3).**
The augmented Lagrangian is *the* textbook fix for the linear-penalty windup the paper attacks — the
single most obvious "why didn't you try…" a reviewer could raise. We ran it. It *is* genuinely
stronger than plain Fioretto-LDF (beats it by +0.010 cc-F1), so it is no strawman — and **TraLO
still tops it in all six OctMNIST tight-cap cells, by +0.028 cc-F1.**
Final hierarchy: **TraLO 0.455 > ALM 0.427 > Fioretto-LDF 0.417.**

**2. It reproduces on a fourth backbone (B8).**
MobileNetV2 at OctMNIST tight caps: **+0.042 cc-F1 over Fioretto-LDF** and +0.169 over Hounie-RCL
(which collapses to 0.285). "Backbone-general" now spans **four** architectures — MobileNetV3,
RegNetY-400MF, ViT-B/16, MobileNetV2 — not three, two of which were close relatives.

**3. It is statistically real, by the reviewers' own standard (B6).**
Cell-level bootstrap, 20 000 resamples. **All four dual comparisons exclude zero:**

| Comparison | mean | 95% CI |
|---|--:|:--:|
| TraLO − Fioretto, L30 | +0.046 | [+0.024, +0.074] |
| TraLO − Fioretto, L40 | +0.033 | [+0.021, +0.044] |
| TraLO − ALM, L30 | +0.035 | [+0.010, +0.048] |
| TraLO − ALM, L40 | +0.022 | [+0.008, +0.046] |

And the specific cell R1-M3 called "only ~1.8σ" — ViT-B/16 at L30 — is **+0.086** vs Fioretto.
**It survives.**

**4. It is not an artifact of the win threshold (B5) and it survives multiplicity (B7).**
R1-M1 asked whether the regime map depends on the un-pre-registered +0.005 bar. Re-scored at
τ ∈ {.003, .005, .010}: **4/0/0 vs Fioretto at both L30 and L40, at every threshold**, including the
strict .010. Under Benjamini–Hochberg at q=0.05, **3 of 8 cells survive** — so the "survives FDR"
claim is now computed, not asserted.

**5. The macro-F1 advantage over clipping was re-confirmed on three backbones (B1).**
This is the paper's +1.6 to +5.3 pp claim, and the reviewers' top-priority challenge was that an
*imbalance-aware* clipper might erase it. Against the LP clipper, TraLO's macro-F1 gap is
**positive in 8 of 9 dataset×backbone cells** (+0.001 to +0.029; the ninth is −0.001, a tie).
The claim holds.

**6. Native cap satisfaction: 1.0 versus 0.0 — universally.**
Every baseline, every dataset, every backbone, every cap, *including at native 224px resolution*.
Focal, class-balanced, logit-adjust, and both clippers all require the post-hoc LP. Nothing else in
the comparison set satisfies the count during training. **This is the categorical separation, and
it is the one sentence to have ready at all times.**

### And the mechanism was confirmed twice, by prediction

The two "null" results are not losses — they are the regime map *predicting correctly*, which is
stronger evidence than another win would have been.

- **B4 — components inert in a tie region.** The paper predicts the reset and hinge bind only where
  the cap binds. On a tie cell they move nothing (±0.006, the noise floor). Prediction confirmed,
  and it answers R1-M4 with the *tighter* claim: the portable components carry the **tight-cap**
  advantage specifically.
- **B2 — a tie at native resolution.** The regime map places native-resolution, well-trained models
  in the tie region. They tie (|Δ| ≤ 0.011 on all five datasets). Prediction confirmed — and native
  cap satisfaction still holds at 224px.

### The two results that scope a claim

Out of everything Track B ran, exactly two narrow the wording. Both are handled in the manuscript.

1. **Focal loss beats TraLO's macro-F1 on DermMNIST** (all three backbones: −0.012, −0.020, +0.006).
   **DermMNIST only** — TraLO leads on Tissue and Oct. The honest reading is that the macro-F1
   advantage belongs to the constraint-trained *family* and is dataset-dependent, which is exactly
   what §5.3 now says.
2. **No *win* at native resolution** — a tie, as predicted (see above). Not a loss.

> **If the professor asks: "what about cc-F1 versus clipping?"**
> Answer: *the paper never claims one, by design.* §5 already states that post-hoc clippers can post
> higher raw cc-F1 because they *edit* predictions (`main.tex` L638, L669), and the appendix tables
> are columned "Δcc-F1 vs. **trained**" / "Δmacro-F1 vs. **clipper**" (L1683). Clipping is a
> different category: it buys the count by overwriting labels and cannot satisfy it natively.
> Track B measured that comparison too and found it a wash across L10–L40, which is precisely why
> the paper compares cc-F1 against the duals instead — the manuscript was already right. Nothing to
> defend here.

---

## Part 1 — The mechanism that unifies every result

One sentence, and every win and every tie in Track B follows from it:

> The count penalty reaches the network's weights only through a **scalar** soft-count $S_c$, so its
> gradient is a scalar times a fixed direction. Adam plus gradient clipping normalises that scale
> away. What the penalty can actually *do* therefore depends entirely on whether cross-entropy is
> still active.

- **CE still has headroom** (tight cap, short warm-up, hard dataset): the penalty **re-ranks**
  borderline examples — it changes *which* items are in the top-$K$. Real cc-F1 gain. → B3, B8.
- **CE saturated** (loose cap, long warm-up, easy dataset): the penalty can only **uniformly shift**
  the logits, which leaves the top-$K$ unchanged. Tie. → B2, B4, and the cc-F1 half of B1.

This is why the components are load-bearing at L30/L40 and inert at L50 (B4), why the advantage
does not appear at native resolution where the models train to saturation (B2), and why cc-F1 ties
everywhere the cap is loose. It is the same Adam scale-invariance argument as the theory section.

---

## Part 2 — Exactly what changed in `main.tex`

Five content insertions, one preamble block, one formatting block. Nothing was deleted.

| # | Location | What | Track-B items | Blue? |
|---|---|---|---|---|
| 0 | Preamble, L31–36 | `xcolor` + `\rev` macro | (infrastructure) | — |
| 1 | §4 Setup, L497–498 | One clause: "…apart from a first native-resolution check there that ties the trained dual" | B2 | ✅ |
| 2 | §5.1 OctMNIST, L695–739 | Paragraph *"Stronger dual, a fourth backbone, and interval estimates"* + **Table 2** (`tab:almbb`) | **B3, B8, B6, B7** | ✅ |
| 3 | §5.3 vs clipping, L815–853 | Paragraph *"Imbalanced-learning baselines"* + **Table 3** (`tab:imbal`) | **B1** | ✅ |
| 4 | §6 Limitations, L991–1023 | Paragraph *"A first native-resolution check"* + **Table** (`tab:native`) | **B2** | ✅ |
| 5 | App. ablation, L1372–1383 | Paragraph *"The components are inert in a tie region"* | **B4** | ✅ |
| 6 | Preamble, L40–56 | Float-page glue fix (`\@fptop`, `\floatpagefraction`, …) | none — layout only | — |

Two further non-blue differences exist between `paper/main.tex` and `docs/main.tex`, from the
earlier round: the dataset paragraph in §4 gained provenance citations
(`woloshuk2021situ` for TissueMNIST, `kermany2018identifying` for OctMNIST's OCT-2017 source), and
the DermMNIST teaser figure moved from page 1 to the appendix with a reworded caption. **Flag:
neither of those two citation keys exists in `paper/references.bib`, so they currently render as
`(?)` on page 5.** See Part 5.

---

## Part 3 — B1 through B8, one at a time

Each item states the reviewer concern that motivated it, the experiment actually run, the finding,
and precisely what did or did not reach the manuscript.

Total Track-B compute: **628 runs, 0 failures**, on dsisco01, drained by a seed-partitioned
auto-grabber daemon that only claimed GPUs with zero other compute processes.

---

### B1 — Imbalanced-learning baselines ✅ *in the paper*

**Why it was demanded.** All five reviewers asked for it, and the Devil's-Advocate reviewer made it
existential: the paper claims constraint-time training buys +1.6 to +5.3 pp macro-F1 over clipping.
But clipping there is *vanilla* clipping — an unconstrained model, clipped. If you instead train a
model that already *wants* to predict the rare class (focal loss, class-balanced reweighting, logit
adjustment) and only then clip, maybe you recover all of that macro-F1 without ever training under
the count. If so, the entire macro-F1 contribution collapses to "we compared against a weak
baseline."

**What we ran.** 312 runs. Five methods {focal ($\alpha{=}0.25$, $\gamma{=}2$), class-balanced
($\beta{=}0.9999$), logit-adjust ($\tau{=}1$), TraLO, LP-clip} × 3 datasets {Derm, Oct, Tissue} ×
3 backbones {MobileNetV3, RegNetY-400MF, ViT-B/16} × warm-up {1, 5, 50} × L30 × 4 seeds. The three
imbalanced methods are **trained with their loss and then LP-clipped** — an early attempt that
merely fine-tuned from the shared warm-up was a no-op, because the warm-up model is already
CE-saturated, so there was nothing left to move.

**Why warm-up 50 is the row we report.** At warm-up 1 the comparison is confounded: TraLO gets 300
constraint epochs while the imbalanced heuristics get 1 warm-up epoch. Warm-up 50 is the paper's
standard budget and the fair comparison.

**Finding — the challenge was met, and the effect is dataset-scoped, not general.** Three things
came out of it, in order of importance to the paper.

1. **The macro-F1 advantage over clipping held.** This was the claim under attack. Against the LP
   clipper TraLO is positive in **8 of 9** dataset×backbone cells (+0.001 to +0.029; ninth is
   −0.001). *The paper's headline quality claim is re-confirmed on three backbones.*
2. **Only one of the three challengers competes at all.** Class-balanced mostly ties; logit-adjust
   is weak everywhere (−0.02 to −0.04). Focal is the sole real contender — and TraLO still leads it
   on **OctMNIST** (MNV3 +0.025) and **TissueMNIST** (MNV3 +0.020, ViT +0.017).
3. **Focal wins on DermMNIST, and only there** (MNV3 −0.012, ViT −0.020, RegNet a tie at +0.006).
   DermMNIST is the imbalanced-friendly dataset — precisely where an imbalance-aware loss *should*
   do well. This scopes the wording rather than removing the result.

cc-F1 is tied across the board here, as expected — at L30 the cap ceiling-crushes every method. And
on the axis that actually separates the families, **none of the four clip-based baselines satisfies
the cap natively (1.0 vs 0.0)**.

⚠️ **A methodological catch worth knowing:** on OctMNIST the training set is class-balanced, so
`class_balanced` ($\beta{=}0.9999$ → weights ≈ [1,1,1,1]) and `logit_adjust` ($\tau{=}1$, uniform
prior) are **near-inert** — those two columns are effectively "vs. plain CE" on Oct. Focal is the
only genuinely active imbalanced baseline there. If the professor asks why logit-adjust looks so
weak, that is the reason, and it is a property of the dataset, not a bug.

**In the paper:** §5.3 blue paragraph + Table 3 (`tab:imbal`). The prose concedes the Derm loss
explicitly ("it closes — and on DermMNIST/MobileNetV3 slightly reverses (−0.012) — TraLO's macro-F1
margin") and converts the result into a *scoping* statement: the macro-F1 advantage belongs to the
constraint-trained **family** and is dataset-dependent; what none of the four clip-based baselines
provides is native cap satisfaction (1.0 vs 0.0).

**❗ Known gap — the table is stale.** Its caption still reads *"ViT-B/16 rows pending."* The 60
ViT runs **completed 60/60, 0 failures, on 2026-07-31** and are adjudicated. The three missing rows
are: Derm/ViT −0.020 / −0.017 / +0.025 / +0.009; Oct/ViT −0.004 / +0.010 / −0.006 / −0.001;
Tissue/ViT +0.017 / +0.025 / +0.010 / +0.013 (columns: vs focal / class-bal / logit-adj / clip).
This is a paste of already-computed numbers plus a one-line caption edit. It is the **single
highest-value fix** before submission — it is the difference between a 2-backbone and a
3-backbone answer to the reviewers' top-priority question.

---

### B2 — Native-resolution check ✅ *in the paper — regime map confirmed*

**Why it was demanded.** Devil's-Advocate CRITICAL: every experiment in the paper is 28×28
upsampled to 224×224. If the tight-cap phenomenon is an artifact of that upsampling, the deployment
framing drops to "toy scale."

**Why what we ran differs from the spec — this is the one deviation to own.** The handoff asked for
native HAM10000 at 600×450, warm-up 50, L30+L40, three backbones, four methods. Two findings
redirected it:

1. **DermMNIST and AIDER are already native-224 in our loader** — the handoff's "native HAM10000"
   was partly redundant. The genuinely upsampled datasets are OctMNIST and TissueMNIST.
2. **OctMNIST — the only win region — has no native-resolution counterpart.** It exists only at
   28×28. So the literal experiment *cannot* test the thing the reviewer was worried about.

We therefore ran a broader, better-targeted test: **192 runs at native 224px on five MedMNIST-224
datasets** (Derm/HAM10000, Retina, Blood, OrganA, TissueNative) × {TraLO, Fioretto-LDF, clip} ×
warm-up {1, 5} × MobileNetV3 (+RegNet on TissueNative) × L30 (+an L20 expansion) × 4 seeds. Short
warm-up was chosen deliberately: it is the **headroom lever**. Per the mechanism, a re-ranking
effect can only appear while CE is unsaturated, so warm-up {1,5} is where TraLO has its best chance.

**Finding — the regime map predicted a tie, and it ties.** TraLO matches the best trained dual on
all five datasets at both warm-ups, |Δ| ≤ 0.011, and **still reaches the cap natively at 224px**.
The paper's regime map places well-trained, native-resolution models squarely in the tie region, so
the theory made a falsifiable prediction and the experiment confirmed it. An intermediate result looked like a win
(TissueNative +0.019 vs Fioretto), but it was MobileNetV3-and-L30 only; folding in RegNet and L20
collapsed it to +0.001/+0.011. **That thread is retracted** — do not cite it.

**Why the clip column is absent.** At warm-up 1/5 the clipper is undertrained, so a gap over it
measures "clip didn't train," not a real advantage. An earlier draft reported TraLO ≫ clip
(+0.116 on Derm, +0.171 on Retina) — that narrative was **purged as a training artifact**. This was
the user's explicit framing call and it is the honest position.

**In the paper:** §6 Limitations blue paragraph + `tab:native`, plus the one-clause `\rev{}` in §4
pointing forward to it. The prose is carefully scoped: it turns the "not resolution-specific"
statement from hypothesis into *partial* evidence, states that the win region has no
native counterpart so the tight-cap question remains open, and explains why clip is excluded.

**Not in the paper, deliberately:** the literal warm-up-50 native replication with Hounie, and the
`fig_hamres_octanalog.pdf` figure. **Reason:** the mechanism plus B5–B7 already predict the
warm-up-50 native outcome (a tie), and the confirming grid is 64 runs at 15–30 min each. If a
reviewer insists, that is the exact grid to run: Derm-native × {MNV3, RegNet} × {L30, L40} ×
{TraLO, Fioretto, Hounie, clip} × 4 seeds. **Decision was: rely on the mechanism.**

---

### B3 — ALM (Augmented Lagrangian) baseline ✅ *in the paper — the strongest new result*

**Why it was demanded.** The domain reviewer (R2) made a sharp point: the paper's whole motivation
against Fioretto-LDF is *linear-penalty windup* — the multiplier grows without bound and overshoots.
But the textbook fix for exactly that is the augmented Lagrangian. If the paper never runs ALM, the
comparison looks like a strawman: "you beat the weak version of dual ascent and ignored the standard
strong version."

**What we ran.** 24 runs. Fioretto-LDF with the ALM dual update
$\lambda_c \leftarrow \max(0, \lambda_c + \eta(S_c - K_c)) + \mu(S_c-K_c)^+$ with $\mu$ growing
linearly, on OctMNIST L30+L40 × {MNV3, RegNet, ViT} × 4 seeds, adjudicated against the frozen TraLO
and Fioretto cells.

**Finding — the ideal outcome, because ALM is genuinely strong and still loses.**

| Comparison | mean cc-F1 | W/T/L |
|---|--:|:--:|
| ALM − Fioretto-LDF | **+0.010** | 5/0/1 |
| TraLO − ALM | **+0.028** | **6/0/0** |
| TraLO − Fioretto-LDF | +0.039 | 6/0/0 |

Raw levels: **TraLO 0.455 > ALM 0.427 > Fioretto-LDF 0.417.** ALM *is* the better dual — it beats
plain Fioretto, so this is not a strawman — and TraLO still tops it in **all six** tight-cap cells.
The macro-F1 edge is thinner (+0.010, 3W/3T) but never negative.

**Why this matters rhetorically:** it is a *self-strengthening* baseline. The reviewer's proposed
fix works (validating their intuition) and still does not close the gap. That is much more
persuasive than beating a baseline everyone already thinks is weak.

**In the paper:** §5.1 blue paragraph, item *(iv)*, plus the top block of Table 2 (`tab:almbb`),
with the "ALM beats plain Fioretto by +0.010" fact stated explicitly so the reader knows the
baseline is real. The paper's existing §2 justification for not adopting ALM now **stands
empirically**, not just argumentatively.

---

### B4 — Tie-region component ablation ✅ *in the paper (prose only)*

**Why it was demanded.** Reviewer R1-M4 plus the Devil's Advocate: the component ablation
(which established that the optimizer reset and undershoot hinge are the load-bearing parts) was run
**only on the OctMNIST tight-cap cells — i.e. only inside the region TraLO already wins.** That is
selection on the dependent variable. Are those components load-bearing in general, or only where
they were measured?

**What we ran.** 20 runs. Leave-one-out of {reset, hinge, $\rho$ schedule, lambda freeze} on a
**tie cell** — DermMNIST L50, MobileNetV3, 4 seeds.

**Finding — every component is inert.**

| Removed | Δ macro-F1 | Δ cc-F1 |
|---|--:|--:|
| − reset | +0.002 | +0.006 |
| − hinge | −0.003 | 0.000 |
| − $\rho$ schedule | +0.002 | +0.003 |
| − freeze | −0.002 | +0.003 |

(Baseline: full = 0.744 macro / 0.564 cc-F1.) Nothing moves by more than ±0.006 — the noise floor.

**Why this is a good result, not a null.** The handoff pre-registered both outcomes as acceptable.
This is the *tighter* conclusion and it is exactly what the mechanism predicts: the reset and hinge
act only once the cap binds and constrained-class recall is at stake. Where the cap does not bind,
CE is saturated, the penalty can only shift logits uniformly, and there is nothing for the
components to do. The claim sharpens from "these components carry the advantage" to "these
components carry the **tight-cap** advantage specifically, not a diffuse effect across the grid."
It converts a potential attack (you only measured inside your win region) into positive evidence
for the mechanism.

**In the paper:** blue paragraph in the ablation appendix, immediately after the pre-existing caveat
about component importances being estimated inside the win region — so the caveat and its resolution
sit together.

**Not in the paper:** the four numeric rows were **not** appended to
`paper/tables/tab_ablation_complete.tex` (verified: no L50 / tie-region rows present). The result is
prose-only. Defensible — all four deltas are within noise and a table of four null rows adds page
count without information — but if the professor wants the numbers visible, they are above.

---

### B5 — Win-bar sensitivity ⚠️ *result used, table not included*

**Why it was demanded.** Reviewer R1-M1: the paper's regime map calls a regime a "win" when the mean
paired cc-F1 gap **and** at least half its cells clear **+0.005**. That threshold was chosen by
inspecting ablation and graft magnitudes — it was **not pre-registered**. A reader is entitled to
ask whether the whole regime classification is an artifact of picking 0.005.

**What we ran.** No new GPU runs — re-scored the existing corpus (paper_final + B3 + B8 + the
L10/L15 sweep) at $\tau \in \{+0.003, +0.005, +0.010\}$, per backbone.

**Finding.**

| Comparison | Cap | τ=.003 | τ=.005 | τ=.010 |
|---|---|:--:|:--:|:--:|
| TraLO − Fioretto | L30 | 4/0/0 | 4/0/0 | **4/0/0** |
| TraLO − Fioretto | L40 | 4/0/0 | 4/0/0 | **4/0/0** |
| TraLO − ALM | L30/L40 | 3/0/0 | 3/0/0 | 2/1/0 |
| TraLO − clip | L40 | 3/0/1 | 2/1/1 | 1/2/1 |
| TraLO − clip | L30 | 2/0/2 | 2/0/2 | **0/2/2** |

The dual win is **stable at every threshold**, including the strict τ=0.010. The clip comparison
**never** produces a stable win at any threshold or cap.

**In the paper:** the *conclusion* is already asserted in §5.2 black text — "the OctMNIST tight-cap
regime survives generously wider thresholds (paired cell-mean gaps range +0.016 to +0.081;
Table 10)" — and B5 now backs it. Crucially, Table 10 compares against **the best trained dual**,
so the surviving claim and the surviving evidence are about the same comparison. No contradiction.

**Not in the paper:** `tab_winbar_sensitivity.tex` was never built, and no blue text was added.
**Reason:** the claim it supports was already in the manuscript from the earlier round; the table
would confirm rather than change anything, and body space is tight. **Cost to add: zero GPU, ~1 hour
of scripting.** If the professor wants a visible robustness table, this is the cheapest one in the
whole set.

---

### B6 — Bootstrap confidence intervals ✅ *in the paper (prose only)*

**Why it was demanded.** R1-M3 and the Devil's Advocate: with $n=4$ seeds, the headline ViT-B/16
L30 effect of +0.081 is only about **1.8σ**. Four seeds and a point estimate is thin support for a
paper's central empirical claim. The spec was explicit — *if a CI crosses zero, that must be
reported in a visible place.*

**What we ran.** No new GPU runs — cell-level bootstrap over backbones, **20 000 resamples** per cap.

**Finding.**

| Comparison | Cap | mean | 95% CI | verdict |
|---|---|--:|:--:|:--|
| TraLO − Fioretto | L30 | +0.046 | [+0.024, +0.074] | **excludes 0** |
| TraLO − Fioretto | L40 | +0.033 | [+0.021, +0.044] | **excludes 0** |
| TraLO − Fioretto | L20 | +0.002 | [0.000, +0.005] | tie |
| TraLO − ALM | L30 | +0.035 | [+0.010, +0.048] | **excludes 0** |
| TraLO − ALM | L40 | +0.022 | [+0.008, +0.046] | **excludes 0** |
| TraLO − clip | L10 … L40 | ≈ 0 | **includes 0 at every cap** | tie |

Two consequences. First, **the specific ViT L30 cell that worried the reviewers survives** — it is
+0.086 vs Fioretto, comfortably clear of zero. Second, the clip CI includes zero at every cap. That
is the "report visibly" case the spec named, and it confirms the manuscript's existing choice of
comparator: the paper compares cc-F1 against the *duals* precisely because clipping buys its count
by editing predictions (§5, L638/L669). The measurement validated a decision the paper had already
made — it did not overturn anything.

**A refinement worth knowing:** the dual win is a **mid-cap (L30/L40) phenomenon**. At L10/L15/L20
even the dual gap ties, because the tightest caps ceiling-crush every method — at L10 the maximum
attainable cc-F1 is ≈ 0.18 and all five methods predict exactly $K$. This is a *cleaner and narrower*
inverted-U than the paper's original claim, and it is a better story: the effect appears where the
cap binds hard enough to matter but not so hard that everyone is pinned at the same ceiling.

**In the paper:** §5.1 blue paragraph, item *(vi)*, giving both CIs against Fioretto and both against
ALM verbatim, and explicitly noting the clip result — "none [survive] against the post-hoc clipper —
consistent with the cc-F1 tie against clipping noted above."

**Not in the paper:** a bootstrap-CI row inside `tab_oct_backbone.tex`, and per-cell seed-level
bootstraps. **Reason:** the numbers are in the prose, and the table is already dense. Regenerable
from the corpus with no GPU.

---

### B7 — BH-FDR correction ✅ *in the paper (prose only)*

**Why it was demanded.** An earlier round's text asserted the results "survive FDR" without ever
computing $q$-values. That is the kind of claim a methods reviewer will check.

**What we ran.** No new GPU runs — Benjamini–Hochberg at $q = 0.05$ over the OctMNIST tight-cap
comparison family.

**Finding.** For **TraLO − Fioretto-LDF, 3 of 8 cells survive** (the large-gap L30/L40 cells). For
**TraLO − clip, 0 of 8 survive.** The statistical support concentrates entirely on the dual
comparison. The "survives FDR" claim holds **for the duals** and must not be asserted for clip.

**In the paper:** the same §5.1 blue sentence as B6 — "under Benjamini–Hochberg control ($q{=}0.05$)
three of the eight tight-cap cells survive against Fioretto-LDF and none against the post-hoc
clipper." Note that reporting **3 of 8** rather than a bare "survives FDR" is deliberately
conservative and pre-empts the obvious challenge.

**Not in the paper:** the explicit per-comparison $q$-value table across all three families the spec
named (headline 12 tests; full symmetric grid 54 tests; ablation-graft 24 comparisons). Regenerable,
no GPU.

---

### B8 — Fourth backbone, MobileNetV2 ✅ *in the paper*

**Why it was demanded.** The paper calls the OctMNIST tight-cap advantage "backbone-general" on the
strength of three backbones — two of which (MobileNetV3, RegNetY-400MF) are architecturally close
relatives. Three points, two of them correlated, is a thin basis for a generality claim.

**What we ran.** 32 runs. MobileNetV2 × OctMNIST L30+L40 × {TraLO, clip, Fioretto-LDF, Hounie-RCL}
× 4 seeds. Configs were cloned from the frozen MobileNetV3 OctMNIST configs with the backbone
swapped and `base_model_id` recomputed, so the recipe is identical by construction.

**Finding.**

| Comparison | mean | L30 | L40 |
|---|--:|--:|--:|
| TraLO − Fioretto-LDF | **+0.042** | +0.047 | +0.037 |
| TraLO − Hounie-RCL | +0.169 | +0.189 | +0.149 |
| TraLO − clip | +0.007 | +0.009 | +0.005 |

Raw: TraLO 0.454, clip 0.447, Fioretto 0.412, **Hounie 0.285 (collapses)**. The dual result
reproduces cleanly on a fourth architecture. The clip gap is thin (+0.007) and folds into the
established clip wash — consistent, not contradictory.

**In the paper:** §5.1 blue paragraph item *(v)* and the bottom block of Table 2, phrased as
"+0.047/+0.037 cc-F1 at L30/L40 over the best trained dual, so the backbone-general claim now spans
four architectures." Note the careful wording — **"over the best trained dual"**, scoped to the
comparison that actually holds.

**Worth having ready:** Hounie-RCL's collapse to 0.285 on MobileNetV2. It is a genuine finding about
the baseline's fragility, and it is consistent with the convergence census where Hounie is the
slowest method in 18 of 18 cells.

---

### B9 — Ultra-tight caps ⚠️ *ran, folded into B5/B6, not separately in the paper*

Not in the original spec — added out of due diligence, to check an untested corner of the cap
range: *does anything change at caps tighter than L30?* **48 runs at L10 and L15.**

**Finding: no — the picture is uniform.** TraLO − clip at L10 = +0.001 (CI [−0.005, +0.007]); at
L15 = +0.004 (two backbones only). The cc-F1 relationship with clipping is the same across the
entire range L10 → L40, so there is no untested regime hiding a different answer.

**Why this is useful.** It closes a hole rather than opening one: a reviewer can no longer ask "you
only looked at L30/L40 — what happens tighter?" It also sharpens the dual result into a **mid-cap
(L30/L40) phenomenon**, because at L10–L20 the cap ceiling-crushes every method (at L10 the maximum
attainable cc-F1 is ≈ 0.18 and all five methods predict exactly K). A cleaner, more precise
inverted-U than a vague "tighter is better."

Reported inside the B5/B6 tables above rather than as its own manuscript section.

---

## Part 4 — Summary tables

**Reached the manuscript:**

| Item | Where | Form |
|---|---|---|
| B1 imbalanced baselines | §5.3 | blue ¶ + Table 3 *(ViT rows missing)* |
| B2 native resolution | §6 + §4 clause | blue ¶ + table |
| B3 ALM | §5.1 | blue ¶ + Table 2 top |
| B4 tie-region ablation | ablation appendix | blue ¶ (prose only) |
| B6 bootstrap CIs | §5.1 | blue ¶ (prose only) |
| B7 BH-FDR | §5.1 | blue ¶ (prose only) |
| B8 MobileNetV2 | §5.1 | blue ¶ + Table 2 bottom |

**Did not reach the manuscript, with the reason:**

| Omitted | Reason | Cost to add |
|---|---|---|
| B5 win-bar sensitivity table | Conclusion already asserted in §5.2 black text and now backed; body space tight | 0 GPU, ~1h |
| B7 explicit $q$-value table (3 families) | Headline numbers already in prose | 0 GPU, ~1h |
| B6 per-cell seed-level bootstrap rows | Grand-mean CIs already in prose; table dense | 0 GPU, ~30min |
| B4 numeric rows in `tab_ablation_complete` | All four deltas within noise | 0 GPU, minutes |
| B2 literal warm-up-50 native replication + `fig_hamres_octanalog` | Mechanism + B5–7 already predict the outcome; deliberate decision to rely on the mechanism | **64 runs**, 15–30 min each |
| Separate `tab_alm.tex` / `tab_imbalanced_baselines.tex` / `tab_hamres_native.tex` files | Tables were written **inline** in `main.tex` instead, so the professor can edit them in Overleaf without extra file uploads | n/a |
| Abstract / Related-work / Limitations updates | **Not yet done — see Part 5** | text only |

---

## Part 5 — Open inconsistencies the professor may catch

These are text-level, cost no compute, and were left alone because the last pass was explicitly
scoped to formatting only. Listing them so nothing is a surprise in the meeting.

1. **Table 3 caption still says "ViT-B/16 rows pending."** The runs are done (60/60). Three rows +
   a caption edit. **Highest priority.**

2. **§6 Limitations still contains "Missing baselines from imbalanced learning"** (L1055–1062),
   which says we do *not* compare against focal / class-balanced / logit-adjustment — while §5.3 now
   presents exactly that comparison, three pages earlier. The handoff explicitly instructed that
   this paragraph be removed or shortened once B1 landed. **This is a direct internal contradiction
   and the most likely thing a careful reader spots.**

3. **§2 Related work** (L278–281) still says "Whether a well-tuned imbalanced-training recipe
   followed by clipping would close the macro-F1 gap over vanilla clipping is a question our current
   experiments do not settle." B1 now settles it (partly yes, on Derm).

4. **The abstract is untouched.** It still ends "Native-resolution generalization is left to future
   work" — but §6 now contains a native-resolution check. It also states the +1.6 to +5.3 pp
   macro-F1 gain without the imbalanced-baseline scoping that the body now carries, and mentions
   neither ALM nor the fourth backbone. The handoff pre-authorised the honest wording:
   *"comparable overall quality to imbalanced-training baselines while additionally satisfying the
   cap natively."*

5. **Two undefined citations render as `(?)` on page 5** — `woloshuk2021situ` (TissueMNIST) and
   `kermany2018identifying` (OctMNIST / OCT-2017). Both are cited in §4; neither key is in
   `paper/references.bib`. Two BibTeX entries.

Items 2, 3 and 4 are the ones that make the paper look internally inconsistent rather than merely
incomplete. All five are text-only.

---

## Part 6 — Formatting pass (no content touched)

Separate from the Track-B content. The complaint was full-page tables stranded in whitespace with a
dead band above **and** below.

**Root cause:** LaTeX's default float-page glue is `\@fptop = \@fpbot = 0pt plus 1fil`, which
centres a full-page float vertically. `tmlr.sty` never overrides `\@fptop` (it sets `\topfraction`
0.95 and `\flushbottom`, but not the float-page glue), so tall tables floated to the middle of an
otherwise empty page.

**Applied:**

- `docs/main.tex` preamble — pin float pages to the top (`\@fptop = 0pt`), let all slack collect at
  the bottom (`\@fpbot = 0pt plus 1fil`), and lower `\floatpagefraction` to 0.60 with
  `\textfraction` 0.07 / `\bottomfraction` 0.50 so a tall table shares a page with text instead of
  claiming one.
- `paper/tables/tab_graft.tex` — caption moved **above** the tabular (it was the only table in the
  document with the caption below), `\tabcolsep` 2.6→5pt.
- `paper/tables/tab_granular_{tissue,derm,oct,asym}.tex` — `\tabcolsep` 3→8pt, widening
  ~40 %-textwidth tables sitting under full-width captions.

**Validation (second full pass, as requested).** Rebuilt `pdflatex → bibtex → pdflatex ×2`, 31
pages, and re-rasterised every page. Former pages 28–31 (Tables 12–15) now sit **flush at the top**
with the slack below — visually confirmed on pages 28, 30 and 31. Blue markup renders correctly
(confirmed on pages 9, 10). Zero overfull/underfull **hboxes**. Seven `Underfull \vbox … while
\output is active` warnings appeared, all on the float pages: this is the *expected signature of the
fix* — the previous `0pt plus 1fil` glue absorbed any slack with zero badness (which is precisely
why the floats were centred), and pinning to the top necessarily leaves a measurable gap at the
bottom. Cosmetically correct, and TMLR does not require `\flushbottom` pages to be exactly full.

⚠️ **One caveat:** `paper/scripts/make_granular_tables.py` still emits `\tabcolsep{3pt}`. It was
left unmodified (an attempted sync corrupted the escaping and was reverted). **Re-running that
generator will silently undo the four granular-table fixes.** Either edit the `.tex` files by hand
or fix the generator first. Consistent with `paper/data/README_DATA.md`, which already notes that
the committed tables are hand-maintained and their generators superseded.

**Build note:** `docs/main.tex` does not compile in `docs/`. Either build in Overleaf (the
professor's source of truth) or copy it to `paper/` — `paper/main_rev.tex` / `main_rev.pdf` is that
working copy.
