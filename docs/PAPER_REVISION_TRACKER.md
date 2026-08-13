# Paper revision tracker — professor's round 2

**Opened 2026-08-13.** This file is the single source of truth for the open work.
Update it as items close; do not open a second tracking document.

Manuscript under edit: **`docs/main.tex`** (the blue revision).
**`paper/main.tex` is the professor's original and must never be edited.**

---

## Status at a glance — updated 2026-08-13

| # | Item | Status |
|---|---|---|
| 1 | ALM across the full grid | **running** — 300 in flight, 84 MobileNetV2 chained |
| 2 | Rebuild Tables 1 and 2 | **blocked on item 1** — nothing else stops it |
| 3 | Figures cover the tables' scope | audit + pipeline repair **done**; Figs 1 & 5 regenerate when ALM lands |
| 4 | Textual coherence | narrative pass **done**; scope sweep + repetition open |
| 5 | Naming consistency | backbone names **done** (12 sites); table headers deferred to item 2, other categories open |
| 6 | Move Tables 3 and 4 | **done** |
| 7 | Figure 4 legend | **done** — all six figures verified by eye |
| 8 | Remove Figure 6, expand dataset appendix | **done** |
| 9 | Are Tables 12–14 earning their place? | **decided: keep**; regenerate when ALM lands |
| 10 | Remove Table 15 | **done** |

Five of ten closed. Of the five open, **three are waiting on ALM** (items 1, 2, 3)
and two are unblocked work: the remainder of the naming sweep and the
scope/repetition pass in item 4.

**One decision is waiting on the professor:** whether the headline margin moves to
the ALM-inclusive number (see item 4). Both are currently stated and correctly
scoped, so nothing is blocked on the answer.

---

## Item 11 — Blind review, round 1: **58/100** (2026-08-13)

Run via the `research-reviewer` skill on `paper/main_clean.tex` (revisions accepted,
marks stripped) by a fresh no-context agent. It recomputed the paper's numbers from
the released corpus and **reproduced Tables 2/3/5/7 and the convergence medians
exactly** — so where it says the data contradict the manuscript, that is worth taking
seriously. I re-verified the three heaviest deductions myself before recording them.

### VERIFIED — must fix

**1. "Never invokes the editing step" is false for 79% of TraLO's own runs.** (−4)

Checked directly against `corpus_final.csv`:

| | |
|---|---|
| TraLO runs in `paper_final` | 324 |
| reporting `sat = 1` | **324** |
| …that *also* record ≥1 flip | **255 (78.7%)**, mean 7.21 flips |
| genuinely edit-free (`sat=1` **and** 0 flips) | **69 (21.3%)** |

Root cause, and it is definitional: **`sat` certifies the GLOBAL cap only**, while the
post-hoc LP also enforces the **per-group** caps — and the joint local+global structure
is what §2 claims as the paper's novelty. So the certifying metric drops the half of
the constraint that is the differentiator.

Affected claims: the abstract ("without editing predictions after training"), App. D
("never invokes the editing step"), §8 ("the count is met during training rather than
by editing predictions afterward").

**The weaker claim is true and verified, and is what the paper should say:** at the
binding caps TraLO averages **1.49** flips vs Fioretto-LDF 5.39, Hounie-RCL 7.03,
TraLO-bounded 5.10, and 102.28 for both clippers. An order-of-magnitude reduction in
editing is a strong, defensible claim. Edit-freeness is not.

**2. The ALM grid contains a loss region.** (−6) Verified on the current snapshot
(134 ALM runs complete, 31 cells at full 4 seeds):

| region | cells | mean Δcc-F1 (TraLO − ALM) |
|---|---|---|
| the band the paper reports (OctMNIST tight) | 6 | **+0.028** |
| every other full cell | 25 | −0.0005 |
| OctMNIST × MobileNetV3, L50–L90 | 5 | **−0.025** |

Per-cap: L50 −0.019, L60 −0.010, L70 −0.030, L80 −0.033, L90 −0.034.

**Context the reviewer could not have:** the ALM expansion is still running, so the
paper reports the 6 cells that existed (the B3 probe), not a selection from 31. But
the finding stands as a forecast — **when the grid lands, there is a loss region and
it must be reported**, not filtered out.

**This does not damage the claim; it fits it.** The loss is at *loose* caps, and the
paper's whole thesis is that the advantage is regime-specific. The honest sentence —
"TraLO leads ALM where the cap binds hard; ALM leads at loose caps on one
dataset×backbone" — *is* the regime map. Filtering to the winning band would be the
thing that damages it.

### Reviewer finding that is a decision, not a defect

**The `+KL` ablation row** (−3): `ablation_complete.csv` has five arms, the table
prints four. The omitted one is +KL (Δ −0.010, winrate 0.21, p 0.04) — i.e. adding a
KL anchor *helps* on these cells. KL material was deliberately removed this round.
The reviewer's point is narrower and fair: §1 and §8 advertise that *every* tie and
negative result is released. Either restore the row with the reason it was retired, or
soften that advertisement.

### FALSE POSITIVE — caused by my own tooling, not the paper

The reviewer deducted (−3) for three structurally broken captions rendering as body
text inside their floats (visible on p. 12). **The manuscript is correct**; the defect
was in `strip_revision_marks.py`. It searched for the literal `{\color{blue}` and
matched the *caption's own opening brace* in `\caption{\color{blue}...}`, deleting
both braces and yielding `\caption\textbf{...}` — which **compiles with zero errors**
while typesetting the caption as body text. Fixed by deleting only the `\color{blue}`
switch and never touching braces. Lesson: a clean build is not a correct build.

### Not yet verified (recorded, not acted on)

Notation collisions ($\lambda_g$ vs $K_g$; $c$ as class and cap index) · the ALM naming
question (reviewer argues it is a PI controller, not an augmented Lagrangian, since no
quadratic term enters the primal) · Fig. 3 panels (a)/(b) inconsistent by ~10× ·
`references.bib` renders two titles blue · hyperref link borders not suppressed ·
Table 1 bolds differences below its own stated noise · "all four backbones" where
Table 1 has three · Hounie slowest in 17 not 18 census cells.

- [ ] Fix the two verified claims (1 and 2)
- [ ] Decide the +KL question
- [ ] Verify and triage the unverified list
- [ ] Re-run the reviewer on a fresh agent after fixes

---

## Ground rules (apply to every item below)

1. **Mark every change.** New text goes in blue (`\rev{...}` or `{\color{blue}...}`).
   Superseded text is struck in red with `\del{...}` — it is *not* deleted from the
   source. A change is never a silent overwrite.
2. **Change what is already there, do not just append.** If a new result affects a
   sentence in the intro, the abstract, a caption, or the limitations, that sentence
   gets edited too. Bolting a new paragraph onto the end of a section is the failure
   mode we are fixing, not a way to fix it.
3. **The paper must read as one paper.** Every insertion has to connect to what
   precedes and follows it. A reader who does not know which reviewer asked what
   should not be able to tell where the revision seams are.
4. **Scope words must match the evidence.** Once a 4th backbone and a 7th method
   land, every "all three backbones" / "the full grid" / "both baselines" in the
   text becomes wrong. These are listed in item 4 and must be swept, not spot-fixed.
5. **One name per thing, everywhere.** See item 5. Abbreviated forms are allowed in
   table headers only, and only if the full form is defined nearby.
6. **Shared files are dangerous.** `paper/tables/` is `\input` by *both* manuscripts.
   Anything that needs revision marks goes in `paper/tables_rev/` and `docs/main.tex`
   is repointed. Never put a `\del` into `paper/tables/`.
7. **Verify rendering, not source.** After any figure or table change, rebuild and
   *look at the rendered page* before calling it done.
8. **A result never changes in one place.** Any time an experimental number moves
   — a rerun, a new arm, a corrected metric, a campaign finishing — it must be
   propagated through the whole chain in one pass, ending in the prose. This is a
   standing rule for the life of the paper, not a task to tick off once. The
   procedure and the wiring diagram are in **`paper/data/PROVENANCE.md`**; the
   short form is below.

### The result-change protocol (run it every time, in order)

```
build_experiment_manifest.py   (server)   ->  manifest/experiments.csv
coverage_report.py                        ->  gaps must be zero or accepted
build_corpus.py --verify                  ->  MUST PASS before overwriting
build_corpus.py                           ->  corpus_final.csv
make_*.py                                 ->  every float that reads the corpus
re-apply the known hand-edits             ->  PROVENANCE.md lists them
propagate into the prose                  ->  the step that gets forgotten
rebuild + LOOK at the changed pages
grep \pending  in docs/main.tex           ->  none may survive submission
```

**Why the prose step needs its own discipline.** Everything above it is
reproducible; the prose is hand-written, so a regenerated table can silently
disagree with a sentence three sections away and the build will still report
zero errors. For each changed number, sweep four places — the float, the float's
**caption** (which lives in the generator, not the `.tex`), the body prose, and
the **abstract / §1 contributions / Conclusion**, which restate headline numbers
and are the most-often-missed. Then check the claims that depend on the number
without containing it: counts ("23 of the 27 cells"), superlatives ("the best
constraint-trained baseline"), scope words ("all three backbones"), and
significance arithmetic (sign tests, BH family size, bootstrap CIs).

This is not hypothetical. When Table 5 gained the isolated-hinge row on
2026-08-13, the table was right and *two* sentences quoting the old row were
wrong — one in §6 and its twin in Appendix B. Both survived a clean build.

---

## Current figure / table numbering (as of 2026-08-13)

| # | Figure | Source | # | Table | Source |
|---|---|---|---|---|---|
| 1 | `fig_octmnist` | `make_octmnist_fig.py` | 1 | headline grid | `tables/tab_ccf1.tex` |
| 2 | `fig_convergence` | `make_convergence_fig.py` | 2 | ALM + MNetV2 patch | inline, `tab:almbb` |
| 3 | `fig_loss_shape` | `make_loss_shape_fig.py` | 3 | imbalanced baselines | inline, `tab:imbal` |
| 4 | `fig_mechanism` | `make_figs.py` | 4 | native resolution | inline, `tab:native` |
| 5 | `fig_deployment` | `make_deployment_fig.py` | 5 | component ablation | `tables_rev/tab_ablation_complete.tex` |
| 6 | `fig_datasets` | `make_datasets_fig.py` | 6 | graft | `tables/tab_graft.tex` |
| | | | 7–8 | deployment | `tables/tab_deploy*.tex` |
| | | | 9 | backbone generality | `tables_rev/tab_backbone_generality.tex` |
| | | | 10–11 | OctMNIST backbone, asym × backbone | `tables/tab_oct_backbone.tex`, inline |
| | | | 12–14 | granular per dataset | `tables/tab_granular_{tissue,derm,oct}.tex` |
| | | | 15 | granular asymmetric | `tables/tab_granular_asym.tex` |

---

## Item 1 — ALM across the full grid

**Status: RUNNING** (launched 2026-08-13 15:56, dsisco01, 4 GPUs).

ALM existed only as a 24-run tight-cap probe (OctMNIST × L30/L40 × 3 backbones ×
4 seeds) under `results/track_b/b3`. It has **zero rows in `corpus_final.csv`**, so
it cannot be tabulated until the expansion lands and is merged.

- Generator: `src/config_generators/gen_alm_full.py`
- Output root: `results/track_b/b3_full` (outside `pending_runs` — the frozen corpus
  cannot be touched)
- Emitted **300 configs** = 3 datasets × 3 backbones × 9 caps × 4 seeds − 24 already done
- Method: clones the frozen `paper_final` Fioretto-LDF config and swaps *only* the
  dual rule, so the CE warmup cache is reused and the pairing is apples-to-apples
- Monitor: `~/alm_full.log` on dsisco01

**ALM on MobileNetV2**: 84 configs generated (`gen_alm_mnv2.py`, cloned from the
existing MobileNetV2 Fioretto-LDF runs since `paper_final` has no MobileNetV2),
chained to launch when the 300 finish.

- [ ] 300-run ALM grid completes, 0 failures
- [x] Generate ALM × MobileNetV2 (84, queued)
- [ ] Run the result-change protocol end to end (ground rule 8) once both land
- [ ] Sanity-check ALM against the 24 B3 runs (the overlapping cells must reproduce)

### Coverage scan — 2026-08-13

Full inventory now exists: **`paper/data/manifest/experiments.csv`**, 10,938 runs
across all six result roots, each row carrying the `config_path` that produced it.
Regenerate with `build_experiment_manifest.py`; re-scan with `coverage_report.py`.

Target = 3 datasets × 4 backbones × 9 caps × 7 methods × 4 seeds at warmup 50
= **3024 runs. Have 2524, missing 500.**

| Missing | Count | Status |
|---|---|---|
| ALM (all backbones, all caps) | 308 | **running now** (300 + 84 queued) |
| MobileNetV2 × the six existing methods | 192 | at L10/L90 and part of L40/L60 only |

**Of the 500, only 99 gate Tables 1–2 — and every one of them is ALM.**
MobileNetV2 is already complete at L30/L50/L70 on all three datasets, so the
headline tables need no new MobileNetV2 runs. The 192 MobileNetV2 gaps sit at cap
levels only the nine-cap figures use.

- [ ] Decide whether the 192-run MobileNetV2 gap-fill is worth running, or whether
      the figures should state that MobileNetV2 covers a partial cap grid

### MobileNetV2 coverage — already better than expected

Surveyed against `corpus_final.csv` at `warmup_epochs==50`: **576 of 648** cells present.

| Cap | L10 | L20 | L30 | L40 | L50 | L60 | L70 | L80 | L90 |
|---|---|---|---|---|---|---|---|---|---|
| coverage | none | full | **full** | partial | **full** | partial | **full** | full | none |

**The three caps the headline tables use (L30/L50/L70) are already complete on all
three datasets, 24/24.** So Tables 1–2 need no new MobileNetV2 runs for the six
existing methods. The 208 missing runs (L10, L90, and L40/L60 partials) matter only
for the nine-cap figures and Table 9.

- [ ] Decide whether the nine-cap gap-fill (208 runs) is worth running

---

## Item 2 — Rebuild Tables 1 and 2

Table 1 is currently one wide table carrying cc-F1 **and** macro-F1 for 6 methods ×
3 backbones × 3 datasets × 3 caps. Table 2 (`tab:almbb`) is a narrow blue patch
holding the ALM and MobileNetV2 results for the OctMNIST tight caps only — it is the
clearest example of a bolted-on revision in the paper.

**Plan:** fold Table 2 into Table 1 (ALM becomes a 7th method column, MobileNetV2 a
4th backbone block), which makes one table far too wide — so split by metric:

- **Table 1 = constrained-class F1**, 7 methods × 4 backbones × 3 datasets × 3 caps
- **Table 2 = macro-F1**, identical shape

`tab:almbb` then disappears as a separate float, and the paragraph introducing it
(`docs/main.tex:735`, "Stronger dual, a fourth backbone, and interval estimates")
must be rewritten — ALM and MobileNetV2 stop being "further checks" and become part
of the main comparison. This is ground rule 2 in action.

- [ ] Extend `paper/scripts/make_main_table.py` to emit two single-metric tables
- [ ] Regenerate with ALM + MobileNetV2 included
- [ ] Re-derive bolding (best / second-best among constraint-*trained* methods)
- [ ] Rewrite `docs/main.tex:735` paragraph; delete `tab:almbb` float in red
- [ ] Check every `\ref{tab:almbb}` and `\ref{tab:ccf1}` still resolves correctly

---

## Item 3 — Figures must cover the same scope as the tables

Audited 2026-08-13. Full generator→data map in `paper/data/PROVENANCE.md`.

| Figure | Needs ALM / MobileNetV2? | Why |
|---|---|---|
| 1 `fig_octmnist` | **Yes, both** | plots every method (top) and per-backbone deltas (bottom) |
| 5 `fig_deployment` | **Yes, both** | per-method and per-backbone deployment properties |
| 2 `fig_convergence` | only if ALM joins the convergence census | currently a 3-method census |
| 3 `fig_loss_shape` | No | analytic, no run data |
| 4 `fig_mechanism` | No | 2-method warmup-1 probe, deliberately off-grid |
| 6 `fig_datasets` | — | being removed (item 8) |

- [x] Audit each figure for which backbones/datasets/methods it plots
- [x] Repair the figure pipeline (all generators pointed at the retired
      `final_AAAI_PAPER/` tree; two crashed, one silently produced a data-less
      figure). All six regenerate correctly now.
- [ ] Regenerate Figures 1 and 5 once ALM lands
- [ ] Decide whether ALM joins the convergence census (Figure 2)
- [ ] Confirm figure scope and table scope agree, and that captions say so

---

## Item 4 — Textual coherence

Full end-to-end read completed 2026-08-13. Verdict: **the skeleton is sound, but the
four new experiments were grafted on without being allowed to change the spine.**
Two diagnostic tells: the Conclusion is still the pre-revision conclusion word for
word, and each blue block sits next to *the objection it answers* rather than at the
place in the argument it changes.

### DECISION REQUIRED — the headline number

The body claims TraLO leads **"the best constraint-trained baseline"** by
**+0.038** cc-F1 (`docs/main.tex:660`). But the blue §5.1 paragraph says ALM beats
Fioretto-LDF by +0.010 and TraLO leads *ALM* by **+0.028**. If ALM is a
constraint-trained baseline — and the paper now calls it one — then "best
constraint-trained baseline" means ALM, and the honest headline is +0.028.

Right now the paper's most-quoted number is computed over a baseline set that
excludes its own strongest baseline. A reviewer will do this arithmetic.

**Recommendation: take the +0.028.** "We beat the textbook augmented Lagrangian"
is a *stronger* claim than "we beat two weaker duals by a bigger margin", because
margins over weak baselines get discounted. Recompute from the full ALM grid rather
than hand-editing, once item 1 lands.

- [ ] Professor sign-off on moving the headline to the ALM-inclusive margin

### Verified defects (spot-checked against the source, not taken on trust)

| ID | Defect | Location | Status |
|---|---|---|---|
| D5 | Body reported the *old* ablation row — said the hinge alone costs $+0.036$ at winrate 1.00, but that is the reset+hinge row; hinge alone is $+0.032$ / 0.92 | `main.tex:972` | **FIXED** |
| D11 | Table 9 caption named the wrong two backbones as clearing $+0.005$ (MobileNetV3 is $+0.004$, ViT-B/16 is $+0.018$) | `tables_rev/tab_backbone_generality.tex` | **FIXED** |
| D1 | Related Work says "Our **two** dual-ascent baselines are…" four lines after the blue text announcing ALM as a third | `main.tex:238` | open |
| D2 | Headline margin supersession (above) | `main.tex:660`, `689`, `697`, `1758` | open |
| D3 | The struck "Missing baselines" limitation named the exact condition under which the claim must tighten — and Table 3 shows that condition **met** on DermMNIST. Deleting it removed the caveat at the moment it came true | `main.tex:1136` | open |
| D4 | §6 and Appendix B are the same argument written twice, same numbers — ~1.5 pages recoverable | `main.tex:965–1027` / `1474–1618` | open |
| D6 | A 96-run positive result + Table 4 filed under Limitations (matches the professor's note) | `main.tex:1051` | see item 6 |
| D10 | §5.3 blue concludes focal's win "reinforces that the advantage belongs to the constraint-trained family" — focal is **not** constraint-trained; the result actually says the margin is over *vanilla* clipping | `main.tex:875` | open |
| D13 | BH correction arithmetic ($6/12$, $12/12$) assumes 2 named components; the table now has 3, so the family is 18 | `main.tex:640` | open |
| D14 | "every method and every backbone in the paper" then enumerates 4 of 6 methods | `main.tex:1055` | open |
| D8 | "the full symmetric grid" describes a table holding 27 of 81 cells; and $L40$ — half the tight-cap evidence — is not in Table 1 at all | `main.tex:654` | open |
| D9 | Table 1 caption cites "Supp. Table S6 / Supp. Sec. D" — leftover from the two-document AAAI version; this is one self-contained PDF | `tables/tab_ccf1.tex:17` | open |
| D12 | MobileNetV2 called a full fourth backbone in the body, "corroboration of sign not size" in its own table caption; it is on a partial cap grid (18 of 27 cells) | `main.tex:743` | open |
| D7 | Abstract calls the native-resolution result a win; §4 calls it a tie. Same 96 runs | `main.tex:98` / `517` | open |

### Repetition to cut (drift, not reinforcement)
- OctMNIST motivation caveat stated **four** times (§4, §7, Broader Impact, App. F)
- Penalty-shape neutrality stated **seven** times
- §6 / App. B duplication (D4) — the largest single space recovery

### Narrative pass — DONE 2026-08-13

The spine was rewritten so the four new experiments change the argument instead of
sitting beside it. In order of the reader's path:

- **Abstract** — was selectively updated (native resolution in, ALM and focal out).
  Now carries all four: the clipping margin is scoped to imbalance-unaware clipping,
  ALM is named as a stronger dual the advantage survives, and the two components are
  stated as individually load-bearing.
- **§1 "Who should care"** — pointed at Limitations for a *positive* result, and the
  pointer went stale when the native-resolution material moved. Repointed at
  App.~\ref{app:native}, with the focal scoping added.
- **§2 Related Work (D1)** — said "Our **two** dual-ascent baselines are…" four lines
  after the blue text announcing ALM as a third, and kept the old *reason not to run
  it* while announcing that we do. Rewritten as one argument: ALM is the sharper
  comparison precisely because its quadratic growth is structurally close to our
  $\rho$ ramp, so beating it cannot be credited to a weak baseline. Three baselines
  named.
- **§4 Setup** — the root cause of ALM reading as a patch: it was never *defined*
  where every other baseline is. Now in the baseline list, in the per-method detail
  (exact multiplier update, verified against `fioretto_alm/train.py`), and in the
  frozen-recipe step sizes.
- **§5.1** — the (iv)(v)(vi) paragraph bundled three reviewer answers in one breath.
  ALM moved into the main claim, where both margins are stated correctly scoped:
  $+0.038$ over the two dual-ascent baselines, $+0.028$ over ALM. **Both numbers
  kept** — the larger one is not lost, and the reviewer's "your best baseline isn't
  in your headline" objection is closed. The paragraph is now (iv) fourth backbone,
  (v) intervals.
- **§5.3 (D10)** — concluded that focal's win "reinforces that the advantage belongs
  to the constraint-trained family." Focal is not constraint-trained; the result says
  the margin is over *vanilla* clipping. Replaced with the conclusion the data
  supports, and it now forward-declares the narrower form.
- **§7 Limitations (D3, D6)** — the struck "Missing baselines" paragraph had named the
  exact condition under which the claim must narrow, and the experiment *met* it, so
  a short replacement states the resulting scope (no numbers — those stay in §5.3).
  The 96-run native-resolution result and its table moved to **App.~E**; Limitations
  keeps only the genuine limitation (OctMNIST has no native counterpart, so the win
  region is untested at full resolution).
- **§8 Conclusion** — was the pre-revision conclusion word for word, the clearest tell
  of bolt-on writing. Now absorbs ALM, the fourth backbone, the narrowed clipping
  margin, native resolution, and the individually load-bearing components.
- **App. B (D4)** — restated §6's findings verbatim, same numbers and sign tests, so
  the reader met the whole argument twice. Struck in the appendix (§6 argues it
  better); the protocol, the caveat, and the tie-region control stay.
- **Table 15 removed** (item 10) — both citing sentences rewritten rather than left
  dangling.

**Verified false positive — do not "fix" it.** The reviewer flagged the BH arithmetic
(`main.tex:672`, "twelve cell-by-component comparisons") as stale now that the
ablation table has three named rows. It is not: the family is six cells $\times$
**two components**, and the third row is a *joint* removal, not a third component.
Twelve stands. The sentence also says "$p{=}0.031$ per load-bearing component" —
which was only strictly true for one component before, and became true for both when
the isolated-hinge arm completed.

### Still open

- [ ] Sweep every remaining scope phrase for the 4th backbone / 7th method
- [ ] Repetition: OctMNIST motivation caveat appears 4$\times$, penalty-shape
      neutrality 7$\times$; §7's "regime-specific" paragraph repeats §5.2
- [ ] `\pending{}` markers must all be resolved before submission — grep for them.
      Currently **1** (the 23-of-27 tie count, which needs ALM's full grid)

---

## Item 5 — Naming consistency

A background agent is building an exhaustive inventory of every named entity and
every spelling variant, across `docs/main.tex` and all table files. Known suspects:
`RegNetY400MF` / `RegNetY-400MF` / `RegNet`, `ViTB16` / `ViT-B/16` / `ViT`,
`MobileNetV3` / `MNetV3`, plus dataset, method, metric and component names.

Papers do get rejected over this. The sweep must be exhaustive, not sampled.

- [x] **Backbone names done** (2026-08-13): 7 bare `RegNet` and 5 bare `ViT`
      completed to `RegNetY-400MF` / `ViT-B/16`. Marked by blue-ing only the
      *completion* (`RegNet{\color{blue}Y-400MF}`) rather than striking the
      professor's word — nothing is removed, so a red strikeout would overstate
      the change and 12 strikeouts would bury the substantive revisions.
- [ ] Table-header abbreviations (`MNetV3`, `Heur.`, `TraLO-b`, `Tissue`/`Derm`
      vs `OctMNIST`) — **deferred to item 2 on purpose**, since those tables are
      being rebuilt and fixing them now means doing it twice
- [ ] Remaining categories: metric `$\Delta$` spacing, winrate forms, `warmup`
      vs `warm-up`, `hard-binding` vs `tight-cap`, the `$eta$` symbol collision
      (hinge weight vs class-balanced hyperparameter)
- [ ] Add a canonical-names table to this file so future edits stay consistent

---

## Item 6 — Move Tables 3 and 4

- **Table 4** (`tab:native`, native resolution): **move to the appendix.** It must not
  sit in the Limitations section — limitations should *reference* the appendix table,
  not display it.
- **Table 3** (`tab:imbal`, imbalanced baselines): may stay in the general results.

- [x] Move Table 4 to the appendix (now App.~E, `app:native`); Limitations keeps
      only the open question and points there
- [x] Table 3 stays in the body. It answers the most obvious challenge to the
      paper's second-biggest claim; burying it would read as evasive.

---

## Item 7 — Figure 4 legend overlaps the plot

`fig_mechanism`'s legend crosses the plotted data. Generator: `paper/scripts/make_figs.py`.

**This check applies to every figure, not just Figure 4.**

- [x] Fixed in the generator via a new `legend_clear()` helper in `fig_style.py`
      that tests candidate positions against the plotted data
- [x] Regenerated
- [x] Rendered and visually confirmed
- [x] All six checked by eye. Figures 1, 2, 3, 5 were already clear; Figure 3's
      apparent axis crowding was a downsampling artifact, confirmed clean at 600 dpi.

---

## Item 8 — Remove Figure 6, expand the dataset appendix

We are not showing images from the datasets. `fig_datasets` (DermMNIST lesions) goes.
In its place the appendix needs a proper written description of **all** datasets used,
not just DermMNIST:

- what the data actually is, and where it comes from
- the full class list per dataset
- basic analysis: class balance, the constrained class and its prevalence, split sizes

- [x] Removed; the float is retained commented-out in the source, and the 
ef in
      Sec.~4 struck without a live reference so it cannot print "Figure ??"
- [x] Written: source, split sizes, full class list with prevalences for all three,
      plus the analysis of *why* they behave differently under a cap (the OctMNIST
      train/test skew -- drusen ~8% train vs 25% test -- is what makes it the
      hard-binding case, and it explains most of the regime map)
- [x] Confirmed: one reference, in Sec.~4, struck

---

## Item 9 — Are Tables 12–14 earning their place?

The granular per-dataset tables (Tissue / Derm / OctMNIST) take a lot of space and
largely restate what the headline tables already show.

- [x] **Decided: keep.** They are the fix for a real problem — Table 1's caption
      says "the full symmetric grid" but shows 27 of 81 cells, and $L40$ (half the
      tight-cap evidence) is not in it at all. These are the only place the full
      grid is visible, so they matter *more* once Table 1 splits by metric, not
      less. Recommend keeping and saying so in App. D's opening.
- [ ] If kept, they must be regenerated with ALM and MobileNetV2 included — a
      granular table that omits two arms the main table has is worse than no table.

---

## Item 10 — Remove Table 15 (asymmetric global vs local)

`tables/tab_granular_asym.tex`. Dense, low coverage, and it does not support the
claim it sits under. **Remove entirely** unless a use is found — and the judgement
is that there isn't one.

- [x] Removed
- [x] Both citing sentences rewritten; the claim now rests on the cross-backbone
      check (Table~11), which supports it better than the removed float did

---

## Verification before this round is called done

- [ ] `docs/main.tex` compiles clean: 0 errors, 0 undefined refs/citations
- [ ] `paper/main.tex` byte-identical to the professor's version (md5 check)
- [ ] Every figure visually inspected in the rendered PDF
- [ ] Rule-1 checker passes (all diffs are revision wrappers only)
- [ ] `main_edited_by_roei.tex` + `overleaf_upload/` regenerated from the final state
