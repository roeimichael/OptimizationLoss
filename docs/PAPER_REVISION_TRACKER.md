# Paper revision tracker — professor's round 2

**Opened 2026-08-13.** This file is the single source of truth for the open work.
Update it as items close; do not open a second tracking document.

Manuscript under edit: **`docs/main.tex`** (the blue revision).
**`paper/main.tex` is the professor's original and must never be edited.**

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

**Still to generate — ALM on MobileNetV2** (36 runs at the table caps, 108 at all
nine). `gen_alm_full.py` clones `paper_final`, which has no MobileNetV2, so these
need a separate source. This is the *only* missing data blocking Tables 1–2.

- [ ] 300-run ALM grid completes, 0 failures
- [ ] Generate + run ALM × MobileNetV2
- [ ] Merge `b3` + `b3_full` into the canonical corpus view
- [ ] Sanity-check ALM against the 24 B3 runs (the overlapping cells must reproduce)

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

Figure 1 (`fig_octmnist`) and the other result figures currently show a subset of
backbones/datasets. Whatever scope the tables claim, the figures must show.

- [ ] Audit each figure for which backbones/datasets/methods it plots
- [ ] Add ALM and MobileNetV2 wherever the figure is a method or backbone comparison
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

- [ ] Fix every occurrence from the agent's report
- [ ] Add a canonical-names table to this file so future edits stay consistent

---

## Item 6 — Move Tables 3 and 4

- **Table 4** (`tab:native`, native resolution): **move to the appendix.** It must not
  sit in the Limitations section — limitations should *reference* the appendix table,
  not display it.
- **Table 3** (`tab:imbal`, imbalanced baselines): may stay in the general results.

- [ ] Move Table 4 to the appendix; leave a reference from Limitations
- [ ] Confirm Table 3's placement reads correctly where it is

---

## Item 7 — Figure 4 legend overlaps the plot

`fig_mechanism`'s legend crosses the plotted data. Generator: `paper/scripts/make_figs.py`.

**This check applies to every figure, not just Figure 4.**

- [ ] Fix the legend placement in the generator (not by hand-editing the PDF)
- [ ] Regenerate the figure
- [ ] **Read the rendered image back and visually confirm** no overlap
- [ ] Repeat the visual check for all six figures

---

## Item 8 — Remove Figure 6, expand the dataset appendix

We are not showing images from the datasets. `fig_datasets` (DermMNIST lesions) goes.
In its place the appendix needs a proper written description of **all** datasets used,
not just DermMNIST:

- what the data actually is, and where it comes from
- the full class list per dataset
- basic analysis: class balance, the constrained class and its prevalence, split sizes

- [ ] Remove `fig_datasets` (red-marked) and its caption
- [ ] Write the expanded appendix section covering every dataset
- [ ] Confirm nothing else referenced the figure

---

## Item 9 — Are Tables 12–14 earning their place?

The granular per-dataset tables (Tissue / Derm / OctMNIST) take a lot of space and
largely restate what the headline tables already show.

- [ ] **Decide: keep or cut.** Give the professor a recommendation with reasoning.
- [ ] If kept, they must be regenerated with ALM and MobileNetV2 included — a
      granular table that omits two arms the main table has is worse than no table.

---

## Item 10 — Remove Table 15 (asymmetric global vs local)

`tables/tab_granular_asym.tex`. Dense, low coverage, and it does not support the
claim it sits under. **Remove entirely** unless a use is found — and the judgement
is that there isn't one.

- [ ] Remove the table and its `\input`
- [ ] Rewrite (red/blue) whatever text referenced it, including the asymmetric-cap
      discussion that leans on it

---

## Verification before this round is called done

- [ ] `docs/main.tex` compiles clean: 0 errors, 0 undefined refs/citations
- [ ] `paper/main.tex` byte-identical to the professor's version (md5 check)
- [ ] Every figure visually inspected in the rendered PDF
- [ ] Rule-1 checker passes (all diffs are revision wrappers only)
- [ ] `main_edited_by_roei.tex` + `overleaf_upload/` regenerated from the final state
