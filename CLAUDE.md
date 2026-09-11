# OptimizationLoss

Thesis project: train neural networks to satisfy **transductive prediction-count constraints**
via soft constraint optimization, and beat a post-hoc clipping baseline.

---

# STOP. READ `docs/FRAMEWORK.md` FIRST.

**It is the only operational document.** It holds the fixed experimental protocol, every idea
that has already failed and why, and the one open question. Everything else in `docs/` is history.

**Do not propose, run, or score anything before reading it.** If any other file disagrees with
it, `docs/FRAMEWORK.md` wins.

## The five rules that get broken most

1. **Warm-up 1 / constraint 29 for trained arms; warm-up 30 / constraint 0 for post-hoc arms.**
   30 optimizer epochs on both sides. **Never run warm-up 50** -- CE saturates and every method
   becomes identical. Never run warm-up 5 -- it is a dead zone; never interpolate between them.
2. **Score at equal compute, with BOTH clippers (`clip` and `focal_clip`) inside the campaign,
   AND COMPARE AGAINST BOTH.** An arm-vs-arm delta is not a result until the bar is in the
   same campaign. ⛔ **THIS LINE SAID "`clip` IS THE STRONGER QUALITY BAR" UNTIL 2026-09-11
   AND IT IS FALSE BY MEASUREMENT.** Over all 33 scored cells in the eleven licensed-unit
   campaigns, `focal_clip` beats `clip` in roughly **20 of 33** and beats **`tralo` in 12 of
   33** -- at EQUAL compute, verified from the configs (`focal_clip` is warm-up 30 /
   constraint 0 = 30 epochs against tralo's 1 + 29, `methodology: heuristic`, the greedy
   clip allocator). So it is a post-hoc clipping baseline, which is the thing CLAUDE.md's
   own first line says to beat, and the sentence above was the reason nothing compared TraLO
   to it. `tralo_wins` scored against `--control clip` and the three rival DUALS only;
   `bcn2rgn`/`L100_G95` is counted a TraLO WIN while `focal_clip` leads it by **11.5 items**.
   ✅ `tralo_wins` now prints the stricter figure BESIDE the old one and names every demoted
   cell -- beside, never instead, because a rule change that only moves a number against the
   method is still a rule change (2(z108)). FRAMEWORK 2(z112).
3. **md5 the raw predictions across arms before reading any metric.** Inert flags are this
   project's most frequent failure mode -- **five** occurrences and counting (the fifth
   is `graph_probe --dump`, an argparse DESTINATION, which `audit_config` cannot see).
   🛑 **BUT md5 IS ONE-SIDED.** Identical predictions prove inertness; DIFFERENT
   predictions prove nothing. `logit_adjust` on iwildcam is mathematically plain CE
   (uniform train prior => a constant added to every logit => shift-invariant) yet its
   predictions differ from `clip` in 24/24, because the constant moves float rounding by
   1e-9 and 30 epochs compound it. To clear a LOSS variant, compare its GRADIENT against
   CE on the real training prior. FRAMEWORK 2(x2).
4. **Atomic cell = (dataset, backbone, cap, method) over 4 seeds. Count cells.** Never pool
   across cap levels, backbones or datasets. Always sweep at least two cap levels -- a
   single-cap claim has been retracted three times.
5. **`flips`, raw count over K, and "proximity to feasibility" are NOT metrics.** Post-hoc
   filling is free. When quality ties, the honest report is "this arm produced nothing."

## Do not run

Anything already in `docs/FRAMEWORK.md` section 2. In particular: penalty-shape variants,
more constraint steps, a dedicated constraint optimizer, the joint objective, the undershoot
hinge, finer constraint granularity. **All of them are measured, and all made things worse.**

🛑 **AND THE WHOLE CONSTRAINT-GRADIENT EXPRESSION IS CLOSED.** The per-logit
gradient is `A_S * p(1-p)`: 2(z56) §5 closed the scope scalar `A_S` (shape, magnitude,
frequency-vs-magnitude, units, granularity, scope SELECTION) and §6 closed the per-item
factor `p(1-p)` (`aim_table`: not one starved cell on `dom1`). Between them that is the
entire expression. Do not propose a count function, a cut window, a margin, a class
re-weighting, or another scope re-weighting.

🔑 **AND PRICE THE MECHANISM BEFORE BUILDING IT -- 2(z77).** At the protocol's
4 seeds the MINIMUM DETECTABLE EFFECT is **6.2-13.5 deployed items**, against a per-cell
prize of **11.7-21.2**. The instrument's resolution and the total prize are the same size,
so a null here is consistent with capturing a third of everything there is to win.

* ⛔ **Stop pricing against the 1.42-item gap to ALM.** Certifying 1.42 items needs
  **78-362 seeds per cell**; the protocol runs 4. Nothing that small is provable here.
* 🟢 **A mechanism worth ~6+ items per cell -- about HALF the smallest cell prize --
  is certifiable at 5-22 seeds**, i.e. one `add_seeds` extension. That is the bar: build
  for half the headroom, not for the gap to ALM. Anything smaller is structurally
  invisible and should not be built.
* ⚠️ The sd behind those figures is ESTIMATED from two recorded medians under a
  normality assumption. `paired_noise --campaign results/dom1` replaces it with a
  measured one; task #109.

## Where things are

```
main.py            dispatcher (kill -INT to stop; interrupted runs reset to pending)
configs/           gen_campaign.py = THE generator (asserts the protocol, refuses to
                   emit a single-cap campaign, always adds both clippers)
data/              ⛔ **THREE RUNNABLE DATASETS, NOT ONE -- THIS LINE SAID
                   "iwildcam -- THE ONLY dataset" UNTIL 2026-09-11 AND HAD BEEN
                   FALSE FOR OVER A WEEK.** `bcn1mn3` is COMPLETE at 228 runs,
                   `fmow1` at 304/304, and `bcn1vit` at 190/190 -- so all
                   three have images on the server, and bcn + fmow supply **3 of the
                   8 licensed units** (D1, E1, E2) including the HEADLINE
                   ViTB16. 🔑 It is not a bookkeeping slip: 2(z105)'s
                   one actionable recommendation is to buy NON-iwildcam units,
                   and a top-of-file line calling iwildcam the only dataset
                   argues the opposite to every reader who gets that far.
                   Only the three `*_meta.csv` pairs are tracked; the `.npy`
                   arrays are gitignored and server-side.
                   The ORIGINAL three are removed and unrunnable, not merely
                   discouraged; see `docs/FRAMEWORK.md` 2(n) -- and they are
                   NOT named here on purpose, because this block is FENCED and
                   `test_no_runnable_command_in_the_docs_names_a_removed_dataset`
                   reads every fenced line as executable. It caught this exact
                   edit on 2026-09-11.
docs/FRAMEWORK.md  THE framework -- protocol, rejected ideas, code purge, open question
docs/MISSION.md    THE RESUME POINT -- goal, knob ledger, priority queue
🛑 **THE RECIPE, AND IT IS THE ONLY CORPUS THAT COUNTS (2026-09-02).**

```
iwildcam + constraint_fp32: True + constraint_grad_mode: normalize
```

**Anything else is a DIFFERENT METHOD, not a variant.** FIVE distinct TraLO
configurations existed across 277 completed `tralo` runs; only 106 were
current. A corpus assembled by campaign NAME rather than by RECIPE mixes
methods, and it did: the one unit that dissented on all three contrasts was the
one campaign running `grad_mode: clip`.

* **18 campaigns / 1,326 configs are archived** at
  `~/optloss-archive-stale-2026-09-02/` (moved, not unlinked, so `results/`
  cannot glob them). `iwc1-4` `loose1` `loosevit1` `vitu1` `xfam1` `taskwin1`
  `uniform1_VOID` `vitdom2_cnn` `vitdom2_vit`, plus seven campaigns on the
  removed leaked-test-set dataset. They are still the receipts for what they
  measured; they are not corpus, and none of them is runnable.
* **`results/` holds `dom1` `dom1b` `equaldose1` `uniform1` `taskwin2`
  `vittask1` `vitdual1` and nothing else**, all one recipe.
* `scripts.rig_status` now REFUSES a campaign off the recipe, and refuses one
  that mixes recipes internally. It caught `vitdual1` staged on `clip` before a
  single run executed. **`gen_campaign` DEFAULTS `--constraint-grad-mode` to
  `clip`, so pass `--constraint-grad-mode normalize` explicitly, every time.**
* ⚠️ **`clip` and `normalize` COINCIDE EXACTLY wherever the raw gradient norm
  is >= 1**, because `clip` scales by `min(raw_norm, 1.0)`. Measured: `loose1`
  (clip) and `dom1` (normalize) produce BYTE-IDENTICAL `tralo` predictions on
  MobileNetV2 in 4/4 seeds despite different commits. So a `clip` campaign is
  not automatically wrong -- it is UNVERIFIABLE, because whether it equals
  `normalize` depends on a norm nobody logged per step.

docs/COVERAGE.md   🗺️ WHAT WE ACTUALLY HAVE vs WHAT THE PAPER NEEDS, built
                   from all 2,671 configs in all 14 worktrees. Read BEFORE
                   proposing a campaign. Carries THE GATE (does TraLO clear
                   its own reseed floor? RECOUNT DONE 2026-09-04: FOUR units
                   had their signs read and only THREE carry a verified `task`
                   cell, so quote BOTH -- ⛔ **SUPERSEDED 2026-09-10, 2(z86): D1 IS NEGATIVE, so task-restricted is 3/4 p=0.3125, not** 4/4 p=0.0625, 3/3 p=0.125
                   task-restricted. `paper_rows` prints the restriction itself)
   🟢 **THE LEDGER LICENSES ELEVEN AS OF 2026-09-11 -- D2, F1 AND G1 WERE
   LICENSED AND READ THE SAME DAY, AND bcn NOW CARRIES ALL FOUR BACKBONES.
   READ THE FOUR TOGETHER OR NOT AT ALL: D2 passes, G1 splits 1 of 2, D1 and
   F1 lose (F1 with a PRICED cell). THE NINTH WAS THE FIRST NON-iwildcam
   UNIT WHERE TraLO **PASSES**.** `bcn1vit` -- bcn / ViTB16,
   COMPLETE at 190 runs -- was licensed as **D2** and read the same hour. At
   its one task cell `L90_G95`, `tralo` beats `clip` AND all three rival duals:
   **+33.00 deployed items against fioretto +17.25, hounie +8.50, alm +3.25**.
   2(z105) had observed that every surviving TraLO unit sat on iwildcam and
   PRE-REGISTERED that a pass off iwildcam refutes it; this is that pass.
   ⛔ **AND IT IS A SIGN, NOTHING MORE.** Zero cells priced (floor 25.5 on 12
   obs, tralo's margin over the best rival 15.75), `deployed_h2h` REFUSES a #1,
   the jackknife flips #1 among {tralo, tralo_linear, tralo_squared}, and
   `tralo_reseed` -- the same null with only the RNG offset changed -- reaches
   **85% of TraLO's advantage over the clipper**. The seed-paired constraint
   contrast is **+4.06 items needing 203 seeds per cell**. FRAMEWORK 2(z107).
   ⛔ **AND THE ACCEPTANCE FIGURE WAS COMPUTED OVER CELLS THE TOOL HAD ITSELF
   DECLARED UNTESTABLE.** `tralo_wins` printed "N OF M CELLS DO NOT POSE THE
   CAP QUESTION" and then put all M in the denominator headed "CELLS THAT CAN
   TEST THE CLAIM" -- the banner is `quarantine.gate()`'s and the arithmetic
   was `tralo_wins`'s, and they never spoke. Fixed, mutation-tested 4/4, and
   **RECOMPUTED the same hour over the eleven licensed-unit campaigns with D2
   in the ledger. THE STANDING FIGURE IS NOW:**

   | denominator | figure |
   |---|---|
   | testable cells | **8 of 27 = 30%**, bar 50%, **VERDICT FAIL** |
   | STRICT `task` cells only | **6 of 18 = 33%** |
   | **...AND ALSO beating `focal_clip`**, the OTHER post-hoc clipper at equal compute | **7 of 27 = 26%** |
   | **per UNIT** (the only axis a p may be computed over) | **2 of 11** (C2, D2) |
   | PRICED cells | **3, tralo wins 1** -- record **1 win 2 losses** |

   Superseded the same day, in order as D2, F1 and G1 were licensed:
   `7 of 23 = 30% / 2 of 9`, `7 of 25 = 28% / 2 of 10`, and before all of them
   `6 of 22 = 27% / 2 of 8`. ⚠️ **QUOTE THE RESTRICTION WITH THE FIGURE** --
   three legitimate denominators give 30% / 33% / 18%, and 2(z66) is the entry
   about exactly that confusion. All three say FAIL.
   🔑 **G1 ADDED TWO CELLS AND ONE WIN, SO THE CELL RATIO ROSE AND THE UNIT
   COUNT DID NOT** -- `bcn2rgn` splits 1 of 2, which is not a majority. With it
   **bcn carries ALL FOUR BACKBONES and they disagree four ways**: D2 passes,
   G1 splits, D1 and F1 lose. The dataset explains nothing in either
   direction. FRAMEWORK 2(z112).
   🛑 **AND THE BAR WAS MISSING ONE OF ITS TWO HALVES UNTIL 2026-09-11.**
   `tralo_wins` scored against `clip` and the three rival DUALS; `focal_clip`
   sat in every campaign at equal compute and was compared to nothing. It beats
   `tralo` in **12 of 33** cells. Adding it demotes exactly one counted win --
   `bcn2rgn`/`L100_G95`, tralo +2.25 against focal_clip **+13.75** -- for
   **7 of 27 = 26%**. Printed BESIDE the old figure, never instead. 2(z112) §3.
   🛑 **AND DO NOT QUOTE D2 WITHOUT F1.** `bcn2mn2` is unit **F1**, bcn /
   MobileNetV2 -- **the SAME DATASET as D2** -- and TraLO loses it 0 of 2 with
   one cell **PRICED** (-22.75 items at L90 against an 18.5 floor on 12 obs),
   at equal 29.00 dose with both cells `task`. So D2's pass is NOT a property
   of bcn, and a single unit cannot establish what makes TraLO work in either
   direction. The pair is the finding. FRAMEWORK 2(z109).
   🔑 The fix dropped ONE win and THREE losses, so the ratio ROSE. A fix that
   only ever moves a number against the method would be the suspicious kind.
   FRAMEWORK 2(z108).
   The superseded line read: **THE LEDGER NOW LICENSES EIGHT AND EVERY SIGN IS
   READ (2026-09-10).**
   `fmow1` completed at 304/304 and licenses TWO -- E1 (MobileNetV3) and E2
   (ViTB16, the HEADLINE backbone) -- so the attainable sign floor moves from
   0.5^6 = 0.0156 to 0.5^8 = 0.0039. ⛔ **AND BOTH CAME BACK 0 OF 2 CELLS**,
   so the ledger grew and the tally did not: the bar is **6 of 22 = 27%,
   per unit 2 of 8**. FRAMEWORK 2(z88). The superseded line read:
   **BUT THE LEDGER NOW LICENSES SIX, AND TWO SIGNS ARE UNREAD
   (2026-09-09).** This line said "the ledger licenses FOUR"; that was true
   when it was written at 17:45 on 2026-09-04 and false by 19:43, when
   `("dom1","MobileNetV3")` was added as C2. `("bcn1mn3","MobileNetV3")` was
   added as D1 today -- a COMPLETE 228-run campaign that was reading
   `UNVERIFIED` and contributing nothing. **`LICENSED` and `SIGN READ` are
   different, and this line collapsed them.** Reading C2 and D1 costs ZERO
   GPU-hours (both are on disk) and is the top of the queue.
   🛑 **BUT READ D1 FIRST, AND KNOW WHICH TALLY 6/6 BELONGS TO
   (2026-09-10).** iwildcam/MobileNetV3 carries `strict class 2: []` -- a
   band measured EMPTY -- and `classify` checks strict before partial, so
   **C2 reads `partial` at every cap on the grid and can never carry a strict
   `task` cell.** `paper_rows` restricts its printed sign test to `task`
   units, so reading BOTH gives ⛔ **REFUTED BY MEASUREMENT 2026-09-10 -- 2(z86). D1 WAS READ AND IS NEGATIVE.** The tally is **5/6 p=0.109** (unrestricted, mean rule) at best and **3/4 p=0.3125** task-restricted; the `worst-cell` rule gives 3/6 p=0.656. C1 and C2 are SPLIT too. Nothing clears 0.05.** **6/6 p=0.0156 unrestricted and 4/4 p=0.0625
   task-restricted** -- the sub-0.05 number is the one the paper-facing scorer
   does NOT print. D1 moves both rows (`bcn1mn3` L80/L90 are verified task
   cells, 2(z58)); C2 moves only the first. A negative is still worth more
   than either. ⛔ DO NOT QUOTE 6/6 BEFORE IT IS READ, AND NEVER WITHOUT
   ITS RESTRICTION. FRAMEWORK 2(z66), **2(z75)**, MISSION 0-UNREAD.
                   COVERAGE also carries the checklist of holes: ViTB16 has zero fioretto/hounie/alm,
                   every run caps the same 2 classes, no symmetric cap ever,
                   1 dataset of 3 -- ✅ THAT HOLE IS CLOSED, NOT
                   CLOSING: `bcn1mn3` 228/228, `bcn1vit` 190/190 and
                   `fmow1` 304/304 are ALL COMPLETE. ⛔ This line said
                   "`bcn1vit` + `fmow1` are running" until 2026-09-11 -- the
                   third instance of 2(z72) on these same two campaigns, and
                   MISSION 0-RUNNING had ALREADY recorded the other two.
                   Re-read COVERAGE before quoting it.
docs/PLAYBOOK.md   WHAT TO DO WHEN A CAMPAIGN LANDS -- the integrity gates in
                   order, how to read the logs and their three traps, and a
                   branch per outcome (win / null / loss / gates red) decided
                   in advance. Read it BEFORE scoring, not after.
docs/THEORY.md     the derivations behind the loss. BACKGROUND, not protocol --
                   where it disagrees with FRAMEWORK, FRAMEWORK wins.
docs/archive/      history, not instructions. 🛑 **AND IT IS IN GIT AGAIN AS OF
                   2026-09-10.** `.gitignore`'s bare `archive/` matches at ANY
                   depth, so `git rm` + `mv docs/archive/` had been silently
                   DELETING for weeks -- 22 files / 1.7 MB were on one disk with
                   no history, including `main.tex` (150 KB) and
                   `BLUE_REVISION_BRIEFING.md` (while its own `.tex` and `.pdf`
                   were tracked). The two `!docs/**/*.tex` / `*.pdf` negations
                   written to protect exactly those could not work: git never
                   descends into an excluded DIRECTORY, so a file-level negation
                   underneath one is dead. Fixed with `!docs/archive/`, which
                   names the directory. Archive by `git mv` and CHECK
                   `git status` shows a rename, not a delete.
                   🛑 **AND BANNER IT -- THE FOLDER DOES NOT TRAVEL WITH THE
                   FILE (2026-09-11).** 14 of the 24 markdown files here had
                   no banner and several read as live instructions: the
                   SUPERSEDED rejected ledger is titled "do not re-introduce
                   without reading this" (the live one is FRAMEWORK section
                   2); `CLEANUP_PROMPT.md` is an imperative MISSION brief
                   executed once; two warm-up-50 tables announce a "Headline
                   F1 win" on TissueMNIST, whose groups are `index % 3`, in
                   the regime where CE saturates and every method ties. A
                   reader arriving by search or grep meets the TITLE.
                   ⚠️ The `launchers/` README was the sharpest: four
                   EXECUTABLE `.sh` wrappers that each start a real campaign,
                   all of them quarantined, under a heading that merely began
                   with the word "Archived". `test_every_archived_doc_SAYS_
                   it_is_archived` now requires the banner in the FIRST FIVE
                   LINES -- below the fold does not count, which is the whole
                   bug. Mutation-tested 2/2, with three negative controls.
docs/paper/        the TMLR manuscript
results/           experiment outputs
scripts/           full_panel.py + score_arm.py = THE scorer; plus dataset prep
src/               the pipeline: losses, methodologies, models, pipeline, training, utils
evidence/          two tarballs: provenance for 14,524 runs, predictions for 128
                   (`mcbar` + `multiclass` only). Extract BOTH into one tree --
                   neither alone yields a scorable run. 0.9% is re-scorable.
```

Nine methodologies, all claimed in the paper: `tralo` - duals `fioretto_ldf` / `hounie_rcl` /
`fioretto_alm` - allocators `heuristic` (greedy clip) / `danits_lp` (LP-LG, Shifman) - and the
imbalanced recipes `focal` / `class_balanced` / `logit_adjust`, each LP-clipped.
⛔ **BUT `class_balanced` AND `logit_adjust` ARE BOTH INERT ON iwildcam**
(FRAMEWORK 2(x1), 2(x2)): the TRAIN set is **exactly 2500/class -- imbalance 1.0x**
(the 4.5x figure is the TEST set). `class_balanced`'s weights are then exactly 1.0 and
weighted CE is plain CE **bitwise**; `logit_adjust` adds `tau*log(prior)`, a CONSTANT
vector, and `log_softmax` is shift-invariant, so its objective is unchanged too.
🛑 **AND THEY FAIL DIFFERENTLY, WHICH IS WHY ONLY ONE WAS CAUGHT.** `cb_lp`'s raw
predictions are byte-identical to `clip`'s in 24/24; **`la_lp`'s DIFFER in 24/24**,
because the constant moves float rounding by ~1e-9 and 30 epochs compound it. So
**md5 divergence is NOT evidence of a live mechanism** -- identical predictions prove
inertness, different ones prove nothing. To clear a LOSS variant, compare its GRADIENT
against CE on the real training prior (`max|g_v - g_ce|` was 9.3e-10 here, eight orders
inside the noise). `focal` survives: it reweights per EXAMPLE and never reads the prior.
✅ `gen_campaign` now REFUSES `cb_lp`/`la_lp` on a dataset whose TRAIN set is balanced
(`--allow-inert-baseline` overrides and says what it let through), and measures the
prior rather than hardcoding iwildcam. Nothing published is affected: all 120
`class_balanced`/`logit_adjust` rows in the corpus are on the three removed datasets,
where the prior really is imbalanced.
🛑 **AND `full_panel` IS ALLOCATOR-BLIND BY CONSTRUCTION**: it re-derives its own
equal-budget allocation from the raw probabilities, so two arms sharing a warm-up model
score `+0.0000` on every budget-equalized metric however differently they allocate.
`lp` vs `clip` reads `+0.0000 p=1.000` while their deployed predictions differ in 23/24.
Compare allocators on `final_predictions.csv` (as-deployed), never on the panel.

**Before launching anything, run all three** -- each refuses a different way to waste a week:

```bash
python -m pytest tests -q                   # 639 regression tests, ~295s, no dataset needed
#   `tests/test_scorers_run_end_to_end.py` EXECUTES every scorer as a subprocess
#   against a campaign carrying a real PARTIAL marker. It exists because three
#   scorers once used `quarantine.` with no module-level import: they PARSED,
#   imported, passed every AST gate and were unrunnable on every input, and the
#   NameError fired only on the branch that a quarantined campaign reaches --
#   the branch that exists to prevent a wrong number. 6/6 mutations caught.
#   🛑 **AND `paper_rows` -- THE TOOL THAT SAYS WHAT MAY BE WRITTEN -- WAS
#   EXEMPT FROM IT UNTIL 2026-09-10, ON A TICKET.** Its exemption reason read
#   "needs a file fixture, task #116" and the ticket sat there, so the one
#   scorer whose output reaches a manuscript had NO end-to-end test: its
#   `--self-test` exercises `build()` in process and never enters `main`,
#   which is exactly the gap that left `order_probe` unrunnable for a day with
#   every gate green (2(z81)). It now has three, running the real CLI as a
#   subprocess: a clean run, the hard-quarantine refusal (with
#   `--allow-quarantined` as its negative control), the PARTIAL drop (dead
#   arms go, LIVE arms stay), and a not-a-cell_table CSV that must be NAMED
#   rather than raise. The campaign names come from `quarantine.REGISTRY`, so
#   the test cannot drift from the registry it checks.
#   🔑 **THE RULE THAT CAME OUT OF IT: AN EXEMPTION WHOSE REASON IS A TICKET
#   IS A DEFECT WITH A COMMENT ATTACHED.** `step_dose` stays exempt and its
#   reason is now a MEASURED one -- `main()` needs `load_data` (the gitignored
#   3.0 GB arrays) and pretrained weights, so no fixture makes it runnable
#   here. Both entries state a fact; neither states an intention.
#   🛑 **AND ITS FIXTURE DID NOT LOOK LIKE A RUN, FOR WEEKS, WITH A COMMENT
#   ABOVE IT SAYING IT MUST (2026-09-09).** It wrote the group column as
#   `Group` while `src/training/logging.py` writes `Group_ID`, so all nine
#   group-aware scorers took their NO-GROUP FALLBACK branch and "25 scorers
#   ran clean" meant they ran on a file no run has ever produced; and it put
#   `seed` at top level while `gen_campaign.py:130` writes only
#   `hyperparams.seed`, so `panel` read `seed: None` throughout -- which
#   crashed `cell_table` on `sorted()` five frames deep the moment the group
#   column was fixed and it got that far. Both are now EXECUTABLE gates that
#   read the authorities (`logging.py`, `gen_campaign.py`, `full_panel.py`)
#   rather than restating them, so renaming a pipeline field turns them red.
#   Mutation-tested 2/2. FRAMEWORK 2(z65).
#   🔑 IT WAS FOUND BY A REFUSAL, NOT BY READING: `paired_noise` began
#   REFUSING a predictions file with no `Group_ID` (2(z63)) and the green
#   test went red the same minute. A tool that guesses cannot find this.
#   🛑 **AND A `--self-test` THAT NEVER ENTERS `main` TESTS THE HELPERS, NOT
#   THE TOOL (2026-09-10).** `order_probe` imported the MODULE
#   `capped_classes` and then defined a FUNCTION of the same name, so
#   `capped_classes.assert_single_dataset` was an **AttributeError on every
#   `--campaign` run** for a day. It parsed, imported, and passed
#   `audit_config`, `doc_commands`, `dead_code`, every AST sweep and the whole
#   suite -- an AttributeError, not a NameError, because the name RESOLVES, to
#   the wrong object. Its own `--self-test` was green: twelve checks, none of
#   them entering `main`.
#   🔑 41 modules carry a `--self-test`; **SEVEN had ever been executed with
#   real arguments.** Two gates now close it:
#     `test_no_module_import_is_shadowed_by_a_local_definition` -- AST, module
#       level only, over `scripts/ configs/ src/`. A function-local `json = 1`
#       is routine and must not fire. 4 controls, mutation-tested.
#     `test_every_gated_tool_fails_CLEANLY_on_an_EMPTY_campaign_root` -- points
#       all 33 root-shaped tools at an empty root and requires a refusal rather
#       than a traceback. **6.2 s at 4-way parallelism.** Eight exemptions,
#       each with a written reason, and the list is checked for ROT so it
#       cannot become a place a tool hides. FRAMEWORK 2(z81).
#   `tests/test_lessons_learned.py` is the CATALOGUE OF LESSONS ALREADY PAID FOR:
#   rejected backbones and datasets with the measured reason each was dropped,
#   the ten deleted config footguns, the BF16/compute-capability split between
#   the two hosts, the oldest allocator bug (an argmax fallback that ignored the
#   cap), the local-scope mirror of it, and a sweep that RUNS all 43 `--self-test`
#   entry points -- nothing else ever ran them together. Every entry is dated and
#   was mutation-tested: 13 mutations, 13 caught, including a false-positive
#   control that a COMMENT naming a deleted key must NOT fire.
python -m scripts.preflight --before-launch # 🛑 THE STAGED GATE. `tests/` gates the CODE;
#   `tests/gates/` gates the EXPERIMENT -- six buckets, one per pipeline stage, each
#   encoding failure modes this project actually PAID for, at the point where each is
#   still cheap to catch. `--before-launch` runs stages 1-4 (data / budget / model /
#   grid): everything answerable from configs and labels alone, no GPU, no dataset for
#   2-4. Then `--stage trainlog` on the FIRST completed run and `--stage results`
#   before quoting a number. `--stage all`, `--list`. Exit code is pytest's, so it
#   drops into CI -- `.github/workflows/preflight.yml` runs the six as a matrix.
#   Every gate carries a NEGATIVE CONTROL in the same test: a gate that has never
#   failed has never been shown to work. A typo'd stage name errors, it does not
#   silently run nothing.
python -m scripts.run_campaign --root <root> --step <step>   # 🛑 THE STEP GATE.
#   Runs the right checks at the right point in a campaign's life and REFUSES
#   to move on when they are red. Five steps: `stage` (before a config exists),
#   `verify` (generated, not yet launched), `launch` (rig health), `firstrun`
#   (the FIRST completed runs) and `score` (before any number is quoted).
#   Each step runs BOTH the `tests/gates` bucket that proves the DETECTOR works
#   and the INSTRUMENT that runs it against THIS campaign -- either alone is a
#   half measure.
#   🔑 THREE OUTCOMES, NOT TWO: pass, FAIL, and UNRUNNABLE. A campaign worktree
#   is PINNED at the commit its configs were generated from, and the gate
#   buckets import training-path modules that may postdate it
#   (`configs.task_cells` on `optloss-domb`). `configs/` is frozen mid-campaign,
#   so that gate genuinely cannot execute there -- and reporting it RED would
#   blame a healthy campaign for version skew. It is named, loudly, as having
#   verified NOTHING. `--skip` toggles a check or a whole step and every skip is
#   announced the same way. `--self-test` gates all three outcomes.
#   ⚠️ `firstrun` IS THE ONE THAT MATTERS: `tralo_uniform` ran at 1/29 dose
#   beside `tralo` at 29/29 in the SAME campaign and still wrote
#   `status: completed`; `iwc3` lost 328 of 1044 steps; `taskwin1` landed 20/29.
#   All three were visible in the first finished run.
python -m scripts.data_present <root>        # 🛑 CAN THESE CONFIGS ACTUALLY
#   READ THEIR DATASET, FROM THIS TREE? Runs inside `--step launch` now, and
#   it exists because on 2026-09-06 a FRESH WORKTREE passed `--step verify`
#   AND `--step launch` with every gate GREEN and then failed 24 runs in 120
#   seconds on `FileNotFoundError: data/iwildcam/oodslice/train_images.npy`.
#   🔑 THE ARRAYS ARE GITIGNORED -- 3.0 GB + 443 MB -- so `git worktree add`
#   gives a tree carrying only the tracked `*_meta.csv`. All fourteen older
#   worktrees had them copied or symlinked in by hand at creation, so nobody
#   had ever made a NEW one and this had never happened.
#   ⚠️ `gate:data` would have caught it and lives in `--step stage`, which
#   runs "before a config exists" -- a campaign generated in one sitting and
#   launched in the next skips straight to `verify`. A gate that only fires
#   in a step people skip is not a gate, so it is duplicated at `launch`,
#   the last thing between a config and a GPU-hour. Costs ~1 second.
#   Follows symlinks (`os.stat`, not `lexists`), so a DANGLING link reads as
#   missing, and a ZERO-BYTE array reads as empty rather than present.
#   Prints the symlink command that fixes it, pointing at the REAL file --
#   the worktree arrays are themselves symlinks into `optloss-audit`, so
#   linking to a sibling makes a chain. `--self-test` gates it, 6 checks, 4
#   negative controls including the exact meta-present/npy-absent shape.
python -m scripts.audit_config              # no config key without a reader, no reader without a key
python -m scripts.doc_commands              # 🛑 AND THE SIBLING RULE FOR THE DOCS:
#   no documented FLAG without an argparse to accept it. The docs carry 110
#   checkable `python -m scripts.<name>` invocations and they are copy-pasted at the
#   worst moment -- a campaign has just landed and a number is wanted. Found by
#   hand 2026-09-07: FRAMEWORK 2(z51) pre-registered
#   `latch_probe --glob <pattern>` and `latch_probe` has never had a `--glob`;
#   the gate then immediately caught a SECOND, `data_present --root <r>`, whose
#   root is POSITIONAL. Both were written, reviewed and committed.
#   STATIC, by AST -- it never executes the modules, so it is safe to run on a
#   host with a live campaign. ⚠️ It ABSTAINS on a module that builds flags
#   dynamically (`audit_config`, `check_parity`) and says which, rather than
#   passing them silently. It does NOT check that a flag DOES anything --
#   that is `flag_live`, and md5 is one-sided (2(x2)).
#   `--self-test` gates it, 8 checks, 5 of them negative controls: a flag in a
#   trailing COMMENT, a flag after a PIPE, and a dynamic module must all NOT
#   fire, while a bad flag on a backslash-CONTINUED line must.
python -m scripts.smoke_arms                # every arm actually RUNS and respects its caps
python -m scripts.smoke_arms --matrix       # + {1,2} capped classes x {L30_G30, L50_G30},
                                            #   caps verified for the TRAINED arms too
python -m scripts.flag_live <armA> <armB>    # md5 across arms: is the new flag LIVE
                                            #   or a fifth inert one? (rule 3)
python -m scripts.verify_caps               # what integer budget each cap tag really produces
python -m scripts.check_parity <root>       # equal compute, same knobs, >=2 caps, sane warm-up sharing
python -m scripts.reachability <early-run>  # CAN the penalty even reach this cell's cut?
#   ⛔ **ITS `live at K` / `flat at K` VERDICT WAS A GLOBAL READING UNTIL
#   2026-09-10 -- THE TENTH SITE, AND THE FIRST ONE THE PER-CALL-SITE GATE
#   FOUND ON ITS OWN (2(z84)).** `slope_at(r[col].to_numpy(), k, ...)` took the
#   globally k-th item of the whole column, off `final_predictions_raw.csv`
#   (the ARGMAX frame). The allocator cuts top-`k_g` WITHIN each group, and on
#   iwildcam -- 7 of 14 ceilings at K=0, groups of wildly different difficulty
#   -- a global top-K is dominated by the confident groups, so it could report
#   `flat` about a cell in which one group's cut is fully live.
#   ✅ It now reads `slope_per_group`, budget-weighted off the DEPLOYED file,
#   and prints the old reading beside it labelled `glob` with the group count.
#   ⛔ DO NOT READ THE GAP AS A DIRECTION (2(z64)). Gated by
#   `test_reachability_reads_the_cut_PER_GROUP_and_the_two_readings_differ`,
#   which also pins the two negative controls: no `Group_ID` yields NaN so the
#   caller falls back to a LABELLED global reading and never a silent one, and
#   a single-group campaign must make the two readings AGREE exactly.
python -m scripts.quarantine --list         # 🛑 IS THIS CAMPAIGN ALREADY DEAD?
#   🔑 **THREE STATES, NOT TWO (2026-09-04, FRAMEWORK 2(z40)).**
#     `scorable=False`              nothing here may be scored
#     `scorable=True` + `dead_arms` PARTIAL: score everything EXCEPT contrasts
#                                   touching the named arms
#     no entry                      live
#   `dom1`, `dom1b` and `equaldose1` (792 runs) are PARTIAL: they carry the
#   SAME 29-vs-28 dose gap that quarantined `vitdual1` -- `fioretto` and
#   `hounie` at 28.00 attempted steps/run, plus **`tralo_lam0` in
#   `equaldose1`** -- but `tralo` vs `clip`/`focal_clip`/`lp`/`alm`/
#   `tralo_uniform`/its own `_null` is at EQUAL dose and UNAFFECTED. A blanket
#   marker would have deleted three of the independent units behind the
#   headline to describe a defect touching two arms. `scorable=True` with NO
#   dead arms is a self-test FAILURE: a registry row that does nothing.
#   🛑 **THE REGISTRY IS THE SOURCE OF TRUTH, NOT `QUARANTINE.json`.**
#   The file is only its on-disk copy, written on ONE host, while scoring
#   happens in fourteen worktrees and on a laptop with no `results/` at all.
#   ✅ **ALL SEVEN SCORERS NOW CALL `quarantine.gate()`.** Until 2026-09-04
#   FIVE checked nothing -- `deployed_h2h`, `paper_rows`, `score_scan`,
#   `paired_noise`, `sensitivity_screen` -- and `paper_rows` is the one that
#   says what may be WRITTEN. It reads a CSV and has no path to walk, so it
#   gates by campaign NAME and DROPS rows for dead arms. Verified on the
#   server: all six path-based scorers exit 1 on `vitdual1`, including on a
#   SUBDIRECTORY of it. Gated 7/7 in `tests/gates/test_g6_results.py`.
#   SEVENTEEN campaigns are marked outright, plus 3 PARTIAL (2026-09-04).
#   🛑 THE TWO NEWEST ARE MARKED FOR A DEFECT NO HEALTH CHECK CAN SEE:
#   `uniform1` (252 runs) and `vittask1` (13) are mechanically PERFECT --
#   clean parity, zero collapse, zero non-finite, 1044/1044 and 29/29 dose --
#   and every one of their cells sits OUTSIDE the measured task window, so
#   they measured the absence of a question. `uniform1` is 9 of 9 cells at
#   L20/L30/L50; `vittask1` is 2 of 2 with class 2 at K/n 0.60 and 0.70
#   against ViTB16's measured [0.80, 0.90]. `vittask1` was ALSO found
#   stalled (34 pending, no dispatcher), and those 34 were dropped rather
#   than resumed. ✅ `quarantine.gate()` now CLASSIFIES the cells of
#   whatever it is about to score and announces every one that poses no
#   question, so this class does not depend on somebody remembering to add
#   a marker. FRAMEWORK 2(z42).
#   The fifteenth was `vitdual1`,
#   the four-dual head-to-head, which ran at UNEQUAL DOSE -- `alm`/`tralo` at
#   29.00 attempted steps/run against `fioretto`/`hounie` at 28.00, with every
#   arm landing 100% of what it ATTEMPTED so no gate was red. Superseded by
#   `vitdual2`. It is still the receipt for the dose gap AND for the ViTB16
#   task window, whose lambda=0 nulls the fix cannot touch. FRAMEWORK 2(z38).
#   This line said TEN, then
#   THIRTEEN -- the three were `dosefix`, `vit_ceskip`, `vit_diag`, and the
#   fourteenth is `taskwin1`, staged WITHOUT --constraint-fp32 and landing
#   20/29 = 69.0%. It is the CLEANEST A/B on that flag in the project:
#   taskwin1 69.0% vs taskwin2 100%, same host, backbone, caps and arms)
#   and `full_panel` REFUSES them (exit 1) unless you
#   pass --allow-quarantined. Each marker names the defect AND what the runs
#   are still a receipt for, because dead and worthless are different: `iwc2`
#   landed 74.6% of its dose with `check_parity` GREEN and is the only evidence
#   that `--constraint-fp32` is load-bearing on ViTB16; the dermmnist campaigns
#   sit on a test set that leaks 38.7% of itself. Both produce a full,
#   plausible panel -- the refusal is the point.
#   `--apply` is a DRY RUN; `--apply --execute` writes markers, drops `pending`
#   runs that must never execute (dead dataset, or a quarantined campaign the
#   dispatcher would still pick up -- a marker does not stop main.py, an absent
#   config does) and corrects a `running` status with no process behind it.
#   ⛔ It NEVER deletes a `completed` run. `results/` is gitignored, disk is
#   31% used with 588T free, so space is never a reason. `--self-test` gates it
#   in both directions: it must refuse the dead AND allow the live.
python -m scripts.cut_gap <roots>           # where is the CUT, and can anything
#   reach it? `gap = hard_count - K` is the distance between the point the
#   penalty pushes (the decision boundary, where `p(1-p)` peaks) and the point
#   the metric reads (rank K, because the allocator emits exactly K). At
#   K/n=0.20 the cut sits at p=0.9999 where `p(1-p)`=0.0001; at K/n=0.90 it is
#   0.59-0.99.
#   🔧 ⚠️ **THOSE TWO FIGURES ARE GLOBAL-SORT NUMBERS AND ARE NOT RE-MEASURED.**
#   `p_K` read `argsort(-P[:,cls])[K-1]` -- the globally K-th item -- while the
#   allocator cuts top-`k_g` WITHIN each group. Fixed 2026-09-09, the FIFTH
#   such site (2(z63) is the fourth). The tool now prints `p_K` (per-group,
#   budget-weighted), `p_K_glob` (the old reading, retained so the docstring
#   table reproduces) and `p_Kmin`/`slope_max`, the DEEPEST group's cut.
#   🛑 DO NOT READ `p_K` vs `p_K_glob` AS A DIRECTION. Only the MINIMUM is
#   ordered against the global value (the global top-K maximises the minimum
#   selected probability); the MEAN sits ABOVE it whenever budgets track group
#   difficulty, which is the normal case. I asserted the opposite, gated it,
#   mutation-tested it 2/2 green, and the end-to-end run refuted it -- the
#   fixture and the claim had come out of the same reasoning. FRAMEWORK 2(z64).
#   🔑 `slope_max` IS THE COLUMN THE GLOBAL READING COULD NOT PRODUCE: whether
#   ANY group's cut carries gradient, not whether the average one does. On
#   iwildcam, with 7 of 14 ceilings at K=0, that is the live-vs-dead question.
#   **NOT YET MEASURED on the corpus** -- needs the server.
#   ⚠️ READ ITS STATUS BLOCK: the geometry is measured, the CAUSAL reading is
#   NOT. Within a warm-up the hard count is constant, so `gap`, `slope_K` and
#   `K/n` are one variable in three costumes (`rho(gap,K) = -1.0000`), both
#   `gap` and `slope_K` REVERSE SIGN once the cap is fixed, and the account's
#   sharp `tralo_uniform` prediction FAILED. Cite it as an unrefuted account,
#   never as a cause. FRAMEWORK 2(y). `--self-test` gates it.
```

## Reading a result

```bash
python -m scripts.pred_integrity <roots>    # 🛑 IS THE PREDICTIONS FILE INTACT?
#   A TORN CSV PARSES. Two dispatchers over shared NFS wrote one run
#   directory and produced a `final_predictions.csv` with SIX EXTRA ROWS, one
#   of them the torn tail of another line (`0.00016164035,218` -- a
#   probability and a group id with no label in front). pandas accepted it;
#   the only tell was that the stray float forced `True_Label` to float64 and
#   sklearn raised FIVE FRAMES DEEP on a dtype, sending the investigation to
#   the metric code when the fault was in the file. An integer fragment would
#   have scored silently with six phantom rows.
#   Two checks, both cheap enough to run every pass: ROW COUNT within a
#   campaign (the test set is fixed, so every run emits the same count --
#   2944 in 111 clean runs, 2950 and 2958 in the torn), and LABEL DTYPE
#   checked LEXICALLY, because pandas is what accepted the file. `full_panel`
#   and `score_scan` now REFUSE rather than score. `--self-test` gates it, 5
#   checks including a POSITIVE control (a different campaign with its own row
#   count must NOT be flagged).
python -m scripts.dose_landed <root>        # 🛑 RUN THIS FIRST, AND ON A RUNNING
#   CAMPAIGN. Per-arm `steps landed / attempted` straight out of config.json --
#   no predictions, no pairing, seconds on a campaign that is 1% done. ONE arm
#   low = the loss shape (`tralo_uniform` 1/29 beside `tralo` 29/29); EVERY arm
#   low = the host (`iwc3` 716/1044, FP16 + GradScaler skips an overflowing
#   step). Read the `amp` column to tell them apart. `--self-test` gates it.
#   🛑 READ ITS `attempted/run` TABLE, NOT ONLY THE PERCENTAGE. Every arm
#   can read 100% and still be at DIFFERENT dose, because that figure is
#   applied/attempted WITHIN an arm. `vitdual1` had `alm`/`tralo` at **29.00**
#   steps/run against `fioretto`/`hounie` at **28.00**: both duals start their
#   multipliers at exactly 0 and updated them AFTER the primal step, so epoch 0
#   took none. ✅ FIXED 2026-09-03 by moving the dual update BEFORE the primal
#   gate (an ordering, not a knob); that campaign was DISCARDED and relaunched
#   as `vitdual2` rather than caveated. Gated end to end in
#   `tests/gates/test_g4_grid.py` (nulls must still attempt ZERO) and in source
#   by lesson 29. FRAMEWORK 2(z38).
python -m scripts.sensitivity_screen --campaign <roots>   # 🛑 COULD THIS CELL HAVE
#   SEPARATED TWO METHODS AT ALL? Run it on the FIRST completed runs, beside
#   `dose_landed`. Three axes, and a FOUR-WAY verdict because "nothing moved"
#   and "we could not have seen it move" are opposite conclusions:
#     GRADIENT  p(1-p) at the per-group cut. Bar 0.0099 = `task_window`'s
#               WIGGLE_MAX pushed through p(1-p), so the two agree by
#               construction. ⚠️ There are already TWO other bars for this
#               quantity, 8x apart (`reachability` 0.040, `cut_gap` 0.005) --
#               SAY WHICH ONE YOU MEAN. And it is read on the FINAL model, so
#               it is a LOWER bound on what the constraint experienced.
#     BAND      items at p in [0.05, 0.95]. Bar is `task_window.MIN_PRIZE`.
#     SPREAD    the typical ARM-PAIR difference in deployed TP, against the
#               RNG floor in the SAME cell.
#   🛑 SPREAD IS PAIRWISE, NEVER `max - min`. A RANGE over k arms grows like
#   `sd*sqrt(2 ln k)` (~3.1*sd at k=10) against a two-arm floor's 1.13*sd, so
#   `range >= floor` certifies PURE NOISE as differentiated at ~2.7x. Measured
#   on the corpus: raw range/floor reads a healthy median 2.51 over 50 cells
#   and the SAME cells read **0.97** once the range is corrected; an sd-based
#   estimator agrees at 0.94. `tests/gates/test_g5_trainlog.py` gates both the
#   arithmetic and the four verdicts, mutation-tested 4/4.
#   🔑 RUN 2026-09-04 OVER dom1 + dom1b + equaldose1 + taskwin2 + vittask1
#   (38 cells): **SENSITIVE 0, FLOOR UNMEASURED 36, SATURATED 2.** The typical
#   arm-pair difference is 2-5 deployed TP items and the RNG floor in the same
#   cell is 1.0-10.5. They are the same size.
#   🛑 **THAT 36 READ `UNDER-POWERED` UNTIL 2026-09-10 AND IT WAS THE
#   WRONG NAME.** All 36 trip the FLOOR branch -- the floor itself rests on too
#   few observations, so the spread is NEVER compared to it and no effect size
#   would have changed the verdict. `UNDER-POWERED` is the DIFFERENT case where
#   the floor IS well estimated and the spread is genuinely smaller. The
#   remedies are opposite: more seeds on the TREATED arms for the second, more
#   seeds or a third STREAM on the lambda=0 arms for the first. Same shape as
#   2(z69). FRAMEWORK 2(z70).
#   ⛔ AND THE FLOOR RESTS ON **FOUR** OBSERVATIONS. Every campaign carries
#   exactly ONE `_null`/`_reseed` pair at 4 seeds, and the four `_null` arms are
#   BYTE-IDENTICAL (FRAMEWORK 2944), so they add no replicates. Below
#   `MIN_FLOOR_OBS` = 8 the screen refuses to decide rather than comparing a
#   well-estimated median against a badly-estimated one.
#   ⛔ **AND `<fam>_reseed` TWINS WOULD NOT HELP EITHER -- I claimed they
#   would and that was WRONG.** `tralo_reseed` is `tralo_null` plus the single
#   key `rng_reseed: True`; the `_null` arms are byte-identical because
#   lambda=0 makes them all plain CE; so an `alm_reseed` is plain CE plus that
#   same key and is byte-identical to `tralo_reseed`. Adding reseed FAMILIES
#   buys nothing. What buys observations is more RNG STREAMS or more seeds:
#     * a third lambda=0 variant (`tralo_reseed2`, a distinct reseed offset)
#       gives 3 pairs x 4 seeds = **12 obs for 8 extra runs**, and needs
#       `rng_reseed` to become an offset rather than a boolean;
#     * seeds 5-8 on the existing pair give **8 obs for 16 extra runs**, and
#       need no code at all.
#   The per-observation price differs 4x, so say which one is being bought.
#   ⚠️ AND DE-SATURATING IS NOT THE INDICATED FIX: FRAMEWORK 2(j) says
#   post-hoc allocation is optimal given the probabilities and that optimality
#   is distribution-free, so a worse model raises the headroom for `clip` too.
#   A bigger prize is not a bigger GAP. `--self-test` gates it, 18 checks.
python -m scripts.deployed_h2h --campaign <roots> --control clip  # 🛑 THE ARM-VS-ARM
#   ONE, and NOT a duplicate of full_panel. full_panel scores its OWN re-derived
#   equal-budget allocation, so it is allocator-blind by design and answers
#   "whose RANKING is better"; this reads `final_predictions.csv` -- what would
#   actually be deployed -- in EXACT captured items. They disagree in RANK
#   ORDER: at dom1/MNv2/L80_G95 the panel puts `tralo` +5.77 over `alm` +5.49
#   while both capture exactly 2602 items, an artefact of cc-F1 being
#   macro-averaged over two classes whose (K+n) differ.
#   🔑 IT REFUSES TO NAME A #1 when the spread is under the RNG floor,
#   and on the clean corpus that is most cells. 🛑 **RECOUNTED 2026-09-04
#   WITH THE DEAD ARMS DROPPED: over the 15 cells that carry rival duals,
#   #1 is namable in 2 -- BOTH `alm`, and TraLO in ZERO.** All four of
#   TraLO's former #1 calls were in verified `task` cells and every one was
#   named because a DEAD ARM sat far enough below it to stretch the spread
#   past the floor, not because it led `alm`. FRAMEWORK 2(z43).
#   ⛔ This line said "19 cells, #1 namable in 6, refused in 13" until
#   2026-09-04; `fioretto` held 2 of that 6. AND that tally does not
#   reproduce -- the same roots now give 8 named, likely a scorer-version
#   difference (server vs local md5). UNVERIFIED; re-run the local scorer.
#   🛑 IT RANKS ON THE SEEDS EVERY ARM SHARES (fixed 2026-09-06). Each arm's
#   delta-vs-control may use every seed that arm shares with the control; an
#   ARM-VS-ARM ORDERING may not, or it compares populations. On the unfinished
#   `vitdual2`/L90-90 -- the ONLY cell with all four duals at equal dose --
#   `alm` ran seeds {1,3} while `tralo` and `hounie` ran {1,2}, and alm's
#   6.5-item lead came from a seed the others had not run. The row also said
#   "3 seeds", the max over arms, not the number behind any comparison.
#   Corrected, on the one common seed: fioretto +6.0, hounie +6.0, alm +3.0,
#   **tralo -8.0 = LAST**, where the old table read tralo +0.50 mid-pack.
#   ⚠️ NOT a bias -- it flattered `tralo` too, and fixing it made TraLO look
#   WORSE. A COMPLETE cell is unchanged, gated as a negative control.
#   FRAMEWORK 2(z50).
#   ✅ THE FLOOR NOW READS **EVERY** lambda=0 STREAM (2026-09-06), not just
#   the `_null`/`_reseed` pair. A campaign carrying a THIRD stream
#   (`<fam>_reseed2`, a distinct `rng_reseed` offset) got NO credit for it:
#   `shape1` reported a floor resting on **2** observations while carrying
#   three streams -- the runs were bought, executed and then not read, the
#   same defect as the `add_seeds` pooling bug. All C(k,2) pairs now count:
#   3 streams x 4 seeds = **12**, which clears `MIN_FLOOR_OBS` = 8.
#   ⚠️ THE VERDICT PRINTS observations AND STREAMS, because they differ: k
#   streams give C(k,2) gaps but only k-1 INDEPENDENT contrasts, so 12 from 3
#   streams is a better median than 4 from 2 and is NOT 12 independent draws.
#   ⛔ `<fam>_lam0` is NOT a stream -- it keeps `lambda_step` and takes real
#   constraint steps, so counting it puts the treatment back in the floor.
#   Gated in both directions plus that negative control.
#   ⚠️ |tralo - rival| median 4.0 items POOLED `alm`+`fioretto`+`hounie`
#   and must be recomputed against `alm` alone (n 180 -> ~60); the value is
#   not restated because it is not measured. The floor it was compared to,
#   |tralo - tralo_reseed| median 4.0, is unaffected. `--self-test` gates it.
python -m scripts.stale_figures              # 🛑 WHICH QUOTED FIGURES CAN TODAY'S
#   CODE NO LONGER REPRODUCE? Holds every date-stamped figure in the docs
#   against the last commit to the scorer that produced it. It exists because
#   `tralo_wins` reported **6 of 17 = 35%** at 09:09 on 2026-09-06 and
#   `deployed_h2h.rank_cell` -- which makes the deltas BOTH halves of that
#   verdict read -- was fixed at 22:48 the SAME DAY. Nothing was red: the
#   figure lives in a doc, the fix lives in git.
#   🔑 RUN 2026-09-10: **56 stale, 16 fresh** over 73 attributable.
#   Reading the `paper_rows` hits found the LICENSED-vs-SIGN-READ collapse
#   still live in three places the 2(z66) recount never reached.
#   🛑 **BUT DO NOT QUOTE THE PER-SCORER TALLY, IT IS INFLATED, AND
#   THAT IS MEASURED.** `paper_rows` appeared to head the list with 12; all
#   twelve were read by hand and came out roughly **5 genuine, 5
#   misattributed, 2 ambiguous**. The bad ones are figures about an md5 audit,
#   a `grep -ciE`, a dose percentage and `dual_cone_probe` that merely sit in
#   an entry naming `paper_rows` above. Attribution by proximity is
#   irreducibly noisy: preferring a script on the figure's OWN line helps and
#   moved that 12 only to 11. So this is a **QUEUE OF FIGURES TO READ**,
#   ordered by how much the number matters -- never a defect count.
#   ⚠️ A HIT IS `UNVERIFIED`, NEVER `WRONG`, and it is a REPORT not a gate --
#   a docstring commit moves the date and changes no number, so it prints the
#   commit SUBJECT and leaves the judgement to a person.
#   ⚠️ ATTRIBUTION IS SECTION-SCOPED, AND THE DEPTH IS MEASURED. FRAMEWORK
#   carries 1 h1, 37 h2 and 218 h3; the **h2s ARE the entries** and the h3s sit
#   inside one. Splitting at h3 ties only 46 of 78; at h2, 72; h1-only reaches
#   77 by letting a figure claim any script in the same chapter, which buys
#   coverage with misattribution. It is also FENCE-AWARE -- a naive split reads
#   every `#   RUN ...` comment in CLAUDE.md's command blocks as a heading and
#   drops this file from 13 figures to 4.
#   🔑 TICK ITEMS OFF WITH `[verified YYYY-MM-DD]` ON THE FIGURE'S OWN
#   LINE. A queue you cannot tick off is a list you re-read forever: a figure
#   somebody had read and confirmed came back identically every run, so the
#   report could only grow. The marker clears a figure ONLY when the
#   verification is STRICTLY AFTER the scorer's last commit, so it
#   SELF-INVALIDATES -- move the scorer again and the marker goes stale with
#   the figure, and it can never become a permanent exemption. Same-day
#   resolves AGAINST the marker (a date carries no hour, and 2(z68) is exactly
#   that case: figure 09:09, scorer 22:48).
#   `--self-test` gates it, 18 checks, 8 negative controls -- incl. that a
#   SAME-DAY figure must not fire, that a marker PREDATING the scorer or
#   sitting on a DIFFERENT line must not clear, that the forward fallback STOPS
#   at the section bound, and that `docs/paper/scripts/make_main_table.py` is
#   not read as ours (it fired on the first real run). Mutation-tested:
#   relaxing the comparison to `>=` turns the same-day control red and nothing
#   else. FRAMEWORK 2(z68).
python -m scripts.stale_provenance          # 🛑 THE OTHER STALENESS AXIS: WHICH QUOTED
#   FIGURES COME FROM DATA THAT NO LONGER COUNTS? `stale_figures` asks whether the
#   SCORER moved; this asks whether the DATA was condemned. Three findings in one day
#   were invisible to the first because none is about code: the seed-budget figures
#   come from `iwc3` (`scorable=False`, outside its own `keep_for`); 2(w3), the only
#   POSITIVE result, is `loose1`, the `clip` recipe that 2(z26-CORRECTED) removed from
#   the corpus BY NAME as "a different method"; and `1.9-9.9 items` is a dermmnist
#   figure that was bare in 17 places across 13 files, one of them a CLI DEFAULT that
#   gates a verdict.
#   It READS the authorities -- `quarantine.REGISTRY` for dead/PARTIAL campaigns,
#   COVERAGE section 0's own table for the recipe census -- so it cannot drift from
#   what it enforces.
#   🔑 IT REFUSES AT ARM GRANULARITY. `scorable=False` condemns everything; PARTIAL
#   does not, so `dom1` fires only when a DEAD ARM is named too. Measured: blanket 41
#   hits, arm-granular **32**, and the nine dropped were about the count function, the
#   scope split and the unit ledger -- none of which reads a dual.
#   ⚠️ **A QUEUE, NEVER A DEFECT COUNT, AND THE NOISE IS MEASURED.** 74 entries
#   DISCLOSE and were cleared. The top SIX of the 32 were read by hand: **3 genuine,
#   2 spurious, 1 ambiguous** -- the same ratio 2(z68) measured for `stale_figures`.
#   Read the entry; the tool orders the reading, it does not judge.
#   `--self-test` gates it, 14 checks, 4 negative controls (a DISCLOSED entry must be
#   cleared, a dead name with NO figure must not fire, a figure with no dead name must
#   not fire, and a `#` inside a fence must not split an entry). FRAMEWORK 2(z78).
python -m scripts.campaign_state            # 🛑 THE THIRD STALENESS AXIS, AND THE ONE A
#   RUN-STATE TABLE CANNOT ASK OF ITSELF: DOES THE CAMPAIGN HAVE A STATE AT ALL?
#   `stale_figures` asks whether the SCORER moved; `stale_provenance` whether the
#   DATA was condemned; this asks whether anybody ever wrote down that the campaign
#   exists. It harvests every campaign-shaped name from CLAUDE.md + the four `docs/`
#   files and holds it against the FOUR authorities, all READ not restated:
#   `quarantine.REGISTRY`, COVERAGE section 0's census, MISSION 0-RUNNING's tables,
#   and CLAUDE.md's own archive list.
#   🔑 IT EXISTS BECAUSE OF `price1`: launched (task #78 COMPLETED), named four
#   times in FRAMEWORK -- twice as "before `price1` launched" -- and load-bearing
#   for 2(z29), with **no run count, dose, host or outcome in any file**. That is
#   2(z72)'s defect INVERTED: a stale row is visible to anyone who re-reads the
#   block, an ABSENT row is visible to nobody, because a table can only be audited
#   for the rows it contains.
#   🔑 TWO HARVEST CHANNELS, AND ONE OF THEM IS THE WHOLE FINDING. A campaign is a
#   PATH when somebody pastes a command and a BACKTICK when somebody writes about
#   it; `price1` is only ever the second, so a path-only harvest returns 24 names
#   and misses it. Both channels: 48.
#   ⚠️ A QUEUE, NEVER A DEFECT COUNT. A campaign named only in a `gen_campaign`
#   command has no state because it has not run, and that is CORRECT. The tool
#   flags mentions carrying a PAST-EXECUTION verb and prints the lines; the verb
#   list is a HEURISTIC and excludes `seeds` and `completed` on purpose (a
#   pre-registration names its seed count; 0-PERM's honest "ZERO completed runs"
#   would otherwise read as a run). FIRST RUN: 48 names, 30 recorded, 18 without a
#   state, 10 with a verb -- read by hand as **3 genuine / 1 near / 6 prose-only**.
#   The three were `price1` (nothing anywhere), `vitdual2` (THREE incompatible
#   progress figures, none dated at its line) and `margin2` (FRAMEWORK said 432
#   runs staged, MISSION had CHECKED that no such campaign exists).
#   ✅ All 18 now carry a row in MISSION 0-RUNNING's campaign-state ledger, and
#   `tests/test_lessons_learned.py` FAILS on any new orphan.
#   `--self-test` gates it, 23 checks, 5 negative controls; mutation-tested 5/5.
#   ⚠️ Its `--all` flag prints the full list. FRAMEWORK 2(z83).
# scripts/cellreport.py                     🛑 NOT A CLI, and not runnable -- the rule-4 per-cell
#   report, in ONE place. `cell_of` returns (backbone, dataset, cap, ARM) or
#   None when the path is too shallow to say; `per_cell_report` prints the
#   table and refuses to call a pooled block legal.
#   ⛔ IT EXISTS BECAUSE THE DUPLICATION WAS MANAGED RATHER THAN REMOVED.
#   Both functions were copied BYTE-IDENTICALLY -- 84 lines -- into
#   `graph_probe` and `scope_probe`; `cell_of`'s own docstring said the two
#   "must stay byte-identical" and `tests/gates/test_g6_results.py` asserted it
#   with `inspect.getsource`. **A test that enforces a duplication is a comment
#   with an assert attached**: it makes drift loud, it cannot make drift
#   impossible. The gate now asserts ONE definition, by object identity rather
#   than by source text -- two `getsource` strings can match while the names
#   point at different functions, which is 2(z81) exactly.
#   ⚠️ `deep_scope.run_parts` is NOT this function and must never be folded in.
#   It returns SIX parts, keeping the campaign and the SEED, because that
#   tool's unit is a cell(seed). It was called `cell_of` until 2026-09-11 --
#   two functions of that name with different arity in one package. Pinned as
#   a negative control. Mutation-tested 3/3.
#   ⚠️ Follows `floors.py`'s rule: it must not import anything reaching `src/`,
#   because its callers run in worktrees PINNED at older commits. Module-level
#   imports: `os`, and nothing else.
python -m scripts.dead_code --paths configs src   # what is DECLARED and never
#   referenced. AST, never grep: a name in a docstring is not a call. A REPORT,
#   not a gate -- a getattr-built call is invisible to it, so confirm by hand.
python -m scripts.full_panel --campaign <root> --control clip   # THE scorer, seed-paired
#   ^ 🛑 READ ITS `CONSTRAINT DOSE` BLOCK ON THE FIRST COMPLETED RUNS,
#     NOT AT THE END. A non-finite constraint gradient makes
#     `finish_constraint_step` drop the update while the run still writes
#     `status: completed`, so an arm can run at 3.4% of its dose and look
#     healthy from every other angle. Measured: `tralo_uniform` 1/29 steps
#     against `tralo` 29/29 in the SAME campaign (FRAMEWORK 2(u)); `iwc3`
#     lost 328 of 1044.
#     ⚠️ **IT WARNS, IT DOES NOT REFUSE.** This line said "full_panel
#     refuses to compare arms more than 5 percentage points apart"; the dose
#     function is print statements and its return value is discarded. Refusing
#     would ALSO be wrong -- `dom1`/`dom1b`/`equaldose1` are PARTIAL, scorable
#     for every contrast not touching the named arms, and a blanket refusal
#     would delete three independent units to describe a defect in two arms.
#     `quarantine.gate()` is what actually gates, at ARM granularity.
#     🔑 AND IT NOW PRINTS **TWO** DOSE STATISTICS, because the percentage
#     alone is blind to the gap that actually happened: applied/attempted is
#     INTERNAL to each arm, so 29/29 and 28/28 both read 100%. The second is
#     ATTEMPTED STEPS PER RUN, which reads 29.00 vs 28.00 and is loud.
python -m scripts.log_health <root>        # what the OPTIMISATION did, per run, from
#   🛑 ITS CROSS-ARM COUNT TABLE IS NOT COMPARABLE. The arms write different
#   log SCHEMAS (tralo* 76 cols, hounie 16, alm 15, fioretto 14), and for every
#   TRAINED arm the last logged `Hard_Class2` disagrees with the model's actual
#   predictions (alm logs 340, emits 467; 0/24 agree) while both nulls agree
#   24/24. Reading that table gave the EXACT OPPOSITE of the truth on dom1.
#   Measure any count from `final_predictions_raw.csv`. FRAMEWORK 3(0c).
                                            #   training_log.csv -- collapse, divergence,
                                            #   satisfaction, count trajectory vs K
python -m scripts.paired_seeds <scan-root>  # each arm minus its OWN lambda=0 twin, per seed
python -m scripts.score_scan <root>         # AUROC / prec@K / Jaccard, grouped by CELL
#   ⛔ **ITS prec@K AND JACCARD WERE A GLOBAL TOP-K UNTIL 2026-09-10 -- THE
#   NINTH SUCH SITE, AND THE FIRST FOUND MECHANICALLY RATHER THAN BY READING
#   (2(z84)).** The comment above the line said "what the cap actually
#   consumes"; the cap consumes top-`k_g` WITHIN each group, and `read()`
#   never loaded `Group_ID` at all.
#   ⛔ **THE DATED FIGURE IS WITHDRAWN: `Jaccard 0.29-0.42 with prec@K
#   identical, 2026-08-20`.** It is a Jaccard between two sets no run ever
#   deployed. The tool now prints the withdrawal in its own footer so it
#   cannot be re-quoted from stale output. ✅ The CHURN MECHANISM stands --
#   that is a claim about what the metric hides, not a number.
#   🔑 THE FIX NEEDED NO RECONSTRUCTION, unlike the other eight:
#   `final_predictions.csv` IS the allocator's output, so the deployed set is
#   `Predicted_Label == c` exactly -- no sort, no budget arithmetic, nothing
#   left to get wrong. The global reading is retained in the RIGHT-HAND
#   columns so old figures reproduce.
#   ⛔ DO NOT READ THE GAP BETWEEN THE TWO HALVES AS A DIRECTION (2(z64)).
python -m scripts.headroom <root>           # items from `clip` to a PERFECT allocator,
                                            #   per cell -- the ceiling any arm is chasing
python -m scripts.paper_rows --cells cells.csv --out paper_rows.csv  # 🛑 THE PAPER ROW,
#   and the one that says what may actually be WRITTEN. Takes `cell_table`'s CSV and
#   emits one line per (cell, CONTRAST) -- vs `clip`, vs the arm's OWN lambda=0 twin
#   (resolved per FAMILY, so `alm`'s effect is never attributed to tralo's model), and
#   vs the `tralo_reseed` RNG floor -- in ITEMS, beside the cell's task-window status
#   and the seeds needed at 80% power. NOTHING is averaged over cells.
#   🔑 IT CARRIES THE INDEPENDENT UNIT, AND THAT IS THE POINT. `dom1` and `loose1` are
#   ONE model byte-identically, and two cap levels in one campaign share a warm-up, so
#   EIGHT cells are FOUR units. A campaign pair absent from `MEASURED_UNITS` reads
#   `UNVERIFIED`, never a free replicate. Sign tests go over UNITS: 4/4 is p=0.0625,
#   not p=0.0039.
#   ⚠️ **THAT `4/4` IS THE ARITHMETIC EXAMPLE, NOT THE LEDGER SIZE, AND THE
#   TWO GOT CONFUSED ONCE ALREADY.** The 8-cells-to-4-units collapse above is a
#   true statement about `dom1`/`loose1`. The LEDGER licenses **SIX** units, of
#   which **four have had their signs read** -- see THE GATE at the top of this
#   file. A unit is LICENSED when it is in `MEASURED_UNITS` and SIGN READ when
#   somebody has scored it; this block quotes the second number and the GATE
#   block quotes the first. FRAMEWORK 2(z66), 2(z68).
#   ⚠️ Run on the corpus 2026-09-01: **1 of 158 strict-task rows clears 2 sd**, and
#   that sd is a rho=0 quadrature, so it is within **sqrt(2)** of the truth in
#   EITHER direction -- `sd(A-B) <= sa+sb <= sqrt(2)*sqrt(sa^2+sb^2)` for any
#   correlation, and positive correlation makes it an OVER-estimate. This line
#   said "a LOWER bound, measured at 6-12x" until 2026-09-03; that figure
#   compares the paired sd to ONE ARM's sd, a quantity the quadrature already
#   contains, so it does not apply. FRAMEWORK 2(z32). Everything else is a SIGN, not a
#   measurement. `items` is approximate -- `full_panel` macro-averages over both
#   capped classes whose (K+n) differ, so no single scale is exact for both.
#   FRAMEWORK 2(z26). `--self-test` gates it, including that the cautious default holds.
python -m scripts.tralo_wins --campaign <roots> --control clip   # 🛑 THE ACCEPTANCE
#   TABLE, and the bar as a command so it is never re-litigated per result.
#   A cell is a TRALO WIN only if `tralo` beats the control AND beats EVERY
#   rival dual present in that SAME cell; TraLO passes at >= 50% of the cells
#   that can test it. Cells holding no rival are EXCLUDED from the denominator
#   and printed separately -- `taskwin2` staged `tralo` alone and cannot test
#   the claim either way.
#   🔑 READ THE `priced` COLUMN, NOT ONLY THE VERDICT. A win is a SIGN;
#   `priced` says the spread cleared the RNG floor AND that floor rests on
#   >= MIN_FLOOR_OBS observations. RUN 2026-09-06 over the whole live corpus:
#   ✅ **RECOMPUTED 2026-09-11 ON THE FIXED TOOL, OVER THE TWELVE
#   LICENSED-UNIT CAMPAIGNS WITH D2, F1 AND G1 ADDED: 8 of 27 = 30%,
#   bar 50%, VERDICT FAIL. Strict-`task` only: 6 of 18 = 33%. PER
#   UNIT: 2 of 11 -- C2 and D2 alone. 3 cells PRICED, tralo wins 1,
#   so the priced record is 1 WIN 2 LOSSES.**
#   ⛔ Superseded the same day, in licensing order: `7 of 23 = 30%
#   / 2 of 9` (D2 only) and `7 of 25 = 28% / 2 of 10` (with F1).
#   🔑 G1 ADDED TWO CELLS AND ONE WIN, so the cell ratio ROSE while
#   the unit count did NOT -- a 1-of-2 split is not a majority. A
#   fix or an addition that only ever moves a number one way would
#   be the suspicious kind; this one moves two numbers apart.
#   ⚠️ QUOTE THE RESTRICTION WITH THE FIGURE -- the three legitimate
#   denominators give 30% / 36% / 22% and all three say FAIL (2(z66)).
#   🔑 The same run FIXED the denominator: 4 cells that pose no cap
#   question were being counted as "cells that can test the claim",
#   and dropping them removed ONE win and THREE losses, so the ratio
#   ROSE 27% -> 30%. FRAMEWORK 2(z108).
#   ⛔ The superseded readings were: **RECOMPUTED 2026-09-10 OVER THE
#   SIX LICENSED UNITS WITH D1 ADDED: 6 of 18 = 33%, bar 50%, per unit
#   2 of 6. ⛔ SUPERSEDED THE SAME DAY BY `fmow1`: 6 of 22 = 27%,
#   VERDICT FAIL, per unit 2 of 8.** fmow is the THIRD dataset and carries the HEADLINE
#   backbone with all four duals at EQUAL 29.00 dose and THREE
#   lambda=0 streams -- the cleanest cells in the corpus. `fmow1`
#   alone is **0 of 4**, tralo NEGATIVE vs clip in 3 of them and LAST
#   of the four duals in all 4. TWO cells are now PRICED and the
#   record is **1 WIN 1 LOSS** -- the loss is fmow1/ViTB16/L20 at
#   -10.00 items against a 6.5-item floor. FRAMEWORK 2(z88), 2(z86).
#   The superseded 2026-09-06 reading was:**
#   **6 of 17 = 35%, bar 50%, VERDICT FAIL -- and 0 of 17 cells are priced**,
#   so every win is a direction and none is reportable. Per unit it is 2 of 6.
#   🛑 **BUT `0 of 17 priced` IS NOT A RESULT ABOUT TraLO -- THE TEST WAS
#   NEVER RUN (2(z69)).** `priced` requires `nfloor >= MIN_FLOOR_OBS` (=8)
#   BEFORE it compares the spread to the floor. Two lambda=0 streams over 4
#   seeds give `1 pair x 4` = **4**, so the third clause is never reached and
#   `priced` is False **by construction, at any effect size**. Every corpus
#   campaign predates `tralo_reseed2` (protocol 2026-09-04, `7f455cb4`).
#   So the corpus is SILENT on the noise question, not negative on it. The
#   WIN count is a separate computation and FAIL still stands.
#   ⛔ **THAT RUN IS 09:09; `rank_cell` GOT THE COMMON-SEEDS FIX AT 22:48
#   THE SAME DAY, so 35% is NOT reproducible from today's code.** Both halves
#   of the verdict read `rank_cell`'s deltas. Of the three same-day fixes only
#   that one bears on the win count: the range->margin change is strictly
#   stricter so `0 of 17 priced` is already at the floor, and de-whitelisting
#   touches PRICING only (`present` reads a FIXED rival list). Direction
#   unknown -- 2(z50) says the fix is not a bias, and the one cell ever
#   examined under it moved tralo +0.50 mid-pack -> **-8.0 LAST**.
#   🔑 QUOTE THE VERDICT, NOT THE FIGURE: FAIL needs 6->9 of 17 to flip
#   and nothing suggests the fix is worth three cells, so say "FAIL, figure
#   pending recompute". FRAMEWORK 2(z68), task #104.
#   `--self-test` gates it in both directions, 13 checks, including that
#   beating the CONTROL but not the RIVAL is NOT a win (the old framing scored
#   that green) and that exactly 50% passes.
#   🛑 **AND IT NOW NAMES THE CELLS THAT NEVER REACHED THE TABLE
#   (2026-09-11).** THREE conditions dropped a cell before it could be
#   scored -- no `tralo`, no control, or `rank_cell` finding the two share
#   NO COMMON SEED -- and all three were a bare `continue`. The summary
#   printed `testable` and `no rival`, so a reader took those two counts
#   for the whole input and a coverage hole in the DENOMINATOR of the
#   acceptance bar was invisible. Each drop is now printed with its reason
#   before the verdict, and before the early return, so it shows even when
#   NOTHING is testable.
#   🔑 THE TWO REASONS ARE KEPT DISTINCT ON PURPOSE: "no tralo" means the
#   arm was never staged, "tralo shares no seed with clip" means it RAN and
#   2(z50)'s common-seed rule dropped it. They look identical from outside
#   and the remedies are opposite -- stage the arm, versus re-run it on the
#   control's seeds. Mutation-tested 4/4: reverting either path to a bare
#   `continue`, collapsing the two reasons into one, and silencing the
#   report block each turn the self-test RED.
python -m scripts.cell_table --campaign <roots> --out cells.csv   # the SURVEY, not the
#   🛑 IT REFUSES A CELL WHOSE RUNS CARRY NO `hyperparams.seed` (2026-09-09).
#   `full_panel.panel` reads the seed from there and NOWHERE else, so a config
#   carrying it at top level made `sorted()` die on
#   `'<' not supported between NoneType and NoneType` -- five frames down, in
#   a paper-facing scorer, reading like an aggregation bug. Same shape as the
#   torn-CSV dtype trap `pred_integrity` exists for. Now named, with a
#   negative control (a cell WITH seeds must not be refused). FRAMEWORK 2(z65).
#   verdict. `full_panel` prints CONTRASTS, so the absolute level an arm reached
#   is nowhere in its output. This emits one row per (campaign, dataset, model,
#   cap, arm) with mean, within-cell seed sd and n_seeds for every metric, plus
#   `dose` and `n_md5`. SEED IS THE ONLY COLLAPSED AXIS and the key is asserted
#   at runtime -- `--self-test` builds cells differing only by backbone and
#   requires they stay separate. Quarantine-gated like `full_panel`.
```

⚠️ `full_panel` now prints a **RESOLUTION** block per contrast: the within-cell
seed sd in items, and the seeds needed at 80% power beside the seeds present. **Read it
before the verdict.** A tie means "no effect" OR "not enough seeds", and those are
opposite conclusions from the same table -- on the live `dualbar2` one contrast reads
`observed +0.36 items, needs ~174 seeds per cell`. It refuses to print a figure at all
when no cell has two seeds, rather than deriving one from nothing.

## Pricing a direction BEFORE spending a GPU

All five run on CPU in minutes against artefacts that already exist, and every one carries
its own liveness control, so a null from them is a measurement rather than silence. Each
closed a direction this project would otherwise have spent a campaign on.

```bash
python -m scripts.budget_share <campaign-root>
#   🛑 WHERE DOES THE FIXED-NORM CONSTRAINT STEP ACTUALLY GO? Under
#   `normalize` the delivered step has norm exactly `lr*clip` whatever the loss
#   is worth, so the RATIOS of the per-scope pull weight
#   `A_S = lambda_S * d(pen)/d(soft)` are the constraint's ENTIRE degree of
#   freedom. This reconstructs both factors from `training_log.csv` +
#   `config.json` -- no GPU, no predictions, no model -- and splits them by
#   what the scope has at stake.
#   🔑 MEASURED ON dom1 2026-09-07: **74.1% of TraLO's step is aimed at K=0
#   scopes whose HARD count is already 0** -- fully compliant, the allocator
#   emits nothing there, no item can change -- because `relu(soft - 0) > 0` for
#   any softmax so the term never switches off. FRAMEWORK 2(z54).
#   ⚠️ READ THE CAVEAT IT PRINTS: `A_S` weights each scope's OWN item set and
#   group sizes differ, so this is the split of the PER-ITEM pull, not of the
#   summed gradient norm. The log carries no per-group item counts.
#   ⚠️ It SKIPS a non-constant ratchet and says so: lambda is reconstructed
#   with the constant formula and applying it to `tralo_dualprop` would report
#   what a constant ratchet WOULD have built. Same trap as 2(z53).
python -m scripts.scale_inversion <campaign-root>
#   🛑 THE COMPANION, AND THE ONE THAT NAMES THE DEFECT. Evaluates TraLO's
#   rule, TraLO's rule WITH `penalty_item_scale`, and ALM's rule on the SAME
#   logged scope-states, so the three are directly comparable.
#   🔑 TraLO divides the excess by `max(K,1)`, making `d(pen)/d(soft)` carry
#   units of 1/budget; ALM's `lambda + mu*r` is in RAW ITEMS with no division.
#   On iwildcam, where 7 of 14 local ceilings are K=0, that INVERTS the scope
#   priority. Measured over 11,136 dom1 scope-epochs:
#     budget      TraLO   item-scaled   ALM
#     K=0         93.5%       15.3%   18.8%
#     K=10..99     4.9%       32.0%   12.0%
#     K>=100       1.7%       52.7%   69.2%
#   ⛔ ALL THREE SHIPPED SHAPES CARRY THE SAME DENOMINATOR (`linear` returns e,
#   `squared` returns e**2, both E/scale), so the penalty-shape ablation in the
#   rejected ledger varied the numerator and could NOT have caught this.
#   ⚠️ IT IS A COUNTERFACTUAL WEIGHTING, NOT A REPLAY. ALM's lambda is
#   path-dependent and ALM trained a different model; this weights ONE fixed
#   set of states by three rules, which is what the `normalize` algebra makes
#   decisive. FRAMEWORK 2(z54).
python -m scripts.penalty_starvation --glob '<runs>/tralo/seed_*'
#   🛑 IS THE PENALTY SHAPE STARVING THE WORST-VIOLATED SCOPE? The shipped
#   `rational_bounded` is BOUNDED in the excess, so its slope is NON-MONOTONE:
#   peaks ~53-58% over and DECAYS toward zero for anything deeper. With one
#   term that divides out under the single clip; with several it sets their
#   RELATIVE weights and sets them BACKWARDS.
#   🔑 MEASURED ON iwildcam 2026-09-06, from `training_log.csv` alone, no GPU:
#   232 epochs over 8 `dom1` runs give **11 live scopes per epoch**, the
#   deepest violated **29.8x** its budget against a median of 0.19x -- a
#   **147x spread** -- and the shipped shape pulls the DEEPEST scope
#   **0.075x (rho=0.5) to 0.014x (rho=100)** as hard as the median one.
#   `linear` reads 92x and `squared` 3926x on the same epochs.
#   ⚠️ The starvation algebra was established on **dermmnist**, which is removed
#   AND whose local scope was empty (0 LP candidates / 52 runs), so the
#   many-term case barely existed where it was shown. iwildcam is 5x worse.
#   ⚠️ READ `spread` FIRST: if the scopes are violated to the same depth the
#   shape has no relative weight to set and the direction closes for free.
#   ⚠️ AND QUOTE A RHO. `rho` ratchets `initial_rho` 0.5 -> `rho_target` 100
#   within a run and the ratio moves 51x -> 167x with it, so a single number is
#   a choice dressed as a measurement -- it prints both ends. `--self-test` is a
#   POSITIVE control against FRAMEWORK 2(a2)'s autograd table at BOTH rho rows.
python -m scripts.latch_probe --campaign <root> --arms tralo tralo_uniform
#   🛑 IS TraLO STILL ADAPTING, AND WHAT IS IT AIMING AT? Two questions from
#   `training_log.csv` alone, no GPU.
#   (1) THE LATCH. `ratchet_gate = satisfaction_epoch is None`, set on the
#   first all-satisfied epoch and NEVER cleared, freezing the lambda ratchet
#   AND the rho ramp for every scope. ⛔ MEASURED: it fires in **0 of 72**
#   dom1 runs -- satisfaction is a global AND and 7 of 14 iwildcam ceilings
#   are K=0, so the conjunction never holds. Closed for free. (`satisfaction_epoch`
#   is NOT persisted; it is reconstructed from `Global_Satisfied` /
#   `Local_Satisfied`, the exact booleans the latch ANDs.)
#   (2) 🔑 FREQUENCY vs MAGNITUDE, and this one is live. TraLO ratchets a
#   CONSTANT per violated epoch, so `lam_c = lam_0 + step * (epochs violated)`
#   -- a FREQUENCY counter whose range is CAPPED BY THE EPOCH COUNT (24.3x at
#   the shipped constants, **2.1x** at the manuscript's). LDF/ALM/Hounie all
#   integrate the violation MAGNITUDE. REPLICATED over SIX campaigns / FOUR
#   backbones / 109 runs (dom1 dom1b equaldose1 vitdual2 taskwin2 uniform1):
#   the latch fires in **0 of 109**, and tralo's lambda range takes only TWO
#   values, both exact -- **24.3x** = (0.01+29*0.05)/(0.01+1*0.05) and
#   **13.3x** with a minimum count of 2. So the range is set entirely by the
#   smallest violation COUNT and **24.3x is the structural CEILING**, already
#   saturated in dom1b and vitdual2. The violations span **282x-1934x**.
#   🛑 READ THE RANGE, NOT THE RHO. Spearman(lam, magnitude) is +0.905, so the
#   ORDERING is about right -- but a rank correlation is invariant to monotone
#   rescaling and hides the 48x range gap. Under `normalize` the summed
#   gradient takes ONE norm over `model.parameters()`, so only the RATIOS
#   across scopes steer. FRAMEWORK 2(z49). `--self-test` gates it, 12 checks,
#   6 negative controls, including a fixture that must INVERT.
python -m scripts.deep_scope --campaign <root> --arms tralo alm lp clip
#   WHICH ARM ACTUALLY CLOSES THE DEEPLY-VIOLATED SCOPES? The OUTCOME test for
#   `penalty_starvation`'s algebra, from `final_predictions_raw.csv` -- the
#   argmax BEFORE the allocator, because the deployed file is forced to exactly
#   K and every scope's excess there is zero by construction.
#   Scopes are bucketed by how deeply the **lambda=0 NULL** violates them, never
#   by the arm under test: bucketing on the arm's own excess sorts scopes by the
#   outcome and guarantees the answer.
#   🔑 ALWAYS PASS A POST-HOC ARM (`clip` / `lp`). They run warm-up 30 /
#   constraint 0, so they take ZERO constraint steps and whatever they score IS
#   the artefact floor. The tool labels them REFERENCE and marks every other arm
#   as clearing the bar or not. It is not decoration: the aiming statistic reads
#   `tralo` +0.447 and the zero-constraint `lp` **+0.400** on the same sets.
#   🛑 MEASURED ON dom1 2026-09-06, 178 violated scopes / 24 cell(seed)s, and it
#   REFUTES the starvation story's outcome form: net of the reference, DEEP is
#   `alm` +12.7 vs `tralo` +12.3 (**TIED**) and the whole gap is at MIDDLE depth
#   (+11.3 vs +6.6). The DEEP bucket is **83% K=0** ceilings, where `s=max(K,1)`
#   makes E/K just the raw count. FRAMEWORK 2(z48).
#   🟢 It also prints the premise nobody had checked: excess removed vs deployed
#   capped-class TP. ✅ **RECOMPUTED 2026-09-09 with `--arms` honoured:
#   rho +0.442 over 6 cells (96 runs), 4 positive (67%)** -- the proxy is not
#   orthogonal to the metric, and the direction survives de-pooling.
#   ⛔ The published figure was **+0.504 over "360 runs"**, and that run count
#   was a BUG, not a design choice: the block iterated every arm on disk while
#   the excess table 150 lines above filters on `--arms`, the same file
#   selecting in one place and not the other. 360 = 60 runs/cell = 15 arms x 4
#   seeds; the documented invocation names FOUR arms, which is the 96 above.
#   ⚠️ **4 of 6 is p=0.34 either way.** De-pooling moved the estimate 12% and
#   changed nothing about its weight: this is a DIRECTION, never a measurement.
#   FRAMEWORK 2(z52).
#   `--self-test` gates it, 10 checks, 5 of them negative controls.
python -m scripts.step_dose --config <config.json> --ce-steps 60
#   🛑 HOW BIG IS THE CONSTRAINT STEP IN WEIGHTS, per delivery rule? Without it
#   a null from `tralo_sgd` cannot be told from underdosing: `sgd` steps a flat
#   `lr*clip` while Adam steps ~`lr*sqrt(N)` (sqrt(N) alone is ~1871 at
#   MobileNetV2 scale), yet only the component ALONG the constraint direction
#   enforces anything. Reports the product -- CONSTRAINT-ALIGNED DISPLACEMENT --
#   for both rules from ONE shared Adam state and ONE gradient, on a REAL
#   backbone.
#   ⛔ **THE DIRECTION FIGURE IS DISPUTED, DO NOT QUOTE 0.009-0.017.** This line
#   used to state it as measured. `constraint_step.py` asserted it uncited, and
#   the string occurs in exactly TWO places in the repo -- there, and FRAMEWORK
#   2(z46) quoting there. Measured here on a real MobileNetV2 it is **0.187 at
#   60 CE steps and 0.258 at 126** -- 15-20x higher and RISING, so "lower after
#   a full epoch" is refuted on its own axis (126 IS the full epoch). It is also
#   NOT the `92.6% stale CE momentum` figure, which is `ortho_survival`'s
#   momentum algebra and does have a receipt.
#   🔑 IT NOW REPORTS THE STATE, NOT ONLY THE COSINE: `|m_ce|`, `r=|m_ce|/clip`
#   and `cos(m_ce, ghat)`. That pair is what sets `shared`'s cos, and the
#   inversion is UNSTABLE in the angle -- cos(dw,ghat)=0.013 needs r=8.5 at
#   angle 0, r=1.8 at -0.05, and is UNREACHABLE at +0.05. So algebra cannot
#   close the dispute; that one measurement can. FRAMEWORK 2(z73), task #107.
#   ⚠️ Real state, not a toy: `hp_liveness_real` exists because the smoke net
#   inverts verdicts, and Adam's `v` is the whole question here.
#   `--self-test` gates it, 10 checks, 2 of them labelled negative controls
#   (empty-state Adam, and no-CE-steps => |m_ce| must read 0). Mutation-tested:
#   zeroing the momentum read turns 2 checks red.
python -m scripts.frozen_head_probe --run-dir <run> --seeds 1 2 3 4 5 6 7 8  # refit ONLY a
                                            #   linear head on the frozen features under
                                            #   a different loss; verdicts in ITEMS, and
                                            #   `seeds_needed` prices any survivor in
                                            #   CAMPAIGN seeds (topk/ptopk: +1.2-1.3
                                            #   items but ~24-36 seeds/cell => unaffordable)
                                            #   ⚠️ EIGHT seeds, not four: the liveness gate
                                            #   is a sign test, p = 2^(1-n), so at
                                            #   --max-sign-p 0.01 it CANNOT pass below 8 at
                                            #   any effect size. At 4 it called a 72-item
                                            #   corruption `NOTHING DETECTED`.
                                            #   ⛔ AND IT DOES NOT TRANSFER TO iwildcam:
                                            #   resolution there is 35.09 items against a
                                            #   0.0-1.0 (tight) / 11.7-21.2 per
                                            #   cell (task) item question -- the
                                            #   `1.9-9.9` here until 2026-09-10 was
                                            #   a dermmnist figure -- so every
                                            #   `NO DIFFERENCE` is an absence of
                                            #   measurement, not a null. FRAMEWORK 2(q)
python -m scripts.prep_iwildcam --annotations <cct.json>     --out data/<name>/oodslice --meta-only  # screen a CANDIDATE dataset with NO
                                            #   images and no GPU: any
                                            #   COCO-CameraTraps annotation file
                                            #   (iWildCam, Terra Incognita/CCT)
python -m scripts.tier_viability <slice-dir> ...  # 🛑 IS THE LOCAL SCOPE A TIER
#   STRUCTURE, OR JUST A SPARSITY PATTERN? `dataset_screen` asks whether
#   per-group LABEL SHIFT exists; that is NECESSARY AND NOT SUFFICIENT, and
#   iwildcam is the proof -- it scores z=96.3, the BEST of 21 candidates, and
#   6 of its 8 classes still cannot carry a local cap at all.
#   🔑 THE PROPERTY: a per-group cap is a real allocation decision only if the
#   class could appear in more than one group -- the way a hospital's
#   gold/silver/bronze tiers are, because any patient could land in any tier.
#   If a class occurs at exactly ONE group then "at most K per group" IS "at
#   most K overall" and the LOCAL scope has silently collapsed onto the GLOBAL
#   one. The campaign still runs and measures a global cap twice.
#   🛑 MEASURED 2026-09-07 over all 21 staged candidates + the incumbent.
#   **iwildcam ranks 20th of 22**: density 0.27, 2 of 8 classes usable, 50% of
#   ceilings K=0 at a 70% cap, and **72% of test items sit in groups holding
#   NONE of the usable classes**. Camera 218 alone is 1657 items = 56% of the
#   test set with zero of BOTH capped classes. Twelve slices read TIER-LIKE
#   with all 8 classes usable (`fitz_atlasfst` 0.94, `bcn_s1` 0.89,
#   `fmow_country_wide` 0.82). ⛔ NONE of the 21 has images on disk.
#   ⚠️ READ `dead` AND `median cell items`, NOT ONLY `usable`. A class spread
#   over 9 groups with 3 items per cell cannot carry a count cap either.
#   `--cap` sets the fraction used to count zero ceilings; `--self-test` gates
#   it, 11 checks, 4 of them negative controls including the DIAGONAL case
#   (each class at exactly one group) which must read DEAD.
python -m scripts.dataset_screen <slice-dir> ...  # CAN a count constraint carry
                                            #   information here? Labels + metadata only,
                                            #   no images/model/GPU. Read the NET column:
                                            #   the DIFFERENTIAL per-group shift, after
                                            #   subtracting BOTH a sampling-noise null and
                                            #   the global shift. octmnist -7, tissuemnist
                                            #   -55 = DEAD (`synth_group` is `index % 3`);
                                            #   derm slice_1 +65 passes stage 1 and STILL
                                            #   nulls, so stage 1 is necessary only --
                                            #   stage 2 is `scope_probe --calibrate`
python -m scripts.task_window --glob '<runs of tralo_null/clip>' --classes 2 7
#   🛑 IS THE CAP A QUESTION AT ALL? Needs a finished UNCONSTRAINED run
#   (not a pre-GPU screen), and it is per (dataset, BACKBONE). Reports the
#   K/n window in which the cap BINDS (evicts >= 10), has a PRIZE (errors
#   inside K) and has WIGGLE (p@K < 0.99). On iwildcam every backbone's
#   window is inside K/n 0.60-1.00, so L20/L30/L50 are ALL non-tasks --
#   24 of 24 cells. The measured windows live in `configs/task_windows.yml`
#   and `gen_campaign` REFUSES caps outside them. `--self-test` gates it,
#   `python -m configs.gen_campaign --self-test` gates the refusal.
python -m scripts.ceiling_screen <slice-dir> --caps L20_G50 L30_G50 --classes 2 7
                                            #   the OTHER half of the question, and it is
                                            #   independent: even where the counts carry
                                            #   information, the PRIZE can be zero.
                                            #   Emitting only K predictions for a class
                                            #   with n true instances caps cc-F1 at
                                            #   `2K/(K+n)`, so the WHOLE prize for any
                                            #   method is `(1-p)*K` items -- no loss, dual,
                                            #   allocator or optimizer changes that bound.
                                            #   🛑 AND THE NOISE MOVES THE SAME WAY, so
                                            #   a prize alone decides nothing. Measured on
                                            #   iwc3 against the PAIRED sd -- the noise the
                                            #   contrast actually run faces -- prize/sd is
                                            #   **0.04-0.09x at L20/L30/L50 and NEVER
                                            #   reaches 1.0, topping out at 0.90x at
                                            #   K/n=90%**.
                                            #   ⛔ **EVERY prize/sd FIGURE HERE IS
                                            #   AGAINST `iwc3`'s TREATED sd AND IS
                                            #   THEREFORE UNVERIFIED** -- `scorable=False`,
                                            #   68.6% dose, `keep_for` = the fp16 receipt
                                            #   only. Same defect as the `paired_noise`
                                            #   block below; the reasoning is spelled out
                                            #   once there. The MECHANISM sentence that
                                            #   follows is a design fact and stands.
                                            #   Re-run on `dom1`. FRAMEWORK 2(z71). 🛑 Pairing GROWS the noise here
                                            #   (7.6-29.1 items vs 0.8-13.5 unpaired):
                                            #   `tralo` and `tralo_null` share one warm-up
                                            #   epoch then train 29 apart, so they are two
                                            #   MODELS, not two readings of one. The
                                            #   RNG-only `tralo_reseed` floor alone already
                                            #   matches the whole prize. So a method
                                            #   capturing 100% of the gap to a PERFECT
                                            #   ranking would not be detectable at 4 seeds
                                            #   at ANY cap. Quote which of the four noise
                                            #   numbers you mean -- FRAMEWORK 2(v) lists
                                            #   them and they differ up to 12x.
                                            #   FRAMEWORK 2(v). K comes from labels + cap
                                            #   policy only and reproduces `headroom`'s K
                                            #   and ceiling exactly with NO model; p@K and
                                            #   the sd are an iwildcam curve and are a
                                            #   guide to WHERE to look, never a substitute
                                            #   for measuring them. `--self-test` gates it,
                                            #   and it CAN say WORTH RUNNING.
python -m scripts.paired_noise --campaign <root>  # 🛑 THE COMPANION TO
                                            #   `ceiling_screen`, AND THE ONE THAT
                                            #   DECIDES ITS VERDICT. A prize is priced
                                            #   against a noise, and FOUR different
                                            #   noises exist here. It prints three of
                                            #   them side by side in the same per-class
                                            #   TP items: `unpaired` (one arm across
                                            #   seeds), `reseed` (RNG only -- the floor
                                            #   under ANY paired contrast) and `treated`
                                            #   (the contrast actually run). ⚠️ **PAIRING
                                            #   GROWS THE NOISE ON THIS DESIGN, 6-12x**:
                                            #   `tralo` and `tralo_null` share ONE
                                            #   warm-up epoch then train 29 apart, so
                                            #   they are two MODELS, not two readings of
                                            #   one. Measured on iwc3, class 2 at
                                            #   K/n=0.2: prize 0.42 items against an
                                            #   unpaired sd of 0.80 (0.52x) but a treated
                                            #   sd of 7.59 (**0.05x**).
                                            #   ⛔ **THE `treated` NUMBER AND THE
                                            #   6-12x RATIO ARE FROM `iwc3`, WHICH IS
                                            #   `scorable=False` (2026-09-10).** Its
                                            #   treated arm ran 716/1044 = 68.6% dose
                                            #   and its `keep_for` is the fp16-dose
                                            #   receipt only. The `unpaired` and
                                            #   `reseed` columns are dose-IMMUNE
                                            #   (lambda=0 arms take zero constraint
                                            #   steps), so HALF this comparison is
                                            #   sound and the half carrying the ratio
                                            #   is not.
                                            #   ⚠️ Direction NOT established: an
                                            #   underdosed `tralo` sits nearer its
                                            #   null, which plausibly SHRINKS the
                                            #   treated sd and would make the true
                                            #   penalty larger -- but that is an
                                            #   argument, not a measurement, and this
                                            #   project has been wrong about exactly
                                            #   this kind of sign before.
                                            #   ✅ THE MECHANISM STANDS REGARDLESS:
                                            #   `tralo` and `tralo_null` share ONE
                                            #   warm-up epoch and then train 29 apart,
                                            #   so they are two MODELS. That is a
                                            #   design fact, not a number. Re-run on
                                            #   `dom1`. FRAMEWORK 2(z71).
                                            #   The 4th number is
                                            #   `full_panel`'s `paired seed sd`, which is
                                            #   macro-averaged `d ccF1` in different
                                            #   units -- NEVER substitute it. `--self-test`
                                            #   🔑 **READ THE `seeds` COLUMN, NOT THE
                                            #   RATIO.** A ratio below 1.0 reads as shut
                                            #   everywhere; the seeds-per-cell at 80% power
                                            #   separates hopeless from merely expensive.
                                            #   iwc3 class 2: **2607 seeds at L20, 546 at
                                            #   L30/L50 -- but only 7-8 at K/n=0.9**, and the
                                            #   protocol already runs 4. So this is closed
                                            #   by the CAP CHOICE, not by physics.
                                            #   ⛔ **THOSE THREE NUMBERS COME FROM A
                                            #   QUARANTINED CAMPAIGN AND ARE UNVERIFIED
                                            #   (2026-09-10).** `iwc3` is `scorable=False`
                                            #   -- fp16 without `--constraint-fp32`, 716 of
                                            #   1044 steps -- and its `keep_for` covers ONLY
                                            #   the fp16-dose receipt, not noise. Worse, the
                                            #   `seeds` column is `seeds_needed(prize,
                                            #   treated_sd)` and the TREATED sd is exactly
                                            #   what a 68.6% dose touches. `paired_noise`
                                            #   now calls `quarantine.gate()`, so it
                                            #   REFUSES `iwc3` and the figures cannot be
                                            #   reproduced without `--allow-quarantined`.
                                            #   ⚠️ Direction UNKNOWN -- an underdosed
                                            #   treated arm could be tighter or looser; it
                                            #   is simply not the design now run. The
                                            #   qualitative shape (tight caps absurd, loose
                                            #   caps cheap) also follows from the prize
                                            #   being ~15x larger at loose caps, so the
                                            #   CONCLUSION is likely safe and the NUMBERS
                                            #   are not. Re-run on `dom1`. FRAMEWORK 2(z71).
                                            #   🔑 AND `stale_figures` CANNOT SEE THIS:
                                            #   the line carries no measurement DATE, and
                                            #   staleness is not the defect anyway. The
                                            #   data was condemned, not the code. ⚠️ The
                                            #   catch, and say it every time: at K/n=0.9
                                            #   the cap barely binds, so where the
                                            #   constraint BINDS nothing is measurable, and
                                            #   where something is measurable the
                                            #   constraint hardly constrains. Half the
                                            #   prize costs 4x the seeds.
                                            #   gates it, and its liveness case proves the
                                            #   tool CAN report that pairing helped.
python -m scripts.scope_probe --campaign <root>   # `L20_G50` and `L50_G20` impose the
                                            #   SAME TOTAL, so the local-vs-global SCOPE
                                            #   question is answerable with the model held
                                            #   fixed. CLOSED the local-cap direction:
                                            #   pinning the split -0.86 items while
                                            #   wrong-shape controls cost 5.3-5.5.
                                            #   `--oracle-split` ALWAYS prints its
                                            #   transfer: the best split found with labels
                                            #   gains +4.18 and transfers at -0.89, so an
                                            #   oracle quoted alone is selection noise
python -m scripts.graph_probe --campaign <root> --dump <csv>  # diffuse the scores over
                                            #   a kNN graph of the stored embeddings -- the
                                            #   one input the allocator provably lacks.
                                            #   ⛔ THE OLD "NULL: +0.50 items, 10/19" WAS
                                            #   MEASURED ON dermmnist, WHICH IS REMOVED AND
                                            #   LEAKS 38.7%. On iwildcam it is NOT a null:
                                            #   +2.01 items, 232/384, controls clean
                                            #   (-13.1 / -16.3).
                                            #   🔑 AND IT IS STILL NOT A DIRECTION.
                                            #   The probe is POST-HOC and scores each arm
                                            #   against its OWN undiffused scores, so a gain
                                            #   every arm shares raises the BASELINE and
                                            #   moves no contrast. Per arm the UNTREATED
                                            #   ones gain MOST: `tralo_null` +2.91 vs
                                            #   `tralo` +1.83, i.e. -1.08 items AGAINST
                                            #   TraLO; 4 of 6 cells positive, sign p=0.34.
                                            #   `--dump` writes the per-arm rows -- it was
                                            #   an INERT FLAG until 2026-09-01, the fifth.
                                            #   `--self-test` gates it: diffusion must WIN
                                            #   on clustered features and the shuffled
                                            #   control must take it away. FRAMEWORK 2(g).
python -m scripts.ortho_survival --compounding  # does a count-function change
                                             #   compound over the 29 steps?
                                             #   ⚠️ MOSTLY NOT, and the first
                                             #   answer here was WRONG. Adam's
                                             #   `(1-b1^k)` accumulation is for
                                             #   CONSECUTIVE steps; train.py puts
                                             #   ~126 CE steps between constraint
                                             #   steps (`b1^126 = 1.7e-6`), so the
                                             #   difference at a constraint step is
                                             #   `(1-b1)/(1-b1^(c+1))` = **0.1000**,
                                             #   the single-step value, forever.
                                             #   The WEIGHT trajectory does
                                             #   compound, weakly: 0.44 -> 2.31 deg
                                             #   over 29 steps under uncorrelated
                                             #   CE and 0.08 -> 0.08 under
                                             #   half-correlated -- a 31x swing on
                                             #   an assumption -- NOW MEASURED: a
                                             #   real net at the trainer's own
                                             #   batch 64 / 126 steps-per-epoch
                                             #   gives lag-1 CE cosine **0.128**
                                             #   at warm-up 1, falling after. At
                                             #   that value the trajectory opens
                                             #   0.26 -> 0.30 deg, i.e. NOT AT
                                             #   ALL, so the per-step figure is
                                             #   the whole story. ✅ The
                                             #   one solid part: the sum-vs-uniform
                                             #   angle is 18.7-49.6 deg, NEVER 180,
                                             #   because `p(1-p)` and its mean are
                                             #   both elementwise non-negative.
python -m scripts.task_window --glob "<root>/<Backbone>/iwildcam/*/tralo_null/seed_*"
                                            #   🛑 RUN THIS BEFORE CHOOSING CAP TAGS.
                                            #   A cap poses a question only if all
                                            #   three hold: it FORCES OUT >=10 items
                                            #   (`hard_count - K`, a count not a
                                            #   boolean -- L90 evicts THREE on class 2
                                            #   and looks binding), there are ERRORS
                                            #   inside K, and p@K < 0.99. Every
                                            #   L20/L30/L50 campaign tested a NON-TASK.
                                            #   🛑 DO NOT QUOTE A K/n RANGE IN PROSE.
                                            #   It was stated three incompatible ways in
                                            #   this repo and two published cells were
                                            #   classified off the wrong one.
                                            #   `configs/task_windows.yml` is the ONLY
                                            #   place a window is a number; ask
                                            #   `configs.task_cells.classify`, which
                                            #   returns K, n, K/n, the window, the
                                            #   grid-snap margin, the row's PROVENANCE
                                            #   and one of task / non_task / no_window /
                                            #   no_data. FRAMEWORK 2(z16), 2(z24).
                                            #   🛑 AND IT NOW CORRECTS FOR THE ARM THE
                                            #   ROW WAS MEASURED WITH (2026-09-10). Four
                                            #   rows -- bcn/fmow x MNv2/RegNet -- carry
                                            #   `reference_arm: clip` because their
                                            #   pilots were staged without `tralo_null`
                                            #   (2(z91)). `clip` runs warm-up 30 /
                                            #   constraint 0 against the declared arm's
                                            #   warm-up 1 + 29 CE epochs, which sharpen
                                            #   the probabilities and move the window UP:
                                            #   the clip band is never HIGHER than the
                                            #   null band at either end, 6 of 6. So the
                                            #   STRICT band is narrowed from below by the
                                            #   median offset (+1 grid step) and a cap
                                            #   that falls out reads **`ref_shifted` ->
                                            #   `unmeasured`**, never `non_task` -- the
                                            #   ratio IS inside a band somebody measured.
                                            #   ⛔ **THE CORRECTION HAD NO READER UNTIL
                                            #   NOW.** It was measured, written into
                                            #   `meta.reference_arm_offset`, and grep
                                            #   found it in TWO places, both comment
                                            #   strings: `classify` took the raw band and
                                            #   `gen_campaign` read the GLOBAL
                                            #   `meta.reference_arm`, never the row's own.
                                            #   `bcn`/MNv2 `L80_G95` would have generated,
                                            #   classified `task`, and entered the unit
                                            #   ledger. `gen_campaign` now REFUSES it
                                            #   (--allow-nontask overrides and says so),
                                            #   printing the arm and BOTH bands -- the
                                            #   generic bucket printed "outside the
                                            #   measured window" beside the CORRECTED
                                            #   floor, which is not in the yml.
                                            #   ⚠️ It NARROWS, never widens, and only the
                                            #   STRICT band: the partial band was not in
                                            #   the six comparisons, so a narrowed cell
                                            #   degrades to `partial`, the weaker claim.
                                            #   ✅ Nothing in the corpus moves -- all 12
                                            #   iwildcam rows use the declared arm, so the
                                            #   shift is 0.0. Gated in both directions on
                                            #   ONE changed field, plus a check that the
                                            #   coded rule reproduces the four bands
                                            #   derived by hand. Mutation-tested 7/7.
                                            #   FRAMEWORK 2(z95).
                                            #   ⚠️ AND IT IS PER CAMPAIGN, NOT JUST
                                            #   PER BACKBONE. The lambda=0 count is 336
                                            #   in dom1/loose1 and 355 in
                                            #   equaldose1/iwc3 on the SAME cached
                                            #   warm-ups; at K=333 that is 3 evicted
                                            #   items against 22. Re-measure on the
                                            #   campaign's OWN reference arm.
                                            #   ⚠️ AND READ `binds n/N`, NOT THE MEAN.
                                            #   The four seeds spread 105 items, so a
                                            #   mean `forced` of 3 is 50 in one seed and
                                            #   -55 in another. A `** PARTIAL n/N **`
                                            #   cap poses its question to n seeds only.
                                            #   ⛔ That does NOT make the other seeds
                                            #   free nulls: md5 says tralo differs from
                                            #   its null in 4/4 slack seeds, because 7
                                            #   of 14 LOCAL ceilings are K=0.
python -m scripts.bias_shift_probe --self-test  # ⛔ REFUTES `tralo_uniform`'s
                                            #   founding claim. Its docstring argues a
                                            #   uniform step in log-odds is "a pure bias
                                            #   shift, which cannot reorder". The step is
                                            #   taken in PARAMETERS, not logits:
                                            #   `dz_i = -lr*g*n*(fbar.f_i + 1)`, which
                                            #   VARIES with `fbar.f_i`. It reorders, and it
                                            #   does so with the backbone FROZEN -- the leak
                                            #   is in the linear head. The only update that
                                            #   provably cannot reorder is one confined to
                                            #   `b_c`, and THAT one is useless: a constant
                                            #   added to `z_c` leaves the within-class order
                                            #   untouched, so the emitted top-K is
                                            #   bit-identical. Pure algebra, no artefact.
python -m scripts.step_direction_probe --glob "<root>/*/iwildcam/*/tralo*/seed_*"
                                            #   🛑 THE ONE THAT PRICES A NEW COUNT
                                            #   FUNCTION BEFORE YOU BUILD IT. Every count
                                            #   `S_c = sum_i phi(p_ic)` has head gradient
                                            #   `sum_i g_i f_i`, a g-weighted MEAN FEATURE,
                                            #   and `normalize` discards the magnitude -- so
                                            #   a new count can only matter if it changes
                                            #   the DIRECTION. Measured on real features the
                                            #   family is THREE clusters: {uniform, 1-p},
                                            #   {sum, margin}, {p, linear, cut-window}, at
                                            #   ~0.99 within and 0.58-0.87 between. So
                                            #   `tralo_margin` is 0.989 from `tralo` and
                                            #   will mostly REPRODUCE it.
                                            #   ⚠️ RUN IT ON REAL FEATURES. A Gaussian
                                            #   toy says 1.0000 for all six and is WRONG;
                                            #   real post-ReLU features are non-negative and
                                            #   anisotropic and give uniform-vs-sum 0.7479.
                                            #   ⛔ **BUT ITS CUT WAS GLOBAL IN TWO PLACES
                                            #   UNTIL 2026-09-10 -- THE SEVENTH SUCH
                                            #   SITE. ⚠️ It read SIXTH for one day:
                                            #   2(z80) recounted to EIGHT after finding
                                            #   that `order_probe` holds TWO of them and
                                            #   the 2(z64) `argsort` audit had cleared
                                            #   the file while listing only one. The
                                            #   audit is per-FILE; the defect is per-CALL
                                            #   SITE.**
                                            #   The mass-at-the-cut band was
                                            #   `argsort(-z)[K-20:K+20]` over the whole test
                                            #   set, and `cut_window`'s centre `tau` was the
                                            #   global K-th logit. The allocator emits
                                            #   top-`k_g` WITHIN each group. So (z12)'s
                                            #   headline "the shipped count puts 0.00% of its
                                            #   gradient at the cut" is UNVERIFIED: `sum` is
                                            #   `p(1-p)`, vanishing as p -> 1, and that
                                            #   table's own `p at the cut` reads
                                            #   0.99984-1.00000. **The FIX had the same flaw**
                                            #   -- on a two-group fixture the globally-aimed
                                            #   `cut_window` puts 0.0000 of its mass on the
                                            #   hard group, the per-group one 0.702.
                                            #   ✅ The THREE-CLUSTER cosine table above is
                                            #   UNAFFECTED (only `cut_window` reads a cut), so
                                            #   the `tralo_margin` prediction stands.
                                            #   ⛔ DIRECTION UNKNOWN -- 2(z64) records that
                                            #   exact claim being fixtured, mutation-tested
                                            #   green and then REFUTED. Both readings now
                                            #   print, with band sizes and a per-ITEM column.
                                            #   It REFUSES a predictions file with no
                                            #   `Group_ID` rather than falling back.
                                            #   Re-run on `dom1`, never `iwc1` (which is
                                            #   `scorable=False` and these figures are outside
                                            #   its `keep_for`). FRAMEWORK 2(z79), task #112.
                                            #   `--self-test` gates it in BOTH directions,
                                            #   now 3 negative controls on the cut, mutation-
                                            #   tested 4/4.
python -m scripts.ortho_survival             # does an intervention installed in
                                            #   `prm.grad` SURVIVE Adam? It mostly does
                                            #   not, and this is the cheapest probe here
                                            #   -- pure algebra, no artefact needed.
                                            #   `ortho_project` delivers **0.0%** of its
                                            #   promised CE-neutrality in 16/16
                                            #   conditions (92.6% of the momentum is
                                            #   stale CE the projection never touches;
                                            #   `sqrt(v)` breaks the orthogonality of
                                            #   the 7.4% it does). And a gradient-MASKED
                                            #   coordinate still steps at **90.4%** of an
                                            #   unmasked one, so `head_only` does not
                                            #   freeze the backbone -- it only keeps
                                            #   constraint INFORMATION out of it.
                                            #   ⚠️ THE GENERAL RULE: `prm.grad` is not
                                            #   the delivery mechanism, Adam is. Verify
                                            #   any grad-level arm at the WEIGHT-DELTA
                                            #   level. `--self-test` gates it.
python -m scripts.straddle_probe --campaign <root>  # how much of the ORACLE headroom is
                                            #   REACHABLE by a step the size ours actually
                                            #   is? `headroom.py` assumes the ranking can
                                            #   be rewritten arbitrarily; 2(a3) measured
                                            #   that we deliver exactly `lr*clip`, so an
                                            #   item misranked by a wide margin is not
                                            #   reachable at any dose. delta is MEASURED
                                            #   from each arm's own `_null` twin, not
                                            #   assumed. `--self-test` gates it.
                                            #   🛑 **IT CUT GLOBALLY UNTIL 2026-09-10 --
                                            #   THE ELEVENTH SUCH SITE, AND THE ONE THE
                                            #   CALL-SITE GATE COULD NOT SEE.** The cut
                                            #   was `np.partition(s, -K)[-K]` and neither
                                            #   `partition` nor `quantile` was in the
                                            #   registry's target list, so the gate that
                                            #   had just produced sites 9 and 10 reported
                                            #   this file CLEAN and held ZERO entries for
                                            #   it. Disclosed in its own docstring from
                                            #   the day it was written ("it ignores the
                                            #   per-group ceilings") and never acted on.
                                            #   Now sums FP-out/TP-in inside each group
                                            #   against that group's own cut; the global
                                            #   reading is retained in a `glob` column.
                                            #   ⛔ NO DIRECTION -- on its own fixture the
                                            #   per-group oracle is LARGER on class 1
                                            #   (2.40 vs 2.20) and SMALLER on class 2
                                            #   (4.40 vs 5.20), in the same run: a
                                            #   per-group selection captures fewer TPs
                                            #   (raises the gap) while
                                            #   `sum_g min(k_g,n_pos_g) <= min(K,n_pos)`
                                            #   (lowers it). FRAMEWORK 2(z85).
                                            #   ⚠️ `--match-contested` is the ONLY
                                            #   ladder comparable ACROSS cap levels:
                                            #   the fraction-of-range one reversed a
                                            #   24/33 trend once density was held
                                            #   fixed. Aggregates key on the ARM too.
                                            #   ⛔ AND IT HAD NO END-TO-END COVERAGE AT
                                            #   ALL until 2026-09-10 -- `--self-test`
                                            #   runs the `--sweep` ladder -- so a
                                            #   mutation reverting it to the global mass
                                            #   left every straddle test green. Two of
                                            #   the three mutations that found gaps hit
                                            #   the CONTROL and the LADDER, not the
                                            #   statistic.
                                            #   `contested` is LABEL-free but NOT
                                            #   model-free -- no model, no ranking, no
                                            #   cut. `dataset_screen` is the pre-GPU one
```

`frozen_head_probe`, `graph_probe`, `scope_probe` and `straddle_probe` need
`test_embeddings.npz`, written by `src/pipeline/features.py` at the end of every run
finished after 2026-08-22. Runs predating it cannot be probed and **must not be
substituted for with synthetic data** -- the probes refuse rather than fall back.
`dataset_screen` is the exception: labels and metadata only, so it runs on a candidate
slice before a single image is ever loaded.

⚠️ **Read `straddle_probe`'s shuffled control in the right DIRECTION.** Shuffling the
scores does not send `reachable` to zero, it RAISES it -- a random top-K scatters
positives on both sides of the cut. It is a *reference* (it depends on n_g, k_g and
prevalence, with the same per-group budgets re-taken on the permuted scores),
and the SIGN of the deviation is the result: `reachable << ctrl` means the ranking already
took the easy swaps, `~= ctrl` means the statistic is reading the score distribution and
means nothing, and `>> ctrl` means positives are parked BELOW the cut -- the one case in
which a cut-local method has something real to win.

⛔ **AND `10.8 vs 11.6` WAS WITHDRAWN 2026-09-10 (2(z85)).** That 1.07x spread does
not reproduce at any seed count under EITHER reading. Re-measured at the widest band:
the reference moves **1.37-1.46x** between the two self-test regimes while the oracle
moves 5.2-6.1x and the real arm 2.5-2.9x. **The licence is that it moves LEAST, not
that it is fixed** -- and the global reading moves the same 1.39x, so this is not
something the per-group fix did.

**Three rules that cost a night each to learn:**

1. **Carry the `_null` arm AND `tralo_reseed`** (`--arms all+null`). The null is the same
   warm-up, allocator and seed with lambda=0, so it isolates the constraint -- and it
   doubles as a post-hoc clipper at equal compute with the allocator held fixed. Without
   it **no count trajectory is attributable**: CE alone swings the capped counts
   242 -> 227 -> 324 -> 233. `tralo_reseed` is that null with the RNG stream perturbed and
   nothing else, and it is the **noise floor**: the constraint moves the capped count RMS
   75-95 items, a reseed moves it 83-95. `gen_campaign` REFUSES a campaign that holds a
   trained arm without it.
2. **Read `d capF1` beside `d macroF1`.** Paired over seeds their precision differs by an
   order of magnitude, and macro-F1 is carried by the UNCAPPED classes, which swing with
   `d capF1` is quantised **PER CLASS**: with exactly K predictions emitted,
   `F1 = 2TP/(K+n)`, so ONE class's `dF1` is an integer multiple of `2/(K+n)`
   (**not** `1/(K+n)`: TP is an integer, so half an item cannot occur).
   ⛔ **BUT THAT RULE IS FALSE FOR THE ccF1 `full_panel` PRINTS, AND USING IT
   AS A BUG DETECTOR THERE IS WRONG.** The printed metric is MACRO-AVERAGED over
   the two capped classes, whose `(K+n)` differ, so the lattice is
   **two-dimensional**: `d ccF1 = a/(K2+n2) + b/(K7+n7)` for integers a, b.
   Measured on `dom1`/`L90_G95` (class 2: K=333 n=370; class 7: K=411 n=456), one
   class-2 item moves ccF1 by `1/703`, which is **0.5583** of the `2/785` quantum
   the old rule predicts -- a HALF-quantum move, routine and legitimate. Since
   `gcd(703,867)=1` the achievable spacing is as fine as `1/609501`, so the
   divisibility test is near-vacuous on the headline metric. Apply it per class,
   or not at all.
   **CONVERT TO ITEMS PER CLASS: `items = dF1 * (K+n)/2`.** `full_panel` prints a
   single scale `sum(K_c+n_c)/2`, and ⚠️ **that scale is exact only when the
   delta splits proportionally to `(K_c+n_c)`, which it never does.** Measured on
   the same cell: +1 real item on class 2 reads as **1.117** items (+11.7%), +1 on
   class 7 as **0.905** (-9.5%), and a NET-ZERO trade of 5 items from class 7 into
   class 2 reports **+1.06 PHANTOM items**. A sub-item delta is not a difference,
   and near one item the SIGN can be an artefact of which class moved.
   ⚠️ **The "1.9-9.9 items" gap from `clip` to a PERFECT allocator is a
   `dermmnist` number**, measured on the REMOVED, 38.7%-leaking dataset. Do not
   quote it for iwildcam, where `headroom` reads 0.0-1.0 on the tight cells.
   🔑 **AND THE CAP CHOICE MOVES IT ~15x. MEASURED 2026-09-06 ON TWO BACKBONES:**
   at LOOSE / task caps the prize is **12.8-20.7 ITEMS PER CELL**, not 0-1.

   ✅ **RE-READ PER BACKBONE 2026-09-09.** The three `dom1` rows used to pool
   MobileNetV2 and MobileNetV3, because `headroom` keyed its cells on
   `(cap tag, class)` with the backbone ABSENT while explicitly refusing to
   pool cap levels. Split:

   | campaign | backbone | cap | c2 | c7 | cell | binds |
   |---|---|---|---|---|---|---|
   | `dom1` | MNv2 | L80_G95 | 7.2 | 6.5 | **13.7** | 4/4, 4/4 |
   | `dom1` | MNv2 | L90_G95 | 11.7 | 9.5 | **21.2** | 4/4, **1/4** |
   | `dom1` | MNv2 | L95_G80 | 6.8 | 8.0 | **14.8** | 4/4, 4/4 |
   | `dom1` | MNv3 | L80_G95 | 8.2 | 3.5 | **11.7** | **3/4**, 4/4 |
   | `dom1` | MNv3 | L90_G95 | 12.0 | 6.8 | **18.8** | **1/4**, 4/4 |
   | `dom1` | MNv3 | L95_G80 | 6.2 | 6.5 | **12.7** | **3/4**, 4/4 |
   | `vitdual2` | ViTB16 | L80-80_G95 | 7.3 | 5.7 | **13.0** | |
   | `vitdual2` | ViTB16 | L90-90_G95 | 12.0 | 8.7 | **20.7** | |

   🔑 **THE POOLED ROW DESCRIBED NEITHER BACKBONE, AND CLASS 7 IS WHERE IT
   SHOWS.** At `L80_G95` the pooled c7 read **5.0**; per backbone it is **6.5**
   (MNv2) and **3.5** (MNv3) -- +30% and -30%, nearly 2x apart. The cell totals
   move less (13.7 vs 11.7) because the two classes' errors partly cancel, so
   reading only the `cell` column would have hidden it.
   ⛔ **AND THE BIGGEST PRIZE IS THE LEAST-BINDING CELL, ON BOTH BACKBONES AT
   ONCE.** `L90_G95` is the 19-21 item row and on each backbone one capped
   class binds in **1 of 4 seeds** -- c7 on MNv2, c2 on MNv3. Different
   classes, same cap. A seed already under budget gets an identically ZERO
   constraint gradient, so most of that prize is quoted from seeds that pose
   no question.

   So "there is nothing to win" was a statement about the TIGHT cells and was
   generalised past its evidence. Observed `tralo` - `clip` deltas are +2 to
   +10 items, i.e. **10-50% of a real prize**, against RNG floors of 3-10. The
   effect and the floor are the same size; the PRIZE is 2-4x both. That is
   under-powered, which is a different conclusion from empty.
   ⛔ **BUT READ `binds n/N` BEFORE QUOTING A CELL.** `vitdual2` L90-90 class 2
   binds in **1 of 3 seeds** and `dom1` L90_G95 in 5 of 8: a seed already under
   budget gets an identically ZERO constraint gradient and is its own null, so
   the cell's mean is diluted by seeds that pose no question.
3. **Check reachability before choosing a cap.** The penalty's per-item gradient scales
   with `p(1-p)`. At the K-th RANKED item that is 0.026 at `L30_G20` (0/4 seeds respond)
   vs 0.055 at `L50_G30` (4/4), and converging the model drops it 60x -- which is what
   "CE saturates" means and why warm-up 50 makes every method identical.
   ⚠️ **But rank K is NOT the decision boundary**, and the two get conflated. When the
   hard count is 300 against K=44, the boundary is at item 300 and rank 44 is buried
   inside the class. At the boundary `p(1-p)` is near its MAXIMUM, and `sum` already puts
   29.4% of its gradient there. Say which point you mean; `docs/FRAMEWORK.md` section 4
   has the measurement.

`smoke_arms` exists because the config gates are structurally blind to a runtime
crash: three arms once shipped with an undefined name in `train()`, burned all 29
constraint epochs, died, were reset to `pending`, and the campaign came back
looking merely unfinished -- with `audit_config` and `check_parity` both green.

**The global cap is redundant at `L30_G30` / `L50_G50` and inert at any `G > L`** -- local caps
are per-group ceilings, so their sum already bounds the count. To make the global scope bind,
sweep `G < L` (e.g. `L50_G30`). See `docs/FRAMEWORK.md` section 1.
Generate a campaign with:

```bash
python -m configs.gen_campaign --root results/<name>     --datasets iwildcam --models MobileNetV3     --caps L80-100_G95 L70-90_G95 --arms all+null --constraint-fp32
```

Add SEEDS to a campaign that already exists -- the thing that is actually
scarce -- with:

```bash
python -m scripts.add_seeds --root results/<live> --seeds 5 6 7 8     --arms clip focal_clip tralo tralo_null tralo_reseed     --out results/<live>seed --execute
```

🛑 **`gen_campaign` CANNOT DO THIS WHILE A CAMPAIGN IS RUNNING, AND THAT IS
WHY THIS EXISTS.** The seed list lives in `configs/protocol.yml`, `configs/` is
frozen mid-campaign, and the seed is not even a config field -- it is baked
into the `base_model_id` hash, so the configs cannot be produced by copying a
sibling and editing a number either. `add_seeds` reads the campaign's own
protocol and writes only into `results/`.

* It **REGENERATES every config already on disk and demands a byte match**
  before writing anything. If it and `gen_campaign` disagree by one default,
  the new seeds are not replicates and the pooled "8 seeds" would be two
  populations of four. It refuses rather than warns.
* It **reads the RECIPE off the campaign**, because `--constraint-fp32` and
  `--constraint-grad-mode` are CLI flags whose protocol defaults are `False`
  and `clip`. A flag that must be typed correctly every time eventually is
  not. It refuses a campaign that mixes two recipes internally.
* `--out` writes the extension to its OWN root. Adding seeds to only some arms
  of a live campaign makes its coverage ragged and turns `check_parity` red;
  the two roots pool because they share a protocol and a `code_version`.
* It refuses a foreign `code_version`, refuses to add an ARM (that is a new
  experiment), and never overwrites or resets anything. `--self-test` gates it
  in both directions, 11 checks, 6 of them negative controls.

🔑 **`--constraint-fp32` IS NOT OPTIONAL, IT IS THE DOSE, AND `gen_campaign`
DEFAULTS IT OFF.** Measured over every completed run in every worktree:
`true` lands **15284 / 15284 constraint steps across 532 runs and 6 campaigns**
(`dom1` `dom1b` `equaldose1` `iwc4` `loose1` `loosevit1`, not one step lost);
`false` lands 86.9% over 189 runs, and that group is the quarantine list.
`taskwin1` was staged without it, landed **20/29 = 69.0%** on `amp=float16`, and
had to be killed at 3/48 and regenerated as `taskwin2`, which lands **29/29** on
the same host. FRAMEWORK 2(u).

⛔ **THE CAPS IN THAT LINE USED TO BE `L20_G50 L30_G50`, AND THAT CAMPAIGN
MEASURES NOTHING.** A cap poses a question only where it evicts >= 10
predictions, leaves errors inside K, and cuts at `p@K < 0.99`. Measured on all
four backbones (`docs/FRAMEWORK.md` 2(z16), 2(z17)): **24 of 24 (backbone x
class x cap) cells at L20/L30/L50 fail at least one of those, and 8 of 8 at
K/n=0.90 pass**. At L20/L30 on ViTB16 both capped classes have literally ZERO
errors inside K. `gen_campaign` now REFUSES those caps against the measured
windows in `configs/task_windows.yml`; `--allow-nontask` overrides it and says
in the output what it let through.

🔑 The per-class form `L<c2>-<c7>_G<g>` exists because the two classes'
windows differ on **some** backbones -- `L80-100_G95` caps class 2 at 80% and
class 7 at 100%. ⚠️ **IT IS NOT "every backbone" ANY MORE, AND THAT IS A
MEASUREMENT.** MobileNetV2's two strict windows coincide at 0.80, and **ViTB16
joined it 2026-09-03**: measured off two distinct `vitdual1` nulls, both classes
come back **[0.80, 0.90]** on the 0.1 grid (the underlying prizes still differ,
class 2 3.5/6.0 items against class 7 4.5/8.0, but the grid verdict does not).
So on half the backbones a single fraction is legal, written per-class or not.
Ask `configs.task_cells.classify`; never assume the form is forced.

## Eight more tools that exist and were invisible here

Audited 2026-08-25: these are in `scripts/`, are useful, and were named in
neither this file nor `docs/FRAMEWORK.md` -- which by this project's own rule
means nobody ran them.

```bash
python -m scripts.rig_status                 # 🛑 RUN THIS BEFORE AND AFTER EVERY
                                             #   LAUNCH. Every operational failure
                                             #   here has been SILENT and every row
                                             #   is one that already happened: a
                                             #   launch that ran 40 runs on CPU
                                             #   because `bash -c` re-sourced
                                             #   .bashrc and flipped conda to base;
                                             #   a killed dispatcher leaving three
                                             #   runners alive writing into a
                                             #   directory a fresh dispatcher had
                                             #   claimed; a sibling checkout sharing
                                             #   the live campaign's git object
                                             #   store; a GPU picking up a second
                                             #   user. None of those raise.
python -m scripts.factorial_control --self-test   # the CONTROL for `dataset_screen`,
                                             #   and it bounds where that screen is
                                             #   valid. 2(n)'s baseline gives an
                                             #   unseen group the global training
                                             #   prevalence -- right for an ATOMIC
                                             #   group (a camera, a trap), TOO
                                             #   GENEROUS for one built as a PRODUCT
                                             #   of factors that both appear in
                                             #   training, because the model can
                                             #   interpolate. Run it on any
                                             #   subpopulation slice before
                                             #   believing the screen.
                                             #   READ `raked`, NOT THE PERCENTAGE.
                                             #   When `--sep` is absent from the
                                             #   label, `split[0]` and `split[-1]`
                                             #   are the SAME string, every unseen
                                             #   group keeps the global prior, and
                                             #   the two arms become one arm -- so
                                             #   `survives` was ~100% by ARITHMETIC.
                                             #   `raked=0` now prints NOT A CONTROL.
                                             #   8 of 21 candidates rake zero,
                                             #   `iwildcam` and every `fmow` among
                                             #   them, so the old table's two
                                             #   top rows (both 100.1%) were never
                                             #   measured and iwildcam was NOT the
                                             #   positive control it was quoted as.
                                             #   fmow is still the clean second
                                             #   dataset -- because a country is
                                             #   ATOMIC and the gate does not apply,
                                             #   NOT because it scored 100.1%.
                                             #   FRAMEWORK 2(w2c).
python -m scripts.hp_liveness_real           # `hp_liveness` answers "which knob can
                                             #   change a result" on the SMOKE NET,
                                             #   where the clip never engages -- so
                                             #   lambda/rho read LIVE and
                                             #   `constraint_grad_clip` reads INERT,
                                             #   and on ViTB16 both verdicts SHOULD
                                             #   invert. A knob sweep justified by the
                                             #   smoke net sweeps cancelled quantities.
                                             #   ⛔ **THAT INVERSION IS A PREDICTION,
                                             #   NOT A MEASUREMENT, AND THIS LINE SAID
                                             #   "INVERT" UNTIL 2026-09-10.** The tool's
                                             #   own docstring says "should inverT" --
                                             #   it is the motivation for building it.
                                             #   ✅ **RUN FOR THE FIRST TIME
                                             #   2026-09-10** on a real ViTB16
                                             #   (`fmow1`/L30/tralo/seed_1),
                                             #   dsisco02 GPU 2. The FIRST HALF
                                             #   of the prediction is now
                                             #   MEASURED and it holds:
                                             #   **max|g| = 2157 against a clip
                                             #   of 1.0, and the clip BINDS in
                                             #   5 of 5 epochs** -- three orders
                                             #   above the smoke net, where it
                                             #   never engages. So the magnitude
                                             #   knobs SHOULD read INERT, which
                                             #   is what the per-knob table
                                             #   decides. ⛔ **AND ALL NINE
                                             #   CAME BACK `LIVE`, WHICH IS THE
                                             #   SIDE md5 CANNOT SPEAK ON.**
                                             #   The tool's own footer gives
                                             #   the valid direction --
                                             #   identical hash = no effect --
                                             #   and the converse is 2(x2)'s
                                             #   trap exactly: `logit_adjust`
                                             #   is algebraically plain CE and
                                             #   still differs in 24/24. So the
                                             #   run closed ZERO directions and
                                             #   the knob sweep is still
                                             #   unjustified; clear a magnitude
                                             #   knob at the GRADIENT level,
                                             #   not the prediction level.
                                             #   🔑 The two columns that ARE
                                             #   measurements: `rho_target`
                                             #   100->10 drops max|g| 2157 ->
                                             #   143.1 and `lambda_step` raises
                                             #   it to 8724, yet the clip binds
                                             #   5/5 in EIGHT of nine, so all
                                             #   of them deliver the identical
                                             #   step size and only the
                                             #   DIRECTION can differ.
                                             #   `lr_constraint x10` is the
                                             #   lone exception at 4/5.
                                             #   FRAMEWORK 2(z89).
                                             #   So the liveness of the ONE scalar
                                             #   `normalize` does not cancel is itself
                                             #   unmeasured. Since the determinism fix
                                             #   its verdict is a HASH COMPARISON at
                                             #   n=1 per setting, five epochs per knob,
                                             #   so this is hours and it GATES the
                                             #   dose-response campaign.
                                             #   FRAMEWORK 2(z82) section 5, task #118.
python -m scripts.derive_dual_weights        # the receipt for FRAMEWORK 2's
                                             #   dual-weight table
python -m scripts.diagnose_run <run-dir>     # stage-by-stage read of ONE run's log
python -m scripts.reset_crashed <root>       # reset CRASHED runs for retry, and
                                             #   nothing else
python -m scripts.prep_isic ...              # candidate slice: held-out
                                             #   SUBPOPULATION, the first
                                             #   non-camera-trap ⇒ screen it with
                                             #   `factorial_control`, not
                                             #   `dataset_screen` alone
python -m scripts.prep_fmow ...              # candidate slice: held-out COUNTRY
```

## Datasets

⛔ **THREE ARE RUNNABLE AND ALL THREE HAVE RUN. THIS LINE SAID
"`iwildcam` is the only RUNNABLE one ... two more pass the screen" UNTIL
2026-09-11**, which framed bcn and fmow as CANDIDATES when `bcn1mn3` (228 runs)
and `fmow1` (304/304) were long complete, `bcn1vit` was COMPLETE TOO at
190/190, and the two of them
supply **3 of the 8 licensed units** -- D1, E1 and E2, the last of which is the
HEADLINE ViTB16. `fmow1` is also the cleanest campaign the project has ever run
(2(z88)).

⚠️ **AND iwildcam IS THE WEAKEST OF THE THREE, MEASURED.**
`tier_viability` 2026-09-11: density **0.27**, **2 of 8** classes usable, **50%**
of ceilings K=0, **72%** of test items in groups holding neither capped class --
verdict **WEAK**, against TIER-LIKE for bcn (0.89) and fmow (0.82). Every
surviving TraLO unit is on it; see 2(z105), and read that entry's p=0.357 before
drawing anything from the coincidence.

`iwildcam` itself: 8 species, classes 2 (impala) and 7 (cattle) capped,
`location` = camera trap, and the test cameras are held out ENTIRE. **No AIDER, no
EuroSAT, no others.**

`dermmnist`, `octmnist` and `tissuemnist` are REMOVED -- the rows below are the evidence
for why, not an offer to run them.

🔑 **Triage a candidate BEFORE downloading it** (`docs/FRAMEWORK.md` 2(n)): the GROUP's
definition decides. Groups built from an index, a randomisation or a balanced assay design
are dead by construction -- that is what killed octmnist (`synth_group = index % 3`) and it
is why **rxrx1 fails too**, despite 1,139 classes and real batch effects: every siRNA
appears in every experiment by design. **A dataset famous for DOMAIN SHIFT is not
automatically one with PER-GROUP LABEL SHIFT, and only the second is usable here.**

🟢 **`iwildcam` is the ONE that can carry a constraint**, and the other three are now
understood not to. Screen them with `scripts.dataset_screen`, which reports the
DIFFERENTIAL per-group novelty net of sampling noise and the global shift:

| dataset | group | NET items | z | unseen groups | status |
|---|---|---|---|---|---|
| **bcn/oodslice** | body site x age | **+2031** | **61.6** | **8** | 🟢🟢 **RUNNABLE. `bcn1mn3` COMPLETE (228 runs), `bcn1vit` COMPLETE (190/190) and UNLICENSED -- 2(z106).** ⚠️ **HALF ITS NOVELTY IS INTERPOLABLE -- 2(z104)** |
| **fmow/oodslice** | **country** | **+2793** | **76.2** | **13** | 🟢🟢 **RUNNABLE. `fmow1` COMPLETE 304/304 (2026-09-10), 3 of 4 cells are TASK -- and TraLO wins 0 of 4. FRAMEWORK 2(z88)** |
| **iwildcam/oodslice** | camera | **+3133** | **96.3** | **7** | 🟡 runnable, but a task in **0 of 24** cells at L20/L30/L50 |
| **terra/oodslice** | camera | **+2546** | **75.8** | **5** | 🟡 screened 2026-08-28, META ONLY |
| dermmnist/slice_1 | synth | +65 | 2.9 | 0 | ⛔ leaked + removed |
| octmnist/slice_1 | `index % 3` | -7 | -0.4 | 0 | ⛔ dead by construction |
| tissuemnist | `index % 3` | -56 | -1.9 | 0 | ⛔ dead by construction |

🛑 **THE bcn AND fmow ROWS WERE MEASURED 2026-09-11, AND THE fmow ROW
REPLACES ONE THAT DOES NOT REPRODUCE.** bcn had never been screened at all --
three dashes, on the slice carrying a COMPLETE 228-run campaign and licensing
unit D1, whose NEGATIVE sign is load-bearing for the acceptance tally. fmow read
`+2969 / 79.7 / 10 unseen`; the slice on disk gives **`+2793 / 76.2 / 13`**.
🔑 **THE GROUP COUNT IS THE DECISIVE FIELD -- BY CONTEMPORANEOUS
RECORD, NOT BY INDEPENDENT MEASUREMENT.** `configs/task_windows.yml` line 267
says `1 of 13`, `tier_viability` counts `grp 13`, and the test meta holds 13
countries over 4168 items with ZERO overlap against 136 train countries.
⛔ **ALL THREE READ THE SAME FILE** -- the yml's own header says "Counted
from the test labels alone" -- so they are CONSISTENT, not independent, and an
earlier draft of this note called them three authorities. ⚠️ The yml
block carries TWO provenances: its WINDOW rows come from `fmow1`'s own nulls,
its GROUP-COUNT table from labels. What defeats `10` is that the yml was written
while `fmow1` was in flight and revised at its completion, so it records which
slice was in play. ⛔ Whether the SERVER worktree holds the same slice is
task **#134**, one md5 per file.
✅ **THE TOOL IS NOT THE VARIABLE**: iwildcam re-reads **+3133 / 96.3 / 7**,
byte-identical to the row above it, and `tier_viability` reproduces `bcn_s1`
0.89 / `fmow_country_wide` 0.82 / iwildcam 0.27 exactly -- so the candidate-slice
names in section 0 and these `oodslice`s are the same data.

⚠️ **AND HALF OF bcn's NOVELTY IS INTERPOLABLE -- 50.4% SURVIVES,
A FIGURE MEASURED 2026-09-01 THAT NEVER REACHED THIS TABLE.** 2(w2c) lists
`bcn_s2` at 50.4 among the factorial rows that reproduce exactly; bcn's row here
stayed three dashes, `bcn1mn3` ran 228 runs, and unit D1 entered the ledger
anyway. Re-read here on the DEPLOYED slice. Its group is
`anterior torso|40s`, a PRODUCT of two factors that both appear in training, so
`dataset_screen`'s baseline (give an unseen group the global training
prevalence) is TOO GENEROUS: the model can interpolate site x age. Raked **8 of
8** unseen groups, against `raked=0` for iwildcam and every fmow -- which is why
this correction has never applied before and why the tool's `NOT A CONTROL`
banner exists. Measured: NET **+2033 global** vs **+1025 additive**, z 62.9 vs
27.3.
⛔ **IT STILL PASSES -- z=27.3 is not a null -- BUT READ THE ITEM COUNTS,
NOT THE RATIO**, which the tool says itself: 100% of bcn's test set is unseen, so
the global shift is computed largely from these very groups and corrects the
raked baseline twice. The defensible statement is that bcn carries roughly HALF
the per-group novelty its unadjusted row implies, and that no other slice in the
corpus has ever been asked this question.

✅ **THE fmow p@K NUMBER WAS GONE AND GOT, 2026-09-09 (FRAMEWORK 2(z59)).**
That question -- "fmow needs local p@K `<= 0.92` at L30, where iwildcam measures
0.9948-0.9972" -- is answered from `fmow1`'s own nulls: **0.842 / 0.882 / 0.952 /
0.973** over the four (backbone, class) pairs, so 2 of 4 clear the bar outright
and all four sit under the 0.99 WIGGLE ceiling. `configs/task_windows.yml` now
carries the measured windows and `gen_campaign` enforces them.
⛔ **AND THE GLOBAL COLUMN WOULD HAVE REJECTED THIS DATASET.** fmow's GLOBAL
p@K at L30 is 0.996-0.999, no better than iwildcam. The allocator is per-group,
so the LOCAL cut is the one that decides -- the fourth time that distinction has
changed an answer here.
🟡 **`terra` still has NO IMAGES.** Rebuild its meta in minutes on CPU with
`prep_iwildcam --annotations <cct.json> --meta-only`, then `dataset_screen`.
⚠️ Stage 1 is NECESSARY ONLY -- dermmnist passed it at z=2.9 and still nulled.

⚠️ **octmnist and tissuemnist are structurally dead** -- `synth_group` is
`np.arange(len(y)) % 3`, so their groups are i.i.d. draws from one distribution
and the local scope is empty **by construction**. Two of the original three
could never have tested the thing being tested. `data/dermmnist/shift_1` looks
better at LOCAL=160 but 110 of that is the global shift replicated across
groups; it has never been used and should not be.

🛑 On `iwildcam`, **7 of 14 per-group ceilings are K=0** ("predict none of this
species at this camera"). A zero ceiling binds regardless of sum slack, so the
LOCAL scope constrains the output at every cap level -- unlike dermmnist, where
`lp_fallback_used` was False with 0 candidates on all 52 runs. `gen_campaign`
now reads the real budgets and says so; do NOT trust the sum-arithmetic line
alone. See `docs/FRAMEWORK.md` section 2(n).

## Backbones

**`ViTB16` IS THE HEADLINE**, fixed a priori 2026-08-20 (FRAMEWORK 1-pre) so a win found on
another backbone cannot be promoted after the fact. `MobileNetV3`, `MobileNetV2`,
`RegNetY400MF` are the other three. **Nothing else** -- these are
exactly the four the paper claims. ShuffleNetV2 and the small CNNs were deleted; they appear in
no `.tex` file.

## Loss

```
L_total = L_ce + lambda_g * L_global + lambda_l * L_local
```

Rational saturation `E/(E+K)` plus bounded quadratic. Soft counts (differentiable) for the
gradient, hard counts (argmax) for verification; post-hoc adjustment closes the gap.
**KL is out of scope.** The `alpha_kl` key and the whole KL anchor are DELETED from the
pipeline -- there is no setting to get wrong. Same for the CE-saturation skip
(`enable_ce_skip`), the undershoot hinge, and the `bounded_only` penalty branch.

## Infrastructure

- **Never run experiments locally.** SSH `dsisco01` / `dsisco02`, `conda activate optloss`.
- 🛑 **NEVER touch `src/`, `configs/` or `main.py` on the SERVER while a campaign is
  running** -- not even a comment. `code_version` is a git hash, so any edit splits the
  campaign into two non-comparable halves and turns `check_parity`'s "every arm from one
  commit" red. Deploy after the last run, never during. `scripts/` is exempt and safe to
  update mid-flight: nothing under it is on `src.experiments.runner`'s import path, which
  is why the scorer and the offline probes can be iterated while runs land. **Check
  `git status --porcelain src/ configs/ main.py` on the server, not just `git status`** --
  a tree dirty only in `scripts/` is the normal working state and says nothing.
- 🛑 **PIN THE CAMPAIGN TREE AT THE COMMIT ITS CONFIGS WERE GENERATED FROM, AND
  STOP MOVING IT.** `code_version` is `git rev-parse HEAD` -- only the `-dirty`
  SUFFIX is scoped to `TRAINING_PATHS`. So **any commit at all, including a
  docs-only one, desynchronises a staged campaign**: the configs keep the old
  stamp while the runner would write the new one, and `check_parity` fails the
  campaign on a change that touched nothing the runner imports. Fast-forwarding
  a staged tree "to pick up the latest docs" is how that happens.
  Either regenerate the configs after moving HEAD (they must end up a single
  stamp -- check with
  `python -c "import glob,json;print({json.load(open(f))['code_version'] for f in glob.glob('<root>/*/*/*/*/*/config.json')})"`)
  or, better, leave the tree alone once staged. A `scripts/` update can still be
  copied in by hand: `scripts/` is outside `TRAINING_PATHS`, so it does not flip
  `-dirty` and does not move HEAD.
- 🛑 **A FRESH WORKTREE HAS NO DATA, AND EVERY LAUNCH GATE STILL READS GREEN.**
  The `.npy` arrays are gitignored (3.0 GB + 443 MB), so `git worktree add`
  produces a tree with only the tracked `*_meta.csv`. Measured 2026-09-06 on
  `optloss-dualprop`: `--step verify` and `--step launch` both GREEN, then 24
  runs failed in 120 seconds. Every one of the fourteen older worktrees had the
  arrays put in by hand at creation, so the failure mode had never occurred.
  ✅ `scripts.data_present` now runs inside `--step launch`. After
  `git worktree add`, link the arrays **at their real location** -- the ones in
  the sibling worktrees are themselves symlinks into `~/optloss-audit`, so
  linking worktree-to-worktree builds a chain that breaks when the middle one
  is removed:
  ```bash
  SRC=~/optloss-audit/data/iwildcam/oodslice
  DST=~/<new-worktree>/data/iwildcam/oodslice
  for f in $SRC/*.npy; do ln -s "$f" "$DST/$(basename $f)"; done
  ```
- 🛑 **THE CAMPAIGN CHECKOUT IS A WORKTREE, SO THE FREEZE COVERS GIT PLUMBING TOO.**
  `~/optloss-audit/.git` is a FILE, not a directory:
  `gitdir: /home/dsi/michaer8/OptimizationLoss/.git/worktrees/optloss-audit`.
  ⛔ **FOURTEEN worktrees share ONE object store** in `~/OptimizationLoss/.git`
  (this file said FOUR until 2026-09-01, and the freeze below is only as wide as
  the list it names). Counted with `git worktree list`:
  `OptimizationLoss` itself, `OL-replication` (**marked `prunable`**),
  `optloss-audit`, `optloss-cutwin`, `optloss-dom`, `optloss-domb`,
  `optloss-equaldose`, `optloss-iwc4`, `optloss-loose`, `optloss-loosevit`,
  `optloss-select`, `optloss-uniform`, `optloss-vitdom2`, `optloss-vitu`.
  🔑 **AND RESULTS ARE SCATTERED ACROSS THEM, NOT COLLECTED.** `loosevit1` --
  the only iwildcam ViTB16 campaign at loose caps, and the source of 2(z20) --
  sits in `optloss-loosevit`, which no doc listed, so nobody had scored it.
  Run `git worktree list` and inventory `*/results/` before concluding that a
  question has no data. So a command run
  in a SIBLING checkout can reach into the running campaign's git. While a campaign is
  running, **never run `git gc`, `git prune`, `git repack`, `git reflog expire` or
  `git worktree prune` anywhere in that family**, including in a checkout that looks
  unrelated. The file-level freeze above is necessary and NOT sufficient: it protects
  `src/ configs/ main.py`, and this protects the objects `code_version` resolves against.
  Branch-level work in a sibling worktree (fetch, checkout, reset of its OWN tree) is fine
  -- different branch, different working tree, no repack.
  `cd ~/OptimizationLoss && git worktree list` prints the whole family; run it before any
  git maintenance, because "I am in a different directory" is not isolation here.
- **Max 2 GPUs.** Run `nvidia-smi` **with owner lookup** first; never share a GPU with another user.
- dsisco01 = Quadro RTX 6000 (FP16 + GradScaler). dsisco02 = RTX PRO 6000 Blackwell (BF16 AMP).
  Record which one a result came from.
- Any hyperparameter that changes what warm-up optimizes **must** be in `compute_base_model_id`,
  or the second arm silently loads the first one's cached model.

## Paper

`docs/paper/main.tex` is the professor's file -- **never edit it**. Edit `docs/paper/main_edited_by_roei.tex`.
Appendix tables stay in the appendix.

🛑 **EVERY MANUSCRIPT IN `docs/paper/` IS ON A CORPUS THAT NO LONGER EXISTS.**
All five `.tex` files are `dermmnist` / `octmnist` / `tissuemnist`; **not one names
`iwildcam`**, and `docs/paper/data/` holds zero iwildcam rows. The paper and the
current experiments are **disjoint generations** -- a finding on one says nothing
about the other, either way. `docs/paper/WHICH_CORPUS.md` is the full statement,
including the manuscripts' OWN separate problems (warm-up 50, the dermmnist leak,
no lambda=0 control in the corpus). Read it before quoting a paper number, and
before saying "MedMNIST" in a sentence about current results.

**Four manuscripts sit in `docs/paper/`. `main_edited_by_roei.tex` is the paper of
record** -- it is the one to edit and the one to read a claim out of.

| File | What it is | Reads |
|---|---|---|
| `main_edited_by_roei.tex` | ✅ **the paper of record**, additions in blue | `tables/` + `tables_rev/` |
| `main.tex` | the professor's file. **Never edit** | `tables/` |
| `main_rev.tex` | the revision `main_edited_by_roei` was branched from | `tables/` + `tables_rev/` |
| `main_clean.tex` | a de-marked-up snapshot | `tables/` + `tables_clean/` |

Only the first two are live. A fix applied to either of the other two has no
effect on anything anyone reads. `main_old.tex` (pre-TMLR) was deleted
2026-09-02 along with 11 one-off `docs/launch_*.sh` wrappers, the dermmnist
`data/dynamics/` tree and two orphaned scouting notes -- 98 files, 10,228
lines. Git history is the archive.
⛔ **THAT LINE ENDED "none of them referenced by anything" AND IT WAS FALSE
(2026-09-11).** `docs/paper/data/dynamics/` -- 84 files, 4,332 lines, deleted in
`e7d9e893` -- is the input to `docs/paper/scripts/make_figs.py`, which emits
**`fig_mechanism`**, and `fig_mechanism` is `\includegraphics`'d by BOTH live
manuscripts, `main.tex` **and `main_edited_by_roei.tex`, the paper of record**.
The generator has been dying on `FileNotFoundError` ever since. Nothing looked
broken because the committed PDF is still there and the paper still builds: a
figure in the paper of record had silently become **unreproducible**, which is
the same shape as 2(z94) one level up -- the artefact survives, its provenance
does not.
✅ RESTORED with `git checkout e7d9e893^ -- docs/paper/data/dynamics`, and
`make_figs.py` runs again (`gates fio@58 tra@63; max lambda fio=53.40
tralo=0.180`, the 297x ratio the figure is about).
🔑 **THE RULE: "UNREFERENCED" MUST BE MEASURED, NOT ASSERTED, AND A DATA TREE
NEEDS A DIFFERENT SEARCH THAN A SCRIPT.** `dead_code` is AST over `configs src`
and would never see a `.csv` path built by `os.path.join` inside a figure
generator. The cheap check is to RUN every `docs/paper/scripts/make_*.py`
before deleting anything under `docs/paper/data/`: all eleven run in under a
minute and every table then regenerates byte-for-byte across `tables/`,
`tables_rev/`, `tables_clean/` and `tables_task/` -- verified 2026-09-11.
⚠️ **AND DO NOT COMMIT WHAT THAT CHECK REGENERATES.** Five of the six figure
PDFs come back byte-different from the committed ones (the known
non-reproducibility above); `git checkout -- docs/paper/figures` after running
them, or the check quietly rewrites the artefacts it was meant to verify.

**EIGHT of the eleven tables in `docs/paper/tables/` regenerate from
`docs/paper/data/corpus/corpus_final.csv` byte-for-byte** via
`docs/paper/scripts/make_*.py` -- run them and `git diff docs/paper/tables/` must
be empty. 🛑 **`make_main_table.py` needs `--two-metrics`**; the bare
invocation writes a DIFFERENT table over the same `tab_ccf1.tex` (verified
2026-08-21: bare = 54 insertions / 63 deletions, `--two-metrics` = byte-identical).
It is the one generator whose default is not the shipped artefact, so run:

```bash
python docs/paper/scripts/make_main_table.py --two-metrics   # tab_ccf1.tex
```

⚠️ `tab_ablation_complete`, `tab_deploy` and `tab_oct_backbone` have
**no generator and never did**, so an empty diff says nothing about those three.
⚠️ **Nor does it say anything about the FIGURES.** Verified 2026-08-25: of the six
PDFs under `docs/paper/figures/`, two regenerate byte-for-byte and **four do not**
(-888 to -4,617 bytes, all smaller). Not the data -- `make_loss_shape_fig.py` reads
no data file and still differs -- and not the toolchain, same matplotlib and font
subsets. The committed figures came from earlier versions of their generators. See `docs/paper/data/PROVENANCE.md`, including what the corpus itself
can no longer be rebuilt from.
