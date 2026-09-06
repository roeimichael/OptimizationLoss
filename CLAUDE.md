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
2. **Score at equal compute, with BOTH clippers (`clip` and `focal_clip`) inside the campaign.**
   `clip` is the stronger quality bar. An arm-vs-arm delta is not a result until the bar is in
   the same campaign.
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

## Where things are

```
main.py            dispatcher (kill -INT to stop; interrupted runs reset to pending)
configs/           gen_campaign.py = THE generator (asserts the protocol, refuses to
                   emit a single-cap campaign, always adds both clippers)
data/              iwildcam -- THE ONLY dataset. The original three are removed and
                   unrunnable, not merely discouraged; see `docs/FRAMEWORK.md` 2(n)
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
                   its own reseed floor? RECOUNT DONE 2026-09-04: the ledger
                   licenses FOUR units and only THREE carry a verified `task`
                   cell, so quote BOTH -- 4/4 p=0.0625 licensed, 3/3 p=0.125
                   task-restricted. `paper_rows` prints the restriction itself) and the
                   checklist of holes: ViTB16 has zero fioretto/hounie/alm,
                   every run caps the same 2 classes, no symmetric cap ever,
                   1 dataset of 3
docs/PLAYBOOK.md   WHAT TO DO WHEN A CAMPAIGN LANDS -- the integrity gates in
                   order, how to read the logs and their three traps, and a
                   branch per outcome (win / null / loss / gates red) decided
                   in advance. Read it BEFORE scoring, not after.
docs/archive/      history, not instructions
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
python -m pytest tests -q                   # 583 regression tests, ~250s, no dataset needed
#   `tests/test_scorers_run_end_to_end.py` EXECUTES every scorer as a subprocess
#   against a campaign carrying a real PARTIAL marker. It exists because three
#   scorers once used `quarantine.` with no module-level import: they PARSED,
#   imported, passed every AST gate and were unrunnable on every input, and the
#   NameError fired only on the branch that a quarantined campaign reaches --
#   the branch that exists to prevent a wrong number. 6/6 mutations caught.
#   `tests/test_lessons_learned.py` is the CATALOGUE OF LESSONS ALREADY PAID FOR:
#   rejected backbones and datasets with the measured reason each was dropped,
#   the ten deleted config footguns, the BF16/compute-capability split between
#   the two hosts, the oldest allocator bug (an argmax fallback that ignored the
#   cap), the local-scope mirror of it, and a sweep that RUNS all 22 `--self-test`
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
python -m scripts.audit_config              # no config key without a reader, no reader without a key
python -m scripts.smoke_arms                # every arm actually RUNS and respects its caps
python -m scripts.smoke_arms --matrix       # + {1,2} capped classes x {L30_G30, L50_G30},
                                            #   caps verified for the TRAINED arms too
python -m scripts.flag_live <armA> <armB>    # md5 across arms: is the new flag LIVE
                                            #   or a fifth inert one? (rule 3)
python -m scripts.verify_caps               # what integer budget each cap tag really produces
python -m scripts.check_parity <root>       # equal compute, same knobs, >=2 caps, sane warm-up sharing
python -m scripts.reachability <early-run>  # CAN the penalty even reach this cell's cut?
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
#   (38 cells): **SENSITIVE 0, UNDER-POWERED 36, SATURATED 2.** The typical
#   arm-pair difference is 2-5 deployed TP items and the RNG floor in the same
#   cell is 1.0-10.5. They are the same size.
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
#   ⚠️ |tralo - rival| median 4.0 items POOLED `alm`+`fioretto`+`hounie`
#   and must be recomputed against `alm` alone (n 180 -> ~60); the value is
#   not restated because it is not measured. The floor it was compared to,
#   |tralo - tralo_reseed| median 4.0, is unaffected. `--self-test` gates it.
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
#     lost 328 of 1044. full_panel refuses to compare arms more than 5
#     percentage points apart -- but only once you look.
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
#   **6 of 17 = 35%, bar 50%, VERDICT FAIL -- and 0 of 17 cells are priced**,
#   so every win is a direction and none is reportable. Per unit it is 2 of 6.
#   `--self-test` gates it in both directions, 7 checks, including that
#   beating the CONTROL but not the RIVAL is NOT a win (the old framing scored
#   that green) and that exactly 50% passes.
python -m scripts.cell_table --campaign <roots> --out cells.csv   # the SURVEY, not the
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
                                            #   1.9-9.9 item question, so every
                                            #   `NO DIFFERENCE` is an absence of
                                            #   measurement, not a null. FRAMEWORK 2(q)
python -m scripts.prep_iwildcam --annotations <cct.json>     --out data/<name>/oodslice --meta-only  # screen a CANDIDATE dataset with NO
                                            #   images and no GPU: any
                                            #   COCO-CameraTraps annotation file
                                            #   (iWildCam, Terra Incognita/CCT)
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
                                            #   K/n=90%**. 🛑 Pairing GROWS the noise here
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
                                            #   sd of 7.59 (**0.05x**). The 4th number is
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
                                            #   by the CAP CHOICE, not by physics. ⚠️ The
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
                                            #   `--self-test` gates it in BOTH directions.
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
                                            #   ⚠️ `--match-contested` is the ONLY
                                            #   ladder comparable ACROSS cap levels:
                                            #   the fraction-of-range one reversed a
                                            #   24/33 trend once density was held
                                            #   fixed. Aggregates key on the ARM too.
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
positives on both sides of the cut. It is a *reference* (it depends on n, K and prevalence
only, measured at 10.8 vs 11.6 items across two regimes whose error structures differ 5x),
and the SIGN of the deviation is the result: `reachable << ctrl` means the ranking already
took the easy swaps, `~= ctrl` means the statistic is reading the score distribution and
means nothing, and `>> ctrl` means positives are parked BELOW the cut -- the one case in
which a cut-local method has something real to win.

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
                                             #   and on ViTB16 both verdicts INVERT.
                                             #   A knob sweep justified by the smoke
                                             #   net sweeps cancelled quantities.
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

**`iwildcam` is the only RUNNABLE one** (the only one with images on the server; two more pass the screen -- see the table). 8 species, classes 2 (impala) and 7 (cattle) capped,
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
| **iwildcam/oodslice** | camera | **+3133** | **96.3** | **7** | 🟢 RUNNABLE, images on the server |
| **fmow/oodslice** | **country** | **+2969** | **79.7** | **10** | 🟡 screened 2026-08-28, META ONLY |
| **terra/oodslice** | camera | **+2546** | **75.8** | **5** | 🟡 screened 2026-08-28, META ONLY |
| dermmnist/slice_1 | synth | +65 | 2.9 | 0 | ⛔ leaked + removed |
| octmnist/slice_1 | `index % 3` | -7 | -0.4 | 0 | ⛔ dead by construction |
| tissuemnist | `index % 3` | -56 | -1.9 | 0 | ⛔ dead by construction |

🟡 **`fmow` and `terra` PASS stage 1 but have NO IMAGES yet.** Rebuild their
meta in minutes on CPU with `prep_fmow --meta-only` / `prep_iwildcam
--annotations <cct.json> --meta-only`, then `dataset_screen`. ⚠️ Stage 1 is
NECESSARY ONLY -- dermmnist passed it at z=2.9 and still nulled. And their
PRIZE is unmeasured: `ceiling_screen` prices them off **iwildcam's** p@K curve,
which it says does not transfer. The number to go and get is fmow's real p@K:
**it needs only `<= 0.92` at L30 to clear twice the noise, where iwildcam
measures 0.9948-0.9972.** See `docs/FRAMEWORK.md` 2(w2).

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
lines, none of them referenced by anything. Git history is the archive.

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
