# MISSION -- make TraLO mathematically the best, and prove it

**This file is the resume point.** A fresh session reads it first, then
`docs/FRAMEWORK.md` section 3(0) (the status board). It is updated at the end of
every working session. If it is stale, that is a defect -- fix it before doing
anything else.

Last updated: **2026-09-10** (🔴🔴🔴🔴 **`fmow1` LANDED AT 304/304 AND IS THE CLEANEST TEST THE PROJECT HAS EVER RUN -- TraLO WINS 0 OF ITS 4 CELLS.** Third dataset (fmow, held-out COUNTRY), the HEADLINE backbone (ViTB16) plus MobileNetV3, all four duals at **EQUAL 29.00 attempted steps/run**, **THREE lambda=0 streams** so the floor rests on **12 observations** and clears `MIN_FLOOR_OBS`=8, predictions intact, and **3 of 4 cells are TASK cells exactly as the windows re-measured this morning predicted**. Every objection ever raised against an earlier result is closed by this campaign at once. Result: `tralo` is **NEGATIVE vs `clip` in all three task cells** and **LAST of the four duals in all four**, and it **LOSES THE ONE PRICED CELL by -10.00 items against a 6.5-item floor**. The corpus-wide priced record is now **1 win, 1 loss**. ⇒ **THE ACCEPTANCE BAR IS 6 of 22 = 27%, per unit 2 of 8, VERDICT FAIL** (was 6 of 18 = 33% over 6 units). FRAMEWORK **2(z88)**. 🔴 **AND IN cc-F1 -- THE OFFICIAL METRIC -- `tralo_reseed2` BEATS `tralo` IN ALL THREE fmow TASK CELLS**: that arm is the lambda=0 twin with a different RNG offset and nothing else. **The first `vs clip` contrast in the whole corpus that RESOLVES appeared today and it is a LOSS** (-0.0132 cc-F1, paired sd 0.0061, 0/4 seeds). Over all 13 task cells only **4 of 50** contrasts resolve; three say the constraint changes the model and the fourth is the only one comparing it to an ALTERNATIVE. FRAMEWORK **2(z87)** carries the paper tables -- `docs/paper/scripts/make_task_cell_table.py` -> `docs/paper/tables_task/`, 13 cells / 6 units in cc-F1 and macro-F1, gated by 10 checks and mutation-tested 5/5. In cc-F1 **1 of 11 testable cells has an unambiguous leader and it is `alm`**; in macro-F1 **zero**, and the lambda=0 arm leads 8 of 10 on iwildcam+bcn. 🟢 dsisco02 GPU 2 is FREE (fmow1 finished); `price1` 39/80 and `price2` 16/80 still running on dsisco01 GPU 0/1. ⛔ All three stalled ViT campaigns -- `vitdual2`, `vitseed1`, `vitcoin1` -- are **dsisco01/fp16** by their dispatcher logs and may NOT be finished on dsisco02. | PREVIOUS: 🔴🔴🔴 **D1 WAS READ AND IT REFUTES THE HEADLINE. THE SIGN TEST CLEARS 0.05 UNDER NO AGGREGATION RULE.** The queue promised that reading units C2 and D1 would give `6/6 p=0.0156` unrestricted and `4/4 p=0.0625` task-restricted. Both were read at zero GPU cost on 2026-09-10 and the measurement is **5/6 p=0.109** at best (unrestricted, mean rule), **3/4 p=0.3125** task-restricted, and **3/6 p=0.656** under the worst-cell rule. `bcn1mn3`/MobileNetV3 is unit **D1** and it is the FIRST licensed unit whose `tralo` does not clear its own RNG floor: **+5.16 items at L80 and -11.61 at L90**, and -2.71 against its own null. C1 and C2 are SPLIT too once every cell is read, so `6/6` failed on THREE units rather than one, and A2's worst cell is **+0.02 items** -- a tie wearing a plus sign. FRAMEWORK **2(z86)**, task #102. 🔴 **AND THE ACCEPTANCE BAR IS RECOMPUTED AND STILL FAILS: 6 of 18 = 33%**, per unit 2 of 6, bar 50%. The stale `6 of 17 = 35%` predated `rank_cell`'s common-seeds fix by 14 hours and 2(z68) said its direction could not be assumed -- measured, the fix was worth ZERO cells. Task #104 discharged. 🟢 **1 of 18 cells is PRICED, the first ever, and `tralo` wins it** -- every previous run read `0 of N`, false by construction because two lambda=0 streams over 4 seeds give 4 observations against `MIN_FLOOR_OBS`=8 (2(z69)). ⛔ **THE SHARPEST NUMBER IS NOT ABOUT THE CAP: `tralo_coin_sgd`, a RANDOM constraint direction of the same norm, OUTRANKS stock TraLO in BOTH of D1's task cells** (2nd vs 3rd at L80, 1st vs 10th at L90), and `tralo_lam0` -- lambda switched off -- recovers +14.74 of TraLO's +27.94 vs `clip`. 2(z29) and 2(z56) replicated on a SECOND dataset and backbone. ⇒ Per the standing bar this is the trigger to change the METHOD, not to run more seeds of it. 🟢 **SSH IS BACK AND THE RIG IS FULL**: `price1` and `price2` relaunched on dsisco01 GPU 0/1 at 15:11 (both were stalled with dead `running` statuses), `fmow1` at 300/304 on dsisco02 GPU 2, `vitdual2` queued for the next free slot. `price1`'s dose is EQUAL at 29.00 attempted steps/run across every trained arm. **`bcn1vit` and `snap2` are COMPLETE and both docs still called them LIVE**; `price2`, `seed58a`, `bcnpilot1/2` and `fmowpilot1/2` exist on disk and NO doc named them. See 0-RUNNING, now a verified CENSUS. Also: an ELEVENTH global-top-K site, `straddle_probe`, found by asking what the call-site gate could not see -- `np.partition` was not in its target list, 2(z85).)

---

## 🔧 0-DOC. THE DOCUMENTATION REVIEW, AND THE ONE THING IN IT THAT CHANGES A NUMBER (2026-09-11)

**READ THIS BEFORE RUNNING `tralo_wins` AGAIN -- ITS OUTPUT HAS A NEW SECTION
AND THE DENOMINATOR MAY MOVE.** FRAMEWORK 2(z101).

Three conditions dropped a cell before the acceptance table could score it --
no `tralo`, no control, or `rank_cell` finding the two share no common seed --
and all three were a bare `continue`. The summary printed `CELLS THAT CAN TEST
THE CLAIM` and `hold no rival`, so a reader took those two counts for the whole
input. Each drop is now printed with its reason, before the verdict.

⛔ **WHETHER ANY LIVE CELL IS ACTUALLY DROPPED IS UNMEASURED.** Both hosts
have refused SSH all of 2026-09-11 (`Connection timed out during banner
exchange`), so the tool has not run against `results/` since the fix. The
standing acceptance figure is **6 of 22 = 27%, per unit 2 of 8, VERDICT FAIL**.
The derived-with-`price1` figure of 6 of 24 = 25% is **DERIVED, NOT RUN** --
task #128 -- and must not be quoted as measured.

🔑 **WHAT TO DO FIRST ON RECONNECT.** Run the recount and READ THE NEW
BLOCK before the verdict:

```bash
# The SCORABLE roots only. `uniform1`, `vittask1` and `vitdual1` are
# `scorable=False` in `quarantine.REGISTRY`, and `tralo_wins` gates on it --
# naming any of them makes the whole command exit 1 before it scores anything.
# `dom1` / `dom1b` / `equaldose1` are PARTIAL (fioretto + hounie dead, and
# `tralo_lam0` in equaldose1); the gate drops those arms and keeps the rest.
python -m scripts.tralo_wins --control clip --campaign \
    results/dom1 results/dom1b results/equaldose1 results/taskwin2 \
    results/price1 results/vitdual2
```

⚠️ **AND THE ROOTS ARE SCATTERED ACROSS WORKTREES**, so a bare
`results/<name>` resolves only inside the worktree that holds that campaign.
Inventory first -- `git worktree list`, then `ls */results/` -- and pass the
real paths. `loosevit1` sat unscored for weeks because no doc listed the
worktree it was in.

If it names dropped cells, the 22 in "6 of 22" was never the whole input and
the acceptance bar has been computed on a denominator nobody could check. If it
names none, that is a result worth having rather than an assumption worth
keeping -- record which it was.

### The rest of the review, none of which moves a number

* **MIN_PRIZE was the one criterion of three that `test_g2_budget` did not
  mirror** against `configs/task_windows.yml`, and it is `sensitivity_screen`'s
  BAND bar as well as the task-window PRIZE bar. Mirrored, mutation-tested 3/3.
  FRAMEWORK 2(z100).
* **14 of 24 archived docs read as live instructions**, the SUPERSEDED rejected
  ledger and four EXECUTABLE campaign launchers among them. All 24 bannered,
  gated in the first five lines, mutation-tested 2/2. FRAMEWORK 2(z102).
* **Two measured NULLS, recorded so nobody builds them**: a key-reader audit
  for `task_windows.yml` is 1 real in 21 (the rest are receipts the yml itself
  documents as receipts), and a dead-doc-path audit is 9 hits and 0 defects.
  Neither gate is buildable at acceptable noise. FRAMEWORK 2(z100) sections 2
  and 4.
* **Verified clean, by EXECUTION not grep**: `stream_family` excludes `_lam0`
  and the treated arm from the RNG floor; `MIN_FLOOR_OBS` is 8; `MEASURED_UNITS`
  is 13 entries over 8 units; `full_panel`'s RESOLUTION block refuses when no
  cell has two seeds and takes `min(ns)` not the median; both dose statistics
  print, with the "BOTH may read 100%" warning.

## 🔴 0-PRICE. `price1` LANDED AND THE NOISE TEST FINALLY RAN. TraLO LOSES BOTH CELLS (2026-09-11)

`price1` completed 80/80 at 02:53. MobileNetV2, `L70-70_G95` + `L80-80_G95`,
**equal dose at 232/232 = 29.00 attempted steps/run on every trained arm**,
predictions intact, all three step gates GREEN. THREE lambda=0 streams, so the
floor rests on **12 observations** and clears `MIN_FLOOR_OBS`=8.

```
cap            sds   tralo    alm    floor  priced  verdict
L70-70_G95       4   -2.25  -0.75   6.0(12)    no     loss
L80-80_G95       4   +2.25  +3.00   6.0(12)    no     loss
```

* 🔑 **`priced = no` IS A MEASUREMENT HERE, FOR THE FIRST TIME.** 2(z69): every
  earlier `0 of N priced` was False BY CONSTRUCTION, because `nfloor >= 8` gates
  the comparison and two streams over 4 seeds give 4. This one reaches the third
  clause, and the spread (2.25, 3.00 items) is well under the floor (6.0).
  ⚠️ ONE unit, TWO cells -- it does not make the rest of the corpus's `priced`
  column retroactively meaningful. It removes the excuse that nobody ran it.
* `sensitivity_screen`: **NOT ONE CELL IS SENSITIVE** -- 1 SATURATED, 3
  genuinely UNDER-POWERED (the floor IS well estimated, so this is not 2(z70)'s
  misnamed branch). 18-27 seeds/cell needed against 4 present.
* 🔑 The SATURATED cell is a **CUT-PLACEMENT** result, not a saturated model:
  `p(1-p)` is 0.00355 at the cut and **0.24980 at the decision boundary**. And
  `tralo_null` train acc runs 0.9581 -> 0.9997 over the constraint phase, so
  warm-up 1 is doing its job.
* 🛑 **IT BUYS NO UNIT. IT IS A2, BYTE-IDENTICALLY** -- `tralo_null` matches
  `equaldose1` and `coin2` in 4 of 4 seeds (e7be738bc8, 7758aef831, d77a1c47be,
  0cf8acc779) while `dom1`'s MobileNetV2 differs at every seed. Entered in
  `MEASURED_UNITS`, which **costs TraLO a unit**: A2 was `2 of 3 cells TRALO`
  and folding these in makes it `2 of 5` -> `rival`. A ledger that only admits
  a replicate when it agrees is not a ledger. FRAMEWORK 2(z97).
* ⚠️ **DERIVED, NOT YET RUN**: corpus-wide this takes **6 of 22 = 27%, per unit
  2 of 8** to **6 of 24 = 25%, per unit 1 of 8**. Both halves are arithmetic on
  cells `tralo_wins` printed, but the recount itself did NOT execute -- the jump
  host went down mid-session. **Quote it as a derivation until the tool prints
  it.** Task #128.

---

## 🟢 0-CAPS. THE FOUR NEW UNITS' CAPS ARE SETTLED, OFFLINE, AND THEY ARE STRICT TASK CELLS ON BOTH LEVELS (2026-09-10)

Task #124's measurement half is done and so is its cap choice. All twelve
(dataset x backbone) windows are now measured, and `configs.task_cells.classify`
runs on bcn and fmow **on a laptop with no GPU** -- the `*_meta.csv` splits are
tracked as of 2(z96), and K comes from labels and the cap policy alone.

| unit | caps | verdict at BOTH levels |
|---|---|---|
| `bcn` / MobileNetV2 | `L90_G95` `L100_G95` | task, both classes STRICT |
| `bcn` / RegNetY400MF | `L90_G95` `L100_G95` | task, both classes STRICT |
| `fmow` / MobileNetV2 | `L40_G95` `L50_G95` | task, both classes STRICT |
| `fmow` / RegNetY400MF | `L40-30_G95` `L50-40_G95` | task, both classes STRICT |

* ⚠️ **fmow/RegNetY400MF NEEDS THE PER-CLASS FORM.** Its corrected class-5 band
  is `[0.30, 0.40]` against class 3's `[0.40, 0.60]`, so the single-fraction
  overlap is ONE grid point and a two-level campaign is impossible without it.
  `L50_G95` reads `partial` there (class 5 slack) and `L60_G95` reads
  `non_task`. bcn and fmow/MNv2 do NOT need it.
* ⚠️ **`L110_G95` IS `L100_G95` ON bcn.** Both give K=1210 / 826, because
  `G95` binds globally at 0.95 before the local cap does. Two cap levels means
  L90 + L100; adding L110 buys a duplicate campaign.
* 🔑 **THE CORRECTION IS VISIBLY LOAD-BEARING AT THESE EDGES.** `bcn` `L80_G95`
  and `fmow` `L30_G95` both read `ref_shifted -> unmeasured`: inside the
  measured band, outside the one corrected for the `clip` reference arm
  (2(z95)). Before the correction had a reader they would have generated and
  classified `task`. The chosen caps sit one grid step clear of that edge.
* ⛔ **`L20`-`L60` ON bcn ARE NON-TASKS AND `L80`-`L110` ON fmow ARE**, so the
  two datasets' task regions do not overlap at all. Never carry a cap tag
  across datasets.

NEXT: generate on the server (the `.npy` arrays are gitignored, so `classify`
runs here but training cannot). Fresh worktree pinned at the commit, arrays
symlinked at their REAL location, `run_campaign --step stage/verify/launch`,
then dsisco01 GPU 2 -- CLEAR as of 23:40, with `price1`/`price2` on GPU 0/1 and
GPU 3 owned by `nirgal`.

---

## 🛑 0-FLOOR. THREE SCORERS REPORTED A GUARD AS IF IT WERE A MEASUREMENT (2026-09-10)

The floor bar `MIN_FLOOR_OBS = 8` is consulted by several tools. In three of
them the guard firing was reported in language that reads as a result. All
three are fixed or documented; the RULE is the point.

| tool | what it said | what it meant |
|---|---|---|
| `tralo_wins` | `0 of 17 priced` | `nfloor = 4 < 8`, so the spread was **never compared** to the floor. False by construction at any effect size. 2(z69) |
| `sensitivity_screen` | `UNDER-POWERED 36` | all 36 tripped the FLOOR branch, not the spread branch. Now `FLOOR UNMEASURED`. 2(z70) |
| `deployed_h2h` | "Naming a #1 here names the RNG" | tested `spread <= floor` BEFORE validating the floor, so it asserted the effect was inside a noise level it was about to call unpriced. Order fixed. 2(z70) |

🛑 **THE STANDING CHECK: wherever `MIN_FLOOR_OBS` is consulted, a reader
must be able to tell "the bar was not met" from "the test came out negative".**
The sweep table of every consumer is in 2(z70) -- start the next audit from it
rather than from a guess. `quarantine` and `add_seeds` mention the constant in
prose only and emit no verdict.

⚠️ **NONE OF THIS MAKES TraLO LOOK BETTER.** No ranking changed, and the
acceptance verdict (FAIL) is untouched -- the win count never reads `nfloor`.
What changed is that the corpus is now known to be **SILENT** on the noise
question rather than negative on it, which is what tasks #104 and #105 exist
to fix. 2(z39) further measured that a properly estimated floor comes out
HIGHER (4.0 -> 6.5 items), so expect these tools to refuse MORE often, not
less.

## 🛑 0-STALE. THE NUMBER ANSWERING OUR OWN BAR IS NOT REPRODUCIBLE (2026-09-10)

`tralo_wins` is the command that answers the acceptance bar. Its figure --
⛔ **RECOMPUTED 2026-09-10 WITH D1 ADDED: 6 of 18 = 33%, VERDICT FAIL UNCHANGED, and 1 of 18 is now PRICED -- the first ever. 2(z86), task #104 discharged. Superseded:** **6 of 17 = 35%, FAIL, 0 of 17 priced** -- was written at **09:09 on
2026-09-06**. `deployed_h2h.rank_cell`, which produces the deltas BOTH halves
of that verdict read, was fixed at **22:48 the same day**. Nothing was red:
the figure lives in a doc, the fix lives in git.

**Only ONE of the three same-day fixes bears on it**, and sorting that out is
the point -- my first reading was that all three pushed the same way:

* range -> pairwise margin: strictly stricter, so `0 of 17 priced` is already
  at the floor and cannot have improved. **That half is SAFE.**
* de-whitelisting: PRICING only. `present = [r for r in RIVALS if r in d]`
  reads a FIXED list, so more arms in `d` never changes which are rivals.
  **Zero effect.**
* common-seeds (22:48): changes `d` itself. **This is the live one.**
  Direction unknown -- 2(z50) says it is not a bias, but the one cell ever
  examined under it moved `tralo` +0.50 mid-pack -> **-8.0 LAST**.

⛔ **QUOTE THE VERDICT, NOT THE FIGURE.** FAIL needs 6/17 -> 9/17 to flip,
from a fix with no demonstrated positive direction in any cell. Say
"FAIL, figure pending recompute".

**THE COMMAND, blocked on SSH (`results/` is not local):**

```bash
python -m scripts.tralo_wins --campaign results/dom1 results/dom1b     results/equaldose1 results/taskwin2 results/vitdual2 --control clip
```

🔑 **AND THE CLASS IS NOW INSTRUMENTED.** `scripts/stale_figures.py`
holds every date-stamped figure in the docs against the last commit to the
scorer that produced it. First real run: **16 stale, 5 fresh**, and SEVEN of
the sixteen were `paper_rows` -- which found the LICENSED-vs-SIGN-READ
collapse still live in three places the 2(z66) recount never reached. Run it
before quoting any number:

```bash
python -m scripts.stale_figures
```

It is a REPORT, not a gate: a docstring commit moves the date and changes no
number, so it prints the commit SUBJECT and a hit means UNVERIFIED, never
WRONG. It also prints the 55 figures it could NOT attribute, because a tool
that silently examines a quarter of its input reads like one that found
nothing. FRAMEWORK 2(z68), task #104.

## 🔑 0-UNREAD. THE FIFTH AND SIXTH UNITS EXIST. READ THEM FIRST. (2026-09-09)

**This is the top of the queue and it needs no GPU.**

The headline is a sign test over independent units, and `0.5^4 = 0.0625` means
four units cannot reach p<0.05 at any effect size. That has been read for a
week as "we need a fifth campaign". **`scripts.paper_rows.MEASURED_UNITS`
licenses SIX.** ⛔ **EIGHT since 2026-09-10 -- `fmow1` licenses E1 and E2,
and both read 0 of 2. FRAMEWORK 2(z88).**

| unit | campaign / backbone | sign |
|---|---|---|
| A1 | `dom1` / MobileNetV2 | READ |
| A2 | `equaldose1` / MobileNetV2 | READ |
| B1 | `dom1b` / RegNetY400MF | READ |
| C1 | `taskwin2` / MobileNetV3 | READ |
| **C2** | **`dom1` / MobileNetV3** | ⚠️ **LICENSED, UNREAD** |
| **D1** | **`bcn1mn3` / MobileNetV3** | ⚠️ **LICENSED 2026-09-09, UNREAD** |

The `4/4, p=0.0625` line in `CLAUDE.md` and `docs/COVERAGE.md` is the
2026-09-04 recount, taken when the ledger held FOUR. Two units have been added
since and **the documents collapse "licensed" and "sign read" into one
number.**

```bash
# 0. THE GATES FIRST. A copy-paste sequence that skips them is how a wrong
#    number gets quoted, and both scorers below are quarantine-gated anyway --
#    better to see the refusal here than five frames deep in `sorted()`.
python -m scripts.pred_integrity results/dom1 results/bcn1mn3   # a TORN CSV PARSES
python -m scripts.dose_landed results/bcn1mn3                   # read attempted/run
python -m scripts.run_campaign --root results/bcn1mn3 --step score

# 1. THE READ. `cell_table` REFUSES a cell whose runs carry no
#    `hyperparams.seed`; `paper_rows` prints the task-window restriction itself.
python -m scripts.cell_table --campaign results/dom1 results/bcn1mn3 --out cells.csv
python -m scripts.paper_rows --cells cells.csv --out paper_rows.csv
```

* both signs positive -> ⛔ **REFUTED BY MEASUREMENT 2026-09-10 -- 2(z86). D1 WAS READ AND IS NEGATIVE.** The tally is **5/6 p=0.109** (unrestricted, mean rule) at best and **3/4 p=0.3125** task-restricted; the `worst-cell` rule gives 3/6 p=0.656. C1 and C2 are SPLIT too. Nothing clears 0.05.** **6/6 p=0.0156 unrestricted, 4/4 p=0.0625
  task-restricted**. 🛑 **THOSE ARE DIFFERENT TALLIES AND THIS LINE USED TO
  QUOTE ONLY THE FIRST (2026-09-10).** iwildcam/MobileNetV3's strict band for
  class 2 is measured EMPTY, so **C2 reads `partial` at every cap on the grid
  and can never carry a strict `task` cell** -- and `paper_rows` restricts its
  printed sign test to `task` units. The sub-0.05 number is the one the
  paper-facing scorer does NOT print. Read **D1 first**: `bcn1mn3` L80/L90 are
  verified task cells (2(z58)), so it is the one unit that moves both rows.
  FRAMEWORK 2(z75)
* either negative -> **the headline is refuted for minutes of compute**, which
  is worth MORE than the first outcome

⛔ **DO NOT QUOTE 6/6 BEFORE IT IS READ.** The ledger licenses a unit; it
does not supply its sign. Until then the honest line is *4/4 read, p=0.0625,
two licensed units unread*.

### D1 was found by asking why a COMPLETE campaign contributed nothing

`bcn1mn3` is finished -- 228 runs, 4 seeds, `L80` and `L90` are verified task
cells -- and was simply absent from the ledger. An absent entry reads
`UNVERIFIED`, which is the correct cautious default; the cost of that default
is that a finished campaign vanishes from every tally, silently. Third
instance of the class (`add_seeds` pooling, `shape1`'s third stream).

🔑 Its independence is **proved, not sampled**. Every other entry needed an
md5 because two iwildcam campaigns really can share a warm-up.
`compute_base_model_id` returns `model_dataset_hash`, so a bcn id begins
`MobileNetV3_bcn_` and cannot collide with `MobileNetV3_iwildcam_`. Gated in
`tests/test_lessons_learned.py`, mutations 2/2, with the negative control that
the SAME dataset must still collide.

⚠️ **The unit count is a claim.** 5 -> 6 moves the attainable floor
0.03125 -> 0.01563, so `tests/test_unit_ledger_matches_docs.py` hardcodes it:
growing the ledger is a decision somebody records, never a side effect.
FRAMEWORK 2(z66).


---

## 🔑 0-CLIP. THE ONE SCALAR `normalize` DOES NOT CANCEL HAS NEVER BEEN MOVED (pre-registered 2026-09-10)

**This is the top SCIENCE item in the queue.** Everything above it is
bookkeeping on numbers already taken.

⛔ The closure this disputes (2(b), 2(b-post)) is `dermmnist` -- removed, and
leaking 38.7% -- and its read-out campaign `dosefix` is `scorable=False`. Those
figures appear below as the thing being QUESTIONED, never as support.

Under `normalize` the delivered constraint step has norm exactly
`lr_constraint * constraint_grad_clip`. That is why `lambda_step`,
`initial_rho`, `rho_target` and `fioretto_step_size` are all inert here -- and
it makes `constraint_grad_clip` **the single scalar that survives the
normaliser**. FRAMEWORK 1a: *"the clip is load-bearing (measured); the VALUE is
not. `protocol.yml` itself says 'Sweep 0.3 / 1.0 / 3.0' -- that sweep has never
run."*

⛔ **2(b) LOOKS LIKE IT CLOSES THIS AND DOES NOT.** Its four bullets change the
step COUNT, the DELIVERY RULE, the OBJECTIVE and the PENALTY TERM. Not one
moves the scalar, and all four are `dermmnist` with the `p(1-p)` count -- the
count 2(r) showed evicts the CORRECT items. 2(w)'s separate refusal names
`lambda` (inert here) and step count (breaks equal compute); neither applies.

⚠️ **The real counter-argument:** the dedicated constraint optimizer recovered
~10x more constraint gradient and cost AP -0.0938, p=0.0006. That is why the
criteria are pre-registered rather than the grid just being launched.

### The order, hours before days

1. **#118 `hp_liveness_real`** -- it has NEVER been run, and CLAUDE.md stated
   its PREDICTION as a measurement until today. Five epochs per knob; since the
   determinism fix the verdict is a HASH COMPARISON at n=1. If the clip reads
   INERT on a real backbone, step 4 is three bit-identical arms.
   ```bash
   python -m scripts.hp_liveness_real <a completed dom1 tralo run dir> --epochs 5
   ```
2. 🔑 **READ `price1` -- THE SAME DIFFERENCE-IN-DIFFERENCES IS ALREADY
   PRE-REGISTERED ON THE ADJACENT AXIS, AND IT MAY ALREADY HAVE RUN.**
   `protocol.yml:511-516`: *"the claim is that `tralo_sgd` - `tralo_coin_sgd`
   EXCEEDS `tralo` - `tralo_coin`"* -- the identical estimator, varying the
   DELIVERY RULE instead of the scalar. All four arms exist and `price1` is
   the campaign staged to run it. ⛔ **Its state is recorded in no file
   anywhere** (2(z83)); task #78 says COMPLETED. Zero GPU-hours if it ran.
   ```bash
   python -m scripts.dose_landed results/price1
   python -m scripts.deployed_h2h --campaign results/price1 --control clip
   ```
   ⚠️ Read it as a DiD, never as `tralo_sgd > tralo`: 2(z46) measured `sgd` at
   **1/89th** the constraint-aligned displacement, so a bare null there is a
   DOSE null. That confound is precisely what step 4 exists to break.
3. **#116 can dose bind at all?** A larger clip buys nothing if the count is
   already satisfied most epochs. `dosefix` says 0 of 29 epochs satisfied at
   2.5-3.5x budget, but that is dermmnist. Re-read on `dom1`:
   ```bash
   python -m scripts.penalty_starvation --glob 'results/dom1/*/iwildcam/*/tralo/seed_*'
   python -m scripts.latch_probe --campaign results/dom1 --arms tralo tralo_uniform
   ```
4. **#117 `clipsweep1`** -- iwildcam x {**ViTB16**, MobileNetV2} x {`L80_G95`,
   `L95_G80`} x clip {0.3, 1.0, 3.0} x 4 seeds, **`tralo` AND `tralo_coin` at
   every clip**, plus `clip` `focal_clip` `alm` and THREE lambda=0 streams so
   the floor rests on 12 observations and `priced` can be True at all (2(z69)).
   🔑 **THE MEASURED QUANTITY IS `|tralo - tralo_coin|` / FLOOR AS A
   FUNCTION OF THE CLIP, NOT `tralo - clip`.** 2(z29) measured that at clip 1.0,
   in task cells, on this recipe, at equal dose, a coin flip of the same norm
   is INDISTINGUISHABLE from the penalty -- 2.0 items, 1.00x the floor. So the
   question is not "does a bigger step capture more" but **"does the DIRECTION
   acquire information as the step grows"**, and 2(z29) tested exactly one
   point on that axis. `tralo_coin` holds the norm, schedule, dose, RNG and
   model fixed and varies only what the direction knows.
   ⛔ **FLAT AT 1.00x ACROSS 10x CLOSES THE WHOLE LOSS-DESIGN PROGRAM** --
   count function, penalty shape, scope weighting, dual rule -- because
   `normalize` keeps only the direction. That is the stronger of the two
   outcomes and the one this project has been circling for a month.
   ⛔ **THE CELLS WERE ASKED OF `configs.task_cells.classify`, NOT ASSUMED,
   AND THE FIRST DESIGN DIED OF IT.** It named MobileNetV2 + MobileNetV3 at
   `L80_G95` / `L90_G95` -- four cells, **ONE** strict task cell. MobileNetV3
   is `partial` at every cap (2(z75): its strict class-2 band is measured
   EMPTY) and `L90_G95` is `partial` on every backbone but ViTB16. The design
   above is 4 cells and **4 strict task cells**, on the headline backbone plus
   one CNN, and `L95_G80` is the `G < L` shape where the GLOBAL scope binds --
   so the two caps differ in SCOPE, not only in tightness.

**PASS / FAIL / KILL are fixed in FRAMEWORK 2(z82) section 4 and must not be
renegotiated after the numbers.** In one line: a >= 6-item span across the three
clips in at least half the task cells, the winner clearing its own floor.

🔑 **BOTH OUTCOMES ARE WORTH THE HOURS.** FLAT retires the dose axis on live
data over a 10x range -- a far stronger closure than 2(b)'s four mechanism
substitutions on a removed dataset. MONOTONE gives TraLO a live knob no rival
has under `normalize`, in a comparison that has been untuned-against-untuned.

**No new code.** Arms are block-composed (`protocol.yml:819`), so a one-key
`constraint_grad_clip` block composes like `tralo_uniform` does. Nothing in
`src/`.

## 🔧 0-INSTR. THE TWO AUDITS, AND WHAT IS STILL UNMEASURED (2026-09-09)

**Nothing here changes a published number.** Real configs and real prediction
files always carried the fields involved. What was wrong is that the tools and
the TESTS did not agree with the pipeline, so several code paths had never
been exercised and two claims were wrong.

### The `argsort` audit -- what does a tool sort probabilities WITHIN?

🛑 **RECOUNTED 2026-09-10, AND THEN AGAIN THE SAME DAY: ELEVEN SITES, NOT SIX
(2(z80), 2(z84), 2(z85)).** The version below reads "six files sort
probabilities" and that is the wrong unit -- **the audit is per-FILE and the
defect is per-CALL SITE**. `order_probe.py` holds SIX `argsort` calls and
carries TWO separate instances; the audit listed four of the six and cleared
them in one verdict.

🛑 **AND THE PER-CALL-SITE GATE HAS ITS OWN BLIND SPOT, ONE LEVEL UP: the
enumeration is per-SPELLING and the defect is per-QUESTION.** Site 11 is
`straddle_probe.cut_score`, which located its cut with `np.partition` --
not in the target list, so the gate that had just produced sites 9 and 10
reported that file CLEAN and held ZERO entries for it.

| # | site | found | state |
|---|---|---|---|
| 1 | `order_probe --evictions` | 2026-08-28 | ✅ **FIXED 2026-09-10**, 13 days after disclosure -- and it held a THIRD defect: `K` came from `budget_for` on the RAW frame, i.e. the HARD count |
| 2-6 | task window, cap screen, fmow window, `paired_noise`, `cut_gap` | 09-01 .. 09-09 | ✅ fixed |
| 7 | `step_direction_probe` | 2026-09-10 | ✅ fixed |
| 8 | `order_probe` band + Jaccard | 2026-09-10 | ✅ fixed |
| 9 | **`score_scan` prec@K + Jaccard** | 2026-09-10 | ✅ fixed, **figure WITHDRAWN** |
| 10 | **`reachability.slope_at`** -- its `live at K` / `flat at K` VERDICT read the globally k-th item of the whole column, off the RAW frame | 2026-09-10 | ✅ fixed |
| 11 | **`straddle_probe.cut_score`** -- `np.partition(s, -K)[-K]`, and it sets the DENOMINATOR of the 2(z77) mechanism bar. Disclosed in its own docstring from the day it was written | 2026-09-10 | ✅ fixed, shuffled-control figure RE-MEASURED |

🔑 **SITE 10 WAS FOUND BY THE GATE THAT SITE 9 PRODUCED, WITHIN THE HOUR.**
Fixing site 1 (`order_probe --evictions`, disclosed 2026-08-28 and unfixed
since) introduced an `np.sort(pn)[::-1][K - 1]` the registry could not see,
because it tracked `argsort` and not `sort`. Widening the target list took it
from 40 call sites to **56**, and the first new entry anyone had to classify
was site 10. Widening it AGAIN -- `partition` and `quantile`, chosen by counting
every cut-locating spelling in the repo rather than guessing -- took it to
**59** and produced site 11. `searchsorted` is a genuine cut spelling and occurs
ZERO times, checked; `max`/`min`/`median`/`argmax` are deliberately excluded and
the reason is written at the list, because a target list that grows without a
stated boundary reaches 495 entries and stops being read. `GLOBAL-OPEN` is now
**0**: nothing is knowingly wrong and unfixed. ⚠️ All three fixes print BOTH
readings and claim NO direction (2(z64)) -- and on `straddle_probe`'s own
fixture the per-group oracle is LARGER on one class and SMALLER on another in
the same run, so there is no direction to claim.

✅ **AND #114 IS DONE, WHICH IS HOW #9 AND #10 WERE FOUND.**
`tests/test_lessons_learned.py::test_every_sort_on_scores_is_classified_per_CALL_SITE`
AST-walks `scripts/ src/ configs/` and requires all sort-on-scores call
sites -- **59** once `sort`, `partition` and `quantile` joined `argsort` in the target list -- to carry a
verdict and a reason, keyed by the sorted EXPRESSION so that
changing what is sorted turns it red. **Site 9 appeared the first time the
sites were enumerated mechanically rather than read** -- and its `prec@K` and
`Jaccard` had been computed on a globally-ranked set no run ever deployed,
including the dated `Jaccard 0.29-0.42` the tool printed in its own footer.
Mutation-tested 3/3. FRAMEWORK 2(z84).

🔑 **THE ENUMERATION ALSO FOUND A CATEGORY NOBODY HAD NAMED: `GREEDY-ROOM`.**
`score_arm.equalize` and the allocator itself sort GLOBALLY and then skip any
item whose group is full -- a third rule, neither global top-K nor per-group
top-k, and the post-hoc clipper's own. A pattern-match that flagged every
unindexed `argsort` would have called the allocator a bug.

Site 8 had a THIRD defect in the same expression, larger than the group one:
`K` came from `budget_for` on `final_predictions_raw.csv`, which holds the
model's ARGMAX -- so it was the HARD COUNT. At L20 that is ~336 against a
deployed K of ~74, and the band `K//2 .. 2K` did not contain the cut in
EITHER sense. Fixed; both readings now print side by side.

⛔ **AND `order_probe` WAS UNRUNNABLE ON EVERY REAL INPUT WHILE THAT WAS TRUE
(2(z81)).** `scripts/order_probe.py` imported the module `capped_classes` and
then defined a function of the same name, so `capped_classes.assert_single_dataset`
was an AttributeError on every `--campaign` run from `2dd84549` (2026-09-09).
Every static gate was green; `--self-test` was green because it never enters
`main`. Fixed, and gated by
`test_no_module_import_is_shadowed_by_a_local_definition` (4 controls,
mutation-tested). **The rule: every documented command needs ONE execution
that starts where a person starts.**

Six files sort probabilities. Four are clean (three group-blind by design, one
a self-test fixture). Two were not:

  * `paired_noise` counted a GLOBAL top-K -- the fourth instance of this class
    (2(z63)). It now takes the per-group top-K and REFUSES a predictions file
    with no `Group_ID`.
  * `cut_gap` read `p_K` at GLOBAL rank K -- the fifth (2(z64)). It now reports
    `p_K` (per-group, budget-weighted), `p_K_global`, and **`p_K_min` /
    `slope_max`, the deepest group's cut**.

⚠️ **AND THE DIRECTIONAL CLAIM I ATTACHED TO THE `cut_gap` FIX WAS WRONG.**
I asserted the global reading understates the gradient at the cut, built a
fixture showing it, mutation-tested 2/2 green -- and the end-to-end run
refuted it (mean per-group cut 0.690 vs global 0.681). Only the MINIMUM is
ordered against the global value; the MEAN sits above it whenever budgets
track group difficulty, which is the normal case. **Do not read `p_K` vs
`p_K_glob` as a direction.**

🔑 **THE FIRST THING TO RUN WHEN SSH RETURNS** is `cut_gap` over the live
corpus, for the `slope_max` column: whether ANY group's cut carries live
gradient, not whether the average one does. With 7 of 14 iwildcam ceilings at
K=0 that is the live-vs-dead reachability question, and the global reading
could not produce the column at all. Task #100.

### The `groupby` audit -- is the key the WHOLE cell?

Sibling question, same shape. 21 call sites; `paired_noise`'s cell is
`(model, dataset, cap)` and is complete, `full_panel`'s keys are complete.
`order_probe` groups on `(model, cap, cls)` and `cut_gap.summarise` on
`(campaign, model, cap, cls)`, both with NO `dataset` -- complete while
iwildcam was the only runnable dataset, and `gen_campaign --datasets` is
`nargs="+"`. bcn and fmow ended that. Both now call
`capped_classes.assert_single_dataset`, which reads `dataset_mode` (the field
`gen_campaign` writes and `data_loader` raises without).

### What the audits found in the TESTS, which is the part worth remembering

`tests/test_scorers_run_end_to_end.py` exists because three scorers were
unrunnable on every input. Its fixture spelled **three** fields unlike the
pipeline (`Group` vs `Group_ID`, `seed` at top level vs `hyperparams.seed`,
`dataset_name` vs `dataset_mode`), so nine group-aware scorers ran only their
FALLBACK branches and every row carried `dataset: None, seed: None`. Fixing
the group column exposed a live crash in `cell_table` -- `sorted()` on Nones,
five frames deep, in a paper-facing scorer. All three are now EXECUTABLE
gates that read the authorities rather than restating them.

⛔ And `capped_classes` -- written this week to stop tools guessing
iwildcam's class pair on a bcn campaign -- said in its own docstring and
fixture that **bcn caps 3 and 5**. bcn caps **0 and 2**;
`configs/task_windows.yml` has said so since the block was measured on the
complete 228-run `bcn1mn3`. Its checks passed because the fixture and the
assertion agreed with each other.

🔑 **THE RULE ALL OF THIS PRODUCES:** a fixture that agrees with the claim it
tests proves only that they agree. Every defect above was found by comparing
against an AUTHORITY outside the test (`src/training/logging.py`,
`configs/gen_campaign.py`, `configs/task_windows.yml`) or by making a tool
REFUSE ambiguous input and seeing what went red. None was findable by reading
the code.

### Also done, and blocked

* ✅ **#114 closed, and it paid for itself immediately.** The per-call-site
  registry found a NINTH global-top-K site the moment the sites were
  enumerated instead of read -- `score_scan`'s prec@K and Jaccard, and the
  dated `Jaccard 0.29-0.42` the tool printed from them, now withdrawn at the
  line. FRAMEWORK 2(z84).
* ✅ **#115 closed, SPLIT, and only half of it was a fixture.** `paper_rows`
  now runs end to end as a subprocess against a real `cell_table` CSV, with
  the hard-quarantine refusal, the PARTIAL drop and a not-a-cell_table
  refusal each pinned -- the campaign names read out of
  `quarantine.REGISTRY`, so the test cannot drift from the registry it
  checks. `step_dose` is NOT fixturable and now says why: `main()` needs
  `load_data` (the gitignored 3.0 GB arrays) and pretrained weights.
  🔑 **THE RULE: AN EXEMPTION WHOSE REASON IS A TICKET IS A DEFECT WITH A
  COMMENT ATTACHED.** `paper_rows` sat exempt on "needs a file fixture, task
  #116" while being the one tool that says what may be WRITTEN. Both entries
  now state a fact rather than an intention.
* ✅ **#98 closed.** The per-item OT family is cited (Sinkhorn Label
  Allocation, Confident Sinkhorn Allocation, OTAMatch; bib 57 -> 60) and
  Limitations carries the SCOPE STATEMENT that bounds the null: it closes
  aggregate count penalties with a scalar dual and says nothing about
  per-item assignment. Clean-room compile, 0 undefined citations.
  ⚠️ Verify a LaTeX edit in a COPY: a stale `main_edited_by_roei.aux`
  in `docs/paper/` beats `-output-directory` and reported all three new
  citations undefined after they resolved.
* ⛔ **BLOCKED ON SSH:** #100 (re-measure `cut_gap`), #101 (deploy the
  corrected scorers by hand into each worktree's `scripts/` -- it is outside
  `TRAINING_PATHS`, so it does not flip `-dirty` and does not move HEAD),
  #96, #97, #99. `fmow1`, `bcn1vit` and `snap2` are on detached dispatchers
  and were not reachable to check.

### The exact commands, so the handoff is instant

```bash
# 1. WHAT IS ALIVE. Both hosts -- one NFS /home, so `results/` is identical
#    from either and only `ps` differs. Never declare a dispatcher dead off
#    one host.
for h in dsisco01 dsisco02; do echo "== $h"; ssh $h   'ps -u michaer8 -o pid,etime,cmd | grep -E "main.py|dispatch" | grep -v grep'; done

# 2. DEPLOY THE CORRECTED SCORERS. `scripts/` is outside TRAINING_PATHS, so
#    this does NOT flip `-dirty` and does NOT move HEAD. Never `git pull` a
#    pinned campaign tree.
#
# !! THE BRANCH IN THIS BLOCK WAS `headroom/small-cnn` UNTIL 2026-09-10 AND
#    THAT IS 7 COMMITS STALE. Verified with `git merge-base --is-ancestor`:
#    every 2026-09-09/10 scorer fix -- order_probe's per-group band,
#    step_direction_probe's cut, score_scan's prec@K -- is on
#    `cleanup/consolidate-pipeline` and NOT on headroom. Running the old block
#    would have copied superseded scorers into all fourteen worktrees and left
#    every self-test green. CONFIRM THE TARGET BEFORE DEPLOYING, do not trust
#    this line either:
#      git rev-list --count origin/headroom/small-cnn..origin/cleanup/consolidate-pipeline
BR=origin/cleanup/consolidate-pipeline
cd ~/OptimizationLoss && git fetch origin cleanup/consolidate-pipeline
for f in paired_noise cut_gap cell_table deployed_h2h capped_classes          deep_scope latch_probe step_direction_probe task_window          arm_identity_check order_probe headroom score_scan campaign_state          stale_provenance; do
  git show $BR:scripts/$f.py > /tmp/$f.py
  for wt in $(git worktree list --porcelain | awk '/^worktree /{print $2}'); do
    [ -d "$wt/scripts" ] && cp /tmp/$f.py "$wt/scripts/$f.py"
  done
done
# then, in ONE worktree, prove they still run there:
python -m scripts.cut_gap --self-test && python -m scripts.capped_classes --self-test   && python -m scripts.paired_noise --self-test && python -m scripts.order_probe --self-test   && python -m scripts.campaign_state --self-test

# 3. THE MEASUREMENT THAT IS ACTUALLY OWED (#100). Read `slope_max`.
python -m scripts.cut_gap results/dom1 results/dom1b results/equaldose1                           results/loose1 results/taskwin2
```

⚠️ `git worktree list` is safe. `git gc` / `prune` / `repack` /
`reflog expire` / `worktree prune` are NOT -- 14 worktrees share one object
store and `code_version` resolves against it.


---

## 0-UNIT. WHICH (BACKBONE, CAP) PAIRS CAN CARRY A UNIT AT ALL (2026-09-06)

**Measured by `gen_campaign`'s own task-window check, which is the authority.
Read this BEFORE choosing a second unit -- three of the four backbones are
constrained in ways that are not obvious.**

Caps where BOTH capped classes are strictly in-window, so the cell poses the
question to both:

| backbone | usable caps | notes |
|---|---|---|
| **MobileNetV2** | `L70-70_G95`, `L80-80_G95` | `L90-90` is PARTIAL for class 2. This is `dualprop1`. |
| **RegNetY400MF** | `L70-70_G95`, `L80-80_G95` | `L90-90` is PARTIAL for class 2. Same caps as `shape1`, so its `tralo`/`alm`/`clip` data is directly comparable. |
| **ViTB16** (headline) | `L80-80_G95`, `L90-90_G95` | `L70-70` is OUTSIDE the window entirely. |
| **MobileNetV3** | ⛔ **NONE** | **class 2 has NO STRICT BAND on this backbone at any cap.** Only class 7 poses a question, so a MNv3 cell asks half the question and cannot be a full unit. |

⛔ **MobileNetV3 IS NOT A CANDIDATE FOR A NEW UNIT**, and that is easy to miss:
it appears throughout the existing corpus (`dom1`, `equaldose1`, `taskwin2`)
because those predate the strict-window measurement. Its cells are not wrong,
they are HALF-STRENGTH, and a new campaign should not spend GPU hours there.

🔑 **THE SEQUENCING, and it follows the standing rule that only a POSITIVE
signal earns hours.** ViTB16 is the headline backbone, fixed a priori, and it is
also the slowest. So:

1. `dualprop1` (MobileNetV2) reports its first cell.
2. **Positive** -> commit ViTB16 at `L80-80_G95` + `L90-90_G95` on the same
   11-arm design. That supersedes `vitdual2` entirely and, unlike it, CAN be
   priced (three lambda=0 streams -> 12 observations).
3. **Negative** -> do not burn ViT hours on it.

`RegNetY400MF` at `L70-70_G95` + `L80-80_G95` is the cheap second unit and is
what goes on GPU 1 the moment `shape1` frees it, regardless of branch -- two
priceable units is the sign-test floor this project has never had.

---

## 🛑 0-PERM. THE ONE CAMPAIGN THAT ANSWERS A MECHANISM QUESTION **AND** CAN PRICE ITS OWN VERDICT (staged 2026-09-10)

`tralo_permbudget` is `tralo` with the per-group budgets PERMUTED across
groups and the TOTAL held fixed. It is a CONTROL, not a variant, and it asks
the one question nothing here has ever asked:

> Does the constraint read the budget **CONTENT**, or is it only responding to
> the fact that a constraint EXISTS?

It is fully implemented (`permute_local_budgets` in `src/utils/data_loader.py`,
on the DATA path, not in the loss), gated in
`tests/gates/test_g2_budget.py::test_the_budget_permutation_control_holds_the_TOTAL_and_refuses_to_be_INERT`,
and has **ZERO completed runs anywhere**. It already survived one FALSE
`INERT` verdict: `smoke_arms` called `compute_local_constraints` directly and
never went through the loader, so an arm whose treatment lives in the data
path was invisible to it. Both callers now go through one function.

🔑 **AND IT WOULD BE THE FIRST PRICEABLE CAMPAIGN, WHICH IS A SEPARATE
WIN.** `deployed_h2h` reads EVERY lambda=0 stream and needs
`MIN_FLOOR_OBS = 8`. Two streams (`_null` + `_reseed`) give C(2,2)x4 = **4**
observations, which is why **0 of 17 cells in the live corpus are priced**.
Adding `tralo_reseed2` -- a third, distinct RNG offset -- gives
C(3,2)x4 = **12**, clearing the bar. So this campaign can say whether its own
result is above the noise, and no campaign so far could.
⚠️ Quote observations AND streams: 12 from 3 streams is a better median
than 4 from 2, and is NOT 12 independent draws (k streams give k-1
independent contrasts).

✅ **VERIFIED IN CODE 2026-09-10, not argued from the docs.**
`floors.is_lambda0_stream` returns True for `tralo_null`, `tralo_reseed` AND
`tralo_reseed2`, and False for `tralo` and `tralo_lam0` (which keeps
`lambda_step` and would put the treatment back in the floor).
`floors.stream_pairs` returns **3 pairs** for the three streams and **1** for
two, so 3 x 4 seeds = 12 >= 8 and 1 x 4 = 4 < 8. The priceability claim is
mechanical, not hopeful.

**BACKBONE: MobileNetV2, and that is forced, not preferred.** Its strict
windows are measured and non-empty -- class 2 `[0.70, 0.80]`, class 7
`[0.60, 0.80]` -- while MobileNetV3's class-2 intersection is **EMPTY**, so
MNv3 cannot carry a strict-task cell at all. Both caps below put BOTH classes
inside the window, and there are two cap levels, as the protocol requires.

```bash
python -m configs.gen_campaign --root results/perm1 --datasets iwildcam --models MobileNetV2 --caps L70-70_G95 L80-80_G95 --arms tralo tralo_permbudget tralo_null tralo_reseed tralo_reseed2 clip focal_clip --constraint-fp32 --constraint-grad-mode normalize
```

⚠️ `--constraint-grad-mode normalize` is typed EXPLICITLY because
`gen_campaign` DEFAULTS it to `clip`, and `--constraint-fp32` because it
defaults OFF and IS the dose (15284/15284 steps with it, 86.9% without).

✅ **BOTH CAPS VERIFIED `task` 2026-09-10** by asking the authority rather
than reading the yml: `configs.task_cells.classify(P, TW, "iwildcam",
"MobileNetV2", cap)` returns **`task`** for `L70-70_G95` and `L80-80_G95`.
Checked against two controls in the same call -- `L60-60_G95` and `L30_G50`
both return **`non_task`** -- because a classifier that says yes to everything
verifies nothing.

🔑 **AND THE PERMUTATION HAS REAL TEETH HERE, WHICH IS NOT TRUE OF
EVERY DATASET.** On iwildcam **7 of 14 per-group ceilings are K=0**, and
`task_cells` says of each that "the budget is real and binding, not a disabled
constraint". So permuting budgets across groups does not merely jitter
numbers -- it moves WHICH cameras are told to predict none of a species.
That is the largest possible change in budget CONTENT at a fixed total, which
is exactly what makes a null here informative rather than weak.

**THEN, IN ORDER:**

```bash
python -m scripts.run_campaign --root results/perm1 --step verify
python -m scripts.run_campaign --root results/perm1 --step launch
python -m scripts.dose_landed results/perm1
python -m scripts.run_campaign --root results/perm1 --step firstrun
```

🛑 **PRE-REGISTERED, BEFORE THE FIRST RUN LANDS.** Write the outcome
against this, not after reading it:

* **PASS (the constraint reads content):** `tralo_permbudget` is WORSE than
  `tralo` on deployed capped-class TP by more than the reseed floor. The
  per-group budgets carry information the mechanism uses.
* **NULL (the constraint reads only that a constraint exists):**
  `permbudget` is within the floor of `tralo`. **This is what I expect**, and
  the reasons are already measured: 74.1% of TraLO's step is aimed at K=0
  scopes that are ALREADY compliant (2(z54)); the constraint is blind to
  violation depth under `normalize`; and the delivered step is a fixed
  `lr*clip` whatever the violation is worth. A null here would say the
  per-group structure -- the thing that makes this a TRANSDUCTIVE constraint
  rather than a global one -- is not being read.
* **THIRD OUTCOME:** `permbudget` BEATS `tralo`. That is not noise being
  charitable, it is evidence the true budgets are actively mis-aimed, and it
  would point straight at the scope-priority inversion in 2(z54).

⛔ **A NULL IS A REAL RESULT HERE AND MUST NOT BE READ AS "no effect of
TraLO".** It bounds the MECHANISM, not the method. Task #99.

## 0-LAUNCH. THE EXACT COMMANDS, BECAUSE A LAUNCH THAT LIVES ONLY IN SCROLLBACK IS LOST

**`dualprop1` ran for a day before its own generating command was written down
anywhere. Any campaign that is running must have its command HERE.**

🛑 **THIS SECTION OWNS COMMANDS, NEVER RUN-STATE. `0-RUNNING` owns run-state,
and it is the only section that may say what is alive.** Every entry below is a
HISTORICAL launch record kept so the command is reproducible; an entry being
here says nothing about whether that campaign is running now. Two sections
claiming run-state is how both of them went stale -- this one still said
`dualprop1 -- LIVE` after it had landed and been scored, while `0-RUNNING`
announced `itemscale` as in-flight two days after it closed a design family.
FRAMEWORK 2(z72).

### DELETED LAUNCHERS -- the registry, because seven doc lines still name them

🛑 **`docs/launch_*.sh` DOES NOT EXIST. NOT ONE OF THEM.** The launchers that
survived were archived to `docs/archive/launchers/` (four files); the rest were
deleted outright, and **seven lines across `docs/MISSION.md` and
`docs/FRAMEWORK.md` still point a reader at the old paths.** Those lines are
left as written -- this project's style is to keep the old claim and correct it
beside, not to rewrite the record -- and this table is the correction. Found by
the 2026-09-06 cleanup, deferred, and discharged 2026-09-10.

| named at | file | recover with |
|---|---|---|
| MISSION 2179, 2561 | `docs/launch_margin2.sh` | `git show e7d9e893^:docs/launch_margin2.sh` (419 lines) |
| MISSION 2554 | `docs/launch_vitdom1.sh` | `git show e7d9e893^:docs/launch_vitdom1.sh` (334 lines) |
| FRAMEWORK 6903 | `docs/launch_iwc4.sh` | `git show e7d9e893^:docs/launch_iwc4.sh` (227 lines) |
| FRAMEWORK 4135, 5956, 6890 | `docs/launch_margin1.sh` | `git show 2c5f292a^:docs/launch_margin1.sh` |

⛔ **THE FOURTH ROW IS WHY THIS IS A TABLE AND NOT A SENTENCE.** The deferred
note said "all recoverable via `git show e7d9e893^:<path>`". That is true for
three of the four and **FALSE for `launch_margin1.sh`**, which was deleted
earlier and elsewhere -- at `2c5f292a`, "margin2 supersedes the never-fired
margin1" -- so the blanket recovery command returns nothing for it. A recovery
instruction that fails on a quarter of its cases is worse than none, because it
is tried once and believed.

⚠️ **AND `margin1` WAS NEVER FIRED**, so its launcher is a PLAN, not a receipt;
`margin2` superseded it and MISSION already records that no `margin2` exists on
disk either. Recovering either buys a design, not data.

✅ Gated by `tests/test_lessons_learned.py`: any `docs/launch_*.sh` named in a
doc must either exist on disk or appear in this table. A new dead launcher path
turns the suite red instead of waiting for the next cleanup sweep.

`dualprop1` -- LANDED and scored, MobileNetV2, 88 runs (read at 72/88). `tralo_dualprop` was REJECTED -- FRAMEWORK 2(z53):

```bash
python -m configs.gen_campaign --root results/dualprop1 --datasets iwildcam \
  --models MobileNetV2 --caps L70-70_G95 L80-80_G95 \
  --arms clip focal_clip lp alm fioretto hounie tralo tralo_dualprop \
         tralo_null tralo_reseed tralo_reseed2 \
  --constraint-fp32 --constraint-grad-mode normalize
```

`dualprop2` -- STAGED, goes on GPU 1 the moment `shape1` frees it. Identical
design, RegNetY400MF, the cheap second unit. **Validated locally 2026-09-07 by
generating it into a scratch root: 88 configs, and all four (cap x class) rows
read `in`** -- c2 K/n 0.700 and 0.800 against window 0.70-0.80, c7 0.700 and
0.798 against 0.60-0.90.

```bash
python -m configs.gen_campaign --root results/dualprop2 --datasets iwildcam \
  --models RegNetY400MF --caps L70-70_G95 L80-80_G95 \
  --arms clip focal_clip lp alm fioretto hounie tralo tralo_dualprop \
         tralo_null tralo_reseed tralo_reseed2 \
  --constraint-fp32 --constraint-grad-mode normalize
```

⚠️ **`--constraint-fp32` AND `--constraint-grad-mode normalize` ARE BOTH
NON-DEFAULT AND BOTH LOAD-BEARING.** The protocol defaults are `False` and
`clip`. Omitting the first cost `taskwin1` 9 of its 29 steps per run; omitting
the second puts the campaign off the recipe and `rig_status` refuses it.

🛑 **A FRESH WORKTREE HAS NO `.npy` ARRAYS.** They are gitignored (3.0 GB +
443 MB), so `git worktree add` yields only the tracked meta CSVs and every run
dies in ~5 s on `FileNotFoundError: train_images.npy`. That is how `dualprop1`
lost its first 24 runs. Link them from the REAL location -- and note the
sibling worktrees are themselves symlinks, so link to `optloss-audit`, never
worktree-to-worktree, or you build a chain:

```bash
ln -s ~/optloss-audit/data/iwildcam/oodslice/*.npy <new-worktree>/data/iwildcam/oodslice/
python -m scripts.data_present results/dualprop2     # gates exactly this
```

Then the step gates, in order, and none of them is optional:

```bash
python -m scripts.run_campaign --root results/dualprop2 --step verify
python -m scripts.run_campaign --root results/dualprop2 --step launch   # includes data_present
# ... launch, then on the FIRST completed run:
python -m scripts.run_campaign --root results/dualprop2 --step firstrun
```

---

## 0-PRICE. `vitdual2` CAN NEVER BE PRICED, AND `dualprop1` IS THE FIRST THAT CAN (2026-09-06)

**The only campaign carrying all four duals at equal dose is structurally
incapable of producing a priced verdict, however many seeds it finishes.**

`vitdual2`'s arms are `alm alm_null clip fioretto fioretto_null focal_clip
hounie hounie_null tralo tralo_null tralo_reseed`. The RNG floor is built from
PAIRS of lambda=0 streams WITHIN a family, and only `tralo` has two
(`tralo_null`, `tralo_reseed`). One pair x 4 seeds = **4 observations**, against
`MIN_FLOOR_OBS = 8`. The `*_null` arms of the other three families each stand
alone and contribute no pair. So even finished, every `vitdual2` cell reads
REFUSED -- UNPRICED.

⛔ It is ALSO STALLED: no dispatcher, one run stuck in `running` with no process
behind it, 29 pending. Same shape as `vittask1`.

🟢 **`dualprop1` IS THE FIRST CAMPAIGN THAT CAN GIVE A PRICED FOUR-DUAL
VERDICT.** It carries `clip focal_clip lp alm fioretto hounie tralo
tralo_dualprop tralo_null tralo_reseed tralo_reseed2` -- all four duals, both
clippers, and THREE lambda=0 streams, so C(3,2) x 4 = **12 observations**, which
clears the bar. That was the design intent and it only became true on 2026-09-06
when `rng_floor` was fixed to read every stream rather than the null/reseed pair
alone (2(z50) sibling commit).

**So resuming `vitdual2` buys more seeds on a DIRECTION and can never buy a
RESULT.** Adding `tralo_reseed2` to it is not possible either: `add_seeds`
refuses to add an ARM, correctly, because that is a new experiment. If the
ViTB16 four-dual cell is wanted as a priced result it needs a NEW campaign on
the `dualprop1` design.

---

## 🔑 0-NEXT. THE PENALTY PULLS HARDEST WHERE THE VIOLATION IS MILDEST (2026-09-06)

**The strongest mechanistic lead in the project, measured from
`training_log.csv` alone at zero GPU cost.**

TraLO's shipped penalty is `rational_bounded`, the manuscript's Eq. 4. It is
BOUNDED in the excess, so its slope `d(pen)/dE` is NON-MONOTONE: near `1/s` at
the boundary, peaking around 53-58% over, and decaying toward zero for anything
deeper. With ONE term that divides out -- the constraint gradient is normalised
as a whole, so the shape is a scalar times a fixed direction. With SEVERAL terms
it sets their RELATIVE weights, and it sets them backwards.

`scripts/penalty_starvation`, 232 epochs over 8 `dom1` runs on iwildcam:

| | |
|---|---|
| live constraint scopes per epoch | **11** |
| deepest scope violated by | **29.8x** its budget |
| median scope violated by | 0.19x |
| **spread across scopes** | **147x** |

and the pull each receives, deepest / median:

| shape | ratio | |
|---|---|---|
| `rational_bounded` (shipped) | **0.075x (rho=0.5) -> 0.014x (rho=100)** | starves the worst violator |
| `linear` | 92x | |
| `squared` | 3926x | |

**TraLO pulls its worst-violated constraint 13x to 71x LESS hard than one that
is 19% over.**

⚠️ **WHY THIS WAS NOT KNOWN.** The algebra is FRAMEWORK 2(a2) and was correct;
it was demonstrated on **dermmnist**, which is removed, leaks 38.7% of its test
set, and whose LOCAL scope was EMPTY (`lp_fallback_used` False with 0 candidates
on all 52 runs). The one dataset where the effect was shown is the one where the
many-term case barely existed. iwildcam's spread is 147x against dermmnist's
~30x, and nobody had measured it.

🔑 **AND IT IS A CANDIDATE MECHANISM FOR THE `alm` GAP.** An augmented
Lagrangian grows its pull with violation depth without bound; this shape shrinks
it. `alm` leads TraLO on the deployed head-to-head. That is a STRUCTURAL
difference between the two methods, not a tuning one -- which is exactly the
kind of asymmetry TraLO needs and has never had.

`penalty_shape: linear` and `squared` are ALREADY IMPLEMENTED in the same
function. **Neither has ever run on iwildcam.** The default stayed
`rational_bounded` only because it is the manuscript's Eq. 4 and changing it
would reinterpret every stored result.

**PRE-REGISTERED, so it cannot be rewritten afterwards:**
* the prediction is about the DEEPLY-VIOLATED scopes specifically. Read
  `Group*_Hard_Class*` against `Group*_Limit_Class*` per scope, not the campaign
  headline. A shape that fixes the weighting and moves no deployed TP is still
  informative and must be reported as such.
* `squared` is UNBOUNDED and iwildcam has 7 zero-K ceilings where the scaled
  excess can be large, so it is the likelier of the two to destabilise. If it
  collapses and `linear` does not, that is the expected ordering.
* NOT predicted to win outright. Aiming the weights correctly is NECESSARY, not
  sufficient -- `headroom` still bounds the whole prize.

---

## ⚠️ 0-DOSE. `tralo_sgd` IS UNDER-DOSED, AND THAT IS MEASURED (2026-09-06) [verified 2026-09-10]

✅ **VERIFIED 2026-09-10 against a scorer that moved the same day.** `step_dose` was edited on 2026-09-10 to report the stale-CE-momentum state, which `stale_figures` correctly flags. Checked by diff: every line touching `dw`, `cos`, `norm` or `aligned` in `measure()` is an ADDITION -- the displacement computation these numbers come from is byte-identical, so they still reproduce.
⚠️ The marker will not clear the figure until 2026-09-11, and that is the design working: a date carries no hour, so a same-day marker is ambiguous and the tie resolves AGAINST it. 2(z68) is exactly that case -- read at 09:09, scorer fixed 22:48 the same day.

`scripts/step_dose` on the real MobileNetV2 config. Constraint-aligned weight
displacement, `||dw|| * cos(dw, descent direction)`:

| rule | `\|\|dw\|\|` | cos | aligned |
|---|---|---|---|
| `shared` | 0.0444 | 0.187 | 0.00828 |
| `sgd` | 0.000100 | 0.9997 | 0.000100 |

**`sgd` delivers 83x LESS movement along the direction the constraint asked
for.** Its direction is perfect and its dose is tiny.

⚠️ **State the caveat with the number -- and it is NOT the one this line
carried until 2026-09-10.** It used to say that `cos = 0.187` is higher than the
0.009-0.017 the framework measures "after a full epoch", and that at the
framework's cosine the gap narrows to ~6x: it attributed the disagreement to
STEP COUNT. 🛑 **That axis was measured, and it moves the other way.**
2(z46) reads **0.187 at 60 CE steps and 0.258 at 126** -- and ~126 CE steps IS
the full epoch between constraint steps -- so going to a full epoch takes the
under-dose from 83x to **115x** and WIDENS the disagreement with the docstring
instead of closing it.

| cos | where it comes from | shared aligned | `sgd` under-dosed |
|---|---|---|---|
| 0.013 | `constraint_step.py` docstring, "measured in this project", uncited | 0.000577 | 5.8x |
| 0.187 | measured, 60 CE steps | 0.00830 | 83x |
| **0.258** | **measured, 126 CE steps = a full epoch** | **0.01146** | **115x** |

⛔ **AND THE DOCSTRING FIGURE HAS NO TRACEABLE MEASUREMENT.** `0.009-0.017`
occurs in exactly two places in this repo: that docstring, and 2(z46) quoting
it. It is also NOT the same quantity as the `92.6%` stale-momentum figure, which
is `ortho_survival`'s momentum algebra -- the two have been read as one claim,
and only the algebraic one has a receipt.

Either way `sgd` is UNDER-dosed, never over-dosed, so **a null from `tralo_sgd`
is about DOSE and must be reported as the dose gap, never as "delivering the
direction does not help"**. That was pre-registered in `protocol.yml` before the
campaign launched and is now quantified. What is NOT settled is the 15-20x
motivation behind the whole delivery program. FRAMEWORK 2(z73).

---

## 🛑 0-HEAD. THE SCORER COULD NOT SEE HALF THE ARMS, AND EVERY VERDICT IS UNPRICED (2026-09-06)

**Two separate defects, both in the instruments, both found with zero GPU.**

### 1. `deployed_h2h` ranked a four-name WHITELIST

`rank_cell(cell, control, get, arms=DUALS)` with
`DUALS = ("tralo", "alm", "fioretto", "hounie")`. Every other completed arm was
structurally invisible, and `tralo_wins` -- the acceptance table that answers
"does TraLO beat the clipper and the duals in >=50% of cells" -- delegates to
it. So the 35% headline was computed over a table that **could not contain the
TraLO variants built to fix TraLO**, nor `focal_clip`, which CLAUDE.md rule 2
requires in every campaign as the stronger quality bar.

It never looked broken. It printed a clean ranking of a subset and called it the
campaign.

FIXED: `rankable_arms(cell, control)` ranks every competitor present and
excludes only the `_null` / `_reseed` twins, which are floor INSTRUMENTS (ranking
them would let a cell's own noise estimate win the cell). Gated in
`deployed_h2h --self-test` with a negative control that an explicit `arms=`
still restricts.

**What the fix immediately made readable, from runs finished weeks ago:**

| | | |
|---|---|---|
| `tralo_cut` vs `tralo`, taskwin2 L70-90 | +0.00 vs **+6.00** | LOSES |
| `tralo_cut` vs `tralo`, taskwin2 L80-100 | +5.25 vs **+10.75** | LOSES |

That independently re-derives `protocol.yml`'s own `rejected_arms` entry for
`tralo_cut`. The ledger was right and nobody could see the evidence.

⚠️ AND THE LEDGER HAD A STALE ENTRY IN THE OTHER DIRECTION: it says
`tralo_coin` has "0 completed runs". It has **24**, in `vitcoin1` (ViTB16),
`coin1` (RegNetY400MF) and `coin2` (MobileNetV2).

### 2. `tralo_wins` priced cells on a RANGE, not a pairwise margin

`spread = max(d.values()) - min(d.values())`. A range over k arms grows like
`sd*sqrt(2 ln k)` against a floor that is a TWO-arm quantity at `1.13*sd`, so a
trailing arm inflates it for free -- and de-whitelisting `deployed_h2h` made it
worse by adding arms. Gated with a cell where the two visibly disagree:
**pairwise 2.0 against a range of 80.0**, a 40x inflation.

FIXED: the margin is now the NARROWER of (tralo vs control) and (tralo vs best
rival present), because a win needs both.

### 3. 🛑 THE REAL BOTTLENECK IS THE FLOOR, NOT THE MECHANISM

With the whitelist gone, the same sentence appears under every cell in every
campaign: **REFUSED, the floor rests on 4 observations, under the bar of 8.**

* acceptance table: 0 of 17 cells priced
* `coin1` + `coin2`: 4 of 4 REFUSED
* `taskwin2`: 2 of 2 REFUSED -- including `tralo` **+10.75 items over `clip`**
* `sensitivity_screen`: 36 of 38 UNDER-POWERED

Every campaign carries exactly ONE `_null`/`_reseed` pair at 4 seeds, so the
floor is a median of four numbers whose order-statistic CI is the whole sample
range. `MIN_FLOOR_OBS = 8` then refuses everything, correctly.

**`tralo_reseed2` already exists** (`protocol.yml`, `rng_reseed: 2`, a distinct
RNG stream), is documented as worth 8 runs, and **has never been put in a single
campaign**. Three lambda=0 streams give `C(3,2) x 4 = 12` observations for 8
extra runs; seeds 5-8 would give 8 for 16. Four times cheaper per observation.

### 4. THE COIN IS NOT A KILL, AND THAT MATTERS

`tralo` vs `tralo_coin` (a RANDOM constraint step of the same norm):

| cell | tralo | coin | gap |
|---|---|---|---|
| coin1 / RegNetY400MF / L70 | +3.00 | -1.25 | **+4.25** |
| coin1 / RegNetY400MF / L80 | +3.00 | +0.75 | +2.25 |
| coin2 / MobileNetV2 / L70 | -2.25 | +1.75 | **-4.00** |
| coin2 / MobileNetV2 / L80 | +2.25 | +0.50 | +1.75 |

3 of 4 in TraLO's favour, sign p=0.31, every cell inside its own floor. So the
direction is **neither proven live nor proven dead** -- it is unmeasured. Do not
quote "TraLO ties a random vector" as settled; it rests on 4 unpriced cells.

### 5. WHAT WAS STAGED AS OF 2026-09-06

⚠️ **RUN STATE IS A SNAPSHOT, NOT A FACT.** This section was written 2026-09-06 and nothing re-dates it when the rig moves. Verify with `python -m scripts.rig_status` and `python -m scripts.quarantine --list` before believing any of it. A present-tense heading with no date is how a reader ends up relaunching a campaign that was quarantined and had its pending runs dropped (2026-09-10).

`tralo_sgd` and `tralo_coin_sgd` are new arms. `constraint_step_rule: sgd`
exists, is guarded against silent fallback (`dual_common.py:228`) and has
**never run on iwildcam**. It attacks the one measured defect nothing has
addressed: under `shared`, 92.6% of each delivered constraint update is stale CE
momentum, so a count function rotated 180 degrees arrives at the weights as 9.1
degrees. That is the standing explanation for why a random step ties TraLO.

🔑 **PRE-REGISTERED, and it is a difference-in-differences, not a headline.**
The claim is NOT `tralo_sgd > tralo`. Plain SGD at `lr_constraint` is a smaller
effective step than Adam's normalised one, so a null there is DOSE, not
mechanism. The claim is that
`(tralo_sgd - tralo_coin_sgd) > (tralo - tralo_coin)`: that direction matters
MORE once it is actually delivered. All four arms are in the campaign so the
contrast is within-campaign.

### 6. A GATE WOULD HAVE REFUSED THE CAMPAIGN, AND THE FIX MADE IT STRONGER

`check_parity.SHARED_KEYS` requires `constraint_step_rule` to be IDENTICAL
across arms. `tralo_sgd` deliberately differs, so `price1` would have been
refused by its own gate. The two obvious moves were both bad: drop the key and
an ACCIDENTAL step-rule split goes silent forever, or keep it and lose the arm.

FIXED by making the exemption **declared and per-campaign** instead of global:
`gen_campaign.declared_contrasts` computes which arms deviate from the shared
block's own value, writes `CONTRAST.json` at the campaign root, and
`check_parity` exempts exactly those (arm, key) pairs while still requiring
every other arm to agree. An undeclared split is still a failure; a declaration
naming an absent arm is a failure; an unreadable marker is a failure, never an
empty exemption. `gen_campaign` does NOT import `scripts.check_parity` to do
this -- `configs/` is on the runner's import path and `scripts/` is not, which is
the only reason `scripts/` is safe to update mid-campaign.

Mutation-tested 3/3: declare every carrier (exempts everybody, checks nothing),
declare nothing, and make `check_parity` ignore the file.

### 7. GPU STATE

`vitdual2` (57/88) and `vitseed1` (22/40) were STOPPED by explicit PID on
2026-09-06. Both ran the shipped TraLO, which is already measured below the
50% bar, and finishing them sharpens an estimate that is on the wrong side of
it. All completed runs preserved. A 20-day-old `watchdog.sh` on dsisco02,
naming arms rejected weeks ago, was killed with it.

**dsisco02 GPU 0 belongs to `nirgal`, not us.** Do not touch it and do not
share it.

---

## 🛑 0-HEAD. THE FOUR-DUAL HEAD-TO-HEAD RESTS ON **TWO CELLS** (2026-09-05)

**The pre-registered goal -- TraLO beats `fioretto` / `hounie` / `alm` -- is
currently being asked of a sample of TWO CELLS, and both are unfinished.**

Measured by running the deployed `scripts.deployed_h2h` over every scorable
campaign in every worktree (`dom1`, `dom1b`, `equaldose1`, `taskwin2`,
`vitdual2`), with the dead-arm quarantine and the new floor-observation guard
both live:

| | cells | why |
|---|---|---|
| carry **all four** duals | **2** | `vitdual2` L80-80_G95 and L90-90_G95 only |
| carry 2 arms | 15 | `fioretto`/`hounie` are DEAD arms in `dom1`/`dom1b`/`equaldose1` (the 28-vs-29 dose gap) and are dropped |
| carry 1 arm | 2 | `taskwin2` staged `tralo` alone |
| **#1 NAMED** | **0 of 19** | |
| REFUSED: spread inside the RNG floor | 13 | a genuine null: the arms differ by less than the noise |
| REFUSED: the floor itself unestimated | 4 | fewer than `MIN_FLOOR_OBS` = 8 observations behind it |
| ONE ARM: nothing to rank | 2 | |

**This supersedes "2 of 15 cells namable, both `alm`".** Those two `alm` calls
were priced against a floor resting on four observations; under the guard they
are UNPRICED, not won. Nothing moves in TraLO's favour -- TraLO was already 0
-- but the rivals' two wins are withdrawn as well, so the honest statement is
that **the head-to-head has not yet been measured anywhere**, rather than that
TraLO lost it.

### What that means for the queue

1. **`vitdual2` finishing is the whole experiment.** At 33/88 it carries 2 and
   1 seeds in its two cells. Every four-way number in this project comes from
   it. Nothing else can substitute: no other campaign holds four live duals.
2. **`vitseed1` is correctly targeted for the FLOOR.** Seeds 5-8 of
   `tralo_null` + `tralo_reseed` take the floor from 4 observations to 8, which
   is the bar `deployed_h2h` now enforces. Without it every cell refuses on
   "floor unestimated" no matter how large the spread.
3. **But `vitseed1` carries NO dual arms** (`clip`, `focal_clip`, `tralo`,
   `tralo_null`, `tralo_reseed` only). So after it lands, `tralo` sits at 8
   seeds against rivals at 4 -- which is an apples-to-apples violation in the
   one comparison that matters. Seeds 5-8 of `alm`, `fioretto`, `hounie` are
   the missing 24 runs.
   ⚠️ **Their `_null` twins are NOT needed and must not be run.** Verified by
   md5 on `vitdual2`: within a (cap, seed) all four families' `_null` arms are
   byte-identical, 0 of 3 groups split, exactly as FRAMEWORK 2944 says. 24
   runs that would produce a file already on disk.
4. **Do NOT size that extension from today's numbers.** `seeds@80%` currently
   reads `tralo` 16 / `alm` 44 / `fioretto` 3 / `hounie` 5 -- computed from TWO
   seeds, so it is an estimate of an estimate. Re-read it when `vitdual2` has
   its four, then buy the seeds.

### The 13 "inside the floor" cells are the real result so far

They are not a measurement failure. Those cells have four seeds and a floor
built from four observations, and the arms still differ by less than the RNG
spread. That is a null, and per the honest-null clause it gets reported in
those words: **on MobileNetV2, MobileNetV3 and RegNetY400MF, at every cap
tested, `tralo` and `alm` are not distinguishable at 4 seeds.**

## 🛑 0-NOW. READ `docs/COVERAGE.md` BEFORE ANY NUMBER BELOW (2026-09-03)

Twelve findings supersede parts of every section that follows.

🛑🛑 **AND ONE MORE SUPERSEDES EVERY DUAL-vs-DUAL SENTENCE IN THIS FILE
(2026-09-04, FRAMEWORK 2(z40) and 2(z43)).** `fioretto` and `hounie` are DEAD
ARMS in `dom1`, `dom1b` and `equaldose1` -- 28.00 attempted constraint steps
per run against 29.00 -- and `tralo_lam0` is one in `equaldose1` too. Those are
the ONLY recipe campaigns carrying rival duals, so **the surviving field is
`tralo` vs `alm` and nothing else**. Recounted as deployed with the dead arms
dropped: **#1 named in 2 of 15 cells, both `alm`, TraLO 0** -- and all four of
TraLO's former #1 calls were in verified `task` cells, every one of them
produced by a dead arm stretching the spread past the RNG floor rather than by
a lead over `alm`. **The dose objection is REOPENED** (item 3 below).

🟢🟢 **WHAT IS UNTOUCHED, AND IT IS THE HEADLINE.**
`scripts/paper_rows.CONTRASTS` is exactly `vs_clip`, `vs_null`
(family-resolved) and `vs_reseed`; **none touches a dead arm.** So `tralo` vs
`clip` **4/4 p=0.0625**, vs its own `_null` **4/4**, vs `tralo_reseed` **3/4**,
and task-restricted **3/3 p=0.125** all stand. **0 of 15 cells are lost**
(144 of 792 runs, 18.2%, are touched; `equaldose1` worst at 42.9% of its paper
rows). **And the paper of record is entirely unaffected** -- disjoint MedMNIST
corpus, zero iwildcam rows, grep count 0, verified 2026-09-04.

**0. NOT ONE CELL IN THE CORPUS COULD HAVE SEPARATED TWO METHODS (2026-09-04,
FRAMEWORK 2(z39)).** `scripts/sensitivity_screen` over `dom1` + `dom1b` +
`equaldose1` + `taskwin2` + `vittask1` -- **38 cells, ~850 runs: SENSITIVE 0,
UNDER-POWERED 36, SATURATED 2.** The models DO saturate globally (93.6% of items
at p > 0.99 or p < 0.01, train accuracy 0.9595 -> 0.9992 THROUGH the constraint
phase) but at loose caps the CUT is fine (p@cut 0.41-0.65), so the blocker is
arithmetic: the arm-PAIR difference is 2-5 deployed TP items against an RNG
floor of 1.0-10.5 in the same cell. **Two measurement defects found:** a
`max - min` RANGE over k arms inflates ~2.7x against a two-arm floor (raw
range/floor median 2.51 over 50 cells, **0.97** once corrected), and the floor
itself rests on FOUR observations. `<fam>_reseed` twins do NOT fix the second --
they are byte-identical to `tralo_reseed`. Read finding 1 below in that light:
it is the same conclusion, now automatic and per cell.

**1. The head-to-head between the duals is measuring the RNG -- and after
2026-09-04 there is barely a head-to-head left.** Scored on the AS-DEPLOYED
predictions in exact captured items (`scripts/deployed_h2h.py`). **Recounted
over the 15 dual-carrying cells with the dead arms dropped: #1 named in 2,
both `alm`, TraLO 0** (FRAMEWORK 2(z43)). All four of TraLO's #1 calls were in
verified `task` cells and all four collapse to REFUSED, each having been named
on a dead arm's distance rather than on a lead over `alm`. At k=2 survivors the
range IS the pairwise difference, so finding 0's ~2.7x inflation factor is 1.0
here and this is the fairest reading available, not the harshest.

⛔ **THIS ITEM READ AS FOLLOWS UNTIL 2026-09-04, over 19 cells: "a #1 arm can
be named in 6 cells and refused in 13, and of the 6 it is `alm` 2, `tralo` 2,
`fioretto` 2".** `fioretto`'s two are void. ⚠️ **AND THAT TALLY DOES NOT
REPRODUCE, WHICH IS A SEPARATE UNVERIFIED ITEM.** The 19-cell root set
reproduces exactly (`dom1`+`dom1b`+`equaldose1`+`taskwin2`+`vittask1`, and both
the jackknife 10 and the items/ccF1-disagree 5 match to the integer) but the
count does not: measured 2026-09-04 it is **NAMED in 8, REFUSED in 11, of the 8
named `tralo` 4, `alm` 2, `fioretto` 2**. Likely cause is scorer version -- the
run used the SERVER's `deployed_h2h.py`, which differs by md5 from local.
**What would settle it:** re-run the current local scorer on the same roots.

⚠️ **`|tralo - a rival dual|` median 4.0 items, n=180, NEEDS RECOMPUTATION.**
It pools `alm` + `fioretto` + `hounie`, two of which are dead. Recompute
against `alm` alone; n falls to roughly 60. **The recomputed value is not
stated here because it has not been measured.** The floor it was compared
against, `|tralo - tralo_reseed|` median **4.0 items**, is unaffected and
stands. **10 of 19 cells change their #1 when one seed is dropped** also
stands, and is if anything understated now.

**2. But `tralo` vs the clippers is now p<0.05, and the old reading could not
have been.** There are **FIVE** independent units, not four: `dom1` carries
MobileNetV3 as well as MobileNetV2 and was never counted, and `taskwin2` +
`equaldose1` MobileNetV3 are md5-identical in 4/4 seeds so they are ONE unit,
not two. As deployed: `tralo` > `clip` and `tralo` > its own `_null` in **4/4 units (p=0.0625)**

> ⚠️ **THE LEDGER LICENSES FOUR, NOT FIVE (2026-09-03).**
> `scripts/paper_rows.MEASURED_UNITS` holds four entries: `dom1`/MobileNetV2,
> `equaldose1`/MobileNetV2, `dom1b`/RegNetY400MF, `taskwin2`/MobileNetV3.
> A one-sided sign test over four floors at `0.5^4 = 0.0625`, so **the
> headline cannot reach p<0.05 on this corpus at any effect size.**
> `dom1`/MobileNetV3 is a CANDIDATE fifth and is NOT in the ledger; adding
> it requires the md5 evidence that its warm-up is distinct from every
> entry already there. Until then the ledger's own doctrine applies: an
> absent entry is UNVERIFIED, not independent, and the default must not be
> the flattering one.
>
> ⛔ **AND ONLY THREE OF THE FOUR CARRY A VERIFIED `task` CELL (2026-09-04).**
> `configs.task_cells.classify`: `taskwin2`/MobileNetV3 -- ledger unit `C1` --
> is `no_strict_band` at `L70-90_G95` (class 2's strict band re-measured EMPTY
> 2026-09-02 under the per-group prize) and `unmeasured` at `L80-100_G95` (c7
> at K/n 0.950). So state BOTH: **4/4 units, p=0.0625** SIGN READ, and
> **3/3 units, p=0.125** over units with a verified `task` cell. Every sign is
> identical either way. `scripts/paper_rows.py` prints the restriction --
> take it from there rather than re-deriving it.
>
> ⚠️ **THAT `4/4` READ `licensed` UNTIL 2026-09-10, AND IT IS NOT.**
> LICENSED means present in `MEASURED_UNITS`; SIGN READ means somebody
> scored it. The ledger licenses **SIX** and only these four are read --
> C2 (`dom1`/MobileNetV3) and D1 (`bcn1mn3`/MobileNetV3) are on disk and
> unread. So this is 4 of 4 READ, not 4 of 4 that exist, and the
> denominator can still move in either direction. ⛔ DO NOT QUOTE 6/6
> BEFORE IT IS READ. FRAMEWORK 2(z66), 2(z68), task #102.
>
> 🛑 **AND THE INCLUSION RULE WAS NOT SIGN-BLIND.** The old "B2",
> `loose1`/RegNetY400MF, was removed in commit `1a7723a0` for running
> `constraint_grad_mode: clip` -- a sound reason -- but it was the
> DISSENTING unit and the commit is titled "the result gets BETTER". A
> sign test is valid only under an inclusion rule fixed BEFORE the signs
> are read. Both revisions moved the headline toward significance. Quote
> this as 4/4 one-sided, uncorrected, on ONE dataset slice, and say that
> the unit set was revised after the signs were known.

The old reading of this line said 5/5 twice. A sign test floors at `0.5^n`,
so four unanimous units could not go below 0.0625 at any effect size.

**3. `full_panel` was the wrong instrument for "which arm wins".** It scores
its OWN re-derived equal-budget allocation, not the deployed file, and the two
disagree in RANK ORDER: at `dom1`/MNv2/`L80_G95` the panel puts `tralo` +5.77
over `alm` +5.49 while both capture **exactly 2602 items**. The ordering is a
macro-averaging artefact over two classes whose `(K+n)` differ.

**4. Two of the four "rival duals" are ONE method, and TraLO's 83-degree
direction difference changes nothing.** Every trained arm's constraint gradient
is `sum_j c_j * dS_j/dtheta` with `c_j >= 0`, and `constraint_grad_mode:
normalize` rescales the result to exactly `constraint_grad_clip` -- scaling UP
below the bound, not only down -- so the magnitude is discarded. At a fixed
model state `fioretto_alm` and `fioretto_ldf` both build weights proportional
to `relu(S_j - K_j)`: **cos = 1.0000 in 192 of 192 stored states**
(`scripts/dual_cone_probe.py`) -- an algebraic identity read at a FIXED model
state, so it does not depend on how many steps an arm took and it survives.

⛔ **THE DEPLOYED HALF DOES NOT, 2026-09-04.** This read: "On the deployed
predictions `|alm - fioretto|` is **0.83x the RNG floor**. So the paper's 'four
duals' is THREE, and a dominance claim counting them separately counts one
comparison twice." **`fioretto` is a dead arm at 28.00 steps**, so
`|alm - fioretto|` compares 29 steps against 28 and cannot establish method
identity. UNVERIFIED; `vitdual2` runs both at 29.00 and would settle it.
🛑 **The conclusion is overtaken anyway, and by more:** on this corpus the
paper's "four duals" is not three, it is **TWO** -- `tralo` and `alm` -- because
`fioretto` and `hounie` are both dead in all 15 dual-carrying cells.

🔑 The sharper half: `tralo` IS a different direction -- median cosine **+0.11
against the duals, 83 degrees, and >60 degrees in 124 of 192 states**, sometimes
anti-aligned at -0.86 -- and **every trained-arm contrast still sits at or below
the RNG floor** (2.0 to 3.5 items against a floor of 3.0). A constraint
direction can be rotated most of a right angle with no measurable effect on what
is emitted. That is the strongest form of the structural null this project has,
and it is evidence FOR section 4's account, not against it. FRAMEWORK 2(z28),
gated by two mutation-tested regression tests.

**5. A COIN FLIP OF THE SAME NORM IS INDISTINGUISHABLE FROM THE CONSTRAINT
(2026-09-03).** Campaign `coin1`, RegNetY400MF, 48 runs, 0 failed, EQUAL DOSE
(`tralo` 232/232 and `tralo_coin` 232/232). `tralo_coin` replaces the constraint
gradient with a random vector of the SAME delivered norm and draws nothing from
the global RNG, so dropout masks and batch order are identical and only the
information in the direction differs. As deployed, 16 paired points:

```
FLOOR |tralo_null - tralo_reseed|   2.0 items
      |tralo - tralo_coin|          2.0     1.00x    <- a coin is as good as the penalty
      |tralo - tralo_null|          2.0     1.00x
      |tralo - clip|                3.0     1.50x    <- the only contrast above the floor
```

🔑 **PRE-REGISTERED.** Predicted from 2(z28)'s 83-degree geometry BEFORE the
campaign was generated. The `clip` gap survives randomisation, so it belongs to
the REGIME (29 extra CE epochs under a fresh Adam), not to the constraint: 3(0)'s
"the win is compute, not method", now isolated by a control instead of inferred.
`coin2` (MobileNetV2, the only other backbone whose two classes have overlapping
strict windows) is running as the replication. FRAMEWORK 2(z29).

⚠️ **AND THE CAP SCREEN THAT CHOSE THESE CAMPAIGNS' CAPS WAS WRONG.**
`task_window` counted the PRIZE over a GLOBAL top-K while every allocator here
is per-group with 7 of 14 ceilings at K=0 -- 8.5 errors global vs 2.0 local on
MobileNetV3 class 2, a 4.25x overstatement -- and passed PRIZE on `errors > 0`
when the RNG floor is 3.0 items. Fixed, gated 23 ways. Consequence:
`taskwin2`'s `L70-90_G95` was never a task cell, which is the whole explanation
for its +0.75 items; and `dom1`/`equaldose1` were wrongly retired as
SUPERSEDED, so both banners are withdrawn.

🔑 **NOTHING HERE IS SETTLED POLICY.** The rebuilt
`configs/task_windows.yml` is a single re-measurement under a prize bar chosen
the same day. `gen_campaign` WARNS on an empty band, it does not refuse.

---

**6. THE PAPER OF RECORD PRINTS A p BELOW ITS OWN FLOOR.** It states that
"cells are the independent units". `compute_base_model_id` hashes the backbone,
the dataset and `warmup_identity_keys` -- **the cap is in none of them**, so two
cap levels at one (backbone, seed) load the SAME cached warm-up. The six
tight-cap cells are `{RegNet, MNv3, ViTB16} x {L30, L40} x seeds 1-4`, i.e.
**THREE warm-up models**, and a one-sided sign test over three floors at 0.125.
The printed `p=0.031` and the `t`-test `p=0.013` on six correlated cell means
are both inadmissible. ✅ Fixed at four sites in `main_edited_by_roei.tex`
(blue, pdflatex clean); `main.tex`, `main_rev.tex` and `main_clean.tex` still
carry the old numbers. FRAMEWORK 2(z33), gated in `test_lessons_learned`.

**7. THE `vs_null` MAGNITUDE IS THE NULL MOVING, NOT TraLO.** Cross-campaign
spread at fixed (backbone, cap), in items: **`tralo` 0.63**, `tralo_reseed`
2.21, `clip` 3.36, **`tralo_null` 6.60**. TraLO lands on the same cc-F1 to four
decimals across two campaigns on two hosts while its own λ=0 twin moves up
to 11.6 items -- and at `equaldose1`/MNv2/`L90_G95` the null BEATS it. So
`dom1`'s "+12 items vs null" is the null being bad there. This is also why the
host clustering shows up in `vs_null` and NOT in `vs_clip`, the headline
contrast, where dsisco01 holds three of the four largest values. The SIGN is
untouched; quoting a `vs_null` magnitude as "the size of the effect" is not.
FRAMEWORK 2(z34).

**8. "1 OF 158 STRICT ROWS RESOLVES" IS BELOW CHANCE.** `resolved` is
`|d| >= 2*sd` with `d` a 4-seed MEAN and `sd` a PER-SEED sd, i.e. `t >= 4` on
**df = 3**, where `P(|t_3| >= 4) = 0.0280` and 158 rows yield **4.43 expected
under the global null**. One observed. The honest sentence is "**0 of 158
resolve beyond chance**". And over all 393 rows the two largest resolved
effects are **`alm`** (+11.80 vs clip, +10.51 vs reseed), not `tralo`.
FRAMEWORK 2(z32)a.

**11. `vitdual1` WAS RUNNING TWO NON-TASK CAPS, AND THE WINDOW THAT SAID
OTHERWISE WAS COUNTED WRONG.** (2026-09-03, FRAMEWORK 2(z38).) The campaign's
ViTB16 window was one `vittask1` seed recorded as PARTIAL [0.70, 0.90]. Two
`vitdual1` nulls have now measured it and the band is **[0.80, 0.90] on BOTH
classes**: at K/n 0.70 class 2 binds 2/2 but its prize is **2.5 items** against
the 3.0 floor and a ~4-item RNG floor, so the whole question there is smaller
than the noise. 0.60 is 1.5.

🛑 **AND THE SEED COUNT WAS INFLATED BY HALF.** `vitdual1` holds THREE
completed `tralo_null` runs and they are **TWO models**: a `lambda = 0` arm has
no constraint term, so its RAW predictions cannot depend on the cap, and
`L60-90/seed_1` and `L70-90/seed_1` are byte-identical (md5 3701265ff7c3e9f2,
one `base_model_id`). **A CAP LEVEL IS NOT A SEED.** This is `dom1`/`loose1`
one level deeper: there the WARM-UP was shared, here the whole 30-epoch model
is. md5 in its valid direction.

✅ **THE CAMPAIGN IS CORRECTED AND RUNNING.** Pending `L60-90_G95` and
`L70-90_G95` dropped to `~/vitdual1_dropped_*`, their COMPLETED runs kept as
receipts (22 and 11; a completed run is never deleted). `L90-90_G95` generated
IN the pinned worktree so the stamp matches, and installed beside the earlier
`L80-80_G95`. Now **88 pending over two cap levels that are `task` on both
classes**, one `code_version` `6658ef8cbc59`, recipes 99 x (fp32, normalize,
29, 1) + 22 x (posthoc, 30, 0). `classify` confirms it independently: L60-90
and L70-90 `non_task`, L80-80 and L90-90 `task`. Dispatcher relaunched on
dsisco01 GPU 3 (`main.py` snapshots its queue ONCE, so new configs require a
restart, not a rescan).

⚠️ **A PINNED WORKTREE CARRIES A PINNED GATE.** `optloss-cutwin` sits at a
commit that PREDATES `configs/task_windows.yml` and `--allow-nontask`: the file
is absent there and the flag is not in its `--help`. The window gate never ran
on this campaign and could not have. "The generator would have refused it" is
not a defence for anything generated in a worktree.

✅ **THREE INSTRUMENT FIXES, ALL GATED, ALL MUTATION-TESTED.**
`task_window` now dedupes byte-identical references and reports
`N run(s) -> M distinct model(s)`; `classify` no longer crashes on an empty
`partial` band (`2: []` unpacked as a 2-tuple and raised); and the 4-seed
self-test fixture, which shared one probability array across its four "seeds",
now perturbs each; and `paired_noise` prints `N (M distinct)` per arm.
Suite 551 tests, 550 pass / 1 skip.

✅ **AND THE FOUR-DUAL HEAD-TO-HEAD WAS NOT AT EQUAL DOSE. FIXED; THE
CAMPAIGN WAS DISCARDED AND RELAUNCHED AS `vitdual2`.**
`dose_landed` on the live campaign: every arm lands **100%** of what it
attempts, but `alm` and `tralo` attempt **29.00** steps/run against
`fioretto` and `hounie` at **28.00**. Both start their multipliers at
exactly 0, so their epoch-1 constraint loss is identically 0 and no
backward runs. `alm` starts at 0 too and still attempts 29, because its
`mu*violation^2` term is nonzero -- which is what proves the cause is the
MULTIPLIER, not the dual family. ⛔ **"It is the method" is not a licence to
ship it**: the gap sits UNDER `full_panel`'s 5-point refusal, so scoring would
have proceeded and the number would have been quoted, which is exactly the
failure mode. The dual update now runs BEFORE the primal gate (an ORDERING
change: same violations, same step size, `lambda_0 = 0` untouched, no new
knob), so all four arms take 29. `alm` is left alone on purpose -- it always
took 29, and it is the control that identified the multiplier as the cause.
Gated end to end in `tests/gates/test_g4_grid.py` with the `lambda=0` twins
required to attempt ZERO, and in source by lesson 29; both mutation-tested.

**10. FIVE THINGS IN THE PAPER OF RECORD ARE FIXED, AND ONE 2(z30) CLAIM IS
WITHDRAWN.** All in `main_edited_by_roei.tex`, blue, `pdflatex` clean at every
step: the four unit-inflation sites (2(z33)); a **same-lesion leakage
disclosure** the paper never carried, though 2(o) measured 38.7% of the derm
test set and **67.3% of melanoma, the capped class**, back on 2026-08-19
(2(z37), gated); the Hounie-RCL rate departure, with `alpha = 10` now stated at
all and the source's `0.1` / `alpha = 1` named; the mechanism figure's "every
method takes a single norm-clipped step" narrowed to the two arms it shows,
because hounie's raw norm ran 0.005-0.1105 against a clip of 1.0 and was never
rescaled; and `focal_alpha`, which is bit-inert above ~1 (10,000x -> argmax
agreement 1.0000) with the shipped 0.25 sitting inside Adam's `eps` regime.
2(z30) has no open items left.
⛔ **WITHDRAWN:** 2(z30)'s "the methods section describes a deleted pipeline".
**The paper reports MedMNIST only** -- no iwildcam anywhere in it -- so warm-up
50, the 300-epoch budget and `ratchet step 0.002` correctly describe the runs
it presents. Do NOT modernise them.
⚠️ `main.tex` is the professor's file and carries **none** of this;
`main_rev.tex` and `main_clean.tex` likewise. That is Roei's call.

**9. THE REPO HAS BEEN UNDER-CLAIMING ITS OWN POWER, AND MY CONE RESULT WAS
FORCED.** (a) `paper_rows`' "the sd is a LOWER bound, measured at 6-12x" is
impossible: `sd(A-B) <= sa+sb <= sqrt(2)*sqrt(sa^2+sb^2)`, so the worst
underestimate is 41% and positive correlation makes it an over-estimate;
measured, `sd(treated)/sd(null)` over 73 cells is median 0.78 with ZERO above
6x. (b) `fioretto_alm` vs `fioretto_ldf` "cos = 1.0000 in 192/192" is the
probe's fixed-state replay restated as algebra, not a measurement -- over
TRAJECTORIES the two reach **90 degrees** on disjoint supports. What carries
that claim is the as-deployed 0.83x-the-floor number, which means
"indistinguishable here", never "identical". FRAMEWORK 2(z32)b, 2(z28).

## 🟢 0-RUNNING. WHAT IS IN FLIGHT

✅ **LAST VERIFIED 2026-09-10 15:04, ON BOTH HOSTS, AGAINST `results/` AND
`ps`.** SSH returned after a full session down. Everything below is a CHECKED
state, not a last-known one, and the check was a CENSUS -- every campaign
directory in every worktree, not the ones the docs happen to name.

🔑 **THE FIRST ACTION ON RECONNECT IS TO VERIFY, NEVER TO RELAUNCH, AND THIS
BLOCK IS DATED WHEN IT WAS LAST *CHECKED* RATHER THAN WHEN IT WAS WRITTEN.**
Until 2026-09-10 it read `WHAT IS IN FLIGHT (2026-09-07)` and announced
`itemscale1` + `itemscale2` running on dsisco01 GPU 0 and GPU 1 -- a campaign
that had LANDED on 2026-09-08 and closed an entire design family. It never
said so anywhere in its 76 lines, and it carried a copy-pasteable **relaunch**
command. A resuming session that ran the resume protocol, saw no `main.py` and
followed the block would have re-run 192 runs of a closed direction; a session
that believed it would have left two GPUs reserved for it. A run-state block
dated at writing reads as current forever. FRAMEWORK 2(z72).

### 🖥️ THE RIG, 2026-09-10 15:04

| host | GPUs | ours | theirs |
|---|---|---|---|
| **dsisco01** Quadro RTX 6000, fp16 + GradScaler | 4 | **GPU 0 `price1`, GPU 1 `price2`** (launched 15:11) | none, and no other user at all |
| **dsisco02** RTX PRO 6000 Blackwell, BF16 | 4 | 🟢 **GPU 2 FREE at 17:5x -- `fmow1` COMPLETED 304/304 and was SCORED** | ⛔ **GPU 0 = `nirgal`, two procs, 83 GB. DO NOT TOUCH** |

⛔ **THE HOST IS PART OF THE UNIT AND NOTHING RECORDS IT.** `config.json` has no
host, amp or device field -- the only receipt is the dispatcher log line
`GPU: Quadro RTX 6000 | CUDA: 12.8 | AMP: float16 + GradScaler`. `vitdual2` was
read off `~/vitdual2.log` this way and is **dsisco01/fp16**; `price1`/`price2`
are pinned to dsisco01 by a comment in their own wrapper. Finish a campaign on
the host it started on, and read the log to find out which that is.

### 📋 THE CENSUS. 21 worktrees, 26 campaigns, 2,715 configs

```
COMPLETE
  dom1        384/384    equaldose1  216/216    uniform1    252/252
  dom1b       192/192    bcn1mn3     228/228    bcn1vit     190/190  <- LANDED
  snap2        96/96     shape1       72/72     dualprop1    88/88
  itemscale1   96/96     itemscale2   96/96     coin1        48/48
  coin2        48/48     seed58a      40/40     taskwin2     48/48
  vitdual1     37/37     bcnpilot1    16/16     bcnpilot2    16/16
  fmowpilot1   16/16     fmowpilot2   32/32
COMPLETE, ADDED 2026-09-11 (checked 02:5x, on dsisco01)
  price1       80/80     SCORED. 0 of 2 cells, TraLO loses both, alm ahead in
                         both. Equal dose 232/232 = 29.00 attempted/run. Unit
                         A2 by md5, NOT a ninth unit. MISSION 0-PRICE,
                         FRAMEWORK 2(z97).
  bcnpilot3    16/16     bcnpilot4  16/16   fmowpilot3  16/16   fmowpilot4  16/16
                         All four windows measured into configs/task_windows.yml.
RUNNING   (counts checked 2026-09-11 02:5x, on dsisco01; the jump host went
           down at ~03:10 so anything after that is LAST-KNOWN, not checked)
  price2       64/80     dsisco01 GPU 1
  bcn2mn2      23/96     dsisco01 GPU 2, claimed 00:02 by queue q01c_newunits.
                         --step firstrun GREEN; dose 58/58 = 29.00 attempted/run
                         on all five trained arms.
  vitdual2     58/88     dsisco01 GPU 0, claimed 02:54 by q01a when price1 ended
QUEUED    (each waiting for its gpu -- holding nothing)
  fmow2mn2      0/96     dsisco01 GPU 2, after bcn2mn2
  bcn2rgn       0/96     dsisco01 GPU 2, after fmow2mn2
  fmow2rgn      0/96     dsisco01 GPU 2, after bcn2rgn
                         All four: worktree ~/optloss-newunits pinned at
                         be37eb2a, 12 arms incl. THREE lambda=0 streams, both
                         cap levels strict task cells. MISSION 0-CAPS.
  vitdual2     58/88     dsisco01 GPU 0, 30 pending, after price1
  vitcoin1     16/17     dsisco01 GPU 0, 1 pending, after vitdual2
  vitseed1     22/40     dsisco01 GPU 1, 18 pending, after price2
COMPLETE SINCE THE 19:10 CENSUS
  fmowpilot3   16/16     dsisco02 GPU 2. WINDOW MEASURED, off `clip`. 2(z91)
  fmowpilot4   16/16     dsisco02 GPU 2. WINDOW MEASURED, off `clip`. 2(z91)
COMPLETE SINCE THE 15:04 CENSUS
  fmow1       304/304    dsisco02 GPU 2 -- SCORED, 0 of 4. FRAMEWORK 2(z88)
WAS STALLED, NOW QUEUED (see QUEUED above; their stale `running` rows were
corrected 19:05, so the pending counts each rose by one)
QUARANTINED
  vittask1     13/14     1 crashed; scorable=False anyway
```

🛑 **`bcn1vit` AND `snap2` ARE COMPLETE AND BOTH DOCS STILL CALLED THEM LIVE.**
`bcn1vit` is 190/190 -- the second-dataset ViTB16 campaign, and CLAUDE.md's
dataset table says `bcn1vit LIVE`. `snap2` is 96/96. Two more instances of
2(z72), found by the census rather than by re-reading the block.

🔑 **AND `price2` EXISTS, WHICH NO DOCUMENT MENTIONED.** The same ten-arm
design as `price1` on **RegNetY400MF** -- a second INDEPENDENT UNIT rather than
more seeds, which is the axis a sign test runs over. Its wrapper says so. That
is exactly 2(z83)'s defect: a campaign with no row at all is visible to nobody.

⚠️ **SEVEN CAMPAIGNS ON DISK ARE NAMED IN NO DOC**: `price2`, `coin2`,
`seed58a`, `bcnpilot1`, `bcnpilot2`, `fmowpilot1`, `fmowpilot2` -- 200 completed
runs between them. `campaign_state` audits doc-names against recorded states; it
cannot audit the other direction, because it never reads `results/`. **That is
the next widening**, and it is the same per-SPELLING/per-QUESTION lesson as
2(z85): a gate answers only for the direction it was pointed in.

### 🟢 THE QUEUE, ARMED 2026-09-10 19:07 -- 111 RUNS ACROSS 3 DATASETS AND 4 BACKBONES

🛑 **LAUNCH FROM `~/queue_runner_v2.sh`, NOT `~/queue_runner.sh`.** bash reads
a script LAZILY, by byte offset, so overwriting the file a runner is executing
makes that instance resume mid-token in the new bytes. Deploy to a VERSIONED
path and launch the new queue from it. The `_v2` copy carries three fixes the
original did not have and all three had already cost a queue (commit
`35fa4929`): the whole script was **CRLF** and died on `set: -: invalid option`
before defining its own logger, so it wrote NOTHING and simply was not there
when looked for -- and `bash -n` PASSES a CRLF script under Git Bash, so the
check that finds it is byte-level and now runs on the DEPLOYED copy; `yes 0 |`
feeds the gpu prompt that every tree pinned before 2026-09-07 still calls; and
`dispatcher_on_gpu` reads ownership off `/proc/<pid>/environ` rather than off
`nvidia-smi`, because a dispatcher RELEASES its cuda context between runs and
an idle card is not a free card.

Three runners are armed:

| runner | host / gpu | frees when | campaigns |
|---|---|---|---|
| `q01a_iwildcam_vit` | dsisco01 GPU 0 | `price1` ends (50/80) | `vitdual2` (30) -> `vitcoin1` (1) |
| `q01b_iwildcam_seed` | dsisco01 GPU 1 | `price2` ends (29/80) | `vitseed1` (18) |
| `q02_newunits` | dsisco02 GPU 2 | **RUNNING `bcnpilot3`** | `fmowpilot3` ✅ -> `fmowpilot4` ✅ -> `bcnpilot3` -> `bcnpilot4` |

🔑 **THE FOUR PILOTS BUY FOUR NEW UNITS, WHICH IS THE ONLY AXIS A p-VALUE MAY
GO OVER.** Measured task windows exist for iwildcam x4 backbones, fmow x2 and
bcn x2; the four missing pairs -- **fmow x MobileNetV2, fmow x RegNetY400MF,
bcn x MobileNetV2, bcn x RegNetY400MF** -- have no window, and `gen_campaign`
refuses a cap outside one. Each pilot is `clip` + `focal_clip` at two caps over
4 seeds, whose ONLY job is to give `task_window` a finished unconstrained
model. Same shape as `bcnpilot1/2` and `fmowpilot1/2`. Ledger 8 -> 12 moves the
attainable sign floor from 0.5^8 to 0.5^12. Task #124.

⛔ **THEY ARE ON dsisco02/bf16 DELIBERATELY**, matching `bcn1mn3` and `fmow1`.
The host is part of the unit and nothing in `config.json` records it, so
putting bcn x MNv2 on fp16 beside bcn x MNv3 on bf16 would have made the two
non-comparable for free.

⚠️ **THE PILOTS ARE `--allow-nontask` BY NECESSITY, AND THAT IS THE ONE
LEGITIMATE USE OF THAT FLAG**: you cannot place a cap inside a window you have
not measured yet. Nothing about method comparison may be read off a pilot --
they carry no trained arm at all.

🛑 **ITS FIRST LAUNCH TRAINED ON THE CPU AND NOTHING RAISED (2(z90)).**
`conda activate` did not survive into the detached shell, the child came up as
base python whose torch is CPU-only, and fmow x MobileNetV2 ran at 120 cores
with GPU 2 at 0% / 3 MiB. The campaign wrote `status: running` throughout. It
is a REPEAT of a row already in `rig_status`'s docstring -- a check that lives
in a tool somebody has to remember to run is not a gate. The runner now
activates the env itself by absolute path and calls `assert_gpu_ready` before
every claim; that CPU run's log was deleted and its config reset to `pending`.

**The runner's own gates, each a failure this project already paid for**, and
every one shown to FAIL on a fixture before it was trusted:
python outside the env -> ABORT (exit 4); torch with no visible cuda device ->
ABORT (exit 4);
foreign user on the gpu -> ABORT the whole queue (never share, never queue
behind); missing root -> SKIP; 0 pending -> SKIP; more than one
`code_version` in a campaign -> SKIP (the tree moved under a staged campaign);
`data_present` RED -> SKIP (a fresh worktree passes every launch gate and then
fails 24 runs in 120 seconds on a gitignored `.npy`). A sixth control confirms
a single-stamp campaign is NOT skipped, so the stamp check is not vacuous.

🔑 **`~/optloss-queue` IS A NEW WORKTREE PINNED DETACHED AT `5a9e2d7f`** with
all three datasets symlinked at their REAL locations in `~/optloss-audit`
(never worktree-to-worktree, which builds a chain). All four pilots are a
single stamp `5a9e2d7f337e` and all four passed `--step verify` GREEN.

⚠️ **TWO STALE `running` STATUSES WERE CORRECTED FIRST.** `vitdual2` and
`vitseed1` each carried a run marked `running` with no process behind it --
verified by checking `/proc/<pid>/cwd` of every live `main.py`, both of which
are `price1`/`price2` in `~/optloss-price`. `quarantine --apply --execute`
moved them `running -> crashed` (2 corrections, 0 markers, 0 removals), and
`reset_crashed` then reset **0** -- correctly, because it requires an
`error_log*.json` and a killed dispatcher leaves none. They were set to
`pending` directly after confirming each directory held only a partial
`training_log.csv`: no predictions, no model.

### 🔴 WHAT THE STALLED ONES OWE, AND WHY `vitdual2` IS THE ONE THAT MATTERS

`vitdual2` is the ONLY campaign that puts `tralo` against `alm`, `fioretto` AND
`hounie` at EQUAL DOSE on the **headline** backbone -- i.e. it is the acceptance
table. It is 66% done and its 29 pending runs span every arm including both
clippers. ⛔ Do NOT restart it on dsisco02: its 58 completed runs are dsisco01
/fp16, and 2(u) measures `--constraint-fp32` on fp16 landing 69% of its dose
when it is absent. Queue it on dsisco01 when `price1` or `price2` frees a GPU.

`vitseed1` carries no dual arms and feeds the FLOOR only, so it ranks below
`price1`/`price2`, which carry three lambda=0 streams and buy the same thing at
3 obs per 8 runs instead of 1 per 4.

🛑 **ALL THREE STALLED CAMPAIGNS ARE dsisco01/fp16, VERIFIED FROM THEIR OWN
DISPATCHER LOGS 2026-09-10** -- `vitdual2` (read earlier from `~/vitdual2.log`),
and now `vitseed1` and `vitcoin1`, both of which print
`GPU: Quadro RTX 6000 | CUDA: 12.8 | AMP: float16 + GradScaler`. **So NONE of
them may be finished on the dsisco02 GPU that just freed.** The host is part of
the unit and there is no field in `config.json` that records it; the log line is
the only receipt. Anything that runs on dsisco02 GPU 2 must therefore be a NEW
campaign or an offline probe -- and 2(z88) says what it should not be, which is
another grid of the arm that just lost its cleanest test.

### LANDED. ⛔ DO NOT RELAUNCH

| campaign | landed | outcome |
|---|---|---|
| `itemscale1` + `itemscale2` | 2026-09-08 | 192 runs, dose 232/232, every gate green. Mechanism CONFIRMED and replicated on two backbones; the deployed score did NOT follow, 25% against the 50% bar, 0 of 4 cells priced, and `tralo_coin` -- a RANDOM constraint direction of the same norm -- took one cell. **Closed the whole per-scope weighting family**, and with it task #89. FRAMEWORK 2(z56) |
| `bcn1mn3` | 2026-09-09 | COMPLETE, 228 runs. ⛔ Its sign is **UNREAD** -- this is unit D1, it is on disk, it costs ZERO GPU-hours, and it is the top of the queue. Task #102, MISSION 0-UNREAD |
| `bcn1vit` | **verified 2026-09-10** | COMPLETE, 190/190. ⛔ **UNSCORED.** ⚠️ Score **L90 ONLY** -- L70 and L80 were archived as non-task to `~/optloss-archive-bcn1vit-L70-nontask-2026-09-09` and nothing there is evidence about any method. FRAMEWORK 2(z58) |
| `snap2` | **verified 2026-09-10** | COMPLETE, 96/96. 🛑 **THE ONE WITH NO RESOLVABLE PROVENANCE** -- it runs from the server branch `snap/slice-provenance`, never merged, absent from the remote-tracking branches here. Its `code_version` resolves in NO other checkout, the 633-test suite has never executed against that code, and no scorer correction since 2026-09-08 applies unless hand-deployed. Fetch and review the branch BEFORE scoring a run. Tasks #103, #97. FRAMEWORK 2(z67) |

### 🛑 EVERY OTHER CAMPAIGN THE DOCS NAME. THE TWO TABLES ABOVE ARE A SAMPLE, NOT A CENSUS (2026-09-10)

`python -m scripts.campaign_state` harvests every campaign-shaped name in
`CLAUDE.md` + the four `docs/` files and holds it against the four places a
state is supposed to live -- `quarantine.REGISTRY`, COVERAGE section 0's
census, the two tables above, and CLAUDE.md's archive list. First run:
**48 names, 30 recorded, 18 with NO recorded state, 10 of those carrying a
past-execution verb.**

⛔ **THE TABLES ABOVE COULD NEVER HAVE SHOWN THIS, AND THAT IS THE POINT.**
2(z72) was a campaign that had LANDED and was still announced as RUNNING -- a
stale row, visible to anyone who re-read the block. This is the complementary
defect: a campaign with **no row at all**, which is visible to nobody, because
a run-state table can only be audited for the rows it contains. FRAMEWORK
2(z83).

🔑 **CALIBRATED BY HAND, because an uncalibrated report is a rumour.** The ten
came out **3 genuine, 1 near-genuine, 6 stated-in-prose-only** -- the same
ratio 2(z68) and 2(z78) measured for the other two staleness tools. Quote that
ratio beside any count taken from here.

| campaign | LAST-KNOWN state, and where the claim is | dated? |
|---|---|---|
| `price1` | 🛑 **NOTHING, ANYWHERE.** Named 4x in FRAMEWORK, twice as `before \`price1\` launched`, and task #78 "Launch price1" is COMPLETED. No run count, no dose, no host, no outcome exists in any file. **VERIFY FIRST ON RECONNECT** | no |
| `vitdual2` | 🛑 **THREE INCOMPATIBLE FIGURES.** `RUNNING 0/88` (FRAMEWORK 2(z67)'s table, quoted there as false), `32/88` (MISSION older queue), `57/88` **STOPPED by explicit PID** (MISSION 0-HEAD §7). None is in an authority and none is dated at its line | no |
| `margin2` | 🛑 **THE TWO DOCS CONTRADICT EACH OTHER.** FRAMEWORK 2(z12) says "`margin2` is 432 runs staged against it"; MISSION's knob ledger says "**NOT staged** -- checked 2026-09-02, no `margin2` exists on disk anywhere; this line said it was". The MISSION reading is the CHECKED one. ✅ FRAMEWORK corrected 2026-09-10 | 09-02 |
| `shape1` | ⚠️ "is running `tralo_linear` and `tralo_squared`" (FRAMEWORK 2(z45)), undated, and MISSION 0-LAUNCH schedules `dualprop2` for "the moment `shape1` frees it". No landing recorded | no |
| `dualprop1` | LANDED and scored, MobileNetV2, 88 runs, **read at 72/88** so every cell is 3 seeds not 4 (MISSION 0-OPEN, 2(z53)). Task #106 re-reads it at completion | 09-06 |
| `dualprop2` | STAGED, queued behind `shape1` on GPU 1 (MISSION 0-LAUNCH) | 09-06 |
| `coin1` | LANDED. RegNetY400MF, 48 runs, 0 failed, EQUAL DOSE 232/232. It is the whole evidence base for 2(z29) | 09-03 |
| `coin2` | LANDED. MobileNetV2, the replication; its two cells are in MISSION 0-HEAD §4's table | 09-06 |
| `vitcoin1` | LANDED. ViTB16. With `coin1`+`coin2` it holds the **24** `tralo_coin` runs `protocol.yml` said were zero | 09-06 |
| `vitseed1` | **22/40, STOPPED by explicit PID** (MISSION 0-HEAD §7). Carries NO dual arms, so it feeds the FLOOR only | 09-06 |
| `perm1` | STAGED 2026-09-10, **ZERO completed runs anywhere** (MISSION 0-PERM). Correct: it has not run | 09-10 |
| `clipsweep1` | PRE-REGISTERED 2026-09-10, not generated (MISSION 0-CLIP, FRAMEWORK 2(z82)). Correct: it has not run | 09-10 |
| `cutwin1` | **DELETED** -- built around a tight-vs-loose contrast 2(z16)/2(z17) removed (MISSION older queue) | 09-02 |
| `vitdom1` | **HELD**, never launched, pending the 0-PRE decision. 240 ViTB16 runs staged | 08-30 |
| `vitdom2` | the family name; `vitdom2_cnn` and `vitdom2_vit` are both ARCHIVED (CLAUDE.md). Nothing runs under the bare name | 09-02 |
| `margin1` | never staged -- the whole `margin` count family is `NEVER RUN` in MISSION's knob ledger | 09-01 |
| `rankpair` | **does not exist on either host**, searched 2026-08-22 (FRAMEWORK 2(c)) | 08-22 |
| `ortho` | not ours: `newdirections/arm_ortho/results/ortho`, a different tree | -- |

✅ **CHECKED AGAINST `results/` ON 2026-09-10 15:04, AND FOUR ROWS WERE
WRONG.** The table above is the ledger as it stood while SSH was down; the
census in this section's header is the measurement. What the check changed:

| row | the claim | the measurement |
|---|---|---|
| `price1` | "NOTHING, ANYWHERE. VERIFY FIRST ON RECONNECT" | **EXISTS**, 27/80, MobileNetV2 x 2 caps x 10 arms, recipe confirmed on disk, dsisco01/fp16. Task #119 closed |
| `vitdual2` | three incompatible figures, `0/88` / `32/88` / `57/88` | **58/88**, 29 pending, stalled with a dead `running` status. None of the three was right |
| `vitseed1` | "22/40, STOPPED by explicit PID" | **22/40 confirmed**, and still carrying a dead `running` status |
| `shape1` | "is running ... No landing recorded" | **LANDED**, 72/72 complete |

⚠️ **AND THE LEDGER WAS INCOMPLETE IN THE OTHER DIRECTION**: `price2`, `coin2`
is here but `seed58a`, `bcnpilot1`, `bcnpilot2`, `fmowpilot1`, `fmowpilot2` are
not, and they hold 200 completed runs. The `dated?` column is the date the
claim was made, and a blank means the claim carries no date at all -- which is
2(z72)'s defect and the reason three of these were unreadable.

🟢 **THE GATE THAT KEEPS THIS TRUE.**
`tests/test_lessons_learned.py::test_no_campaign_is_discussed_without_a_recorded_state`
runs the audit and fails on any campaign that carries a past-execution verb and
no state. Adding a campaign to a doc without a row here now turns the suite red.

### The resume protocol for a tree you have just reconnected to

```bash
# 1. WHAT IS ALIVE -- on BOTH hosts. One NFS /home means `results/` is
#    identical from either, and only `ps` differs, so a dispatcher on the
#    other host reads as dead from this one.
ssh dsisco01 'ps -u michaer8 -o pid,etime,cmd | grep main.py | grep -v grep'
ssh dsisco02 'ps -u michaer8 -o pid,etime,cmd | grep main.py | grep -v grep'
# 2. WHAT IS DONE -- from the results tree, which is the authority. Never
#    conclude a campaign is unfinished because no process is running.
python -m scripts.dose_landed results/<root>
python -m scripts.rig_status
```

⛔ **A campaign with no process is not automatically a campaign to relaunch.**
Check `dose_landed` and this section's LANDED table first. Both of those are
cheap; a wrong relaunch costs a week of GPU and, worse, produces runs that look
like corpus.

## 📌 0-OPEN. NUMBERS THAT ARE NOT FINISHED YET (swept 2026-09-10)

**TEN FRAMEWORK entries state a result and then say, in their own text, that
the number is not final.** An obligation is not a state claim -- it does not go
stale, it only gets discharged -- so this is a checklist and every line names
the task that owns it. ⛔ **Do not quote a number from these entries without
its caveat.** The failure mode is not that the number is wrong; it is that the
caveat is one line above it and gets left behind on the way to a table.

| entry | what is unfinished | owner |
|---|---|---|
| 2(z53) | `dualprop1` was **72 of 88** runs, so every cell is **3 seeds, not 4**. The DIRECTION is safe (sign 6 of 6, `tralo_dualprop` stays rejected); every NUMBER, and the ranking-damage claim, is not | **#106** |
| 2(z59) | the `fmow` task windows are written from **1-2 seeds** | #96 |
| 2(w4) / 2(z84) | **`order_probe --evictions` HAS NO CURRENT ITEM FIGURE.** Both of its published numbers are superseded by the 2026-09-10 fix and neither has been re-measured: `+16.50 items per cell`, and `overstates by 6.5x`, which was `16.50 / 2.53` -- a wrong reading over a right one, so it sizes a defect that no longer exists rather than any property of the tool | **#121** |
| 2(z84) | `reachability`'s `live at K` / `flat at K` verdicts were all GLOBAL readings (site 10). Every verdict it has ever printed is UNVERIFIED until it is re-run per group | **#121** |
| 2(z67) | every `snap` result, because its branch is unfetched and its `code_version` resolves nowhere here | #103, #97 |
| 2(z68) | the acceptance figure `6 of 17 = 35%` predates the scorer both halves of its verdict read. Say **"FAIL, figure pending recompute"**, never the figure | #104 |
| 2(w3) / 2(z53) | the two results measure the SAME contrast with OPPOSITE signs and were never reconciled; `+0.0253` is additionally POOLED over 3 backbones x 2 caps, which rule 4 forbids | **#108** |
| 1b-pre | the two `coin` rows, measured on the instrument that was broken until 2026-08-20 | -- |
| 2(z52) | the per-backbone derivation must be re-read per backbone -- ✅ **discharged** by #86 for `headroom` | ✅ #86 |
| 2 | the feature-space claim must be re-read on every new dataset | #90, #96 |
| 3 | the regime-beats-method table is recorded as provisional against a named misreading | -- |

🔑 **THE TWO WITH NO OWNER ARE THE TWO THAT CANNOT BE DISCHARGED BY RUNNING
ANYTHING.** 1b-pre's instrument no longer exists to re-measure on, and entry 3
is provisional by construction rather than by seed count. They are listed so
that "no task" is visibly a judgement and not an oversight.

## 🧹 0-CLEAN. THE STEP GATE AND THE SYNC (2026-09-02)

**`python -m scripts.run_campaign --root <root> --step <step>`** is now the way
a campaign moves forward. Five steps -- `stage`, `verify`, `launch`,
`firstrun`, `score` -- each running BOTH the `tests/gates` bucket that proves
the detector works and the instrument that runs it against THIS campaign.

🔑 **THREE OUTCOMES, NOT TWO.** pass / FAIL / **UNRUNNABLE**. A campaign
worktree is pinned at the commit its configs were generated from, the gate
buckets import `configs.task_cells`, and that module postdates `1d921173`.
`configs/` is frozen mid-campaign, so on `optloss-domb` that gate genuinely
cannot execute. It is reported as having verified NOTHING -- not as a failure
of the campaign. A gate that cries wolf gets switched off, and that is how this
project lost `taskwin1`'s dose.

### What the cleanup actually found

The tracked Python was NOT the bloat: `dead_code` reports three dead symbols in
`configs+src+scripts`, and an orphan audit finds zero orphaned scripts. The
redundancy was one-off documents and never-run staging debris.

* **-10,228 lines / 94 files** deleted: seven `docs/launch_*.sh` for archived
  or never-run campaigns, `docs/paper/data/dynamics/` (dermmnist, a removed and
  leaking dataset), two orphaned scouting notes, `main_old.tex`.
* **The four live-corpus launch scripts were RESTORED** after the first sweep
  took all eleven and eight gates went red. `dom1`, `dom1b`, `equaldose1` and
  `uniform1` launch scripts are provenance for how the corpus was made.
* **Three local `results/` roots were off-recipe staging debris** -- 152 configs,
  ZERO completed runs, arms since rejected, not on the server. They kept
  `rig_status` permanently red. Removed. No completed run was touched.

### The sync gap, and it was worse than expected

**`tests/gates/` existed on ZERO server worktrees**, and the worktree running
`seed58a` was missing 11 scripts including `task_window`, `deployed_h2h`,
`cell_table` and `paper_rows` -- every current scorer. Both trees are outside
`TRAINING_PATHS`, so `scripts/` and `tests/` were copied by hand into
`optloss-domb`, `optloss-cutwin` and `OptimizationLoss`. `code_version` is
unchanged in all three and the training paths are clean.

🛑 **THE STANDING RULE THIS IMPLIES.** A campaign worktree is pinned, so
it drifts from `main` the moment anything lands. Re-sync `scripts/` and
`tests/` by hand before scoring anything on the server, and NEVER by moving its
HEAD.

Running all 23 self-tests on the server at each pinned commit found
`collateral_probe` uninvokable -- inverted flag behind a required `--campaign`.
Fixed with a standalone `--self-test` and a no-op injection as its negative
control. 23/23 now.

---

## ⚠️ 0-PRE. THE DECISION THAT BLOCKS EVERYTHING (2026-08-30)

**`grep -ril iwildcam docs/paper/` returns ZERO files.** The manuscript is a
DermMNIST / OctMNIST / TissueMNIST / HAM10000 paper at **warm-up 50 with no
lambda=0 twins**. Every campaign run since 2026-08-21 is **iwildcam at warm-up
1 with twins**, and all three MedMNIST datasets are removed from disk.

⇒ **the paper and the evidence base share no dataset, no warm-up regime and
no controls.** Every iwildcam result currently has no destination, and every
paper claim rests on data that is quarantined or deleted.

**This is Roei's call and it should be made before the next campaign:**

| option | cost | consequence |
|---|---|---|
| **A. Rewrite the empirical section around iwildcam** | the 8 corpus-generated tables go, and the corpus **cannot be rebuilt** | the paper matches the evidence, and inherits warm-up 1 + real controls |
| **B. Restore a MedMNIST dataset** | re-download + re-run; `octmnist` was CLEAN in the leakage audit (keeps MedMNIST's official split), `dermmnist` leaked 38.7% | the corpus tables survive, but `octmnist`'s groups are `index % 3` -- **dead by construction** for the local scope (FRAMEWORK 2(n)) |
| **C. Both** | most expensive | iwildcam as the headline, MedMNIST as the legacy comparison |

⚠️ Option B has a trap already measured: `octmnist` passes the leakage audit
but **fails the information screen** (NET -7 items, z=-0.4), so a count
constraint carries nothing there. B restores the tables at the cost of a
dataset that cannot test the thesis.

---

## 🟢 0-RESULT. WHAT THE TASK WINDOW CHANGED, 2026-09-01

🛑 **CORRECTED THE SAME DAY BY FRAMEWORK 2(z24). READ THAT FIRST.** Three
defects sat under the numbers below: the window was a **MEAN over seeds whose
unconstrained counts spread 105 items** (so a cell can be a "task" whose cap
binds in half its seeds); each window row was **imported from another
campaign's model** (MobileNetV3 class 2 reads 336 in `dom1`/`loose1` and 355 in
`equaldose1`/`iwc3`, on the SAME cached warm-ups); and `dom1` and `loose1` are
**byte-identical on MobileNetV2's lambda=0 arm in 8 of 8 pairs**, so they are
one model, not two campaigns.

⇒ **3 distinct lambda=0 models carry every strict task cell in the project.**
`4 of 4, p = 0.0625` is **`3 of 3, p = 0.125`**. Every SIGN below is unchanged
and the dilution biases toward zero, so the positive readings are conservative;
what falls is the UNIT COUNT. And **ViTB16 has ZERO strict task cells at any
cap ever run** -- its classes' per-seed windows are 0.70 and 0.90 and do not
overlap, so only `L70-90_G95` can express one.


2(z17) made it possible to ask, for the first time, whether a cell poses a
question. Applying that to campaigns **already complete on disk** -- no GPU --
changes the picture. Read each arm against **its own lambda=0 null**, and price
it against the **`tralo_reseed` floor measured inside the same campaign**.

| campaign | backbone(s) | task cells | `tralo` vs ITS floor, ccF1 items |
|---|---|---|---|
| **`dom1`** (384 runs, 16 arms) | MobileNetV2/V3 | 4 of 6 | **12.46 vs 8.12 = 1.53x**, above in 4/4 cells |
| **`dom1b`** (192 runs, 16 arms) | **RegNetY400MF** | 3 of 3 | **4.38 vs 1.76 = 2.49x**, above in 3/3 |
| **`loose1`** (144 runs) | MNv2/MNv3/RegNet | 5 of 6 | **9.65 vs 6.27 = 1.54x**, above in 4/5 |
| **`loosevit1`** (48 runs) | **ViTB16 (headline)** | 1 of 2 | **1.41 vs 0.51 = 2.8x**, only arm positive on all four metrics |
| `equaldose1` (216 runs, 9 arms) | MobileNetV2/V3 | 4 of 6 | 2.32 vs 3.39 = **0.68x** |

🛑 **THE CELL COUNTS ABOVE ARE NOT INDEPENDENT UNITS.** Cells at different cap
levels on the same backbone share ONE lambda=0 warm-up model, so "4/4 cells" is
2 independent units. **The honest statistic is the 4 (campaign, backbone) pairs
in 2(z23): `tralo - tralo_reseed` is positive in 4 of 4, sign p = 0.0625.**
Counting cells gives 7/8 and p=0.035, which is anticonservative -- do not quote it.

✅ **IT REPLICATES ACROSS CAMPAIGNS AND HARDWARE.** `loose1`'s MobileNet subset
reads 12.74 items / 3-of-3 / 1.42x against `dom1`'s 12.46 / 4-of-4 / 1.53x --
different campaign, different `code_version`, same answer. RegNet reads 2.49x
(fp16, one commit) and 2.24x (bf16, another commit).

⛔ **THIS ALSO SAID "The arm ORDERING `tralo > alm ~ fioretto > hounie` is
identical in `dom1` and `dom1b`" UNTIL 2026-09-04.** Two of the four positions
are dead arms, so a four-place ordering is not readable. **The surviving
two-place ordering does replicate:** `dom1` tralo 12.46 > alm 10.98; `dom1b`
tralo 4.38 > alm 2.87.

⛔ **THIS READ "`tralo` leads every rival dual on ccF1 in the task cells of
both `dom1` and `equaldose1`" UNTIL 2026-09-04.** Two of the three rivals it
counted are dead. **Against the one survivor the claim holds and should be
stated that way:** in `equaldose1`'s 4 task cells `tralo` is +2.32 items against
`alm`'s -0.73. ✅ **Untouched:** `tralo` is still the only arm above its OWN
floor in 4/4 of `dom1`'s task cells, and `alm` still leads on AP in `dom1`
(+0.0426 vs +0.0403), so the ordering is metric-dependent -- say which metric
every time.

⛔⛔ **THE DOSE OBJECTION IS REOPENED, 2026-09-04. THIS READ: "The dose
objection is closed. `equaldose1`: `tralo` +0.0275 AP against the dose-matched
`tralo_lam0` +0.0287. The 3.4% step head start is not the source of the lead."**
`tralo_lam0` attempts **28.00** steps and is itself a DEAD ARM in `equaldose1`.
**The control built to close the objection is the one the defect landed on**,
and every other arm at 28.00 there (`fioretto`, `hounie`) is dead too, so **no
dose-matched control survives anywhere in the corpus.** Under the registry as
written `drop_dead_runs` removes those runs before any scorer sees them, so the
number is not recomputable at all. **`vitdual2` is the only campaign that can
close it** (all four duals at 29.00, verified), at **32 of 88 complete**.

⚠️ **A TENSION IS RECORDED HERE, NOT RESOLVED.** `scripts/quarantine.py` calls
`tralo_lam0`'s 28.00 a defect; section 4's launch note for `equaldose1` below
says the 28 was **DELIBERATE**, the arm existing precisely to match the duals'
28 so TraLO's extra step could be priced, with its void check recorded as
PASSING on exactly that basis. Both readings are on file and they disagree
about whether an arm mismatched BY DESIGN counts as a dead arm. **UNVERIFIED --
this needs a human decision, not a recount.** What would settle it: decide, and
write the decision into the `equaldose1` registry entry.

⛔ **`tralo_uniform` never leads anywhere**, and clears its own floor in
**3 of 12** measured task cells (`dom1` 0/4, `loose1` 1/5, `dom1b` 2/3), plus -3.50
items against a 0.51 floor on ViTB16. It was already below the floor at tight caps
(2(z11)). Do not run it again -- but "refuted in every regime" overstates it.

🔴 **NOT ESTABLISHED, and do not let the above imply it.** Every figure here is
a mean over cells or a 4/4 sign count -- min sign-test p is 0.0625, and
`full_panel`'s paired reading calls every line UNDERPOWERED (9-17 seeds needed).
`equaldose1` puts `tralo` BELOW its floor at 0.68x on the same backbones and
caps where `dom1` puts it above at 1.53x, and the two campaigns' floors differ
2.4x in magnitude and in SIGN. **The floor is not portable between campaigns**
(2(z21)); measure it inside whatever is being scored.

🔑 **The one structural claim that IS supported across all three campaigns:
which method looks best depends on whether the cell poses a question.** `alm`
is best on ccF1 in `equaldose1`'s non-task cells and second-worst in its task
cells; `tralo_uniform` is the best arm in one `dom1` non-task cell and the worst
in all four task cells. Every historical ranking in this project pooled cells
without asking.

---

## 🗺️ 0-MAP. THE CLEAN CORPUS -- `docs/COVERAGE.md`

**Rebuilt 2026-09-02 after clearing the stale corpus.** Walking every
`config.json` in all 14 worktrees found **FIVE distinct TraLO configurations**
across 277 completed `tralo` runs. Only one is current:

> **`iwildcam` + `constraint_fp32: True` + `constraint_grad_mode: normalize`.**

**18 campaigns / 1,326 configs moved to
`~/optloss-archive-stale-2026-09-02/`** -- ten stale-recipe iwildcam campaigns,
seven dermmnist ones on the leaked test set, and `vitdom2_*` (current recipe but
staged entirely on rejected caps). `results/` now holds **1,228 configs, all one
recipe**.

🔑 **CLEARING CHANGED THE RESULT.** The old "unit B2" that dissented on all
three contrasts was `loose1`, which ran `grad_mode: clip`. With it gone:

| contrast | units | sign p | |
|---|---|---|---|
| `tralo` vs `clip` | **4/4** | 0.0625 | ✅ beats the bar everywhere |
| `tralo` vs its own null | **4/4** | 0.0625 | ✅ attributable |
| `tralo` vs `tralo_reseed` | **3/4** | 0.3125 | ⛔ fails on MobileNetV3 |
| ~~#1 of the four duals~~ | ~~**3/6 cells**~~ | ~~0.66~~ | ⛔ **SUPERSEDED 2026-09-04: 2 of 15 cells namable, both `alm`, TraLO 0 (FRAMEWORK 2(z43)). Two of the four rivals are dead arms, so this row counted a field that does not exist** |

⛔ **AND THE DENOMINATOR IS 3, NOT 4, ONCE A VERIFIED `task` CELL IS REQUIRED
(2026-09-04).** `taskwin2`/MobileNetV3 -- unit `C1` -- classifies
`no_strict_band` at `L70-90_G95` and `unmeasured` at `L80-100_G95`, so it
carries none. Over the three task-carrying units each of the three CONTRAST
rows above reads **3/3 units, p=0.125**, because C1 was also the MobileNetV3
dissent on the reseed row. Cleaner AND less significant. `scripts/paper_rows.py` computes and
prints the restriction; take it from there.

🛑 **THE ONE FAILURE IS SPECIFIC.** On MobileNetV3 (`taskwin2`, C1) `tralo`
beats `clip` by +7.32 items but beats its own `_null` by **+0.75** -- below the
one-item quantum -- and **loses to a pure RNG reseed by 0.27**. So there the
gain is the REGIME, not the constraint.

⛔ **"THE HEAD-TO-HEAD PATTERN IS THE CAP" IS WITHDRAWN 2026-09-04.** It read:
"TraLO is #1 in **2 of 3** cells at the tighter `L80_G95` and **1 of 3** at the
looser `L95_G80`. Where the cap binds hard TraLO leads; where it is slack `alm`
and `fioretto` overtake." **Every one of TraLO's `L80_G95` #1 calls was named
on a dead arm's distance** -- `dom1`/MNv2 on `fioretto`/`hounie` at -0.75 while
`tralo` and `alm` tie exactly at +4.25, `dom1b` on `hounie` at -5.75,
`equaldose1`/MNv2 on `hounie` at -10.25 -- and all three collapse to REFUSED
once those arms drop. The pattern was the dead arms, not the cap.
FRAMEWORK 2(z43). ⚠️ Note also that 2(z43)'s own premise, `L95_G80` being
"looser", is separately refuted: it and `L80_G95` emit 660 vs 661 predictions,
the same budget through a different SCOPE.

⛔ **DO NOT EXPAND THE GRID.** More datasets / backbones / class-counts is
written down in COVERAGE section 5 and is explicitly NOT queued. The only
question on the table is making `tralo` clear its reseed floor and lead the
duals. Two staged campaigns answer it: `vittask1` (running) and **`vitdual1`
(88 runs, the four duals on ViTB16 -- the paper's core comparison on the
paper's headline backbone, never yet run)**.

---

## 🛑 0-PAPER. WHAT THE CORPUS ACTUALLY SUPPORTS, 2026-09-01

Broken to paper-level items with `scripts/paper_rows.py` -- one row per
(cell, contrast), NOTHING averaged over cells. 393 rows from `dom1` + `dom1b`
+ `loose1` + `equaldose1`. FRAMEWORK 2(z26) has the full tables.

**The number that decides how this is written up:**

> **1 of 158 strict-task rows separates from its own seed noise at 2 sd**, and
> that sd is within sqrt(2) of the truth in either direction, NOT the
> "6-12x lower bound" this line used to claim (2(z32)). Everything else we
> quote is a SIGN, not a
> measurement.

**The evidence is sign consistency over FOUR independent units, not 8 cells:**

| contrast | units | sign p |
|---|---|---|
| `tralo` vs its own null (attribution) | **4/4** | **0.0625** |
| `tralo` vs `clip` (the quality bar) | 3/4 | 0.3125 |
| `tralo` vs `tralo_reseed` (RNG floor) | 3/4 | 0.3125 |

⛔ **AND ONLY THREE OF THE FOUR CARRY A VERIFIED `task` CELL (2026-09-04).**
Unit C1 (`taskwin2`/MobileNetV3) contributes `no_strict_band` + `unmeasured`
and nothing else, so restricted to task-carrying units this is
**3/3 units, p=0.125**, every sign unchanged. Quote both, and take the
restriction from `scripts/paper_rows.py`, which prints it.

> ⚠️ **THOSE ARE SIGN-READ COUNTS, AND THE LEDGER NOW LICENSES SIX
> (2026-09-09).** Units C2 (`dom1`/MobileNetV3) and D1 (`bcn1mn3`/MobileNetV3)
> are licensed and UNREAD, so every `n/4` above is a numerator AND a
> denominator that can still move. Reading both costs zero GPU-hours -- both
> are on disk. Reading both gives ⛔ **REFUTED BY MEASUREMENT 2026-09-10 -- 2(z86). D1 WAS READ AND IS NEGATIVE.** The tally is **5/6 p=0.109** (unrestricted, mean rule) at best and **3/4 p=0.3125** task-restricted; the `worst-cell` rule gives 3/6 p=0.656. C1 and C2 are SPLIT too. Nothing clears 0.05.** **6/6 p=0.0156 unrestricted, 4/4 p=0.0625
> task-restricted** -- C2's strict band on class 2 is measured EMPTY so it can
> never carry a `task` cell, and the `n/4` counts above ARE the restricted
> ones (2(z75)). A single dissent takes either the other way. ⛔ DO NOT
> QUOTE 6/6 BEFORE IT IS READ, AND NEVER WITHOUT ITS RESTRICTION.
> FRAMEWORK 2(z66), 2(z68), **2(z75)**, task #102.

* 🔑 **0.0625 is the FLOOR at four units.** No amount of agreement in this
  corpus reaches p<0.05. **The bar is crossed by adding a FIFTH INDEPENDENT
  UNIT, not by another knob.** That is exactly what `taskwin2` (MobileNetV3,
  which has ZERO task cells today) and `vittask1` (ViTB16, the headline
  backbone, also ZERO) are for. They are the highest-value runs available.
  ⛔ **AND `taskwin2` DID NOT BUY IT (2026-09-04).** It completed, and both its
  cells classify non-task -- `L70-90_G95` -> `no_strict_band`, `L80-100_G95` ->
  `unmeasured` -- so unit C1 carries no verified `task` cell. `vittask1` is the
  live candidate.
* ⛔ `B2` (`loose1`/RegNetY400MF/`L80_G95`) dissents on all three contrasts.
  It goes in the table.
* ⛔ **Dominance over the rival duals is NOT MERELY UNSHOWN, IT IS
  UNANSWERABLE (2026-09-04).** This read: "`tralo` is #1 of four in **3 of 6**
  strict cells. The `dom1` 'leads all four' reading included `L90_G95`, now
  PARTIAL." Two of the four rivals are DEAD ARMS at 28.00 steps, so there is no
  field of four to be #1 of. Recounted as deployed: **2 of 15 cells namable,
  both `alm`, TraLO 0** (FRAMEWORK 2(z43)). The only surviving rival dual is
  `alm`, and `vitdual2` is the only campaign that can restore the other two.
### 🎯 THE COROLLARY: TWO STAGED CAMPAIGNS CROSS p<0.05, AND NOTHING ELSE DOES

At `n` unanimous units the one-sided sign test is exactly `0.5^n`:

| units | p | |
|---|---|---|
| **3** (today, restricted to units carrying a verified `task` cell) | **0.125** | the honest floor -- see the `taskwin2` correction below |
| 4 (today, every unit the ledger licenses) | 0.0625 | above the bar, and it CANNOT go lower |
| **5** | **0.03125** | **below** |
| 6 | 0.01562 | below |

Verified against `configs.task_cells.classify` on 2026-09-01 -- and the
`taskwin2` rows RE-VERIFIED 2026-09-04, where the first one FELL:

| campaign | backbone | cap | status | buys |
|---|---|---|---|---|
| `taskwin2` | MobileNetV3 | `L70-90_G95` | ⛔ **`no_strict_band`** (read **task** on 2026-09-01) | **NOTHING** -- unit 5 never arrived |
| `taskwin2` | MobileNetV3 | `L80-100_G95` | `unmeasured` | nothing -- c7 sits at K/n 0.950 |
| `vittask1` | ViTB16 | `L60-90_G95` | **task** | **unit 6**, and it is the HEADLINE backbone |
| `vittask1` | ViTB16 | `L70-90_G95` | **task** | same unit (one campaign, one warm-up) |

⛔ **THE FIRST ROW WAS TRUE ON 2026-09-01 AND FALSE ON 2026-09-02, AND IT IS
THE ROW THE TALLY RESTED ON.** The cap screen behind the 09-01 reading counted
the PRIZE over a GLOBAL top-K while every allocator here is per-group;
re-measured with the per-group prize, MobileNetV3 class 2's strict band is
**EMPTY** on the dsisco01 model `taskwin2` uses -- at every 0.1-grid fraction
either the cap binds 4/4 and the local prize is under the 3.0-item floor, or
the prize clears the floor and the cap has gone slack in some seed. So
`taskwin2` / MobileNetV3 -- ledger unit **C1** -- contributes **ZERO** verified
`task` cells, and the deciding experiment is `vittask1` alone.

🔑 **`no_strict_band` IS A MEASUREMENT; `unmeasured` IS AN ABSENCE; NEITHER
IS `non_task`.** C1's other cell, `L80-100_G95`, is the opposite failure: c7
sits at K/n 0.950, a fraction nobody has looked at. Do not collapse the three.

⚠️ **CONSEQUENCE FOR THE TALLY, AND BOTH NUMBERS BELONG IN ANY WRITE-UP:**
**4/4 units, sign p=0.0625** over the units the ledger licenses;
**3/3 units, sign p=0.125** once restricted to units carrying a verified
`task` cell. The SIGNS do not change -- dropping C1 flips nothing, and it
removes the one unit that was FAILING the `vs tralo_reseed` contrast, so the
corpus gets CLEANER and LESS significant at once. `scripts/paper_rows.py`
computes the restriction itself and prints "UNITS CARRYING AT LEAST ONE
VERIFIED `task` CELL: N of M"; read it there rather than re-deriving a number
that has now gone stale twice.

Both are already staged, single `code_version`, `constraint_fp32: true`,
warm-up 1 / constraint 29, six arms including `tralo_null` and
`tralo_reseed`. `taskwin2` is at 39/48 with dose **203/203 and 174/174**.

🔑 **So the deciding experiment is not a new idea -- it is finishing the two
campaigns already on the disk.** No knob, no loss variant and no extra cap
level moves the headline p below 0.05; only a fifth and sixth independent unit
does. Anything that delays those two campaigns costs the result directly.

⚠️ **And it can go the other way.** Unit 5 disagreeing takes 4/5 to p=0.1875 --
WORSE than today. The two campaigns are the test, not a formality.

🔑 **AND UNITS 7 AND 8 ARE FREE.** The unit is `(backbone, HOST)`, measured
2026-09-01 (FRAMEWORK 2(z27)): there are exactly TWO null models per
(backbone, seed) across all 14 worktrees, and they are dsisco02/bfloat16 vs
dsisco01/float16. So `taskwin2` and `vittask1` re-run on **dsisco02** are units
7 and 8 at no design cost -- 6/6 is p=0.0156, 8/8 is p=0.0039. Blocked today
only because all four dsisco02 GPUs are held by other users.
⚠️ **AND THOSE ARE UNRESTRICTED COUNTS.** A unit only enters the
TASK-RESTRICTED tally if it carries a strict `task` cell, and `taskwin2`'s
MobileNetV3 has ZERO of them for the same reason C2 does -- the class-2 strict
band on that backbone is measured EMPTY (2(z75)). A ViTB16 unit does carry
them. Say which tally before quoting either p.
⛔ **And the converse: another MobileNetV2 or RegNetY400MF campaign on either
host buys NO unit.** It adds cells and moves no p-value.

* 🔑 **Power tracks `K/n`.** The single cell that resolves needs 2 seeds and is
  the highest-`K/n` cell present; `L80` cells need 13-37 and we run 4. If a
  cell must resolve on its own, run it at high `K/n` or run 10+ seeds.

---

## 🔴 0-NOW. THE TWO DEFECTS, FOUND 2026-08-31/09-01

Four measurements, in the order they were made. FRAMEWORK 2(z11), 2(z12),
2(z16), 2(z17). (1)-(3) are the AIM of the gradient; (4) is the PLACEMENT of
the cap, and they are independent.

**(1) At the item level the constraint is at the RNG floor.** `boundary_probe
--control tralo_null`, every arm against its OWN lambda=0 twin. A pure reseed
moves **3357** items where the constraint arms move 3362-3647, and nets **+89**
at tight / **+167** at loose against `tralo_uniform`'s **-43** / **+148**. So
`tralo_uniform` does not clear its own noise floor in either regime; only
`tralo` at loose (+221) and `alm` (+255) do.

**(2) `tralo_uniform`'s founding claim is false.** Its docstring argues a
uniform step in log-odds is "a pure bias shift, which cannot reorder". The step
is taken in PARAMETERS, not logits: `dz_i = -lr*g*n*(fbar.f_i + 1)`, which
varies with `fbar.f_i`. It reorders, with the backbone FROZEN -- the leak is in
the linear head. `scripts/bias_shift_probe.py`. The only provably harmless
update is one confined to `b_c`, and that one is useless: a constant added to
`z_c` leaves the within-class order untouched, so the emitted top-K is
bit-identical.

**(3) AND THE ROOT CAUSE. The shipped count puts 0.00% of its gradient at the
cut.** `p(1-p)` is maximal at p=0.5 and vanishing at p=1; the tight-cap cut sits
at **p = 0.99984 to 1.00000**. Fraction of gradient mass on the 40 items
straddling rank K, measured on real stored features over 24 (run, class) pairs:

| weighting | mass at the cut |
|---|---|
| `cut_window` | **0.3486** |
| `p` | 0.1039 |
| `uniform` | 0.0136 |
| **`sum` -- THE SHIPPED COUNT** | **0.0001** |
| `margin` -- the BOUNDARY window | **0.0000** |

⇒ The penalty spends its whole budget where movement cannot change the
emitted set, and nothing where the metric reads. **That is (1) explained**: the
reordering is at the RNG floor because it is arbitrary with respect to the
metric. It also **derives the regime reversal with no new assumption** -- at
loose caps the cut falls to p=0.59-0.99 where `p(1-p)` finally has mass, which
is exactly where `sum` wins.

🛑 **AND IT PRICES `margin2` BEFORE IT RUNS.** `tralo_margin` windows the
DECISION BOUNDARY, puts **exactly 0.0000** at the cut, and sits at cosine
**0.989** from `tralo` -- so its 432 staged runs would mostly reproduce `tralo`.
That is CLAUDE.md rule 3's conflation costing a campaign. Run the cut window
first.

**(4) AND THE SECOND DEFECT IS THE CAP, NOT THE MODEL. Every campaign this
project ran cut ABOVE the region where the model is uncertain.** Roei's worry
was that warm-up CE saturates and leaves no wiggle room at the constraint
border. Measured on all four backbones, `tralo_null`, iwildcam (2(z16),
2(z17)): a cap poses a question only when it BINDS (evicts >= 10), has a PRIZE
(errors inside K) and has WIGGLE (p@K < 0.99).

| K/n | errors@K over the 8 (backbone, class) | p@K | cells that are a TASK |
|---|---|---|---|
| 0.20 | 0.0 - 2.5 | 0.99978 - 1.00000 | **0 / 8** |
| 0.30 | 0.0 - 3.0 | 0.99945 - 1.00000 | **0 / 8** |
| 0.50 | 0.0 - 7.8 | 0.99381 - 1.00000 | **0 / 8** |
| **0.90** | **14.5 - 43.8** | **0.48820 - 0.96096** | **8 / 8** |

⇒ **24 of 24 cells at L20/L30/L50 pose no question, and 8 of 8 at K/n=0.90 do.**
The saturation is real but it is LOCAL: move the cut to 0.90 and p@K falls from
~1.0 to 0.49-0.96 with 14-44 fixable errors appearing. **The wiggle room was
always there and every campaign cut above it**, ViTB16 included -- at L20/L30
both its capped classes have literally ZERO errors inside K.

🔑 **This is the best explanation on record for why ~20 arms tied**, and it is
independent of (1)-(3): (3) says the gradient is aimed away from the cut, (4)
says the cut was placed where there is nothing to win. Both had to be fixed
before a null means anything. `taskwin2` is the first campaign with both fixed.

✅ Two independent lines now name the same cap: `paired_noise` prices K/n=0.90
at ~7 seeds per cell against 546-2607 at L20/L30/L50, and the task window says
0.90 is the only single fraction that is a task for both classes on all four
backbones. **The cheap regime and the answerable regime are the same regime.**

⚠️ **WHAT IS NOT CLAIMED: that aiming at the cut WINS.** Necessary, not
sufficient. At tight caps the clipper's set is already 99.6% correct and
`headroom` reads 0.0-1.0 items, so a correctly-aimed gradient can still find
nothing to take -- it may fix the aim and still lose in the very regime it was
built for. What IS predicted is that the count trajectory responds where `sum`
measurably cannot.

---

## 0. THE GOAL, stated so it can be failed

Make **TraLO** the best of the constrained-optimization methodologies, on the
mathematics, and show it. Not "not worse". Not "wins on one metric on one
backbone at one cap". The bar the work is held to:

| axis | required | have now | gap |
|---|---|---|---|
| **datasets** | **3** | **1** (iwildcam) | `fmow` screened; the factorial gate DOES NOT APPLY (country is atomic) -- 2(w2c). Needs ~21k images + its own p@K. Third TBD |
| **backbones** | **3** | **4 exist**: MobileNetV2/V3 (`dom1`), RegNetY400MF (`dom1b`), **ViTB16 (`vitu1` tight + `loosevit1` loose, already complete)** | not coverage -- CELLS. ViTB16 has 3 tight + 2 loose contrast cells and **no rival-dual arms at all** |
| **constraint pairs** | **varied**, both **equal and unequal** local:global ratios | 3, **all loose**, only 1 unequal-binding | `margin2`'s matched 2x2 (4 tags, 2 budgets x 2 scopes) closes this the moment a GPU frees |
| **consistency** | wins across **regimes**, not one | wins at L80-L95; **loses at L20-L50, and now we know why** | the mechanism is found (2(y)); the question is whether ANY count function can fix it |
| **metrics** | ccF1 **and** macroF1 both defensible | backbone-dependent: macroF1 **-0.0022 on MobileNet** (`dom1`) but **+0.0196 tight / +0.0021 loose on ViTB16** | the damage is a REPRESENTATION effect (2(z1)), not allocation -- and it is not universal |

🛑 **Winning only at L80/L90 is not a result.** If TraLO loses at every other
constraint pair, the claim is "TraLO helps when the constraint barely binds",
which is not the thesis.

---

## 1. WHERE WE ACTUALLY ARE (read the numbers, not the vibe)

### What is established

- **`dom1` (384 runs, complete, LOOSE caps, MobileNetV2+V3).** ⛔ **This said
  "TraLO is #1 of five on ccF1 / AP / AUROC, 6/6 cells each" until 2026-09-04.**
  Two of the five are dead arms, so the field is THREE, and on ccF1 `tralo`
  +0.0080 vs `alm` +0.0075 is a TIE at the top by this file's own one-item rule.
  As deployed, TraLO is #1 in **0 of 15** cells (FRAMEWORK 2(z43)).
- ⛔⛔ **"First campaign at equal dose. All five trained arms 100.0%; `hounie`
  672/672" IS THE EXACT OPPOSITE OF THE TRUTH, AND IT IS THE WORST SENTENCE
  THIS PROJECT WROTE.** **672 = 24 x 28 and 696 = 24 x 29.** The number quoted
  as proof of parity IS the defect. The "100.0%" is applied/attempted computed
  WITHIN each arm and is structurally blind to a gap BETWEEN arms. `dom1` is
  not the first campaign at equal dose; it is the first campaign whose dose was
  written down per arm, which is how the gap was eventually found. **The clause
  that survives is the last one: no earlier dual-vs-dual number is safe -- and
  now, neither is this one.** FRAMEWORK 2(x), 2(z40).
- **The four lambda=0 nulls are byte-identical 24/24**, so the compute term is
  shared exactly and arm differences are the method.

### What is NOT established, and must be said every time

| claim | reality |
|---|---|
| ~~TraLO > fioretto~~ | ⛔ **UNANSWERABLE 2026-09-04.** ~~AP 3/6 cells, p=1.00. A coin flip.~~ `fioretto` is a DEAD ARM at 28.00 steps in every campaign that ran it on the recipe |
| TraLO > alm **(the only surviving rival dual)** | 4/6 on everything, p=0.69. Not shown. As deployed, #1 in **0 of 15** cells (FRAMEWORK 2(z43)) |
| ~~TraLO > hounie~~ | ⛔ **UNANSWERABLE 2026-09-04.** ~~6/6 AP+AUROC on dom1, p=0.031, fails BH~~ -- `hounie` is a DEAD ARM at 28.00 steps. **The dom1b half SURVIVES via `alm`**, which is live: on RegNet `tralo` is **2nd of 3 on both AP and AUROC, behind `alm`** (+0.0314 vs +0.0458; +0.0044 vs +0.0069) and **below its own reseed floor** on each. The ranking lead does not reproduce. FRAMEWORK 2(z5) |
| Anything survives correction | **0 of 20 contrasts** -- and it is worse than that: the independent unit is **(model, seed) = 8**, not 6 cells, because a lambda=0 twin is byte-identical across cap tags. **8 of 9 dom1 sweeps evaporate at n=8**; only class 4's allocated damage survives (0/8, p=0.0078). FRAMEWORK 2(z) |
| macroF1 | **-0.0022, 2/6 on MobileNet** -- but **+0.0196 (3/3) tight and +0.0021 loose on ViTB16**, against a reseed floor of -0.0366 loose. Backbone-dependent, and the damage is REPRESENTATION drift, not allocation (RAW -0.0107 is 44% LARGER than deployed -0.0074) |
| TraLO enforces better | **REFUTED.** ⛔ This said "Pulls +6.2 items vs hounie +23.4. The WEAKEST of the four" until 2026-09-04; `hounie` is a dead arm. **The finding survives on `alm`, which is live: +6.2 items against `alm`'s +17.8, a third of the enforcement, and still the weaker of the two comparable arms** |
| Constraints ever satisfied in training | **0 of 696 epochs.** The post-hoc allocator does all of it |
| dom1 is the headline | **No.** FRAMEWORK 1-pre fixed **ViTB16** a priori; dom1 has none |

### The regime step is REAL. The explanation for it is NOT. (corrected 2026-08-30)

✅ **The solid part.** The CNN warm-ups are shared across campaigns, which
gives a within-model tight-vs-loose contrast with no confound. Paired on the
**12 warm-up models present in both regimes**: `tralo` moves **+6.24 items
from tight to loose, 12/12, sign p = 0.00049** (the exact floor at n=12), while
**the reseed floor does not move at all** (5/12, p = 0.774). Floor-corrected,
+5.30 items, 12/12. This is the cleanest attributable result the project has.

⛔ **The part that failed.** The geometric account -- that the penalty aims at
the decision boundary while the metric reads the cut, `gap = hard - K` -- is
**consistent but not discriminated, and its sharp prediction failed**:

* `gap`, `slope_K` and `K/n` are one variable in three costumes within a model
  (`rho(gap, K) = -1.0000` exactly, hard count constant in 40/40 groups).
* Both `gap` and `slope_K` **reverse sign** once the cap is held fixed.
* `tralo_uniform` was predicted to order oppositely. It does not -- same sign
  at every level.

⇒ quote the geometry as an unrefuted account, never as a measured cause.
Testing it needs `gap` varied at FIXED `K/n`. FRAMEWORK 2(y).

🛑 **And the absolute loose-cap win does not survive honest units.** At the
cell level `tralo - null` reads 15/20 (p=0.041); at the **16 distinct warm-up
models it is 11/16, p = 0.21**, and it beats the reseed floor 11/16, p = 0.21.
**macroF1 and uncapped F1 are NEGATIVE in 11 of 16 cells.** The relative
(loose-minus-tight) statement survives; the absolute one does not.

⚠️ **`dom1` is not independent of `loose1`.** Its L80_G95 and L90_G95 cells
are byte-identical to `loose1`'s in 80/80 files; `dom1` contributes only
L95_G80. Only **20 distinct warm-up models** exist across all five campaigns.

## 2. THE KNOB LEDGER -- what has been tried on TraLO itself

✅ = keep · 🟡 = live, unresolved · ⛔ = rejected, **do not retry**

| knob | verdict | evidence |
|---|---|---|
| `soft_count_mode: sum` (shipped) | 🟡 wins LOOSE, loses TIGHT -- **and the reason is now known**, 2(y) | AP +0.0253..+0.0064 loose / -0.0572..-0.0933 tight. The gradient sits at the boundary, 200-440 ranks from the cut when the cap is tight |
| `soft_count_mode: uniform` | ⛔ **DO NOT RUN AGAIN 2026-09-01** (weaker than 'refuted': see the cross-campaign count). Was logged as "the tight-cap tool", but tight caps are now measured NON-TASKS (2(z17)) and in the cells that ARE tasks it clears its own reseed floor in **3 of 8** of the cells 2(z23) counts (`loose1` 1/5, `dom1b` 2/3) and in **0 of 4** `dom1` task cells (2(z21)) -- **3 of 12 overall** and is **-3.50 items against a 0.51 floor on ViTB16** (2(z20), 2(z21), 2(z23)). It never LEADS anywhere. Its founding order-preservation claim was also refuted (0-NOW (2)) | old row: ViTB16 AP +0.0087 tight / -0.0091 loose, `uniform1` -0.0754 -> +0.0030. Those tight-cap numbers stand as measurements and no longer support the verdict |
| `soft_count_mode: margin` | ⛔ **NEVER RUN, AND NOW REPRICED DOWNWARD** (2026-09-01). **NOT staged** -- checked 2026-09-02, no `margin2` exists on disk anywhere; this line said it was. It windows the BOUNDARY, and the boundary is measured to carry **exactly 0.0000** of the gradient at the cut. 🛑 **Run `taskwin2` first** | cosine **0.989** to `tralo` on real features, so 432 runs would mostly reproduce `tralo`. FRAMEWORK 2(z12) |
| `soft_count_mode: cut` (`tralo_cut`) | ⛔ **REJECTED on its first campaign, 2026-09-02.** `taskwin2` landed 48/48 at 232/232 dose and `tralo_cut` is WORSE than `tralo` in **both** cells on **all three** contrasts. In the `L70-90_G95` cell -- called the STRICT task cell when this was written, re-measured `no_strict_band` 2026-09-02 -- it is NEGATIVE against the clipper. Aiming the gradient at the cut was necessary and is now measured to be insufficient. 🟡 One confirmation outstanding: `vittask1` runs it on ViTB16 and will give a second backbone before this is final | cell `L70-90_G95` (⛔ `no_strict_band`, NOT strict -- re-measured 2026-09-02): `tralo_cut` **-0.46 / -7.02 / -8.05** items (vs clip / null / reseed) against `tralo` **+7.32 / +0.75 / -0.27** -- **7.8 items behind on every contrast**. Unmeasured cell `L80-100_G95`: **+6.64 / +1.24 / -0.95** against `tralo` **+10.86 / +5.47 / +3.28** -- 4.2 behind. The build was sound (mass at the cut 0.0001 -> 0.3486, chunked gradient == full-N exactly, md5-distinct on every binding seed); the HYPOTHESIS was wrong |
| `tralo_st` (hard-count value fix) | ❓ **NEVER RUN** -- same campaign | isolates VALUE from PLACEMENT |
| `straight_through` | ✅ keeps count value exact | -- |
| `constraint_grad_mode: normalize` | ✅ **required** -- `clip` gives a ~20x dose spread across duals | `check_parity` refuses `clip` |
| `--constraint-fp32` | ✅ removes the FP16 skipped-step dose loss | iwc3 lost 328/1044 without it |
| `tralo_head` (head-only) | ⛔ 1.7x floor, tie uninformative; masking does not freeze the backbone (90.4% step) | |
| `tralo_ortho` (CE-orthogonal) | ⛔ delivers **0.0%** of its guarantee in 16/16 conditions | `ortho_survival` |
| `tralo_coin` (random direction) | ❓ never run -- **the control** for any placement claim | in `launch_margin2.sh` |
| penalty-shape variants | ⛔ FRAMEWORK 2 | all measured worse |
| more constraint steps | ⛔ **worse** | 2(c) |
| dedicated constraint optimizer | ⛔ | 2 |
| joint objective | ⛔ overfits, -0.067 AP | |
| undershoot hinge | ⛔ not budget-equalized; +16.3% free fill | |
| finer granularity (LLP) | ⛔ refuted | |
| KL anchor | ⛔ deleted from the pipeline | |
| `select` arm | ⛔ worst measured, -22 items | |
| `rank` / `beta` arms | ⛔ null / rejected | |
| cut-centred count `sigma((p-tau_K)/T)` | ⛔ **CLOSED BY ALGEBRA** -- counts items above the K-th largest = K-1 for ANY model. Detaching `tau` gives a gradient but was not shown to measure the violation | `margin_window` docstring; re-checked 2026-08-30 |

🎯 **The next knob is `margin` + `st` + `coin`** -- `docs/launch_margin2.sh`,
**432 runs, 12 cells**, re-validated 2026-08-30 (`gen_campaign` emits 432,
`check_parity` PASSES), never fired. `margin` is the only untested corner of
the count-function 2x2 and the only arm whose per-item gradient is not a
function of `p_ic` alone -- every other penalty this project ships has the form
`f(sum_i p_ic)`, whose logit gradient `f'(S) p_ic(1-p_ic)` is a monotone map
and therefore **cannot move an item across another on the direct channel**.

Its cap grid is a **matched 2x2**, which is what makes it answer the regime
question rather than just adding cells:

| tag | K (cls 2 / cls 7) | budget | what is pinned |
|---|---|---|---|
| `L30_G50` | 111 / 137 | K/n=0.30 | the DISTRIBUTION across groups |
| `L50_G30` | 111 / 137 | K/n=0.30 | only the TOTAL |
| `L80_G95` | 296 / 364 | K/n=0.80 | the DISTRIBUTION across groups |
| `L95_G80` | 296 / 365 | K/n=0.80 | only the TOTAL |

Each row-pair imposes the **same total budget through a different scope**, so
scope is isolated with tightness held fixed. ⚠️ 7 of 14 per-group ceilings are
K=0, and a zero ceiling binds however much slack the sum has -- so
"global-binding" never means the local scope is off. Say "pinned vs free
distribution", not "local vs global".
⛔ Do NOT add `L30_G30`: at `L30_G50` the global K=185 sits above the local sum
111, so the global term is INERT and the two tags are ONE cap level.

---

## 3. THE STANDING RULES THIS WORK IS HELD TO

1. **Never idle.** A campaign running is not a reason to stop; it is a reason to
   do the cheap offline work beside it.
2. **Cells, not seeds.** 4 seeds cannot resolve any of these effects (46-91
   needed). Everything rests on sign consistency across cells. **>= 9 cells** for
   a `***`, and **>= 10** if more than a couple of contrasts are tested.
3. **Pre-register ONE primary contrast** before scoring. 20 contrasts at 9 cells
   still cannot survive BH. This is the cheapest fix in the project.
4. **Always quote the `tralo_reseed` floor beside any win.** A 6/6 sweep is not
   evidence when the RNG floor also sweeps 6/6.
5. **Always quote macroF1 beside ccF1.** ccF1 alone hides the uncapped damage.
6. **Read the logs, but never compare counts across arms from
   `training_log.csv`** -- the schemas differ (76/16/15/14 cols) and trained arms'
   logged counts disagree with their predictions. Use
   `final_predictions_raw.csv`. FRAMEWORK 3(0c).
7. **md5 the raw predictions** before reading any metric (`_raw` = model, plain
   = allocator).
8. **Update this file and FRAMEWORK 3(0) at the end of every session.** A knob
   that failed goes in the ledger so it is never retried.

---

## 4. THE QUEUE -- in priority order

Work top-down. When one finishes, score it, update sections 1-2 of this file
and FRAMEWORK 3(0), then start the next.

⚠️ **THE LIVE QUEUE IS ITEMS 0-6 BELOW, REBUILT 2026-09-10. EVERY ITEM NAMES
ITS TASK ID, AND THE TASK LIST -- NOT THIS FILE -- IS THE AUTHORITY ON WHETHER
IT IS DONE.** The previous live queue was dated 2026-09-07 and all five of its
items had completed: items 1 and 3 were `dualprop1` reads that produced 2(z53),
item 2 was the `dualprop2` launch, items 4 and 5 were the `deep_scope` and
`headroom` recomputes. A queue that says "work top-down" whose top is finished
work is the same defect as a run-state block dated at writing. FRAMEWORK
2(z72).

0. 🔴 **RE-ESTABLISH CONTACT.** SSH to both hosts has been down since
   2026-09-06; the jump host `dsihead.lnx.biu.ac.il` (132.70.60.180) shows 100%
   packet loss, so it is the VPN and not a host. **Nothing in `0-RUNNING` has
   been verified since 2026-09-09.** On reconnect: VERIFY before relaunching
   anything -- `0-RUNNING` carries the LANDED table for exactly this reason.

1. 🟢 **#102 -- READ THE SIGNS FOR UNITS C2 AND D1. ZERO GPU-HOURS, TOP OF
   QUEUE.** Both are on disk. `("dom1","MobileNetV3")` and
   `("bcn1mn3","MobileNetV3")` are LICENSED in `MEASURED_UNITS` and have never
   had their signs read; `bcn1mn3` is a COMPLETE 228-run campaign currently
   contributing nothing.
   🛑 **READ D1 FIRST, AND SAY WHICH TALLY (2026-09-10).** C2's strict
   band on class 2 is measured EMPTY, so it reads `partial` at every cap and
   can never carry a strict `task` cell -- and `paper_rows` restricts its sign
   test to `task` units. Reading both gives ⛔ **REFUTED BY MEASUREMENT 2026-09-10 -- 2(z86). D1 WAS READ AND IS NEGATIVE.** The tally is **5/6 p=0.109** (unrestricted, mean rule) at best and **3/4 p=0.3125** task-restricted; the `worst-cell` rule gives 3/6 p=0.656. C1 and C2 are SPLIT too. Nothing clears 0.05.** **6/6 p=0.0156 unrestricted, 4/4
   p=0.0625 task-restricted**; the sub-0.05 number is the one the paper-facing
   scorer does not print. `bcn1mn3` L80/L90 ARE verified task cells (2(z58)),
   so D1 is the one unit that moves both rows. A negative is worth more than
   either. ⛔ **DO NOT QUOTE 6/6 BEFORE IT IS READ, AND NEVER WITHOUT ITS
   RESTRICTION.** 2(z66), **2(z75)**, 0-UNREAD.

2. 🔴 **#108 -- RESOLVE 2(w3) vs 2(z53). THE ONLY POSITIVE RESULT AND THE SHARPEST NEGATIVE ONE MEASURE THE SAME QUANTITY WITH OPPOSITE SIGNS.** `loose1` gives `tralo` vs its own null AP **+0.0253, 5/1**, and what survives its control is specifically the RANKING; `dualprop1` gives **-0.0138 / -0.0160 / -0.0051, 6 of 6 negative** on the same contrast. They overlap at MobileNetV2/L80. TWO unchecked things decide it: WHICH cell was the 1 of that 5/1 (never identified anywhere), and whether they still disagree once the `+0.0253` is split per backbone -- it pools THREE, which rule 4 forbids.
   🔑 **AND THE CHECK IS A LOOKUP, NOT A CAMPAIGN.** `loose1`'s six cells rest
   on THREE warm-up models (two cap levels in one campaign share one), and FOUR
   of the six are BYTE-IDENTICAL to `dom1`'s -- 96/96 md5 on MobileNetV3 +
   MobileNetV2 at both caps, embeddings too -- i.e. units **A1** and **C2**.
   Only the two RegNetY400MF cells are `loose1`'s alone (`dom1b` vs `loose1` is
   0/12). So the candidate set for the negative cell is SIX, four of which can
   be read off `dom1` instead. Zero GPU, both campaigns complete and on disk.
   2(z74).
   ⛔ **AND `loose1` IS THE `clip` RECIPE, WHICH 2(w3) NEVER SAID
   (2026-09-10).** COVERAGE section 0 puts it in recipe row 2 with `iwc4`
   `loosevit1` `vitu1`, and 2(z26-CORRECTED) removed `loose1`/RegNetY400MF
   from the unit corpus BY NAME as *a different method* -- while the same
   campaign stayed the headline positive. The correction was applied where the
   campaign hurt and never propagated to where it helped. Reading:
   the 4 CNN cells are byte-identical to `dom1`'s so `clip` and `normalize`
   provably coincide there and the recipe cannot touch them -- **which is also
   why they are not independent of `dom1`**; the 2 RegNet cells ARE the
   excluded unit. So drop the RegNet row when splitting, and read the other two
   as `dom1`'s. On the current recipe 2(w3) is a **two-unit** result and both
   units are `dom1`. The SIGN is untouched; the INDEPENDENCE and the SIZE are
   not. **2(z76)**.

3. 🛑 **#104 -- RECOMPUTE THE ACCEPTANCE TABLE.** `6 of 17 = 35%` was produced
   at 09:09 on 2026-09-06; `deployed_h2h.rank_cell`, which makes the deltas
   BOTH halves of that verdict read, was fixed at 22:48 the same day. Until it
   is re-run, say **"FAIL, figure pending recompute"** and never the figure.
   2(z68).

4. 🔑 **#107 -- RESOLVE THE 0.013-vs-0.258 COSINE. THIS GATES THE ONLY
   UNCLOSED MECHANISM LAYER.** 2(z56) closed the per-scope weighting family and
   2(z53) rejected `tralo_dualprop`; both act UPSTREAM of `normalize`, which
   discards magnitude -- which is why `tralo_coin`, a random direction of the
   same norm, took a cell. The delivery layer is what is left, and its entire
   motivation is a cosine with no citation that a real-backbone measurement
   contradicts 15-20x. `step_dose` now reports `r` and `cos(m_ce, ghat)`,
   the one measurement that decides it. 2(z73).

5. **#106 -- RE-READ 2(z53) AT COMPLETION.** `dualprop1` was 72 of 88 runs when
   it was scored, so every cell carries 3 seeds against the protocol's 4. The
   DIRECTION is safe (sign 6 of 6, the rejection stands); every NUMBER is not.
   Zero GPU if the campaign has since finished.

6. **#105 -- MAKE THE EXISTING CORPUS PRICEABLE.** Every corpus campaign
   predates `tralo_reseed2`, so its RNG floor rests on 2 streams x 4 seeds = 4
   observations against `MIN_FLOOR_OBS` = 8, and every `priced` column reads
   false because the comparison is never reached -- not because the spread lost
   to the floor. 16 runs per campaign fixes it. 2(z69).
   ⚠️ **AND SAY WHAT IT DOES NOT BUY (2026-09-10).** It buys
   `MIN_FLOOR_OBS`, which is the whole point, and it does NOT buy the ALM
   comparison: at 8 seeds the minimum detectable effect is still **4.4-9.6
   deployed items** against a 1.42-item gap, which needs **78-362 seeds per
   cell**. Selling seeds 5-8 as "now we can resolve TraLO vs ALM" would be the
   2(z69) defect again in the other direction. 2(z77), task #109.

7. 🟢 **#111 -- READ THE THREE GENUINE STALE-PROVENANCE HITS. ZERO GPU,
   DOCUMENT READS.** `python -m scripts.stale_provenance` (new, 2(z78)) asks
   the staleness question `stale_figures` cannot: not "has the SCORER moved"
   but "was the DATA condemned". 74 entries disclose, 32 do not; the top six
   were hand-read at **3 genuine / 2 spurious / 1 ambiguous**, so it is a
   QUEUE and never a count. The three:
   `(z12)`'s 53 figures are `iwc1`'s and sit outside its `keep_for`;
   `(z8)`'s count-function reversal is on `loose1`, the `clip` recipe
   (apply 2(z76)'s cell-by-cell reading); FRAMEWORK:6833's attribution
   analysis is on `iwc3` at 68.6% dose (2(z71) already gives the split --
   the lambda=0 columns are dose-immune, the treated ones are not).

8. 🟢 **#110 -- RUN `straddle_probe` ON `dom1`. IT SETS THE DENOMINATOR
   OF THE BAR ABOVE AND HAS NEVER BEEN RUN.** 2(o) still calls it "an
   INSTRUMENT not yet a result". The `11.7-21.2` prize is `headroom`'s ORACLE
   quantity; 2(a3) measured that the delivered displacement is exactly
   `lr*clip` per step, so part of it was never reachable. A smaller reachable
   prize makes the bar HARDER, so this decides whether "half the prize" is
   achievable at all. Zero GPU.

9. 🟢 **#109 -- MEASURE THE sd INSTEAD OF ESTIMATING IT.** 2(z77)'s whole
   ladder rests on `median|d| = 0.6745*sd`, a normality assumption over two
   medians from different campaigns. `paired_noise --campaign results/dom1`
   and `ceiling_screen` already compute it directly and both currently quote
   the quarantined `iwc3` (2(z71)), so ONE re-run on `dom1` replaces every
   figure in 2(z77) §1 with a measured one AND closes 2(z71). Zero GPU,
   minutes of CPU, blocked only on host access.

---

🔑 **THE BAR A NEW MECHANISM MUST CLEAR, AND IT IS NOT THE GAP TO ALM
(2(z77), 2026-09-10).** At the protocol's 4 seeds the minimum detectable effect
is **6.2-13.5 deployed items** against a per-cell prize of **11.7-21.2** -- the
instrument's resolution and the entire prize are the same size, so every null
in this corpus is equally consistent with capturing a third of everything there
is to win. Therefore:

* ⛔ do not build for the **1.42-item** gap to ALM. It needs 78-362 seeds
  per cell and the protocol runs 4, so no mechanism of that size is provable
  on this design at any dose.
* 🟢 do build for **~6+ items per cell**, about HALF the smallest cell
  prize. That is certifiable at **5-22 seeds** -- one `add_seeds` extension,
  no new design.

The constraint-gradient expression itself is closed on both factors (2(z56)
§5 for `A_S`, §6 for `p(1-p)`), so a mechanism that clears this bar has
to act somewhere else: DELIVERY (task #107), the SNAPSHOT (`tralo_snap`, #97),
or the dataset.

---

### The older queue, kept for its reasoning

0. 🟢 **RUNNING (204/216): `equaldose1`** -- and it has ALREADY ANSWERED
   BOTH ITS OWN QUESTION AND `taskwin2`'s. FRAMEWORK 2(z19).
   ⛔⛔ **DOSE: REOPENED 2026-09-04.** This read: "Dose: closed, in TraLO's
   favour. `tralo` +0.0275 AP against `tralo_lam0` +0.0287 -- indistinguishable,
   so the 3.4% step head start is NOT the source of the lead." **`tralo_lam0`
   is itself at 28.00 steps and is a DEAD ARM in this campaign**, so the
   control built to close the objection is the one the defect landed on, and no
   dose-matched control survives anywhere. `vitdual2` (32/88) is the only
   campaign that can close it. See item 3 in 0-NOW for the design-vs-defect
   tension, which is recorded and NOT resolved.
   ✅ **4 of its 6 cells are inside the measured task window** (all three
   MobileNetV2 caps + `MobileNetV3/L90_G95`), and in those cells `tralo` leads
   **`alm`, the one surviving rival**, on ccF1 (+2.32 items vs -0.73, both
   clippers below -2.7) and is the ONLY arm with near-zero macroF1 damage
   (-0.0011). ⛔ **This read "leads every rival on ccF1 (+2.32 items vs
   `fioretto` +1.62, `alm` -0.73, `hounie` -2.30)" until 2026-09-04**; two of
   those three rivals are dead arms.
   🔴 **But the RNG reseed floor in those cells is 3.39 items against
   `tralo`'s 2.32 -- 0.68x, BELOW the floor**, and restricting to task cells
   makes that ratio worse, not better (1.10x over all 6).
   🔑 **The ordering CHANGES with cell selection**: `alm` is best on ccF1 in
   the non-task cells (+5.85) and second worst in the task cells (-0.73).
   ⚠️ Means over cells, not paired tests; the nulls' effective n is 2, not 6.
   Directions and ordering only -- no ratios until 216/216.
   (original framing below)
0b. 🟢 **RUNNING: `equaldose1`** (216 runs, dsisco01 GPU 3, pin `10d37518`).
   Is the dominance claim a 3.4% head start? `tralo` and `alm` attempt 29
   constraint steps per run, `fioretto` and `hounie` 28, at identical
   `constraint_epochs: 29` -- verified at the gradient level (epoch-1 grad norm
   3.09 vs **0.0**). `tralo_lam0` starts lambda at 0 so its first step carries a
   zero gradient exactly as theirs does. **The first thing to check on the
   first completed run is `Grad_Norm` at epoch 1: 0.0 for `tralo_lam0`, ~3.09
   for `tralo`. If not, the arm is inert and the campaign is void.**
   ✅ **VOID CHECK PASSED 2026-08-30 on the first completed run**
   (`MobileNetV2/L80_G95/tralo_lam0/seed_1`). Epoch-1 `Grad_Norm`:
   `tralo` **2.16** (steps), `tralo_lam0` **0.0**, `fioretto` **0.0**,
   `hounie` **0.0** -- the control now matches the duals exactly. And it is
   NOT the null: its `Lambda_Global` rises 0.025 -> 0.05 -> 0.075 and it steps
   from epoch 2 (grad 18.69), where `tralo_null` stays 0.0 forever.
   ⚠️ 8 independent (model, seed) units, so only 8/8 (p=0.0078) is significant;
   7/8 is p=0.0703 and is a DIRECTION. Say which was met.
1. 🟢🟢 **RUNNING NOW: `taskwin2`** (48 runs, dsisco01 GPU 3, ours exclusively,
   `~/optloss-cutwin`, tree pinned `6658ef8c`, dispatcher PID 18190).
   🛑 **`taskwin1` WAS KILLED AT 3/48 AND REGENERATED.** It was staged without
   `--constraint-fp32`, and its first trained run landed **20 / 29 = 69.0%** on
   `amp=float16` -- dead centre of `dose_landed`'s documented FP16 + GradScaler
   host signature. Measured across every completed run in every worktree:
   `constraint_fp32: true` is **15284 / 15284 steps over 532 runs and 6
   campaigns**, `false` is 86.9% over 189. `taskwin2` carries the flag and its
   `tralo` lands **29 / 29 = 100.0%** on the same host and the same arm.
   ⚠️ **`gen_campaign` DEFAULTS THE FLAG OFF**, which is how this happened; put
   `--constraint-fp32` in every launch line until that default changes.
   It replaces `cutwin1`, which was deleted: `cutwin1` used `L30_G50`, and 2(z16)/2(z17) then established that
   L30 poses no question on any backbone. **It is the first campaign in this
   project whose caps were chosen by MEASURING that the cap poses a question.**
   MobileNetV3 x {`L80-100_G95`, `L70-90_G95`} x {`tralo_cut`, `tralo`,
   `tralo_null`, `tralo_reseed`, `clip`, `focal_clip`} x 4 seeds, `normalize`
   so the arms differ in DIRECTION and not in dose.
   ✅ **THE BUDGETS ARE VERIFIED AGAINST THE TASK WINDOW**, not assumed:

   | cap tag | class 2 K/n | class 7 K/n | binding scope |
   |---|---|---|---|
   | `L80-100_G95` | **0.800** (K=296) | **0.950** (K=433) | GLOBAL |
   | `L70-90_G95` | **0.700** (K=259) | **0.901** (K=411) | LOCAL |

   🛑 **RE-MEASURED PER SEED 2026-09-01 -- HALF OF THIS CAMPAIGN IS
   NOT A STRICT TASK CELL.** FRAMEWORK 2(z24b). The windows above came from
   `task_windows.yml`, which still carried the MEAN-based ranges that 2(z24)
   retired. Reading `binds n/N` on the same reference runs, MobileNetV3's
   strict windows are class 2 **0.70 only** and class 7 **0.90 only**:

   | cap tag | class 2 | class 7 | verdict |
   |---|---|---|---|
   | `L70-90_G95` | ⛔ **NO STRICT BAND** (read `0.700 strict 4/4` here on 2026-09-01) | 0.901 **strict 4/4** | ⛔ **NOT A TASK CELL** |
   | `L80-100_G95` | 0.800 **PARTIAL 3/4** | 0.950 **UNMEASURED** | ⚠️ label it |

   0.950 is halfway between the strict 0.90 and the partial 1.00, ten times the
   0.005 snapping tolerance from either, so nobody measured that fraction.
   ⛔ **AND ON 2026-09-02 THE OTHER HALF FELL TOO.** This read "⇒ **The
   arm-vs-arm claim rides on the `L70-90_G95` half**", and that is WITHDRAWN.
   The cap screen behind it counted the PRIZE over a GLOBAL top-K while every
   allocator here is per-group; re-measured with the per-group prize,
   MobileNetV3 class 2 has **NO strict band at any 0.1-grid fraction** on the
   dsisco01 model `taskwin2` uses. `classify` returns `no_strict_band` for
   `L70-90_G95` -- a measured EMPTY band, which is neither the `unmeasured`
   absence of one nor `non_task`. **So `taskwin2` carries no verified `task`
   cell, and ledger unit C1 buys none either:** the headline is
   **4/4 units, p=0.0625** over the licensed set, and
   **3/3 units, p=0.125** over units with a verified `task` cell.
   `scripts/paper_rows.py` computes and prints that restriction -- do not
   re-derive it here.

   The `L80-100_G95` half is a second reading, conservative if positive (a
   slack seed dilutes toward zero) and NOT evidence of no effect if null. Say
   PARTIAL / UNMEASURED wherever it is quoted. `classify` now returns seven
   statuses (four when this was written) and `gen_campaign` prints the label,
   so this cannot be staged unlabelled again.

   🟢 **`vittask1` IS CLEAN ON BOTH CELLS** -- ViTB16 strict windows are
   class 2 0.60-0.70 and class 7 0.90, so `L60-90_G95` and `L70-90_G95` are
   both 4/4 on both classes. It is the FIRST ViTB16 campaign that is, which
   retires "ViTB16 has zero strict task cells" (true of L20/L30/L50 only).

   ✅ **EARLY GATES, RUN AT 24 OF 48 (2026-09-01) -- ALL GREEN.**
   - **DOSE 100.0%**: `tralo` 116/116 and `tralo_cut` 116/116 on
     `amp=float16`. The `--constraint-fp32` regeneration did its job; the
     killed `taskwin1` landed 69% on the same host and arm.
   - **RULE 3 (md5 across arms): `tralo_cut` is DISTINCT from `tralo` in 4 of
     4** completed (cell, seed) pairs. The new count function is LIVE, not a
     sixth inert flag. ⚠️ Do not skip this: `cb_lp` was byte-identical to
     `clip` in 24/24 with every config gate green.
   - **Consistency**: `clip`, `focal_clip`, `tralo_null` and `tralo_reseed`
     are byte-identical ACROSS the two cap levels, as they must be -- they
     share a warm-up and the cap is applied downstream by the allocator --
     while `tralo` and `tralo_cut` differ across caps, as they must.
   - The dispatcher interleaves the two caps, so both sit at 12/24: **no cell
     is complete and nothing is scorable yet.** Do not read `full_panel`
     until a cell has all 4 seeds on every arm.

   Windows on MobileNetV3: class 2 **0.70-0.90**, class 7 **0.90-1.00**. All
   four land inside. The two tags also differ in WHICH SCOPE BINDS, so the
   local-vs-global question is carried for free.
   ⚠️ **2 cells, so it CANNOT reach significance on any metric** -- the
   generator says so itself. It is a mechanism check, not a verdict, and must
   be reported as direction + per-cell consistency only.
   **What it must show before anything is built on it**, in this order:
   `dose_landed` 100% and dose-matched; `flag_live` md5-distinct from `tralo`
   on the real runs; `log_health` for collapse/divergence and the count
   trajectory vs K; then `full_panel --control clip` reading its CONSTRAINT
   DOSE block on the FIRST completed runs; then `boundary_probe --control
   tralo_null` against the `tralo_reseed` floor.
   🔑 **THE PRE-REGISTERED PREDICTION.** Both cap tags are now LOOSE-ish
   (K/n 0.70-0.95), and 2(z18) has now MEASURED what that costs: at these
   budgets `cut_window` sits at cosine **0.926-0.951** from `sum` (against
   0.716-0.728 at the tight caps), with only 4.5-6.9x its cut mass rather than
   thousands. So the tight-vs-loose contrast `cutwin1` was built around is GONE,
   `tralo_cut` is expected to behave like `tralo` here, and the prediction
   changes with it:
   > `tralo_cut` and `tralo` are aimed at nearly the same place here and should
   > behave ALIKE on the count. The discriminating quantity is the **EMITTED
   > top-K set** against the `tralo_reseed` floor: with the cap finally inside
   > the window, **at least one trained arm must clear the reseed floor on
   > `d capF1` in items**, in both cells. If NO arm clears the floor even with
   > a cap that is a measured task, then the cap placement was never the
   > binding problem and cluster C closes with it.
   🛑 **THIS IS THE ONE PREDICTION THAT CANNOT BE SATISFIED BY REPAIRING THE
   CAP.** Every prior null had the escape hatch "the cap was in the wrong
   place". 2(z17) removes that hatch for these two cells specifically, which is
   the whole point of running them before any grid.
   ⚠️ **MobileNetV3 only, on purpose** -- small first, per Roei 2026-09-01.
   The same two cap tags are ALSO inside ViTB16's measured windows (class 2
   0.60-0.90, class 7 0.90-1.00), so the ViTB16 extension needs no new cap
   design, only GPUs.
2. 🔴🔴 **ViTB16 AT SEVERAL TASK CELLS -- now the highest-value campaign.**
   2(z20): on the headline backbone, in the ONE cell that is a measured task
   (`loosevit1/ViTB16/L90_G95`), `tralo` is the only arm positive on AP, AUROC,
   ccF1 and macroF1, and its 1.41 items is **2.8x** its own reseed floor. One
   cell cannot reach significance; more task cells can.
   🔑 ViTB16's two windows overlap only at K/n=0.90, so more cells REQUIRE
   per-class caps (2(z16)). Generate with:
   ```
   python -m configs.gen_campaign --root results/vittask1 --datasets iwildcam \n     --models ViTB16 --caps L70-90_G95 L80-95_G95 L85-100_G95 --arms all+null
   ```
   ⛔ **AND IT REPLACES `vitdom2_vit`, WHICH IS STAGED WRONG.** That campaign
   sits at 0/108 in `~/optloss-vitdom2` with caps `L30_G50 L60_G95 L90_G95`:
   on ViTB16 only `L90_G95` is a task, so **2 of its 3 cap tags -- 72 of 108
   runs -- would measure nothing.** It predates the window gate, which now
   refuses exactly this. Do not launch it as staged.
2b. 🔴 **ViTB16 LOOSE, from 2 cells to >= 6.** (superseded framing, kept for
   its numbers) `loosevit1` already exists, is
   100% dose, md5-clean, single `code_version`, carries `tralo_null` +
   `tralo_reseed` + both clippers -- and on it **`tralo` is positive on every
   metric including macroF1** (AP +0.0064, ccF1 +0.0017, macroF1 +0.0021, all
   2/0) against a reseed floor that is NEGATIVE (AP -0.0113, macroF1 -0.0366).
   That is the best-looking result in the project and it sits on **2 cells,
   min attainable p = 0.500, NOT CALLABLE**. More loose cap tags on ViTB16
   (L85, L95, plus the matched `L95_G80`) is the cheapest route to a callable
   headline. **Highest value per GPU-hour available.**
3. ⏸️ **`vitdom1` -- HELD pending the 0-PRE decision.** 240 ViTB16 runs
   pointed at a paper section that may not exist. Ready and validated; do not
   launch until A/B/C is chosen.
   🔴 **`vitdom1`** (`docs/launch_vitdom1.sh`, 240 runs, 6 cells, validated
   2026-08-30). **ViTB16 has never run a single rival dual on iwildcam**, so
   the dominance claim cannot be reproduced on the pre-registered headline
   backbone. Six LOOSE cap tags, five distinct budgets plus the `L80_G95` /
   `L95_G80` scope pair at an identical K=296. Also takes `loosevit1`'s
   NOT-CALLABLE 2-cell positive to 6 cells. Deliberately loose-only, and
   deliberately without per-family nulls -- the header says why.
4. 🔴 **`margin2`** (`docs/launch_margin2.sh`, 432 runs, 12 cells, validated
   2026-08-30, pre-registration fixed). Now carries a falsifiable prediction
   from 2(y): gain in the 6 LOOSE cells, none in the 6 TIGHT ones.
   ⛔ Blocked on a GPU, not on readiness.
5. ✅ ~~ViTB16 needs rival duals~~ -- **this is now queued as `vitdom1`, item 2.**
   Kept here only so the hole is not re-discovered: `hounie`, `fioretto`, `alm`, `danits_lp` have
   **never run on ViTB16 on iwildcam** -- they exist there only in the dermmnist
   `vit_diag`/`vit_ceskip` campaigns, which are 86/97 pending on a dataset that
   is removed from disk. So the `dom1` dominance claim **cannot be reproduced on
   the pre-registered headline backbone** without new GPU time. This is the
   single biggest hole in the dominance story.
6. ✅ **`dom1b` -- DONE and scored.** 192/192, all gates green. The ccF1 lead reproduces on RegNetY400MF (2.49x the floor) but the **AP and AUROC lead does NOT** -- `tralo` is 4th and 3rd, both below its own reseed floor, with `alm` first. Confounded with the numeric regime (Blackwell bf16 vs Quadro fp16), so it is scored standalone. Nothing in it is significant and nothing could be: 4 warm-up units, sign floor p=0.125. FRAMEWORK 2(z5).
7. 🟡 **Unequal L:G ratios beyond L95_G80** -- `margin2`'s matched 2x2 covers two
   budgets; `L50_G20` / `L70_G40` would extend it.
8. 🟢 **`fmow` images (~21k)** -- the only route to dataset #2. Needs the user's
   go-ahead for the download.
   ⛔ **CORRECTED 2026-09-01 -- BOTH GATES ON THIS DECISION WERE
   RETURNING iwildcam's ANSWER.** FRAMEWORK 2(w2c), 2(z25).
   `factorial_control` scored `fmow_s1` at **100.1%** and that number was
   never measured: a country is ATOMIC, `--sep` never occurs in the label,
   so 0 of 10 groups were raked and the two arms were the same arm. The
   0.1% was the null draw. **8 of the 21 candidates read that way,**
   `iwildcam` included -- which means this gate had no positive control at
   all until `--self-test` grew a synthetic one. And `ceiling_screen`
   printed `PRIZE BELOW THE NOISE` for fmow off **iwildcam's** p@K curve,
   which it says does not transfer; it now refuses and prints the p@K to
   go and measure instead.
   🔑 **fmow is STILL the right ask** -- atomic group, 2(n)'s baseline
   sound, stage-1 NET +2766 at z=80.4. What changed is the GROUND: say
   "the factorial gate does not apply", never "it scored 100.1%". The one
   open number is **fmow's own p@K at the cap**, and that needs the images
   plus one unconstrained run -- there is no cheaper route to it.

⛔ **Do NOT re-run ViTB16 tight caps.** `vitu1` is complete, 100% dose, and says
`tralo` is 6.6x WORSE than the RNG floor there (AP -0.0933 vs -0.0142). 2(y)
explains why and predicts no count function fixes it. `iwc2` is also ViTB16 tight
but ran at **74.6% dose** (fp16 without `--constraint-fp32`) -- drop it.

---

## 5. RESUME PROTOCOL -- what to read, in order

A fresh session with no context should do exactly this:

```bash
# 1. state of the world -- 60 seconds
cat docs/MISSION.md                      # this file: goal, ledger, queue
sed -n '/^### 3(0)/,/^### 3(1)/p' docs/FRAMEWORK.md   # the live status board

# 2. WHAT WAS RUNNING AS OF 2026-08-30

⚠️ **RUN STATE IS A SNAPSHOT, NOT A FACT.** This section was written 2026-08-30 and nothing re-dates it when the rig moves. Verify with `python -m scripts.rig_status` and `python -m scripts.quarantine --list` before believing any of it. A present-tense heading with no date is how a reader ends up relaunching a campaign that was quarantined and had its pending runs dropped (2026-09-10).
for h in dsisco01 dsisco02; do ssh $h 'nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | while IFS=, read -r u p; do echo "$(ps -o user= -p ${p// /} 2>/dev/null)"; done | sort | uniq -c'; done
ssh dsisco01 'cd ~/optloss-domb && ~/anaconda3/envs/optloss/bin/python -m scripts.rig_status'

# 3. progress of every campaign
ssh dsisco02 '~/anaconda3/envs/optloss/bin/python - <<PY
import glob,json,os,collections
seen=collections.defaultdict(collections.Counter)
for t in sorted(glob.glob(os.path.expanduser("~/optloss-*"))):
    for c in glob.glob(os.path.join(t,"results","*","*","*","*","*","seed_*","config.json")):
        p=c.split(os.sep); seen[p[p.index("results")+1]][json.load(open(c)).get("status","?")]+=1
for k,v in sorted(seen.items(), key=lambda kv:-sum(kv[1].values())):
    print("%-14s %s"%(k,dict(v)))
PY'

# 4. gates, before ANY launch
python -m pytest tests -q          # must be 634 passed + 1 skipped (635 collected)
#   NOTE 2026-09-11: this read 591 for weeks, a drift of 43, while
#   CLAUDE.md and FRAMEWORK.md stayed current -- because
#   `test_the_documented_test_count_is_the_real_one` reads those TWO
#   files and not this one. Its own comment said 'bump when you add
#   one'. An instruction to a human is not a gate.
python -m scripts.audit_config
python -m scripts.smoke_arms
```

**Then pick up item 1 of the queue that is not already running.**

🛑 **AND ON THE FIRST TRAINED RUN OF ANYTHING YOU LAUNCH, NOT AT THE END:**

```bash
python -m scripts.dose_landed <root>     # `amp` column beside the percentage
```

A trained arm landing 25-31% below its attempted steps on `amp=float16` is the
HOST, not the loss shape, and the fix is `--constraint-fp32`. Measured over
every completed run in every worktree: with the flag, **15284 / 15284 steps
across 532 runs and 6 campaigns**; without it, 86.9% over 189 runs, and that
group is the quarantine list. `gen_campaign` now REFUSES a campaign with
trained arms and `constraint_fp32: false`, so this cannot recur from the
generator -- but a campaign staged before 2026-09-01 can still carry it.
`taskwin1` did, landed 20/29, and was killed at 3/48 and relaunched as
`taskwin2`. Deciding on run one cost thirty minutes; deciding at 48/48 would
have cost seven hours.

### Reading a landed campaign, in this order and no other

```bash
python -m scripts.dose_landed <root>                        # FIRST. always.
python -m scripts.full_panel --campaign <root> --control clip
python -m scripts.full_panel --campaign <root> --control tralo_null
python -m scripts.family_split --campaign <root> --families tralo fioretto hounie alm
python -m scripts.log_health <root>                         # read 3(0c) first
python -m scripts.order_probe --campaign <root> --arm tralo
```
Then: **per-cell breakdown, never a pooled digit**; the **reseed row** beside
every win; **macroF1 beside ccF1**; and an **exact sign test** with the cell
count stated.

---

## 6. WORKING IN PARALLEL

The GPU is the scarce resource; context is the other one. While a campaign runs,
delegate independent read-only analysis to subagents (the user has standing
approval for this) and keep only the conclusions:

- one agent per landed campaign that has never been scored
- one agent per offline probe that prices a direction (`ceiling_screen`,
  `paired_noise`, `dataset_screen`, `factorial_control`, `straddle_probe`)
- one agent to re-audit a defect class already found once (inert flags,
  incommensurable logs, unequal dose, pooled digits hiding per-cell reversals)

Never delegate a launch, a `git push`, or anything that writes to `src/`,
`configs/` or `main.py` while a campaign is running.
