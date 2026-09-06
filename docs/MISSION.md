# MISSION -- make TraLO mathematically the best, and prove it

**This file is the resume point.** A fresh session reads it first, then
`docs/FRAMEWORK.md` section 3(0) (the status board). It is updated at the end of
every working session. If it is stale, that is a defect -- fix it before doing
anything else.

Last updated: **2026-09-06** (the arm-vs-arm scorer was ranking a four-name whitelist, so half of every campaign was invisible; the acceptance table priced cells on a RANGE; and with both fixed, EVERY verdict in EVERY campaign reads REFUSED because the RNG floor rests on 4 observations against a bar of 8. `tralo_reseed2` -- the fix, 12 observations for 8 runs -- exists and had never been put in a campaign. See 0-HEAD.)

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

## ⚠️ 0-DOSE. `tralo_sgd` IS UNDER-DOSED, AND THAT IS MEASURED (2026-09-06)

`scripts/step_dose` on the real MobileNetV2 config. Constraint-aligned weight
displacement, `||dw|| * cos(dw, descent direction)`:

| rule | `\|\|dw\|\|` | cos | aligned |
|---|---|---|---|
| `shared` | 0.0444 | 0.187 | 0.00828 |
| `sgd` | 0.000100 | 0.9997 | 0.000100 |

**`sgd` delivers 83x LESS movement along the direction the constraint asked
for.** Its direction is perfect and its dose is tiny.

⚠️ **State the caveat with the number.** That reading used 60 CE steps, so
`cos = 0.187` is higher than the 0.009-0.017 the framework measures after a full
epoch; at that cos the gap narrows to ~6x. Either way `sgd` is UNDER-dosed, not
over-dosed, so **a null from `tralo_sgd` is about DOSE and must be reported as
the dose gap, never as "delivering the direction does not help"**. That was
pre-registered in `protocol.yml` before the campaign launched and is now
quantified.

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

### 5. WHAT IS NOW STAGED

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
> at K/n 0.950). So state BOTH: **4/4 units, p=0.0625** licensed, and
> **3/3 units, p=0.125** over units with a verified `task` cell. Every sign is
> identical either way. `scripts/paper_rows.py` prints the restriction --
> take it from there rather than re-deriving it.
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

## 🟢 0-RUNNING. WHAT IS IN FLIGHT RIGHT NOW (2026-09-02, both GPUs)

We hold **2 of 4 GPUs on dsisco01, which is the cap.** GPU 0 is `nirgal` and
GPU 2 is `zehavid`; never share. The next free slot is one of OURS finishing,
not another user leaving.

| root | worktree | GPU | commit | runs | what it decides |
|---|---|---|---|---|---|
| `vitdual1` | `optloss-cutwin` | 3 | `6658ef8c` | 88 | the four duals on **ViTB16**, the headline backbone, which has zero fioretto/hounie/alm. Also carries 8 `tralo_null` seeds at two caps -- the ONLY way to replace the 1-seed ViTB16 task window |
| `seed58a` | `optloss-domb` | 1 | `1d921173` | 40 | **seeds 5-8** of `dom1b`/RegNetY400MF at `L80_G95` + `L90_G95` |

### Why `seed58a` is seeds and not a new campaign

A new campaign on an already-measured (backbone, host) buys NOTHING -- the
warm-up is cached under the same `base_model_id`, so it is the same model. All
four backbones on dsisco01 are spent or in flight. What is NOT spent is SEEDS.

`paired_noise` prices the powered corner exactly: at K/n = 0.9 a cell needs
**7-8 seeds at 80% power**, against 546 at K/n = 0.5 and 2607 at 0.2. The
protocol runs 4. So `L90_G95` is the one cap where doubling the seeds crosses
the power threshold instead of chasing it -- and this is the direct answer to
"1 of 158 strict-task rows resolves". It is generated at `dom1b`'s EXACT commit
`1d92117363d2` and lands on the same fp16 + GradScaler regime, so seeds 5-8
merge into dom1b's existing cells rather than forming new ones.

⚠️ **The catch, and state it every time the result is quoted:** at
K/n = 0.9 the cap barely binds. Where the constraint BINDS nothing is
measurable, and where something is measurable the constraint hardly
constrains. `L80_G95` is strict on BOTH classes for RegNetY400MF and
`L90_G95` is strict on class 7 only (class 2 is PARTIAL there), so the two
caps are not interchangeable and must not be pooled.

### When a GPU frees

Refill it. Queued in order:
1. **Re-measure the ViTB16 task window** the moment `vitdual1`'s `tralo_null`
   arms land -- `configs/task_windows.yml` currently records ViTB16 from ONE
   seed and both strict bands are empty for that reason alone.
2. **Seeds 5-8 for MobileNetV2** (`equaldose1`, commit `10d37518`) at the same
   two caps -- the second unit to cross 8 seeds.
3. **Units 7-8 are free on dsisco02**: RegNetY400MF and ViTB16 there have never
   been run. Those are NEW units, worth more than seeds, but dsisco02 is fully
   occupied by other users.

---

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

# 2. what is running RIGHT NOW
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
python -m pytest tests -q          # must be 501 (bump when you add one)
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
