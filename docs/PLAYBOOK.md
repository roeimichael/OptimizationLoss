# PLAYBOOK -- what to do when a campaign lands, whichever way it lands

`docs/MISSION.md` says what we are trying to prove and where we are.
**This file says what to DO next, branching on the result**, so that a landing
campaign is never met with an improvised decision. Every branch below was
written before the data, which is the only time such a rule is worth anything.

Companion documents: `docs/FRAMEWORK.md` is the law and the rejected ledger;
`docs/MISSION.md` is the state and the queue.

Last updated: **2026-08-30**.

---

## 0. THE ONE-PAGE VERSION

```
campaign lands
      |
      +-- 1. INTEGRITY GATES  (before ANY metric -- section 2)
      |      dose_landed -> quarantine --check -> flag_live -> code_version
      |      any gate red? -> STOP. The result is not a result. Section 6.
      |
      +-- 2. READ THE LOGS  (section 3)
      |      log_health per run, then verify every count from predictions
      |
      +-- 3. SCORE  (section 4)
      |      full_panel x2 controls -> per cell -> macroF1 BESIDE ccF1
      |      -> reseed floor beside every win -> correct independent n
      |
      +-- 4. BRANCH  (section 5)
             WIN  -> replicate before believing; hardest cell next
             NULL -> was it measured or merely unresolved? they differ
             LOSS -> ledger it, and mine it for the mechanism
```

---

## 1. THE STANDING RULES THAT DECIDE EVERY BRANCH

1. **A result is (effect, floor, n).** Never one of the three. `tralo_reseed`
   minus `tralo_null` is the floor, and a win inside it is not a win.
2. **The independent unit is the WARM-UP MODEL, not the cell.** A lambda=0 twin
   is byte-identical across cap tags, so cap levels within a seed are
   correlated replicates -- and it is worse than that: the CNN warm-ups are
   **shared across campaigns**, so `uniform1` + `loose1` + `dom1` together hold
   only **20 distinct warm-up models**, and `dom1`'s L80/L90 cells are
   byte-identical to `loose1`'s in 80/80 files. FRAMEWORK 2(z): 8 of 9 `dom1`
   sweeps evaporated once this was applied, and the loose-cap win went from
   15/20 (p=0.041) to **11/16 (p=0.21)**. **State which unit you are quoting,
   every time**, and check for cross-campaign duplication before pooling two
   campaigns.
3. **macroF1 and uncF1 go beside ccF1 in every table.** ccF1 alone hides
   uncapped damage: `dom1` reads ccF1 +0.0141 (6/6) and macroF1 -0.0022 (2/6).
4. **Convert to ITEMS.** `items = dF1 * (K + n_pos) / 2`. The whole prize from
   `clip` to a PERFECT allocator is 1.9-9.9 items, so a sub-item delta is a
   re-allocation, not a difference.
5. **Pre-register ONE primary before scoring.** Twenty contrasts cannot survive
   correction at any n this project can afford.
6. **`flips`, raw count over K, and proximity to feasibility are not metrics.**
   Post-hoc filling is free.

---

## 2. THE INTEGRITY GATES -- run these BEFORE any metric

Run in this order. Each refuses a different way to waste a week, and each has
fired at least once on a campaign that looked healthy from every other angle.

```bash
python -m scripts.quarantine --check <root>   # is this campaign already dead?
python -m scripts.dose_landed <root>          # FIRST, and on a RUNNING campaign
python -m scripts.flag_live <armA> <armB>     # md5: is the new arm LIVE?
python -m scripts.check_parity <root>         # equal compute, one commit, >=2 caps
python -m scripts.cut_gap <root>              # where is the cut, and is it reachable?
```

| gate | what it catches | the receipt |
|---|---|---|
| `quarantine --check` | scoring a campaign already known dead | `iwc2` at 74.6% dose with `check_parity` GREEN |
| `dose_landed` | an arm that ran at a fraction of its dose and still wrote `completed` | `tralo_uniform` 1/29 beside `tralo` 29/29; `iwc3` lost 328 of 1044 |
| `flag_live` | a flag that changes nothing -- four have shipped | `cb_lp` byte-identical to `clip` in 24/24 |
| `check_parity` | unequal compute, mixed commits, one cap level | -- |
| `cut_gap` | a cut in a dead zone, where no count function can act | 12 of 26 points at K/n <= 0.3 |

🛑 **`check_parity` being green proves less than it looks.** It is green on
`iwc2`, which lost a quarter of its dose. Green parity plus red dose is a
common combination; run both.

### The AMP trap, now a GATE because saying it was not enough

`--constraint-fp32` is mandatory. Without it, fp16 + GradScaler skips
overflowing steps and the dose quietly drops: ViTB16 landed **173/232 (74.6%)**
without it and **232/232** with it, on the same host. Check `amp` and
`constraint_fp32` in `config.json` for every campaign you compare.

🛑 **THIS PARAGRAPH EXISTED AND `taskwin1` WAS STILL STAGED WITHOUT THE FLAG**
(2026-09-01). Its first trained run landed **20 / 29 = 69.0%** and it was
killed at 3/48 and regenerated as `taskwin2`, which lands **29 / 29** on the
same host and the same arm. The generator's default is `false`, and no amount
of prose survives that. So `configs/gen_campaign.py` now **REFUSES** a campaign
with trained arms and `constraint_fp32: false`, quoting the measurement:

| `constraint_fp32` | landed / attempted | runs | campaigns |
|---|---|---|---|
| **true** | **15284 / 15284 = 100.0%** | 532 | dom1, dom1b, equaldose1, iwc4, loose1, loosevit1 |
| false | 4684 / 5393 = 86.9% | 189 | iwc1, iwc2, iwc3, taskwin1, uniform1_VOID, xfam1 |

`--allow-fp16-constraint` overrides it and says in the output what it let
through. Note the `false` column IS the quarantine list.

🔑 **AND DECIDE ON THE FIRST TRAINED RUN, NOT THE LAST.** `dose_landed`'s
own rule is "one arm low = the loss shape, EVERY arm low = the host". Do not
wait for every arm to prove the host: the `amp` column plus the table above
settles it on run one. Restarting at 3/48 cost 30 minutes; the same decision at
48/48 would have cost seven hours.

---

## 3. HOW TO READ THE LOGS -- the procedure, and its three traps

```bash
python -m scripts.log_health <root>       # per-run: collapse, divergence, trajectory
python -m scripts.diagnose_run <run-dir>  # stage-by-stage read of ONE run
```

**What you are looking for, in order:**

1. **Terminal collapse.** Does the last epoch's metric fall off a cliff? The
   pipeline keeps the LAST epoch, not the best. A `clip` seed once ended
   0.9934 -> 0.9116 and that single collapsed control reversed a headline.
   ⇒ **always look at the final epoch of the CONTROL arms, not just the treated
   ones.** A collapsed control manufactures a win.
2. **Divergence.** Loss going to NaN/inf, or the constraint term dominating CE.
3. **The count trajectory against K.** Does the hard count move toward the
   budget, and does it move MORE than `tralo_reseed`'s does? The constraint
   moves the capped count RMS 75-95 items; a reseed moves it 83-95. So a count
   movement is only a result as a RATIO to the floor.
4. **Satisfaction.** On `dom1`, **0 of 696 epochs** ever satisfied the
   constraint; the post-hoc allocator does all of it. If a campaign reports
   satisfaction, check it against the predictions before believing it.

### 🛑 THE THREE TRAPS IN THE LOGS

**(a) The cross-arm count table is NOT comparable.** The arms write different
log SCHEMAS -- `tralo*` 76 columns, `hounie` 16, `alm` 15, `fioretto` 14. And
for every TRAINED arm the last logged `Hard_Class2` disagrees with the model's
actual predictions (`alm` logs 340 and emits 467; 0 of 24 agree), while the
nulls agree 24/24. Reading that table gave the EXACT OPPOSITE of the truth once.
⇒ **measure any count from `final_predictions_raw.csv`, never from the log.**

**(b) `_raw` and plain are different questions.** `final_predictions_raw.csv` is
the MODEL's argmax -- compare MODELS with it, but it is **not budget-equalized**,
so a raw ccF1 difference mixes quality with count. `final_predictions.csv` is
after the allocator and emits exactly K -- compare ALLOCATORS with it.
`full_panel` re-derives its own equal-budget allocation from the probabilities,
so it is **allocator-blind**: two arms sharing a warm-up score `+0.0000` however
differently they allocate.

**(c) A rising lambda is not pressure.** The clip delivers exactly `lr * clip`
against a raw norm of 2,560-12,400, so magnitude is void. Only step DIRECTION
and step COUNT are live levers.

### The minimum log evidence for any claim

- the arm's dose, landed/attempted, beside its siblings'
- the final-epoch value for the treated arm **and its control**
- the hard count from predictions, beside K and beside the reseed's
- the schema you read, named, so the next person knows what is comparable

---

## 4. SCORING -- the fixed sequence

```bash
python -m scripts.full_panel --campaign <root> --control clip        # the quality bar
python -m scripts.full_panel --campaign <root> --control tralo_null  # attribution
python -m scripts.family_split --campaign <root> --families tralo fioretto hounie alm
python -m scripts.paired_noise --campaign <root>    # which of the FOUR noises
python -m scripts.headroom <root>                   # is there anything to win?
python -m scripts.cut_gap <root>                    # is the cut reachable?
```

Then, and this is the reporting contract:

- **per cell, never a pooled digit** -- a pooled mean hides sign reversals
- **the reseed floor row beside every win**
- **macroF1 and uncF1 beside ccF1**, in items as well as F1
- **an exact two-sided sign test**, with the cell count AND the independent-unit
  count both stated
- **the RESOLUTION block read before the verdict** -- a tie means "no effect" OR
  "not enough seeds", and those are opposite conclusions from the same table

---

## 5. THE BRANCHES -- decided in advance

### 5A. IF IT WINS

A win is the most dangerous outcome, because it is the one nobody audits.

1. **Price it against the floor first.** `tralo_reseed - tralo_null` on the same
   metric and cells. If the win is inside the floor, it is not a win -- record it
   as "inside the RNG floor" and move on. On `dom1` a pure reseed swept 6/6 at
   +0.0043 ccF1, which is most of what `tralo` scored.
2. **Restate at the correct independent n.** If it survives only at the cell
   count, say so in the same sentence as the result.
3. **Check macroF1 did not pay for it.** A ccF1 win with a macroF1 loss is a
   trade, not a gain. Report both or neither.
4. **Replicate on the hardest cell, not the easiest.** The next campaign goes to
   the regime or backbone where the mechanism predicts it should FAIL. A win
   that survives its own predicted failure is worth ten that do not test it.
5. **Then, and only then, write it into `MISSION.md` §1 and FRAMEWORK 3(0).**

### 5B. IF IT IS NULL

**Separate the two nulls, because they are opposite conclusions.**

| reading | how to tell | what to do |
|---|---|---|
| **measured null** -- the effect is ~0 | the RESOLUTION block says the design could have seen an effect this size; the floor is well below the observed delta | ledger it in FRAMEWORK 2 as rejected, never retry |
| **unresolved** -- we cannot see | `seeds_needed` >> seeds present; `NOT CALLABLE`; min attainable p above 0.05 | do NOT ledger it. Price what resolution would cost, then decide |

Quote `seeds_needed` and the attainable p every time. "No difference" without
them is an absence of measurement wearing the costume of a null.

If unresolved, the options in cost order: more cells (cap tags or backbones)
before more seeds -- seeds buy precision, cells buy significance, and at 4 seeds
the sign test over cells is the only thing that can reach a verdict.

### 5C. IF IT LOSES

A loss is the cheapest information available and must not be wasted.

1. **Write it into FRAMEWORK 2 with the evidence and the dose**, so it is never
   retried. A rejected direction that gets re-proposed is a whole wasted cycle.
2. **Update `MISSION.md` §2 (the knob ledger)** with the verdict symbol.
3. **Ask what it rules OUT, not just what failed.** `tralo_uniform` losing at
   loose caps is not a null -- it is confirmation of the `cut_gap` mechanism,
   because that mechanism predicted it.
4. **Check the loss is not an artifact** before believing it: dose, collapsed
   final epoch, an inert flag, a mismatched control. A loss is as capable of
   being a bug as a win is.

### 5D. IF IT IS AMBIGUOUS OR THE GATES ARE RED

Do not score it. Fix the gate, or quarantine the campaign
(`scripts/quarantine.py`, add a registry entry with a reason and a `keep_for`)
and say what it is still a receipt for. **A campaign is either scorable or
quarantined -- there is no third state**, because the third state is where
numbers get quoted from campaigns nobody re-checked.

---

## 6. SPECIFIC CONTINGENCIES FOR WHAT IS QUEUED NOW

### `taskwin2` -- 48 runs, RUNNING NOW, and `vittask1` -- 48 runs, STAGED

The first two campaigns in this project whose caps were chosen by MEASURING
that the cap poses a question, and the first carrying `tralo_cut`. Together
they are 4 cells over 2 backbones at ONE `code_version` (`6658ef8cbc59`), so
score them together, never separately.

`taskwin2` MobileNetV3 x {`L70-90_G95`, `L80-100_G95`};
`vittask1` ViTB16 x {`L60-90_G95`, `L70-90_G95`}. Both carry
`clip focal_clip tralo tralo_cut tralo_null tralo_reseed` x 4 seeds,
`normalize`, `--constraint-fp32`.

*Primary:* `tralo_cut - tralo` in ITEMS on `d capF1`, per cell, against
`tralo_reseed - tralo_null` measured IN THE SAME CAMPAIGN. 4 cells is 2
independent (campaign, backbone) units, so the honest ceiling is 2/2, sign
p = 0.25. These campaigns can report DIRECTION and per-cell consistency and
nothing else; the generator says so in its own POWER block.

| outcome | what it means | next move |
|---|---|---|
| **`tralo_cut` > `tralo`, above the campaign's own floor, 4/4 cells** | aiming the count function at the cut buys something where the cap actually poses a question | the first positive on a MEASURED task cell. Add the other two backbones at their OWN per-class task caps before any claim |
| **`tralo_cut` ~ `tralo`** | the count function is not the lever, and 2(z12)'s cosine predicted it (`margin` is 0.989 from `tralo`) | ledger it as run-and-null. Do NOT then run `margin2`: same family, and its caps are dead |
| **`tralo_cut` < `tralo`** | moving gradient mass to the cut COSTS, a real result about where the mass should sit | FRAMEWORK 2 immediately; it retires the whole cut-window family |
| **both below the reseed floor** | the cap poses a question and no method answers it | the strongest negative available here, and worth more than another arm |

⚠️ **READ THE PER-SEED BINDING BEFORE THE VERDICT.** 2(z24): `L80-100_G95`
puts MobileNetV3 class 2 at K/n 0.800, where the cap binds in 3 of 4 seeds on
the reference model, while `L70-90_G95` binds in 4 of 4. Run
`scripts.task_window` on THIS campaign's own `tralo_null` runs rather than on
the yml row, and quote `binds n/N` beside every cell.

### `vitdom1` -- 240 runs, ViTB16, 6 loose caps, the rival duals

*Primary:* `tralo` ranks first of the five trained arms in >= 5 of 6 cap tags,
with its margin over the runner-up exceeding the reseed floor.

| outcome | what it means | next move |
|---|---|---|
| **TraLO first, clear of the floor** | the dominance claim survives on the pre-registered backbone | replicate at TIGHT caps on ViTB16? **No** -- `vitu1` already says it loses there. Go to **dataset #2 (`fmow`)**, because generality is the remaining axis and cells within one dataset cannot buy it |
| **TraLO first, inside the floor** | ranking is real, magnitude is not | do NOT report as dominance. Price the seeds needed; if unaffordable, report as a direction with the floor quoted |
| **TraLO not first** | the `dom1` ordering was MobileNet-specific | this is a major negative and it goes in FRAMEWORK 2 immediately. The paper's headline must then be restated as backbone-conditional |
| **any arm off-dose** | not a result | fix and relaunch; do not partially score |

*Secondary already fixed:* `tralo_uniform` should LOSE here. ⚠️ But note 2(y) was CORRECTED on
2026-08-30: the geometric account is unrefuted but NOT discriminated, and its
sharp `uniform` prediction already failed once (both arms' slopes have the same
sign). So treat a `uniform` result here as a fresh test, not a confirmation.

### `margin2` -- 432 runs, 12 cells, tight + loose

*Primary:* `tralo_margin - tralo` on AP, >= 10 of 12 independent (model, seed)
units.

| outcome | what it means | next move |
|---|---|---|
| **gains LOOSE, not TIGHT** | consistent with 2(y), which is an account and not a measured cause | the regime-conditional claim strengthens, but do NOT write "by geometry" -- 2(y)'s discrimination failed. Report the regime step (12/12, p=0.00049 paired on the warm-up), which IS solid |
| **gains BOTH** | 2(y) is WRONG | the most valuable outcome available. Re-derive the mechanism before running anything else; the `cut_gap` reading must be re-examined first |
| **gains NEITHER** | placement is not the lever | ledger `margin`; check `tralo_coin` -- if a random direction moves the metric as much, the whole placement family is dead |
| **`tralo_coin` matches `tralo_margin`** | direction does not matter, only norm | this kills the placement family regardless of sign. Report it prominently |

### `dom1b` -- landing now

Score `dom1` + `dom1b` together **at the (model, seed) unit**, not 9 cells.
If RegNetY400MF reverses the `dom1` ordering, the ordering is backbone-specific
and every dominance sentence needs that qualifier.

### If a GPU frees and nothing is ready

Order: `vitdom1` -> `margin2` -> extend `loosevit1`'s cap tags. Never launch to
look busy; the cheap offline probes (`cut_gap`, `ceiling_screen`,
`paired_noise`, `headroom`, `straddle_probe`) all price a direction for free and
each has closed one this project would otherwise have spent a campaign on.

---

## 7. THE FAILURE MODES THIS PROJECT ACTUALLY HAS

Each has happened. Check for each by name.

| failure | how it presents | the check |
|---|---|---|
| inert flag | a new arm scores like an old one | `flag_live` md5 -- four have shipped |
| silent dose loss | `status: completed`, everything green | `dose_landed`, and read `amp` |
| collapsed control | the treated arm "wins" | final epoch of the CONTROL |
| pooled digit | one number, no per-cell | always break out cells |
| correlated units counted as independent | 6/6 sweeps that mean less than they look | the (backbone, seed) unit |
| budget not equalized | a raw ccF1 gain that is really more emitted | compare at matched K |
| ccF1 quoted alone | a trade reported as a gain | macroF1 + uncF1 beside it |
| log counts read across arms | the exact opposite of the truth | count from predictions |
| a dead campaign scored | plausible numbers from invalid data | `quarantine --check` |
| a cap with no prize | chasing an effect smaller than the noise | `headroom`, `cut_gap`, `ceiling_screen` |

---

## 8. HYGIENE -- keep the evidence base honest

```bash
python -m scripts.quarantine --list            # what is live, what is dead, why
python -m scripts.quarantine --apply           # dry run
python -m scripts.quarantine --apply --execute # write markers, drop dead configs
python -m scripts.rig_status                   # before AND after every launch
```

**The quarantine rules, which are deliberate:**

- **never delete a `completed` run** -- completed runs are receipts, `results/`
  is gitignored, and the corpus already cannot be rebuilt. Disk is 31% used
  with 588T free, so space is never the reason.
- **do delete a `pending` run that must never execute** -- a dead dataset, or a
  quarantined campaign the dispatcher would still pick up. A marker file does
  not stop `main.py`; an absent config does.
- **do correct a `running` status with no process behind it.** That is not
  clutter, it is false, and it reads as a live campaign.
- `full_panel` refuses a quarantined root and **exits 1**. Override only with
  `--allow-quarantined`, and only while saying so in the report.
