# MISSION -- where this project stands

**A state file, not a narrative.** What is running, what is next, and the course.
A finished campaign's FINDING goes to [`LEDGER.md`](LEDGER.md); its story is
deleted. Completed audit receipts go to `docs/archive/`. Keep this file short.

| File | Answers |
|---|---|
| [`RULESET.md`](../RULESET.md) | How to work. Rules, gates, the re-entry checklist. |
| **this file** | Where we are now, what is running, what is next. |
| [`LEDGER.md`](LEDGER.md) | What is proved, what is measured, what is closed. |
| [`FRAMEWORK.md`](FRAMEWORK.md) | The protocol. Outranks all three. |

Evidence was reset on 2026-09-14 at the user's request: historical results and
folder names establish no winners and no universal rejections. **`keep working`**
reloads all four from disk, checks both hosts, does the next highest-value thing,
and writes what it learns back here and into the ledger.

---

## THE COURSE -- read this first, every session

*This section is the contract. If a result changes the course, edit THIS section.
Do not leave it stale and correct it further down.*

### The one question

**Can a training-time constraint beat a post-hoc clipper on a capped-class task**,
at matched data, compute, allocator and precision?

**The bar, set by the user:** leading group on **cc-F1** AND not dominated on the
rest of the metric profile. A win on accuracy or macro-F1 while losing cc-F1 is a
**TRADE**, not a win, and is reported as one. A valid negative result is a real
deliverable. Do not start a new research task to manufacture an advantage.

### Where the answer stands today

**TraLO does not clear the bar on any backbone.** The backbone was thought to be
a real moderator; under the corrected cap it is not -- the one exception reversed.

- On MobileNetV2, MobileNetV3 and RegNetY400MF the constraint contributes nothing:
  every apparent clipper win is delivered by `tralo_null` at the same or a larger
  margin, and in `live11` the constraint is actively negative while its null posts
  the corpus's largest clipper win.
- ⛔ **The ViTB16 exception is DEAD, reversed 2026-09-17 and re-confirmed from raw
  predictions 2026-09-20.** It was the one cell in 1,364 fmow2 runs where
  `tralo` - `tralo_null` survived its own control, and it was measured under the
  PREVALENCE cap, where the ceiling is an affine image of the per-group label
  histogram (LEDGER 2.9). Re-run under the corrected policy cap (`polcv2`, 112
  runs, 8 seeds/cell, balanced), ViTB16 **flips to the worst arm of seven**:
  `tralo` - `tralo_null` is **-0.0139** (t -2.0) on cc_f1 at L75 and **-0.0165**
  (t -2.4) on F1 Macro at L25; ceiling precision 0.4610, last of seven; 805 true
  positives evicted against 784-798 for every rival. **TraLO now has no cell
  anywhere in which the attributable contrast is positive.**
- **`focal_clip` beats `tralo` on ViTB16 with CIs excluding zero**, and does so
  again under the corrected cap (-0.0181, t -3.3, the largest single contrast in
  the corpus). The best arm measured anywhere is **`aug_clip` at budget 6** -- a
  post-hoc clipper with flip-and-crop and early stopping.

- 🔑 **The strongest objection is now closed too (2026-09-26, LEDGER settled #9).** On the
  rebuild branch, end-to-end knee (trainable ResNet18), two preregistered studies of 24 seeds
  each: with the controller, trigger, binding and dose all fixed, TraLO's direction stepped
  to the exact radius that meets the hard cap is **no better than a random move of the same
  size**. target-sham is +0.14 [-1.21, +1.48] at cap 76 and -0.75 [-2.45, +0.96] at cap 50.
  The step evicts 83-87% the same items as the post-hoc cut. The published arm's extra damage
  is a 10x overshoot of the step size.

**The mechanism is proved, not guessed** (LEDGER PART 2). The loss is a function
of the multiset of test probabilities while the allocator is a function of their
ranks (M1), so nothing in the objective can prefer a correct ordering over the
worst ordering with the same multiset. The literature says post-hoc thresholding
IS the optimum for selection-rate constraints, with one crack: it is optimal only
when the score is Bayes-optimal (Woodworth et al., COLT 2017). **A training-time
win is permitted only by improving the SCORE.**

⛔ **The "root cause" in LEDGER 2.3 was RETRACTED 2026-09-20.** It said the
constraint runs on TRAIN data where violation is identically zero. It does not:
`runner.py:109` binds `X_test = data.X_test`, and `tralo` plus all four duals run
their whole backward pass over the deployment pool. In the logs the term is
violated in **288 of 288 constraint epochs across 60 runs (0.0% satisfied)**,
growing ~1500x to a gradient norm of 6399 while the capped class sits 4x over
budget. **The term is maximally alive and TraLO still loses** -- which closes
"the term was never alive" as an escape and leaves M1 as the operative
mechanism. The held-out-fold direction this opened is cancelled.

### What is settled

Eight settled findings, five proved mechanism results (M1-M5) and the closed
directions live in [`LEDGER.md`](LEDGER.md). Do not re-open any without new
evidence. The five most often re-litigated:

1. The four rival duals share one per-item gradient and are ONE family. Measuring
   them tied is a confirmed prediction, not a disappointment.
2. Eviction given the probabilities is already ~87% optimal; the prize is not in
   the loss shape.
3. The exact (LP) allocator makes real results worse in 14/14 cells. Cancelled.
4. The cap IS binding and there IS headroom -- but ~39-46% of wasted slots are
   held by items the model is certain about and wrong, so the reachable prize is
   **~12-15% of the budget, not 20-25%**.
5. **Liveness is not the barrier.** At ~100% live (`small60`) the constraint is
   negative in 6 of 6 primary cells against its own null. A short budget and a
   live window explain why the constraint is INERT; they do not explain, and do
   not repair, the fact that it is HARMFUL when active. `tralo_stab` closed the
   one escape route M5 left open.

### Standing decision rules

Full set in [`RULESET.md`](../RULESET.md); it is the authority. The four that bite
most often: **`gate:saturation` is a screen, not a diagnosis** (it reads TRAIN
accuracy); **average over SEED only** and de-duplicate by prediction hash;
**four seeds is a pilot** (~15% power against a seed sd of ~0.011); **report a
trade as a trade, and retract in place.**

---

## RUN STATE -- checked 2026-09-26 12:45 IDT (server clock)

All live work is on **dsisco01**, under `~/tralo-rebuild/runs/`. Releases are immutable clones by SHA in `~/tralo-rebuild/releases/`, and the code is on branch `claude/bandcons-20260926`. dsisco02 is fully held by another user. All four dsisco01 GPUs are ours only.

| run | release | seeds | state | scored by |
|---|---|---|---|---|
| `claude-bandcons-a1cap50` (BANDCONS, cap 50, deviation D-A1) | c5634488 | 2401-2424 | 5/24 | `analysis/score_bandcons.py`, then the snapshot-ensemble prereg |
| `claude-step-probe-20260926` (targeted step at depths f) | e7e02085 | 2601-2624 | 12/24 | `analysis/score_step_probe.py` (fixed 615d55eb) under amendment 1 |
| `claude-cutpair-cap76` (CUTPAIR, cap 76) | 6fb21e48 | 2701-2724 | launched 12:42; the pilot 2700 passed its gate | `analysis/score_cutpair.py` under amendment 1, then the snapshot-ensemble prereg |

Queued: the BANDCONS cap-76 block (2501-2524). It is decided after the cap-50 result.

## RUN STATE (superseded) -- checked 2026-09-16 22:57 IDT (server clock)

🛑 **ALL SEVEN PREVIOUS CAMPAIGNS WERE STOPPED 2026-09-16 22:30** by explicit PID
(runners first, then INT on each trainer), on the user's instruction. Every one
of them ran the PREVALENCE cap (`L80_G95` / `L90_G95`), which LEDGER 2.9 shows
makes the local ceiling an affine image of the per-group label histogram. **They
were measuring the wrong constraint**, so completing them had no value.

**Results moved, not deleted**, to `~/quarantine_2026-09-16/<tree>__results`
(29G, five trees). `/home` is at 93% and a copy was impossible -- 16G free
against a 16G tree -- so the move is a same-filesystem rename. ⚠️ **Reclaiming
the space is BLOCKED**: the deletion was refused by the permission layer and
needs the user. ~110G is safely removable (`optloss-history-20260914` 46G,
`optloss-archive-stale-2026-09-02` 18G, `isic_cache` 13G, `_cct_chunks` 2.2G,
`hp_liveness_out` 607M, plus the 29G quarantine). **No dataset is in that list.**

🛑 **The only real dataset bytes live in `~/optloss-audit/data/`** (fmow2 3.1G of
.npy). Every other tree reaches them through a symlink chain
`optloss-rank/data/fmow2 -> optloss-probe/data/fmow2 -> optloss-audit/...`.
Deleting `optloss-audit` destroys the data. Nothing else symlinks into the
removable list (verified by `find -type l` + `readlink -f`).

### DONE: Option C on MobileNetV3 -- REFUTED

`polc2_a` + `polc2_b`, 252/252, 12 seeds, completed 02:00. Gates green
(saturation `ok` at 50% live; both campaigns validate; 252 distinct prediction
hashes, no re-runs counted as seeds). **Verdict taken once against the mapping
fixed at 130/252: outcome 3, REFUTED, on both primary endpoints.** Full numbers
and the mechanism in LEDGER PART 4.

### DONE: the augmentation cell -- scored 2026-09-25, OUTCOME 2 (AMBIGUOUS), as pre-registered

168/168 runs, 12 seeds/cell, balanced, 168 distinct prediction hashes, dsisco02/bf16.
`scripts/score_augfin.py`.

| cc_f1 | L25_G25 | L75_G75 |
|---|---|---|
| **decisive** `aug_tralo` - `aug_clip` | +0.0011 (t +0.3) | +0.0033 (t +0.6) |
| **attributable** `aug_tralo` - `aug_tralo_null` | +0.0011 (t +0.4) | +0.0028 (t +0.8) |

F1 (Macro) against `aug_clip`: -0.0016 / -0.0031. The large `aug_tralo` - `focal_clip`
margins (t +2.2 / +2.5) are the AUGMENTATION and, per the pre-registration, are not a
TraLO result. **First TraLO arm non-negative against both its matched rival and its own
null at both caps under the corrected cap -- and every interval crosses zero.** At
~0.2 seed-SD, resolving it would take ~250 seeds; more seeds are not the path.

Gates: `gate:saturation` FAILS at 41% live (4.1 of 10 epochs) against the gate's 50%;
the pre-registered kill threshold was 33%, so the cell stands, flagged.
`feasibility_check` FAILS identically in every arm and seed (e.g. g5/c2 33 > 3): the
checker recomputes PREVALENCE caps and ignores `group_budget_shares`; the deployed
counts equal the configured POLICY caps exactly. A checker bug, not a run defect --
fixed on `codex/tralo-decision-20260921` (`e8a3bacc`), not yet merged here.

Mechanism note from the training logs: `aug_tralo`'s hard count for c2 ROSE 653 -> 718
against K=136 over the constraint phase (satisfied 0/120 epochs). One constraint step
per epoch against ~276 task steps per epoch does not move the count.

### (superseded) LIVE: the augmentation cell -- checked 2026-09-21 13:23 IDT (server clock)

| campaign | host | GPU | runner | caps | seeds | runs | budget |
|---|---|---|---|---|---|---|---|
| `augfin_a` | dsisco02 | 0 | `afa2` | L25_G25 L75_G75 | 1-6 | 84 | 11 (1+10) |
| `augfin_b` | dsisco02 | 1 | `afb`  | L25_G25 L75_G75 | 7-12 | 84 | 11 (1+10) |

Seven arms each, single stamp `99504a126ff1`, bf16, no scaler. Launched 13:16
and 13:17, both confirmed training at 88% GPU. GPUs 2-3 free; dsisco01 idle.
**Pre-registered reading is above -- the decisive contrast is `aug_tralo` -
`aug_clip`, NOT `aug_tralo` - `clip`.**

⚠️ **NOT YET DONE: the `firstrun` gate.** Nothing may be read from these runs
until `gate:saturation` and `gate:trainlog` pass. Augmentation should hold the
boundary live for ~5 of 10 constraint epochs; below 33% the cell is killed early.
Run: `python -m scripts.run_campaign --root augfin_a --step firstrun`

⚠️ The runner resolves `<worktree>/<root>` RELATIVE TO ITS CWD. Launching from
`~/optloss-rank` makes it look for `optloss-rank/optloss-rank/augfin_a` and it
logs `SKIP -- does not exist`, then exits cleanly with the GPU idle. **Launch
from `$HOME`.** The first `afa` runner died this way and was relaunched as
`afa2`.

### DONE: Option C on ViTB16 -- AMBIGUOUS by the mapping, NEGATIVE in direction

`polcv2_a` + `polcv2_b`, **112 of 168 runs** (56 lost to `OSError 28`, disk
full). The loss is UNIFORM -- every one of the 14 (cap, arm) cells kept exactly
8 of 12 seeds -- so the survivors are balanced, not a truncated prefix, but
**n=8, not the 12 the pre-registration assumed**. Quote that wherever the
numbers are quoted.

Verdict against the registered mapping: **outcome 2, AMBIGUOUS**, because no
contrast clears the pre-registered threshold. Every point estimate that moves,
moves against TraLO: `cc_f1` L25 +0.0007 / L75 **-0.0139** (t -2.0), F1 Macro
L25 **-0.0165** (t -2.4) / L75 **-0.0187**. Ceiling precision at L75 is 0.4610,
the **worst of all seven arms**, and it evicts 805 true positives against
784-798 for every rival. **This REVERSES the old corpus's ViT-only positive**,
which was measured under the prevalence cap. Full numbers in LEDGER PART 4.

**Both Option C campaigns are therefore closed.** MobileNetV3 refuted, ViTB16
ambiguous-and-negative. No GPU work is queued behind them.

⚠️ **`polcv_a`/`polcv_b` (budget 7) were KILLED at 14 runs** for failing the
firstrun saturation gate at 33% live, and quarantined. The replacement runs at
budget 5 because the live window is a property of (backbone, dataset).
`gen_campaign --total-epochs` was added for exactly this and moves every arm in
a campaign together, so equal dose still holds within each campaign.

### Disk -- DIAGNOSED 2026-09-17, and the first diagnosis was WRONG

⚠️ **The 200G / 100% figure is a per-user QUOTA on the home export, not a full
disk.** Bare `df` shows the array is `psa:/ifs/a/home` **846T at 32%**, 578T free.
`df <path>` returns the quota view; the two disagree and only the quota binds.

🔑 **CAUSE: `optloss-rank/model_cache/fresh-v1`, 29G, 357 entries, last
written 04:48 -- the minute runs began failing.** It is the warm-up checkpoint
cache, keyed per (backbone, cap, seed, arm-family); ViTB16 checkpoints are
~330MB each and last night's campaigns minted hundreds. **This, not the old
archive trees, consumed the 16G and caused `OSError 28` on 56 ViT runs.**
⛔ An earlier entry here blamed `optloss-history-20260914` / `isic_cache` and
sent the user after 110G of archives. That was wrong and is retracted.

**The cheap, safe fix:** `model_cache/fresh-v1` is REGENERABLE. Deleting it
loses nothing scientific -- warm-ups recompute, at a speed cost only
(`warmup=0.17s cached=True` is the benefit forgone). 29G, and it alone clears
the quota. Still needs the user: deletion is refused at the permission layer.

**The structural fix -- decided by the user 2026-09-19: stay inside our own
folder, do not involve the admin.**

First, the contamination worry is already answered: ✅ **nothing of ours has
ever been outside `/home/dsi/michaer8`**, which is mode `drwx------`, i.e.
private to us. `/private/shared` and `/home/fast` hold none of our bytes.
Neither is writable by us anyway (`shared_mngr` group; root-owned), so using
either would require the admin the user wants to avoid. **The roomy-export
option is closed by choice, not by failure.**

| path | free | writable by us | ours on it |
|---|---|---|---|
| `/private/shared` | 5.2T | NO -- `shared_mngr` group, admin-managed | none |
| `/home/fast` | 200G (0 used) | NO -- root-owned | none |
| `/home/dsi/michaer8` | 0 | yes, but quota-capped | **everything** |

So the whole problem is **37 `optloss-*` trees inside our own folder**, not
where the folder sits. Measured 2026-09-19, the keep-set is **44G of 260G**:

| keep | size | why |
|---|---|---|
| `anaconda3` | 21G | the `optloss` env; rebuilding it costs hours |
| `optloss-audit/data` | 17G | 🛑 **the ONLY real dataset bytes** |
| `optloss-rank` minus `model_cache` | 6.2G | the working tree + scored results |

🔑 **`optloss-probe` is no longer load-bearing.** The data symlink was repointed
`optloss-rank/data/fmow2 -> optloss-audit/data/fmow2`, collapsing the old
two-hop chain, and verified by LOADING: `(3442, 224, 224, 3) uint8`. The other
35 trees can now go without breaking a path.

⚠️ Deletion is still **refused at the permission layer** and needs the user to
run it. Nothing on the removable list is a dataset.

⚠️ **`model_cache` grows without bound across campaigns and nothing prunes it.**
Any future campaign plan must budget for it or clear it first.

---

## PRE-REGISTERED READINGS -- fixed before the numbers, do not reinterpret

### `augfin_a` + `augfin_b` -- the augmentation cell, written 2026-09-21 BEFORE launch

**Why this exists.** Every other direction is closed (PART 4). The augmentation x
constraint interaction is the ONLY one in which the constraint has never been
harmful: positive in 5 of 6 budget-8 cells across three backbones (+0.004 to
+0.010), with ViTB16 the lone negative and also the cell that fails the
saturation gate. It has never run at adequate power, and never under the
corrected policy cap. This is the last live lead.

**Design.** MobileNetV3 x fmow2, caps `L25_G25` and `L75_G75`, 7 arms, seeds
1-12 split across two roots, dsisco02/bf16. `total_epochs 11` because
augmentation roughly doubles the live window (3 -> ~5) and the budget rule is
`2 x live + 1`; every arm in the campaign carries the same 11, so the dose is
equal. `group_budget_shares: proportional_to_group_size` -- the corrected cap.

**The two contrasts, fixed now.**
- **Decisive:** `aug_tralo` - `aug_clip`. Same augmentation, same budget, same
  allocator; differs only in whether the constraint trained. This is the bar.
- **Attributable:** `aug_tralo` - `aug_tralo_null`. Isolates the constraint
  inside the augmented column.

🛑 **`aug_tralo` - `clip` and `aug_tralo` - `focal_clip` are NOT the test.**
`stab8` already showed those large wins (cc-F1 +0.018 to +0.020, t 5.6-7.0) are
the AUGMENTATION, not the constraint. Quoting them as a TraLO win is the exact
error this pre-registration exists to prevent.

**Outcomes, fixed before the numbers.**
1. **CONFIRMED** -- `aug_tralo` - `aug_clip` > 0 on cc_f1 with a 95% CI excluding
   zero, AND `aug_tralo` - `aug_tralo_null` > 0. This would be the project's
   first positive result that survives both its rival and its own control.
2. **AMBIGUOUS** -- the attributable contrast is positive but the decisive one is
   not, or either CI includes zero. Reported as underpowered, not as a win.
3. **REFUTED** -- `aug_tralo` - `aug_clip` <= 0 on cc_f1. The interaction is the
   augmentation, and the last non-harmful direction closes with it.

**Gate before reading anything:** `gate:saturation` at firstrun. Augmentation
should hold the boundary live for ~5 of 10 constraint epochs (~50%). If it comes
in below 33% the cell is killed early, as `polcv_a`/`polcv_b` were.


### Option C on ViTB16 (`polcv_a` + `polcv_b`, written 2026-09-17 02:10, BEFORE launch)

**Why this exists.** Option C is REFUTED on MobileNetV3 (LEDGER PART 4). ViTB16
was the only backbone in the old corpus where `tralo - tralo_null` was positive
on both caps and both endpoints, so it is the one place the refutation could
fail to generalise. This is a REPLICATION of a closed result on a new backbone,
not a new question.

**Design.** ViTB16 x fmow2 x {`L25_G25`, `L75_G75`} x 7 arms x 12 seeds
(`polcv_a` seeds 1-6, `polcv_b` 7-12; same host, precision and stamp, so they
pool). Only the two EXTREME caps, so 12 seeds buys real power on the cap
contrast rather than 6 seeds spread over three levels. Same budget (7 epochs),
same share rule (`proportional_to_group_size`).

**Endpoints and contrasts.** Identical to the MobileNetV3 pre-registration
above: primary `cc_f1` then `F1 (Macro)`; the only attributable contrast is
`tralo - tralo_null`; every `tralo - rival` printed beside `tralo_null - rival`;
average over SEED only; de-duplicate by prediction hash.

**Outcomes, fixed now:**

1. **REFUTATION GENERALISES** -- pooled mean inside the seed sd on both
   endpoints, as on MobileNetV3. Option C closes across backbones.
2. **BACKBONE-SPECIFIC** -- `tralo - tralo_null` positive at BOTH caps on cc_f1
   with a pooled mean exceeding the seed sd. Then the MobileNetV3 refutation is
   real but does not generalise, and ViTB16 becomes the claim's only support --
   which must then be stated as a single-backbone result.
3. **AMBIGUOUS** -- anything else, including a sign split across caps. Reported
   as ambiguous; no direction is claimed from it.

🛑 The mechanism probe (precision of the filled ceilings, evictions per arm) is
run REGARDLESS of outcome, because on MobileNetV3 it explained the result better
than the endpoint did.


### Option C (`polc2_a` + `polc2_b`, written 2026-09-17 00:30 at 130/252 runs, BEFORE any score was read)

**Unit.** (MobileNetV3, dsisco02). `polc2_a` and `polc2_b` differ ONLY by seed --
same backbone, host, precision and code stamp `b2051c88` -- so they POOL into one
12-seed set. Nothing from dsisco01 may join it. Average over SEED only.
De-duplicate by prediction hash first: training is bit-deterministic, so a re-run
is not a seed.

**Endpoints.** Primary `cc_f1`, then `F1 (Macro)`. Everything else is exploratory
and may not be quoted as the result.

**The only attributable contrast is `tralo - tralo_null`** at a fixed cap. Every
`tralo - rival` number must be printed beside its matching `tralo_null - rival`
on the same line; a `tralo - clip` win with an equal-or-larger `null - clip`
beside it is a RECIPE effect and is reported as such. This rule exists because
an earlier corpus pass produced 8 false hits out of 12 for want of it.

**Cap ladder and what it means.** L25 / L50 / L75, equal percentages, so
`sum(Phi) == Psi` and both scopes bind. Eviction pressure falls as the cap
loosens, so a real constraint effect should be LARGEST at L25.

**Outcomes, fixed now:**

1. **DOSE-RESPONSE** -- `tralo - tralo_null` positive at all three caps on the
   primary endpoint AND larger at L25 than at L75. The strong result.
2. **FLAT BUT PRESENT** -- positive at 2 of 3 caps with a pooled mean clear of
   the seed sd (~0.011), no ordering.
3. **REFUTED** -- positive at 1 or 0 caps, or a pooled mean inside the seed sd.

**Power.** 12 seeds against a seed sd of ~0.011 detects about 0.009. An effect
smaller than that is NOT measurable here and "not significant" is not evidence of
absence.

🛑 **Any read before 252/252 is EXPLORATORY** and is labelled so wherever it
appears. The verdict is taken once, at completion.


### The budget sweep (`bud_*`, written 2026-09-16 10:45, before any run finished)

**Why first.** The panel has 50+ metrics and the grid has 25 arms x 2 caps x 6
budgets; searching that surface for a winner would find one whether or not
anything is real.

**Grid:** fmow2, caps L80_G95 + L90_G95, 25 arms = {tralo, tralo_null, clip} x
{30, b12, b8, b7, b6, b5} + {alm, fioretto} x {30, b7, b6} + focal_clip.
Frozen per host (dsisco01 fp16 + grad scaler, dsisco02 bf16 no scaler).
🛑 **DO NOT POOL ACROSS HOSTS** -- the precision regimes differ, so
(backbone, HOST) is the unit and dsisco02 is not extra seeds.

**Discovery / confirmation split.** dsisco01 (seeds 1-6, fp16) is DISCOVERY;
dsisco02 (seeds 7-12, bf16) was CONFIRMATION. Nothing found on dsisco01 is a
result until it reproduces, same sign, on dsisco02 -- and **that confirmation set
no longer exists** (see RUN STATE). It must not be silently substituted for by the
247 partial runs.

**PRIMARY endpoints, in order:** `cc_f1`, then `F1 (Macro)`. Everything else
(constrained_precision, collateral_f1, ECE, Brier, per-class) is EXPLORATORY.

**The three contrasts, at each budget, averaged over SEED only:**
`tralo_bN` - `tralo_null_bN` (does the constraint do anything?);
`tralo_bN` - `clip_bN` (does it beat its MATCHED clipper?);
`tralo_bN` - `alm_bN` (b6, b7 only).

**What each outcome means, decided in advance:**
- **CONFIRMED** if `tralo - clip` is negative at 30/b12 and positive at b6/b5, on
  both primary endpoints, in discovery, with the sign reproduced in confirmation.
  The between-campaign evidence predicts exactly this (`live11` 30% live -0.0089;
  `live6b` 60% live +0.0079).
- **NULL** if the contrast is flat across budgets. Then the ~8 pp `track_b`
  warm-up effect did not survive to fmow2 + current code, and the regime lever
  closes with it.
- **TraLO SPECIFICALLY LOSES** if `tralo - alm` <= 0 wherever `tralo - clip` is
  positive. That is the recorded `track_b` outcome (ALM +9.18 vs TraLO +7.17) and
  would mean the FAMILY wins, not our method. **A real possible outcome, written
  down before the numbers existed.**
- **A budget that helps EVERY arm equally is not a constraint result.** The matched
  `clip_bN` and `tralo_null_bN` exist to catch that.

**Power.** 6 seeds per cell per host resolves ~0.013 at t=2 against a prior seed sd
of ~0.011. Effects below that are not measurable here.
Interim reading at 302/900 runs is in LEDGER PART 3 and is **not** a verdict.

### The ViT campaign (`vit_a` + `vit_b`, rewritten 2026-09-16 15:10)

Grid: ViTB16 x fmow2 x {L80_G95, L90_G95} x 12 seeds x 7 arms = {`clip`,
`focal_clip`, `tralo`, `tralo_null`, `focal_tralo`, `focal_tralo_null`, `alm`},
168 runs, ~14h. 🛑 **`vit_a` and `vit_b` differ ONLY in seed and share host,
precision and stamp, so they POOL to one 12-seed campaign. This is the one legal
pooling in the project and it is legal for exactly those reasons.**

🛑 **CORRECTIONS MADE IN PLACE, both before any evaluation metric was read.**
(a) The first version named `focal_tralo` - `focal_tralo_null` as the decisive
test of whether focal "unlocks the constraint"; **that question cannot be asked by
this campaign** and the claim is withdrawn -- ViTB16 reaches train accuracy >= 0.95
at epoch 1 (`alm` 0.906 -> 0.956) and epoch 2 (`clip` 0.917 -> 0.951) against a
29-epoch constraint phase, a live fraction of **3-7%**, the dead regime.
(b) A later version called `tralo` - `tralo_null` a negative control "expected null
BY CONSTRUCTION"; **that was wrong and is withdrawn** -- `fm2_vit` ran at the
IDENTICAL budget and its `tralo` - `tralo_null` is positive on both caps and both
endpoints (+0.0117/+0.0141 cc_f1, +0.0230/+0.0149 macroF1), so predicting zero
would have contradicted the existing data.

**What it CAN answer, and why it is worth 14 hours:** it is a like-for-like 12-seed
replication of `fm2_vit`, which used the identical budget at only 4 seeds.

- **PRIMARY: `focal_clip` - `clip` on cc_f1, then F1 (Macro).** Does the 2-point
  win replicate at n=12? The pre-registered headline.
- **PRIMARY: `focal_tralo` - `focal_clip`.** Can TraLO match the best known
  configuration once both carry focal? A tie is a real result; TraLO has never had
  a fair comparison against `focal_clip` at power.
- **CO-PRIMARY: `tralo` - `tralo_null`.** A straight 12-seed replication of the
  only cell in the fmow2 corpus where the attributable contrast survives its
  control. **Positive on both caps on cc_f1 at n=12 confirms it; a sign flip or a
  collapse toward zero refutes it and closes ViT as well.** `focal_tralo` -
  `focal_tralo_null` is the same question with focal attached.
- `alm` is the rival bar. Beating a null is not beating a rival.

🛑 **This campaign is NOT evidence about TraLO in the live regime and must never be
cited as such.** The live-regime ViT question needs constraint ~2-3 epochs
(live >= 50%), i.e. `focal_tralo_bN` arms that do not exist in
`configs/protocol.yml`. Adding them means editing a file inside
`source_inventory()` while five campaigns run, so it needs a SEPARATE WORKTREE and
is a compute-budget decision to be ASKED, not taken.

### The cap sweep (`cap_a` + `cap_b`, written 2026-09-16 16:05, at 0/180 runs)

**The question.** `tralo` - `tralo_null` on ViTB16 rests on n=4 and two adjacent
LOOSE caps. Does it hold across constraint strength? The dose axis (LEDGER PART 3)
spans 6x, from 3.6% of test evicted at `L90_G95` to 21.5% at `L40_G95`.

**Units.** `vit_a`, `vit_b`, `cap_a`, `cap_b` are all ViTB16 x fmow2 on dsisco02,
bf16 no-scaler, stamp `55c1be530de9`, differing only in seed and cap, so they pool
into ONE 5-cap x 12-seed ladder. 🛑 Nothing from dsisco01 joins it.

**Primary endpoint `cc_f1`, then `F1 (Macro)`. Averaged over SEED only.**

1. 🟢 **DOSE-RESPONSE (the strong result).** `tralo` - `tralo_null` should GROW as
   the cap tightens. Registered as a positive Spearman correlation between the
   per-cap mean difference and the eviction share, with the effect at `L40_G95`
   exceeding that at `L90_G95`. **This is the outcome that would make TraLO
   relevant**, and it is harder than a flat effect because it must order five rungs.
2. 🟡 **FLAT BUT PRESENT.** Positive at >= 4 of 5 caps at pooled n=12 but no
   ordering. Real, weaker, consistent with a constant offset.
3. 🔴 **REFUTED.** The mean crosses zero, or the sign flips at the tighter caps.
   This closes ViTB16, and with it the last open cell on fmow2 -- at which point
   the honest conclusion is that TraLO does not beat its own null anywhere we have
   looked.

**The rival bar, reported alongside and NOT as the headline.** `focal_clip`
currently beats `tralo` on ViT at both loose caps (-0.0221 / -0.0202 cc_f1).
`tralo` - `focal_clip` is exploratory at each cap; overtaking at the tight end is
the strongest available result but must be reported with the dose curve, not as a
single winning cell.

🛑 **Reading rule for all three.** Every `tralo` - clipper number is reported with
its matching `tralo_null` - clipper number on the same line; the first corpus
scoring pass produced 8 false hits out of 12 for want of exactly that.

---

## Open work

- [ ] **Decide the next experiment with the user** once the live campaigns land.
      The candidates are in LEDGER PART 5, priced. The cheapest and most decisive
      is the **budget-permuted twin**; the one motivated by a measured zero is
      **computing the constraint on a held-out fold of the train groups**; the one
      the surrogate diagnosis points at is **differentiable top-K**.
- [ ] Use the same greedy deployment allocator and the same saved probabilities for
      every arm. Clippers currently allocate with 256-item inference while eval
      saves a separate 512-item pass, and trained arms use a different allocator.
      **Correct this as a deployment-protocol change, not a TraLO gain.**
- [ ] Complete cc-F1-first, fixed-class metric reporting with paired native-unit
      uncertainty. Missing declared classes must count as zero, not disappear.
- [ ] Integrate shared structured logs: rival CSV initialisation erases warm-up
      history, and warm-up/rival task-step application plus rival displacement and
      local-scope state are missing. Make the first-run gates consume the records.
      Missing evidence is unknown, not absent.
- [ ] Deploy the `log_progress_to_csv` satisfaction fix (LEDGER PART 1) to the
      server **only after** the current campaigns finish -- it touches
      `source_inventory()`.
- [ ] Enforce exclusive canonical campaign ownership and safe crash recovery; the
      read-only audit found duplicate-root admission and a stale-running recovery
      mismatch.
- [ ] Require a fresh campaign identity and explicit source/config/data/quota
      inventory for reporting; test rejection of archived, mixed and unmarked runs.
      Use a new `OPTLOSS_MODEL_CACHE` namespace so fresh runs cannot reuse
      historical warm-up checkpoints.
- [ ] Audit development-cut saturation on the current datasets **without selecting
      on a TraLO win**.
- [ ] Remove the remaining unused weighted-CE option and orphan dependencies.

The first GPU experiment of any new direction tests pipeline validity and dataset
headroom, not superiority. Use an audited development split, one backbone and one
host before expanding. New loss changes stay deferred; the reference loss, dual
update ordering and training behaviour do not change during structural cleanup.

**Held-out selection.** There is no validation split on fmow2 -- every curve ever
looked at was read against the test set -- and a group-disjoint one IS
constructible. Full measurement and its caveats: LEDGER PART 5.

---

## Infrastructure notes

- `dsisco01` uses older GPUs and fp16; `dsisco02` is Blackwell and bf16, and it is
  **3.1x faster** (LEDGER PART 3). Storage is shared NFS, so **check processes on
  both hosts**.
- Canonical arrays are under `/home/dsi/michaer8/optloss-audit/data`. The dataset in
  use is `fmow2` (LEDGER PART 3, Datasets).
- `/home/dsi/michaer8/optloss-reset-validation-20260914` is an OLDER source
  snapshot; its earlier CPU test pass does not validate current source.
  **Re-sync and verify actual bytes before any campaign.**
- Recovery paths and backup verification: [`GIT_TRACKING.md`](GIT_TRACKING.md).
- Data provenance: `docs/archive/paper/data/PROVENANCE.md` -- **archived and
  contaminated with retired datasets.** Read it only for the fmow2 rebuild keys,
  never for results.

## Preservation

No old scientific result is promoted or erased by the reset. Folder titles carry no
evidential meaning. Archive records must state original path, destination,
inventory/hash verification and restore procedure. Superseded run state and
retired pre-registrations are in `docs/archive/mission_history_2026-09-16.md`.
