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

**TraLO does not clear the bar, and the backbone is a real moderator.**

- On MobileNetV2, MobileNetV3 and RegNetY400MF the constraint contributes nothing:
  every apparent clipper win is delivered by `tralo_null` at the same or a larger
  margin, and in `live11` the constraint is actively negative while its null posts
  the corpus's largest clipper win.
- On **ViTB16 only**, `tralo` - `tralo_null` is positive on 4 of 4 (cap x metric)
  combinations while the null LOSES to `clip` -- the one cell in 1,364 fmow2 runs
  where the attributable contrast survives its own control. **n=4, only 1 of 4
  reaches |t| >= 2, and seed 3 reverses everywhere.** A candidate, not a result.
- **`focal_clip` beats `tralo` on ViTB16 with CIs excluding zero** (-0.022 to
  -0.032 on F1 Macro and Precision Macro), so even the ViT cell is a live-null
  result, not a rival-beating one. The best arm measured anywhere is **`aug_clip`
  at budget 6** -- a post-hoc clipper with flip-and-crop and early stopping.

**The mechanism is proved, not guessed** (LEDGER PART 2). The loss is a function
of the multiset of test probabilities while the allocator is a function of their
ranks (M1), so nothing in the objective can prefer a correct ordering over the
worst ordering with the same multiset. The literature says post-hoc thresholding
IS the optimum for selection-rate constraints, with one crack: it is optimal only
when the score is Bayes-optimal (Woodworth et al., COLT 2017). **A training-time
win is permitted only by improving the SCORE.**

**And the root cause is measured** (LEDGER PART 2.3): every constraint term this
project has run is computed on TRAIN data, where the model reaches 0.9999 accuracy
and the violation is **identically zero in 174 of 174 cells**. The term is
multiplied by an empty support for ~80% of training. **Fixing the loss FUNCTION
cannot fix this -- the defect is which DATA the term is computed on.**

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

## RUN STATE -- checked 2026-09-16 22:57 IDT (server clock)

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

### LIVE: Option C on ViTB16

| campaign | host | GPU | caps | seeds | runs | budget |
|---|---|---|---|---|---|---|
| `polcv2_a` (runner `pcv2a`) | dsisco02 | 0 | L25_G25 L75_G75 | 1-6 | 84 | 5 (1+4) |
| `polcv2_b` (runner `pcv2b`) | dsisco02 | 1 | L25_G25 L75_G75 | 7-12 | 84 | 5 (1+4) |

Seven arms each, stamp `8085c356`, bf16, no scaler. Launched 02:33, ETA ~05:30.
Saturation gate re-checked at 56 runs: **50% live, `ok`**. GPUs 2-3 are
`liverty`; dsisco01 is idle but for `dvorata1` on GPU 0.

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

**The structural fix:** this work should not sit under a 200G home quota.

| path | free | writable by us |
|---|---|---|
| `/private/shared` | 5.2T | NO -- `shared_mngr` group, admin-managed |
| `/home/fast` | 200G (0 used) | NO -- root-owned |
| `/home/dsi/michaer8` | 0 | yes, but quota-capped |

**Ask the cluster admin for `/private/shared/michaer8`.** Neither roomy export
is writable by us today.

⚠️ **`model_cache` grows without bound across campaigns and nothing prunes it.**
Any future campaign plan must budget for it or clear it first.

---

## PRE-REGISTERED READINGS -- fixed before the numbers, do not reinterpret

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
