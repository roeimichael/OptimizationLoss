# MISSION -- where this project stands

**Updated 2026-09-15 (Asia/Jerusalem).** Evidence was reset on 2026-09-14 at the
user's request: historical results and folder names establish no winners and no
universal rejections.

Three files carry everything:

| File | Answers |
|---|---|
| [`RULESET.md`](../RULESET.md) | How to work. Rules, gates, the re-entry checklist. |
| **this file** | Where we are now, what is running, what is next. |
| [`LEDGER.md`](LEDGER.md) | What is proved, what is measured, what is closed. |

[`FRAMEWORK.md`](FRAMEWORK.md) is the protocol and outranks all three.

**`keep working`** reloads all four from disk, checks both hosts, does the next
highest-value thing, and writes what it learns back here and into the ledger.
Run it whenever the thread needs picking up.

---

## THE COURSE -- read this first, every session

*This section is the contract. If a result changes the course, edit THIS section.
Do not leave it stale and correct it further down.*

### The one question

**Can a training-time constraint beat a post-hoc clipper on a capped-class
task**, at matched data, compute, allocator and precision?

**The bar, set by the user:** leading group on **cc-F1** AND not dominated on the
rest of the metric profile. A win on accuracy or macro-F1 while losing cc-F1 is a
**TRADE**, not a win, and is reported as one.

A valid negative result is a real deliverable. Do not start a new research task
to manufacture an advantage.

### Where the answer stands today

**TraLO does not clear the bar.** On the two complete 30-epoch cells it trails
every one of the six rivals on cc-F1, three of those gaps beyond their own seed
sd -- including its own zero-constraint null. The best arm measured anywhere is
`aug_clip`, a post-hoc clipper with augmentation.

**The mechanism for why is proved, not guessed** (`LEDGER.md` PART 2): the loss
is a function of the multiset of test probabilities while the allocator is a
function of their ranks, so nothing in the objective can prefer a correct
ordering over the worst ordering with the same multiset. Cross-entropy supplies
the ordering while it is alive. Once train CE reaches ~0 it stops, and the
constraint step is rescaled to full size regardless of how small the violation
is -- a full-size push with nothing opposing it and no information about
correctness.

**Every cell memorises in 2-4 epochs**, so in a 30-epoch budget roughly 3 epochs
are useful and 26 push a frozen boundary.

🔑 **The direction that is actually new.** The constraint's soft count is an
UNWEIGHTED sum of probabilities, so every item in a scope receives the same
`dL/dp_i(c)` and the only per-item differentiation is `p_i(1-p_i)`. Everything it
knows about an item is `p_i(c)` -- the quantity the allocator already ranks by,
which is why PART 2.1 proves it cannot re-order. The harm lemma assumes the
scores exhaust the available label information, and **they do not**: measured
2026-09-15, neighbourhood agreement in the model's own embedding space
identifies the confidently-wrong at **AUC 0.87**, against 0.68 for anything
derivable from `p`. `tralo_stab` in LEDGER PART 5 is the falsifiable
modification that follows, with its derivative, log signature, matched control
and failure criterion.

**The other live thread** is that the damage appears to stop when the constraint
acts while the boundary is still moving. Four campaigns at four budgets order
monotonically on a fixed backbone, and the same sign flip shows up independently
on the deployed selection. Neither is powered: p = 0.20 across campaigns, and the
pre-registered interaction is p = 0.22 at n = 4. **This is the open question, and
it is open because nothing run so far could have resolved it.**

### What is settled

Eight settled findings and five proved results live in [`LEDGER.md`](LEDGER.md).
Do not re-open any of them without new evidence. The four that most often get
re-litigated:

1. The four rival duals share one per-item gradient and are one family. Measuring
   them tied is a confirmed prediction, not a disappointment.
2. Eviction given the probabilities is already ~87% optimal; the prize is not in
   the loss shape.
3. The exact allocator makes real results worse in 14/14 cells. Cancelled.
4. The cap IS binding and there IS headroom: every slot is filled, ~1,000 of
   ~4,400 hold wrong items, ~1,500 true positives sit outside the cut. **But
   39-46% of those wasted slots are held by items the model is certain about and
   wrong, so the reachable prize is ~12-15% of the budget, not 20-25%.**

### Standing decision rules

Full set in [`RULESET.md`](../RULESET.md). The four that bite most often:

- **`gate:saturation` is a screen, not a diagnosis.** It reads TRAIN accuracy,
  and FRAMEWORK gate 2 says that cannot diagnose test-cut saturation. Pair it
  with the regime check at the real allocation cut.
- **Average over SEED only.** A cap level is not a seed, a copied warm-up is not
  a seed, and a re-run is not a seed -- training is bit-deterministic, so
  de-duplicate by prediction hash before counting n.
- **Four seeds is a pilot.** Effects here are ~0.01 against a seed sd of ~0.011.
- **Report a trade as a trade**, and retract in place.

---

## Run state, checked 2026-09-15 19:13 IDT (server clock)

**STAGE 1 OF THE RANKING PIVOT IS LIVE: three campaigns on dsisco01 GPUs 1, 2, 3**
(`rank1_MobileNetV3`, `rank1_MobileNetV2`, `rank1_RegNetY400MF`, 40 runs each,
budget 30, fmow2). Tree: **a SEPARATE worktree `~/optloss-rank` pinned at
`338110cc`**, so the four completed campaigns in `~/optloss-probe` keep their
frozen `source_inventory` and can still be re-scored. 20/20 pre-launch gates.

🛑 **dsisco02 is FULLY OCCUPIED by `liverty`; dsisco01 GPU 0 is `dvorata1`.**
Never share a card. Only GPUs 1-3 on dsisco01 are ours.
🛑 **Do not touch `src/`, `configs/`, `scripts/` in `~/optloss-rank` until these
finish.**

### The pivot, and why it is not another arm

The user authorised a staged pivot on 2026-09-15, having accepted that the
count-based question is answered NO. LEDGER PART 2.1 proves a count penalty
reads the MULTISET while the allocator reads the RANKS; the literature (now in
PART 2) proves post-hoc thresholding is the OPTIMUM for selection-rate
constraints, with one crack -- it is optimal only when the score is
Bayes-optimal (Woodworth et al. COLT 2017). **A training-time win is permitted
only by improving the score.**

`rank_clip` does that: a hinge around the per-group K-th order statistic -- the
exact point the allocator cuts -- on TRAIN labels with the budget simulated. The
cut stays in the graph, so items compete for the K slots. No test label enters
any gradient.

### STAGE 1 GATE, pre-registered

**Does gAP(`rank_clip`) - gAP(`clip`) > 0?** gAP is allocation-free, so it
isolates the SCORE. `rank_clip` carries no constraint at all -- deliberately: if
the ranking channel is real it must appear here, with the dual machinery out of
the picture entirely.

- **Flat in all three backbones** -> the ranking channel is dead. Stop. Write the
  negative result, which is now a strong one with citations.
- **Moves** -> Stage 2 is justified: differentiable top-K through the allocator
  (Petersen ICML 2022; Xie NeurIPS 2020; Berthet NeurIPS 2020), plus
  `rank_tralo` / `rank_tralo_null`, which are BUILT and gated but deliberately
  not yet run -- their result is uninterpretable until the channel is shown.

Honest prior, recorded before the seeds land: ~60% that gAP moves, ~25% that it
helps TraLO specifically more than its null. **If it lifts the clipper too, that
is still the paper** -- the claim becomes "for budgeted deployment the training
signal that matters is the ranking at the cut, not count satisfaction", which is
a positive result motivated by our own negative one.

### First-run checks owed on these

`AMP: float16 + GradScaler` (dsisco01), and that `rank_clip` is NOT byte-identical
to `clip` -- the warm-up is cached and `rank_weight` is an identity key precisely
to stop that, but it must be confirmed on the first completed pair.

---

## The data

**`fmow2` is the dataset.** 17,670 train / 3,442 test, row counts consistent
across images, labels and meta. **139 train countries vs 10 test countries, zero
overlap. Zero cross-split exact image duplicates.** Passes **8 of 8** conditions
in `scripts.candidate_gate` (density 0.82, 6% dead items, 6/26 zero ceilings,
class balance 0.57). Capped classes come from this slice's own labels: **1
crop_field, 2 place_of_worship, 7 ground_transportation_station**, present in
10/10, 9/10 and 8/10 groups.

It replaced the original `fmow/oodslice`, which was **withdrawn**: `prep_fmow`
joined metadata to images on `os.path.basename`, but the archive is laid out
`split/class/class_seq/aoi/file` and the filename does not encode the AOI. On
`val-metadata.tar.gz`, 7,429 of 53,041 basenames appear under more than one AOI,
so **16.4% of records were silently dropped or mis-joined**. Both sides now key
on `class_seq/aoi/file` and `load()` refuses a non-unique key. Old arrays are
preserved; the rebuild is a separately versioned slice.

**Measured hardness on fmow2 / MobileNetV3.** Train CE saturates by epoch 6 at
99.7% train accuracy, but **test accuracy is 0.634-0.648 over 4 seeds** and a
cell carries **187 errors inside K**. On ~11 of 30 ceilings p@K >= 0.99 -- the
model is confidently wrong, and the penalty's `p(1-p)` gradient is near zero
exactly there. **That is a calibration limit, not a data limit, and it is the
open question on this slice.**

**`bcn` is blocked on integrity.** Two exact duplicate pairs cross train/test
with conflicting class labels and different official lesion IDs; public source
JPEG and annotation checks confirm the conflict is upstream, not introduced by
our export. No images or labels were changed. It is otherwise the
best-structured slice available (candidate_gate 7/8, failing only class balance
at 0.04), so repairing it is worth doing rather than abandoning. A versioned
curation policy and a renewed whole-split audit are still required.

**iwildcam is RETIRED.** It passes 2 of 8 conditions: 2 of 8 classes can carry a
local cap, half the per-group ceilings are K=0 before training starts, and 72% of
test items sit in groups holding NEITHER capped class, so those groups cannot
produce a single allocation decision. Its per-group label shift is the best in
the corpus (TV 0.737) and that is the SAME fact as its density of 0.27 -- the
shift IS the sparsity. **Every earlier TraLO number was measured through it, so
treat pre-fmow2 results as describing iwildcam rather than the method.** The two
tools that could have caught this disagree by construction (`dataset_screen`
rewards shift, `tier_viability` rewards density), so whichever was run said "it
passes"; `scripts.candidate_gate` now screens all eight at once.

---

## Open work

- [ ] **Decide the next experiment with the user.** The budget sweep is designed
      and gated but unapproved. The strongest untested alternative in the ledger
      is the **budget-permuted twin** (`LEDGER.md` PART 5): identical code and
      schedule, budgets permuted across groups within a class. It isolates
      whether the budgets do any work at all, and it is cheap.
- [ ] Use the same greedy deployment allocator and the same saved probabilities
      for every arm. Clippers currently allocate with 256-item inference while
      eval saves a separate 512-item pass, and trained arms use a different
      allocator. **Correct this as a deployment-protocol change, not a TraLO
      gain.**
- [ ] Complete cc-F1-first, fixed-class metric reporting with paired native-unit
      uncertainty. Missing declared classes must count as zero, not disappear.
- [ ] Integrate shared structured logs: rival CSV initialisation erases warm-up
      history, and warm-up/rival task-step application plus rival displacement
      and local-scope state are missing. Make the first-run gates consume the
      records. Missing evidence is unknown, not absent.
- [ ] Enforce exclusive canonical campaign ownership and safe crash recovery; the
      read-only audit found duplicate-root admission and a stale-running recovery
      mismatch.
- [ ] Require a fresh campaign identity and explicit source/config/data/quota
      inventory for reporting; test rejection of archived, mixed and unmarked
      runs. Use a new `OPTLOSS_MODEL_CACHE` namespace so fresh runs cannot reuse
      historical warm-up checkpoints.
- [ ] Audit development-cut saturation on the current datasets **without
      selecting on a TraLO win**.
- [ ] Remove the remaining unused weighted-CE option and orphan dependencies.

The first GPU experiment of any new direction tests pipeline validity and dataset
headroom, not superiority. Use an audited development split, one backbone and one
host before expanding. New loss changes stay deferred; the reference loss, dual
update ordering and training behaviour do not change during structural cleanup.

---

## RUN STATE -- checked 2026-09-16 18:37 IDT (server clock).

🛑 **dsisco02 IS WEDGED AS OF ~18:15. `vit_a`, `vit_b`, `cap_a`, `cap_b` ARE
BLOCKED, NOT LOST.** Diagnosed from dsisco01 over shared NFS, because ssh to
dsisco02 fails at the banner exchange:

* `ping` succeeds and TCP 22 is OPEN from dsisco01, so the **host is up** and
  it did not reboot. sshd accepts the connection and then never speaks.
* `vit_a` and `vit_b` are frozen at **21/84 each**. Two samples two minutes
  apart show identical completion counts, identical queue-log sizes and an
  identical mtime, against a ~10.3 min/run rate. The jobs are not progressing.
* That single cause -- an NFS/IO hang on dsisco02 -- explains all of it: sshd
  blocks reading `/home` for auth, and the trainers block writing to `/home`.

**42 completed ViT runs are safe on NFS and readable from dsisco01.** Nothing
is deleted and nothing needs regenerating. 🛑 **Do NOT relaunch these on
dsisco01**: they are frozen for dsisco02/bf16 and `validate_campaign` refuses a
cross-host release, so a relaunch would be a different unit, not a recovery.
Recovery needs console or admin access to dsisco02.

**RUN-STATE ON NFS, checked 18:37 and again 19:34 (unchanged, confirming the
jobs are blocked rather than slow):**

| campaign | completed | running (stuck) | pending |
|---|---|---|---|
| `vit_a` | 21 | 1 | 62 |
| `vit_b` | 21 | 1 | 62 |
| `cap_a` | 0 | 0 | 90 |
| `cap_b` | 0 | 0 | 90 |

### RECOVERY RUNBOOK for dsisco02 -- follow in order, do not skip step 2

1. Confirm ssh answers: `ssh dsisco02 'echo ALIVE; uptime'`.
2. 🛑 **Confirm NO trainer of ours is alive before touching any config:**
   `pgrep -u michaer8 -f 'main\.py'` and `pgrep -u michaer8 -f queue_runner`.
   If a trainer is alive it may simply have UNBLOCKED and be finishing its run.
   Resetting its config to `pending` while it still holds the directory gives
   two writers to one run. Either wait for it, or stop it by EXPLICIT PID
   (INT, then TERM/KILL), scoped by reading `/proc/<pid>/environ` for
   `EXPERIMENT_DIR`. Never `pkill`, never `grep main.py`.
3. Reset ONLY the stuck run in each campaign. `scripts/reset_crashed.py` is the
   maintained tool and is conservative by construction: a run is eligible only
   if it has NO usable result (no `results.accuracy`, no full `training_log.csv`
   of >= 5 rows), so the 21 completed runs cannot be clobbered. Dry run first:
   `python -m scripts.reset_crashed results/vit_a` then `--apply`. Same for
   `vit_b`. `cap_a`/`cap_b` need nothing -- they never started.
4. Re-validate each: `python -m src.pipeline.campaign validate --root results/vit_a`.
   🛑 Do NOT re-freeze. The campaigns are already frozen against dsisco02/bf16
   and the stamp `55c1be530de9` must not move.
5. Relaunch with a **NEW label** so a fresh log file is created:
   `setsid nohup bash ~/queue_runner_v2.sh 0 vita2 "$WT:results/vit_a" </dev/null >>~/queue_logs/vita2_launch.log 2>&1 &`
   (and gpu 1 / `vitb2` / `vit_b`). 🛑 Never reuse the old label: the previous
   log may still be held open, and rewriting a live log is what orphaned
   `vita_vit_a.log` earlier today (RULESET 6).
6. Re-queue the cap sweep behind them, same pattern, labels `capa2` / `capb2`.
7. `~/camp_status.sh` needs no edit -- it resolves logs by `*_<campaign>.log`.

🛑 **Do NOT relaunch any of these on dsisco01.** They are frozen against
dsisco02/bf16 and `validate_campaign` refuses a cross-host release. A dsisco01
run would be a different (backbone, HOST) unit that cannot pool with the 42
completed runs, so it is a new experiment, not a recovery.

**Decision if dsisco02 stays down.** Rebuilding the ViT ladder on dsisco01 means
Turing fp16 at an estimated 15-20 min/run, so a 5-cap x 12-seed x 4-arm ladder
is 240 runs / 60-80 GPU-hours, and dsisco01 has no free card -- it would cost
stopping one or two `bud_*` campaigns. That is 30-80 hours to reproduce what
dsisco02 does in ~25. **Recommendation: wait for the box and keep `bud_*`
running.** Revisit only if dsisco02 is down for more than about a day.

**dsisco01 is unaffected** and its three campaigns are advancing normally
(`bud_mn3` 96/300, `bud_mn2` 77/300, `bud_rgn` 100/300 at 18:37).

## RUN STATE -- campaign grid (checked 2026-09-16 14:39 IDT)

**The budget sweep. 1,500 runs, five GPUs, both hosts.** Launched after
LEDGER 2.1b established that rank1/rank2/rank3 (360 runs) all ran at an 8-10%
live fraction -- the dead regime -- because `saturation_gate.py` skipped every
run with `constraint_epochs = 0` and reported "no training_log.csv matched".

| campaign | host | GPU | backbone | seeds | runs |
|---|---|---|---|---|---|
| `bud_mn3` | dsisco01 | 1 | MobileNetV3 | 1-6 | 300 |
| `bud_mn2` | dsisco01 | 2 | MobileNetV2 | 1-6 | 300 |
| `bud_rgn` | dsisco01 | 3 | RegNetY400MF | 1-6 | 300 |
| `vit_a` | dsisco02 | 0 | ViTB16 | 1-6 | 84 |
| `vit_b` | dsisco02 | 1 | ViTB16 | 7-12 | 84 |
| `cap_a` | dsisco02 | 0 | ViTB16 | 1-6 | 90 (queued behind `vit_a`) |
| `cap_b` | dsisco02 | 1 | ViTB16 | 7-12 | 90 (queued behind `vit_b`) |

**`bud_mn3b` and `bud_mn2b` were STOPPED on dsisco02 on 2026-09-16, immediately before the ViT launch at 14:28:34**, by
explicit PID, runners first so neither could dispatch a replacement, then the
trainers (`kill -INT`, no escalation needed). **247 completed runs are preserved
on disk** -- `bud_mn3b` 134, `bud_mn2b` 113 -- nothing was deleted. The cost of
the stop is the CONFIRMATION half of the budget sweep: the dsisco01 discovery
set still runs to completion, but its pre-registered confirmation on a second
precision regime no longer exists and must not be silently substituted for by
the partial 247. **Those 247 runs are a truncated, non-random prefix of the
grid** (the runner walks the grid in order), so they are NOT a 45% sample of the
sweep and must not be scored as one.

The two freed Blackwell cards carry the ViT campaign instead. **Grid:** ViTB16 x
fmow2 x {L80_G95, L90_G95} x 12 seeds x 7 arms = {`clip`, `focal_clip`, `tralo`,
`tralo_null`, `focal_tralo`, `focal_tralo_null`, `alm`}, 168 runs, ~14h.
Generated AND frozen on dsisco02 (bf16, no scaler); single `code_version` stamp
`55c1be530de9` on all 168 configs, the same pinned tree the `bud_*` runs use.

🛑 **`vit_a` and `vit_b` differ ONLY in seed (1-6 vs 7-12) and are the same host,
same precision, same stamp. They POOL to one 12-seed campaign.** This is the one
legal pooling in the project and it is legal for exactly those reasons; it is
not a licence to pool anything else.

**PRE-REGISTERED READING for the ViT campaign (rewritten 2026-09-16 14:52, after
the first 4 runs' TRAINING LOGS but before any evaluation metric was read).**

🛑 **CORRECTION, MADE IN PLACE.** The first version of this block, written at
14:40, named `focal_tralo` - `focal_tralo_null` as the decisive test of whether
focal "unlocks the constraint". **That question cannot be asked by this
campaign** and the claim is withdrawn here. Measured from the first four
training logs: ViTB16 reaches train accuracy >= 0.95 at **epoch 1** (`alm`,
0.906 -> 0.956) and **epoch 2** (`clip`, 0.917 -> 0.951), against a **29-epoch
constraint phase**. Live fraction is **3-7%** -- the DEAD regime, the same
condition LEDGER 2.1b showed had invalidated rank1/rank2/rank3. The constraint
steps here act on a frozen boundary.

**What this campaign CAN answer, and why it is still worth its 14 hours.** It is
a like-for-like 12-seed replication of `fm2_vit`, which used the identical
budget (warmup 1 + constraint 29) at only **4 seeds** and is where `focal_clip`'s
2-point cc-F1 win was observed. At 4 seeds this project has ~15% power against a
seed sd of ~0.011, so that 2-point win is currently one lightly-measured number.

- **PRIMARY: `focal_clip` - `clip` on cc_f1, then F1 (Macro).** Does the 2-point
  win replicate at n=12? This is the pre-registered headline.
- **PRIMARY: `focal_tralo` - `focal_clip`.** Can TraLO match the best known
  configuration once both carry focal? A tie is a real result; TraLO has never
  had a fair comparison against focal_clip at power.
- 🛑 **SECOND CORRECTION, 15:10, still before any `vit_a`/`vit_b` evaluation
  metric has been read (only training logs were opened).** The 14:52 version of
  this bullet called `tralo` - `tralo_null` a negative control "expected null BY
  CONSTRUCTION" at 3-7% live. **That was wrong and is withdrawn.** `fm2_vit` ran
  at the IDENTICAL budget (warmup 1 + constraint 29) and its `tralo` -
  `tralo_null` is **positive on both caps and both endpoints** (+0.0117/+0.0141
  cc_f1, +0.0230/+0.0149 macroF1), with the null LOSING to `clip` while `tralo`
  beats it. A low live fraction evidently does not force this contrast to zero on
  ViT, so predicting zero here would have been a prediction the existing data
  already contradicts.

- **CO-PRIMARY, and the reason this campaign matters: `tralo` - `tralo_null`.**
  This is a straight 12-seed replication of the only cell in the entire fmow2
  corpus where the attributable contrast survives its own control. The `fm2_vit`
  prior is n=4, carried by 3 of 4 seeds with seed 3 reversing at both caps, and
  only 1 of 4 (cap x metric) combinations reaches |t| >= 2.
  **Pre-registered reading:** positive on both caps on cc_f1 at n=12 confirms it;
  a sign flip or a collapse toward zero refutes it and closes ViT as well.
  `focal_tralo` - `focal_tralo_null` is the same question with focal attached.
- `alm` is the rival bar. Beating a null is not beating a rival.

🛑 **This campaign is NOT evidence about TraLO in the live regime and must never
be cited as such.** The live-regime ViT question needs a short budget
(constraint ~2-3 epochs, so live >= 50%), which requires `focal_tralo_bN` arms
that do not exist in `configs/protocol.yml`. Adding them means editing a file
inside `source_inventory()` while five campaigns run, which would invalidate
every frozen release, so it must be done in a SEPARATE WORKTREE and is a
compute-budget decision to be ASKED, not taken.

Do not reinterpret any of these after seeing the numbers.

Grid: fmow2, caps L80_G95 + L90_G95, 25 arms = {tralo, tralo_null, clip} x
{30, b12, b8, b7, b6, b5} + {alm, fioretto} x {30, b7, b6} + focal_clip.
Frozen per host (dsisco01 fp16 + grad scaler, dsisco02 **bf16, no scaler** --
`validate_campaign` refuses a cross-host release, correctly).

🛑 **DO NOT POOL ACROSS HOSTS.** The precision regimes differ, so
(backbone, HOST) is the unit. dsisco02 is not extra seeds.

## BUDGET SWEEP -- PRE-REGISTERED READING (written 2026-09-16 10:45, before any run finished)

**Why this is written first.** The standing goal is "a valid metric on which we
beat the rivals". The panel has 50+ metrics and the grid has 25 arms x 2 caps x
6 budgets; searching that surface for a winner and reporting it would find one
whether or not anything is real. The protections are fixed here, in advance.

**DISCOVERY / CONFIRMATION SPLIT.** dsisco01 (3 campaigns, seeds 1-6, fp16) is
the DISCOVERY set. dsisco02 (2 campaigns, seeds 7-12, bf16, independent
numerics) is the CONFIRMATION set. **Nothing found on dsisco01 is a result
until it reproduces, same sign, on dsisco02.** The two MobileNet backbones are
run on both hosts precisely so this is possible.

**PRIMARY endpoints, in this order, fixed now:**
1. `cc_f1` -- the deployed endpoint.
2. `F1 (Macro)` -- the user's alternative, equally admissible.

Everything else in the panel (constrained_precision, collateral_f1, ECE,
Brier, per-class) is EXPLORATORY: reportable, but only as a hypothesis for the
confirmation set, never as a headline from discovery alone.

**The three contrasts, at each budget, averaged over SEED only:**
- `tralo_bN` - `tralo_null_bN` -- does the constraint do anything?
- `tralo_bN` - `clip_bN` -- does it beat its MATCHED clipper?
- `tralo_bN` - `alm_bN` -- do we beat ALM? (b6, b7 only; these are the new arms)

**What each outcome means, decided now:**
- **The live-regime account is CONFIRMED** if the `tralo - clip` contrast is
  negative at budget 30/b12 and positive at b6/b5, on both primary endpoints,
  in the discovery set, and the sign reproduces in confirmation. The
  between-campaign evidence predicts exactly this (`live11` 30% live -0.0089;
  `live6b` 60% live +0.0079).
- **The regime is a null** if the contrast is flat across budgets. Then the
  ~8 pp `track_b` warm-up effect did not survive to fmow2 + current code, and
  the regime lever is closed with it.
- **TraLO specifically loses** if `tralo - alm` is <= 0 wherever `tralo - clip`
  is positive. That is the recorded `track_b` outcome (ALM +9.18 vs TraLO
  +7.17) and would mean the FAMILY wins, not our method. **This is a real
  possible outcome and it is written down before the numbers exist.**
- **A budget that helps EVERY arm equally is not a constraint result.** The
  matched `clip_bN` and `tralo_null_bN` exist to catch exactly that; the
  protocol's own comment says short budgets may simply help everything.

**Power.** 6 seeds per cell per host, 12 across hosts but NOT poolable. Prior
seed sd on cc-F1 is ~0.011, so a 6-seed cell resolves ~0.013 at t=2. Effects
below that are not measurable here and must not be reported as findings.

## CAP SWEEP -- PRE-REGISTERED READING (written 2026-09-16 16:05, at 0/180 runs)

**The question.** `tralo` - `tralo_null` on ViTB16 is the only contrast in the
entire 1,364-run fmow2 corpus that survives its own control (LEDGER PART 3). It
is n=4 and rests on two adjacent LOOSE caps. This sweep asks whether it holds
across constraint strength.

**The dose axis, verified 2026-09-16 with the pipeline's own
`compute_local_constraints` on `data/fmow2/oodslice/test_meta.csv`** (not with a
hand-rolled approximation -- `_round_to_K` rounds where an earlier check
floored, and the numbers differ):

| cap | binding ceilings (evict >= 10) | items evicted | share of test |
|---|---|---|---|
| `L40_G95` | 18 / 30 | 740 | **21.5%** |
| `L55_G95` | 15 / 30 | 553 | 16.1% |
| `L70_G95` | 12 / 30 | 370 | 10.7% |
| `L80_G95` | 9 / 30 | 247 | 7.2% |
| `L90_G95` | 5 / 30 | 123 | **3.6%** |

Zero K=0 ceilings where the class is present. The three K=0 warnings
(NLD/c7, IND/c2, EGY/c7) are groups with no true instance of that class and are
correct. **This is a 6x dose range, and the two caps this project has always
used are the two WEAKEST rungs.**

**Units and pooling.** `vit_a`, `vit_b`, `cap_a`, `cap_b` are all ViTB16 x fmow2
on **dsisco02, bf16 no-scaler, single code_version `55c1be530de9`**, differing
only in seed and cap. They therefore pool into ONE 5-cap x 12-seed ladder. This
is the same narrow licence recorded for `vit_a`/`vit_b`: same host, same
precision, same stamp. 🛑 Nothing from dsisco01 joins it.

**Primary endpoint `cc_f1`, then `F1 (Macro)`. Averaged over SEED only.**

**Pre-registered outcomes, in order of strength:**

1. 🟢 **DOSE-RESPONSE (the strong result).** If the constraint does real work,
   `tralo` - `tralo_null` should GROW as the cap tightens, because a tighter cap
   evicts more items and leaves more for the constraint to influence. Registered
   as: a positive Spearman correlation between the per-cap mean difference and
   the eviction share above, with the effect at `L40_G95` exceeding the effect at
   `L90_G95`. **This is the outcome that would make TraLO relevant**, and it is a
   much harder target than a flat effect because it must order five rungs.
2. 🟡 **FLAT BUT PRESENT.** Positive at >= 4 of 5 caps with the pooled n=12 but
   no ordering. Real, weaker, and consistent with a constant offset rather than
   a constraint that responds to its own budget.
3. 🔴 **REFUTED.** The mean crosses zero, or the sign flips at the tighter caps.
   This closes ViTB16, which closes the last open cell on fmow2 -- at which point
   the honest conclusion is that TraLO does not beat its own null anywhere we
   have looked.

**The rival bar, reported alongside and NOT as the headline.** `focal_clip`
currently beats `tralo` on ViT at both loose caps (-0.0221 / -0.0202 cc_f1).
`tralo` - `focal_clip` is registered as exploratory at each cap. If TraLO
overtakes `focal_clip` at the tight end that is the strongest available result,
but it must be reported with the dose curve, not as a single winning cell.

🛑 **Reading rules.** Every `tralo` - clipper number is reported with its
matching `tralo_null` - clipper number on the same line; the first pass of the
corpus scoring produced 8 false hits out of 12 for want of exactly that. Do not
reinterpret any outcome after seeing it.

## RUN STATE -- rank3 (superseded, kept for provenance)

**NOTHING IS RUNNING. dsisco01 GPUs 1, 2, 3 are free by decision, not by
accident** -- Stage 1 is answered and Stage 2 is a compute-budget question for
the user. GPU 0 is `dvorata1`; dsisco02 GPUs 1-3 are `liverty`.

- `rank1_*` -- DEAD. 24 usable / 16 failed per campaign on the unpack defect.
  Its 72 CONTROL runs are valid and were used to measure the gAP noise envelope.
- `rank2_*` -- DEAD. Stopped by explicit PID at 2/40 on the warm-up cache
  defect, before it could write wrong numbers.
- `rank3_*` -- **COMPLETE. 120/120 runs, zero failures, score gates GREEN on all
  three.** 84 distinct models, 36 shared hashes all of which are cap-invariant
  controls, zero rank-arm collisions. Answer in LEDGER PART 2.2.

🛑 The tree stays pinned at **323edf44**. `rank_min_group` is declared an
identity key in git but deliberately NOT deployed there: changing
`configs/protocol.yml` would break `rank3`'s frozen `source_inventory` and cost
the ability to re-score it with the maintained reporter. It applies when the
next campaign is generated.

**Stage 1 verdict: the budgeted ranking loss does not supply the "which".**
`rank_clip` - `clip` is 4 of 18 cells positive at cell-mean gAP -0.0090;
`aug_rank_clip` - `aug_clip` is 7 of 18 at -0.0039; on cc-F1 the ranking arm
trails its control in 11 of 12 cells. Diagnosis in LEDGER PART 2.2: the gradient
is uncertainty-weighted, not cut-anchored -- partial correlation with distance
from the cut is -0.145 once the softmax Jacobian is held fixed, against +0.604
the other way.

**The decision now open, and it is the user's** (changes the compute budget):

| option | what it tests | rough cost |
|---|---|---|
| **Stage 2: differentiable top-K** (Petersen / Xie / Berthet) | differentiate the SELECTION itself, so the gradient stays rank-dependent through the backward pass -- the specific defect PART 2.2 identifies | build + 1 campaign, ~5.5h on 3 cards |
| **Fix the dose first** (group-batched sampler) | whether the 8-of-139-groups deficit was the binding constraint after all | sampler change + 1 campaign, ~5.5h |
| **Val-split stopping rule** (LEDGER PART 5) | the only candidate so far that could produce a POSITIVE reportable result | retrain on 83% of train, ~5.5h |
| **Stop and write the negative result** | PART 2.1 + PART 2.2 are a coherent, well-evidenced story | 0 |

My reading: the dose fix is the weakest of the three, because PART 2.2's
diagnosis points at the surrogate rather than the dose -- more gradient of a
still-uncertainty-weighted term.

---

## Stage 1 ranking gate -- PRE-REGISTERED READING (written 2026-09-15, before any rank arm landed)

Three campaigns live on dsisco01 GPUs 1/2/3: `rank1_MobileNetV3`,
`rank1_MobileNetV2`, `rank1_RegNetY400MF`; 40 runs each; arms `clip`,
`focal_clip`, `rank_clip`, `aug_clip`, `aug_rank_clip`. **No new scorer is
needed** -- `scripts/rank_paired.py` already gives per-cell paired gAP with the
seed sd beside it:

```
python3 scripts/rank_paired.py --glob 'results/rank3_*/*/*/*/*/seed_*' --a rank_clip     --b clip
python3 scripts/rank_paired.py --glob 'results/rank3_*/*/*/*/*/seed_*' --a aug_rank_clip --b aug_clip
```

**The mechanism check comes free, and it is read FIRST.** `rank_paired` marks an
arm `(cap-inert)` when its probabilities are byte-identical across L80 and L90.
Both control arms are already marked that way, correctly: no constraint reaches
their model, so the cap acts only in the post-hoc allocator. But `rank_frac` is
read from the cap (`warmup.py:165` -- 0.8 at L80, 0.9 at L90), so a LIVE ranking
loss trains two different models and `rank_clip` **cannot** be cap-inert.

| what `rank_paired` shows for `rank_clip` | what it means | what follows |
|---|---|---|
| `(cap-inert)` | **AMBIGUOUS, do not read it as death.** Until 323edf44 this was guaranteed by a CACHE bug -- both caps shared one `base_model_id`, so L90 loaded L80's warm-up. In `rank3_*` that cause is removed and the two caps carry different digests (verified in the generated configs), so cap-inertness would now mean the loss really did not reach the model. Check the digests before reading it either way. | if the digests differ and the arm is still cap-inert: a sixth dead flag, discard, do NOT read gAP |
| separate L80 / L90 rows | the loss moved the model | proceed to read gAP |

**Then, and only then, the gate: is gAP(`rank_clip`) - gAP(`clip`) > 0?**
Read as cells, never pooled -- 3 backbones x 3 constrained classes = 9 cells per
contrast, with `|mean|/sd` beside each.

- **>= 6 of 9 cells positive, on both contrasts** -> the ranking channel moves
  the score. This is the first positive result in the project. Proceed to
  Stage 2 (differentiable top-K through the allocator).
- **mixed, or <= 3 of 9 positive** -> **NOT a refutation of the ranking
  channel.** LEDGER PART 3 records why: this loss fires on 8 of 139 train
  groups and trains a 2.3rd-of-12 order statistic to serve a 41st-of-363
  decision. A null here is confounded with a 29x estimator deficit. The next
  move would be group-batched sampling, which is a relaunch and therefore a
  question for the user -- not a unilateral change, and not a closed direction.

🛑 This table is fixed BEFORE the numbers. Honour it; do not reinterpret after
seeing them. Every headline this project has had to retract came from reading a
result and then choosing what it meant.

---

## Open user question -- ANSWERED 2026-09-15

**Are there untouched evaluation groups or splits on the current datasets?**

**No untouched split exists, and one IS constructible.** Measured on
`data/fmow2/oodslice/` (`scripts/val_split.py`):

- The slice ships **train and test only**. There is no validation split, so
  every epoch, checkpoint or hyperparameter ever chosen by looking at a curve
  was chosen against the test set.
- Train and test are **group-disjoint by construction**: 139 countries vs 10,
  **zero overlap** (test is CAN, DZA, EGY, IND, IRQ, JPN, MEX, NLD, PHL, TUR).
  The deployment shift this dataset poses is a GROUP shift, so a validation
  split drawn by shuffling rows would leave the same countries on both sides and
  measure a strictly easier problem.
- 35 train countries carry >= 80 items, which is enough to carve a
  **group-disjoint** val split that imitates the test profile: e.g. ARG, BRA,
  CHE, CHL, DEU, KEN, KOR, PER, SVN, SYR -- 2920 items in 10 groups against the
  test set's 3442 in 10, constrained-class shares matching to a total mismatch of
  0.021, leaving 83% of the training data behind.

**Consequence.** This is the missing piece under `scripts/epoch_curve.py`, whose
best epoch is currently an ORACLE chosen on the test set and therefore not
reportable. With a group-disjoint val split, "stop at the epoch the constraint
stops helping" becomes a deployable rule measured off held-out data. It costs a
retrain of every arm, so it is a compute-budget decision and is NOT being taken
unilaterally -- see LEDGER PART 5.

Standing caution still applies: do not describe rerunning the inspected test
split as fresh confirmatory evidence.

---

## Infrastructure notes

- `dsisco01` uses older GPUs and fp16; `dsisco02` is Blackwell and bf16. Storage
  is shared NFS, so **check processes on both hosts**.
- Canonical arrays are under `/home/dsi/michaer8/optloss-audit/data`.
- `/home/dsi/michaer8/optloss-reset-validation-20260914` is an OLDER source
  snapshot; its earlier CPU test pass does not validate current source.
  **Re-sync and verify actual bytes before any campaign.**
- Recovery paths and backup verification: [`GIT_TRACKING.md`](GIT_TRACKING.md).
- Data provenance: `docs/archive/paper/data/PROVENANCE.md` -- **archived and
  contaminated with retired datasets.** Read it only for the fmow2 rebuild
  keys, never for results.

## Preservation

No old scientific result is promoted or erased by the reset. Folder titles carry
no evidential meaning. Archive records must state original path, destination,
inventory/hash verification and restore procedure. **Keep this file concise** --
completed audit receipts go in `docs/archive/audits/`, findings go in
`LEDGER.md`, and neither belongs here as a growing narrative.
