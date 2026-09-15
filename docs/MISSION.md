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

## RUN STATE -- checked 2026-09-15 22:16 IDT

**`rank1_*` is DEAD. All three campaigns finished 24 usable / 16 FAILED.** Every
`rank_clip` and `aug_rank_clip` run died in its first logged epoch on
`ValueError: too many values to unpack (expected 2)` -- see LEDGER PART 1. The
24 control runs per campaign (`clip`, `focal_clip`, `aug_clip`) are valid and
retained; the Stage 1 gate was unanswerable from them.

**`rank2_*` is LIVE on dsisco01 GPUs 1/2/3**, claimed 22:15:39 IDT:
`rank2_MobileNetV3`, `rank2_MobileNetV2`, `rank2_RegNetY400MF`, 40 runs each,
generated and frozen at **bfb33a97** (the fix commit), same design as rank1
(fmow2, L80_G95 + L90_G95, arms `clip` `focal_clip` `rank_clip` `aug_clip`
`aug_rank_clip`, seeds 1-4, pretrained, warm-up 30).

Why a NEW campaign rather than re-dispatching rank1's 16 pending runs: the
campaign freeze refused it, correctly -- `validate_campaign` raised `source
bytes differ from frozen release` because the fix changed `src/` under a frozen
inventory. That guard is doing its job and was not overridden.

**The mixed-`code_version` question was settled by execution, not argument.**
The fix touches a function called INSIDE the warm-up loop, and iterating a
`shuffle=True` loader draws from the global RNG, so a changed number of draws
would have silently altered training. Measured pre-fix vs post-fix on a 2-tuple
loader: identical accuracy, **identical RNG state**, identical parameters,
identical mode. The fix is a true no-op for every historical arm, so rank1's
controls remain byte-comparable with anything produced at bfb33a97.

**The decisive check arrives early.** Rank arms are runs 2 and 5 of every
campaign, not last -- so each campaign's own second run is the smoke test, about
10 minutes in. A separate smoke campaign was started and then stopped as
redundant once the interleaving was confirmed. Watch for the unpack signature;
`scripts/rank_status.sh` now surfaces distinct error strings.

**Cost so far: 2h21m x 3 cards spent on rank1, recovered as control arms only.**

**FIRST RESULTS IN, checked 2026-09-15 22:40 IDT -- two checks discharged:**

1. **The ranking loss is NOT inert.** `aug_rank_clip` seed 1 differs from
   `aug_clip` seed 1 on both backbones that have reached run 2
   (MobileNetV3 `6daea2293b08` vs `8c7a9316fa59`; RegNetY400MF `e560332f9134`
   vs `fa08e8cc461c`). The loss reached the model. This is the failure mode
   five earlier flags died of, and it is now excluded. The formal
   pre-registered version of this check -- `rank_paired` marking `rank_clip`
   `(cap-inert)` -- still runs at scoring.

2. **The metrics.py fix is confirmed a no-op ON REAL DATA, not just in
   miniature.** Every control run present in BOTH rank1 (338110cc) and rank2
   (bfb33a97) reproduces byte-identically: **3 of 3 pairs so far**, and the set
   grows as rank2 advances. This is the corpus-level evidence behind the
   mixed-`code_version` argument; re-check it when rank2 completes.

---

## Stage 1 ranking gate -- PRE-REGISTERED READING (written 2026-09-15, before any rank arm landed)

Three campaigns live on dsisco01 GPUs 1/2/3: `rank1_MobileNetV3`,
`rank1_MobileNetV2`, `rank1_RegNetY400MF`; 40 runs each; arms `clip`,
`focal_clip`, `rank_clip`, `aug_clip`, `aug_rank_clip`. **No new scorer is
needed** -- `scripts/rank_paired.py` already gives per-cell paired gAP with the
seed sd beside it:

```
python3 scripts/rank_paired.py --glob 'results/rank2_*/*/*/*/*/seed_*' --a rank_clip     --b clip
python3 scripts/rank_paired.py --glob 'results/rank2_*/*/*/*/*/seed_*' --a aug_rank_clip --b aug_clip
```

**The mechanism check comes free, and it is read FIRST.** `rank_paired` marks an
arm `(cap-inert)` when its probabilities are byte-identical across L80 and L90.
Both control arms are already marked that way, correctly: no constraint reaches
their model, so the cap acts only in the post-hoc allocator. But `rank_frac` is
read from the cap (`warmup.py:165` -- 0.8 at L80, 0.9 at L90), so a LIVE ranking
loss trains two different models and `rank_clip` **cannot** be cap-inert.

| what `rank_paired` shows for `rank_clip` | what it means | what follows |
|---|---|---|
| `(cap-inert)` | the ranking loss did not reach the model at all | a sixth dead flag; fix the wiring, discard the campaign, do NOT read gAP |
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
