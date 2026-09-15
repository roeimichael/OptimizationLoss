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

**The one live thread** is that the damage appears to stop when the constraint
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

## Run state, checked 2026-09-15 13:10

**Nothing is running. All eight GPUs across both hosts are idle**, because every
remaining direction needs a launch decision the user has not made. This is a
deliberate hold, not a stall.

Completed and scored, fmow2 / MobileNetV3 unless noted:

| Campaign | Budget | Runs | Reading |
|---|---|---|---|
| `fm2_mn3` | 30 | 56/56 | Frozen-boundary reference. TraLO trails all six rivals on cc-F1. |
| `fm2_mn2` | 30 | 56/56 | Same, MobileNetV2. |
| `fm2_vit` | 30 | 56/56 | **Failed the gate hardest** (1.6 live epochs). Archived as reference, not read as a verdict. |
| `gx2` | 30 | 56/56 | Adds the augment and focal columns. **Its overlap with `fm2_mn3` is byte-identical -- a re-run, not a replication.** |
| `live11` | 11 | 72/72 | First campaign with per-column nulls. Carries the pre-registered 2x2. |
| `live6b` | 6 | 56/56 | First campaign to PASS `gate:saturation`. Ranking damage stops. |

**Abandoned before it produced evidence:** `sweep1` / `sweep1_mn2`, a within-
campaign budget sweep, stopped at 6 of 384 runs. The firstrun saturation gate
showed the budget grid was badly placed -- four of five points sat deep in the
frozen regime with nothing between 43% and 75% live fraction, which is where the
sign flip is. The `b5`/`b6`/`b7`/`b8`/`b12` machinery that made per-arm budgets
expressible is committed and tested but **is not currently authorised for use**;
the user has not approved the budget-sweep design.

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

## Open user question

**Are there untouched evaluation groups or splits on the current datasets?**
Until this is answered, do not describe rerunning the inspected splits as fresh
confirmatory evidence. Code cleanup and data-integrity checks can proceed.

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
