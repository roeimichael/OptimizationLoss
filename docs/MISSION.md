# TraLO reset: execution state

Updated 2026-09-14 (Asia/Jerusalem). User approved recoverable large cleanup, fresh evidence,
validated logging/data/code, then a within-TraLO modification and monitored SSH
experiments. No historical acceptance tally is carried forward.

**Latest user direction (2026-09-14): take charge and drive it.** Manage the
classes and the data, get them to fit the requirements, then run on the GPU
servers; if results look decent, improve them by (1) hyperparameters, (2) extra
seeds for replication, (3) other modalities -- and validate that what comes back
is true. The acceptance bar, chosen by the user: **leading group on cc-F1 and
not dominated elsewhere**, judged on the full metric profile rather than a single
hard-capped condition.

**The dataset is `fmow2`.** iwildcam is RETIRED (2 of 8 candidate conditions;
see the retirement note in `configs/protocol.yml`) and the original `fmow`
oodslice is withdrawn (basename join collapsed distinct AOIs). Rebuilt fmow2
passes all 8 conditions of `scripts/candidate_gate.py`.

## Current stage

### 🔬 PRE-REGISTERED 2026-09-14, BEFORE THE SEEDS LANDED

**CLAIM: TraLO's ranking damage is caused by UNDERSHOOT, not by the constraint.**

TraLO decides a scope is violating from a SINGLE epoch's count. That count has a
measured epoch-to-epoch sd of **27-113 items** and oscillates just as much with
the constraint switched off -- `tralo_null` takes zero constraint steps and its
class-2 count still swings 468 -> 869. So the ratchet fires on noise:
`Local_Satisfied` was 0 in EVERY epoch of every run inspected, multipliers
climbed monotonically for all 29, and class 7 ended at **111-122 predictions
against a permitted 304 on BOTH backbones** -- a class whose global excess in
the null was +8 (MNv3, 0.17 sd) and -83 (MNv2, already compliant).

Evidence at pre-registration: 12 paired `tralo` vs `tralo_null` contrasts over
2 backbones x 2 caps x 3 classes.

| group | mean d gAP | n |
|---|---|---|
| ended UNDER budget | **-0.0273** | 5 |
| ended AT/OVER budget | **+0.0001** | 7 |

`pearson(excess/sd, d gAP) = +0.524`. The two largest GAINS in the table
(+0.0482, +0.0247) are both cases where TraLO corrected and stopped.

**PREDICTION.** At 4 seeds across `fm2_mn3`, `fm2_mn2` and `fm2_vit`, the split
holds: contrasts ending under budget stay negative, contrasts ending at/over
budget stay >= 0 within seed noise.

⛔ **FALSIFIED IF** the two groups are equally negative at 4 seeds, or if
`pearson` falls to ~0. Then the damage is not overshoot and this account is
wrong -- record it in the rejected ledger rather than rescuing it.

⚠️ At pre-registration n=12 and the contrasts are NOT independent: L80 and L90
share a warm-up and some share seeds. r=+0.524 at n=12 is suggestive, not
conclusive. What makes it worth testing is the mechanism, not the p-value:
undershoot forces the allocator to backfill to K from lower-ranked items, which
is mechanically guaranteed to cost quality.

🔑 **THE FIX THIS IMPLIES IS NOT IN THE REJECTED LEDGER.** Everything closed
there is the gradient EXPRESSION -- penalty shape, count function, cut window,
margin, scope re-weighting. This is the MEASUREMENT that triggers it: require a
scope's excess to clear its own measured epoch noise before ratcheting, or
average the count over recent epochs before declaring a violation. No gradient
changes and no extra compute. Test as a `tralo_hyst` arm against unmodified
`tralo` at equal compute, pre-registered above.

⚠️ With `L80`/`L90` local against `G95` global, `sum(local K) < global K` for
every capped class, so the GLOBAL cap is inert and these campaigns are a pure
LOCAL-cap experiment. Do not read them as evidence about the global scope.

**Core and tool thinning committed and independently reviewed. Fresh identity,
common deployment, metrics and logging gates remain.**
Current source checkpoint `de760d40`: seven public arms; 52 historical scripts and the
task-window config machinery retired. All 17 formerly populated server result
trees remain in the external history archive, not the active results roots.

**RUN STATE, CHECKED 2026-09-14 19:42 Asia/Jerusalem.** Three campaigns live on
dsisco02, tree pinned at `1072a229`, at the 3-GPU ceiling and zero failures:

| campaign | backbone | GPU | done / planned | launcher PID |
|---|---|---|---|---|
| `fm2_mn3` | MobileNetV3 | 1 | 27 / 56 | 3290265 |
| `fm2_mn2` | MobileNetV2 | 0 | 15 / 56 | 3320254 |
| `fm2_vit` | ViTB16 | 2 | 7 / 56 | 3306622 |

`sat_fmow2` completed 16/16 earlier as the clipper saturation baseline. GPU 3 is
free and deliberately unused -- the standing ceiling is three GPUs total, not
three per host. This is a snapshot, not a reservation; recheck BOTH hosts before
any dispatch. **The source tree is FROZEN while these run** -- `src/`, `configs/`,
`scripts/` and `main.py` are all inside `source_inventory()`, so a deploy of any
of them splits the campaign identity.

## Ordered work

- [x] Preserve the pre-reset dirty diff and exact instruction files under `.codex/`.
- [x] Move the four large operational narratives into the local history archive;
  replace them with a short current protocol and state file.
- [x] Preserve historical artifacts outside Git tracking with verified backups;
  see `docs/GIT_TRACKING.md`. Archival copies are not fresh-clone dependencies.
- [x] Archive old local and remote result trees with manifests; keep data arrays,
  checkpoints, predictions, parent-extension links, and recovery paths intact.
- [x] Complete independent review of the committed operational-tool/test thinning;
  retained mathematical, AMP, caps, data, recovery and real-CLI checks stay in Git.
- [x] Reduce the public comparison to `tralo`, `tralo_null`, `clip`, `focal_clip`,
  `fioretto`, `hounie`, `alm`; remove retired variant runtime branches.
- [x] Generate the reference with `constraint_fp32: true` and
  `constraint_grad_mode: normalize`; trained 1+29/posthoc 30+0 task epochs.
- [ ] Remove the remaining unused weighted-CE option and protocol metadata;
  remove orphan package dependencies without changing the installed environment.
- [ ] Require a fresh campaign identity and explicit source/config/data/quota
  inventory for reporting; test rejection of archived, mixed and unmarked runs.
  Use a new `OPTLOSS_MODEL_CACHE` namespace so fresh runs cannot reuse historical
  warm-up checkpoints; permit sharing only inside the newly frozen release.
- [x] Replace the stale `keepworking` skill and forward-test the new reference.
- [x] Fix and regression-test AMP step/event accounting. The separately repaired
  optional uniform estimator was subsequently retired with the noncore variants.
- [ ] Use the same greedy deployment allocator and the same saved probabilities
  for every arm. Clippers currently allocate with 256-item inference, while eval
  saves a separate 512-item pass; trained arms also use a different allocator.
  Correct this explicitly as a deployment-protocol change, not a TraLO loss gain.
- [ ] Complete cc-F1-first, fixed-class metric reporting with paired native-unit
  uncertainty. Missing declared classes must count as zero, not disappear.
- [ ] Integrate shared structured logs: rival CSV initialization currently erases
  warm-up history; warm-up/rival task-step application and rival displacement/
  local-scope state are missing. Preserve model state/RNG while fixing producers
  and make the first-run gates consume the records. Missing evidence is unknown.
- [ ] Enforce exclusive canonical campaign ownership and safe crash recovery;
  fix queue failure propagation and per-runner orphan detection. The read-only
  audit found duplicate-root admission and a stale-running recovery mismatch.
  Initial two-GPU execution uses two manifest-disjoint complete campaign roots,
  one dispatcher per root, one queue per card; no multi-GPU scheduler rewrite.
- [ ] Audit current datasets and development cut saturation without selecting on
  a TraLO win. Resolve untouched holdout availability with the user.
- [ ] Commit the validated reference release and verify SHA-256 parity on the
  target host. Leave exact generate/freeze/verify/launch/inspect/report commands
  usable from any terminal or Claude Code, without Codex-only dispatch logic.
- [ ] Launch first-run pilots on two GPUs after gates pass, attach monitoring,
  inspect logs, then expand only if healthy (maximum three GPUs).

Implementation order: reviewed core/tool thinning; fresh identity/common deployment
and reporting; logging integration; dispatch/recovery repair; whole-change
verification; target-host/data
validation; then monitored reference experiments. New loss changes are deferred.
The reference loss, dual update ordering and training behavior stay unchanged
during structural cleanup. The named shared-allocator correction is separate.
Unknown or removed config keys must fail clearly, not be silently ignored.
Archive historical tests/probes through the existing recovery process; retain
compact tests for gradients, caps, metrics, data splits, logging and recovery.

The first GPU experiment tests pipeline/log validity and dataset headroom, not
superiority. Use an audited development split, one backbone and one host before
expanding. Inspect per-group allocation-cut errors, soft/hard residuals, dual
trajectories, actual applied updates and parameter displacement alongside cc-F1.
After those checks, compare the seven core methods at two distinct cap levels
with at least four seeds, paired native-metric uncertainty, equal task-epoch
budgets and recorded extra constraint compute. A later candidate requires a
separate reviewed change; no redesign is needed to run this reference comparison.

## Validation and release state

Core checkpoint `8e684211`: 77 deterministic CPU model/probability/deployment/RNG
arrays match the pre-thinning reference exactly. Independent review cleared its
dataset-scope and fixture-label fixes. Tool checkpoint `2f33fce7`: **335 passed,
1 skipped, no warnings** in 99.95 seconds. Its five review findings were fixed in
`de760d40`: **273 passed, 1 skipped, no warnings** in affected integration, then
all five independently cleared. This was not another full-suite run; the
real-log skip is not a pass. Receipts/review ledger are in the ignored
`.superpowers/sdd/lean-cleanup-plan/`; source recovery is in Git and the verified
external archive. These are software checks, not GPU or superiority evidence.

**fMoW: REPAIRED AND IN USE AS `fmow2` (2026-09-14).** The defect below was real
and is fixed. `prep_fmow` joined metadata to images on `os.path.basename`, but
the archive is laid out `split/class/class_seq/aoi/file` and the filename encodes
only `<class>_<class_seq>_<idx>` -- the AOI is not in it. Measured on
`val-metadata.tar.gz`: 63,422 records, 53,041 unique basenames, **7,429 basenames
under more than one AOI**, so 16.4% of records were silently dropped or
mis-joined. That is the mechanism behind the 436/146 rows: the `false_detection`
metadata row was dropped by `DROP`, but its IMAGE was still popped by a
surviving row sharing the basename. Both sides now key on `class_seq/aoi/file`
and `load()` refuses a non-unique key. The old arrays are preserved; the rebuild
is a separately versioned slice.

`fmow2` re-audited on the rebuilt arrays: 17,670 train / 3,442 test with row
counts consistent across images, labels and meta; **139 train countries vs 10
test countries, zero overlap**; **zero cross-split exact image duplicates**. It
passes **8 of 8** conditions in `scripts.candidate_gate` (density 0.82, 6% dead
items, 6/26 zero ceilings, class balance 0.57). Capped classes are drawn from
this slice's own labels: **1 crop_field, 2 place_of_worship, 7
ground_transportation_station** -- present in 10/10, 9/10 and 8/10 groups. Not
the 3 and 5 the old config declared; class 3 lives in 6 of 10 groups with 4 zero
ceilings.

Measured hardness on `fmow2`/MobileNetV3: train CE saturates by epoch 6 (99.7%
train accuracy), but **test accuracy is 0.634-0.648 over 4 seeds** and a cell
carries **187 errors inside K** against iwildcam's 11.7-21.2 item prize.
⚠️ On ~11 of 30 ceilings p@K >= 0.99 -- the model is confidently wrong, and the
penalty's `p(1-p)` gradient is near zero exactly there. That is a calibration
limit, not a data limit, and it is the open question on this slice.

**BCN is not launch-ready:** two exact duplicate pairs cross train/test with
conflicting class labels and different official lesion IDs. Public source JPEG
and annotation checks now confirm the conflict is upstream, not introduced by
our resize/export for these pairs. No images or labels were changed. A versioned
curation policy and renewed whole-split audit remain necessary; see
`docs/audits/2026-09-14-reset.md`. `fmow2` passes the exact cross-split image
check with zero duplicates and its crop/label failure is repaired above, so BCN
is the only runnable slice still blocked on integrity. Near-duplicates and
unused-holdout status remain open. BCN is otherwise the best-structured slice
available (candidate_gate 7/8, failing only class balance at 0.04), so repairing
it is worth doing rather than abandoning.

**iwildcam is RETIRED (2026-09-14).** It passes **2 of 8** conditions: 2 of its 8
classes can carry a local cap, half the per-group ceilings are K=0 before
training starts, and **72% of test items sit in groups holding NEITHER capped
class** -- those groups cannot produce one allocation decision. Its per-group
label shift is the best in the corpus (TV 0.737) and that is the SAME fact as
its density of 0.27: the shift IS the sparsity. Removed from `protocol.yml`,
from `data_loader.IMAGERY_DATASETS` and from every test fixture; data and the 41
completed runs are archived, not deleted.
⛔ **Every earlier TraLO number was measured through that**, so treat pre-fmow2
results as describing iwildcam rather than the method.
🔑 The two tools that could have caught it disagree by construction --
`dataset_screen` rewards shift, `tier_viability` rewards density -- and nothing
combined them, so whichever was run said "it passes". `scripts.candidate_gate`
now screens all eight conditions at once and its exit code is the verdict.

The server validation checkout `/home/dsi/michaer8/optloss-reset-validation-20260914`
is an OLDER source snapshot; its earlier CPU test pass does not validate the lean
source. No new cleanup commits have been pushed or synced. The old app display
+117,078/-33,372 was the committed `origin/main...62581d90` comparison, not
uncommitted dirt. **Re-sync and verify actual bytes before any campaign.**
iwildcam/fMoW arrays are linked there; BCN is deliberately not linked. Canonical
arrays are under `/home/dsi/michaer8/optloss-audit/data`.
dsisco01 uses older GPUs/fp16; dsisco02 Blackwell/bf16. Storage is shared NFS.
Server static-analysis dependency is isolated at
`/home/dsi/michaer8/optloss-reset-validation-deps-20260914` (`pyflakes==3.4.0`);
the shared training environment was not upgraded.

## Open user question

Are there untouched evaluation groups/splits on the three current datasets?
Until answered, do not describe rerunning the inspected splits as fresh
confirmatory evidence. Code cleanup and data-integrity checks can proceed.

The user's separate loss-research task owns `docs/research/RESEARCH_LEDGER.md`.
Its shortlist is a proposal, not an approved algorithm change or launch protocol.

## Preservation

No old scientific result is promoted or erased by the reset. Folder titles have
no evidential meaning. Archive records must say original path, destination,
inventory/hash verification, and restore procedure. Keep this state concise;
put completed audit receipts in `docs/audits/`, not a growing resume narrative.
