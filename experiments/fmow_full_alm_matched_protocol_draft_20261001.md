# Draft protocol: persistent local TraLO versus inexact PHR-ALM on fmow2

**Status: implementation and independent gate review in progress; no GPU run.**
Draft runner, scorer, queue and fixed configs now exist, but independent review
identified gate and replay checks that must be completed before release. It
does not supersede any frozen campaign. In particular, the completed 12-seed
PHR study at `2f6a68eb` compared *side-copy directions* at PTO snapshots; its
PHR dual advanced, but the model always reset to PTO. It cannot answer whether
ALM training changes subsequent supervised learning. The older knee/CIFAR ALM
study trained frozen ResNet18 heads, with a different data, model and schedule;
its scores are not a matched fmow2 comparator. This draft must be reviewed,
implemented, tested and given a fresh immutable release before its pilot can
run. The prospective compute decision below is limited to this block.

## Question and limits

On the byte-audited fmow2 country split and trainable MobileNetV3-Large, does a
**persistent** label-free pooled-plus-country TraLO correction improve deployed
class-1 ranking over its schedule-matched zero-step control, post-hoc clippers,
and a persistent inexact Powell-Hestenes-Rockafellar (PHR) ALM correction? The
PHR arm is an inexact primal method, not an exact classical ALM solve or a
replication of a particular published ALM training recipe. Compare the two
methods under a shared correction schedule and maximum parameter-space dose;
their directions, activation and PHR dual rule are the interventions. Do not
interpret a TraLO-over-ALM result as a gain unless TraLO also beats its null
and post-hoc controls.

The five 1,673-image development countries have been repeatedly scored.
Fresh seeds on them are **exploratory**, not geographic confirmation. The five
reserved countries and Chen test stay unscored. No parameter, cap, epoch,
backbone or method may be chosen from this development block's results. This
study tests one fixed policy, not a radius/rho/grid search.

## Frozen data, caps, model and deployment

Use the existing six SHA-256-pinned fmow2 files, country partition and sample
IDs from `tralo.fmow_yuval.load(..., include_pool_labels=False)`: 15,841
supervised training images, 1,829 country-disjoint stop images, 1,673 unlabeled
development images, and 1,769 reserved-country images. The 2026-09-30
read-only data audit found no cross-role exact pixel duplicates, but eight
extra duplicate training rows and no scene-identity proof; retain and disclose
these limits. Recheck array hashes, shape, train-label alignment, unique IDs,
country separation and duplicate hashes on the new release. The runner must
neither load nor retain development or reserved labels. Only an independent
offline scorer may read development labels after all label-blind gates pass.

Keep the current ImageNet-V2 MobileNetV3-Large eight-class initialization,
full-frame RGB 224 preprocessing, training augmentation, class-balanced
sampler, FP32/TF32-off arithmetic, batch size 32, task Adam `lr=1e-4` and the
existing epoch LR decay. Pin the exact checkpoint bytes and torchvision/
PyTorch versions in the release receipt. Fix **weight_decay=0** and no label
smoothing for every new arm, honoring the user's explicit rejection recorded
in main `RULESET.md`. This is a prospective matched recipe amendment to the
inherited Yuval setting of 1e-4 weight decay. It means the new null/PTO must
be trained afresh and cannot claim byte parity or numerical comparability
with older weight-decayed PTO campaigns. Do not decide this setting from
development scores.

Use exactly two pooled class-1 caps on the 1,673-image pool: `G=167` and
`G=83`. For each, country upper bounds are Hamilton size shares of
`ceil(5G/4)` (totals 209 and 104), computed solely from unlabeled country
sizes. The local bounds sum above the pooled cap, so country bounds are upper
bounds, not a requirement to fill each one. Save the entire quota derivation.
Every arm deploys through the same label-free `local_capped_first` allocator,
with deterministic ID ties. Separately report raw predictions and feasibility.
This PTO plus local allocator is a **local Clipper analogue**, not a result for
the historical `clip` or `focal_clip` training code.

## Matched training algorithm (proposed fixed design)

**Proposed dated amendment, 2026-10-01, to main `FRAMEWORK.md`'s 30-epoch
default:** train a new, complete **seven-epoch** block for every arm; do not
recycle PTO snapshots from older campaigns. Thirty was a comparability
convention, not a validated optimum for this Yuval MobileNet recipe; recent
fixed PTO trajectories stopped after six to seven epochs and the long
constraint phase was previously documented as saturated. Seven fixes the
same task-update budget for every new branch without choosing an epoch from
the development metric. The PI's 30 September instruction delegated ongoing
experiment design and asked us to continue without waiting for another reply.
This document fixes the seven-epoch matched amendment prospectively for this
distinct study; it cannot change after a pilot or development score. It is a
separate study, not a silent extension of old
30-epoch results. The same amendment narrows the rival panel to local TraLO,
inexact PHR-ALM, CE/focal clippers and schedule-matched nulls. The full
`FRAMEWORK.md` panel additionally includes Fioretto and Hounie; this narrow
study cannot be described as the final all-rivals thesis comparison.

For CE-based arms, epoch 1 is a common supervised CE warm-up. At its end,
clone the complete model *and task optimizer state* into the CE/null,
cap-specific TraLO and cap-specific PHR branches. The **focal Clipper branch
starts instead from the same seed's initial model before epoch 1**, with its
own optimizer, and uses the specified focal loss from its *first* update; it
must never inherit the CE warm-up. All branches run seven task epochs with
the same prerecorded seed-local sample orders and augmentation RNG streams,
the same optimizer type/nominal learning-rate schedule and training image
recipe. Focal's loss is the declared exception; any gate-derived per-batch
learning rate is computed from that arm's own logits under a common audited
rule, never copied from the CE arm. Do not let evaluating one arm change
another's RNG, BatchNorm buffers, optimizer moments or checkpoint state.

At the **end** of each epoch 5, 6 and 7, save the model checkpoint and its
pool probabilities **after** that epoch's accepted or skipped constraint
correction and, for PHR, after the dual-state update. Hash each checkpoint,
probability artifact and dual record. The fixed deployed output is an
equal-weight ensemble of those three post-correction probability arrays for
every branch. For CE/null/focal branches the same point is after the explicit
no-op correction slot. Neither stop-country nor development labels select an
epoch or checkpoint. Stop-country loss is logged for diagnosis only. A full
zero-step branch must replay the independently run CE-only PTO branch
byte-for-byte.

After each supervised epoch 2-7, each constrained branch evaluates the same
unlabeled development images and takes at most one full-model correction on
its **own persistent model**. The next supervised epoch starts from those
corrected weights; no branch resets to a PTO snapshot. The PHR dual is separate
for each seed/cap and carries across all six correction opportunities. All
branches use the same maximum L2 radius 0.1 and the already defined label-free
boundary acceptance rule; if it rejects, apply zero and log why. This dose
policy makes the comparison controlled, while actual realized radii/skip rates
remain outcomes. It is a specified inexact ALM implementation, not an attempt
to claim that its penalty weight was optimized.

For pooled scope and each development country `s`, let
`g_s(theta) = (sum_{i in s} p_theta(i,class 1) - K_s)/max(K_s,1)` with signed
soft residuals. The TraLO branch uses the existing hard-active normalized
pooled-plus-country soft-count direction and its boundary rule. The PHR branch
uses `A = sum_s (max(0,lambda_s + rho*g_s)^2 - lambda_s^2)/(2*rho)`, with
`rho=0.5`, `lambda_s(0)=0`, and a step in `-grad_theta A` under the **same**
boundary acceptance rule. Recompute `g_s` at the PHR model *after* its accepted
or skipped correction, then set
`lambda_s <- max(0,lambda_s + rho*g_s)` before the next epoch. A skipped
parameter step does not skip this dual update. The gradient and dual access
development images/country IDs/caps, never development labels. Report the
penalty and per-scope signed residual, multiplier, derivative, activation,
probe and applied displacement before and after each opportunity.

Per seed and cap, save persistent TraLO and persistent PHR branches. Run a
schedule-matched null with all constraint observations/copying but zero
parameter correction; a second method-named null is required in the pilot and
must be byte-identical to the first, including task batches/optimizer and all
predictions. Once proved identical, the full block stores one shared CE/null/
Clipper trajectory per seed; its identical names never count as independent
evidence or separate method contrasts. A separate `focal_clip` branch uses
the main protocol's declared focal `alpha=0.25, gamma=2` **from epoch 1**,
with the same seven-epoch budget, augmentation, sample order and allocator.
Validate its loss, its independent initial trajectory and recipe parity before
the pilot. It is a different supervised loss and is never called a null.
No historical `clip` or `focal_clip` score is substituted for these branches.

The primary trained-arm comparison keeps all other factors common. Native
TraLO and PHR convergence claims are out of scope; an ALM arm that is active
for only some epochs is still reported as such. Also report how frequently
the controller accepted a nonzero step and whether raw violations remained.

## Integrity gates before dispatch or scoring

1. **Math and software:** independently hand-check pooled/country residuals,
   PHR value and dual projection, then compare autograd and central
   finite-difference gradients on active, slack and conflicting-scope cases.
   Require chunked/full gradient parity, finite/zero-gradient behavior,
   deterministic fixed-weight probability replay, maximum/radius dose,
   exact accepted-probe replay, model-buffer/RNG neutrality and a mutated
   calculation that makes the relevant test fail. Cover both the runner and
   independent scorer with real-path fixtures, not names or source-text tests.
2. **Training parity:** identical initial model hashes in all arms and
   identical epoch-1 warm-up model/optimizer hashes in CE-based arms;
   focal's epoch-1 model/optimizer must instead follow its own focal loss.
   Require per-epoch sample-order and first-batch image hashes, equal task
   update counts, and byte-exact CE-only/TraLO-null/ALM-null PTO predictions
   in the pilot. The focal branch must differ from the CE warm-up in its
   first-epoch loss/gradient and maintain its own optimizer trajectory.
   Independently replay and hash each epoch-5-to-7 **post-correction** model,
   probability array and PHR dual state. Log all attempted/applied/skipped
   constraint updates and task/constraint gradient norms, raw versus actual
   parameter displacement,
   per-scope residual/dual, training and stop losses, timing, GPU/precision,
   nonfinite failures and checkpoint identities. Null instrumentation may
   perform inference but may not mutate model, RNG or task optimizer.
3. **Data and regime:** verify all file/checkpoint/source/config hashes and
   image-label alignment, country roles, exact cross-role duplicates and
   the independent quota recount; inspect learnable mistakes at actual
   pooled/country allocation cuts for this backbone without choosing a more
   favorable country or cap. Confirm development labels are unavailable to
   the runner and no reserved-country predictions are produced.
4. **Release and pilot:** commit/push tested source, protocol/config/scorer to
   GitHub and DSI, deploy a *new* immutable detached release, require both-host
   tracked-byte/native test/CLI parity and exclusive arithmetic smoke on a
   genuinely free card. Fix fresh pilot seed **6700** and full seeds
   **6701–6712**, each with separate, non-reused roots. Run seed 6700 as an
   exclusive step-on pilot; a distinct same-host zero-step reference using
   seed 6700 proves PTO identity. The label-blind gate
   validates all above plus artifact hashes, applied doses, dual continuity,
   quota/allocator feasibility and projected full-study cost. A failed gate
   stops expansion and keeps its artifacts. Do not inspect pilot development
   scores to choose settings or decide whether the fixed full block runs.
5. **Full block:** only after the pilot passes, execute twelve *fresh* paired
   seeds in exclusive roots, with at most the authorized concurrent GPU count
   and no overlap with any earlier seed family. Never retry an ambiguous or
   failed root without a forensic inventory and a separately documented
   decision. Independently score only the complete block after all identity
   and label-blind gates pass.

## Outcomes and uncertainty fixed before labels are opened

Primary endpoint is allocated class-1 `cc-F1` (one constrained class here).
The seed is the sole replication unit; caps, epochs, country cells and shared
warm-up copies are not independent replicates. Report both caps, every seed,
mean, seed SD, paired deltas and two-sided 95% Student-t intervals conditional
on this fixed, repeatedly viewed development pool. Predeclare a Holm family
of **eight unique contrasts**: at each cap, TraLO minus the shared CE/null/
Clipper, PHR minus that shared control, TraLO minus PHR, and TraLO minus
focal Clipper. The CE Clipper and the schedule-matched null must be proven
exactly identical; testing both as separate contrasts would duplicate one
test. Report all other pairwise values as descriptive. A TraLO lead over this
matched panel requires at least +0.01 absolute cc-F1 in mean paired benefit,
with positive Holm-adjusted two-sided evidence versus the
shared CE/null/Clipper, focal Clipper and PHR-ALM at a cap, with no wholly
negative accuracy, macro-F1 or weighted-F1 interval versus those controls.
Beating only one rival does not qualify; an unresolved TraLO-PHR contrast does
not establish superiority over ALM. Report constrained precision and
recall/support, uncapped-class F1, exact cap violations, true/correct slot
entries/exits, raw call counts, snapshot/branch activation and measured
GPU-hours. No favorable cap is selected post hoc. Neither a negative nor a
positive development result establishes geographic generalization.

## Compute projection and delegated budget decision

The closest measured complete MobileNetV3 pilot is the boundary snapshot
study: on dsisco01, six PTO epochs plus five side arms at both caps cost
`1176.81 s` for step-on and `409.56 s` for the CE-only reference. The prior
PHR snapshot pilot on dsisco02 cost `568.21 s` step-on and `184.01 s`
reference. These host-specific numbers are **not** a benchmark of persistent
training. They bound neither the extra optimizer trajectories nor the PHR
full-pool gradient/line-search cost. The six **unique** trajectories per full
seed are one shared CE/null/Clipper, one focal trained independently from
epoch 1, two cap-specific persistent TraLO models and two cap-specific
persistent PHR models. As a rough CE-only scale check, six independent
trajectories at dsisco01's measured 409.56-second reference rate would be
`6 * 409.56 / 3600 = 0.683 GPU-h` per seed, or 8.19 GPU-h for twelve seeds.
Shared CE warm-up can save part of that, while the seventh epoch, focal
trajectory and four branches' full-pool gradient/line-search passes add work.
Neither number is a bound. The independent scorer replays real-image
checkpoint predictions on an exclusive GPU, and its measured card time must
also enter the pilot projection and final study cost.

**Separate ceiling: 24 GPU-hours**, fixed as a self-limiting decision under
the PI's 30 September instruction to keep research moving and make experiment
decisions without waiting. This is not a claim that the run fits.
After a complete same-host pilot, measure each of the six unique trajectories:
`T_seed = T_CE_epoch1 + T_CE_epochs2to7 + T_focal_epochs1to7
+ T_TraLO167_epochs2to7 + T_TraLO83_epochs2to7
+ T_PHR167_epochs2to7 + T_PHR83_epochs2to7`. Thus the common CE warm-up
is counted once, while focal is charged from its first update and the four
treated trajectories include their full constraint gradient/probe streams.
Separately measure the
pilot's duplicate method-named null, independent CE reference and arithmetic/
memory smokes (`T_extra`). Project
`T_spent_so_far + 12 * T_seed`, where `T_spent_so_far` includes the complete
pilot, `T_extra`, and any failed/aborted attempts under this new protocol.
Use summed GPU card-hours if arms run concurrently, not wall-clock hours. Stop
before full dispatch if this exceeds 24; do not drop arms/seeds or shorten
training to make it fit. The existing eight-hour boundary snapshot ceiling
and 24-hour ViT ceiling do **not** transfer to this new experiment. The user
authorized research comparing TraLO with ALM, Clipper and nulls, including
autonomous decisions while unavailable. This prospective amendment records
the scientific recipe and a finite ceiling under that delegation; it does
not bypass validation. No GPU dispatch may occur until the implementation,
source/data/preprocessing/gradient gates and immutable release pass. If the
measured pilot projects beyond 24 GPU-hours, stop rather than shorten or
alter the design; a larger ceiling needs a new decision.

## Sources inspected

Worktree `tralo/alm.py`, `tralo/local_alm.py`, `tralo/fmow_local.py`,
`tralo/knee_yuval.py`, `tralo/fmow_yuval.py`,
`experiments/fmow_local_alm_direction_result_20260930.md`,
`experiments/alm_two_dataset_20260924.md`,
`experiments/fmow_boundary_mnv3_protocol_20260930.md`,
`experiments/fmow_local_alm_data_preflight_20260930.md`, and the two pilot
gate receipts under `C:/Users/roeym/.codex/rebuild-audit-20260922/`.
The main checkout's `docs/FRAMEWORK.md`, `RULESET.md`, `docs/MISSION.md` and
`docs/LEDGER.md` set the comparison, evidence and approval boundaries.
