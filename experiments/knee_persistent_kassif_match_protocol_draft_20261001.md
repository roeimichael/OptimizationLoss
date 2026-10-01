# Draft: sustained TraLO versus Kassif PAO on knee images

**Status, 1 October 2026:** prospective design for review and implementation.
No run is authorized by this file, no new seed has started, and no Chen test
image is to be scored. The 826-image development cohort has been examined many
times; every result from it remains exploratory even with new seeds. Freeze this
design and a new immutable runner before the first quality score is read.

## The question

Does repeatedly applying TraLO's label-free capacity direction *during* the
training of a knee classifier improve the identity of the grade-3 patients who
receive scarce slots, compared with (1) plain prediction plus the same exact
allocator, (2) a displacement-matched random direction, and (3) Kassif and
Singer's cost-sensitive PAO retraining? Earlier positive knee results were
steps on **side copies** of snapshots; the supervised PTO training path was
unchanged. This study asks a different question: whether TraLO can improve the
learned model trajectory. A negative answer is a valid outcome.

The [paper's accessible preview](https://www.sciencedirect.com/science/article/pii/S0952197626022736)
describes iterative cost-sensitive training against a class capacity. Its
numerical tables are not publicly exposed in that preview. The source baseline
is the audited [author repository](https://github.com/YuvalKassif/ConstrainedClassification)
at `413d96c`, interpreted in
[the source audit](claude_yuval_repo_audit_20260927.md). Published percentages
must not be treated as paired numbers on our data and split.

## Common data, models, and capacity

Use the audited Chen OAI v1 knee images with 5,778 train, 826 development and
1,656 sealed test images. Keep both knees of one subject together. Carve a
subject-stable stopping split only from train, as in
[`knee_yuval.py`](../tralo/knee_yuval.py); never use development labels to stop,
select a model, calibrate a step or construct a cap. All arms see the same
decoded images, train/development IDs, 224-pixel RGB transform, Kassif-style
augmentation, balanced training sampler, Adam task optimizer, learning-rate
decay, early-stop rule and initial ImageNet weights. Record exact weight-file,
data, split, source and config hashes. Use FP32 on **dsisco02 only** for the
scientific block. Do not pool precision or hardware regimes.

The three prespecified backbones are Kassif's EfficientNet-B5 (fidelity cell),
MobileNetV3-Large (a strong lightweight modern network), and ViT-B/16 (a
transformer). Include B5 even if its prior TraLO direction was negative;
dropping it would select an apparent winner. ResNet18 and RegNetY remain
historical diagnostics, not substitutes for these three modern cells. Verify
cached pretrained weight hashes and the same five-class head initialization
within each seed.

The constrained class is KL grade 3. The development capacity reference is
`round(826 * 757 / 5778) = 108`, derived only from the **training** prevalence.
The paper's 40%, 50%, 60%, 70%, 80%, 90% pressure levels therefore correspond
here to caps 43, 54, 65, 76, 86, 97. The first inference block trains at the
two prespecified, separated levels **54 and 86**. They were selected to span
different quota pressures, not because of a method's observed score. Show a
six-cap *deployment-only* curve for every saved model, clearly distinguished
from training at those six caps. A later all-six-cap training block would be a
separate prospective study with its own cost decision and new seeds.

Our current standing rule rejects weight decay and label smoothing, so the
new **matched** arms use both at zero. This is an explicit deviation from the
author's original `weight_decay=1e-4`; the older exact-recipe B5 PAO/PTO
results remain contextual, not numerically pooled with this study. All arms
share the same zero-decay recipe. Any later exact-paper-recipe stratum must run
both methods and controls anew, under a separate fixed protocol.

## Five matched arms

One `(backbone, seed)` starts from one hashed initialization. One plain PTO
training is shared across caps rather than trained twice. Its stopping epoch
sets a **fixed task horizon H** for the persistent target, sham and null. Those
three run exactly H supervised epochs, select the lowest carved-stop loss
among those epochs and restore that checkpoint. Their common horizon and batch
order make their task-update dose comparable, and ensure that a target radius
exists for every sham epoch. PTO itself retains Kassif's original
patience-based stopping. PAO retains its own variable retrain stopping. This
task-horizon rule is an explicit comparison adaptation, not an exact copy of
the author's pipeline. For each trained cap:

1. **PTO / Clipper:** supervised cross-entropy only, best checkpoint restored
   using the carved training stopping split. The exact `capped_first` allocator
   supplies the grade-3 quota at deployment.
2. **PAO:** the audited Kassif cost-sensitive loss and outer loop, with its
   training-false-positive weights updated from the *unlabeled* development
   argmax count; retrain from the same initial state and same augmentation and
   sampler draws, up to eight retrains. Record convergence and all retrains.
3. **Persistent TraLO:** after each completed supervised epoch, if the current
   development argmax grade-3 count exceeds the cap, take the existing
   label-free streamed soft-count gradient and the smallest Euclidean step
   along its negative direction that reaches the hard cap. Continue the next
   supervised epoch from that changed model. Keep the task optimizer lifetime
   and its state; log the actual displacement, raw gradient and number of
   line-search evaluations. No development label enters the update.
4. **Persistent sham:** same H supervised epochs and *the target arm's exact
   per-epoch parameter-tensor displacement norms*, applied along seeded random
   directions from the sham model's own state. This tests direction, not the
   benefit of adding an extra model perturbation. The sham cannot compute a
   favorable radius from its own outcome.
5. **TraLO-null:** the same epoch hooks, development inference, stopping rule,
   snapshot schedule and optimizer path as persistent TraLO, but no parameter
   perturbation. It must replay PTO probabilities exactly; if it does not, the
   block is invalid. This is a separate execution only for the integrity pilot;
   the full block may use a byte-verified PTO alias to avoid duplicate training.

PAO's variable number of full retrains is part of its method, not equal compute.
Report GPU-seconds, epochs, forward/backward passes and peak memory for each
arm. A matched-cost PTO restart/ensemble is an additional secondary baseline
if the pilot demonstrates that PAO spends materially more compute. Do not
claim an accuracy gain without the equal-cost context.

The line-search method above is a **persistent calibrated TraLO variant**, not
the old separate-Adam TraLO and not Kassif PAO. No new supervised rank loss is
in the confirmatory arms. The unrun
[hard-pair proposal](knee_hard_pair_protocol_20260924.md) remains a possible
later mechanism study and must not be silently added after seeing these scores.

## Estimand, metrics and fixed reading

For each `(backbone, cap, method)` average over the same 12 independent seeds
**only**. The primary endpoint is deployed grade-3 F1 under the identical
`capped_first` allocator, which spends at most the fixed cap without consulting
labels. Report its mean, seed SD, every paired seed difference and two-sided
95% paired Student-t interval. The two primary contrasts per backbone/cap are
TraLO minus sham (direction attribution) and TraLO minus PTO (deployment
advantage); report TraLO minus PAO as the prespecified rival comparison. Correct
the 3 backbones x 2 caps x 3 contrasts family by Holm before calling a win.
Do not select one favorable cap/backbone or omit the B5 cell.

Secondary metrics on the same deployed predictions: overall accuracy,
five-class macro-F1, prevalence-weighted F1, grade-3 precision and recall,
per-grade F1/support, confusion matrix, and filled/wasted grade-3 slots. Also
report raw argmax counts, raw metrics, upper-bound-correction allocation,
feasibility, and compute. A true success needs an attributable cc-F1 gain over
sham and PTO **without a material deterioration** in accuracy, macro-F1 or
weighted-F1. If only grade-3 F1 rises while common grades fall, call it a
trade. A CI crossing zero is inconclusive; 12 seeds do not prove equivalence.
These are repeated-development-cohort intervals, not a test-set claim.

## Gates and stopping rule

Before any GPU training, verify source identity, real array hashes and
train/stop/development/test subject and pixel separation, no individual
development labels in training, exact quota construction, model weight hashes,
hand-checkable allocator and metric fixtures, analytic/autograd/finite
difference count gradient, chunked/full gradient parity, deterministic
augment/sampler replay, sham radius and per-tensor norm equality, no-op/null
equality, AMP/nonfinite paths even though the first block is FP32, logging
state neutrality and exclusive crash behavior. Mutation tests must fail when
labels or a wrong radius are introduced. Re-audit on both hosts from a new
immutable release.

Pilot: seed **6700**, cap **76**, on B5, MobileNetV3 and ViT; it is outside
the study and its quality score is not read. Start on two genuinely free
dsisco02 GPUs, inspect actual forward/backward progress, owner, physical UUID,
memory, applied/skipped task and constraint updates, raw and post-step counts,
soft counts, gradient norms, parameter displacement, best checkpoint,
outputs and source/config/data/artifact hashes. Only after these checks pass
may a third and fourth GPU take distinct jobs. The full block uses fixed seeds
**6701–6712** on all three backbones at caps 54 and 86, one exclusive job per
GPU, at most four simultaneous. No duplicate, resume or overwrite. A dead PID
or SSH outage requires forensic inspection before any continuation.

The previous [Kassif-recipe result](claude_yuval_pipeline_result_20260927.md)
shows PAO can converge while failing to improve deployed F1, and the
[snapshot study](claude_stepens_additional_result_20260928.md) shows a small
MobileNetV3 gain but B5 harm. Both signs must remain in the record. The pilot
gate depends on integrity and cost, **never** on a favorable development F1.

## Compute decision before dispatch

The old Quadro B5 seed 4100 used 3,211.5 seconds for three PAO retrains plus
PTO/step evaluation; MobileNetV3 seed 4300 used 1,284.5 seconds for its prior
single-training block. These are different hardware and method variants, so
they are only a starting cost scale. A ViT knee time and the per-epoch persistent
line-search overhead are **unknown**. No honest full-block projection exists
until the score-blind pilot is timed on dsisco02. Proposed *pilot ceiling*:
**12 total GPU-hours** across at most four dsisco02 cards. Stop safely at the
ceiling, preserve partial results, then compute the full-block estimate from
measured arm/backbone times and seek a separate explicit full-block ceiling.
Using four cards is concurrency permission, not permission for unbounded time.

The sealed Chen test may be evaluated only after this method, caps, backbone
scope, checkpoint and analysis are locked and a separate held-out decision is
authorized. No development result becomes confirmatory because its seed is new.
