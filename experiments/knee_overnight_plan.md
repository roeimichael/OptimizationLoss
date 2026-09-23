# 2026-09-23 daytime amendment (supersedes overnight blocker)

User explicitly authorizes a documented similar medical/knee dataset if Yuval's exact release remains unknown. No further fallback approval is needed. Deadline: 17:00 Asia/Jerusalem today. Existing fixed stages below remain the design; dataset identity and actual split audit must pass before training. The original Pingjun Chen OAI archive is downloading to /home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1; download_launch.json records process and URL. This is an independent mimic, not a verified replication of Kassif/Singer. Monitor knee-tralo-overnight-research is reactivated through 17:00. Preserve historical overnight notes below as a record, not a current blocker.

# Knee TraLO overnight: 22--23 September 2026

## Current status and authorization

User authorizes overnight incremental global-only knee imagery research, one
backbone, several matched seeds, TraLO versus Clipper, source parity and a
morning summary. DermaMNIST is excluded. No requirement to manufacture a win.
Heartbeat `knee-tralo-overnight-research` runs every20minutes, until08:00
Asia/Jerusalem 2026-09-23, then delivers a summary and pauses itself.

**Data gate unresolved:** supplied paper HTM has no full experimental section.
Exact Kassif/Singer knee release/backbone cannot yet be verified. Named local
and DSI data stores contain no knee data. A question asks whether an explicitly
independent public-knee pilot is acceptable if exact source cannot be found.
Do not treat the unanswered question or elapsed time as approval to substitute.
Until resolved, prepare and validate methods; do not launch a claimed knee run.

Documented candidate only: Pingjun Chen's original
[Mendeley v1 dataset](https://data.mendeley.com/datasets/56rmx5bjcr/1), DOI
10.17632/56rmx5bjcr.1, CC BY4.0, linked from the
[author repository](https://github.com/PingjunChen/GradingKneeOA). It is not
verified as Kassif/Singer's release. The
[source paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9531250/) describes splitting
whole bilateral radiographs. Actual subject IDs, split boundaries and image
duplicates still require inspection. Do not relabel an arbitrary Kaggle mirror
as the original or mix the commonly circulated image counts.

## Bounded design once the data gate passes

One backbone: ImageNet ResNet18, initially frozen features + a five-class linear
head. This isolates the loss and allocator while retaining the present simple
pipeline. It is an independent exploratory design unless verified against the
target paper. No backbone search or silent claim of paper replication.
If the user requires the exact paper backbone, resolve it before dispatch.

Use the documented training and validation partition; do not optimize on test
labels or repeatedly inspect test scores. The unlabeled validation pool is the
transductive constraint population; validation labels go to metrics only.
This is a development experiment, not untouched-test evidence.

Starting budget:20head epochs, warm-up5, batch256, Adam lr0.001, FP32 on one host.
Four seeds901--904, fixed beforehand. All arms get the same initial weights,
warm-up and supervised batches. No early stopping or checkpoint choice on labels.
Choose caps before scoring: class3 <= floor(0.10*N), class4 <= floor(0.02*N),
other grades uncapped. These are explicitly synthetic operational upper bounds,
not medical recommendations or paper-matched budgets. Inspect whether they bind;
if not, report that regime instead of silently tightening to force a result.

Stages (maximum3 recipes x3 arms x4 seeds =36short head runs):

1. Base: Clipper, phase-matched no-constraint control, TraLO. Separate persistent
   Adam for constraint steps; initial lambda0.01, lambda increment0.05, rho fixed
   at0.5. This deliberately avoids importing the failed five-epoch rho100 ramp.
   This combined practical recipe is not a causal ablation of its two changes.
2. Add a training-label margin term with weight0.1 and logit margin1 to TraLO
   and its matched null only; retain the same ordinary Clipper baseline.
3. Replace that auxiliary term with false-positive suppression weight0.1,
   again in TraLO and its matched null. Do not combine the two or tune weights.

Each stage: one-seed execution/gradient/logging audit first, then remaining seeds
only if numerically and procedurally valid. Poor scores are valid negative
results and are retained. Repeated Clipper trajectories are controls, not extra
independent evidence. All three stages remain visible. No extra architectures,
hyperparameter sweeps, data switching or new families merely to fill the night.

## Math and intended interpretation

For training example i, true label y_i, logits z_i, CE remains the supervised
objective. Optional margin:

`L_margin = mean_i max(0, 1 + max_{c != y_i} z_ic - z_i,y_i)`.

It increases a logit gap; it is not normalized geometric distance to a boundary.
It covers all training classes, not only the constrained ones.
This familiar supervised idea has no novelty claim. It might improve ranking,
or it might hurt calibration; the null with the same term isolates that effect.

For originally constrained classes C, false-positive auxiliary:

`L_FP = mean_{(i,c): c in C, y_i != c} [-log(1 - softmax(z_i)_c)]`.

This acts on all eligible negative training pairs, not only hard false positives.
Its normalization averages pairs; only the constrained-class mask is used, not
the numeric cap. Numeric capacities remain in the separate count penalty.
Use a difference of log-normalizers for numerical stability. It may reduce
recall; report that trade rather than merely celebrating fewer predictions.

Supervised minibatch update: `L_task = CE + 0.1 * L_aux`, only after warm-up
for TraLO and its matched null. Clipper keeps ordinary CE. Separate subsequent
constraint update uses the existing bounded soft-count penalty on unlabeled
validation features. No evaluation-label argument exists in `train_arm`.

Independent optimizer moments address the specific documented contamination
mechanism; they do not guarantee convergence or better generalization. Tests
verify task moments unchanged by constraint steps and exact null parity when
the constraint is disabled. Preserve old shared mode for historical replay.

## Gates and reporting

- Data: exact source/release/licence; checksums; label mapping; image decoding;
  subject/radiograph-aware disjointness; byte and available near-duplicate checks;
  actual train/validation counts. Do not call unknown patient independence verified.
- Software: tests, analytic/autograd checks, zero-coefficient parity, meaningful
  nonzero intervention, fixed warm-up/batch identities, finite outputs and logs.
- Deploy: testing-on-server skill, commit/push local/GitHub/DSI, immutable release,
  source hashes, native tests. Check both hosts and actual GPU process owners;
  prefer free Blackwell only before campaign start; never mix host/precision.
- Preserve exclusive receipts, failed attempts and raw probabilities. Check
  process and launch receipts before any dispatch; never duplicate after timeout.
- Score raw and BOTH named allocators under original caps. The achieved-count
  cap experiment is separate and is not silently adopted for this campaign.
- Report accuracy, all-class macroF1, fixed constrained-class F1, constrained
  precision/recall and per-class counts, violations, changed labels and actual
  updates. Add supervised CE/auxiliary contributions and gradient/displacement
  diagnostics. Compare seed-paired TraLO-minus-Clipper and TraLO-minus-null,
  every seed plus mean/SD/95% t interval; four seeds remain exploratory.
- Recompute metrics from saved labels/predictions independently. Test labels
  never enter loss, caps, selection or allocation. Do not equate feasibility,
  larger margins, or a single winning seed with useful prediction quality.

At08:00 report completed results AND any blocked/incomplete stages. Pause the
heartbeat; do not stop valid running jobs without checking ownership and purpose.
If dataset identity remains unresolved, state plainly that no knee result exists.

## Prepared code (not yet knee evidence)

`tralo/global_comparison.py`: optional separate constraint Adam; supervised
auxiliary configured independently; original defaults remain compatible.
`tralo/sample_losses.py`: label-dependent margin/FP definitions with hand values,
gradient checks, extreme-logit checks and explicit no-op tests.
Dataset-specific runner awaits confirmed dataset layout; don't adapt CIFAR
paths by pretending another dataset is CIFAR100.
