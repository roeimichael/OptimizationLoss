# Persistent pooled-plus-country fMoW comparison, 2 October 2026

## Decision

The fixed 12-seed MobileNetV3 comparison completed and passed its independent
complete-block audit. It does **not** establish a TraLO gain. At the tighter
capacity, TraLO scored 0.01393 lower constrained-class F1 (cc-F1) than the
matched focal-loss classifier followed by the same post-hoc clipper; the paired
95% interval was [-0.01964, -0.00821] and the eight-comparison Holm-adjusted
`p` was 0.00184. At the looser capacity, TraLO's +0.00502 difference from its
schedule-matched zero-correction control had a 95% interval spanning zero.
The inexact PHR comparator, not TraLO, had the only positive adjusted primary
contrast at the looser capacity. TraLO also had lower accuracy and weighted F1
than the zero-correction control at both capacities.

These are **exploratory development-country** results. Those country labels
were inspected in prior studies, and no reserved country was scored here. The
result is useful for rejecting a claim about this fixed configuration and for
diagnosing training, but it is not an independent geographic confirmation or a
claim that the loss family cannot work elsewhere.

## Frozen comparison and audit

All 12 prespecified seeds, 6701–6712, finished with exit code zero under
immutable runner/scorer release `6cf4982410efd60ddbf0a33c53a24585991b5770`
on dsisco02. Each seed had a cross-entropy (CE) zero-correction control, a
focal-loss-plus-clipper control, and persistent TraLO and inexact PHR-ALM arms
at two fixed pooled/country capacities. All arms shared the MobileNetV3
backbone, seed, image pipeline and deployment allocator. The scorer checked
source/config/data/split/weight identity, events, outputs, constraint dose,
and checkpoint-to-probability replay before opening development labels. Its
fixed deployed score averages probabilities from **epochs 5–7**, then allocates
the allowed positive calls; it does not select the lowest-stop-loss epoch.

The accepted raw result is
[`independent_full_score.json`](C:/Users/roeym/.codex/rebuild-audit-20260922/fmow_full_score_20261002/independent_full_score.json),
SHA-256 `95a095f32dee65c93bd367f0ec245dfed080352cb79dc84d8ebe6ff7f9534637`.
That file retains every seed, count, metric, contrast, interval, replay result,
and cost receipt. All 12 seeds' logs and checkpoints have separate verified
off-host backups in the same audit directory. Earlier failed preflight/gate
attempts remain preserved in the cost registry. Total accounted study compute
was 6.148 aggregate GPU-hours, below the finite ceiling; the independent
replay added 0.0224 GPU-hours.

## Actual deployed metrics

Numbers are means over 12 paired seeds; cc-F1 is shown as mean ± seed standard
deviation. The other columns are mean values. The capacity labels `10` and
`20` are the frozen quota divisors, not a percentage of patients.

| Capacity | Arm | cc-F1 | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| 10 | CE + clipper / matched null | 0.48085 ± 0.01249 | **0.53626** | **0.47480** | **0.55143** |
| 10 | Focal + clipper | **0.49347 ± 0.00722** | 0.53297 | 0.46835 | 0.55009 |
| 10 | TraLO | 0.47955 ± 0.01064 | 0.52446 | 0.46574 | 0.54053 |
| 10 | Inexact PHR-ALM | 0.47955 ± 0.01423 | 0.52232 | 0.46251 | 0.53834 |
| 20 | CE + clipper / matched null | 0.39576 ± 0.01205 | **0.53198** | **0.45955** | **0.54415** |
| 20 | Focal + clipper | 0.40078 ± 0.01121 | 0.52984 | 0.45376 | 0.54393 |
| 20 | TraLO | 0.40078 ± 0.01625 | 0.52889 | 0.45653 | 0.54260 |
| 20 | Inexact PHR-ALM | **0.40524 ± 0.01674** | 0.52535 | 0.45455 | 0.54085 |

Bold marks a largest mean within a capacity and metric, **not** a significant
win. cc-F1 is class-1 F1 after the actual pooled/country allocation. Accuracy,
macro-F1 and weighted-F1 guard against a narrowly improved capped class at
the expense of the full classifier. The raw JSON also gives constrained
precision/recall, class supports, per-country admitted true positives and
the per-seed deployed counts. Feasibility is checked by the common allocator;
meeting the caps alone does not establish classification quality.

## Paired primary comparisons

The unit is a matched seed on the fixed, already viewed development countries.
Intervals are two-sided 95% Student-t intervals on seed-paired cc-F1
differences; `p` values are Holm-adjusted over the frozen primary family.

| Capacity | Contrast | Mean difference | 95% interval | Adjusted p |
|---|---|---:|---:|---:|
| 10 | TraLO − CE/null | -0.00131 | [-0.00643, +0.00382] | 1.0000 |
| 10 | TraLO − PHR | 0.00000 | [-0.00490, +0.00490] | 1.0000 |
| 10 | TraLO − focal clipper | **-0.01393** | **[-0.01964, -0.00821]** | **0.00184** |
| 20 | TraLO − CE/null | +0.00502 | [-0.00311, +0.01315] | 1.0000 |
| 20 | TraLO − PHR | -0.00446 | [-0.01028, +0.00137] | 0.7209 |
| 20 | TraLO − focal clipper | 0.00000 | [-0.01117, +0.01117] | 1.0000 |
| 20 | PHR − CE/null | +0.00948 | [+0.00334, +0.01561] | 0.04150 |

The omitted cap-10 PHR − CE/null comparison is -0.00131
[-0.00623, +0.00362], adjusted p=1.0000. A confidence interval around zero
is not evidence of equivalence; the viewed development countries and small
seed count further limit inference.

## What the training logs say

The correction was genuinely active. Across 12 seeds, each TraLO and PHR
capacity arm attempted and applied all 72 scheduled constraint corrections;
there were no dose-veto skips in those arms. The mean actual parameter
displacement per correction was 0.01457 (TraLO, cap 10), 0.00669 (TraLO,
cap 20), 0.01522 (PHR, cap 10), and 0.00806 (PHR, cap 20). The null and focal
controls correctly applied zero constraint corrections. Thus a zero effect
cannot be dismissed as a silently inactive TraLO step.

The CE training loss fell from a mean 0.455 at epoch 2 to 0.102 at epoch 6,
while CE stop loss rose from 1.429 to 2.104. TraLO showed the same broad
divergence. An immediate TraLO correction increased stop loss in 55.6% of
cap-10 and 54.2% of cap-20 correction events; its mean immediate changes were
+0.00464 and +0.00466. These are **mechanism diagnostics**, not independent
quality tests, and the focal loss is on a different scale. The fixed scoring
ensemble uses late epochs 5–7, so worsening stop loss during that interval is
a plausible contributor to the weak deployed scores. The logs do not establish
that overfitting alone caused the TraLO–clipper gap.

## Consequence for the thesis

The local, persistent correction is an actual training intervention, unlike a
single snapshot step or an output-only clipper. Yet this experiment provides
no positive TraLO paper claim on fMoW. It supports a sharper question: can a
future constraint objective improve **ranked case selection** while ordinary
classification generalizes, under a matched late-checkpoint policy? Do not
retune this repeatedly viewed fMoW development set to force a win. Resolve the
separate CelebA inactive-arm gate and ISIC checkpoint-replay gate from their
preserved failures, then run fresh, independently justified tabular-group
studies with matched null, clipper and ALM controls. This fMoW PHR arm is an
inexact primal/dual comparator, **not** an exact full ALM solution.
