# Registered constraint-strength sweep, 24 September 2026

User explicitly requests broader iteration and more replications after the first
48-fit comparison. Register before observing any new-seed outcome. Runtime math
is unchanged from 6411f2b2; only configuration and replication change.

## Question and scope

Is the TraLO gain a narrow step-size accident, or does a range of strengths
improve the same allocated output over a phase-matched null? Does tuning the
initial ALM baseline over an equal seven-setting budget change its outcome?
This study estimates training-seed sensitivity on two fixed development splits;
it cannot establish generalization to new datasets/patients or a tuned method's
untouched-test performance. Seeds901-904 remain prior exploratory evidence and
are NOT pooled into this study. No new data access or test scoring.

## Fixed design

384 fits = 2 datasets x 12 fresh seeds x (7 TraLO settings + 7 ALM settings +
Clipper + common null). ALM Null and TraLO Null have identical implementation
behavior, already verified across all eight earlier dataset/seed pairs. Run the
common null once per new pair; do not count duplicate null runs as more evidence.
Seeds1001-1012. First seed is a numerical/operational pilot for every setting;
remaining11 proceed only after every pilot passes, regardless of accuracy.

Frozen ImageNet ResNet18 feature caches and dataset identities exactly as in
alm_two_dataset_20260924.md. Knee5778/826 train/development, CIFAR100 fixed
10000/2000 subset. Same20 head epochs,5 warmup,batch256,taskAdam.001,FP32,
TF32off,Quadro dsisco01 for the entire study. No backbone retraining. All methods
use identical initialization and supervised batches for each dataset/seed.
No model selection/checkpoint choice from labels. Final epoch only.

TraLO constraint Adam learning rates: .00001,.00003,.0001,.0003,.001,.003,.01.
This logarithmic grid brackets both original rates and includes intermediate
and stronger/weaker steps. All other TraLO settings remain fixed: lambda.01,
increment.05,rho.5,separate constraint moments,one opportunity per postwarmup epoch.
ALM fixed rho grid: .005,.015,.05,.15,.5,1.5,5; multiplier starts0. Same joint
PHR formulation and one projected dual update per epoch. Seven values each is
equal configuration count, NOT equal compute or equal intervention frequency.
Do not claim optimal tuning of either method. No additions to this grid after
seeing scores. Extreme settings that fail are retained and reported, not replaced.

## Output budgets, metrics and inference

Primary view: capped_first cc-F1, identical exact counts at every constrained
class: knee82/16 for grades3/4; CIFAR10 each for classes0-9. Other classes
unrestricted. Retain raw and upper_bound_correction; accuracy,macro-F1,perclass
TP/FP/FN,count violations,time and applied dose accompany every setting.
These diagnostic metrics do not establish a universally preferred utility.

Primary statistical family: 28 contrasts =2datasets x2methods x7settings, each
versus its dataset/seed-matched common null. Two-sided paired t-tests, Holm
familywise adjustment at .05. Report paired means,SD,ordinary95%t intervals and
Bonferroni simultaneous95%t intervals over28 contrasts. Positive adjusted result
requires positive mean AND Holm p<.05; negative adjusted outcomes are failures
of that setting under this protocol. If all deltas exactly0 use p1/zero interval;
nonzero constant deltas must be flagged, not silently given infinite confidence.
Report wins/ties/losses across12seeds. All other metric/policy contrasts are
exploratory, without pretending to be part of the corrected primary family.

Plot the whole response curve; report adjacent-setting behavior and both datasets
without choosing the best setting and presenting its unadjusted interval as
confirmation. A positive result only on this already inspected development set
needs a separately registered independent evaluation before publication claims.
No posthoc change to caps,epochs,initialization,labels or data subsets.

## Operational acceptance

Reuse only cache receipts whose hashes match; retain source/GPU/precision and
exclusive launch records. Both-host GPU PID ownership before dispatch. Four
queues on four free GPUs, per-job exclusive outputs, abort queue on failure.
Native tests/source hashes on both hosts before the pilot. Every completed run:
artifact hashes,finite applied updates,matched warmup/batches,exact output quotas,
independent confusion-based metrics. Retain all probability vectors/checkpoints.
Pilot validation concerns execution, not whether a metric improved.
