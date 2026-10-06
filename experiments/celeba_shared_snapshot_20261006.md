# CelebA shared-trajectory snapshot block -- 6 October 2026

The human gives standing authorization to implement and run CelebA experiments,
explicitly overriding repeated skill/project approval questions. We adopt the
already presented additional24 aggregate GPU-hour operating ceiling (agent-selected
limit, not a verbatim human number), including new validation/pilot/failures/study/
scoring,6h/job. Historical96-account partial82.13144744416667333333333332 and
uncertified remaining room stay separate; no retrospective credit or claim that
old unknowns disappeared. Source, privacy, ownership and finite-time validation
remain required. No new run is launched merely by writing this protocol.

## Fixed task and readout

Smiling target; Male annotation supplies female/male local groups. Reuse the
existing identity-disjoint train140166/stop31966/development30467 cohort and
the two pre-existing unlabeled-size caps: level1 pooled7617/female6211/male4453;
level2 pooled10664/female7986/male5725. Synthetic capacity constraints are not
demographic fairness/clinical requirements. Development targets were previously
viewed, so new comparisons are exploratory, not untouched-test evidence.

Image-only first, then image+Male. Metadata is an additive two-logit coefficient
of the supplied Male indicator, initially zero, atop the same image classifier.
Every rival/null receives identical inputs within its condition. This assumes
the annotation is available at inference; it does not estimate metadata and
does not admit Smiling or unspecified annotations as features. Separate modality
main effects from TraLO-minus-null; a feature gain is not a TraLO contribution.

MobileNetV3-Large/ImageNet V2 sha256
5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997.
Authenticate and load the same cached bytes, native1000-way strict load then a
fresh2-way head. Six full epochs, batch32, Adam1e-4, wd0, unfrozen FP32/TF32off,
existing train/eval transforms, deterministic algorithms/CUBLAS :4096:8. No
subsampling, independent arm selection or outcome-chosen window.

One CE/PTO task trajectory per input condition; corrections on isolated copies
only after epochs2-6, epoch1 warm-up. Shared model/BN/Adam/gradients/modes/RNG
remain unchanged by side work. Every arm averages probabilities4/5/6 FIRST,
then the same joint capped-first deployment policy. Stopping-carve CE loss is
measured before/after correction as a diagnostic; it selects no arm checkpoint.

Arms: matching null/common CE clipper (ONE baseline); original bounded TraLO
scalar with.0045/.0074, rho.5, scope multipliers1; native PHR.0027/.0112;
PHR direction at actual TraLO displacement; random direction at actual TraLO
displacement with a distinct fixed local RNG; focal gamma2, no class weighting,
on a separate matched task trajectory. The PHR dual starts zero and advances
from shared pre-copy counts, even if native dose is vetoed. Native and matched
branches reuse the authenticated SAME pre-copy PHR gradient and dual recurrence,
with distinct model copies/dose controllers; no corrected-count recurrence or
dual reset. This reuse removes duplicate forwards, not an independent replication.

The bounded objective is the retained sum of lambda_s times
u_s/(1+u_s)+rho*u_s^2/(1+u_s^2), u_s=relu(soft_s-K_s)/max(K_s,1).
Its derivative is unchanged and unnormalized. A common.1 displacement ceiling
is not a dose target. Dose matching uses the measured TraLO before/after norm,
absolute tolerance1e-6 plus relative1e-4 fixed BEFORE new native output. A
positive reference with zero comparator direction or an actual mismatch fails
matching and remains visible; no direction fabrication/tolerance growth/scale
retuning/arm dropping. Keep intended and realized doses separate. Raw count
reduction, stopping-loss change and hash differences are not cc-F1 gains.

Prospective pilot7100, then study7101-7104. Verify historical named-root non-use
and shared exclusive claims before dispatch, refuse collisions without blind
substitution/resume. Pilot is label-free and never enters quality inference.
Only trusted scoring accesses private Smiling targets after four complete
paired common-source/input/recipe blocks pass label-free integrity. No sealed
Chen/test/reserved fMoW access. No old completed model/scorer/operator replay.

All five TraLO contrasts per cap/input versus null, focal, native PHR, matched
PHR and sham: family20. Four independent seeds, native cc-F1 plus the full fixed
metric profile, every paired delta/mean/SD/t(3)95%CI; Holm20 secondary exploratory.
Zero variance CI/p unavailable; retain the full family. No superiority promise.
Failures/negative estimates/trades remain reportable. This is a snapshot policy,
not proof about every persistent or simultaneous penalty implementation.

Before study expansion, measured pilot plus four forecasts at1.5 factor and
remaining verification/scoring bounds must fit24. Choose/declare technical
repairs autonomously within standing authorization, preserving old attempts;
do not manufacture a positive setting from viewed quality. Prefer a verified
free Blackwell card; fresh both-host UUID/PID/owner/command checks, one visible
physical UUID with actual runtime agreement, no sharing/stopping foreign PIDs.
Freeze source/config and retain exclusive output/logs, finite job/aggregate
limits, active heartbeat and phase costs. Idle is not ownership.

Implementation evidence: new core/native pipeline phases under
C:/Users/roeym/.codex/rebuild-audit-20260922/celeba_native_core_* and
celeba_native_pipeline_*. Synthetic CPU passes validate new paths only; they
do not certify the actual model/data/CUDA/private/device/cost/science campaign.
The original RED/import/scalar-conversion failures are retained. Old13:xx/14:03/
16:03 arithmetic preparations and all older completed stages remain terminal.
