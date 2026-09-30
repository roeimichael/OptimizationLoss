# Draft: matched local ALM direction on fmow2

Status: fixed pre-run protocol with a reviewed CPU implementation, independent
scorer, exclusive launcher and fixed configs. No pilot or GPU run had occurred
at this freeze. Any scientific change after release requires a named amendment
and new immutable commit.
The five fmow2 development countries have been repeatedly scored, so this is
an exploratory mechanism comparison, not independent confirmation.

## Question and estimand

At the same post-training snapshot and maximum parameter displacement, does a
Powell-Hestenes-Rockafellar (PHR) inequality step policy select better class-1
items under pooled-plus-country deployment quotas than the completed TraLO
hard-active local step policy? The primary comparison includes each method's
native activation rule and direction; a separately labeled both-active
diagnostic examines direction behavior only where both methods step. This is
**one-step policy quality**, not full ALM training versus full TraLO training,
and cannot support that broader claim. The old frozen-feature global ALM result
is not a control.

Use the existing byte-audited fmow2 training/stop/development roles, Yuval
MobileNetV3-Large FP32 recipe, class-1 policy, fixed ensemble window, and
`local_capped_first` allocator from the fixed-dose protocol. Do not train on
or score the five reserved countries. Their image bytes were read only for
cross-role exact-duplicate hashing in the recorded data preflight; reserved
labels and model outcomes remain sealed. The trainer sees unlabeled development
images, sample IDs and country IDs only. It does not materialize development
labels. The independent offline scorer alone reads those labels after runs are
complete. Neither pilot scores nor full-block scores can select a setting.

For each pool size N=1673 use G=floor(N/10)=167 and G=floor(N/20)=83. The
corresponding local totals B=ceil(5G/4) are 209 and 104, apportioned by
Hamilton largest-remainder rounding from unlabeled country sizes. These are
the previously declared policy quotas, not estimates of class prevalence.

## Arms and mathematical definition

Each seed trains one unchanged PTO trajectory. At every recorded epoch snapshot,
make independent copies of the same model for the arms below; select the
pre-existing ensemble window only after the unchanged PTO stopping decision.
Advance PHR multipliers across all recorded epochs, never only selected ones:

* PTO: no side step.
* TraLO-local: the completed fixed-dose hard-active pooled-plus-country
  normalized soft-count direction, with 0.1 full-parameter L2 displacement.
* PHR-local: one side step in the negative gradient of the PHR inequality
  augmented penalty below, normalized to the same 0.1 displacement when its
  gradient is finite and nonzero.
* Pooled-only and sham: the same controls and dose rules as the fixed-dose
  study. The sham matches per-tensor TraLO-local displacement norms.

Each arm retains its native label-free activation rule: TraLO-local follows
hard-call activation; PHR-local follows the PHR penalty gradient. A finite,
nonzero gradient receives exactly 0.1 total displacement; an inactive arm takes
no step and logs why. Thus the primary contrast compares the two fixed
snapshot policies at a common **maximum** dose, not an equal realized dose on
every snapshot. Report activation and actual dose for every arm/snapshot. A
separately named directional diagnostic may examine snapshots where both arms
were active, but it cannot replace the complete primary denominator.

For the pooled scope and each of the five countries, define
`g_s = (sum_{i in s} p_i(class 1) - K_s) / max(K_s, 1)`. Use **signed** residuals,
not hard-call activation. For fixed rho=0.5 and nonnegative multipliers, the
PHR penalty is
`A(g,lambda,rho) = sum_s (max(0,lambda_s + rho*g_s)^2 - lambda_s^2)/(2*rho)`.
Initialize all multipliers to zero. After each snapshot, recompute signed soft
residuals on the PHR side model (which is the unchanged PTO copy if no PHR step
was applied), and update
`lambda_s <- max(0, lambda_s + rho*g_s)` for the next snapshot. The model
itself always resets to the common PTO snapshot. Maintain a distinct multiplier
vector per cap level and seed. A zero penalty gradient means no PHR model step,
but does not suppress its dual update or another arm's step. Never inject an
arbitrary direction. This is a registered
snapshot-PHR heuristic, not a converged classical ALM solve.

Rho=0.5 is the existing explicit ALM baseline, and radius=0.1 is the prior
fixed stress dose; neither was chosen from this new study's scores. Record the
fact that the latter dose harmed TraLO in the completed study. No radius/rho
sweep or favorable cap, epoch, country, seed or arm selection is allowed here.

## Outcomes and statistical plan

Primary outcome: allocated class-1 cc-F1, which equals class-1 F1 here.
Primary paired contrasts at each cap: PHR-local minus TraLO-local, and
PHR-local minus PTO (four contrasts in one Holm family). A positive exploratory
signal requires both contrasts positive after Holm adjustment at a cap and no
secondary-metric domination by either comparator at that cap. Domination means
a wholly negative two-sided 95% paired Student-t interval for accuracy,
macro-F1 or weighted-F1. Report all per-seed outcomes, paired differences,
mean, SD, intervals and adjusted p values, including negative and null results.
Use two-sided paired t tests at familywise alpha 0.05, with Holm adjustment over
the four primary p values. The displayed 95% paired t intervals are unadjusted
descriptive intervals; they do not override Holm inference.
Use seed as the replication unit; caps and snapshots are not extra replicates.
Keep raw hard/soft counts, every scope residual/multiplier, gradient norms,
directional derivatives, actual parameter displacement, quota utilization,
selected IDs, TP/FP/FN, correct entries/exits and wall time as diagnostics.
Independently recount the exact pooled cap and each country's upper bound;
individual country quotas need not fill because their sum exceeds the pooled cap.

## Validation and execution gate

Before GPU work, validate exact data/source/config/preprocessing hashes,
train/stop/dev/reserved country separation, sample-ID uniqueness and split
overlap, exact image duplicates across roles, and image/label/meta alignment.
The read-only audit in `fmow_local_alm_data_preflight_20260930.md` passed the
fixed data gates and found four training-only class-0 duplicate-image groups;
those rows remain in place. New-release source/config/preprocessing parity
must still be checked after deployment.
Prove no development labels enter trainer
objects, quotas, gradients, stopping or checkpoint selection. Test PHR against
hand values and finite differences, signed slack projection, zero/nonfinite
gradients, full/chunked replay parity, BatchNorm and RNG neutrality, independent
pooled/country scopes, exact 0.1 total dose, common PTO trajectory, allocator
ties/quotas and independent scorer recomputation on runner-shaped CPU fixtures.
Mutation-test the key integrity gates. Commit and deploy an immutable release;
verify tracked-byte parity, native tests and CLI on both hosts.

Reserve pilot seed **6300** and the fixed 12-seed full block **6301–6312**;
they do not overlap the completed fmow2 blocks. Run the step-on/step-off pilot
on the same host/precision, with exclusive
roots and exact PTO parity. Pilot scores may not select settings. Require every
source/config/data/split/artifact/gradient/dose/allocator gate to pass before
expansion. The label-blind gate must observe at least one finite nonzero PHR
step at each cap; a wholly inactive PHR pilot cannot validate direction
quality and stops expansion regardless of scores. Check same host, GPU model
and FP32 metadata and record each physical GPU UUID from both pilot launch
receipts, in addition to exact PTO
snapshot parity. A dead process or SSH loss calls for forensic inspection,
never a blind retry. The launcher must atomically claim each fixed seed/job
across all new roots before execution so a second queue cannot duplicate it.
Preserve failed claims and run evidence; never overwrite a run.
On 2026-09-30, read-only `findmnt` on both dsisco01 and dsisco02 placed the
runs directory under the same NFS source
`psa.local.biu.ac.il:/ifs/a/home` mounted at `/home`. The shared claim uses
atomic `mkdir`; a CPU mock test also checked concurrent duplicate refusal.

The independent full block contains exactly 12 paired PTO fits. The
completed fixed-dose pilot projected 1.498 GPU-hours for 13 step-on fits plus
one step-off reference on dsisco02. That is a historical runtime reference,
**not this study's budget or a valid new projection**. Measure the
new pilot on the target host and project `13 * step_on_hours + step_off_hours`.
The proposed separate ceiling is **8 GPU-hours**. Stop if the pilot-based
projection exceeds 8 GPU-hours; do not silently reduce seeds, arms or checks.
GPU ownership checks and at most three concurrent cards remain mandatory.
The user's earlier instruction to pursue local constraints, compare TraLO with
ALM, use free GPUs and make research decisions without waiting covers this
bounded exploratory direction comparison. The 8 GPU-hour ceiling is a
self-imposed limit within that authorization, not a target to spend. A launch
still requires all source, data, preprocessing, gradient, scorer, queue and
server integrity gates in this protocol; failure leaves the GPUs unused for
this study. A different data source, held-out evaluation or larger budget
requires a separate decision. This draft itself is not an executable release.

If a later full algorithm comparison is desired, separately specify matched
training schedules, nulls and intervention doses for end-to-end TraLO and ALM.
Do not interpret this direction comparison as that study.
