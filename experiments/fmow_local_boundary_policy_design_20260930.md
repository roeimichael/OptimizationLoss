# Candidate: label-free boundary-calibrated local snapshot step

Status: **design and CPU preparation only; no GPU pilot authorized by this
document**. This is a new hypothesis motivated by the completed negative
fixed-dose studies, not an amendment to their immutable runs. The same fmow2
development countries were repeatedly scored; any later outcomes on them
are exploratory. Nothing here licenses scoring reserved countries or choosing
a favorable setting from development labels.

## Scientific question

Can a constraint direction retain useful class-1 rankings when its step stops
near the pooled resource boundary rather than moving a fixed 0.1 parameter
distance? The observed 0.1 joint and PHR moves reduced about 233 raw class-1
calls to 15–17 at both pooled caps and evicted more correct selections than
they added. The earlier raw-feasibility search also failed its pilot because
it required one step to satisfy every pooled and country *hard* ceiling. This
candidate asks a narrower, falsifiable question: whether a label-free safety
rule can avoid over-suppression while improving allocation quality. Avoiding
over-suppression alone is not evidence of a model gain.

## Proposed single-step rule

At each PTO epoch snapshot, on unlabeled development images, use the existing
joint pooled-plus-country normalized soft-count gradient direction `d`.
Countries have **upper bounds**, not individual fill targets. For pooled cap
`G` and each country cap `K_s`, let `C_s(α)` be the sum of class-1 probabilities
after parameter displacement `α d`; let `H(α)` be the pooled raw hard class-1
count. Define positive normalized soft violation
`V_s(α)=max(0,C_s(α)-K_s)/max(K_s,1)` and total `V(α)=Σ_s V_s(α)` including
the pooled scope. No evaluation labels appear in these quantities.

At α=0, calculate each positive residual `r_s=(C_s(0)-K_s)/max(K_s,1)`
and directional derivative `v_s=d r_s/dα` by autograd. If **any** positively
violated scope has `v_s>=0`, skip and record the conflicting scopes: the
declared rule requires local first-order descent for all violated scopes.
Otherwise start at
`α_0=min(0.1, min_{r_s>0,v_s<0} r_s/(-v_s))`. Probe the fixed sequence
`α_0, α_0/2, …, α_0/2^12`, each with a fresh exact inference on the same
unlabeled pool. Accept the **first** candidate for which all of these hold:

1. `H(α) >= max(0,min(H(0),G)-1)` and
   `C_pool(α) >= max(0,min(C_pool(0),G)-1)`;
2. every pooled/country `V_s(α) <= V_s(0)+1e-6`;
3. total positive violation `V(0)-V(α) >= 1e-6`.

If none passes, take α=0 and log all probes and the skip reason. Validate
finite counts, derivatives, probabilities and actual displacement. This is a
discrete bounded search, not an optimizer guarantee. The one-call tolerance is
relative to the **smaller of the starting count and pooled cap**, not relative
to the starting count itself: from about 233 raw calls, it can allow 166 at
G=167 or 82 at G=83. It protects the ability to fill the pooled allocation,
but it does not guarantee preservation of individual correct selections or
ranking. Report correct slot entries and exits before interpreting any gain.
It is **not a local quota target**. Since country upper bounds sum to 209>167 or 104>83, requiring
every country to fill near its cap would be inconsistent with the pooled cap.
The rule intentionally does not require all raw hard caps to be feasible:
the unchanged allocator applies both exact upper bounds at deployment.
The full 0.1 steps in the completed block all lie far below these floors;
therefore the rule rejects the known collapse by construction. It may also
reject all positive radii; that is an honest null outcome.
For a read-only scale check, seed 6301's first G=167 snapshot logged pooled
soft count 194.57 and normalized directional derivative −23.41, giving a
linearized boundary radius about 0.0071, far below 0.1. This arithmetic uses
no outcome labels and is not a measured quality gain or a selected radius.

The radius 0.1 is only an upper ceiling inherited from the completed stress
test, not a value selected by its F1. The one-call and `1e-6` tolerances are
method definitions and require independent numerical and edge-case tests.
Because the five countries partition the pool, reject any initial or probe
record whose pooled soft count disagrees with the sum of country soft counts
beyond a declared floating-point recount tolerance.
Do not alter them after reading candidate scores. Record full per-scope
before/after soft and hard counts, derivatives, trial radii, actual model L2
dose, selected IDs and timing. The main PTO model, BatchNorm state, RNG,
optimizer, early stopping and ensemble window must remain byte-identical
to a no-step reference. Unlabeled development sample IDs and country IDs
may enter; individual development labels may not enter gradients, quotas,
stopping, or candidate selection.

## Fair controls and reading

Use the same PTO snapshots, seed, MobileNetV3-Large FP32 recipe, two fixed
caps G=167/83, Hamilton size-share country upper bounds, fixed ensemble
window and `local_capped_first` allocator as the completed studies. Proposed
arms are: no-side-step PTO (the zero-step post-hoc Clipper analogue), calibrated
joint TraLO, a sham with the joint arm's **realized** per-tensor displacement,
pooled-only direction at the joint arm's realized radius, and PHR direction
under the **same** boundary acceptance rule with its preregistered dual update.
If joint skips, its sham and matched pooled-only controls skip too. PHR may
have a different accepted dose; this makes joint-versus-PHR a **policy**
comparison. A both-active common-dose diagnostic can separate direction
from dose, but cannot replace the complete policy denominator. This is not
full ALM training or a replication of the earlier frozen-head ALM result.

For a later fixed campaign, primary allocated class-1 cc-F1 contrasts at
each cap should be joint-minus-PTO, joint-minus-sham and joint-minus-PHR,
one six-contrast Holm family over seed-paired differences. A lead requires
joint better than both PTO and sham at a cap, with no wholly negative 95%
paired interval on accuracy, macro-F1 or weighted-F1; joint-minus-PHR is
reported whether favorable or not. Show both caps, every seed, activation,
dose, complete slot turnover and raw feasibility. A null or negative full
block remains a result. Do not select a cap, radius, epoch, arm, country or
backbone after viewing candidate development scores.

## Gates before a new release or run

Implement the scalar policy separately from GPU inference and first test
hand-calculated examples, quota sums exceeding G, conflicting derivatives,
hard-count jumps, probe nonmonotonicity, invalid/nonfinite values and group
order invariance. Then test parameter-space directional derivatives against
finite differences, full versus chunked inference/gradient parity, exact
selected probability and artifact hashes, model/BN/RNG neutrality, the
same-host step-on/off PTO reference, label-free quotas, exact allocator
recount and independent metric scoring. Reject a pilot if any test, source,
data, split, preprocessing, gradient, dose, log or artifact gate fails.

Choose pilot and full seed blocks **before** launch, distinct from all prior
studies. The pilot's outcome scores may be read only after label-blind gates
and cannot select settings or stop a registered full block. Measure the pilot
on the intended host; a later full block must fit a separately recorded
ceiling no greater than 8 GPU-hours for this exploration, otherwise stop
before expansion. The extra inference probes may make the cost larger than
the previous 2.1 GPU-hour PHR block; no runtime assumption replaces a
measurement. Check actual GPU UUID/PID/owner on both hosts and take only a
genuinely free authorized card. Never share/stop another user's process.

Independent confirmation needs an audited new group-aware dataset or a
separately justified held-out protocol. The original fmow2 reserved countries
were not scored in these rebuild runs, but earlier project analyses inspected
the original test pool; they are not automatically a pristine confirmation
cohort. The 1,656-image Chen test remains sealed. A distinct dataset, data
access or held-out evaluation needs a concrete fixed plan and decision under
`AGENTS.md` and the main `docs/FRAMEWORK.md`.
