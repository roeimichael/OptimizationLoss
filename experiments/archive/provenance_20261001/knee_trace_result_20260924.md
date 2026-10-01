# Why the knee constraint updates sometimes hurt — overnight diagnosis

## Answer in plain language

The constraint update is doing the mathematics we asked it to do: it reduces
excess **summed probability** for constrained classes. But that is different
from selecting the right images for the available prediction slots.

For the original larger step, the dominant observed effect is a broad reduction
in grade-3 scores. Both false positives and genuine grade-3 cases cross into
other classes. The next supervised epoch often restores those genuine cases.
This creates repeated opposing movements. A smaller step reduces that damage.

That is not the entire story: reassigning constrained slots can recover correct
cases, and the smaller-step final update actually improves allocated F1.
Therefore “the constraint always hurts” and “null always wins” are both too
strong. We have identified concrete harmful behavior, but not a reliable
constraint-training advantage or a proof that the method cannot work.

## What we ran and verified

24 traced fits, each with its own uninstrumented reference: three recipes
(original step, smaller step, adapted features), four seeds, TraLO and matched
null. All 48 executions completed on Quadro in FP32. Each traced run exactly
matched its same-host reference in final parameters and probabilities.

All 11,880 snapshot hashes were checked. We saved logits after every supervised
minibatch and before/after every constraint update. Constraint snapshots include
parameters, gradients and Adam state. Every fit applied 460 supervised updates;
TraLO applied 15 constraint updates, null zero. No backbone retraining occurred.

The prior Blackwell parameter/probability values are not bit-identical, so this
is a separately identified diagnostic campaign, not pooled replication evidence.
Data, frozen feature caches, caps (grade 3 <=82; grade 4 <=16), and seeds remain
fixed. Evaluation labels enter only offline scoring, never training or allocation.
The test split remains unscored.

## What changes inside training

Numbers below are **repeated image-update events**, not distinct images or final
accuracy differences. Each recipe has 60 constraint updates across four seeds.

| Recipe | Wrong → correct immediately | Correct → wrong immediately | Updates starting with both hard caps satisfied | Spoiled events with a next epoch | Restored by that next supervised epoch |
|---|---:|---:|---:|---:|---:|
| Original step | 973 | 1,678 | 40/60 | 1,561 | 1,248 |
| Smaller step | 165 | 195 | 47/60 | 182 | 105 |
| Adapted features + smaller step | 18 | 30 | 0/60 | 26 | 16 |

Of the original recipe's 1,678 spoil events, 1,601 concern true grade 3.
Its average grade-3 logit movement (after removing common per-image offsets)
is -0.483, with mean within-update image-to-image standard deviation 0.058.
That is a predominantly broad shift, not a perfectly uniform one. Smaller step:
-0.047 with standard deviation 0.0056.

The subsequent supervised epoch strongly opposes the original update's centered
logit movement (mean cosine -0.871; -1 means exactly opposite directions).
This opposition is much weaker for smaller-step and adapted recipes. It is a
descriptive trajectory measurement, not proof that every recovered prediction
would have stayed correct under an alternative training algorithm.

## Why clipping changes the interpretation

Raw output chooses each image's highest-probability class. Upper-bound correction
only changes predictions exceeding a cap. Capped-first allocates constrained
slots first, so it can change predictions even when raw counts already comply.
These are different rules; do not pool them.

| Across all constraint updates | Net correct raw events | Net correct capped-first events |
|---|---:|---:|
| Original step | -705 | +105 |
| Smaller step | -30 | +26 |
| Adapted features | -12 | 0 |

These sums are not final gains: the same image may move repeatedly. They show
that raw damage alone cannot establish damage to final allocated predictions.

A particularly useful final-epoch comparison (four-seed mean constrained-class
F1, on a 0–100 scale):

| Recipe | Null after training | TraLO immediately before final constraint update | TraLO immediately after it, then allocated |
|---|---:|---:|---:|
| Original | 41.078 | 39.915 | 40.132 |
| Smaller step | 41.078 | 40.497 | 42.191 |
| Adapted features | 68.694 | 68.827 | 68.827 |

All three columns use capped-first allocation. The smaller-step last update
creates the observed positive endpoint difference. Earlier updates can do the
opposite: at epoch 10 it lowers this score from 31.027 to 29.865. We did not
select an earlier checkpoint after seeing these trajectories.

## Final differences, seed by seed

Each entry is TraLO minus matched null in constrained-class F1 points. F1 here
averages grades 3 and 4; it is not overall accuracy.

| Recipe | Seed | Raw | Upper-bound correction | Capped-first |
|---|---:|---:|---:|---:|
| adapted | 901 | -0.336 | +0.000 | +0.000 |
| adapted | 902 | -1.319 | +0.532 | +0.532 |
| adapted | 903 | +0.188 | +0.000 | +0.000 |
| adapted | 904 | -0.308 | +0.000 | +0.000 |
| none | 901 | -9.643 | -9.643 | +1.064 |
| none | 902 | -11.110 | -11.110 | -1.262 |
| none | 903 | -21.523 | -21.523 | -4.651 |
| none | 904 | -15.183 | -15.183 | +1.064 |
| smallstep | 901 | -0.660 | -0.660 | +0.532 |
| smallstep | 902 | -1.394 | -1.394 | +0.532 |
| smallstep | 903 | -1.857 | -1.857 | +0.532 |
| smallstep | 904 | +0.887 | +0.887 | +2.857 |

The smaller-step capped-first mean advantage is +1.11 points, with a paired
four-seed Student-t 95% interval approximately [-0.74,+2.96]. The adapted gain
is +0.13, interval [-0.29,+0.56]. These are exploratory, unadjusted intervals
from four seeds, not independent confirmation after development inspection.

## Was the gradient or optimizer implemented incorrectly?

Three independent checks found no arithmetic mismatch in these recorded steps:

- Analytic gradients reconstructed from probabilities and feature vectors match
  saved parameter gradients within 1.12e-7 absolute coordinate error.
- Adam displacements reconstructed from gradients, moments and hyperparameters
  match actual displacements within 1.42e-8.
- All 180 updates decrease the intended constraint loss with its pre-update
  multipliers and coefficient held fixed.

Thus the observed damage is compatible with correctly minimizing this loss.
These checks do not prove the whole pipeline bug-free or establish that its
scientific allocation definition is the right one.

## What the loss asks for, mathematically

For class c, soft count S_c = sum_i p_ic; hard count H_c counts argmax predictions.
A model can have H_c=73 under cap 82, yet S_c=105.19. This loss still pushes.

For the positive caps used here, e_c=max(S_c-K_c,0)/K_c, and

    L_constraint = sum_c lambda_c [e_c/(1+e_c) + rho*e_c²/(1+e_c²)].

Define a_c as the derivative with respect to S_c, zero when inactive/uncapped:

    a_c = lambda_c/K_c [1/(1+e_c)² + 2*rho*e_c/(1+e_c²)²].
    dL/dz_ij = p_ij [a_j - sum_c a_c p_ic].

The class-wide pressure a_c comes from aggregate excess. Probabilities determine
how each image feels it, but this derivative contains no information about that
image's true class. Identical probability vectors receive identical logit
pressure even when true labels differ. Shared feature weights can still change
rankings; the formula does not prove rankings cannot improve. Supervised
cross-entropy, and the previously tested training-label auxiliaries, do provide
correctness information in the supervised phase.

“Smaller step” changes constraint Adam learning rate from .001 to .0001; task
learning rate stays .001. It changes how far the correction moves parameters,
not what the objective rewards. At the same state/gradient/moments, displacement
is ten times smaller. Whole-run trajectories need not differ by that factor.

## What is justified next

We have sufficient evidence to stop treating this as an unexplained optimizer
collapse. The next controlled question is whether pressure during already-hard-
feasible states is useful or harmful for the chosen allocation rule. A gated
update may protect raw predictions but remove beneficial slot swaps. It requires
a separately registered comparison reporting both outcomes, not an assumed fix.
No gating intervention has been run in this overnight diagnostic so far.

An eventual stronger objective would need to improve selection among competing
images while respecting caps. These results neither validate such an objective
nor justify more arbitrary loss-weight sweeps. One dataset/backbone/cap pair,
head-only constraint training and four seeds limit generalization.

## Evidence and reproduction

Immutable trace source: 270b46a429002da6b6a0d7f99e68aace34b0b4c4.
Remote snapshots/receipts: /home/dsi/michaer8/tralo-rebuild/runs/knee-trace-quadro-20260923.
Local analysis directory: C:/Users/roeym/.codex/rebuild-audit-20260922/knee_sample_audit.
The latter retains scripts and JSON for trace_steps, trace_slots, trace_margins,
trace_adam, trace_loss and trace_gradients, plus every changed image ID. Offline
scripts use saved artifacts; they do not change training or select checkpoints.
