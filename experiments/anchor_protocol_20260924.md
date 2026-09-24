# Second-stage anchored count update, registered after far-error results

The earlier four-seed far-error study has already been inspected. Knee's
sample-aware Null improved relative to the old Null, but the separate count
step gave no consistent extra benefit. This follow-up is **adaptive exploratory
work**, not independent confirmation of a selected method.

## Hypothesis and intervention

The count update can move correct samples across the decision boundary because
its unlabeled soft-count gradient does not know which predictions are correct.
Test whether placing a training-label signal in that very update protects task
quality while retaining pressure toward the original quotas.

At each post-warm-up epoch, compute on the training set only:
`A = CE_train + 0.1 L_far_train`, where L_far is specified in
far_error_protocol_20260924.md. Apply one separate-Adam step using:

- Anchored Null: `0.1 A` (zero count term).
- Anchored TraLO: `L_count(unlabeled development) + 0.1 A`.

Both receive the same 15 extra training-label passes and 15 extra optimizer
steps. Their only loss difference during this step is L_count. Both also receive
identical supervised minibatches with CE + 0.1 L_far after warm-up. This tests
the incremental effect of the count term under a stronger matched control.
It increases compute and training-label dose relative to the earlier sample-only
study, so cross-study gains cannot be assigned to TraLO's count term.

## Fixed execution and stopping

Reuse both verified frozen ResNet18 feature caches, original global caps, two
datasets and seeds1001-1004. Twenty head epochs, five warm-up, batch256,
task Adam0.001, separate Adam count/anchor learning rate0.0001, lambda0.01 +
0.05 controller, rho0.5, FP32 on one Quadro host. No parameter grid or
checkpoint selection. First seed on both datasets gates arithmetic, matched
warm-up/batches, exact anchor and count dose, finite output, artifact hashes,
and independent allocation/metric checks. If these pass, run remaining three
even if first-seed scores are poor. Save predictions, checkpoints and logs.

The primary comparison is anchored TraLO minus anchored Null in capped-first
constrained-class F1 under the same original exact output slots. Report raw and
upper-bound policies, accuracy, macro-F1, per-class counts/confusions, and
original-cap raw feasibility. The earlier sample-only study is a labelled
exploratory reference. Four seeds and reused development splits cannot
establish a general result. No development labels may enter any gradient,
quota, allocation or checkpoint choice.
