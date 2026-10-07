# Reuse the unchanged snapshot origin probability pass

The null copy already predicts the complete development pool before any
correction. TraLO and native PHR copies have the same authenticated model state,
pool order and fixed evaluation transform. Their coefficient construction now
uses those exact CPU probabilities and IDs. Each method retains its own complete
backward pass and the existing sample-order/probability-parity checks. Native and
matched PHR still use the same gradient and equal dual recurrence as before.

Only coefficient-pass input decoding/forward evaluation is removed. A corrected
epoch with two caps has fourteen development forwards instead of eighteen;
all ten stopping passes remain. Warm-up, task training, cap-specific null passes,
post-correction observations, doses, scientific settings and ensemble windows
remain unchanged. This is a source-level pass count, not a measured speedup or
evidence that the scientific pilot/campaign fits the remaining budget.

The caller must supply probabilities from the same unchanged eval model and
ordered pool. Snapshot copies authenticate their model state against the null
origin; the null observation now also rejects model-state mutation. The optional
gradient input is local to this fresh observation and is not a persistent cache.
No whole-cohort image/tensor cache or additional data access is introduced.

New directly relevant fixed-input cases compare fresh and reused gradients,
duals, snapshot probabilities/doses/stopping observations, PTO/Adam/gradient
state and pool traversal counts. They also reject changed probabilities, IDs and
nonfinite inputs. These CPU cases do not certify CUDA numerical equivalence,
private-boundary integrity, scientific quality or complete-pilot runtime.
