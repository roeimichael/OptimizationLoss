# CelebA norm logging transfer reduction

The terminal 7107 run exceeded its fixed runtime bound. Source inspection found
that task and matched/sham norm logging converts one GPU scalar per parameter
to a Python float. This can introduce many synchronizations per batch. The
terminal timing does not isolate this cost, so no measured speedup is claimed.

This bounded implementation collects the same per-parameter float64 squared
sums into one tensor, transfers that small vector once, and performs the same
ordered Python sum and square root. Task and snapshot callers use the same
helper. The derivative, gradient accumulation, Adam step, scales, caps, intended
and actual displacement calculations, precision, cohort, conditions and epoch
schedule stay unchanged.

The measurable signature is at most one scalar extraction for the explicit
new multi-parameter logging case, with bit-identical norm floats, unchanged
parameter/gradient/mode/RNG state, zero for missing gradients, and preserved
nonfinite output for the existing caller checks. A different norm float,
hidden nonfinite value, state mutation, or repeated scalar extraction fails
this change. Native CUDA timing and campaign completion remain unverified.

Only the new directly relevant CPU cases are run; completed native/model or
scientific stages are not replayed. Any later GPU verification needs its own
prospective case, fresh ownership/source/private checks and finite reservation
within the existing 24 GPU-hour ceiling. No seed is substituted or resumed and
no scientific pilot claim is made by this implementation.
