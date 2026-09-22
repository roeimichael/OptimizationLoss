# Collapse debugging: one seed, controlled interventions

Preserve original `0c355ba4` results. This is a causal diagnostic of seed701,
not a new winner search or replacement research comparison.

1. Exact replay of the original model, config, feature cache and batches with
   per-step count, gradient and Adam-state measurements. Require bitwise equal
   final probabilities to the saved seed701 result before interpreting anything.
2. Change only rho growth: hold rho at its initial0.5; retain lambda adaptation,
   optimizer, data, schedule and all other settings.
3. Change only optimizer-state sharing: preserve independent persistent Adam
   state for task and constraint updates, with the same parameters, gradient,
   learning rate, controller and update opportunities. Conditional activation
   can differ as trajectories change; report actual dose.

At the first supervised batch of epoch10 of the exact replay, clone identical
parameters and gradients and compare the original Adam state with its first
moment zeroed and with fresh Adam state. Measure task loss, gradient/displacement
inner product and predicted counts. This distinguishes a harmful update caused
by optimizer history from a harmful current supervised gradient.

The observer has a training-parity test. No development labels enter it or
`train_arm`. Scores are computed after each trajectory. Report all outcomes,
including failures to alleviate collapse. No precision changes or live-source
edits. Verify loss derivatives analytically and against the old formula; avoid
calling a design choice an algebraic bug without evidence.
