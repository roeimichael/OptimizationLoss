# fMoW persistent scorer provenance amendment, 1 October 2026

Seed 6700's step and reference pilots completed on dsisco02 under immutable
runner release `1bacdb448210a2083b181d86aa46bb8f0b29c6db`. They must not be
repeated. The first scorer invocation accidentally supplied the two queue
roots, rather than the child seed directories. It failed before replay and its
0.4239701218903065-second attempt is preserved, with byte hashes, under
`.fmow-persistent-local-operator-evidence/operator_invocation_failure_20261001_213226Z`.
The next gate charges this measured time through `--prior-gpu-hours`. The
original off-host receipt is `fmow_persistent_gate_v2_attempt`; no development
labels were opened.

The correctly addressed second gate then found a **scorer-only provenance
error before labels**: the runner's development-row manifest contains
`split=val`, `sample_id`, and `location`, while the independent reconstruction
contains only the two common identity fields. Its failure and registered GPU
time remain in the cost ledger. The amended scorer requires every runner row
to have exactly those three fields and the `val` split, projects it to the two
common fields, then compares every row in order to its independent recount.
No data, split, quota, gradient, replay or cost tolerance has been relaxed.

The amended release changes only the scorer, fixed queue and tests. Its
`tralo/*.py` runner source bytes are identical to the seed-6700 runner release.
The scorer pins that exact pilot release, authenticates the common complete
runner source hash, and binds a new full run to the amended scorer release.
The queue accepts only this exact pilot/scorer pairing and passes the child
seed directories to the fresh label-blind gate. The prior failure receipts
remain readable and the scorer recounts every still-registered attempt.

This amendment repairs provenance handling; it does not authorize a setting
change or imply an experimental improvement. The fixed full seed family
6701-6712 and both country cap levels remain unchanged. The fMoW development
countries have been viewed historically, so any resulting quality score is
exploratory rather than a confirmatory paper result.
