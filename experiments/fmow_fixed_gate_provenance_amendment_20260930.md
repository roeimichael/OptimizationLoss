# Fixed-dose pilot provenance gate correction

The prespecified seed-6199 step-on and step-off runs both completed with exit 0
on dsisco02 under immutable release `aae5f599ef9b4e8c124de536da92259e979a8640`.
The step-off reference finished at 2026-09-29 23:16:44 UTC. Neither run is
repeated or altered.

The first label-blind gate invocation failed before evaluating PTO parity. Its
scorer compared the runner's recorded hash of the immutable **input** config
against the reserialized `config.json` in the run root. Both files contain the
same JSON values, but the input has a terminal newline and the run copy does
not. The input hash is
`ab154887a663f70db6d0ceff8d5e731e80be319fa7e6025b0b7b93a87ea1fb70`;
the run-copy hash is
`9c8b869e3890bc01b24b605eee3e15cbb3395c88c1c8f69b67cd7e5efd82daee`.
The source, data, manifest, and split-count comparisons passed. The failed
gate and diagnosis are retained in the local audit receipt
`fmow_fixed_6199_gate_failure_20260929.json`.

The scorer now checks the recorded hash against the fixed input config in the
immutable release and checks parsed input/run config equality. This changes
only provenance verification, not the model, data, step, allocator, outcome,
or experimental settings. A new release is required before rerunning the
label-blind gate and independent pilot scoring. No 6200–6211 jobs may dispatch
until every original integrity and cost gate passes.
