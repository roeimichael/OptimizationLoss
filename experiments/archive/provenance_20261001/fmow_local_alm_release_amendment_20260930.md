# Native launcher-test amendment before the ALM pilot

The first frozen ALM release, `87cc645949686c52d622f21a2bc1f1bae02f2530`,
was pushed and deployed on 2026-09-30. Both hosts passed tracked-file byte
parity, but native pytest failed before any ALM GPU run. The failure receipts
are retained under
`/home/dsi/michaer8/tralo-rebuild/verification/87cc645949686c52d622f21a2bc1f1bae02f2530/`
on the shared server filesystem and in the local deployment audit directory.

All eight new queue tests reached a Windows-only path helper that indexed
`Path.drive[0]`. Linux absolute paths have an empty `drive`, so seven fixture
setups errored and the syntax test failed. This was a cross-platform test
error, not evidence that a launcher job ran. The launcher passed native
`bash -n`; no GPU job, study claim or output root was created.

The correction returns an existing POSIX absolute path unchanged and converts
Windows drive paths to WSL paths as before. The study question, configs,
training implementation, scorer, seed set and budget remain fixed. A new
immutable commit must pass native tests and tracked-byte checks on both hosts
before an exclusive GPU smoke or pilot. The failed release remains untouched.
