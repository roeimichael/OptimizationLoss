# Explicit device selection for independent tabular replay

The archived CelebA complete-block scoring invocation on 4 October did not set
`CUDA_VISIBLE_DEVICES`. Its preceding inventory showed three GPUs occupied by
other users and one free GPU. The recorded scorer PID establishes its process
owner, but no preserved observation binds that PID to a physical GPU UUID.
Exclusive ownership during that replay is therefore **unverified**. The training
queue's GPU receipt does not establish the scorer's device. This is an operational
evidence gap; it is not proof that a foreign GPU was shared or a new quality result.

Future `analysis.score_tabular_persistent` model replay now refuses a missing,
index-based, abbreviated, malformed or multiple-device selection before reading
run artifacts or querying CUDA. It requires one complete physical GPU UUID in
`CUDA_VISIBLE_DEVICES` and records that **requested** UUID separately from the
training queue's ownership receipt. Actual hardware availability and exclusive
ownership still require the existing both-host inventory and launch gate. A
well-formed UUID alone establishes neither availability nor ownership.

The change is confined to replay device selection and its audit metadata. The
objective, gradients, allocator, quotas, private-label boundary and scientific
settings are unchanged. Completed releases and scores remain immutable; none was
replayed to repair missing historical evidence.

Independent examples reject seven unsafe selections before CUDA or artifact
access and accept a complete UUID without making an ownership claim. They failed
against the prior implementation. The focused scorer, runner, pool preparation
and quota suite passes **26 CPU tests locally**. Deployment verification must
check committed bytes and repeat that suite plus the actual CLI's unpinned
refusal on both hosts before the new release is described as deployed.

Accounting evidence is preserved outside source in
`C:/Users/roeym/.codex/rebuild-audit-20260922/`:

- `gpu_block_score_original_boundaries_20261005.json` links 18 original tool
  records to the failed ISIC scorer, corrected ISIC scorer and completed CelebA
  scorer. Only exit metadata, checksums and historical process/device observations
  were inspected; no labels, model quality or original commands were replayed.
- `global_gpu_budget_block_score_reserve_20261005.json` conservatively reserves
  their full synchronous invocation bounds: 480.981, 484.711 and 1,021.986 seconds.
  Unknown successful replay UUIDs remain unknown; nonoverlap is checked against
  every possible physical device on each host. The failed ISIC PID has a recorded
  physical UUID. These are outer bounds including CPU, SSH and waiting, not exact
  GPU or kernel durations.
- `global_gpu_budget_block_score_reserve_verification_20261005.json` independently
  checks the 1,987,678 milliseconds, nine evidence hashes and 18 original record
  hashes. The ledger SHA256 is
  `a2db51aafcf3469bc432ab6e82cbed118be1b2f3f75d54862b243c1930b00812`.

The conservative planning charge is now **73.787372700 GPU-hours**. Subtracting
it from 96 gives **22.212627300 nominal hours**, not a certified spendable balance.
Original/corrected pilot gates, the fresh fMoW full gate, knee verification/replay,
the additional-root census and calendar scope still need closure. No missing cost
is treated as zero, and this reservation does not authorize another campaign.

For a future authorized replay, use a newly verified free UUID with the new
immutable release and an exclusive new output path. Preserve its timing, process
and actual device observation separately from training receipts. ISIC2024's
scientific protocol, adapter, source/label boundary, cost/dose and finite budget
gates remain pending; this guard does not approve them.
