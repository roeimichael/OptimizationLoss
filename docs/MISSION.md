# TraLO reset: execution state

Updated 2026-09-14. User approved recoverable large cleanup, fresh evidence,
validated logging/data/code, then a within-TraLO modification and monitored SSH
experiments. No historical acceptance tally is carried forward.

## Current stage

**Recoverable archival and Git cleanup done; runtime/config thinning is NOT done.**
New experiments have not launched. Local `results/` is empty. The seven remaining
old server `optloss-*/results` directories contain no files or links; the 17
populated trees are in the external history archive. Historical task-window
measurements still influence the generator/scorers: the fresh workflow must
remove that dependency before producing new comparisons.

At 2026-09-14 11:24 Asia/Jerusalem, both SSH hosts were reachable. Each had four
GPUs at 0% utilization / 0 MiB used, no reported GPU compute process, and no
Python/torchrun process owned by this account. This is a snapshot, not a GPU
reservation. Recheck both hosts before any remote move or dispatch.

## Ordered work

- [x] Preserve the pre-reset dirty diff and exact instruction files under `.codex/`.
- [x] Move the four large operational narratives into the local history archive;
  replace them with a short current protocol and state file.
- [x] Preserve historical artifacts outside Git tracking with verified backups;
  see `docs/GIT_TRACKING.md`. Archival copies are not fresh-clone dependencies.
- [x] Archive old local and remote result trees with manifests; keep data arrays,
  checkpoints, predictions, parent-extension links, and recovery paths intact.
- [ ] Retire obsolete probe/source prose and historical-prose test ratchets while
  retaining mathematical, pipeline, allocator, metric and failure-path coverage.
- [ ] Reduce `configs/protocol.yml` and `configs/gen_campaign.py` from 42 declared
  arms to the fresh comparison: TraLO, its zero-constraint control, clip,
  focal_clip, Fioretto-LDF, Hounie-RCL and ALM. Rival zero-constraint cases remain
  correctness fixtures, not extra research arms. Remove the old options AND
  their runtime readers/branches; do not hide them behind disabled defaults.
- [ ] Fix reference generation to FP32 constraint arithmetic and normalized
  gradients. Current YAML defaults are `constraint_fp32: false` and
  `constraint_grad_mode: clip`; do not generate a fresh campaign from them.
- [ ] Replace historical `configs/task_windows.yml` decisions with newly measured
  development diagnostics. Require a fresh campaign identity and explicit input
  inventory for reporting; test rejection of archived, mixed and unmarked runs.
  Use a new `OPTLOSS_MODEL_CACHE` namespace so fresh runs cannot reuse historical
  warm-up checkpoints; permit sharing only inside the newly frozen release.
- [x] Replace the stale `keepworking` skill and forward-test the new reference.
- [x] Fix and regression-test AMP step/event accounting and the optional uniform
  estimator's chunk-dependent weight. Neither establishes a classification gain.
- [ ] Complete cc-F1-first reporting and test metric definitions end to end.
- [ ] Finish logging validation: TraLO now has timed JSONL scope/update events;
  rival-dual event integration and real-backbone cost validation remain pending.
- [ ] Audit current datasets and development cut saturation without selecting on
  a TraLO win. Resolve untouched holdout availability with the user.
- [ ] Review the within-TraLO change proposal with the user, implement/test it,
  commit a frozen release, and verify SHA-256 parity on the target host.
- [ ] Launch first-run pilots on two GPUs after gates pass, attach monitoring,
  inspect logs, then expand only if healthy (maximum three GPUs).

Implementation order: remove obsolete arms/knobs with reference-behavior tests;
then thin their probe/test consumers; then validate fresh reporting/data/logging;
then freeze/sync and launch. The current reference loss, dual update ordering,
allocator and retained baselines must not change during structural cleanup.
Unknown or removed config keys must fail clearly, not be silently ignored.
Archive historical tests/probes through the existing recovery process; retain
compact tests for gradients, caps, metrics, data splits, logging and recovery.

The first GPU experiment tests pipeline/log validity and dataset headroom, not
superiority. Use an audited development split, one backbone and one host before
expanding. Inspect per-group allocation-cut errors, soft/hard residuals, dual
trajectories, actual applied updates and parameter displacement alongside cc-F1.
Only after those checks and a reviewed modification: compare the seven core
methods plus the approved candidate at two distinct cap levels with at least
four seeds, paired native-metric
uncertainty, equal training budgets and recorded extra constraint compute.

## Validation and release state

Snapshot `abb18d27`: local full suite **668 passed, 1 skipped**. The skip needs
real campaign logs and is not a pass. Target-host full suite initially found a
brittle rounded-hash ALM test and two CRLF/LF historical-table comparisons.
After those test repairs, the **server full suite also passes 668, with 1 skipped**
(195.37 seconds, CPU only). The ALM fixture checks hand-computed actual gradients
with positive/negative controls. XML receipts are in the ignored local `.codex/`
directory; tests do not certify GPU behavior or data validity.

The alpha-liveness test now compares raw probabilities against a float32
tolerance instead of rounded hashes. It passes locally and on the server.
The masked-gradient AMP counter bug is reproduced and fixed with CPU GradScaler
controls. It is not evidence of a default FP32 TraLO failure.

**BCN is not launch-ready:** two exact resized images cross train/test with
conflicting class labels and different official lesion IDs. No images were
deleted. See `docs/audits/2026-09-14-reset.md`. Other datasets pass the exact
cross-split image check; near-duplicates and unused-holdout status remain open.

Server validation checkout: `/home/dsi/michaer8/optloss-reset-validation-20260914`.
All 424 tracked files of `abb18d27` matched actual local bytes before tests.
It now includes the two test-file repairs; runtime code is unchanged. Subsequent
Git-hygiene packaging was committed and pushed at `62581d90`; local HEAD and the
remote branch were verified equal, with no staged or unstaged changes. The app's
large display is exactly `origin/main...62581d90`: 117,078 additions / 33,372
deletions, a committed branch comparison, not an uncommitted working-tree diff.
The last cleanup/research pair against parent `d17306b3` is +2,914 / -61,364.
**Re-sync the final clean commit before any campaign.** This checkout is not
launch-approved; the older prepared tree remains intact. iwildcam/fMoW arrays
are linked; BCN is deliberately not linked while its data defect is unresolved.
Canonical arrays are under `/home/dsi/michaer8/optloss-audit/data`.
dsisco01 uses older GPUs/fp16; dsisco02 Blackwell/bf16. Storage is shared NFS.
Server static-analysis dependency is isolated at
`/home/dsi/michaer8/optloss-reset-validation-deps-20260914` (`pyflakes==3.4.0`);
the shared training environment was not upgraded.

## Open user question

Are there untouched evaluation groups/splits on the three current datasets?
Until answered, do not describe rerunning the inspected splits as fresh
confirmatory evidence. Code cleanup and data-integrity checks can proceed.

The user's separate loss-research task owns `docs/research/RESEARCH_LEDGER.md`.
Its shortlist is a proposal, not an approved algorithm change or launch protocol.

## Preservation

No old scientific result is promoted or erased by the reset. Folder titles have
no evidential meaning. Archive records must say original path, destination,
inventory/hash verification, and restore procedure. Keep this state concise;
put completed audit receipts in `docs/audits/`, not a growing resume narrative.
