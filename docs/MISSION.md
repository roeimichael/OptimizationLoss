# TraLO reset: execution state

Updated 2026-09-14. User approved recoverable large cleanup, fresh evidence,
validated logging/data/code, then a within-TraLO modification and monitored SSH
experiments. No historical acceptance tally is carried forward.

## Current stage

**Recoverable reset and initial software/data audit done; launch gates remain open.**
New experiments have not launched. Old result trees are outside active paths.
Both SSH hosts were reachable and all GPUs idle at the latest check.
Recheck processes on both before any remote move or dispatch.

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
Git-hygiene packaging changes are being finalized by the user's other task.
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
