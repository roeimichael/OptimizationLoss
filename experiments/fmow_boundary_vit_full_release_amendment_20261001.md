# ViT-B/16 full-block release compatibility, 1 October 2026

The fixed `local_boundary_vit_v2` pilot seed 6600 was run under immutable release
`0f12bde246ccfabc64d549046550e9bd44cfda9a`. Its step-on and step-off
artifacts, numerical preflights, and saved label-blind gate are preserved.
Subsequent independent scorer and queue corrections require a second immutable
release for the already fixed full seeds 6601–6612. This amendment does not
change the ViT model, pretrained weights, data, transforms, configs, five arms,
caps, seeds, metrics, or 24 GPU-hour ceiling. It authorizes no additional
experiment or use of development labels to select settings.

The full queue must identify **both** immutable commits. Before it can claim a
seed, it verifies the pilot checkout is pinned and clean, and compares the exact
bytes and filenames of all `tralo/*.py` sources, all 14 v2 config JSON files,
and both synthetic-memory and real-image preflight generators between the pilot
and full releases. The full release may change the independent scorer, queue,
tests, and documentation only. A mismatch, missing file, changed saved gate,
or modified release refuses dispatch and leaves evidence in the attempted
root. The cross-release identity receipt and pilot/full commit IDs are checked
again before every seed claim. The independent scorer also audits these bytes.

The cost projection charges measured pilot seed duration and both pilot
invocations' preflight time to the **pilot** release, and the fresh full-queue
preflight time and projected 12 seeds to the **full** release. It independently
rechecks the original pilot gate exactly, including the pilot commit, then
requires the existing 0.5 GPU-hour reserve and cumulative projection to remain
below 24 GPU-hours. All source, config, data, split, checkpoint, log, and
artifact provenance gates from the v2 protocol remain in force. A successful
compatibility check establishes that the full jobs execute the same scientific
procedure; it does not imply any efficacy result.

The old seed 6500 failures and seed 6600 pilots must never be rerun, resumed,
overwritten, or scored on reserved countries. The full block is still
6601–6612 exactly once in new roots on exclusively available cards, followed
by independent complete per-seed scoring. Any interruption requires inspection
of receipts, ownership, PIDs, and completed/pending seeds before action.
