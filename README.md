# TraLO rebuild

A new implementation for auditing classification under global and group-level
prediction quotas. No legacy code is imported. This branch is a development
workspace, not a validated replacement for historical experiments.

Start with [DESIGN.md](DESIGN.md). It defines the first small deliverable and the
order for adding training and methods. Nothing here establishes a research win.
No primary research metric or training recipe has been chosen. Existing metric
functions are diagnostics with explicit, revisable definitions. Tests check
those definitions, not whether F1, Adam, or a historical recipe should be used.

## Read and run

Requires Python 3.10 or later; this first slice has no third-party dependencies.

```powershell
python -m unittest discover -s tests -v
python -m tralo.inspect_predictions examples/hand_predictions.json runs/hand-example
```

The second command **inspects predictions supplied in the example**. It does not
train a model or run an allocator. A second invocation with the same output
directory is refused. The hand example has raw accuracy 2/3, raw class-0 F1 2/3,
and supplied corrected accuracy/class-0 F1 of 1. These are illustrative values,
not experimental results.

Read these four modules in order:

1. `tralo/metrics.py`: definitions and hand-checkable confusion counts.
2. `tralo/quotas.py`: count predictions and check global/local upper bounds.
3. `tralo/events.py`: plain-value JSON logging with no training access.
4. `tralo/inspect_predictions.py`: validation, identities, artifacts and CLI.

Outputs are `input.json` (exact input bytes), `report.json` (raw and allocated
metrics, counts and sample predictions), and `events.jsonl` (start/completion
with source/input/report hashes). Success means the supplied predictions were
audited, not that their named allocation policy has been verified.

**Next:** settle Clipper's allocation contract, implement it with independent
tiny cases, then add the first supervised training path. Logging neutrality
currently covers Python RNG state; model/optimizer/Torch parity requires the
training milestone and is not claimed yet. Server cleanup remains pending access.

## Evidence preservation

The predecessor is Git commit `7a5b55b555b8ccaa946349e4e77da733ef5b0715`.
The original checkout and research outputs remain untouched. Inherited files were
moved outside this worktree to
`C:/Users/roeym/.codex/rebuild-audit-20260922/legacy-worktree`.
All 411 files were checked against SHA-256 hashes in the adjacent
`legacy_inventory.json`. This is preservation, not a runtime dependency.

Both server connections failed at the DSI gateway on 2026-09-22. Server process
state is unknown; no remote cleanup, deployment, or training has been performed.
