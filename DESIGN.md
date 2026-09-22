# Small rebuild: design and implementation plan

## Contract

The user approved a fresh, readable implementation beside the old repository.
This is a replacement design, not behavior-preserving refactoring. Old modules,
experiments, documentation, and tests are not carried into the runtime tree.
Historical artifacts remain recoverable outside it.

## First deliverable

A CPU-only diagnostic path: explicit probability matrix and group IDs -> named
allocation rule -> predictions -> independently calculated metrics and JSON logs.
No model training, dataset download, parameter search, or new scientific campaign.
Start with standard-library Python. Add numerical/training libraries when needed,
not a generic plugin framework or a custom experiment scheduler.

Inputs specify finite nonnegative probabilities with each row summing to one
(absolute tolerance 1e-6, no implicit renormalization),
unique sample IDs, integer class indices, group IDs, global caps, and local caps.
Caps are supplied integers, not inferred from evaluation labels. An uncapped
class is represented explicitly. Ties use sample ID then class index, so input
row order is not an accidental scientific choice.

Allocation has no labels. Metrics receive labels separately. Raw argmax and
allocated predictions are saved separately. No metric chooses a checkpoint or
allocator. Macro-F1 includes every declared class; zero denominator means zero.
Constrained-class F1 averages only the explicitly declared constrained classes.

The scientific choice between upper-limit correction and filling constrained
slots remains open for the user. Diagnostic implementations must be named and
cannot silently establish the research baseline. Greedy failure does not prove
global infeasibility. Exhaustive enumeration on tiny fixtures is an independent
test oracle, not a production solver or a claim about which objective is best.

## Logging contract

One exclusive run directory per invocation. Never overwrite an existing run.
Append JSONL events with increasing sequence numbers; finite JSON values only.
Log supplied plain values, never callbacks, model objects, or data iterators.
Logger must not draw random numbers or perform inference. A write failure fails
the run; a success event is written only after artifacts are saved and hashed.
Record source/config/input hashes, timestamps, named policy, counts, violations,
raw/deployed predictions and metrics. Record failures without marking success.
No restart/resume mechanism in the first slice. Preserve interrupted directories.

## Modules and acceptance checks

1. `tralo/metrics.py`: confusion counts, per-class precision/recall/F1, fixed-class
   macro-F1 and cc-F1. Hand examples include absent classes and invalid labels.
2. `tralo/events.py`: exclusive JSONL writer; tests cover nonfinite values,
   immutable event snapshots, failed writes, sequence, and random-state neutrality.
3. `tralo/allocation.py` (pending the policy decision): explicit inputs and named allocation behavior. Tests
   cover competing classes, local/global caps, zero caps, ties and failure cases.
4. `tralo/inspect_predictions.py` implements the independent inspection slice:
   strict JSON input, supplied predictions, separate raw/deployed metrics. It
   audits feasibility and scores but does not verify the named allocation policy.
   The input embeds the quota/policy configuration, hashed separately as config.
   Source identity uses hashes of all package modules, not a claim that an old
   Git release produced these predictions. Tests recompute artifacts and reject reruns.
5. Independent review checks the specification and actual paths, not just green
   tests. Run the example and preserve an audit receipt outside source.

Implementation sequence: write hand-checkable tests; observe failure; implement
the smallest function; run tests; inspect artifacts; review; commit this slice.

## Subsequent milestones (not implemented in the first slice)

1. Explicit dataset manifest: sample identity, split overlap, grouping, externally
   supplied quota provenance. No inherited quota construction.
2. One backbone and supervised training: explicit optimizer lifetime, local RNGs,
   a logging-on/off parity check, no evaluation pass through the training loader.
3. One-step constraint objective and derivative checks, then alternating TraLO.
   The null is the same engine with the intervention disabled.
4. ALM, Fioretto and Hounie as small method modules with their own mathematical
   specifications. Share plumbing, not an incorrect forced update schedule.
5. Only after validation: the professor's three research directions. Their
   definitions have not yet been supplied; do not invent them.

## Cleanup boundaries

Local new tree is clean; old evidence is externally archived and hashed.
Server cleanup requires live process ownership, symlink targets, disk usage and
an inventory before any move. Gateway timeout means no permission to assume idle.
Cleanup organizes files; it is not a reason to erase research evidence.
