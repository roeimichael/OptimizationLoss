# Checkpoint record template

Keep one concise record per coherent change/run in the project's maintained
experiment notes. Receipts and bulky logs live outside source, linked by SHA.

- **Question and change:** what behavior changed, and why this is needed.
- **Definition:** objective/algorithm, mathematical expected result, alternatives
  and assumptions; what is intentionally not established by this test.
- **Identity:** branch, full commit, GitHub push, server release, tracked source
  hashes, config hash, data/split identities, parent/failed attempt if any.
- **Execution:** exact command, interpreter, cwd, environment keys, host, physical
  GPU UUID/index, ownership check time, architecture, precision and effective batch.
- **Data:** source/license/access, sample IDs, split/duplicate checks, class support,
  quota provenance and any evaluation exposure. No labels used for allocation.
- **Validation:** relevant local and remote tests, independent expected values,
  actual outputs, failed/skipped checks and numeric tolerances with rationale.
- **Results:** raw and allocated metrics, feasibility, quota counts, changed
  predictions, training dose/finite checks, elapsed time and output hashes.
- **Interpretation:** what passed, what remains uncertain, next decision. A single
  short pilot is not a superiority claim or proof of dataset difficulty.
- **How to reproduce:** checkout SHA, install/identify dependencies, supply the
  recorded configuration, choose a newly verified free GPU and a new output path.

Never silently edit the record of a failed attempt. Record the fix as a new
checkpoint and retain original logs. Updating an experiment setting is legitimate;
present it as a changed condition, not a continuation of identical evidence.
