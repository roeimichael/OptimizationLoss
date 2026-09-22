# Rebuild work queue

## Checkpoint rule

For each small coherent change: state the behavior/assumption -> write an
independent check -> implement -> run relevant regressions -> inspect actual
artifacts -> commit -> push GitHub and DSI -> verify the deployed commit and
source bytes -> run remote checks -> record pass/fail and limits of the claim.
Do not commit every keystroke or count passing tests as scientific validation.
Failed checkpoints remain visible; do not overwrite results or claim completion.

DSI mirror: `/home/dsi/michaer8/tralo-rebuild.git`.
Releases: `/home/dsi/michaer8/tralo-rebuild/releases/<commit>`.
Verification receipts belong outside source, under
`/home/dsi/michaer8/tralo-rebuild/verification/<commit>/<host>/<attempt>`.
Both hosts share files, not GPUs. Check actual ownership before GPU use.
Never update old research checkouts or a release in place.

## Tasks in dependency order

- [x] Create separate branch with no legacy runtime imports.
- [x] Implement independent metrics, explicit quota auditing and plain-value logger.
- [x] Test hand-calculated cases, sample/class permutations, invalid input,
  exclusive outputs and logging neutrality for Python random state.
- [x] Establish remote checkpoint: matching commit/source on both hosts, CPU
  regressions and real CLI example; GPU arithmetic/logging smoke on a free card.
- [ ] Agree Clipper's scientific definition: upper-bound correction versus
  preferentially filling slots; score/objective; global/local interaction;
  deterministic ties; behavior when greedy allocation cannot complete.
- [ ] Implement the selected named allocator without evaluation labels. Validate
  tiny hand examples and brute-force feasible assignments independently. If
  greedy, report objective gaps without pretending it is exact. Validate quota
  feasibility separately from prediction quality and optimality.
- [ ] Save probabilities, raw labels, allocated labels, counts and metrics from
  the same inference result. Verify a report can be recomputed from those files.
- [ ] Specify dataset manifest: IDs, split separation, grouping, quota provenance
  and permitted aggregate information. Check actual arrays; do not import an old
  loader or infer validity from directory names.
- [ ] Implement one supervised baseline: explicit model, initialization, loss,
  learning rate, epochs, optimizer lifetime and independent RNG ownership.
  Justify settings; no inherited training recipe is mandatory.
- [ ] Test logging on/off and frequency changes: exact same batches, model and
  optimizer state, raw predictions. Include train/eval mode and tensor mutation.
- [ ] Verify planned/attempted/applied/skipped updates and nonfinite failures;
  record precision, host, source, data and config identities. Add checkpoint
  resume only with optimizer/RNG/data-order restoration tests.
- [ ] Specify TraLO objective and controller, derive/finite-difference gradients,
  inspect one real update, then add alternating training. Null uses the same
  engine with its intervention off; verify this property, not just arm names.
- [ ] Add ALM, Fioretto and Hounie one at a time with method-specific mathematical
  contracts and controls. A common interface must not change their algorithms.
- [ ] Choose evaluation criteria and uncertainty from the research question;
  register these before outcomes. No metric is globally privileged.
- [ ] Record the professor's three proposed directions before implementing them;
  their definitions have not yet been supplied in this conversation.
- [ ] Only then launch a small validated research campaign; inspect early runs
  before expansion, with explicit compute/precision and evidence boundaries.

## What current tests establish

Checkpoint `310cade` (2026-09-22): all 24 regressions and the CLI example passed
on both hosts with matching tracked-file hashes. On dsisco01 GPU0, float32 mean
cross-entropy equaled ln(2), gradients matched +/-1/4, the first Adam step matched
its bias-corrected formula, and event logging preserved Torch CPU/CUDA RNG,
parameter, gradient and optimizer state. These are analytic fixtures, not a
chosen training recipe. dsisco02 GPUs belonged to another user and were untouched.
Receipts: `/home/dsi/michaer8/tralo-rebuild/verification/310cadeaca29125c5d3b6b1f412570ef59ff2212/`.
The five-minute heartbeat was deleted at the user's request; no scheduled
monitoring remains for this task.

Checkpoint `7b96b90` (2026-09-22): 24 regression tests and the example passed on
both hosts with matching source hashes. The first GPU smoke failed while logging
PyTorch's version, which is a string subclass rather than an exact plain string.
The fix explicitly converts framework metadata to strings; logger validation is
not weakened. Failed receipts remain under that commit's verification directory.

`inspect_predictions` audits user-supplied corrected predictions. It does not
generate them, train a model, or verify the named allocation rule. GPU smoke checks
small arithmetic and logger behavior, not a scientific result or real allocator.
The numerical values in that smoke are analytic fixtures, not research settings.
