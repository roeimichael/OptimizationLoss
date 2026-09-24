# Rebuild work queue

## Completed: 384-fit strength sweep, 24 September

Runtime bf00a7fbfe7c2cdf8e27b53e51b37992e82386ca, unchanged method math.
All384 fits passed across knee/CIFAR, seeds1001-1012,7TraLO rates,7ALM rhos,
Clipper/common-null. No positive Holm-adjusted primary contrast (family28).
Earlier .0001 knee optimum did not reproduce; weak/moderate range near null,
stronger settings can harm. See experiments/constraint_sweep_result_20260924.md,
JSON, seedCSV and plot. All run artifacts retained remotely and in local archive.
No sweep queues remain active. Do not duplicate or silently extend the grid.
Next informative direction: separate objective from update schedule; frozen-head
sweep does not answer end-to-end constrained backbone learning.


## Active: two-dataset ALM comparison, 24 September

User confirmed knee + CIFAR-100, four seeds, Clipper/TraLO/TraLO Null/ALM/ALM Null.
See experiments/alm_two_dataset_20260924.md for the fixed design.
ALM integration deployed as 6411f2b2db81cb7b071a09441cd26d7f82347e98; 90 tests
passed locally and natively on both hosts, source parity and Quadro smoke passed.
All 48 fits completed: knee/CIFAR, 4 seeds, 5 arms plus TraLO .001 sensitivity.
Artifact/dose/metrics/null/quotas audits passed. Results and paired intervals:
experiments/alm_two_dataset_result_20260924.md/json. No reliable TraLO-over-null
advantage; initial ALM harms knee cc-F1. Do not launch duplicate runs.
Old overnight automation is paused. Do not duplicate the completed traces.


## Completed overnight diagnosis, 23–24 September

User requests per-image mid-training mechanism analysis through the night.
See experiments/knee_trace_protocol_20260923.md. Instrumented replay saves
every task minibatch endpoint and both sides of constraint updates. 85 local
tests pass, including observer state neutrality and exact training parity.
Deploy and verify before pilot; do not infer native GPU parity from CPU tests.
At 20:04 UTC Blackwell GPUs were occupied by others and Quadro was idle.
Use separately labeled same-host references; historical equality is a separate
recorded check. Morning report due 08:00 Israel, 24 September. New overnight
heartbeat is authorized; the earlier daytime heartbeat remains paused.

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

Collapse debug checkpoint `3a70ba52`:54 tests passed; seed701 exactly replayed.
With identical current parameters/gradient, clearing only Adam's first moment
changed an uphill supervised step into a loss-decreasing step. Fixed rho and
separate persistent optimizer states each prevented severe suppression in this
single-seed diagnostic. [Evidence](experiments/global_tralo_debug_result_20260922.md).
Next: explicit optimizer ownership and state-isolation tests before another
matched small comparison. Do not interpret this as established method superiority.

Global TraLO checkpoint `0c355ba4`: nine matched fits completed (three seeds,
Clipper/null/TraLO),53 tests passed on both servers, all400 supervised updates
per fit applied. [Results](experiments/global_tralo_result_20260922.md) show
excessive constrained-class suppression for this recipe, with no mean benefit
over the matched controls under either final allocation. This does not close
the global-only method family. Any follow-up should isolate controller strength
or shared Adam-state effects; do not tune multiple changes against these scores.

Global-only checkpoint `44f485e` (2026-09-22): 40 tests passed locally and on
both DSI hosts; tracked bytes match, and the dsisco01 CUDA smoke passed. Two
named greedy allocators are implemented as diagnostics while the research
allocation definition remains open. A separate enumeration audit of 500 tiny
problems found every output feasible, but confirmed neither greedy policy is
always optimal for sum of assigned probabilities. This is not an accuracy claim.
The CIFAR-100 pilot protocol is in `experiments/global_clipper_pilot.md`.

Next global-only steps:
- Select the intended allocation objective and upper-bound semantics; consider
  an exact assignment reference before interpreting a greedy baseline's losses.
- Inspect the real image pilot's data, updates, saved scores and independent
  metric recomputation before changing a method or increasing the budget.
- Add an explicit logging-frequency intervention test for full training.
- Establish an adequately trained baseline and justified real quota policy;
  synthetic pilot caps and frozen features are not the final research setting.
- Only then implement a mathematically specified global constraint loss and
  its matched no-constraint control. Group/local constraints remain deferred.

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


## 2026-09-22: achieved-budget evaluation and author-baseline replication

- DONE: four seeds701–704, three unchanged training arms, original and raw-TraLO-derived budgets, both allocators;60 verified evaluation rows. Results: experiments/achieved_counts_result_20260922.html. Original high-rho/shared-Adam failure remains unresolved by this evaluation-only change.
- DONE: data features fully regenerated (12,000 rows, exact parity);59 tests plus independent artifact/statistical review.
- ACTIVE: replicate Kassif/Singer predict-then-optimize baseline before TraLO. See experiments/yuval_clipper_replication.md.
- STAGED: official DermaMNIST28 candidate, checksum and schema verified; original splits preserved. No paper-specific training launched.
- REQUIRED: full paper PDF or experimental sections/tables; supplied HTM contains abstract only. Resolve backbone, preprocessing/native resolution, knee data identity, training schedule, quotas, allocator and target metrics. Author repository pending.
- NEXT: match raw baseline first, then paper allocator on identical probabilities; compare checkpoint/logits when author code arrives. Do not label guessed settings an exact replication.


## 2026-09-22 overnight scope supersedes DermaMNIST staging

User requests knee-only global TraLO/Clipper work through morning, one backbone, incremental sample-aware losses. See experiments/knee_overnight_plan.md for fixed stages, seeds, gates and data blocker. Automation knee-tralo-overnight-research is ACTIVE every20minutes until morning summary08:00Asia/Jerusalem2026-09-23, then pauses. DermaMNIST staged data remain archived/read-only; no further Derma experiments.

Prepared separate persistent constraint Adam and supervised margin/false-positive terms;76 local tests and independent math/integration review passed. These are not knee results. Exact paper dataset/backbone remain unknown; fallback clarification pending. Do not launch substitute data without resolving this. Server verification receipts must be checked before claiming deployment.


## 2026-09-23 daytime: fallback authorized
User explicitly authorizes similar medical data and requires results by17:00. Download of original Chen knee archive launched05:36UTC on dsisco02, PID1625027, /home/dsi/michaer8/tralo-rebuild/data/knee-chen-v1/download_launch.json. Check before retrying. Both hosts were idle. Continue dataset audit, minimal knee runner, deploy and matched4seed stages. Old pending-approval text is superseded; no waiting for exact Yuval release.

First36 knee fits COMPLETE; see experiments/knee_first_stage_20260923.md. Next fixed diagnostic: constraint_lr0.0001,base only, same901-904 seeds. No new data approval needed. All original source releases immutable.

Smallstep12fits COMPLETE, hashes/dose audited, baseline/null byte-identical. RawccF1mean37.24vsnull38.00; capped-first42.19vsnull41.08,paired95%CI delta[-0.74,+2.96]points. No established benefit. Next: experiments/knee_adaptation_protocol_20260923.md prespecifies5training-only epochs perseed then base smallstep comparisons. Newprepare optionalADAPTATION_SEED; deploybeforelaunch, pilot901gate then902-904.

2026-09-23 10:27Israel: All4adaptations+12adaptedhead fits COMPLETE and audited. See knee_adapted_result_20260923.md/json. No jobs remain from these campaigns. CappedccF1TraLO68.827vsnull68.694,pairedmean+.133CI[-.290,+.556]. Continue consolidate/audit60headfits+4backbones; no arbitrary extra sweeps. Scheduled17:00report remainsactive.
