# TraLO research ledger

## Ownership and authority

Research task: `01a09ed4-aa90-7971-b431-a47ee02a1cd3` (Research Trello loss improvements).
Cleanup, core validation, release and dispatch: `01a09b18-ed8e-7452-8b07-a7ed5f858696` (Check project SSH access).

Both tasks confirmed this division on 2026-09-14. Research owns this directory and isolated CPU diagnostics. The cleanup task owns live source edits, tests, server sync, and launch. No algorithm candidate has been implemented in production; no new GPU job was dispatched here. Keep current state here short. The [report](TRALO_LOSS_RESEARCH.md) contains derivations, literature and experiment specifications. FRAMEWORK/MISSION/REJECTED retain protocol authority.

## Search and promotion rules

1. Search the current rejected ledger, archived intervention record, source/config keys and probe implementations before naming an idea new. Use mechanism synonyms, not only arm names. "No receipt located" is not "never tested."
2. Record objective, derivative, optimizer delivery, expected actual allocation change, matched controls, extra data access, cost and falsification before a campaign. An unexplained positive score does not retroactively validate the mechanism.
3. Retain statuses **supported**, **unfavorable in tested setting**, **inconclusive**, **invalid comparison**, or **not tested**, and state their scope. A bug can be supported on a fixture while its real-data impact remains unknown.
4. Before rerunning an old idea, identify a material mechanism difference or a specific invalidity in its previous experiment. Evidence reset alone is not a reason to repeat every sweep.
5. Do not promote loss reduction, applied-step counts, differing hashes, or constraint satisfaction to quality evidence. Use deployed cc-F1, collateral class metrics, paired uncertainty and feasibility.
6. Research diagnostics may use training labels and prespecified development analysis. Individual evaluation labels cannot enter gradients, anchor selection or hyperparameter choice. Request a protocol decision for changed data access, held-out evaluation, scientific question or compute budget.
7. Share shortlist, proposed file ownership, diagnostic findings and proposed dispatch with the cleanup owner before overlapping edits. Do not write dynamic logs into active instructions or overwrite the other task's working tree.
8. After an execution, add its identity/receipt and disposition. Preserve failed and negative evidence. A ledger does not update itself: the task reading a result owns the corresponding update and message.

## Candidate inventory

| ID | Mechanism and historical overlap | Present status | Promotion or revisit condition |
|---|---|---|---|
| R01 | Shapes, rho, scalar lambda, item scaling, finer groups; many archived settings | Historical unfavorable/inconclusive evidence, not fresh validation | New causal mechanism beyond moving the same aggregate proxy; no default sweep |
| R02 | Raw CE-gradient orthogonalization, head-only, separate Adam, direct SGD; all recorded | Existing ideas; some historical comparisons confounded by restore or dose | Real update attribution and matched delivered displacement before renewed quality experiment |
| R03 | Uniform/margin/cut-window counts; code exists | Not new; **uniform chunk inconsistency diagnosed and locally repaired** | Post-fix full-set weight parity and actual trainer tests pass; remote release validation remains |
| R04 | Snapshot averaging and focal CE | Existing general-training ideas | Match both controls and treatment; do not claim constraint novelty |
| R05 | Delivered-update protection | Precursor explicitly recorded around archived FRAMEWORK line 6661; specific joint finite-step design not located | **Not tested** on real data; first audit same-state task/constraint/Adam displacement; then design review |
| R06 | Signed-residual PI controller | Ratchet/proportional controls extensively discussed; nuPI-specific execution not located | **Not tested**; require temporal multi-scope instability and live relative effect under normalization |
| R07 | KL-constrained posterior targets | Posterior regularization cited; SLA already discovered in z61 | **Not tested** here; substantive extension/comparator, not immediate default; no novelty claim |
| R08 | Budget-content permutation | Implemented, archived z62 described staged/unrun at that time | Current fresh execution **not tested**; must preserve deployment caps and account for group-size confounding |
| R09 | Per-group allocation-boundary diagnostics | Many old global-cut substitutions documented | Existing diagnostic direction; rerun on prespecified development data and actual allocator, not global top-K |
| R10 | Label-proportion/LLP-DC ideas | Recent 2026 related work added to report | Literature comparison only; exact proportions are not arbitrary upper caps |

## New evidence

| Receipt | Claim | Status and limits |
|---|---|---|
| `receipts/20260914T073749818489Z/probe.json` | Epsilon-aware penalty derivative matches autograd and finite differences | **Supported** on five CPU float64 fixtures |
| Same | Default sum count full/chunk gradient equality | **Supported** on one heterogeneous fixture; broader release gate remains |
| Same | Optional uniform count changes gradient with chunks | **Supported**; max difference `0.11287593`; reported to cleanup owner |
| Same | Orthogonal raw gradient can yield task-increasing Adam displacement | **Supported** as constructed counterexample; not a measured real-data rate |
| Same | Zero-current-gradient Adam differs from skip and carries historical drift | **Supported** in counterexample; distinguishes attribution controls |
| Same plus `tralo/constraint_events.jsonl` | Logged pre-step count/objective reconstruction | **Supported** on two synthetic trainer epochs; maximum loss reconstruction error `1.5e-9` |
| None | Candidate improves deployed cc-F1 | **Not tested**; no claim of improvement |

Repair follow-up: `receipts/20260914T074403033401Z/probe.json` records the new helper's full-population-weight chunk parity (maximum difference `0`) and retains the per-chunk-mean negative control (`0.11287593`). The cleanup owner wired this weight through the trainer. Independent command `rtk proxy python -m pytest tests/test_constraint_step_audit.py -q -k uniform` returned **2 passed, 9 deselected**, exit 0. Source hashes are in the new receipt. This closes the local defect on the checked fixtures; full remote release/data gates remain with cleanup.

Coordination follow-up: cleanup reports **all 11 audit tests passing**, plus independent review of default-helper equivalence to the old source and uneven-chunk parity in float32/float64, including saturation. These broader checks are reported by the cleanup task, not rerun here. Full-suite verification and its commit remain pending at this update. Cleanup is adding the MISSION pointer and owns the Git index while staging; no research commit is planned concurrently. Subsequent docstring edits change file hashes, so use the final frozen release identity for future replay. Cleanup also flagged a misleading ALM module header; derive its update from the actual function before making literature-correspondence claims.

The earlier timestamped partial receipt directory records a failed diagnostic construction (`[]` instead of `None` for absent global constraints). It contains partial synthetic logs, not a successful audit or classification experiment. Keep it recoverable.

## Next bounded execution

Await cleanup's frozen source and data validation. Select a small, prespecified checkpoint schedule and a fixed training-only anchor; capture model **and optimizer** state. Measure task and constraint gradients and actual Adam displacements for skip, zero-gradient and live-gradient counterfactuals. Price memory and time. Use untouched evaluation only if its availability is established; otherwise label development and final inspected-split results exploratory.

Promote R05 only if that audit establishes the premise. The report specifies a one-sided projection with task and constraint directional checks, a norm cap, finite-step checks, and explicit optimizer-state handling. This is an implementation proposal awaiting design review, not permission to launch. Preserve the two-GPU initial limit, both-host process check, SHA-256 release parity, and heartbeat requirements from FRAMEWORK.

## Search coverage

Searched current and archived rejections, archived FRAMEWORK intervention headings and terms `proximal`, `mirror`, `posterior`, `trust region`, `Fisher`, `natural gradient`, `nuPI`, `extragradient`, `transport`, `projection`, `cone`, `Adam`; inspected live loss/trainer/step/count/allocator code and existing probe implementations. Historical remote receipts were not independently rescored for this report.

Primary-source searches covered multiplier control, gradient surgery, trust-region constrained optimization, proxy-Lagrangians, posterior regularization, constrained CNNs, Sinkhorn allocation, and label-proportion identifiability/recent hard assignments. Ten sources are cited in the report. Stop broad searches now; the next decision depends on real-state mechanism evidence, not another list of papers.

## Git packaging update, 2026-09-14

Git hygiene is coordinated with the cleanup task. Research prose and the reproducible CPU probe remain versioned; timestamped receipts stay local and ignored. The pre-hygiene cleanup snapshot is preserved by a local backup branch and bundle; see docs/GIT_TRACKING.md. A new Git commit changes packaging identity, so server parity must be re-established against the final committed snapshot before any campaign. The earlier verification notes above describe their specific snapshots, not a launch-ready release.
