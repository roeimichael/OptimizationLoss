> HISTORICAL RECEIPT — superseded by the 2026-09-14 evidence reset. Old tables are not current acceptance evidence. See [the reset receipt](2026-09-14-reset.md).

# Repository and evidence audit — 2026-09-13

Status: discovery and bounded safeguard pass; the broader experimental overhaul
is incomplete. This is an audit receipt and implementation record;
`docs/FRAMEWORK.md` remains the protocol and `docs/MISSION.md` owns remote state.

## Latest preparation update — 20:55 Jerusalem time

The user requested fresh remote experiments and a multi-method/backbone metric
table with highlighted best means and an honest conclusion. No new training has
been launched. The matched all-arm snapshot design is awaiting user approval;
the current implementation averages only TraLO-family snapshots.

- Created `/home/dsi/michaer8/optloss-thesis-20260913` on the new branch
  `codex/thesis-20260913`, based on local HEAD `901de2de`. Imported the missing
  local Git objects by bundle without moving any existing remote branch or
  running automatic garbage collection. Older campaign worktrees are unchanged.
- Copied the current local training/config/scoring/test sources. SHA-256 checks
  on both machines matched **all 160 files** in `LOCAL_SOURCE_SHA256.json`.
  Manifest SHA-256:
  `38fceececd4bfb279c314cc2866cc09b00039d03e4459c4eca00eb7803a8c897`.
  This is a dirty preparation checkout, not an immutable campaign release yet.
  Windows CRLF bytes were preserved; the semantic diff matches the local patch,
  and `git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --check`
  passes. No claim is made that manuscript/archive files were mirrored.
- Linked all four non-empty NumPy arrays per active dataset directly to their
  resolved canonical files under `/home/dsi/michaer8/optloss-audit/data`.
- Fresh local suite: **658 passed, 1 skipped** (347.43 s). Fresh server suite,
  forced CPU: **656 passed, 2 skipped, 1 failed** (205.03 s). JUnit receipts are
  `.codex/thinning-baseline-2026-09-13.xml` and
  `.codex/server-synced-tests-2026-09-13.xml` locally; the server receipt is
  `/tmp/optloss-review-20260913-ZILZxX/server-synced-tests.xml`.
- The server failure reproduces in isolation:
  `python -m pytest tests/test_baseline_fidelity.py -q -k alpha_liveness_gate`.
  `_run_arm` hashes softmax values rounded to six decimal places. In a
  single-thread diagnostic, the old low-dose alpha sweep changes unrounded
  probabilities by at most **2.9802322387695312e-08 on both machines**. Windows
  has zero rounded differences; Linux has 1–2 elements crossing rounding bins.
  Thus rounded hash equality is not a stable numerical-tolerance test. At the
  shipped dose, measured differences are 1.31e-6–5.42e-6 instead. Local PyTorch
  is 2.5.1+cu121; server PyTorch is 2.11.0+cu128. This characterizes a harness
  defect, not evidence of a useful empirical effect or a dead production arm.
  No test was weakened or skipped to force a launch.
- At 20:52, dsisco02's four GPUs were idle and no owned dispatcher/runner/queue
  process was found. Recheck before dispatch; the project limit remains two
  GPUs, with every arm of a campaign on the same host/AMP regime.

Thinning census: **68,585 Python lines**, including **32,438 script lines**,
**24,149 test lines**, and **7,445 src lines**. The **77,660 dataset CSV rows**
are not executable code. Large historical prose and tightly coupled probe tests
need a provenance-backed retirement pass; none was bulk-deleted in this update.
The earlier sections below record the preceding audit phase, not the current
server-sync state.

## Scope and preservation boundary

The requested overhaul covers evidence, infrastructure, cleanup, reproducibility,
mathematical correctness, and fair benchmarking. A clean test suite establishes
software behavior; it cannot establish empirical superiority or novelty.

Starting checkout: `901de2de`, branch `cleanup/consolidate-pipeline`.
Pre-existing work: `scripts/deployed_h2h.py` modified; `.codex/` and `AGENTS.md`
untracked. Preserve these changes. Do not edit the professor's manuscript,
overwrite recorded predictions, remove provenance archives, or change a running
server's training code. Local synthetic regression tests are allowed; dataset
training belongs on the university cluster.

## Implementation sequence

- [x] Inspect protocol, current status entries, prior failure ledger, pipeline
  boundaries, dispatch/recovery code, configuration audits, and local artifacts.
- [x] Retry SSH after VPN activation; authenticate successfully to both compute hosts.
- [x] Establish the complete regression baseline and classify existing failures.
- [x] Repair demonstrated recovery and persistence defects with regression tests.
- [x] Correct entry-point guidance and archive the obsolete cleanup launcher.
- [ ] Audit shared training, data leakage protections, configuration identity,
  and statistical comparison boundaries; record what cannot be verified locally.
- [x] Verify the changed code and record the exact benchmark/reconnect gates.
- [x] Verify both hosts and gate three recently completed campaigns before scoring.
- [ ] Execute and evaluate a preregistered, fair benchmark when the evidence and
  infrastructure gates are satisfied.

## Findings and evidence classification

| ID | Finding | Evidence and status |
|---|---|---|
| A01 | SSH restored after VPN activation | Initial TCP timeout was superseded by successful authentication through dsihead to both hosts. Latest process/GPU observation: 2026-09-13 19:47 Jerusalem time; all eight GPUs idle, no owned dispatcher/runner/queue process. |
| A02 | No current local campaign results | Local `results/` has no entries. The two local evidence tarballs belong to the older experimental generation. Current acceptance counts are recorded findings, not recomputed in this audit. |
| A03 | The active architecture is already substantially shared | `TrainInputs`/`TrainOutputs`, common warm-up, runtime setup, constraint-step implementation, dual scaffolding, and evaluation are in use. A second repository would duplicate established contracts. |
| A04 | Documentation gives incompatible operational advice | README included KL and pointed to a separate rewrite; old FRAMEWORK headings named the wrong headline/only dataset. README corrected; FRAMEWORK and MISSION now begin with a dated current reading while preserving historical entries. |
| A05 | Recovery can overwrite a legitimate zero-accuracy result | Reproduced and fixed: completed status is protected independently; any non-null accuracy, including 0.0, is treated as a result. |
| A06 | Dispatcher hides unsuccessful execution from its caller | Reproduced and fixed: main returns failure for unsuccessful/blocked work, preserves interruption as 130, and the actual entry point uses sys.exit. Detached SIGINT handling restored. OS-level signal delivery remains a separate integration check. |
| A07 | Configuration persistence can truncate the only record | Reproduced and fixed using a same-directory temporary file, flush/fsync, and atomic replacement. Serialization/replacement failure preserves the old file and cleans up the temporary file. This does not provide cross-dispatcher locking or a directory-fsync durability guarantee. |
| A08 | Dead-code report omits the top-level consumer | Default scan now includes main.py; the reported print_status_summary candidate was live. No source deletion was justified by it. |
| A09 | Strict determinism can silently degrade | Reproduced and fixed: failure enabling deterministic algorithms now stops setup instead of warning and continuing. This is not a guarantee of bitwise equality across hardware/software versions. |
| A10 | Historical cleanup launcher is detached from its inputs | No live references found. Moved by git mv to docs/archive/cleanup_2026-09-06/run_autoclean.sh, with an archive banner and immediate refusal before any side effects. Verified as a rename, not loss of history; explicit invocation exits 1. |
| A11 | Dependency groups are conflated | `medmnist` has a live reader in the historical manuscript figure generator. It is not a current training dependency and must not simply be deleted. |
| A12 | Declared disjointness can pass without evidence | Missing metadata, missing group columns and null group values previously bypassed verification. Reproduced and fixed: a declared disjoint split must be verifiable; valid disjoint and permitted non-disjoint cases retain their behavior. |
| A13 | Numbered RNG controls can win the method ranking | Numbered reseeds failed the scorer's suffix filter even though the floor reader recognized them. Reproduced on snap3 and synthetic data; fixed with the canonical stream predicate. Controls remain available to estimate noise. |
| A14 | Restricted views remove their own noise floor | Review reproduced that --arms deleted unnamed RNG streams while promising the floor was unchanged. Fixed and regression-tested; every canonical floor stream is retained. Existing twelve-seed results explicitly included all streams and are numerically unchanged. |
| A15 | Seed-count banner and ranking use different populations | Review reproduced a four-seed banner before an actual twelve-seed ranking when only controls were thin. Fixed: both use competitors plus the control. A genuinely thin competitor still limits the reported count. |
| A16 | Existing Windows shell test chose WSL bash | Baseline failure was caused by system32/bash.exe receiving Windows paths. The test now selects Git Bash and obtains its native pwd form. The guard itself was not relaxed. |

## Scientific interpretation

Recorded acceptance, from the latest inspected framework/mission entries: 8/27
testable cells against clip plus rival duals; 7/27 when focal_clip is also required;
6/18 strict-task cells on the former definition; 2/11 units; priced record 1 win,
2 losses. These are distinct denominators and must travel with their definitions.
They have not been recomputed over the complete corpus in this audit. In
particular, the completed D2 seed extension changes its mean ordering, so the
old global tally must not be presented as freshly verified or updated by hand.

The snapshot finding in FRAMEWORK 2(z115) is promising as a variance reduction
method, but its lambda-zero twin receives the same gain. It does not establish a
constraint-specific improvement. The same entry calls for snapshotting every arm
before a fair head-to-head. No new count-function, penalty-shape, scope-weighting,
or dose variant is licensed merely by this cleanup request.

The manuscript's historical corpus and current experiments are different
generations. Historical figures, raw archives, and data generators remain evidence
even when their datasets cannot support a new experiment. A tracked or untracked
artifact is not disposable merely because it is old.

## Validation receipts

- Baseline: 639 passed, 1 failed, 1 skipped, 7 warnings, 365.68 seconds.
  JUnit receipt: `.codex/audit-baseline-2026-09-13.xml`. The failed shell test
  is A16, not an observed training failure.
- Targeted regression/dispatcher/data/baseline run: 127 passed, 3 warnings,
  47.81 seconds before the two review-derived cases were added.
- Intermediate full run: 656 passed, 1 skipped, 7 warnings, 444.94 seconds.
  This preceded the final two review-derived cases; it is not the final receipt.
- All 18 new regression cases passed in 11.87 seconds.
- Final suite: **658 passed, 1 skipped, 7 warnings, 435.13 seconds** (659
  collected), after the two review-derived scorer corrections. JUnit receipt:
  `.codex/audit-final-2026-09-13.xml`. No mathematical or empirical superiority
  conclusion follows from this software result.
- Before-launch staged detector tests: 44 passed, 30 deselected, 22.05 seconds.
  This proves the detectors, not the readiness of an ungenerated campaign.
- `scripts.audit_config`: 3,456 generated configurations, no hallucinated keys;
  192 distinct warm-up IDs for 192 identities. Reported defaulted and tooling-only
  reads require interpretation; exit zero does not mean every reader is explicit.
- `scripts.doc_commands`: 161 invocations parse; six modules explicitly abstain
  because their arguments are built dynamically.
- `scripts.dead_code`: initially 979 definitions and one false-positive
  candidate; final scan includes main.py, finds 987 definitions and no
  unreferenced candidates. This is a static check, not blanket deletion authority.
- `git diff --check HEAD`: passed. Archive execution refuses before side
  effects. Curated audit logs are explicitly unignored so receipts survive a
  future commit; generic runtime logs remain ignored.

## Remote verification and re-scoring

SSH succeeded on both hosts after VPN activation. dsisco01 has four Quadro
RTX 6000 cards (23,040 MiB each); dsisco02 has four RTX PRO 6000 Blackwell cards
(97,887 MiB each). Both share NFS. Neither Slurm squeue nor PBS qstat was found;
the observed pipeline uses main.py and queue_runner.sh. An authenticated shell
and idle cards do not prove automatic dispatch/recovery end to end.

The remote main tree is e93f36ec, behind the local 901de2de starting checkout;
campaign trees remain pinned at their own commits. All 26 worktree entries,
remote modifications and untracked paper artifacts were preserved. Scoring used
an isolated temporary export, never an in-place campaign code update.

Three completed campaigns passed run_campaign's score step (results detector,
parity instrument, landed-dose instrument) and pred_integrity. Training logs
were checked for dose/collapse/finite values before scoring. Several post-hoc
runs used shared warm-up caches and had no separate training log; full per-run
log coverage is not claimed. Prediction MD5 differences were checked first and
are one-sided evidence only, not proof of a live loss mechanism.

| Campaign | Completed | Runner version prefix | Audit observation |
|---|---:|---|---|
| snap3 | 156 | 34d005f83650 | Trained arms at 29/29 dose; RNG controls at zero constraint dose |
| clipsweep2 | 88 | 37f842c97be8 | Two caps, four seeds; sweep effects inside the measured noise floor |
| bcn1vitseed | 144 | 4b980ca08cff | Verified extension of bcn1vit; L80/L90 main-arm seeds pool to 1–12 |

### Descriptive results, not significance claims

| Cell | Seeds | TraLO variant vs clip, items | Variant vs focal_clip, items | Variant vs its own null, items |
|---|---:|---:|---:|---:|
| bcn / MobileNetV2 / L90 / snapshot | 4 | +10.50 | +9.75 | -4.50 |
| bcn / MobileNetV2 / L95 / snapshot | 4 | +10.75 | +10.50 | -1.75 |
| bcn / MobileNetV2 / L100 / snapshot | 4 | +12.25 | +9.75 | +1.00 |
| bcn / ViTB16 / L90 / TraLO | 12 | +10.83 | -11.08 | +11.83 |

At the ViT L90 task cell, focal_clip's +21.92 items exceeds TraLO's +10.83;
fioretto also exceeds it at +12.17. Corresponding cc-F1 gains over clip are
+0.01170, +0.00555 and +0.00633. D2's old four-seed +33.00 result therefore
does not represent the completed extension. The same dataset's MobileNetV2
stock-TraLO result remains negative: -22.75 items at L90 in snap3, consistent
with the earlier F1 loss. Neither dataset-wide success nor failure follows
from choosing one backbone.

ViT L80 is non-task; L70 is non-task and has only two seeds. They are not extra
successful task cells. The snapshot and clip-sweep scorers name no clear #1.
Snapshot/non-snapshot floors differ; the pooled floor and its sqrt(n) reading
are diagnostics, not independent inferential observations. No bootstrap,
multiple-testing correction, or new unit-level significance claim was made.

See [the receipts](2026-09-13/README.md) for raw paired differences, official
cc-F1, actual scorer output, environment freeze and exact scorer patch. Final
scorer reruns produced JSON rows identical to all three corrected receipts.

### Dataset consistency

On canonical server arrays, metadata labels exactly match label arrays and
train/test image-row counts match metadata on all three active datasets.
No train/test group or filename overlaps were found; no metadata nulls were
present. Train/test rows: iwildcam 20,000/2,943; bcn 8,270/3,900; fmow
15,386/4,168. Test groups: 7/8/13 respectively. iwildcam training is exactly
2,500 per class, confirming the balanced-prior premise of the inert-baseline
findings. This check did not hash image contents or establish patient/source/
pretraining independence. See dataset-integrity.json for hashes and counts.

## Remaining work and launch boundary

1. Recompute the complete acceptance ledger using verified seed extensions and
   the corrected competitor definition; preserve old figures beside new ones.
2. Before new training, decide and preregister the matched all-arm snapshot
   comparison called for by FRAMEWORK 2(z115), including inference/storage
   costs and identical post-hoc selection. The current measurements do not
   identify a new constraint-specific mechanism worth an expanded campaign.
3. Validate dispatch interruption/restart and exclusive cross-host run claiming
   in an isolated CPU fixture. Atomic JSON replacement does not prevent two
   dispatchers claiming the same run. No production job was submitted to test it.
4. Separate historical-paper and training dependencies; build/test per-host
   reproducible environment profiles instead of treating pip freeze as a lock.
5. Complete image-level/source leakage and baseline-fidelity audits. Existing
   gradient invariance tests do not establish novelty or blanket mathematical
   correctness, and host/AMP differences prohibit silently pooling campaigns.
6. Broader cleanup must use a provenance manifest. No checkpoints, predictions,
   tarballs, branches or remote worktrees were deleted. All local topic branches
   were reachable from HEAD, but archival references may still be needed.

No commit, push, manuscript edit, data rewrite, new GPU campaign or destructive
artifact purge was performed. Source fixes are local and uncommitted; deploy
them only in a fresh, gated campaign environment after review.
