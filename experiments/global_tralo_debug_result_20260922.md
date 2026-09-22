# Why the first global TraLO run collapsed

**A concrete optimizer-history failure was reproduced.** A large constraint
gradient entered the same Adam state used by supervised training. Its stored
first moment then made a subsequent supervised step increase its own loss and
further suppress constrained predictions. This is not a reversed derivative or
a demonstrated PyTorch bug; it is a harmful interaction in the training recipe.

The earlier result should be read as a failure of that recipe, not as a clean
test showing the constraint idea itself cannot help. Carrying the reference
rho target 100 into a five-epoch phase was not an established safe basic setting.

## Exact reproduction before interventions

Diagnostic commit `3a70ba52e8ef1399770132f2ddae620daf6d259f`.
The registered [debug protocol](global_tralo_debug_protocol.md) specifies one
seed 701, not a search for winners. Same dataset bytes, frozen features, head
initialization, batches, hyperparameters and FP32 host. The instrumented replay
produced **bitwise-identical final probabilities** to the original run.
The observer also passed a separate training-parity regression test.

The loss formula and analytic derivative were checked independently. The full
suite (54 tests) passed locally and on both DSI hosts. This does not prove there
are no other bugs, but the collapse is reproducible without a reporting error.

## The decisive one-step test

At the start of epoch 10, the current supervised gradient norm was 1.284, while
Adam's stored first-moment norm was 19.118. The preceding constraint gradient
norm had been 192.421. Cloning the exact same model and current supervised
gradient produced these results on the same training minibatch:

| Adam state used for this one step | Training loss before | Training loss after | Constrained predictions before → after |
|---|---:|---:|---:|
| Original state | 1.248982 | **1.288651 (worse)** | 64 → 43 |
| Only stored first moment set to zero | 1.248982 | **1.241068 (better)** | 64 → 64 |
| Fresh Adam state | 1.248982 | **1.100242 (better)** | 64 → 77 |

No model weights, labels, current gradient or learning rate changed between
these clones. The second row preserves Adam's second moment and step counter.
It therefore isolates the stored first moment as the cause of this particular
uphill update. It does not isolate every contribution to the complete trajectory.

For a gradient g and actual parameter change d, `g·d > 0` predicts an increase
in the current loss to first order. The original update gave +0.03352; clearing
the first moment gave -0.007979. The measured loss changes confirm those signs.

In the unmodified replay, the first five supervised updates of epoch 10 reduced
constrained predictions `64 → 43 → 34 → 27 → 19 → 14`. All five update directions
had positive inner product with their current supervised gradients. The final
constraint loss was zero, so there was no final constraint step. Skipping a
zero-loss constraint step was insufficient: its earlier history remained in Adam.

## Two controlled full-trajectory diagnostics

Each row below changes only one part of the original recipe. No changes were
combined. All variants share warm-up and batch-order hashes. Values are for
seed 701 only; F1 uses the 0–100 scale.

| Training procedure | Raw total predictions across capped classes | Upper-bound accuracy | Upper-bound constrained F1 | Capped-first accuracy | Capped-first constrained F1 |
|---|---:|---:|---:|---:|---:|
| Original TraLO replay | 14 | 57.40% | 7.22 | 59.80% | 47.31 |
| Hold rho at 0.5; keep shared Adam | 140 | 60.45% | 49.01 | 60.75% | 53.49 |
| Keep original rho ramp; separate Adam states | 121 | 61.00% | 51.49 | 61.00% | 52.27 |
| Original matched TraLO-null reference | 168 | 60.95% | 52.83 | 60.95% | 52.83 |

The raw maximum is 100 combined slots, with individual caps 10/class. The two
diagnostics avoid the severe suppression but remain **raw-infeasible** and need
the same final allocator. Every allocated result satisfies all caps. Avoiding
collapse is not the same as satisfying raw constraints or outperforming the null.

Separate states means task updates keep their own persistent Adam moments and
constraint updates keep another persistent set, acting on the same model.
It is not a fresh optimizer at every step. The diagnostic restores each state
around the relevant update. Same objective, learning rate and opportunities;
actual active constraint dose changes from 4 to 5 as the trajectory changes.

Holding rho fixed removes the severe suppression; separating optimizer states
also removes it while retaining the original ramp. Together with the cloned
one-step experiment, this supports an interaction between constraint pressure
and shared optimizer history. It does not establish that one factor alone
explains every epoch or seed. Lambda still adapts in the fixed-rho diagnostic.

## What is established and what is not

- Established: the original result is exactly reproducible; the constraint
  intervention is active; stale first-moment history causes a specific harmful
  supervised update; two single-change interventions avert collapse in this seed.
- Not established: a general TraLO advantage, safe settings for every backbone,
  three-seed replication of either diagnostic, or absence of every possible bug.
- Next controlled comparison: implement separate optimizer ownership explicitly
  in the ordinary runner, verify state isolation, then repeat the matched small
  comparison. Keep rho fixed initially if that is chosen as a new recipe, but
  distinguish the two changes instead of attributing their combined effect to one.

No main training defaults were silently changed. The new observer and diagnostic
tool leave the original algorithm intact; the earlier evidence is preserved.

## Evidence

Independent scikit-learn recomputation agrees with all metrics within 1e-12;
artifact hashes and predicted counts were independently verified. Complete
trace includes soft counts, hard counts, gradient norms, Adam-state norms and
gradient/displacement products. The full raw and allocated reports and saved
probabilities are retained for all three trajectories.

Remote:
`/home/dsi/michaer8/tralo-rebuild/runs/debug-tralo-3a70ba52-20260922T140611Z/`.
Local:
`C:/Users/roeym/.codex/rebuild-audit-20260922/debug-tralo-run/`.
Summary: [numerical results](global_tralo_debug_result_20260922.json).
Tool: [debug_global_tralo.py](../tools/debug_global_tralo.py).
dsisco01 GPU0, Quadro RTX 6000/Turing, FP32 with TF32 off; other users' Blackwell
jobs were untouched. Original cached-feature cost is excluded from these runs.
