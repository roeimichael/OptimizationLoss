# Code audit of the rebuild, 2026-09-25 (Claude, 38-agent workflow wf_1c6808f3-0cf)

Nine subsystem reviewers; every medium+ finding checked by two independent skeptics
(one by reading the full code path, one by EXECUTING a CPU check), with a tiebreak where
they split. Full machine-readable output: `code_audit_20260925.json`.

## The arithmetic is correct -- verified by execution, not by reading

| subsystem | how it was checked |
|---|---|
| metrics (accuracy, macro-F1, cc-F1) | 300 random cases vs sklearn, max diff 1e-12 |
| allocators (capped_first, upper_bound_correction) | 20k random cases incl. heavy ties: always feasible, exact fill, order-invariant |
| TraLO count loss + two-pass streamed gradient | vs full-batch autograd: 1e-12 (fp64), 6e-8 (fp32, BN+dropout, 7-sample chunks) |
| ALM | correct inequality PHR: penalty, dual sign, projection at 0 |
| cutoff / hard-pair ranking | analytic vs autograd, 300 random cases, <= 1.4e-9 |
| published statistics | every paired mean, t interval, Holm p and Bonferroni interval recomputed with scipy: exact |
| arm matching | init, batch order and warm-up hashes identical across arms; equal task updates |
| label hygiene | no development label reaches a gradient, a cap or a checkpoint |

## What is wrong is the EXPERIMENTAL DESIGN -- 10 confirmed findings

| # | severity | finding | consequence |
|---|---|---|---|
| D1 | high (x4 reviewers) | The constraint step uses its own Adam, one step per epoch. Adam is scale-invariant, so lambda, rho and violation depth cancel: each step is ~ constraint_lr x sign(g). | The lambda/rho controller never acted. Every "TraLO" arm since 23 Sept is a fixed-size nudge; the sweep's only real lever was constraint_lr. |
| D2 | high | ALM (and TraLO) constrain the SOFT count E[count] <= K. For a minority class soft >> hard, so meeting it drives the hard count far below K (ALM: 20-31 of 98 on knee). | ALM's -8.1 knee loss is this surrogate, not "fixed rho / short budget". |
| D3 | high | Knee 82/16 caps never bind on hard counts; the controller freezes at the first opportunity in every knee fit. | Knee TraLO is a frozen lambda=0.01 arm; only the fill policy differs between arms. |
| D4 | high | Frozen linear head on non-negative ResNet features: the count gradient is +lr on every W_c coordinate, so it can only re-rank items by feature L1 norm (label-blind). | ~530 frozen-head fits cannot test whether a count constraint improves WHICH items are selected. |
| D5 | medium->low | Shared-optimizer mode: the TraLO step replays ~90% task momentum the Null never gets. | Shared-mode TraLO-Null (22 Sept studies) is constraint + one extra task step. |
| D6 | medium | Frozen-head fits oscillate; final-epoch endpoint samples an arbitrary phase. | "Near null" in the sweep is lack of power, not an effect estimate. |
| D7 | low-medium | knee_data.audit stores test labels in every manifest; docs quote the test grade-3 count (223). | Aggregate test-label information was read. Planned confirmation caps would be built on it. |

Three findings were refuted (the soft/hard mechanics are real but were over-rated in
severity; rank+count summation is the declared design).
