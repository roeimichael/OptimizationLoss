# TraLO-derived allocation budgets: four-seed diagnostic

Requested 2026-09-22. Training is held fixed to isolate the evaluation change.
The original high-rho/shared-Adam recipe has a documented failure; this experiment
does not repair it, extend training to convergence, or claim to test a repaired
TraLO. Four seeds 701--704, all three arms, unchanged 10 epochs / 5 warm-up,
FP32 dsisco01, frozen ResNet18 CIFAR100 features and the existing data split.
Seeds 701--703 are replays, not new independent evidence beyond those seeds.

For constrained class c define K'_c = count(raw TraLO prediction == c).
Leave all other classes uncapped. No labels determine K'. Each seed supplies its
own TraLO counts to ALL three arms: Clipper, TraLO-null, TraLO. This implements
both interpretations of the request without choosing one after seeing scores.

Report raw predictions, both existing allocators with original K, and both with
derived K'. Reallocation uses each arm's own probabilities. Never apply one
allocator after another. At every output show accuracy, macro-F1, constrained F1,
changed labels, capped-class counts, and excess above BOTH K and K'. Keep the
constrained-class metric set fixed even when a derived cap is zero.

Required identity: upper-bound correction on TraLO at K' equals raw TraLO.
Capped-first at K' fills the observed constrained counts, since their total is
at most the sample population. This may select different samples. Neither
condition proves correctness against labels or global optimality.

All derived-budget comparisons are exploratory alternative-policy diagnostics.
A relaxed cap cannot be presented as satisfying the original cap. Preserve the
fixed-policy results alongside them. Summaries average four paired seeds only;
show every seed and paired deltas, SD and t intervals, without winner claims.

Before training: validate source hashes and tests on both hosts; verify feature
artifact hashes, dataset bytes, split uniqueness/disjointness, image duplicates,
label/sample alignment and finite feature shapes. Retain the original extraction
provenance. Audit update counts and matched warm-up/batches after training.
Independently recompute confusion metrics and capacity counts from saved outputs.
These checks cannot certify absence of all bugs, near-duplicate images or
pretraining overlap. This is inspected development data, not untouched testing.
