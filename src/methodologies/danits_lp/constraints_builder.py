"""
Utility that converts this project's `(feature_pct, target_pct, constrained_class)`
spec into the Psi (global) and Phi (local) vectors that the paper's LP expects.

The OptimizationLoss project encodes constraints as either:
  * a single `constrained_class: int` - one class is binding
  * a list `constrained_class: [int, int, ...]` - multiple classes are
    simultaneously binding (e.g. the `dermmnist_round2_conflicting_constraints`
    experiments constrain both MEL and BCC, or MEL+BKL+BCC).

All constrained classes use the same `(feature_pct, target_pct)` pair,
which the paper's experimental protocol also assumes ("the percentages for
feature-based and target-based constraints were assumed to be the same
across all subsets and classes" -- paper [5] section 3.2).

Psi(c) = round(target_pct * #{samples with true label == c})
Phi_lambda(c) = round(feature_pct * #{samples in group lambda with true label == c})

IMPORTANT: we use `np.round` here to match the project's own bound-derivation
in `src/training/constraints.py` (functions `compute_global_constraints` and
`compute_local_constraints`). This keeps the paper [5] LP's feasibility
reference identical to the bounds the `heuristic` and `tralo` runners
actually see at training/eval time. Mismatching the rounding convention
(e.g. floor vs round) turns a `9.6` bound into either 9 or 10 and flags
perfectly-feasible project runs as "off-by-one violations" -- which is a
bug in the benchmark, not in the methods being benchmarked.

Unconstrained classes / groups get None.
"""

from __future__ import annotations

from typing import Iterable, Union

import numpy as np


def build_psi_phi_from_percentages(
    y_true: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    constrained_class: Union[int, Iterable[int]],
    feature_pct: float,
    target_pct: float,
) -> tuple[list, dict]:
    """
    Build (psi, phi) in the format expected by solve_lp_assignment /
    solve_greedy_assignment.

    Bounds use `int(np.round(count * percentage))`, matching
    `src.training.constraints.compute_global_constraints` and
    `compute_local_constraints`.

    Parameters
    ----------
    constrained_class
        Either a single int or an iterable of ints. All listed classes
        share the same (feature_pct, target_pct) pair; unlisted classes
        are unconstrained.

    Returns
    -------
    psi : list of length n_classes
        entries are int bounds or None (unconstrained).
    phi : dict[group_value -> list of length n_classes]
    """
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)

    if isinstance(constrained_class, (int, np.integer)):
        constrained_classes: list[int] = [int(constrained_class)]
    else:
        constrained_classes = [int(c) for c in constrained_class]

    psi = [None] * n_classes
    for c in constrained_classes:
        total_in_class = int((y_true == c).sum())
        psi[c] = int(np.round(target_pct * total_in_class))

    phi: dict = {}
    for g in np.unique(groups):
        in_group = groups == g
        bounds = [None] * n_classes
        for c in constrained_classes:
            total_in_group_class = int(((y_true == c) & in_group).sum())
            bounds[c] = int(np.round(feature_pct * total_in_group_class))
        phi[g.item() if hasattr(g, "item") else g] = bounds

    return psi, phi
