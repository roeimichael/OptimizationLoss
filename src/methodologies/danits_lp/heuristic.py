"""
Paper [5] Algorithm 1 — naive greedy heuristic (baseline for the LP).

Paper setup: every class has a finite Psi(i) and the algorithm greedily
fills each class in priority order by ascending expected cost.

This implementation extends Algorithm 1 to the "only a subset of classes
is constrained" case that we actually run in OptimizationLoss (usually
a single `constrained_class`):

  1. For every class `c` in `class_order` with a FINITE Psi(c), greedy-fill
     it in the paper's exact Algorithm 1 manner: sort samples ascending by
     E[:, c], assign one at a time respecting Phi_lambda(c), stop when
     Psi(c) is reached. (This is paper [5] verbatim.)
  2. Once all constrained classes are exhausted, each leftover sample is
     assigned to the cheapest UNCONSTRAINED class (Psi=None) that still
     has Phi budget in the sample's group. Unconstrained classes are the
     natural "dumping ground" — this matches what the paper's algorithm
     would degenerate to if the unconstrained Psi were set to +infinity
     and the unconstrained classes were processed last.

Fixes relative to the colleague's notebook implementation:
  - sort ascending (not descending)
  - no "argmin == c" gate
  - no dataset-specific group==1 shortcut
  - handles Psi=None without collapsing to a single class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .lp_solver import _broadcast_cost_matrix, _expected_cost_matrix, _normalize_psi, _normalize_phi


@dataclass
class HeuristicResult:
    R: np.ndarray                # (N, C) int assignment
    y_pred: np.ndarray           # (N,) argmax
    objective_value: float       # realised expected-cost sum under Yhat
    runtime_seconds: float
    unassigned_fallback_count: int  # how many samples were placed in the fallback class


def solve_greedy_assignment(
    y_proba: np.ndarray,
    groups: np.ndarray,
    cost_matrix: np.ndarray,
    psi: dict[int, int] | list | None,
    phi: dict | None,
    class_order: Optional[Sequence[int]] = None,
) -> HeuristicResult:
    """
    Parameters match solve_lp_assignment (see lp_solver.py for details).

    Extra:
    class_order
        Iterable of class indices, the order in which the heuristic fills
        Psi(i). Defaults to 0..C-1. The LAST class in this ordering acts as
        the fallback for any leftover samples.
    """
    import time

    y_proba = np.asarray(y_proba, dtype=np.float64)
    groups = np.asarray(groups)

    n_samples, n_classes = y_proba.shape
    assert groups.shape == (n_samples,)
    cost_matrix = _broadcast_cost_matrix(cost_matrix, n_samples, n_classes)
    E = _expected_cost_matrix(y_proba, cost_matrix)  # (N, C)

    psi_map = _normalize_psi(psi, n_classes)
    phi_norm = _normalize_phi(phi, n_classes)

    if class_order is None:
        class_order = list(range(n_classes))
    else:
        class_order = list(class_order)
        assert sorted(class_order) == list(range(n_classes)), (
            f"class_order must be a permutation of 0..{n_classes - 1}, got {class_order}"
        )

    constrained_classes = [c for c in class_order if psi_map.get(c) is not None]
    unconstrained_classes = [c for c in class_order if psi_map.get(c) is None]

    t0 = time.perf_counter()

    R = np.zeros((n_samples, n_classes), dtype=np.int64)
    already_assigned = np.zeros(n_samples, dtype=bool)

    # Running counts of R-per-group-per-class
    local_used: dict = {g: {i: 0 for i in range(n_classes)} for g in np.unique(groups)}

    def _local_bound(group_value, class_idx) -> Optional[int]:
        bounds = phi_norm.get(group_value)
        if bounds is None:
            return None
        return bounds.get(class_idx)

    # ---- Phase A: paper Algorithm 1 over the FINITELY-constrained classes ----
    for c in constrained_classes:
        psi_c = psi_map[c]  # finite int by construction

        # paper Alg 1 line 5: sort ASCENDING by expected cost for class c
        candidate_order = np.argsort(E[:, c], kind="stable")

        filled = 0
        for s in candidate_order:
            if already_assigned[s]:
                continue
            if filled >= psi_c:
                break  # paper Alg 1 line 13: "until Psi(i) is met"

            g = groups[s]
            phi_ci = _local_bound(g, c)
            if phi_ci is not None and local_used[g][c] >= phi_ci:
                continue  # paper Alg 1 line 9: feature bound saturated for this group

            R[s, c] = 1
            already_assigned[s] = True
            local_used[g][c] += 1
            filled += 1

    # ---- Phase B: leftover -> cheapest unconstrained class with phi-slack ----
    # This is our extension of Algorithm 1 for the "Psi is None for some
    # classes" case that the paper doesn't explicitly handle. For each
    # leftover sample, assign it to the unconstrained class that minimizes
    # E[s, c] while still having Phi room in the sample's group. If no such
    # class exists, fall back to the absolute argmin over all classes
    # (violating Phi if we must — the heuristic isn't guaranteed feasible;
    # only the LP is).
    fallback_count = 0
    if unconstrained_classes:
        unconstrained_arr = np.array(unconstrained_classes, dtype=np.int64)
    else:
        unconstrained_arr = None

    unassigned = np.nonzero(~already_assigned)[0]
    for s in unassigned:
        g = groups[s]

        if unconstrained_arr is not None:
            # Check phi for each unconstrained class
            feasible_classes = [
                c for c in unconstrained_classes
                if (
                    (bound := _local_bound(g, c)) is None
                    or local_used[g][c] < bound
                )
            ]
            if feasible_classes:
                costs_here = E[s, feasible_classes]
                best = feasible_classes[int(np.argmin(costs_here))]
                R[s, best] = 1
                already_assigned[s] = True
                local_used[g][best] += 1
                continue

        # No feasible unconstrained class (or none exist): last-resort
        # fallback to global argmin. May violate Phi.
        best = int(np.argmin(E[s]))
        R[s, best] = 1
        already_assigned[s] = True
        local_used[g][best] += 1
        fallback_count += 1

    # y_pred
    y_pred = R.argmax(axis=1)
    # Handle a pathological row where two classes could tie in R (shouldn't
    # happen given the logic above, but argmax is well-defined regardless).

    runtime = time.perf_counter() - t0

    # Realised objective under Yhat
    objective_value = float(E[np.arange(n_samples), y_pred].sum())

    return HeuristicResult(
        R=R,
        y_pred=y_pred,
        objective_value=objective_value,
        runtime_seconds=runtime,
        unassigned_fallback_count=fallback_count,
    )
