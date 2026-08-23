"""
Phase-2 linear program from Shifman et al. (2025), Eqs. (1)-(4).

    min   sum_s sum_i  R[s,i] * ( sum_j  Omega[s, i, j] * Yhat[s, j] )     (1)
    s.t.  sum_{s in lambda}  R[s, i]  <=  Phi[lambda][i]   for all i, lambda  (2)
          sum_s             R[s, i]  <=  Psi[i]           for all i          (3)
          sum_i             R[s, i]  =   1                for all s          (4)
          R >= 0                                                             (TU)

Cost convention (from the paper, sec 2.1):
    Omega[s, i, j] = cost of labelling sample s as class i when the true class is j.
    Rows i  = predicted, cols j = true.

Totally-unimodular constraint matrix -> LP relaxation returns integer R.
We therefore declare R as continuous NumVar(0, +inf) and let GLOP do the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ortools.linear_solver import pywraplp


@dataclass
class LPResult:
    R: np.ndarray                # (N, C) int assignment matrix
    y_pred: np.ndarray           # (N,) argmax of R
    objective_value: float       # LP optimum (expected cost under Yhat)
    status: str                  # "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | ...
    runtime_seconds: float
    num_variables: int
    num_constraints: int


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _broadcast_cost_matrix(cost_matrix: np.ndarray, n_samples: int, n_classes: int) -> np.ndarray:
    """
    Accept either a shared (C, C) cost matrix or per-sample (N, C, C).
    Return a (N, C, C) array with the paper's [predicted, true] convention.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.ndim == 2:
        assert cost_matrix.shape == (n_classes, n_classes), (
            f"2D cost_matrix must be ({n_classes}, {n_classes}), got {cost_matrix.shape}"
        )
        return np.broadcast_to(cost_matrix, (n_samples, n_classes, n_classes)).copy()
    if cost_matrix.ndim == 3:
        assert cost_matrix.shape == (n_samples, n_classes, n_classes), (
            f"3D cost_matrix must be ({n_samples}, {n_classes}, {n_classes}), "
            f"got {cost_matrix.shape}"
        )
        return cost_matrix
    raise ValueError(f"cost_matrix must be 2D or 3D, got ndim={cost_matrix.ndim}")


def _expected_cost_matrix(
    y_proba: np.ndarray, cost_matrix: np.ndarray
) -> np.ndarray:
    """
    E[s, i] = sum_j  Omega[s, i, j] * Yhat[s, j]

    Shapes:
        cost_matrix: (N, C, C), axis 1 = predicted i, axis 2 = true j.
        y_proba:     (N, C), axis 1 = true class probability.
        y_proba[:, None, :]: (N, 1, C) -> broadcast across predicted axis.
        product sums over axis 2 (true class).

    Returns (N, C).
    """
    # (N, C, C) * (N, 1, C) -> broadcast y_proba on the 'true' axis
    return (cost_matrix * y_proba[:, None, :]).sum(axis=2)


# ----------------------------------------------------------------------
# main API
# ----------------------------------------------------------------------

def solve_lp_assignment(
    y_proba: np.ndarray,
    groups: np.ndarray,
    cost_matrix: np.ndarray,
    psi: dict[int, int] | list | None,
    phi: dict | None,
    solver_name: str = "GLOP",
    verbose: bool = False,
) -> LPResult:
    """
    Solve the paper's Phase-2 LP.

    Parameters
    ----------
    y_proba
        (N, C) float array, rows sum to 1. Phase-1 predicted probabilities.
    groups
        (N,) array of group labels (any hashable dtype). The feature-based
        constraint is applied per unique group value appearing here.
    cost_matrix
        Either (C, C) or (N, C, C). cost_matrix[..., i, j] is the cost of
        predicting class i when the true class is j (paper convention).
    psi
        Global target-based constraint Psi(i). Either:
          - None: no global constraints.
          - list/tuple/array of length C: entries are int bounds or None/NaN
            for "unconstrained for this class".
          - dict[int, int]: mapping class index -> bound, missing keys are
            treated as unconstrained.
    phi
        Feature-based constraint Phi[lambda, i]. Either:
          - None: no local constraints.
          - dict[group_value -> list/array of length C] with int bounds or
            None/NaN entries. Missing group keys mean "no local constraint
            for that group". Missing class entries within a group are
            treated as None.
    solver_name
        OR-Tools solver name. GLOP (LP, what the paper uses) is the default.
    verbose
        Print solver progress.

    Returns
    -------
    LPResult
    """
    import time

    y_proba = np.asarray(y_proba, dtype=np.float64)
    groups = np.asarray(groups)

    assert y_proba.ndim == 2, f"y_proba must be (N, C), got {y_proba.shape}"
    n_samples, n_classes = y_proba.shape
    assert groups.shape == (n_samples,), (
        f"groups must be ({n_samples},), got {groups.shape}"
    )
    # Normalize if rows don't quite sum to 1 (float drift is OK, big drift isn't)
    row_sums = y_proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(
            f"y_proba rows must sum to 1 (max deviation {np.abs(row_sums - 1).max():.2e})"
        )

    cost_matrix = _broadcast_cost_matrix(cost_matrix, n_samples, n_classes)

    # ------- per-sample per-class expected cost (coefficients of R) ------
    E = _expected_cost_matrix(y_proba, cost_matrix)  # (N, C)

    # ------- solver ------------------------------------------------------
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise RuntimeError(f"Could not create OR-Tools solver {solver_name!r}")
    if verbose:
        solver.EnableOutput()

    # R[s, i] continuous in [0, +inf); TU guarantees integer optimum.
    R = np.empty((n_samples, n_classes), dtype=object)
    for s in range(n_samples):
        for i in range(n_classes):
            R[s, i] = solver.NumVar(0.0, solver.infinity(), f"R_{s}_{i}")

    # Objective: sum_s sum_i E[s, i] * R[s, i]
    obj = solver.Objective()
    for s in range(n_samples):
        for i in range(n_classes):
            obj.SetCoefficient(R[s, i], float(E[s, i]))
    obj.SetMinimization()

    # (4) Σ_i R[s, i] == 1 — paper uses equality
    for s in range(n_samples):
        ct = solver.Constraint(1.0, 1.0, f"one_class_{s}")
        for i in range(n_classes):
            ct.SetCoefficient(R[s, i], 1.0)

    # (3) Global target-based: Σ_s R[s, i] ≤ Psi(i)
    psi_map = _normalize_psi(psi, n_classes)
    for i, bound in psi_map.items():
        if bound is None:
            continue
        ct = solver.Constraint(-solver.infinity(), float(bound), f"psi_{i}")
        for s in range(n_samples):
            ct.SetCoefficient(R[s, i], 1.0)

    # (2) Feature-based: Σ_{s ∈ λ} R[s, i] ≤ Phi_λ(i)
    phi_norm = _normalize_phi(phi, n_classes)
    unique_groups = np.unique(groups)
    # If phi specifies a group key that doesn't appear in `groups`, it is harmless
    # (empty sum ≤ bound); no warning needed.
    for group_value, per_class_bounds in phi_norm.items():
        mask = groups == group_value
        sample_idxs = np.nonzero(mask)[0]
        if sample_idxs.size == 0:
            continue
        for i, bound in per_class_bounds.items():
            if bound is None:
                continue
            ct = solver.Constraint(
                -solver.infinity(), float(bound), f"phi_{group_value}_{i}"
            )
            for s in sample_idxs:
                ct.SetCoefficient(R[s, i], 1.0)

    # ------- solve -------------------------------------------------------
    num_vars = solver.NumVariables()
    num_cts = solver.NumConstraints()
    t0 = time.perf_counter()
    status = solver.Solve()
    runtime = time.perf_counter() - t0

    status_names = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }
    status_name = status_names.get(status, f"UNKNOWN_{status}")

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return LPResult(
            R=np.zeros((n_samples, n_classes), dtype=np.int64),
            y_pred=np.full(n_samples, -1, dtype=np.int64),
            objective_value=float("nan"),
            status=status_name,
            runtime_seconds=runtime,
            num_variables=num_vars,
            num_constraints=num_cts,
        )

    R_values = np.array(
        [[R[s, i].solution_value() for i in range(n_classes)] for s in range(n_samples)],
        dtype=np.float64,
    )
    # TU: every entry of R should be exactly 0 or 1 at the optimum (up to fp noise).
    R_int = np.round(R_values).astype(np.int64)
    # Sanity: after rounding, each row is a one-hot vector
    row_sums = R_int.sum(axis=1)
    if not np.all(row_sums == 1):
        # Extremely rare (GLOP tol issue). Repair by argmax.
        bad = np.nonzero(row_sums != 1)[0]
        for s in bad:
            R_int[s] = 0
            R_int[s, int(np.argmax(R_values[s]))] = 1

    y_pred = R_int.argmax(axis=1)

    return LPResult(
        R=R_int,
        y_pred=y_pred,
        objective_value=float(solver.Objective().Value()),
        status=status_name,
        runtime_seconds=runtime,
        num_variables=num_vars,
        num_constraints=num_cts,
    )


# ----------------------------------------------------------------------
# constraint normalization helpers
# ----------------------------------------------------------------------

def _normalize_psi(psi, n_classes: int) -> dict[int, Optional[int]]:
    if psi is None:
        return {}
    if isinstance(psi, dict):
        return {int(k): (None if _is_none(v) else int(v)) for k, v in psi.items()}
    psi_arr = list(psi)
    if len(psi_arr) != n_classes:
        raise ValueError(f"psi must have length {n_classes}, got {len(psi_arr)}")
    return {i: (None if _is_none(v) else int(v)) for i, v in enumerate(psi_arr)}


def _normalize_phi(phi, n_classes: int) -> dict:
    """Return dict[group -> dict[class_idx -> int | None]]."""
    if phi is None:
        return {}
    out: dict = {}
    for g, bounds in phi.items():
        if isinstance(bounds, dict):
            out[g] = {
                int(k): (None if _is_none(v) else int(v)) for k, v in bounds.items()
            }
            continue
        bounds_list = list(bounds)
        if len(bounds_list) != n_classes:
            raise ValueError(
                f"phi[{g!r}] must have length {n_classes}, got {len(bounds_list)}"
            )
        out[g] = {
            i: (None if _is_none(v) else int(v)) for i, v in enumerate(bounds_list)
        }
    return out


def _is_none(v) -> bool:
    if v is None:
        return True
    try:
        return bool(np.isnan(v))
    except (TypeError, ValueError):
        return False
