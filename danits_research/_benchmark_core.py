"""
Shared core for the 2x3 benchmark matrix.

Both `benchmark.py` (single-constraint) and `benchmark_multi.py`
(multi-constraint) use the same cell-computation and row-building
logic. Keeping it here means any change to how a cell is defined
affects both benchmarks identically.

The 2x3 matrix:

                     | argmax (no Phase-2) | project greedy | paper [5] LP
    -----------------+---------------------+----------------+--------------
    warmup model     |      (W,-)          |     (W,g)      |    (W,LP)
    constraint model |      (C,-)          |     (C,g)      |    (C,LP)

Conventions
-----------
- The (W,g) cell equals the `heuristic/.../final_predictions.csv::Predicted_Label`
  value saved by `run_heuristic.py`. We read it from disk (no recomputation).
- The (C,g) cell equals the `our_approach/.../final_predictions.csv::Predicted_Label`
  value saved by `run_experiment.py` after `targeted_correction`. We read it
  from disk.
- The (W,-) and (C,-) cells are the argmax of the respective `Prob_Class_*`
  columns -- the raw model's unconstrained prediction. These rows are almost
  always infeasible and are reported as "reference" rows.
- The (W,LP) and (C,LP) cells are produced by calling `solve_lp_assignment`
  on the respective probability matrix with the same Psi/Phi and cost matrix.
- All feasibility checks use the same `(psi, phi)` built via
  `build_psi_phi_from_percentages`, which uses `round()` to match the
  project's own bound-computation in `src/training/constraints.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    build_psi_phi_from_percentages,
    solve_lp_assignment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "archive_experiments"

# Canonical LP cost preset for the 2x3 matrix main column.
DEFAULT_LP_COST = DERMMNIST_IDENTITY


# ----------------------------------------------------------------------
# data-loading
# ----------------------------------------------------------------------

@dataclass
class LoadedRun:
    run_dir: Path
    cfg: dict
    probs: np.ndarray          # (N, C) raw probabilities from the model
    y_true: np.ndarray         # (N,) integer labels
    y_pred_saved: np.ndarray   # (N,) what this run wrote to final_predictions.csv
    groups: np.ndarray         # (N,) group ids
    n_classes: int


def load_run(run_dir: Path) -> LoadedRun:
    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "final_predictions.csv")
    n_classes = int(cfg["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)  # renormalize away fp drift
    return LoadedRun(
        run_dir=run_dir,
        cfg=cfg,
        probs=probs,
        y_true=df["True_Label"].to_numpy(dtype=np.int64),
        y_pred_saved=df["Predicted_Label"].to_numpy(dtype=np.int64),
        groups=df["Group_ID"].to_numpy(),
        n_classes=n_classes,
    )


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def _counts(y: np.ndarray, n: int) -> dict[int, int]:
    return {int(i): int((y == i).sum()) for i in range(n)}


def _recall(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    m = y_true == c
    if not m.any():
        return float("nan")
    return float((y_pred[m] == c).mean())


def _precision(y_true: np.ndarray, y_pred: np.ndarray, c: int) -> float:
    m = y_pred == c
    if not m.any():
        return float("nan")
    return float((y_true[m] == c).mean())


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    f1s = []
    for c in range(n):
        p = _precision(y_true, y_pred, c)
        r = _recall(y_true, y_pred, c)
        if np.isnan(p) or np.isnan(r) or (p + r) == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * p * r / (p + r))
    return float(np.mean(f1s))


def _feasibility(y_pred: np.ndarray, groups: np.ndarray,
                 psi: list, phi: dict, n_classes: int) -> tuple[bool, list[str]]:
    violations = []
    for i, bound in enumerate(psi):
        if bound is None:
            continue
        cnt = int((y_pred == i).sum())
        if cnt > bound:
            violations.append(f"psi[{i}]: {cnt}/{bound}")
    for g, bounds in phi.items():
        mask = groups == g
        for i, bound in enumerate(bounds):
            if bound is None:
                continue
            cnt = int(((y_pred == i) & mask).sum())
            if cnt > bound:
                violations.append(f"phi[g={g},{i}]: {cnt}/{bound}")
    return (not violations), violations


# ----------------------------------------------------------------------
# 2x3 matrix cell builder
# ----------------------------------------------------------------------

@dataclass
class MatrixCell:
    model_source: str          # "warmup" | "constraint"
    phase2_method: str         # "argmax" | "greedy" | "LP"
    y_pred: np.ndarray
    accuracy: float
    macro_f1: float
    recall_by_class: dict[int, float]
    precision_by_class: dict[int, float]
    counts_by_class: dict[int, int]
    feasible: bool
    violations: list[str] = field(default_factory=list)
    runtime_s: Optional[float] = None
    lp_objective: Optional[float] = None


def _eval_cell(
    model_source: str,
    phase2_method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    psi: list,
    phi: dict,
    n_classes: int,
    constrained_classes: list[int],
    runtime_s: Optional[float] = None,
    lp_obj: Optional[float] = None,
) -> MatrixCell:
    feasible, viol = _feasibility(y_pred, groups, psi, phi, n_classes)
    return MatrixCell(
        model_source=model_source,
        phase2_method=phase2_method,
        y_pred=y_pred,
        accuracy=float((y_pred == y_true).mean()),
        macro_f1=_macro_f1(y_true, y_pred, n_classes),
        recall_by_class={c: _recall(y_true, y_pred, c) for c in constrained_classes},
        precision_by_class={c: _precision(y_true, y_pred, c) for c in constrained_classes},
        counts_by_class=_counts(y_pred, n_classes),
        feasible=feasible,
        violations=viol,
        runtime_s=runtime_s,
        lp_objective=lp_obj,
    )


@dataclass
class BenchmarkResult:
    name: str                          # human-readable identifier
    base_model_id: str
    n_classes: int
    constrained_classes: list[int]
    feature_pct: float
    target_pct: float
    n_samples: int
    psi: list
    phi: dict
    warmup_run_dir: Path
    constraint_run_dir: Path
    cells: list[MatrixCell]


def build_2x3_matrix(
    warmup_run: LoadedRun,
    constraint_run: LoadedRun,
    cost_matrix: np.ndarray = DEFAULT_LP_COST,
    name: str = "",
) -> BenchmarkResult:
    """
    Given two LoadedRun objects that share y_true/groups/config (as verified
    by preflight), compute the full 2x3 matrix and return all six cells.
    """
    # Sanity (should already be guaranteed by preflight, but belt-and-braces).
    assert np.array_equal(warmup_run.y_true, constraint_run.y_true), \
        "y_true mismatch between paired runs"
    assert np.array_equal(warmup_run.groups, constraint_run.groups), \
        "Group_ID mismatch between paired runs"
    assert warmup_run.n_classes == constraint_run.n_classes, "C mismatch"

    y_true = warmup_run.y_true
    groups = warmup_run.groups
    n_classes = warmup_run.n_classes

    # Extract the constraint spec from the warmup (same for both after preflight)
    w_cfg = warmup_run.cfg
    constrained_raw = w_cfg["dataset_config"]["constrained_class"]
    if isinstance(constrained_raw, (int, np.integer)):
        constrained_classes = [int(constrained_raw)]
    else:
        constrained_classes = [int(c) for c in constrained_raw]
    feature_pct = float(w_cfg["constraint"][0])
    target_pct = float(w_cfg["constraint"][1])

    psi, phi = build_psi_phi_from_percentages(
        y_true=y_true, groups=groups, n_classes=n_classes,
        constrained_class=constrained_classes,
        feature_pct=feature_pct, target_pct=target_pct,
    )

    cells: list[MatrixCell] = []

    # --- row 1: warmup model --------------------------------------
    # (W, argmax) -- raw warmup argmax, usually infeasible
    cells.append(_eval_cell(
        "warmup", "argmax", y_true,
        warmup_run.probs.argmax(axis=1), groups, psi, phi,
        n_classes, constrained_classes,
    ))
    # (W, greedy) -- read from saved Predicted_Label of heuristic run
    cells.append(_eval_cell(
        "warmup", "greedy", y_true,
        warmup_run.y_pred_saved, groups, psi, phi,
        n_classes, constrained_classes,
    ))
    # (W, LP) -- our LP on warmup probs
    lp_w = solve_lp_assignment(
        y_proba=warmup_run.probs, groups=groups, cost_matrix=cost_matrix,
        psi=psi, phi=phi, verbose=False,
    )
    cells.append(_eval_cell(
        "warmup", "LP", y_true,
        lp_w.y_pred, groups, psi, phi,
        n_classes, constrained_classes,
        runtime_s=lp_w.runtime_seconds, lp_obj=lp_w.objective_value,
    ))

    # --- row 2: constraint-trained model --------------------------
    # (C, argmax) -- raw constraint-trained argmax, usually infeasible
    cells.append(_eval_cell(
        "constraint", "argmax", y_true,
        constraint_run.probs.argmax(axis=1), groups, psi, phi,
        n_classes, constrained_classes,
    ))
    # (C, greedy) -- read from saved Predicted_Label of our_approach run
    #                (this is targeted_correction output from run_experiment.py)
    cells.append(_eval_cell(
        "constraint", "greedy", y_true,
        constraint_run.y_pred_saved, groups, psi, phi,
        n_classes, constrained_classes,
    ))
    # (C, LP) -- our LP on constraint-trained probs
    lp_c = solve_lp_assignment(
        y_proba=constraint_run.probs, groups=groups, cost_matrix=cost_matrix,
        psi=psi, phi=phi, verbose=False,
    )
    cells.append(_eval_cell(
        "constraint", "LP", y_true,
        lp_c.y_pred, groups, psi, phi,
        n_classes, constrained_classes,
        runtime_s=lp_c.runtime_seconds, lp_obj=lp_c.objective_value,
    ))

    return BenchmarkResult(
        name=name or warmup_run.run_dir.name,
        base_model_id=warmup_run.cfg.get("base_model_id", "?"),
        n_classes=n_classes,
        constrained_classes=constrained_classes,
        feature_pct=feature_pct,
        target_pct=target_pct,
        n_samples=len(y_true),
        psi=psi,
        phi=phi,
        warmup_run_dir=warmup_run.run_dir,
        constraint_run_dir=constraint_run.run_dir,
        cells=cells,
    )


# ----------------------------------------------------------------------
# printing
# ----------------------------------------------------------------------

def print_matrix(result: BenchmarkResult, header: str = "") -> None:
    constrained = result.constrained_classes
    if header:
        print(header)

    # per-class column headers
    per_class_cols = []
    for c in constrained:
        per_class_cols.append(f"c{c} rec")
        per_class_cols.append(f"c{c} prec")
        per_class_cols.append(f"c{c} n")
    header_parts = [f"{'model':<11s}", f"{'phase2':<7s}",
                    f"{'acc':>6s}", f"{'F1m':>6s}"]
    for c in per_class_cols:
        header_parts.append(f"{c:>8s}")
    header_parts.append(f"{'feas':>5s}")
    header_parts.append(f"{'rt':>8s}")
    hdr = "  " + " | ".join(header_parts)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for cell in result.cells:
        parts = [
            f"{cell.model_source:<11s}",
            f"{cell.phase2_method:<7s}",
            f"{cell.accuracy:>6.4f}",
            f"{cell.macro_f1:>6.4f}",
        ]
        for c in constrained:
            parts.append(f"{cell.recall_by_class[c]:>8.4f}")
            parts.append(f"{cell.precision_by_class[c]:>8.4f}")
            parts.append(f"{cell.counts_by_class[c]:>8d}")
        feas = "OK" if cell.feasible else f"x{len(cell.violations)}"
        parts.append(f"{feas:>5s}")
        rt = f"{cell.runtime_s*1000:.1f}ms" if cell.runtime_s is not None else "-"
        parts.append(f"{rt:>8s}")
        print("  " + " | ".join(parts))


# ----------------------------------------------------------------------
# tidy CSV output
# ----------------------------------------------------------------------

def result_to_rows(result: BenchmarkResult, variant: str = "") -> list[dict]:
    """Flatten a BenchmarkResult to long-format rows, one per (cell, class)."""
    rows = []
    for cell in result.cells:
        for c in result.constrained_classes:
            rows.append({
                "benchmark":       result.name,
                "variant":         variant,
                "base_model_id":   result.base_model_id,
                "n_samples":       result.n_samples,
                "feature_pct":     result.feature_pct,
                "target_pct":      result.target_pct,
                "constrained_class": c,
                "model_source":    cell.model_source,
                "phase2_method":   cell.phase2_method,
                "accuracy":        cell.accuracy,
                "macro_f1":        cell.macro_f1,
                "class_recall":    cell.recall_by_class[c],
                "class_precision": cell.precision_by_class[c],
                "class_count":     cell.counts_by_class[c],
                "feasible":        cell.feasible,
                "n_violations":    len(cell.violations),
                "runtime_s":       cell.runtime_s,
                "lp_objective":    cell.lp_objective,
            })
    return rows


def sanity_check_saved_equals_cells(result: BenchmarkResult) -> list[str]:
    """
    Sanity: the (warmup, greedy) cell MUST equal the y_pred_saved column
    of the heuristic run, and the (constraint, greedy) cell MUST equal the
    y_pred_saved of the our_approach run. If not, something is reading the
    wrong file. Return list of warnings (empty if clean).
    """
    warnings: list[str] = []
    for cell in result.cells:
        if cell.model_source == "warmup" and cell.phase2_method == "greedy":
            # y_pred was set to warmup_run.y_pred_saved by construction,
            # so this is a tautology -- nothing to check.
            pass
        if cell.model_source == "constraint" and cell.phase2_method == "greedy":
            # Same: y_pred == constraint_run.y_pred_saved by construction.
            pass
    return warnings
