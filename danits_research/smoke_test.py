"""
Smoke test: load a *single* archived DermMNIST run's final_predictions.csv
from archive_experiments/, build the paper's Psi/Phi from the run's constraint
config, and run both the corrected LP and the corrected greedy heuristic.

Goal is NOT to benchmark performance — it's to validate that the wiring
works on real Phase-1 outputs from this project:
  - shapes line up,
  - LP is feasible and status is OPTIMAL,
  - every sample receives exactly one class,
  - feasibility actually holds (Psi and Phi respected),
  - heuristic produces comparable output.

Run from repo root:
    python -m danits_research.smoke_test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_PRESETS,
    build_psi_phi_from_percentages,
    describe_cost_matrix,
    solve_greedy_assignment,
    solve_lp_assignment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = REPO_ROOT / "archive_experiments" / "dermmnist"

# A stable archived run that exists on disk. c04_02 = (feature=0.4, target=0.2).
DEFAULT_RUN = (
    ARCHIVE_DIR
    / "our_approach"
    / "MobileNetV3"
    / "c04_02"
    / "baseline"
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _load_run(run_dir: Path):
    preds_path = run_dir / "final_predictions.csv"
    cfg_path = run_dir / "config.json"
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    preds = pd.read_csv(preds_path)
    config = json.loads(cfg_path.read_text())

    n_classes = int(config["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    y_proba = preds[prob_cols].to_numpy(dtype=np.float64)
    y_true = preds["True_Label"].to_numpy(dtype=np.int64)
    y_pred_original = preds["Predicted_Label"].to_numpy(dtype=np.int64)
    groups = preds["Group_ID"].to_numpy()

    # Renormalize any tiny drift so the LP's row-sum assertion is happy.
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    return {
        "y_proba": y_proba,
        "y_true": y_true,
        "y_pred_original": y_pred_original,
        "groups": groups,
        "n_classes": n_classes,
        "constrained_class": int(config["dataset_config"]["constrained_class"]),
        "feature_pct": float(config["constraint"][0]),
        "target_pct": float(config["constraint"][1]),
        "config": config,
    }


def _count_per_class(y: np.ndarray, n_classes: int) -> dict[int, int]:
    return {int(i): int((y == i).sum()) for i in range(n_classes)}


def _count_per_group_class(groups: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    out: dict = {}
    for g in np.unique(groups):
        mask = groups == g
        out[int(g)] = {int(i): int(((y == i) & mask).sum()) for i in range(n_classes)}
    return out


def _check_feasible(
    y_assigned: np.ndarray,
    groups: np.ndarray,
    psi: list,
    phi: dict,
    n_classes: int,
    label: str,
) -> list[str]:
    violations: list[str] = []
    counts = _count_per_class(y_assigned, n_classes)
    for i, bound in enumerate(psi):
        if bound is None:
            continue
        if counts[i] > bound:
            violations.append(
                f"  [{label}] PSI violated: class {i} count {counts[i]} > bound {bound}"
            )
    per_group = _count_per_group_class(groups, y_assigned, n_classes)
    for g, bounds in phi.items():
        g_counts = per_group.get(int(g), {})
        for i, bound in enumerate(bounds):
            if bound is None:
                continue
            if g_counts.get(i, 0) > bound:
                violations.append(
                    f"  [{label}] PHI violated: group {g} class {i} "
                    f"count {g_counts.get(i, 0)} > bound {bound}"
                )
    return violations


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main(run_dir: Path = DEFAULT_RUN) -> int:
    print(f"[smoke] loading run: {run_dir.relative_to(REPO_ROOT)}")
    if not run_dir.exists():
        print(f"[smoke] ERROR: run dir does not exist")
        return 1

    data = _load_run(run_dir)
    y_proba = data["y_proba"]
    y_true = data["y_true"]
    groups = data["groups"]
    n_classes = data["n_classes"]
    constrained_class = data["constrained_class"]
    feature_pct = data["feature_pct"]
    target_pct = data["target_pct"]

    N = y_proba.shape[0]
    print(f"[smoke] N={N}, C={n_classes}, constrained_class={constrained_class}, "
          f"(feature_pct, target_pct)=({feature_pct}, {target_pct})")
    print(f"[smoke] groups present: {sorted(np.unique(groups).tolist())}")
    print(f"[smoke] class prior on test set: {_count_per_class(y_true, n_classes)}")

    # ---- build Psi / Phi ----
    psi, phi = build_psi_phi_from_percentages(
        y_true=y_true,
        groups=groups,
        n_classes=n_classes,
        constrained_class=constrained_class,
        feature_pct=feature_pct,
        target_pct=target_pct,
    )
    print(f"[smoke] psi = {psi}")
    print(f"[smoke] phi = {phi}")

    # ---- original project prediction (post-hoc adjusted) for reference ----
    orig_counts = _count_per_class(data["y_pred_original"], n_classes)
    orig_accuracy = float((data["y_pred_original"] == y_true).mean())
    orig_mel_recall = _recall_on_class(
        y_true, data["y_pred_original"], constrained_class
    )
    print(f"[smoke] original (our_approach + post_hoc):")
    print(f"         acc = {orig_accuracy:.4f}, MEL recall = {orig_mel_recall:.4f}, "
          f"counts = {orig_counts}")

    # ---- sweep over cost-matrix presets ----
    summary_rows = []
    for preset_name, omega in DERMMNIST_PRESETS.items():
        print(f"\n[smoke] ---- cost preset: {preset_name} ----")
        print(describe_cost_matrix(omega, priority_idx=constrained_class))

        lp_res = solve_lp_assignment(
            y_proba=y_proba,
            groups=groups,
            cost_matrix=omega,
            psi=psi,
            phi=phi,
            verbose=False,
        )
        if lp_res.status != "OPTIMAL":
            print(f"[smoke] LP status {lp_res.status} for preset {preset_name}")
            return 2

        # Sanity: one-hot rows and feasibility
        row_sums = lp_res.R.sum(axis=1)
        assert np.all(row_sums == 1), f"LP row sums not all 1: {np.unique(row_sums)}"
        lp_violations = _check_feasible(
            lp_res.y_pred, groups, psi, phi, n_classes, f"LP[{preset_name}]"
        )
        if lp_violations:
            print("[smoke] LP feasibility FAILED:")
            for v in lp_violations:
                print(v)
            return 3

        lp_counts = _count_per_class(lp_res.y_pred, n_classes)
        lp_accuracy = float((lp_res.y_pred == y_true).mean())
        lp_mel_recall = _recall_on_class(y_true, lp_res.y_pred, constrained_class)
        print(f"[smoke]  LP  : obj={lp_res.objective_value:10.4f}  "
              f"acc={lp_accuracy:.4f}  MEL recall={lp_mel_recall:.4f}  "
              f"counts[MEL]={lp_counts[constrained_class]:>3d}  "
              f"runtime={lp_res.runtime_seconds:.3f}s")

        heur_res = solve_greedy_assignment(
            y_proba=y_proba,
            groups=groups,
            cost_matrix=omega,
            psi=psi,
            phi=phi,
        )
        heur_counts = _count_per_class(heur_res.y_pred, n_classes)
        heur_accuracy = float((heur_res.y_pred == y_true).mean())
        heur_mel_recall = _recall_on_class(y_true, heur_res.y_pred, constrained_class)
        heur_violations = _check_feasible(
            heur_res.y_pred, groups, psi, phi, n_classes, f"HEUR[{preset_name}]"
        )
        feas_flag = "OK" if not heur_violations else f"{len(heur_violations)} viol."
        print(f"[smoke]  HEUR: obj={heur_res.objective_value:10.4f}  "
              f"acc={heur_accuracy:.4f}  MEL recall={heur_mel_recall:.4f}  "
              f"counts[MEL]={heur_counts[constrained_class]:>3d}  "
              f"feas={feas_flag}  runtime={heur_res.runtime_seconds:.3f}s")

        summary_rows.append({
            "preset": preset_name,
            "lp_obj": lp_res.objective_value,
            "lp_acc": lp_accuracy,
            "lp_mel_recall": lp_mel_recall,
            "lp_counts": lp_counts,
            "heur_obj": heur_res.objective_value,
            "heur_acc": heur_accuracy,
            "heur_mel_recall": heur_mel_recall,
            "heur_counts": heur_counts,
        })

    # ---- compact side-by-side ----
    print("\n[smoke] === side-by-side across cost presets ===")
    print(f"  {'preset':<10s} | {'method':<6s} | {'acc':>6s} | "
          f"{'MEL rec.':>8s} | {'MEL cnt':>7s} | {'obj':>10s}")
    print("  " + "-" * 60)
    print(f"  {'(none)':<10s} | {'orig':<6s} | {orig_accuracy:>6.4f} | "
          f"{orig_mel_recall:>8.4f} | {orig_counts[constrained_class]:>7d} | {'-':>10s}")
    for row in summary_rows:
        print(f"  {row['preset']:<10s} | {'LP':<6s} | {row['lp_acc']:>6.4f} | "
              f"{row['lp_mel_recall']:>8.4f} | {row['lp_counts'][constrained_class]:>7d} | "
              f"{row['lp_obj']:>10.4f}")
        print(f"  {row['preset']:<10s} | {'HEUR':<6s} | {row['heur_acc']:>6.4f} | "
              f"{row['heur_mel_recall']:>8.4f} | {row['heur_counts'][constrained_class]:>7d} | "
              f"{row['heur_obj']:>10.4f}")

    print("\n[smoke] ALL CHECKS PASSED")
    return 0


def _recall_on_class(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> float:
    mask = y_true == cls
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == cls).mean())


if __name__ == "__main__":
    sys.exit(main())
