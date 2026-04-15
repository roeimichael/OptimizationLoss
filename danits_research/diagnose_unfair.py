"""
Diagnose why our_approach and the LP appear almost identical.

Hypothesis: `our_approach/.../final_predictions.csv` has its Predicted_Label
column produced by `targeted_correction` (a greedy post-hoc projection)
applied to the constraint-trained model's probabilities. The benchmark
therefore compares:

    heuristic    : (warmup probs)            + apply_allocation_heuristic
    LP           : (warmup probs)            + paper [5] LP
    our_approach : (constraint-trained probs) + targeted_correction

If the constraint-trained probs are close to the warmup probs (which is
plausible when lambda is small and the constraint loss barely moves the
weights), then (A) our_approach's Predicted_Label is essentially the same
as running the project greedy on warmup probs, and (B) the LP is also
running on warmup probs. Everyone ends up at the same feasible polytope
vertex.

This script checks four things on an exp1 archive pair (heuristic vs
our_approach sharing one warmup hash):

(1) How much do the constraint-trained probabilities differ from the
    warmup-only probabilities? (mean diff, top-1 agreement, per-class KL)
(2) What would our_approach's predictions look like WITHOUT the post-hoc
    adjustment? (raw argmax of its own probs -- should be infeasible if
    the constraint training did not drive the counts to the budget)
(3) What are the differences between:
        (a) warmup raw argmax
        (b) constraint-trained raw argmax
        (c) our_approach saved Predicted_Label (post-adjusted)
        (d) our paper [5] LP applied to warmup probs
        (e) our paper [5] LP applied to CONSTRAINT-TRAINED probs
(4) Is the LP on constraint-trained probs different from LP on warmup?
    If yes, THAT is the fair comparison the benchmark is missing.

Run:
    python -m danits_research.diagnose_unfair
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    build_psi_phi_from_percentages,
    solve_lp_assignment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MULTI_ROOT = (
    REPO_ROOT / "archive_experiments" / "dermmnist_round2_conflicting_constraints"
)

# exp1 is the canonical shared-warmup pair.
EXP = "exp1_MobileNetV3_MEL_BCC_L20G80"
HEUR_DIR = MULTI_ROOT / EXP / "heuristic"
OA_DIR = MULTI_ROOT / EXP / "our_approach"


def _load(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "final_predictions.csv")
    n_classes = int(cfg["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "probs": probs,
        "y_true": df["True_Label"].to_numpy(dtype=np.int64),
        "y_pred": df["Predicted_Label"].to_numpy(dtype=np.int64),
        "groups": df["Group_ID"].to_numpy(),
        "n_classes": n_classes,
        "constrained": cfg["dataset_config"]["constrained_class"],
        "feature_pct": float(cfg["constraint"][0]),
        "target_pct": float(cfg["constraint"][1]),
        "base_model_id": cfg["base_model_id"],
    }


def _counts(y, n):
    return {int(i): int((y == i).sum()) for i in range(n)}


def _recall(y_true, y_pred, c):
    m = y_true == c
    if not m.any():
        return float("nan")
    return float((y_pred[m] == c).mean())


def _precision(y_true, y_pred, c):
    m = y_pred == c
    if not m.any():
        return float("nan")
    return float((y_true[m] == c).mean())


def main():
    heur = _load(HEUR_DIR)
    oa = _load(OA_DIR)

    assert heur["base_model_id"] == oa["base_model_id"], "warmup mismatch"
    assert np.array_equal(heur["y_true"], oa["y_true"]), "y_true mismatch"

    print("=" * 78)
    print(f"DIAGNOSIS on {EXP}")
    print(f"shared base_model_id: {heur['base_model_id']}")
    print("=" * 78)

    y_true = heur["y_true"]
    groups = heur["groups"]
    n = heur["n_classes"]
    mel = 4  # MEL is in the constrained list for this experiment
    N = heur["probs"].shape[0]

    p_warm = heur["probs"]          # warmup-only probabilities
    p_cons = oa["probs"]            # constraint-trained probabilities

    # ================ (1) prob distribution diff =========================
    print("\n(1) PROBABILITY DISTRIBUTION DIFFERENCES")
    print("-" * 78)
    diff = np.abs(p_warm - p_cons)
    per_sample_l1 = diff.sum(axis=1)
    print(f"  per-sample L1 probability diff:")
    print(f"    mean    = {per_sample_l1.mean():.4f}")
    print(f"    median  = {np.median(per_sample_l1):.4f}")
    print(f"    p90     = {np.quantile(per_sample_l1, 0.9):.4f}")
    print(f"    p99     = {np.quantile(per_sample_l1, 0.99):.4f}")
    print(f"    max     = {per_sample_l1.max():.4f}")

    a_warm = p_warm.argmax(axis=1)
    a_cons = p_cons.argmax(axis=1)
    agreement = int((a_warm == a_cons).sum())
    print(f"\n  top-1 agreement (argmax == argmax): "
          f"{agreement}/{N} = {agreement/N:.4f}")

    print(f"\n  raw argmax class counts:")
    print(f"    warmup             : {_counts(a_warm, n)}")
    print(f"    constraint-trained : {_counts(a_cons, n)}")

    print(f"\n  P(MEL) distribution:")
    print(f"    warmup              mean={p_warm[:, mel].mean():.4f}  "
          f"median={np.median(p_warm[:, mel]):.4f}  "
          f"max={p_warm[:, mel].max():.4f}  "
          f">0.5: {(p_warm[:, mel] > 0.5).sum()}")
    print(f"    constraint-trained  mean={p_cons[:, mel].mean():.4f}  "
          f"median={np.median(p_cons[:, mel]):.4f}  "
          f"max={p_cons[:, mel].max():.4f}  "
          f">0.5: {(p_cons[:, mel] > 0.5).sum()}")

    # ================ (2) 5 prediction variants ===========================
    print("\n(2) FIVE PREDICTION VARIANTS on same test set")
    print("-" * 78)

    # Build Psi / Phi (same for both since y_true/groups are identical)
    psi, phi = build_psi_phi_from_percentages(
        y_true, groups, n, oa["constrained"],
        oa["feature_pct"], oa["target_pct"],
    )
    print(f"  Psi: {psi}")
    print(f"  Phi: {phi}")

    # (a) warmup raw argmax
    pred_a = a_warm
    # (b) constraint-trained raw argmax
    pred_b = a_cons
    # (c) our_approach saved Predicted_Label (already post-hoc'd by targeted_correction)
    pred_c = oa["y_pred"]
    # (d) LP on warmup probs
    lp_warm = solve_lp_assignment(
        y_proba=p_warm, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
        psi=psi, phi=phi,
    )
    pred_d = lp_warm.y_pred
    # (e) LP on constraint-trained probs
    lp_cons = solve_lp_assignment(
        y_proba=p_cons, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
        psi=psi, phi=phi,
    )
    pred_e = lp_cons.y_pred

    variants = {
        "(a) warmup argmax              ": pred_a,
        "(b) constraint-trained argmax  ": pred_b,
        "(c) our_approach (post-hoc'd)  ": pred_c,
        "(d) LP on warmup probs         ": pred_d,
        "(e) LP on constraint-trained   ": pred_e,
    }

    print(f"\n  {'variant':<35s} | {'acc':>6s} | {'c4 rec':>6s} | "
          f"{'c4 prec':>7s} | {'c4 n':>5s} | {'c1 n':>5s}")
    print("  " + "-" * 74)
    for name, y in variants.items():
        acc = float((y == y_true).mean())
        print(f"  {name:<35s} | {acc:.4f} | {_recall(y_true, y, 4):.4f} | "
              f"{_precision(y_true, y, 4):.4f}  | "
              f"{int((y == 4).sum()):>5d} | {int((y == 1).sum()):>5d}")

    # ================ (3) pairwise diffs ===================================
    print("\n(3) PAIRWISE DISAGREEMENTS (num samples where predictions differ)")
    print("-" * 78)
    names = list(variants.keys())
    print(f"  {'':37s}", end="")
    for n2 in names:
        print(f"  {n2[:5]:>5s}", end="")
    print()
    for i, (ni, yi) in enumerate(variants.items()):
        print(f"  {ni:<37s}", end="")
        for j, (nj, yj) in enumerate(variants.items()):
            if i == j:
                print(f"  {'-':>5s}", end="")
            else:
                d = int((yi != yj).sum())
                print(f"  {d:>5d}", end="")
        print()

    # ================ (4) summary interpretation ==========================
    print("\n(4) WHAT THIS MEANS")
    print("-" * 78)
    if per_sample_l1.mean() < 0.02:
        print("  The constraint-trained probabilities are ESSENTIALLY IDENTICAL to")
        print("  the warmup probabilities (mean L1 < 0.02). The constraint loss")
        print("  barely moved the model. That's why our_approach (c) and LP (d)")
        print("  produce similar predictions -- they're operating on the same")
        print("  posterior, just via different projections.")
    elif per_sample_l1.mean() < 0.10:
        print("  The constraint-trained probabilities are moderately different")
        print("  from warmup. Some samples moved; the LP on constraint probs (e)")
        print("  may differ meaningfully from the LP on warmup (d).")
    else:
        print("  The constraint-trained probabilities are substantially different")
        print("  from warmup. The LP on constraint probs (e) SHOULD diverge from")
        print("  the LP on warmup probs (d).")
    print()
    print("  The fair 'our_approach vs paper [5]' comparison is (c) vs (e):")
    print("  both pipelines use the constraint-trained model, and the only")
    print("  difference is targeted_correction (project greedy) vs paper [5] LP.")
    print()
    print("  What we've been reporting as 'our_approach vs LP' is actually (c) vs (d):")
    print("  a confounded comparison that mixes two variables (model + Phase-2).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
