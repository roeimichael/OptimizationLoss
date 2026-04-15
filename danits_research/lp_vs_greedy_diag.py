"""
Is the LP really doing anything the greedy can't?

Three experiments:

(1) Sample-level diff: how many predictions differ between LP and greedy
    on the SAME probability input? If zero or tiny, they're effectively
    the same algorithm at this operating point.

(2) Multi-constraint stress test: add a second constraint on NV (the
    majority class) in addition to MEL. Under identity cost this forces
    the LP to reallocate hundreds of samples — a regime where the LP's
    globally-optimal nature should actually show up.

(3) Tight constraint sensitivity: sweep Psi(MEL) from very loose to very
    tight and plot accuracy / MEL recall / LP vs greedy diff. Shows where
    the LP's value actually lives.

Run:
    python -m danits_research.lp_vs_greedy_diag
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    DERMMNIST_MEL_PRIORITY_MODERATE,
    build_psi_phi_from_percentages,
    solve_greedy_assignment,
    solve_lp_assignment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR = (
    REPO_ROOT / "archive_experiments" / "dermmnist" / "heuristic" /
    "ResNet18" / "c05_03" / "bs32" / "final_predictions.csv"
)


def _load():
    df = pd.read_csv(ANCHOR)
    cfg = json.loads((ANCHOR.parent / "config.json").read_text())
    n_classes = int(cfg["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    probs = df[prob_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "probs": probs,
        "y_true": df["True_Label"].to_numpy(dtype=np.int64),
        "y_pred_greedy_saved": df["Predicted_Label"].to_numpy(dtype=np.int64),
        "groups": df["Group_ID"].to_numpy(),
        "n_classes": n_classes,
        "mel": int(cfg["dataset_config"]["constrained_class"]),
        "feature_pct": float(cfg["constraint"][0]),
        "target_pct": float(cfg["constraint"][1]),
    }


def _counts(y, n):
    return {int(i): int((y == i).sum()) for i in range(n)}


def _recall(y_true, y_pred, c):
    m = y_true == c
    if not m.any():
        return float("nan")
    return float((y_pred[m] == c).mean())


def _check_feasible(y_pred, groups, psi, phi, n_classes):
    for i, b in enumerate(psi):
        if b is None:
            continue
        if int((y_pred == i).sum()) > b:
            return False
    for g, bounds in phi.items():
        mask = groups == g
        for i, b in enumerate(bounds):
            if b is None:
                continue
            if int(((y_pred == i) & mask).sum()) > b:
                return False
    return True


def run_experiment_1_sample_diff(data):
    print("=" * 78)
    print("(1) SAMPLE-LEVEL DIFF: LP vs greedy under the current single-constraint setup")
    print("=" * 78)

    probs = data["probs"]
    y_true = data["y_true"]
    groups = data["groups"]
    n = data["n_classes"]
    mel = data["mel"]

    psi, phi = build_psi_phi_from_percentages(
        y_true, groups, n, mel, data["feature_pct"], data["target_pct"],
    )
    print(f"Psi = {psi}")
    print(f"Phi = {phi}")

    lp = solve_lp_assignment(
        y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
        psi=psi, phi=phi,
    )
    heu = solve_greedy_assignment(
        y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
        psi=psi, phi=phi,
    )
    y_lp = lp.y_pred
    y_heu = heu.y_pred
    y_argmax = probs.argmax(axis=1)

    diff_lp_heu = int((y_lp != y_heu).sum())
    diff_lp_argmax = int((y_lp != y_argmax).sum())
    diff_heu_argmax = int((y_heu != y_argmax).sum())

    print(f"\nN samples total           : {len(y_true)}")
    print(f"LP != our greedy          : {diff_lp_heu}  (if 0, the two algorithms are equivalent here)")
    print(f"LP != raw argmax          : {diff_lp_argmax}")
    print(f"our greedy != raw argmax  : {diff_heu_argmax}")

    if diff_lp_heu == 0:
        print("\n  => LP and greedy are producing identical assignments at this operating point.")
        print("     This is NOT a bug. Under identity cost with a single constrained class,")
        print("     both algorithms reduce to 'keep the top-Psi(MEL) samples by P(MEL),")
        print("     argmax the rest'. See experiments (2) and (3) for regimes where they diverge.")
    else:
        # Locate the disagreeing samples and show which classes they involve.
        mask = y_lp != y_heu
        idxs = np.nonzero(mask)[0][:10]
        print(f"\n  Sample rows where LP and greedy disagree (first 10):")
        print(f"  {'idx':>5s}  {'LP':>4s}  {'HEU':>4s}  {'argmax':>6s}  {'true':>4s}  {'top-3 probs'}")
        for i in idxs:
            top3 = np.argsort(-probs[i])[:3]
            top3_str = ", ".join(f"{c}:{probs[i, c]:.3f}" for c in top3)
            print(f"  {i:>5d}  {y_lp[i]:>4d}  {y_heu[i]:>4d}  "
                  f"{y_argmax[i]:>6d}  {y_true[i]:>4d}  {top3_str}")

    # Also compare against the project's saved greedy result
    saved = data["y_pred_greedy_saved"]
    diff_lp_saved = int((y_lp != saved).sum())
    print(f"\nLP != project's saved greedy: {diff_lp_saved}")


def run_experiment_2_multi_constraint(data):
    print("\n" + "=" * 78)
    print("(2) MULTI-CONSTRAINT STRESS: bound BOTH MEL and NV at the same time")
    print("=" * 78)
    print("Under identity cost with multiple tight constraints, the LP must")
    print("make global trade-offs across classes; the greedy allocates class-")
    print("by-class and can get stuck with suboptimal leftovers.")

    probs = data["probs"]
    y_true = data["y_true"]
    groups = data["groups"]
    n = data["n_classes"]
    mel = data["mel"]
    nv = 5  # majority class, 1341 true samples out of 2003

    prior_mel = int((y_true == mel).sum())
    prior_nv = int((y_true == nv).sum())
    print(f"\ntrue prior: MEL={prior_mel}, NV={prior_nv}  (test N={len(y_true)})")

    # Sweep: MEL budget fixed at 0.3, NV budget tightened.
    nv_pcts = [1.00, 0.80, 0.60, 0.50, 0.40, 0.30, 0.20]
    print(f"\nNV budget %  |  NV cap  |  LP acc    greedy acc  |  LP mel rec  heu mel rec  |  "
          f"LP feas  heu feas  |  LP!=heu")
    print("-" * 100)

    for nv_pct in nv_pcts:
        psi = [None] * n
        psi[mel] = int(np.floor(0.30 * prior_mel))
        psi[nv]  = int(np.floor(nv_pct * prior_nv))

        # Construct phi so MEL has a local budget too (same as before),
        # and NV gets 50% of each group's NV count (loose local).
        phi: dict = {}
        for g in np.unique(groups):
            in_g = groups == g
            bounds = [None] * n
            bounds[mel] = int(np.floor(0.50 * int(((y_true == mel) & in_g).sum())))
            bounds[nv]  = int(np.floor(nv_pct * int(((y_true == nv) & in_g).sum())))
            phi[int(g)] = bounds

        lp = solve_lp_assignment(
            y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
            psi=psi, phi=phi,
        )
        heu = solve_greedy_assignment(
            y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
            psi=psi, phi=phi,
        )

        if lp.status != "OPTIMAL":
            print(f"  NV={nv_pct:.2f} | LP status {lp.status}")
            continue

        lp_acc = float((lp.y_pred == y_true).mean())
        heu_acc = float((heu.y_pred == y_true).mean())
        lp_rec = _recall(y_true, lp.y_pred, mel)
        heu_rec = _recall(y_true, heu.y_pred, mel)
        lp_feas = _check_feasible(lp.y_pred, groups, psi, phi, n)
        heu_feas = _check_feasible(heu.y_pred, groups, psi, phi, n)
        diff = int((lp.y_pred != heu.y_pred).sum())

        print(f"  {nv_pct:>6.2f}    |  {psi[nv]:>5d}   | "
              f" {lp_acc:.4f}    {heu_acc:.4f}   |  {lp_rec:.4f}       {heu_rec:.4f}     |  "
              f"{'OK' if lp_feas else 'FAIL':>6s}    {'OK' if heu_feas else 'FAIL':>6s}    |  {diff:>5d}")

    print("\n  Reading: if 'LP!=heu' is large and LP stays feasible while greedy doesn't,")
    print("           then the LP is doing real work the greedy cannot.")


def run_experiment_3_mel_sweep(data):
    print("\n" + "=" * 78)
    print("(3) MEL BUDGET SWEEP: how tight does the single-class budget have to be")
    print("    for LP and greedy to start diverging?")
    print("=" * 78)

    probs = data["probs"]
    y_true = data["y_true"]
    groups = data["groups"]
    n = data["n_classes"]
    mel = data["mel"]

    prior_mel = int((y_true == mel).sum())
    raw_mel_count = int(probs.argmax(axis=1).sum() == mel) + int((probs.argmax(axis=1) == mel).sum())
    # fix (typo): just count raw argmax MEL
    raw_mel_count = int((probs.argmax(axis=1) == mel).sum())
    print(f"\nraw argmax MEL count={raw_mel_count}, true MEL count={prior_mel}")
    print(f"\n  psi_pct  |  psi(MEL)  |  LP acc  heu acc  |  LP mel rec  heu mel rec  "
          f"|  LP obj    heu obj    |  LP!=heu")
    print("  " + "-" * 98)

    for pct in [1.00, 0.80, 0.50, 0.30, 0.20, 0.10, 0.05, 0.02]:
        psi = [None] * n
        psi[mel] = int(np.floor(pct * prior_mel))
        # symmetric local budget so it's never the binding constraint
        phi: dict = {}
        for g in np.unique(groups):
            in_g = groups == g
            bounds = [None] * n
            bounds[mel] = int(np.floor(pct * 2 * int(((y_true == mel) & in_g).sum())))
            phi[int(g)] = bounds

        lp = solve_lp_assignment(
            y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
            psi=psi, phi=phi,
        )
        heu = solve_greedy_assignment(
            y_proba=probs, groups=groups, cost_matrix=DERMMNIST_IDENTITY,
            psi=psi, phi=phi,
        )
        if lp.status != "OPTIMAL":
            print(f"  pct={pct} | LP status {lp.status}")
            continue

        lp_acc = float((lp.y_pred == y_true).mean())
        heu_acc = float((heu.y_pred == y_true).mean())
        lp_rec = _recall(y_true, lp.y_pred, mel)
        heu_rec = _recall(y_true, heu.y_pred, mel)
        diff = int((lp.y_pred != heu.y_pred).sum())

        print(f"  {pct:>5.2f}    |    {psi[mel]:>4d}    |  {lp_acc:.4f}  {heu_acc:.4f}  |  "
              f"{lp_rec:.4f}       {heu_rec:.4f}     |  "
              f"{lp.objective_value:>7.2f}  {heu.objective_value:>7.2f}  |  {diff:>5d}")


def main():
    data = _load()
    run_experiment_1_sample_diff(data)
    run_experiment_2_multi_constraint(data)
    run_experiment_3_mel_sweep(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
