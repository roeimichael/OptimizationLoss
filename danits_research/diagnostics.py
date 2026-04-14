"""
Why is the LP barely changing predictions? Diagnostic script.

Hypothesis 1: the probabilities we feed to the LP come from a model that
was already trained with the constraint loss, so its posterior already
prefers feasible assignments. The LP then has nothing to do.

Hypothesis 2: even a pure warmup (CE-only) model is so confident on the
easy majority class (NV) that only a tiny handful of samples are "in play"
for re-assignment, so the LP changes few predictions.

This script takes two runs from the archive:
    (A) a `heuristic/...` run -> warmup-only probabilities (pure CE,
        no constraint loss during training).
    (B) a `our_approach/...` run -> probabilities from a model trained
        with the full constraint loss.

For each, it computes:
    * raw-argmax class counts (what the model would do unconstrained)
    * probability-distribution summary (entropy, top-1 conf, MEL mass)
    * how "in play" MEL is (rank distribution of MEL in the posterior)
    * LP output under identity and moderate MEL-priority costs
    * number of predictions the LP flips vs the raw argmax
    * comparison of (A) vs (B) prob distributions on the same samples

Run from repo root:
    python -m danits_research.diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from danits_research import (
    DERMMNIST_IDENTITY,
    DERMMNIST_MEL_PRIORITY_MODERATE,
    build_psi_phi_from_percentages,
    solve_lp_assignment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = REPO_ROOT / "archive_experiments" / "dermmnist"

# Use the same constraint ratio (c05_03 = feature_pct 0.5, target_pct 0.3)
# across both runs so the comparison is apples-to-apples.
WARMUP_RUN = (
    ARCHIVE / "heuristic" / "ResNet18" / "c05_03" / "bs32" / "final_predictions.csv"
)
CONSTRAINT_RUN = (
    ARCHIVE / "our_approach" / "MobileNetV3" / "c04_02" / "kl0.5" / "final_predictions.csv"
)


def _load_run(csv_path: Path) -> dict:
    cfg_path = csv_path.parent / "config.json"
    preds = pd.read_csv(csv_path)
    cfg = json.loads(cfg_path.read_text())
    n_classes = int(cfg["dataset_config"]["num_classes"])
    prob_cols = [f"Prob_Class_{i}" for i in range(n_classes)]
    probs = preds[prob_cols].to_numpy(dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "label": str(csv_path.relative_to(ARCHIVE)),
        "probs": probs,
        "y_true": preds["True_Label"].to_numpy(dtype=np.int64),
        "y_pred_saved": preds["Predicted_Label"].to_numpy(dtype=np.int64),
        "groups": preds["Group_ID"].to_numpy(),
        "n_classes": n_classes,
        "constrained_class": int(cfg["dataset_config"]["constrained_class"]),
        "feature_pct": float(cfg["constraint"][0]),
        "target_pct": float(cfg["constraint"][1]),
        "methodology": cfg.get("methodology", "?"),
    }


def _counts(y, n):
    return {int(i): int((y == i).sum()) for i in range(n)}


def _entropy(probs: np.ndarray) -> np.ndarray:
    """Per-sample entropy in bits, clipped log."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def _mel_rank(probs: np.ndarray, mel_idx: int) -> np.ndarray:
    """For each sample, the rank of MEL in the posterior (0 = top-1)."""
    order = np.argsort(-probs, axis=1)  # descending
    ranks = np.empty(len(probs), dtype=np.int64)
    for s in range(len(probs)):
        ranks[s] = int(np.where(order[s] == mel_idx)[0][0])
    return ranks


def _recall(y_true, y_pred, cls):
    mask = y_true == cls
    if not mask.any():
        return float("nan")
    return float((y_pred[mask] == cls).mean())


def _mel_precision(y_true, y_pred, cls):
    mask = y_pred == cls
    if not mask.any():
        return float("nan")
    return float((y_true[mask] == cls).mean())


def _summary_block(run: dict) -> None:
    probs = run["probs"]
    y_true = run["y_true"]
    y_pred_saved = run["y_pred_saved"]
    groups = run["groups"]
    n_classes = run["n_classes"]
    mel = run["constrained_class"]
    N = probs.shape[0]

    print(f"\n=== RUN: {run['label']}  (methodology={run['methodology']}) ===")
    print(f"    N={N}, C={n_classes}, constrained_class(MEL)={mel}")
    print(f"    constraint (feature_pct, target_pct) = "
          f"({run['feature_pct']}, {run['target_pct']})")
    print(f"    true class prior   : {_counts(y_true, n_classes)}")
    print(f"    saved prediction   : {_counts(y_pred_saved, n_classes)}")

    # ---- raw argmax (what the model would do with NO constraint) ----
    y_raw = probs.argmax(axis=1)
    print(f"    RAW argmax counts  : {_counts(y_raw, n_classes)}")
    print(f"    RAW argmax acc     : {(y_raw == y_true).mean():.4f}")
    print(f"    RAW MEL recall     : {_recall(y_true, y_raw, mel):.4f}")
    print(f"    RAW MEL precision  : {_mel_precision(y_true, y_raw, mel):.4f}")

    # ---- how many samples does raw-argmax put into MEL vs Psi/Phi ----
    psi, phi = build_psi_phi_from_percentages(
        y_true=y_true, groups=groups, n_classes=n_classes,
        constrained_class=mel,
        feature_pct=run["feature_pct"],
        target_pct=run["target_pct"],
    )
    psi_mel = psi[mel]
    print(f"    Psi(MEL)={psi_mel}, Phi(MEL) per group = "
          f"{ {g: bounds[mel] for g, bounds in phi.items()} }")
    raw_mel = int((y_raw == mel).sum())
    print(f"    RAW MEL count      : {raw_mel}  "
          f"(Psi(MEL)={psi_mel} -> "
          f"{'BINDING - LP MUST reduce' if raw_mel > psi_mel else 'SLACK - LP has nothing to do on MEL'})")

    # ---- probability distribution health ----
    ent = _entropy(probs)
    top1 = probs.max(axis=1)
    print(f"    posterior entropy  : mean={ent.mean():.3f} bits  "
          f"median={np.median(ent):.3f}  p90={np.quantile(ent, 0.9):.3f}")
    print(f"    top-1 probability  : mean={top1.mean():.3f}  "
          f"median={np.median(top1):.3f}  "
          f"frac > 0.95 = {(top1 > 0.95).mean():.3f}")

    # ---- MEL mass analysis ----
    mel_prob = probs[:, mel]
    print(f"    P(MEL) over test   : mean={mel_prob.mean():.3f}  "
          f"median={np.median(mel_prob):.3f}  "
          f"max={mel_prob.max():.3f}")
    ranks = _mel_rank(probs, mel)
    print(f"    MEL rank           : top-1={int((ranks == 0).sum())}  "
          f"top-2={int((ranks <= 1).sum())}  "
          f"top-3={int((ranks <= 2).sum())}  "
          f"last={int((ranks == n_classes - 1).sum())}")

    # "In play for MEL" = samples where MEL is a plausible alternative (top-3)
    # but not the top-1. These are the samples an LP could flip TO MEL.
    in_play = int(((ranks <= 2) & (ranks > 0)).sum())
    print(f"    samples 'in play for MEL' (MEL in top-3, not top-1): {in_play}")

    # ---- apply LP under identity and moderate-priority cost ----
    for preset_name, omega in [
        ("identity",  DERMMNIST_IDENTITY),
        ("moderate",  DERMMNIST_MEL_PRIORITY_MODERATE),
    ]:
        lp = solve_lp_assignment(
            y_proba=probs, groups=groups, cost_matrix=omega,
            psi=psi, phi=phi, verbose=False,
        )
        y_lp = lp.y_pred
        flips_vs_raw = int((y_lp != y_raw).sum())
        flips_vs_saved = int((y_lp != y_pred_saved).sum())
        lp_counts = _counts(y_lp, n_classes)
        lp_acc = float((y_lp == y_true).mean())
        lp_mel_rec = _recall(y_true, y_lp, mel)

        # Where do flips happen?
        if flips_vs_raw > 0:
            flip_mask = y_lp != y_raw
            from_classes = _counts(y_raw[flip_mask], n_classes)
            to_classes = _counts(y_lp[flip_mask], n_classes)
        else:
            from_classes = to_classes = {}

        print(f"    LP[{preset_name:>9s}]: status={lp.status}  "
              f"flips vs raw argmax={flips_vs_raw:>4d}  "
              f"flips vs saved={flips_vs_saved:>4d}")
        print(f"                 counts={lp_counts}  "
              f"acc={lp_acc:.4f}  MEL recall={lp_mel_rec:.4f}  "
              f"MEL count={lp_counts[mel]}")
        if flips_vs_raw:
            print(f"                 moved FROM classes: {from_classes}")
            print(f"                 moved TO   classes: {to_classes}")


def main():
    if not WARMUP_RUN.exists():
        print(f"missing: {WARMUP_RUN}")
        return 1
    if not CONSTRAINT_RUN.exists():
        print(f"missing: {CONSTRAINT_RUN}")
        return 1

    print("#" * 72)
    print("# DIAGNOSIS: why is the LP barely changing anything?")
    print("#" * 72)

    warmup = _load_run(WARMUP_RUN)
    _summary_block(warmup)

    constraint = _load_run(CONSTRAINT_RUN)
    _summary_block(constraint)

    # --------- head-to-head comparison of the two runs -----------
    print("\n" + "#" * 72)
    print("# Head-to-head: warmup-only vs constraint-trained posterior")
    print("#" * 72)
    # Note: these are DIFFERENT backbones (RN18 vs MV3) and DIFFERENT
    # constraint ratios (c05_03 vs c04_02). We cannot directly compare
    # probabilities sample-by-sample — only distributional properties.
    p_warm = warmup["probs"]
    p_cons = constraint["probs"]
    mel_w = warmup["constrained_class"]
    mel_c = constraint["constrained_class"]

    print(f"                           | warmup-only     | constraint-trained")
    print(f"  mean top-1 prob          | {p_warm.max(1).mean():>14.4f}  "
          f"| {p_cons.max(1).mean():>14.4f}")
    print(f"  mean entropy (bits)      | {_entropy(p_warm).mean():>14.4f}  "
          f"| {_entropy(p_cons).mean():>14.4f}")
    print(f"  mean P(MEL)              | {p_warm[:, mel_w].mean():>14.4f}  "
          f"| {p_cons[:, mel_c].mean():>14.4f}")
    print(f"  frac top1 = MEL          | {(p_warm.argmax(1) == mel_w).mean():>14.4f}  "
          f"| {(p_cons.argmax(1) == mel_c).mean():>14.4f}")
    print(f"  frac P(MEL) > 0.10       | {(p_warm[:, mel_w] > 0.10).mean():>14.4f}  "
          f"| {(p_cons[:, mel_c] > 0.10).mean():>14.4f}")
    print(f"  frac P(MEL) > 0.30       | {(p_warm[:, mel_w] > 0.30).mean():>14.4f}  "
          f"| {(p_cons[:, mel_c] > 0.30).mean():>14.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
