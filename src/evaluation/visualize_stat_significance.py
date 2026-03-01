"""
Visualize statistical significance results:
 - Bar charts for accuracy, F1, kappa, ECE, Brier
 - Class-4 prediction analysis (TP/FP per model, aggregate)
 - Ensemble vs individual comparison
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.posthoc_adjustment import adjust_predictions_to_constraint, enforce_local_constraints
from src.training.metrics import compute_ece
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# ── Paths (relative to project root) ──────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent.parent
SEED_DIR = BASE / "archive_experiments" / "dermmnist" / "statistical_significance" / "ResNet18" / "c05_03"
HEUR_DIR = BASE / "archive_experiments" / "dermmnist" / "heuristic" / "ResNet18" / "c05_03" / "baseline"
OUT_DIR  = BASE / "archive_experiments" / "dermmnist" / "statistical_significance" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 7
CONSTRAINED_CLASS = 4
GLOBAL_LIMIT = 67
LOCAL_LIMITS = {0: 72, 1: 40}

COLORS = {
    "heuristic": "#888888",
    "seeds":     "#4C72B0",
    "ensemble":  "#DD8452",
    "seed_individual": ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
}


def load_preds(path):
    df = pd.read_csv(path)
    y_true = df["True_Label"].values
    y_pred = df["Predicted_Label"].values
    prob_cols = [f"Prob_Class_{c}" for c in range(NUM_CLASSES)]
    y_proba = df[prob_cols].values.astype(np.float64)
    group_ids = df["Group_ID"].values
    return y_true, y_pred, y_proba, group_ids



def main():
    # ── Load all data ──────────────────────────────────────────────────────
    h_true, h_pred, h_proba, h_groups = load_preds(HEUR_DIR / "final_predictions.csv")

    seeds_data = []
    all_proba = []
    for i in range(1, 6):
        t, p, pr, g = load_preds(SEED_DIR / f"seed_{i}" / "final_predictions.csv")
        seeds_data.append((t, p, pr, g))
        all_proba.append(pr)

    y_true_ref = seeds_data[0][0]
    g_ref = seeds_data[0][3]

    # Ensemble
    avg_proba = np.mean(all_proba, axis=0)
    avg_proba = avg_proba / avg_proba.sum(axis=1, keepdims=True)
    ens_pred = np.argmax(avg_proba, axis=1)

    # ── Compute metrics for all 7 ─────────────────────────────────────────
    def metrics(y_true, y_pred, y_proba):
        from sklearn.metrics import cohen_kappa_score
        acc = np.mean(y_true == y_pred)
        _, _, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        _, _, f1w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)
        ece = compute_ece(y_true, y_proba)
        oh = np.zeros_like(y_proba); oh[np.arange(len(y_true)), y_true] = 1.0
        brier = np.mean(np.sum((y_proba - oh) ** 2, axis=1))
        return {"acc": acc, "f1_macro": f1m, "f1_weighted": f1w,
                "kappa": kappa, "ece": ece, "brier": brier}

    # Saturated ensemble (post-hoc fill to constraint limit)
    sat_pred = ens_pred.copy()
    sat_proba = avg_proba.copy()
    sat_pred, _ = adjust_predictions_to_constraint(
        sat_pred, sat_proba, GLOBAL_LIMIT, CONSTRAINED_CLASS)
    local_con = {
        gid: {c: (LOCAL_LIMITS[gid] if c == CONSTRAINED_CLASS else 1e9)
              for c in range(NUM_CLASSES)}
        for gid in LOCAL_LIMITS
    }
    sat_pred, _ = enforce_local_constraints(
        sat_pred, sat_proba, g_ref, local_con, CONSTRAINED_CLASS)

    h_m = metrics(h_true, h_pred, h_proba)
    s_ms = [metrics(d[0], d[1], d[2]) for d in seeds_data]
    e_m = metrics(y_true_ref, ens_pred, avg_proba)
    sat_m = metrics(y_true_ref, sat_pred, sat_proba)

    names  = ["Heuristic"] + [f"Seed {i}" for i in range(1, 6)] + ["Ensemble", "Ens+Sat"]
    all_ms = [h_m] + s_ms + [e_m, sat_m]

    # ── Class 4 analysis ───────────────────────────────────────────────────
    def c4_stats(y_true, y_pred, group_ids):
        pred_c4 = (y_pred == CONSTRAINED_CLASS)
        true_c4 = (y_true == CONSTRAINED_CLASS)
        tp = (pred_c4 & true_c4).sum()
        fp = (pred_c4 & ~true_c4).sum()
        fn = (~pred_c4 & true_c4).sum()
        total_pred = pred_c4.sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / true_c4.sum() if true_c4.sum() > 0 else 0
        # Per-group
        g0_count = (pred_c4 & (group_ids == 0)).sum()
        g1_count = (pred_c4 & (group_ids == 1)).sum()
        return {"tp": tp, "fp": fp, "fn": fn, "total_pred": total_pred,
                "precision": prec, "recall": rec, "g0": g0_count, "g1": g1_count}

    h_c4 = c4_stats(h_true, h_pred, h_groups)
    s_c4s = [c4_stats(d[0], d[1], d[3]) for d in seeds_data]
    e_c4 = c4_stats(y_true_ref, ens_pred, g_ref)
    sat_c4 = c4_stats(y_true_ref, sat_pred, g_ref)
    all_c4 = [h_c4] + s_c4s + [e_c4, sat_c4]

    # ── Aggregate across 5 seeds ───────────────────────────────────────────
    # Which samples predicted as class 4 by ANY seed, by ALL seeds, total counts
    seed_c4_masks = [(d[1] == CONSTRAINED_CLASS) for d in seeds_data]
    any_seed_c4 = np.any(seed_c4_masks, axis=0)
    all_seeds_c4 = np.all(seed_c4_masks, axis=0)
    total_c4_across = sum(m.sum() for m in seed_c4_masks)  # total predictions summed

    true_c4_mask = (y_true_ref == CONSTRAINED_CLASS)
    # Samples predicted c4 by any seed: how many correct?
    any_tp = (any_seed_c4 & true_c4_mask).sum()
    any_fp = (any_seed_c4 & ~true_c4_mask).sum()
    # Samples predicted c4 by all seeds: how many correct?
    all_tp = (all_seeds_c4 & true_c4_mask).sum()
    all_fp = (all_seeds_c4 & ~true_c4_mask).sum()

    # ── FIGURE 1: Main metrics bar chart ───────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Statistical Significance: 5 Seeds + Heuristic + Ensemble\nConstraint [0.5, 0.3] on Class 4 (DermaMNIST-C)", fontsize=14, fontweight="bold")

    metric_keys = ["acc", "f1_macro", "f1_weighted", "kappa", "ece", "brier"]
    metric_labels = ["Accuracy", "F1-Macro", "F1-Weighted", "Cohen's Kappa", "ECE (lower=better)", "Brier Score (lower=better)"]

    for idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx // 3][idx % 3]
        vals = [m[key] for m in all_ms]
        colors = [COLORS["heuristic"]] + [COLORS["seed_individual"][i] for i in range(5)] + [COLORS["ensemble"], "#E45756"]
        bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        # Seed mean line
        seed_vals = [m[key] for m in s_ms]
        seed_mean = np.mean(seed_vals)
        ax.axhline(y=seed_mean, color=COLORS["seeds"], linestyle="--", alpha=0.7, linewidth=1.5, label=f"Seed mean: {seed_mean:.3f}")

        ax.set_ylabel(label)
        ax.set_title(label, fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=7, loc="lower right")

        # For ECE and Brier, lower is better
        if key in ("ece", "brier"):
            ax.set_ylim(0, max(vals) * 1.25)
        else:
            ax.set_ylim(min(vals) * 0.9, max(vals) * 1.08)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'metrics_comparison.png'}")
    plt.close()

    # ── FIGURE 2: Class 4 TP/FP breakdown ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Class 4 Prediction Analysis (Constrained Class)\nGlobal Limit: 67 | True Class 4 Samples: 223", fontsize=13, fontweight="bold")

    # Panel A: TP vs FP stacked bar
    ax = axes[0]
    tps = [c["tp"] for c in all_c4]
    fps = [c["fp"] for c in all_c4]
    x = np.arange(len(names))
    w = 0.6
    bars_tp = ax.bar(x, tps, w, label="True Positives (correct)", color="#55A868", edgecolor="black", linewidth=0.5)
    bars_fp = ax.bar(x, fps, w, bottom=tps, label="False Positives (wrong)", color="#C44E52", edgecolor="black", linewidth=0.5)
    ax.axhline(y=GLOBAL_LIMIT, color="black", linestyle="--", linewidth=1.5, label=f"Constraint limit ({GLOBAL_LIMIT})")

    for i, (tp, fp) in enumerate(zip(tps, fps)):
        ax.text(i, tp / 2, str(tp), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        if fp > 3:
            ax.text(i, tp + fp / 2, str(fp), ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Class 4 Predictions: TP vs FP", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel B: Precision and Recall
    ax = axes[1]
    precs = [c["precision"] for c in all_c4]
    recs = [c["recall"] for c in all_c4]
    w2 = 0.35
    ax.bar(x - w2/2, precs, w2, label="Precision", color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + w2/2, recs, w2, label="Recall", color="#DD8452", edgecolor="black", linewidth=0.5)

    for i in range(len(names)):
        ax.text(i - w2/2, precs[i] + 0.01, f"{precs[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w2/2, recs[i] + 0.01, f"{recs[i]:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Class 4 Precision & Recall", fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9)

    # Panel C: Seed agreement analysis
    ax = axes[2]
    # How many seeds agree on each class-4 prediction?
    vote_counts = np.sum(seed_c4_masks, axis=0)  # 0-5 for each sample
    # Among samples predicted c4 by at least 1 seed: vote distribution
    at_least_1 = vote_counts >= 1
    vote_of_predicted = vote_counts[at_least_1]
    true_labels_predicted = y_true_ref[at_least_1]

    bins = [1, 2, 3, 4, 5]
    correct_counts = []
    wrong_counts = []
    for v in bins:
        mask_v = vote_of_predicted == v
        correct_v = ((true_labels_predicted[mask_v] == CONSTRAINED_CLASS)).sum()
        wrong_v = mask_v.sum() - correct_v
        correct_counts.append(correct_v)
        wrong_counts.append(wrong_v)

    bx = np.arange(len(bins))
    ax.bar(bx, correct_counts, 0.6, label="Truly Class 4", color="#55A868", edgecolor="black", linewidth=0.5)
    ax.bar(bx, wrong_counts, 0.6, bottom=correct_counts, label="Not Class 4 (FP)", color="#C44E52", edgecolor="black", linewidth=0.5)

    for i, (c, w) in enumerate(zip(correct_counts, wrong_counts)):
        total = c + w
        if total > 0:
            ax.text(i, total + 0.5, f"{total}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            if c > 0:
                ax.text(i, c / 2, str(c), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
            if w > 2:
                ax.text(i, c + w / 2, str(w), ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    ax.set_xticks(bx)
    ax.set_xticklabels([f"{v}/5 seeds" for v in bins])
    ax.set_ylabel("Samples")
    ax.set_title("Seed Agreement on Class 4 Predictions", fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "class4_analysis.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'class4_analysis.png'}")
    plt.close()

    # ── FIGURE 3: Ensemble vs Individual scatter ───────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Ensemble vs Individual Models", fontsize=13, fontweight="bold")

    for i, m in enumerate(s_ms):
        ax.scatter(m["acc"], m["f1_macro"], color=COLORS["seed_individual"][i],
                   s=120, zorder=5, edgecolor="black", linewidth=0.8, label=f"Seed {i+1}")
    ax.scatter(h_m["acc"], h_m["f1_macro"], color=COLORS["heuristic"],
               s=180, marker="s", zorder=5, edgecolor="black", linewidth=1.2, label="Heuristic")
    ax.scatter(e_m["acc"], e_m["f1_macro"], color=COLORS["ensemble"],
               s=250, marker="*", zorder=6, edgecolor="black", linewidth=1.2, label="Ensemble")
    ax.scatter(sat_m["acc"], sat_m["f1_macro"], color="#E45756",
               s=250, marker="D", zorder=6, edgecolor="black", linewidth=1.2, label="Ens+Saturated")

    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Accuracy vs F1-Macro for All Models", fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "ensemble_vs_individual.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR / 'ensemble_vs_individual.png'}")
    plt.close()

    # ── Print class 4 analysis summary ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("  CLASS 4 PREDICTION ANALYSIS")
    print("=" * 80)
    print(f"\nTrue class 4 samples in test set: 223")
    print(f"Constraint limit (global): {GLOBAL_LIMIT}")
    print()

    print(f"{'Model':<14} {'Pred C4':>8} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Recall':>7}")
    print("-" * 55)
    for name, c4 in zip(names, all_c4):
        print(f"{name:<14} {c4['total_pred']:>8} {c4['tp']:>5} {c4['fp']:>5} {c4['fn']:>5} {c4['precision']:>7.4f} {c4['recall']:>7.4f}")

    print(f"\n--- Aggregate across 5 seeds ---")
    print(f"Total class-4 predictions (summed): {total_c4_across}")
    print(f"Unique samples predicted as class 4 by ANY seed: {any_seed_c4.sum()}")
    print(f"  - of which truly class 4 (TP): {any_tp}")
    print(f"  - of which NOT class 4 (FP):   {any_fp}")
    print(f"Samples predicted as class 4 by ALL 5 seeds: {all_seeds_c4.sum()}")
    print(f"  - of which truly class 4 (TP): {all_tp}")
    print(f"  - of which NOT class 4 (FP):   {all_fp}")
    if all_seeds_c4.sum() > 0:
        print(f"  - consensus precision: {all_tp / all_seeds_c4.sum():.4f}")

    print(f"\n--- Ensemble (probability averaging) ---")
    print(f"Ensemble predicted class 4: {e_c4['total_pred']} (well within limit of {GLOBAL_LIMIT})")
    print(f"  TP={e_c4['tp']}, FP={e_c4['fp']}")
    print(f"  Precision: {e_c4['precision']:.4f}  Recall: {e_c4['recall']:.4f}")

    print(f"\n--- Saturated Ensemble (post-hoc fill to limit) ---")
    print(f"Ens+Saturated predicted class 4: {sat_c4['total_pred']} (filled to limit of {GLOBAL_LIMIT})")
    print(f"  TP={sat_c4['tp']}, FP={sat_c4['fp']}")
    print(f"  Precision: {sat_c4['precision']:.4f}  Recall: {sat_c4['recall']:.4f}")
    print(f"  Flipped {sat_c4['total_pred'] - e_c4['total_pred']} additional samples to class 4")
    added_tp = sat_c4['tp'] - e_c4['tp']
    added_fp = sat_c4['fp'] - e_c4['fp']
    print(f"  Of the {sat_c4['total_pred'] - e_c4['total_pred']} added: {added_tp} were truly class 4, {added_fp} were false positives")

    # Compare ensemble quality to seeds
    seed_precs = [c["precision"] for c in s_c4s]
    seed_recs = [c["recall"] for c in s_c4s]
    print(f"\n--- All Variants vs Seeds (class 4) ---")
    print(f"  Seed mean precision: {np.mean(seed_precs):.4f} +/- {np.std(seed_precs, ddof=1):.4f}")
    print(f"  Ensemble precision:  {e_c4['precision']:.4f}  ({'better' if e_c4['precision'] > np.mean(seed_precs) else 'worse'})")
    print(f"  Ens+Sat precision:   {sat_c4['precision']:.4f}  ({'better' if sat_c4['precision'] > np.mean(seed_precs) else 'worse'})")
    print(f"  Seed mean recall:    {np.mean(seed_recs):.4f} +/- {np.std(seed_recs, ddof=1):.4f}")
    print(f"  Ensemble recall:     {e_c4['recall']:.4f}  ({'better' if e_c4['recall'] > np.mean(seed_recs) else 'worse'})")
    print(f"  Ens+Sat recall:      {sat_c4['recall']:.4f}  ({'better' if sat_c4['recall'] > np.mean(seed_recs) else 'worse'})")

    print(f"\n--- Seed agreement breakdown ---")
    for v in [1, 2, 3, 4, 5]:
        mask_v = vote_counts == v
        n_total = mask_v.sum()
        n_true = (mask_v & true_c4_mask).sum()
        n_false = n_total - n_true
        pct = n_true / n_total * 100 if n_total > 0 else 0
        print(f"  {v}/5 seeds agree class 4: {n_total:>4} samples ({n_true} true, {n_false} false, {pct:.0f}% correct)")


if __name__ == "__main__":
    main()
