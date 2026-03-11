# Statistical significance evaluation for constraint optimization experiments.
# Evaluates 1 heuristic baseline, 5 seed models, 1 ensemble (majority vote on probabilities).
# Reports accuracy, F1, constraint satisfaction, calibration, and statistical tests.

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support, confusion_matrix,
    classification_report, cohen_kappa_score
)
from src.utils.posthoc_adjustment import adjust_predictions_to_constraint, enforce_local_constraints
from src.training.metrics import compute_ece
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
SEED_DIR = BASE / "archive_experiments" / "dermmnist" / "statistical_significance" / "ResNet18" / "c05_03"
HEURISTIC_DIR = BASE / "archive_experiments" / "dermmnist" / "heuristic" / "ResNet18" / "c05_03" / "baseline"

CONSTRAINED_CLASS = 4
GLOBAL_LIMIT = 67
LOCAL_LIMITS = {0: 72, 1: 40}
NUM_CLASSES = 7


def load_predictions(path):
    df = pd.read_csv(path)
    y_true = df["True_Label"].values
    y_pred = df["Predicted_Label"].values
    prob_cols = [f"Prob_Class_{c}" for c in range(NUM_CLASSES)]
    y_proba = df[prob_cols].values.astype(np.float64)
    group_ids = df["Group_ID"].values
    return y_true, y_pred, y_proba, group_ids


def constraint_check(y_pred, group_ids):
    global_count = (y_pred == CONSTRAINED_CLASS).sum()
    global_ok = global_count <= GLOBAL_LIMIT

    local_results = {}
    for gid, limit in LOCAL_LIMITS.items():
        mask = group_ids == gid
        count = (y_pred[mask] == CONSTRAINED_CLASS).sum()
        local_results[gid] = {"count": int(count), "limit": limit, "ok": count <= limit}

    all_local_ok = all(v["ok"] for v in local_results.values())
    return {
        "global_count": int(global_count),
        "global_limit": GLOBAL_LIMIT,
        "global_ok": global_ok,
        "local": local_results,
        "all_local_ok": all_local_ok,
        "fully_satisfied": global_ok and all_local_ok,
    }


def full_metrics(y_true, y_pred, y_proba, group_ids, name):
    acc = np.mean(y_true == y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    kappa = cohen_kappa_score(y_true, y_pred)
    ece = compute_ece(y_true, y_proba)

    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    brier = np.mean(np.sum((y_proba - one_hot) ** 2, axis=1))

    confidences = np.max(y_proba, axis=1)
    correct_mask = y_true == y_pred

    constr = constraint_check(y_pred, group_ids)

    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "cohen_kappa": kappa,
        "ece": ece,
        "brier_score": brier,
        "mean_confidence": float(confidences.mean()),
        "conf_correct": float(confidences[correct_mask].mean()) if correct_mask.any() else 0,
        "conf_incorrect": float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0,
        "f1_class4": float(f1_per_class[CONSTRAINED_CLASS]),
        "constraint_satisfied": constr["fully_satisfied"],
        "global_class4_count": constr["global_count"],
        "local_g0_class4_count": constr["local"][0]["count"],
        "local_g1_class4_count": constr["local"][1]["count"],
    }


def build_ensemble(all_proba):
    avg_proba = np.mean(all_proba, axis=0)
    avg_proba = avg_proba / avg_proba.sum(axis=1, keepdims=True)
    ensemble_pred = np.argmax(avg_proba, axis=1)
    return ensemble_pred, avg_proba


def main():
    print("=" * 80)
    print("  STATISTICAL SIGNIFICANCE EVALUATION")
    print("  Constraint: [0.5, 0.3] on class 4 (DermaMNIST-C)")
    print("  Global limit: 67 | Local limits: group_0=72, group_1=40")
    print("=" * 80)

    h_true, h_pred, h_proba, h_groups = load_predictions(
        HEURISTIC_DIR / "final_predictions.csv"
    )
    heuristic_metrics = full_metrics(h_true, h_pred, h_proba, h_groups, "Heuristic")

    seed_metrics = []
    all_proba = []
    y_true_ref = None
    group_ref = None

    for seed_num in range(1, 6):
        seed_path = SEED_DIR / f"seed_{seed_num}" / "final_predictions.csv"
        y_true, y_pred, y_proba, group_ids = load_predictions(seed_path)

        if y_true_ref is None:
            y_true_ref = y_true
            group_ref = group_ids
        else:
            assert np.array_equal(y_true, y_true_ref), f"Seed {seed_num} has different labels!"

        m = full_metrics(y_true, y_pred, y_proba, group_ids, f"Seed {seed_num}")
        seed_metrics.append(m)
        all_proba.append(y_proba)

    ens_pred, ens_proba = build_ensemble(np.array(all_proba))
    ensemble_metrics = full_metrics(y_true_ref, ens_pred, ens_proba, group_ref, "Ensemble (5)")

    sat_pred = ens_pred.copy()
    sat_proba = ens_proba.copy()

    sat_pred, g_info = adjust_predictions_to_constraint(
        sat_pred, sat_proba, GLOBAL_LIMIT, CONSTRAINED_CLASS
    )
    print(f"\nPost-hoc global adjustment: {g_info}")

    local_con = {
        gid: {c: (LOCAL_LIMITS[gid] if c == CONSTRAINED_CLASS else 1e9)
              for c in range(NUM_CLASSES)}
        for gid in LOCAL_LIMITS
    }
    sat_pred, n_local = enforce_local_constraints(
        sat_pred, sat_proba, group_ref, local_con, CONSTRAINED_CLASS
    )
    if n_local > 0:
        print(f"Post-hoc local adjustments: {n_local} samples flipped")

    sat_ens_metrics = full_metrics(y_true_ref, sat_pred, sat_proba, group_ref, "Ens+Saturated")

    all_metrics = [heuristic_metrics] + seed_metrics + [ensemble_metrics, sat_ens_metrics]

    print("\n" + "-" * 80)
    print("  RESULTS SUMMARY")
    print("-" * 80)

    header = f"{'Model':<16} {'Acc':>7} {'F1-M':>7} {'F1-W':>7} {'Kappa':>7} {'ECE':>7} {'Brier':>7} {'C4':>4} {'G0':>4} {'G1':>4} {'Sat':>4}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        sat = "YES" if m["constraint_satisfied"] else "NO"
        g_mark = "" if m["global_class4_count"] <= GLOBAL_LIMIT else "*"
        l0_mark = "" if m["local_g0_class4_count"] <= LOCAL_LIMITS[0] else "*"
        l1_mark = "" if m["local_g1_class4_count"] <= LOCAL_LIMITS[1] else "*"
        print(
            f"{m['name']:<16} "
            f"{m['accuracy']:>7.4f} "
            f"{m['f1_macro']:>7.4f} "
            f"{m['f1_weighted']:>7.4f} "
            f"{m['cohen_kappa']:>7.4f} "
            f"{m['ece']:>7.4f} "
            f"{m['brier_score']:>7.4f} "
            f"{m['global_class4_count']:>3}{g_mark} "
            f"{m['local_g0_class4_count']:>3}{l0_mark} "
            f"{m['local_g1_class4_count']:>3}{l1_mark} "
            f"{sat:>4}"
        )

    print("\n" + "-" * 80)
    print("  STATISTICAL ANALYSIS (5 Seeds)")
    print("-" * 80)

    key_metrics = ["accuracy", "f1_macro", "f1_weighted", "cohen_kappa", "ece",
                   "brier_score", "f1_class4", "global_class4_count"]
    values = {k: [m[k] for m in seed_metrics] for k in key_metrics}

    print(f"\n{'Metric':<24} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9} {'95% CI':>18}")
    print("-" * 80)
    for k in key_metrics:
        v = np.array(values[k])
        mean = v.mean()
        std = v.std(ddof=1)
        ci = stats.t.interval(0.95, df=len(v) - 1, loc=mean, scale=std / np.sqrt(len(v)))
        print(
            f"{k:<24} {mean:>9.4f} {std:>9.4f} {v.min():>9.4f} {v.max():>9.4f} "
            f"[{ci[0]:>7.4f}, {ci[1]:>7.4f}]"
        )

    print("\n" + "-" * 80)
    print("  PAIRED COMPARISONS VS HEURISTIC")
    print("-" * 80)

    h_acc = heuristic_metrics["accuracy"]
    h_f1 = heuristic_metrics["f1_macro"]

    seed_accs = np.array([m["accuracy"] for m in seed_metrics])
    seed_f1s = np.array([m["f1_macro"] for m in seed_metrics])

    t_acc, p_acc = stats.ttest_1samp(seed_accs, h_acc)
    t_f1, p_f1 = stats.ttest_1samp(seed_f1s, h_f1)

    print(f"\nHeuristic accuracy:  {h_acc:.4f}")
    print(f"Seeds mean accuracy: {seed_accs.mean():.4f} +/- {seed_accs.std(ddof=1):.4f}")
    print(f"  t-statistic: {t_acc:.4f}, p-value: {p_acc:.4f} {'(significant at p<0.05)' if p_acc < 0.05 else '(not significant)'}")

    print(f"\nHeuristic F1-macro:  {h_f1:.4f}")
    print(f"Seeds mean F1-macro: {seed_f1s.mean():.4f} +/- {seed_f1s.std(ddof=1):.4f}")
    print(f"  t-statistic: {t_f1:.4f}, p-value: {p_f1:.4f} {'(significant at p<0.05)' if p_f1 < 0.05 else '(not significant)'}")

    print("\n" + "-" * 80)
    print("  ENSEMBLE COMPARISON")
    print("-" * 80)

    ens = ensemble_metrics
    best_seed = max(seed_metrics, key=lambda m: m["accuracy"])

    print(f"\n{'':20} {'Accuracy':>10} {'F1-macro':>10} {'F1-weighted':>12} {'Kappa':>8} {'Constraint':>12}")
    print("-" * 75)
    print(f"{'Heuristic':<20} {h_acc:>10.4f} {h_f1:>10.4f} {heuristic_metrics['f1_weighted']:>12.4f} {heuristic_metrics['cohen_kappa']:>8.4f} {'YES' if heuristic_metrics['constraint_satisfied'] else 'NO':>12}")
    print(f"{'Best Seed ('+best_seed['name']+')':<20} {best_seed['accuracy']:>10.4f} {best_seed['f1_macro']:>10.4f} {best_seed['f1_weighted']:>12.4f} {best_seed['cohen_kappa']:>8.4f} {'YES' if best_seed['constraint_satisfied'] else 'NO':>12}")
    print(f"{'Ensemble (5)':<20} {ens['accuracy']:>10.4f} {ens['f1_macro']:>10.4f} {ens['f1_weighted']:>12.4f} {ens['cohen_kappa']:>8.4f} {'YES' if ens['constraint_satisfied'] else 'NO':>12}")

    print("\n" + "-" * 80)
    print("  CONSTRAINT SATISFACTION DETAILS")
    print("-" * 80)

    print(f"\n{'Model':<16} {'Global(<=67)':>14} {'G0(<=72)':>12} {'G1(<=40)':>12} {'All OK':>8}")
    print("-" * 65)
    for m in all_metrics:
        g_ok = "OK" if m["global_class4_count"] <= GLOBAL_LIMIT else "VIOL"
        l0_ok = "OK" if m["local_g0_class4_count"] <= LOCAL_LIMITS[0] else "VIOL"
        l1_ok = "OK" if m["local_g1_class4_count"] <= LOCAL_LIMITS[1] else "VIOL"
        all_ok = "YES" if m["constraint_satisfied"] else "NO"
        print(
            f"{m['name']:<16} "
            f"{m['global_class4_count']:>6} ({g_ok:>4}) "
            f"{m['local_g0_class4_count']:>5} ({l0_ok:>4}) "
            f"{m['local_g1_class4_count']:>5} ({l1_ok:>4}) "
            f"{all_ok:>8}"
        )

    print("\n" + "-" * 80)
    print("  PER-CLASS F1 SCORES (Seeds Mean +/- Std)")
    print("-" * 80)

    per_class_f1s = []
    for seed_num in range(1, 6):
        seed_path = SEED_DIR / f"seed_{seed_num}" / "final_predictions.csv"
        y_true, y_pred, _, _ = load_predictions(seed_path)
        _, _, f1_pc, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        per_class_f1s.append(f1_pc)

    per_class_f1s = np.array(per_class_f1s)

    _, _, h_f1_pc, h_support = precision_recall_fscore_support(h_true, h_pred, average=None, zero_division=0)
    _, _, e_f1_pc, _ = precision_recall_fscore_support(y_true_ref, ens_pred, average=None, zero_division=0)

    class_names = [f"Class {c}" for c in range(NUM_CLASSES)]
    print(f"\n{'Class':<12} {'Heuristic':>10} {'Seeds Mean':>12} {'Seeds Std':>10} {'Ensemble':>10} {'Support':>8}")
    print("-" * 65)
    for c in range(NUM_CLASSES):
        print(
            f"{class_names[c]:<12} "
            f"{h_f1_pc[c]:>10.4f} "
            f"{per_class_f1s[:, c].mean():>12.4f} "
            f"{per_class_f1s[:, c].std(ddof=1):>10.4f} "
            f"{e_f1_pc[c]:>10.4f} "
            f"{h_support[c]:>8}"
        )

    print("\n" + "-" * 80)
    print("  CONFUSION MATRIX (Ensemble)")
    print("-" * 80)
    cm = confusion_matrix(y_true_ref, ens_pred)
    print(f"\n{'':>10}", end="")
    for c in range(NUM_CLASSES):
        print(f"{'Pred_'+str(c):>8}", end="")
    print()
    for c in range(NUM_CLASSES):
        print(f"{'True_'+str(c):>10}", end="")
        for j in range(NUM_CLASSES):
            print(f"{cm[c][j]:>8}", end="")
        print()

    output = {
        "experiment": "statistical_significance",
        "constraint": [0.5, 0.3],
        "model": "ResNet18",
        "num_seeds": 5,
        "results": {m["name"]: {k: v for k, v in m.items() if k != "name"} for m in all_metrics},
        "seed_stats": {
            k: {
                "mean": float(np.mean(values[k])),
                "std": float(np.std(values[k], ddof=1)),
                "min": float(np.min(values[k])),
                "max": float(np.max(values[k])),
            }
            for k in key_metrics
        },
        "t_tests_vs_heuristic": {
            "accuracy": {"t": float(t_acc), "p": float(p_acc)},
            "f1_macro": {"t": float(t_f1), "p": float(p_f1)},
        },
    }

    out_path = SEED_DIR.parent.parent / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
