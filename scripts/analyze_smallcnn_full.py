"""Headroom-test analysis: TraLO vs baselines on derm_smallcnn_full.

Reads training_log + evaluation_metrics for each cell, applies the
end-to-end non-saturation rule (max train_acc < 0.995 across ALL epochs),
and computes paired d_F1 per backbone.
"""
import csv
import glob
import os
from collections import defaultdict

ROOT = "results/pending_runs/derm_smallcnn_full"
SAT_THRESHOLD = 0.995


def read_metrics(path):
    m = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            m[r["Metric"]] = r["Value"]
    return m


def read_max_train_acc(log_path):
    if not os.path.exists(log_path):
        return None
    max_acc = 0.0
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ("Train_Acc", "train_acc", "Train Accuracy", "train_accuracy"):
                if k in r and r[k]:
                    try:
                        max_acc = max(max_acc, float(r[k]))
                    except ValueError:
                        pass
                    break
    return max_acc


def main():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/*/*/seed_1")):
        parts = d.split("/")
        model, method = parts[-3], parts[-2]
        mp = os.path.join(d, "evaluation_metrics.csv")
        lp = os.path.join(d, "training_log.csv")
        if not os.path.exists(mp):
            print(f"MISSING evaluation_metrics: {model}/{method}")
            continue
        m = read_metrics(mp)
        max_tr = read_max_train_acc(lp)
        rows.append({
            "model": model, "method": method,
            "f1": float(m.get("F1 (Macro)", "nan")),
            "acc": float(m.get("Accuracy", "nan")),
            "flips": float(m.get("Flips Required", 0)),
            "sat": m.get("Raw All Satisfied", "0") == "1",
            "max_tr": max_tr,
        })

    print(f"{'model':12s} {'method':14s} {'F1':>7s} {'Acc':>7s} {'Flips':>6s} {'Sat':>4s} {'max_tr':>8s} contam?")
    print("-" * 78)
    for r in rows:
        contam = r["max_tr"] is not None and r["max_tr"] >= SAT_THRESHOLD
        flag = "YES" if contam else ""
        mt = f"{r['max_tr']:.4f}" if r["max_tr"] is not None else "    n/a "
        print(f"{r['model']:12s} {r['method']:14s} {r['f1']:7.4f} {r['acc']:7.4f} "
              f"{r['flips']:6.0f} {'Y' if r['sat'] else 'N':>4s} {mt:>8s} {flag}")

    print()
    print("=== paired d_F1 vs each baseline (TraLO - baseline) — clean cells only ===")
    print(f"{'model':12s} {'TR_F1':>7s} {'vs_fio':>8s} {'vs_hou':>8s} {'vs_dan':>8s} {'vs_heu':>8s} note")
    print("-" * 78)
    for model in sorted(set(r["model"] for r in rows)):
        sub = {r["method"]: r for r in rows if r["model"] == model}
        tr = sub.get("tralo")
        if not tr:
            continue
        # contamination check across ALL methods in this cell
        contam_methods = [m for m, r in sub.items()
                          if r["max_tr"] is not None and r["max_tr"] >= SAT_THRESHOLD]
        note = ""
        if contam_methods:
            note = f"contam:{','.join(contam_methods)}"
        deltas = {}
        for bl in ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]:
            if bl in sub:
                deltas[bl] = tr["f1"] - sub[bl]["f1"]
        print(f"{model:12s} {tr['f1']:7.4f} "
              f"{deltas.get('fioretto_ldf', float('nan')):+8.4f} "
              f"{deltas.get('hounie_rcl', float('nan')):+8.4f} "
              f"{deltas.get('danits_lp', float('nan')):+8.4f} "
              f"{deltas.get('heuristic', float('nan')):+8.4f} {note}")

    print()
    print("=== headroom hypothesis verdict ===")
    print("Saturated MobileNetV3 paper baseline (derm tight cells): d_F1 ~ +0.005")
    print("If d_F1 GROWS on clean small-CNN cells -> hypothesis supported.")


if __name__ == "__main__":
    main()
