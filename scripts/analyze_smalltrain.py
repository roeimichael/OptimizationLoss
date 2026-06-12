"""Analyze the 'smaller train = TraLO wins more' hypothesis.

For each CIFAR-100 train regime (full / subset50 / subset10 / subset5),
extract per-cell warmup train_acc + final macro_f1, then compute paired
(TraLO - Hounie) and (TraLO - Fioretto) deltas.

Hypothesis: as train set shrinks, TraLO's advantage over Hounie grows.
Test: plot/report mean d_macro vs train-size; check correlation.
"""
import csv
import glob
import json
import os
from collections import defaultdict
import numpy as np

SWEEPS = {
    "full_50k":   ("results/pending_runs/new_dataset_probes/cifar100", 50000),
    "subset50":   ("results/pending_runs/cifar100_smalltrain/subset50", 5000),
    "subset10":   ("results/pending_runs/cifar100_smalltrain/subset10", 1000),
    "subset5":    ("results/pending_runs/cifar100_smalltrain/subset5", 500),
}


def extract_max_train_acc(cell_dir):
    log = os.path.join(cell_dir, "training_log.csv")
    if not os.path.exists(log): return None
    try:
        with open(log) as f:
            rows = list(csv.DictReader(f))
        if not rows or "train_acc" not in rows[0]: return None
        accs = []
        for r in rows:
            v = r.get("train_acc", "")
            try: accs.append(float(v))
            except (ValueError, TypeError): pass
        return max(accs) if accs else None
    except Exception:
        return None


def collect(root):
    out = []
    for p in glob.glob(f"{root}/**/config.json", recursive=True):
        cell_dir = os.path.dirname(p)
        metrics_p = p.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(metrics_p): continue
        try:
            with open(p) as f: cfg = json.load(f)
            m = {}
            with open(metrics_p) as f:
                for r in csv.DictReader(f): m[r["Metric"]] = r["Value"]
            out.append({
                "method": cfg["methodology"],
                "tight": cfg["constraint_tag"],
                "seed": cfg["hyperparams"]["seed"],
                "max_train_acc": extract_max_train_acc(cell_dir),
                "macro_f1": float(m["F1 (Macro)"]),
                "raw_satisfied": int(m.get("Raw All Satisfied", "0") == "1"),
                "flips": float(m.get("Flips Required", "nan")),
            })
        except Exception:
            continue
    return out


def paired_delta(rows, baseline):
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["tight"], r["seed"])][r["method"]] = r
    deltas = []
    for k, by_m in by_key.items():
        if "tralo" not in by_m or baseline not in by_m: continue
        deltas.append({
            "tight": k[0], "seed": k[1],
            "tralo_macro": by_m["tralo"]["macro_f1"],
            "bl_macro": by_m[baseline]["macro_f1"],
            "d_macro": by_m["tralo"]["macro_f1"] - by_m[baseline]["macro_f1"],
            "tralo_train_acc": by_m["tralo"]["max_train_acc"],
            "bl_train_acc": by_m[baseline]["max_train_acc"],
            "tralo_flips": by_m["tralo"]["flips"],
            "bl_flips": by_m[baseline]["flips"],
            "tralo_sat": by_m["tralo"]["raw_satisfied"],
            "bl_sat": by_m[baseline]["raw_satisfied"],
        })
    return deltas


def main():
    print(f"\n{'='*88}")
    print(f"HYPOTHESIS: smaller train set -> TraLO advantage grows (test-side fuzziness lever)")
    print(f"{'='*88}\n")

    print(f"{'sweep':<14}{'n_train':>8}{'cells':>6}{'TR_acc_mean':>13}{'TR_acc_max':>11}"
          f"{'macro_tralo':>13}{'macro_hounie':>14}{'d_macro_v_H':>13}{'W/L':>7}")
    print("-" * 105)

    summary_rows = []
    for sweep_label, (root, n_train) in SWEEPS.items():
        rows = collect(root)
        if not rows:
            print(f"  {sweep_label}: NO DATA"); continue
        tralo = [r for r in rows if r["method"] == "tralo"]
        hounie = [r for r in rows if r["method"] == "hounie_rcl"]
        fior = [r for r in rows if r["method"] == "fioretto_ldf"]
        # train acc stats over TraLO cells (representative)
        tr_acc = [r["max_train_acc"] for r in tralo if r["max_train_acc"] is not None]
        if not tr_acc:
            tr_mean = tr_max = "n/a"
        else:
            tr_mean = f"{np.mean(tr_acc):.4f}"
            tr_max  = f"{np.max(tr_acc):.4f}"
        # paired vs Hounie
        deltas_h = paired_delta(rows, "hounie_rcl")
        if deltas_h:
            d = np.array([x["d_macro"] for x in deltas_h])
            w = int((d > 1e-4).sum()); l = int((d < -1e-4).sum())
            wl = f"{w}/{l}"
            d_mean = f"{np.mean(d):+.4f}"
        else:
            wl = "0/0"; d_mean = "n/a"
        macro_tr = f"{np.mean([r['macro_f1'] for r in tralo]):.4f}" if tralo else "n/a"
        macro_h  = f"{np.mean([r['macro_f1'] for r in hounie]):.4f}" if hounie else "n/a"
        print(f"{sweep_label:<14}{n_train:>8}{len(rows):>6}"
              f"{tr_mean:>13}{tr_max:>11}{macro_tr:>13}{macro_h:>14}{d_mean:>13}{wl:>7}")
        summary_rows.append({
            "sweep": sweep_label, "n_train": n_train,
            "n_cells": len(rows),
            "tr_acc_mean": tr_mean, "tr_acc_max": tr_max,
            "macro_tralo": macro_tr, "macro_hounie": macro_h,
            "d_macro_vs_H": d_mean, "W/L_vs_H": wl,
            "n_paired_h": len(deltas_h),
            "n_paired_f": len(paired_delta(rows, "fioretto_ldf")),
        })

    print(f"\n--- Detail: per-cell train_acc + d_macro (vs Hounie) ---\n")
    print(f"{'sweep':<14}{'tight':<10}{'seed':>4}{'tralo_tr_acc':>14}"
          f"{'tralo_macro':>13}{'hounie_macro':>14}{'d_macro':>10}")
    print("-" * 90)
    for sweep_label, (root, _) in SWEEPS.items():
        rows = collect(root)
        for d in sorted(paired_delta(rows, "hounie_rcl"), key=lambda x: (x["tight"], x["seed"])):
            tracc = f"{d['tralo_train_acc']:.4f}" if d['tralo_train_acc'] else "n/a"
            print(f"{sweep_label:<14}{d['tight']:<10}{d['seed']:>4}{tracc:>14}"
                  f"{d['tralo_macro']:>13.4f}{d['bl_macro']:>14.4f}{d['d_macro']:>+10.4f}")


if __name__ == "__main__":
    main()
