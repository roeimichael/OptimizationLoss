"""Summarize AIDER class-rotation results across constrained-class choices.

Reads results/pending_runs/class_rotation/aider/MobileNetV3/<cfg>/<method>/seed_*/
and reports mean F1 + d_F1 (TraLO - baseline) per constrained class.

If the mechanism hypothesis is right, TraLO d_F1 should be smaller when constraining
the MAJORITY class (easier task — warmup already nails it) and larger for minority
classes (harder task — warmup uncertain, post-hoc forced into bad flips).
"""
import csv
import glob
import os
from collections import defaultdict

ROOT = "results/pending_runs/class_rotation/aider/MobileNetV3"


def main():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/*/*/seed_*")):
        parts = d.split("/")
        cfg, method, seed = parts[-3], parts[-2], parts[-1]
        mp = os.path.join(d, "evaluation_metrics.csv")
        if not os.path.exists(mp):
            continue
        m = {}
        with open(mp) as f:
            for r in csv.DictReader(f):
                m[r["Metric"]] = r["Value"]
        rows.append({
            "cfg": cfg, "method": method, "seed": seed,
            "f1": float(m.get("F1 (Macro)", 0)),
            "flips": float(m.get("Flips Required", 0)),
            "sat": int(m.get("Raw All Satisfied", "0") == "1"),
        })

    agg_f1 = defaultdict(list)
    agg_flips = defaultdict(list)
    for r in rows:
        agg_f1[(r["cfg"], r["method"])].append(r["f1"])
        agg_flips[(r["cfg"], r["method"])].append(r["flips"])

    print(f"{'cfg':50s} {'method':14s} {'F1':>8s} {'flips':>8s} {'n':>3s}")
    print("-" * 90)
    for (cfg, m), vals in sorted(agg_f1.items()):
        fl = agg_flips[(cfg, m)]
        print(f"{cfg:50s} {m:14s} {sum(vals)/len(vals):8.4f} "
              f"{sum(fl)/len(fl):8.1f} {len(vals):3d}")

    print()
    print("=== d_F1 (TraLO - baseline) per constrained-class config ===")
    print(f"{'cfg':50s} {'vs_fio':>8s} {'vs_hou':>8s}")
    print("-" * 70)
    for cfg in sorted(set(c for c, _ in agg_f1)):
        tr = agg_f1.get((cfg, "tralo"))
        if not tr:
            continue
        tr_mean = sum(tr) / len(tr)
        deltas = {}
        for bl in ["fioretto_ldf", "hounie_rcl"]:
            bl_vals = agg_f1.get((cfg, bl))
            if bl_vals:
                deltas[bl] = tr_mean - sum(bl_vals) / len(bl_vals)
        print(f"{cfg:50s} "
              f"{deltas.get('fioretto_ldf', float('nan')):+8.4f} "
              f"{deltas.get('hounie_rcl', float('nan')):+8.4f}")


if __name__ == "__main__":
    main()
