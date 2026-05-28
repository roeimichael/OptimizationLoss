"""Warmup-headroom report: TraLO dF1 vs Fioretto/Hounie as a function of
warmup_epochs, per (dataset, backbone). The headline curve — does TraLO's F1
edge grow as warmup shrinks (less saturation = more headroom)?

Reads results/pending_runs/warmup_headroom/**. Pairs by (ds,bb,warmup,seed).
Prints, per (ds,bb), a row per warmup level: TraLO F1/acc, dF1 vs each baseline,
flips. Read top-to-bottom (warmup 0 -> 50) to see the trend.

Usage (server): python -m src.evaluation.warmup_headroom_report
"""
import csv, glob, json, os
from collections import defaultdict
from statistics import mean

# Globs the full sweep, the fast probe, and the cross-backbone confirmation.
ROOTS = ["results/pending_runs/warmup_headroom", "results/pending_runs/warmup_probe",
         "results/pending_runs/warmup_confirm"]
BASELINES = ["fioretto_ldf", "hounie_rcl"]


def lm(p):
    o = {}
    try:
        for r in csv.reader(open(p)):
            if len(r) == 2:
                o[r[0]] = r[1]
    except Exception:
        pass
    return o


def fn(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def load():
    # (ds, bb, warmup, seed, method) -> (f1, flips, sat, acc)
    d = {}
    files = []
    for root in ROOTS:
        files += glob.glob(os.path.join(root, "**/config.json"), recursive=True)
    for f in files:
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        hp = c.get("hyperparams", {})
        key = (c.get("dataset_mode"), c.get("model_name"),
               hp.get("warmup_epochs"), hp.get("seed"), c.get("methodology"))
        m = lm(ev)
        d[key] = (fn(m.get("F1 (Macro)")), fn(m.get("Flips Required")),
                  fn(m.get("Raw All Satisfied")), fn(m.get("Accuracy")))
    return d


def main():
    d = load()
    dss = sorted({k[0] for k in d})
    bbs = sorted({k[1] for k in d})
    warmups = sorted({k[2] for k in d if k[2] is not None})
    print("=" * 78)
    print(f"WARMUP-HEADROOM ABLATION  (cells found: {len(d)}; warmups {warmups})")
    print("Trend to look for: TraLO dF1 grows as warmup_epochs shrinks.")
    print("=" * 78)
    for ds in dss:
        for bb in bbs:
            rows = []
            for w in warmups:
                tr = {k[3]: v for k, v in d.items()
                      if k[0] == ds and k[1] == bb and k[2] == w and k[4] == "tralo"}
                if not tr:
                    continue
                trf1 = [v[0] for v in tr.values() if v[0] is not None]
                tracc = [v[3] for v in tr.values() if v[3] is not None]
                trfl = [v[1] for v in tr.values() if v[1] is not None]
                line = (f"  w={w:<3} n={len(tr)}  TraLO F1={mean(trf1):.4f} "
                        f"acc={mean(tracc):.3f} flips={mean(trfl):.1f}"
                        if trf1 else f"  w={w:<3} (no tralo cells)")
                for b in BASELINES:
                    df1 = []
                    for seed, tv in tr.items():
                        bv = d.get((ds, bb, w, seed, b))
                        if bv and tv[0] is not None and bv[0] is not None:
                            df1.append(tv[0] - bv[0])
                    if df1:
                        pos = sum(1 for x in df1 if x > 1e-9)
                        line += (f"  | d{b[:4]}={mean(df1):+.4f}({pos}/{len(df1)})")
                rows.append(line)
            if rows:
                print(f"\n## {ds} / {bb}")
                for r in rows:
                    print(r)
    print("\n" + "=" * 78)
    print("dFior/dhoun = TraLO F1 minus that baseline (positive = TraLO wins). "
          "Want positive growing as w -> 0.")


if __name__ == "__main__":
    main()
