"""Model-search smoke report: per (model, dataset), TraLO macro-F1 minus each
baseline (Fioretto, Hounie). Positive = TraLO wins. Reads
results/pending_runs/model_search/{MODEL}/smoke/{ds}/{method}/seed_1/.

Usage (server): python -m src.evaluation.model_search_report [MODEL]
(no arg = all models found).
"""
import csv, glob, json, os, sys
from collections import defaultdict

ROOT = "results/pending_runs/model_search"
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


def main():
    want = sys.argv[1] if len(sys.argv) > 1 else None
    # (model, ds, method) -> (f1, flips, sat, acc)
    d = {}
    for f in glob.glob(os.path.join(ROOT, "**/smoke/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        key = (c.get("model_name"), c.get("dataset_mode"), c.get("methodology"))
        m = lm(ev)
        d[key] = (fn(m.get("F1 (Macro)")), fn(m.get("Flips Required")),
                  fn(m.get("Raw All Satisfied")), fn(m.get("Accuracy")))
    models = sorted({k[0] for k in d}) if not want else [want]
    print("=" * 74)
    print("MODEL-SEARCH SMOKE  (TraLO macro-F1 minus baseline; + = TraLO wins)")
    print("=" * 74)
    for model in models:
        dss = sorted({k[1] for k in d if k[0] == model})
        if not dss:
            print(f"\n## {model}: (no smoke cells yet)")
            continue
        print(f"\n## {model}")
        for ds in dss:
            tv = d.get((model, ds, "tralo"))
            if not tv or tv[0] is None:
                print(f"  {ds:<12} (no tralo cell)")
                continue
            line = f"  {ds:<12} TraLO F1={tv[0]:.4f} acc={tv[3]:.3f} flips={tv[1]:.0f}"
            for b in BASELINES:
                bv = d.get((model, ds, b))
                if bv and bv[0] is not None:
                    line += f"  d{b[:4]}={tv[0]-bv[0]:+.4f}"
            print(line)
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
