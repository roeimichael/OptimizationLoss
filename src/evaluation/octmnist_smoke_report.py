"""OCTMNIST smoke report: did the hard dataset restore TraLO's F1 edge?

Reads results/pending_runs/octmnist_smoke/**. Pools matched per-seed/per-
tightness diffs (TraLO - baseline) for F1-macro (higher better) and Flips
(lower better). The headline question: is TraLO's dF1 clearly POSITIVE here
(unlike the derm/easy tie)? Also surfaces warmup train acc from the logs as
the regime check (want <=~80%).

Usage (server): python -m src.evaluation.octmnist_smoke_report
"""
import csv, glob, json, os
from statistics import mean

ROOT = "results/pending_runs/octmnist_smoke"
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
    d = {}
    for f in glob.glob(os.path.join(ROOT, "**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        hp = c.get("hyperparams", {})
        key = (c.get("constraint_tag"), hp.get("seed"), c.get("methodology"))
        m = lm(ev)
        d[key] = (fn(m.get("F1 (Macro)")), fn(m.get("Flips Required")),
                  fn(m.get("Raw All Satisfied")), fn(m.get("Accuracy")))
    return d


def main():
    d = load()
    print("=" * 68)
    print("OCTMNIST SMOKE (cls=2 DRUSEN, synth_group, L20+L50, 2 seeds)")
    print(f"cells found: {len(d)} / 12 expected")
    print("=" * 68)
    tr = {k[:2]: v for k, v in d.items() if k[2] == "tralo"}
    for mth in ["tralo"] + BASELINES:
        f1s = [v[0] for k, v in d.items() if k[2] == mth and v[0] is not None]
        fls = [v[1] for k, v in d.items() if k[2] == mth and v[1] is not None]
        sat = [v[2] for k, v in d.items() if k[2] == mth and v[2] is not None]
        acc = [v[3] for k, v in d.items() if k[2] == mth and v[3] is not None]
        if f1s:
            print(f"   {mth:<14} F1={mean(f1s):.4f}  acc={mean(acc):.4f}  "
                  f"flips={mean(fls):.1f}  sat={mean(sat):.2f}  (n={len(f1s)})")
    for b in BASELINES:
        df1, dfl = [], []
        for (tight, seed), tv in tr.items():
            bv = d.get((tight, seed, b))
            if not bv:
                continue
            if tv[0] is not None and bv[0] is not None:
                df1.append(tv[0] - bv[0])
            if tv[1] is not None and bv[1] is not None:
                dfl.append(bv[1] - tv[1])
        if df1:
            pos = sum(1 for x in df1 if x > 1e-9)
            tag = "TraLO+" if mean(df1) > 0 else "TraLO-"
            print(f"     vs {b:<13} dF1={mean(df1):+.4f} ({pos}/{len(df1)} seeds) "
                  f"dFlips={mean(dfl):+.1f}  -> {tag}")
    print("=" * 68)
    print("Want: dF1 clearly POSITIVE (unlike derm tie) AND warmup train acc "
          "<=~80% (grep the log: 'Warmup 50/50').")


if __name__ == "__main__":
    main()
