"""Backbone smoketest report — per backbone, does TraLO's F1 edge over the
trained baselines (Fioretto, Hounie) widen vs the MobileNetV3 tie?

Reads results/pending_runs/backbone_smoke/**. For each backbone, pools the
matched per-seed/per-tightness differences (TraLO - baseline) for F1-macro
(higher better) and Flips (lower better) and prints mean diff + sign count.
Only 2 seeds x 2 tightness => 4 pairs/baseline, so this is a SIGNAL, not a
significance test; magnitude and sign are what matter.

Usage (on server): python -m src.evaluation.backbone_smoke_report
"""
import csv, glob, json, os
from collections import defaultdict
from statistics import mean

ROOT = "results/pending_runs/backbone_smoke"
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
    # (backbone, tight, seed, method) -> (f1, flips, sat)
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
        key = (c.get("model_name"), c.get("constraint_tag"), hp.get("seed"),
               c.get("methodology"))
        m = lm(ev)
        d[key] = (fn(m.get("F1 (Macro)")), fn(m.get("Flips Required")),
                  fn(m.get("Raw All Satisfied")))
    return d


def main():
    d = load()
    backbones = sorted({k[0] for k in d})
    print("=" * 72)
    print("BACKBONE SMOKETEST (dermmnist cls=4, L20+L50, 2 seeds)")
    print(f"cells found: {len(d)} / 36 expected")
    print("=" * 72)
    for bb in backbones:
        tr = {k[1:3]: v for k, v in d.items() if k[0] == bb and k[3] == "tralo"}
        print(f"\n## {bb}")
        # absolute means
        for mth in ["tralo"] + BASELINES:
            f1s = [v[0] for k, v in d.items()
                   if k[0] == bb and k[3] == mth and v[0] is not None]
            fls = [v[1] for k, v in d.items()
                   if k[0] == bb and k[3] == mth and v[1] is not None]
            sat = [v[2] for k, v in d.items()
                   if k[0] == bb and k[3] == mth and v[2] is not None]
            if f1s:
                print(f"   {mth:<14} F1={mean(f1s):.4f}  flips={mean(fls):.1f}  "
                      f"sat={mean(sat):.2f}  (n={len(f1s)})")
        # paired diffs
        for b in BASELINES:
            df1, dfl = [], []
            for (tight, seed), tv in tr.items():
                bv = d.get((bb, tight, seed, b))
                if not bv:
                    continue
                if tv[0] is not None and bv[0] is not None:
                    df1.append(tv[0] - bv[0])
                if tv[1] is not None and bv[1] is not None:
                    dfl.append(bv[1] - tv[1])  # positive = TraLO fewer flips
            if df1:
                pos = sum(1 for x in df1 if x > 1e-9)
                tag = "TraLO+" if mean(df1) > 0 else "TraLO-"
                print(f"     vs {b:<13} dF1={mean(df1):+.4f} ({pos}/{len(df1)} seeds) "
                      f"dFlips={mean(dfl):+.1f}  -> {tag}")
    print("\n" + "=" * 72)
    print("Compare dF1 here to MobileNetV3 (TIE on derm). A clearly larger +dF1 "
          "means this backbone surfaces TraLO's advantage better.")


if __name__ == "__main__":
    main()
