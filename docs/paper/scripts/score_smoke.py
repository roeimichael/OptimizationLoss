"""Score an arm smoke: paired, per cell, and inert-flag check first.

The FIRST question is never the score. It is whether the flag did anything at
all. This project has shipped two ablations whose flags silently never took
effect, and the first version of the checkpoint fix produced md5-identical
output across 48 runs. So compare the raw prediction files byte for byte before
comparing any metric: if the arms agree exactly, the flag is inert and every
number below is meaningless.

Cells are never pooled. One row per (dataset, backbone, cap), seeds averaged.

    python paper/scripts/score_smoke.py --root newdirections/arm_geom/results/geom --base geom_off
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402

CELL = ["dataset", "model", "cap"]
METRICS = ["ccF1eq", "AP", "macroEq", "count_raw"]


def md5(p):
    if not os.path.exists(p):
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def arm_of(cfg_path):
    try:
        return json.load(open(cfg_path)).get("arm")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--base", required=True, help="the control arm name")
    args = ap.parse_args()

    # ---- 1. inert-flag check, on raw predictions ----
    by_key = {}
    for cfg in glob.glob(args.root + "/**/config.json", recursive=True):
        a = arm_of(cfg)
        if not a:
            continue
        d = os.path.dirname(cfg)
        c = json.load(open(cfg))
        key = (c["dataset_mode"], c["model_name"], c["constraint_tag"],
               c["hyperparams"]["seed"])
        by_key.setdefault(key, {})[a] = md5(os.path.join(d, "final_predictions_raw.csv"))

    arms = sorted({a for v in by_key.values() for a in v})
    others = [a for a in arms if a != args.base]
    print("=" * 78)
    print("INERT-FLAG CHECK  (raw predictions, md5)")
    print("=" * 78)
    for o in others:
        same = tot = 0
        for k, v in by_key.items():
            if v.get(args.base) and v.get(o):
                tot += 1
                same += int(v[args.base] == v[o])
        verdict = ("INERT -- flag did nothing" if tot and same == tot
                   else "live (%d of %d cells identical)" % (same, tot))
        print("  %-14s vs %-14s  %s" % (o, args.base, verdict))
    print()

    # ---- 2. scores, per cell ----
    d = A.rows_for(args.root)
    if d.empty:
        print("no scorable runs")
        return 1
    # rows_for now emits `arm` directly. It has to: both arms share one
    # methodology and one (dataset, model, cap, seed) key, so any map built
    # from that key silently keeps whichever arm was globbed last.
    if "arm" not in d.columns:
        print("!! analyze_headroom.rows_for does not emit `arm` -- patch it first")
        return 1
    d = d.dropna(subset=["arm"])

    print("=" * 78)
    print("PER CELL (seeds averaged, cells never pooled)")
    print("=" * 78)
    t = d.groupby(CELL + ["arm"])[METRICS].mean().round(4)
    print(t.to_string())

    print()
    print("=" * 78)
    print("PAIRED DELTA vs %s, per cell" % args.base)
    print("=" * 78)
    for o in others:
        print("\n--- %s ---" % o)
        rows = []
        for cell, g in d.groupby(CELL):
            piv = g.pivot_table(index="seed", columns="arm", values=METRICS)
            r = {"dataset": cell[0], "model": cell[1], "cap": cell[2]}
            for m in METRICS:
                if (m, args.base) in piv.columns and (m, o) in piv.columns:
                    delta = piv[(m, o)] - piv[(m, args.base)]
                    r[m] = round(float(delta.mean()), 4)
                    r[m + "_w"] = "%d/%d" % (int((delta > 0).sum()), len(delta))
            rows.append(r)
        tt = pd.DataFrame(rows)
        print(tt.to_string(index=False))
        for m in METRICS:
            if m in tt.columns:
                w = int((tt[m] > 0.005).sum())
                l = int((tt[m] < -0.005).sum())
                print("   %-10s WIN %d cells  LOSS %d cells  TIE %d  (of %d)"
                      % (m, w, l, len(tt) - w - l, len(tt)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
