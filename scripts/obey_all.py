"""Who obeys the caps NATIVELY, before the allocator -- every arm, one scale.

The per-arm training logs are not comparable: only the TraLO family writes
Group*_Hard_* columns, and clip/focal_clip write none at all. So the counts are
recomputed here from each run's own raw predictions, identically for every arm:
argmax of the pre-allocation probabilities, against the same local budget
K = floor(true count in that group * L).

LOCAL cells only -- verify_caps shows the global cap is inert at G95.
"""
import glob, json, os, sys, collections, statistics as st
import numpy as np, pandas as pd


def run(d):
    fp, cj = os.path.join(d, "final_predictions_raw.csv"), os.path.join(d, "config.json")
    if not (os.path.exists(fp) and os.path.exists(cj)):
        return None
    cfg = json.load(open(cj))
    L = float(cfg["constraint"][0])
    cc = cfg["dataset_config"]["constrained_class"]
    df = pd.read_csv(fp)
    y, g = df.True_Label.to_numpy(), df.Group_ID.to_numpy()
    pred = df[[c for c in df.columns if c.startswith("Prob_Class_")]].to_numpy().argmax(1)
    ok = tot = 0
    excess = 0
    for gg in np.unique(g):
        m = g == gg
        for c in cc:
            k = int((y[m] == c).sum() * L)
            n = int((pred[m] == c).sum())
            tot += 1
            if n <= k:
                ok += 1
            else:
                excess += n - k
    return cfg["arm"], cfg["constraint_tag"], cfg["hyperparams"]["seed"], ok, tot, excess


def main(globs):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for gl in globs:
        for d in glob.glob(gl):
            r = run(d)
            if r:
                per[r[1]][r[0]].append(r[3:])
    for cap in sorted(per):
        print("")
        print("=== %s -- NATIVE compliance from raw argmax (global cap inert at G95) ===" % cap)
        print("  %-12s %4s %14s %14s" % ("arm", "n", "compliant", "excess items"))
        for arm in sorted(per[cap], key=lambda a: st.mean([x[2] for x in per[cap][a]])):
            v = per[cap][arm]
            print("  %-12s %4d %14s %14s" % (
                arm, len(v), "%.1f/%d" % (st.mean([x[0] for x in v]), v[0][1]),
                "%.0f" % st.mean([x[2] for x in v])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
