"""Per-cell paired gAP with dispersion, so a mean is never read without its floor.

rank_probe reports paired MEANS only. The house rule is that an arm-vs-arm gap
must be read against the RNG floor, so this reports, per cell (backbone x cap x
class): the seed-paired delta, its sd ACROSS seeds, and how many seeds are
paired. Cells are never pooled into one average -- they are counted.
"""
import glob, json, os, sys, collections
import numpy as np, pandas as pd


def ap(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    o = np.argsort(-scores); l = labels[o]
    tp = np.cumsum(l)
    return float(((tp / np.arange(1, len(l) + 1)) * l).sum() / l.sum())


def collect(globs):
    out = collections.defaultdict(dict)
    for g in globs:
        for d in glob.glob(g):
            fp, cj = os.path.join(d, "final_predictions_raw.csv"), os.path.join(d, "config.json")
            if not (os.path.exists(fp) and os.path.exists(cj)):
                continue
            cfg = json.load(open(cj)); df = pd.read_csv(fp)
            y, gid = df.True_Label.to_numpy(), df.Group_ID.to_numpy()
            seed = cfg["hyperparams"]["seed"]
            for c in cfg["dataset_config"]["constrained_class"]:
                p = df["Prob_Class_%d" % c].to_numpy(); lab = (y == c).astype(float)
                gaps, wts = [], []
                for gg in np.unique(gid):
                    m = gid == gg; a = ap(p[m], lab[m])
                    if not np.isnan(a):
                        gaps.append(a); wts.append(lab[m].sum())
                if gaps:
                    key = (cfg["model_name"], cfg["constraint_tag"], c)
                    out[key].setdefault(cfg["arm"], {})[seed] = float(np.average(gaps, weights=wts))
    return out


def main(argv=None):
    import argparse
    P = argparse.ArgumentParser(description=__doc__)
    P.add_argument("--glob", nargs="+", required=True)
    P.add_argument("--a", required=True, help="treatment arm")
    P.add_argument("--b", nargs="+", required=True, help="reference arm(s)")
    args = P.parse_args(argv)
    cells = collect(args.glob)
    for b in args.b:
        print("")
        print("=== %s minus %s ===" % (args.a, b))
        print("  %-12s %-10s %4s %5s %10s %10s %8s" %
              ("backbone", "cap", "cls", "seeds", "mean d", "sd over", "|mean|/sd"))
        rows = []
        for key in sorted(cells):
            arms = cells[key]
            if args.a not in arms or b not in arms:
                continue
            s = sorted(set(arms[args.a]) & set(arms[b]))
            if len(s) < 2:
                continue
            d = np.array([arms[args.a][x] - arms[b][x] for x in s])
            m, sd = float(d.mean()), float(d.std(ddof=1))
            rows.append(m)
            print("  %-12s %-10s %4d %5d %+10.4f %10.4f %8s" %
                  (key[0], key[1], key[2], len(s), m, sd,
                   "%.2f" % (abs(m) / sd) if sd > 0 else "inf"))
        if rows:
            pos = sum(1 for x in rows if x > 0)
            print("  -> %d cells, %d positive, %d negative; cell-mean %+0.4f (NOT a pooled estimate)"
                  % (len(rows), pos, len(rows) - pos, float(np.mean(rows))))
        else:
            print("  no cell has 2+ paired seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
