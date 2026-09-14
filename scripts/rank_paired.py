"""Per-cell paired gAP with dispersion, so a mean is never read without its floor.

rank_probe reports paired MEANS only. The house rule is that an arm-vs-arm gap
must be read against the RNG floor, so this reports, per cell (backbone x cap x
class): the seed-paired delta, its sd ACROSS seeds, and how many seeds are
paired. Cells are never pooled into one average -- they are counted.

And the cap is NOT always a cell. An arm that takes no constraint step trains
one model and lets the cap act only in the post-hoc allocator, so its raw
probabilities are byte-identical across L70/L80/L90 -- verified by md5 on bcn
for `clip` and `tralo_null`, while `tralo` differs in all three. gAP is
allocation-free, so counting three caps as three cells inflates n threefold.
When BOTH arms in a contrast are cap-invariant the cap dimension is collapsed to
a single row rather than printed three times.
"""
import glob, hashlib, json, os, sys, collections
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
                    out[key].setdefault(cfg["arm"], {})[seed] = (
                        float(np.average(gaps, weights=wts)),
                        hashlib.md5(p.tobytes()).hexdigest())
    return out


def cap_invariant(cells, model, cls, arm):
    """True when this arm's probabilities do not change with the cap.

    The cap then never reached the model -- it acted only in the allocator, and
    gAP is allocation-free -- so its cap rows are the SAME observation repeated.
    """
    per_seed = collections.defaultdict(set)
    caps = 0
    for (m, cap, c), arms in cells.items():
        if m != model or c != cls or arm not in arms:
            continue
        caps += 1
        for seed, (_, h) in arms[arm].items():
            per_seed[seed].add(h)
    return caps > 1 and all(len(v) == 1 for v in per_seed.values())


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
        seen = set()
        for key in sorted(cells):
            arms = cells[key]
            if args.a not in arms or b not in arms:
                continue
            collapse = (cap_invariant(cells, key[0], key[2], args.a)
                        and cap_invariant(cells, key[0], key[2], b))
            label = "(cap-inert)" if collapse else key[1]
            va, vb = arms[args.a], arms[b]
            if collapse:
                if (key[0], key[2]) in seen:
                    continue
                seen.add((key[0], key[2]))
                # The caps hold the SAME probabilities, so a seed run under one
                # cap and not another is still one observation. Union them --
                # taking the first cap alone silently halved n on bcn/ViTB16.
                va, vb = {}, {}
                for (m2, _, c2), a2 in cells.items():
                    if (m2, c2) != (key[0], key[2]):
                        continue
                    va.update(a2.get(args.a, {}))
                    vb.update(a2.get(b, {}))
            s = sorted(set(va) & set(vb))
            if len(s) < 2:
                continue
            d = np.array([va[x][0] - vb[x][0] for x in s])
            m, sd = float(d.mean()), float(d.std(ddof=1))
            rows.append(m)
            print("  %-12s %-10s %4d %5d %+10.4f %10.4f %8s" %
                  (key[0], label, key[2], len(s), m, sd,
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
