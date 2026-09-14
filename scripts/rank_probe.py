"""Does the arm RE-RANK better? The only channel by which it can win.

Post-hoc allocation is optimal GIVEN the probabilities and that optimality is
distribution-free, so an arm cannot beat the clipper by allocating better -- it
can only beat it by producing a better ORDER. This scores the order directly,
allocation-free, on the raw pre-allocator probabilities.

PER-GROUP average precision (gAP), because the allocator cuts top-k WITHIN each
group; a global AP measures an ordering the system never uses. Groups are
weighted by their budget so the summary matches what the allocator spends.
Global AP is printed beside it because the two disagree, not as a substitute.

A de-saturation that raises p(1-p) at the cut but leaves gAP unchanged has moved
the numbers without moving the order, and cannot change one deployed item.
"""
import glob, json, os, sys, collections
import numpy as np
import pandas as pd


def ap(scores, labels):
    """Average precision, computed directly so no sklearn version skew."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(-scores)
    l = labels[order]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())


def main(globs):
    out = collections.defaultdict(list)
    for g in globs:
        for d in glob.glob(g):
            fp = os.path.join(d, "final_predictions_raw.csv")
            cj = os.path.join(d, "config.json")
            if not (os.path.exists(fp) and os.path.exists(cj)):
                continue
            cfg = json.load(open(cj))
            df = pd.read_csv(fp)
            y = df.True_Label.to_numpy(); gid = df.Group_ID.to_numpy()
            seed = cfg["hyperparams"]["seed"]
            for c in cfg["dataset_config"]["constrained_class"]:
                p = df["Prob_Class_%d" % c].to_numpy()
                lab = (y == c).astype(float)
                gaps, wts = [], []
                for gg in np.unique(gid):
                    m = gid == gg
                    a = ap(p[m], lab[m])
                    if not np.isnan(a):
                        gaps.append(a); wts.append(lab[m].sum())
                if not gaps:
                    continue
                out[(cfg["model_name"], cfg["constraint_tag"], cfg["arm"], c)].append(
                    dict(seed=seed,
                         gAP=float(np.average(gaps, weights=wts)),
                         capAP=ap(p, lab)))
    if not out:
        print("no runs matched"); return 0
    cells = collections.defaultdict(dict)
    for (mdl, cap, arm, c), v in out.items():
        cells[(mdl, cap, c)][arm] = {r["seed"]: r for r in v}
    for key in sorted(cells):
        arms = cells[key]
        print(""); print("%s / %s / class %d" % key)
        print("  %-12s %5s %10s %10s" % ("arm", "n", "gAP", "capAP"))
        for a in sorted(arms):
            rs = list(arms[a].values())
            print("  %-12s %5d %10.4f %10.4f"
                  % (a, len(rs), np.mean([r["gAP"] for r in rs]),
                     np.mean([r["capAP"] for r in rs])))
        if "tralo" in arms:
            print("  -- TraLO minus arm, on common seeds (gAP is the one that can win) --")
            for a in sorted(set(arms) - {"tralo"}):
                s = sorted(set(arms["tralo"]) & set(arms[a]))
                if not s:
                    print("  %-12s shares no seed" % a); continue
                d = np.mean([arms["tralo"][x]["gAP"] - arms[a][x]["gAP"] for x in s])
                dg = np.mean([arms["tralo"][x]["capAP"] - arms[a][x]["capAP"] for x in s])
                print("  %-12s seeds=%d  d gAP %+0.4f   d capAP %+0.4f" % (a, len(s), d, dg))
    return 0


def cli(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", nargs="+", required=True,
                    help="campaign roots" if "--glob" == "--campaign" else "run-dir globs")
    args = ap.parse_args(argv)
    return main(getattr(args, "--glob".lstrip("-").replace("-", "_")))


if __name__ == "__main__":
    sys.exit(cli())
