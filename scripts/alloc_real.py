"""Does the allocator gap survive on REAL probabilities?

`alloc_gap.py` priced the shipped greedy allocator against an exact
transportation LP on synthetic, well-calibrated softmaxes and found +0.0095
accuracy at the separation our models have. Real models are overconfident, and
overconfidence is exactly what makes greedy look good -- nothing is contested.
This runs the identical comparison on stored predictions, so the synthetic
number either survives or it does not.

Same model, same probabilities, two allocators: the only difference is the
allocation rule, so any gap is PURELY the allocator.
"""
import glob, json, os, sys, collections
import numpy as np, pandas as pd
from scripts.alloc_gap import optimal
from src.methodologies.heuristic.train import _build_hierarchy, apply_allocation_heuristic
from src.training.constraints import compute_global_constraints, compute_local_constraints
from src.utils.constants import UNLIMITED


def f1(y, pred, classes):
    out = []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def one(d):
    fp, cj = os.path.join(d, "final_predictions_raw.csv"), os.path.join(d, "config.json")
    if not (os.path.exists(fp) and os.path.exists(cj)):
        return None
    cfg = json.load(open(cj))
    df = pd.read_csv(fp)
    cols = sorted((c for c in df.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    probs = df[cols].to_numpy(dtype=float)
    y, groups = df.True_Label.to_numpy(), df.Group_ID.to_numpy()
    K = probs.shape[1]
    classes = cfg["dataset_config"]["constrained_class"]
    classes = classes if isinstance(classes, list) else [classes]
    lp_frac, gp_frac = cfg["constraint"]
    frame = pd.DataFrame({"label": y, "g": groups})
    gcon = compute_global_constraints(frame, "label", gp_frac,
                                      constrained_class=classes, num_classes=K)
    lcon = compute_local_constraints(frame, "label", lp_frac, "g",
                                     constrained_class=classes, num_classes=K)
    greedy, _ = apply_allocation_heuristic(
        probs, groups, _build_hierarchy(K, gcon, classes), gcon, lcon, K)
    exact, exact_obj = optimal(probs, groups, gcon, lcon, K)
    # The LP maximises ASSIGNED PROBABILITY, not accuracy. Only the objective
    # is guaranteed to improve; accuracy is a proxy and can move either way.
    greedy_obj = float(probs[np.arange(len(probs)), greedy].sum())
    return (cfg["model_name"], cfg["dataset_mode"], cfg["constraint_tag"], cfg["arm"],
            (greedy == y).mean(), (exact == y).mean(),
            f1(y, greedy, classes), f1(y, exact, classes),
            float((greedy != exact).mean()), greedy_obj, float(exact_obj))


def main(globs):
    per = collections.defaultdict(list)
    for g in globs:
        for d in sorted(glob.glob(g)):
            r = one(d)
            if r:
                per[r[:4]].append(r[4:])
    if not per:
        print("no runs matched")
        return 1
    print("shipped GREEDY allocator vs the exact LP, same stored probabilities")
    print("  %-11s %-8s %-9s %-11s %3s %9s %9s %9s %9s %8s %8s" %
          ("backbone", "data", "cap", "arm", "n",
           "acc", "d_acc", "ccF1", "d_ccF1", "moved", "d_obj"))
    for k in sorted(per):
        v = np.array(per[k])
        print("  %-11s %-8s %-9s %-11s %3d %9.4f %+9.4f %9.4f %+9.4f %7.1f%% %7.3f%%" %
              (k[0], k[1], k[2], k[3], len(v), v[:, 0].mean(),
               (v[:, 1] - v[:, 0]).mean(), v[:, 2].mean(),
               (v[:, 3] - v[:, 2]).mean(), 100 * v[:, 4].mean(),
               100 * ((v[:, 6] - v[:, 5]) / v[:, 6]).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
