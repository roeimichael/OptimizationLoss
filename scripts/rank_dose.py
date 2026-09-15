"""How much of the budgeted ranking loss actually SURVIVES the batch sampler?

The loss skips a group with fewer than `rank_min_group` items, and skips a
(group, class) with no positives or no negatives. fmow2's TEST split has ten
groups over 3442 items and the batch is 64, so a batch holds only a handful of
items per group -- possibly below the min_group of 8. If most batches produce
zero usable terms, `rank_clip` is a near-inert flag: it would run, log, and move
the model barely or not at all, which is the exact shape of the five dead flags
already in this project's ledger. Worth knowing NOW, not after 120 runs.

Reads `train_meta.csv` and `train_labels.npy` only -- the group and label
columns of the TRAIN split. No images, no model, no gradient, no test data.
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd


def main(run_dir, batch_size=64, min_group=8, n_batches=2000):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    dc = cfg["dataset_config"]
    classes = dc["constrained_class"]
    classes = classes if isinstance(classes, list) else [classes]
    group_col = dc["group_column"]
    data_dir = dc.get("data_dir") or os.path.join("data", cfg["dataset"])

    y = np.load(os.path.join(data_dir, "train_labels.npy")).ravel()
    meta = pd.read_csv(os.path.join(data_dir, "train_meta.csv"))
    g = meta[group_col].to_numpy()
    assert len(g) == len(y), "train_meta and train_labels disagree in length"

    sizes = collections.Counter(g.tolist())
    print("train items %d | groups %d | constrained classes %s | batch %d | min_group %d"
          % (len(y), len(sizes), classes, batch_size, min_group))
    print("group shares: %s"
          % ", ".join("%s:%.1f%%" % (k, 100.0 * v / len(y))
                      for k, v in sizes.most_common()))
    print("constrained-class shares in TRAIN: %s"
          % ", ".join("cls%d:%.1f%%" % (c, 100.0 * (y == c).mean()) for c in classes))

    rng = np.random.default_rng(0)
    terms_per_batch = []
    hits = collections.Counter()
    for _ in range(n_batches):
        idx = rng.choice(len(y), size=batch_size, replace=False)
        by, bg = y[idx], g[idx]
        terms = 0
        for gid in np.unique(bg):
            m = bg == gid
            if m.sum() < min_group:
                continue
            for c in classes:
                n_pos = int((by[m] == c).sum())
                if n_pos == 0 or n_pos == int(m.sum()):
                    continue
                terms += 1
                hits[gid] += 1
        terms_per_batch.append(terms)

    t = np.array(terms_per_batch)
    print("")
    print("USABLE (group, class) TERMS PER BATCH over %d simulated batches" % n_batches)
    print("  mean %.2f   median %d   ZERO-term batches %.1f%%   max possible %d"
          % (t.mean(), int(np.median(t)), 100.0 * (t == 0).mean(),
             len(sizes) * len(classes)))
    print("")
    print("WHICH groups ever contribute (the loss is only as wide as this list):")
    for gid, n in hits.most_common():
        print("   %-14s in %5.1f%% of batches  (group is %.1f%% of train)"
              % (gid, 100.0 * n / (n_batches * len(classes)),
                 100.0 * sizes[gid] / len(y)))
    silent = sorted(str(k) for k in sizes if k not in hits)
    print("   NEVER contributes: %s" % (", ".join(silent) if silent else "none"))
    print("")
    if (t == 0).mean() > 0.5:
        print("VERDICT: over half of batches produce NO gradient from this loss. "
              "rank_clip is close to inert; min_group or the sampler must change "
              "before this campaign means anything.")
    elif len(hits) <= 2:
        print("VERDICT: the loss is carried by <=2 groups, so it is live but it "
              "is NOT training the per-group cut the allocator actually makes.")
    else:
        print("VERDICT: live across %d of %d groups, mean %.1f terms per batch."
              % (len(hits), len(sizes), t.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
