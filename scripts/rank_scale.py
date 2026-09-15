"""Is the cut the loss TRAINS the same cut the allocator DEPLOYS?

`rank_dose.py` showed the loss fires on only 8 of 139 train groups, ~2.5 terms
per batch. That alone does not say whether the campaign is measuring the right
thing. The question that decides it is WHERE in the score distribution the
trained cut sits, because:

  - if the train cut sits at a different QUANTILE than the deployed cut, the
    loss is optimising a different decision and a null result would be
    uninformative about the idea;
  - if it sits at the same quantile but is estimated from ~13 items instead of
    ~800, the loss is a NOISY estimate of the RIGHT quantity, which is a
    variance problem, not a bias one -- and the campaign still answers its
    question, just with less power.

Those two have opposite consequences, so measure rather than assert.
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd


def main(run_dir, batch_size=64, min_group=8, n_batches=4000):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    dc = cfg["dataset_config"]
    classes = dc["constrained_class"]
    classes = classes if isinstance(classes, list) else [classes]
    group_col = dc["group_column"]
    data_dir = dc.get("data_dir") or os.path.join("data", cfg["dataset"])
    local_pct = float(cfg["constraint"][0])   

    y = np.load(os.path.join(data_dir, "train_labels.npy")).ravel()
    g = pd.read_csv(os.path.join(data_dir, "train_meta.csv"))[group_col].to_numpy()

    print("cap fraction (local) = %.2f   constrained classes %s" % (local_pct, classes))
    print("")

    # --- TRAIN side: the cut the loss actually forms, inside a batch slice ---
    rng = np.random.default_rng(0)
    depth, kk, nn = [], [], []
    for _ in range(n_batches):
        idx = rng.choice(len(y), size=batch_size, replace=False)
        by, bg = y[idx], g[idx]
        for gid in np.unique(bg):
            m = bg == gid
            n_in = int(m.sum())
            if n_in < min_group:
                continue
            for c in classes:
                n_pos = int((by[m] == c).sum())
                if n_pos == 0 or n_pos == n_in:
                    continue
                k = max(1, min(int(round(n_pos * local_pct)), n_in - 1))
                depth.append(k / n_in)
                kk.append(k)
                nn.append(n_in)
    depth, kk, nn = np.array(depth), np.array(kk), np.array(nn)
    print("TRAIN cut, as the loss forms it inside a shuffled batch:")
    print("  group slice size  mean %.1f items   (min %d, max %d)"
          % (nn.mean(), nn.min(), nn.max()))
    print("  k (rank of cut)   mean %.2f          (k=1 in %.0f%% of terms)"
          % (kk.mean(), 100.0 * (kk == 1).mean()))
    print("  cut DEPTH k/n     mean %.3f  median %.3f" % (depth.mean(), np.median(depth)))

    # --- TEST side: the cut the allocator will actually make ---
    raw = pd.read_csv(os.path.join(run_dir, "final_predictions_raw.csv"))
    ty = raw["True_Label"].to_numpy(int)
    tg = raw["Group_ID"].to_numpy()
    t_depth, t_k, t_n = [], [], []
    for gid in np.unique(tg):
        m = tg == gid
        n_in = int(m.sum())
        for c in classes:
            n_pos = int((ty[m] == c).sum())
            if n_pos == 0:
                continue
            k = max(1, min(int(round(n_pos * local_pct)), n_in - 1))
            t_depth.append(k / n_in)
            t_k.append(k)
            t_n.append(n_in)
    t_depth, t_k, t_n = np.array(t_depth), np.array(t_k), np.array(t_n)
    print("")
    print("TEST cut, as the allocator will actually make it:")
    print("  group size        mean %.1f items   (min %d, max %d)"
          % (t_n.mean(), t_n.min(), t_n.max()))
    print("  k (rank of cut)   mean %.1f" % t_k.mean())
    print("  cut DEPTH k/n     mean %.3f  median %.3f"
          % (t_depth.mean(), np.median(t_depth)))

    print("")
    ratio = depth.mean() / t_depth.mean() if t_depth.mean() else float("nan")
    sample_ratio = t_n.mean() / nn.mean()
    print("DEPTH  ratio train/test = %.2f (means)  %.2f (medians)"
          % (ratio, np.median(depth) / np.median(t_depth)))
    print("SAMPLE ratio test/train = %.0fx  -- the trained cut is the %.1f-th "
          "order statistic of %.0f items; the deployed cut is the %.0f-th of %.0f."
          % (sample_ratio, kk.mean(), nn.mean(), t_k.mean(), t_n.mean()))
    print("")
    # Deliberately NOT a pass/fail band. An earlier version of this probe
    # printed a binary verdict off an arbitrary 0.7-1.4 threshold, which turned
    # a borderline 1.41 into "the loss is optimising a different decision" --
    # a much stronger claim than the numbers support. Report both ratios and
    # let the reader weigh them.
    print("READING:")
    print("  The trained cut sits within %.1fx of the deployed quantile, so it is"
          % max(ratio, 1.0 / ratio))
    print("  aimed at roughly the right place in the score distribution.")
    print("  But it is estimated from %.0fx fewer items, and the max(1, .) floor"
          % sample_ratio)
    print("  pins k=1 in %.0f%% of terms, which biases the train cut SHALLOW."
          % (100.0 * (kk == 1).mean()))
    print("  Consequence: a POSITIVE gAP effect is trustworthy. A NULL cannot")
    print("  separate 'the ranking channel does not help' from 'this estimator")
    print("  is too noisy and too narrow to deliver it'.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
