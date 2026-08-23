"""Oscillation vs monotone-descent statistics for the reconstructed hounie_rcl
soft-count trajectory, split on whether the CE batch loop was running.

    step        mean |soft[t] - soft[t-1]|   (epoch-to-epoch churn)
    drift       (soft[last] - soft[first]) / n_steps  (net movement per epoch)
    |drift|/step  ~0 = pure oscillation, ~1 = pure monotone descent

    python paper/scripts/hounie_osc.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import hounie_dyn as H   # noqa: E402
import glob             # noqa: E402
import json             # noqa: E402

ROOTS = {"lrc1e-4": "results/headroom/headroom_b30_lrc0.0001_noceskip",
         "lrc5e-5": "results/headroom/headroom_b30_lrc5e-05"}


def seg(s):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return None
    d = np.diff(s)
    step = float(np.mean(np.abs(d)))
    drift = float((s[-1] - s[0]) / len(d))
    return step, drift, abs(drift) / step if step > 0 else np.nan, len(s)


def main():
    rows = []
    for tag, root in ROOTS.items():
        for p in sorted(glob.glob(root + "/**/config.json", recursive=True)):
            cfg = json.load(open(p))
            if cfg.get("methodology") != "hounie_rcl":
                continue
            r = H.load_run(p)
            if r is None:
                continue
            T = H.hounie_traj(r)
            ce_off = r["log"]["ce_loss"].isna().to_numpy()
            for lbl, mask in [("CE on", ~ce_off), ("CE off", ce_off)]:
                v = seg(T["soft"][mask])
                if v is None:
                    continue
                rows.append(dict(campaign=tag, dataset=r["dataset"], model=r["model"],
                                 cap=r["cap"], seed=r["seed"], phase=lbl,
                                 n_ep=v[3], step=v[0], drift=v[1], ratio=v[2],
                                 K=r["K"]))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("mean over runs; 'step' = |change in soft count| per epoch, "
          "'drift' = net change per epoch (negative = toward/through the cap)")
    print("=" * 100)
    print(d.groupby(["campaign", "dataset", "phase"]).agg(
        runs=("seed", "size"), mean_epochs=("n_ep", "mean"), K=("K", "mean"),
        step=("step", "mean"), drift=("drift", "mean"),
        drift_over_step=("ratio", "mean")
    ).to_string(float_format=lambda x: "%.3f" % x))
    return 0


if __name__ == "__main__":
    sys.exit(main())
