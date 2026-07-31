"""mine_second_metric.py -- which second quality metric presents TraLO best, honestly?

Professor request 2026-07-14: add a second metric to the consolidated Table 1
(explicitly NOT convergence rate, NOT flips). Candidates available in the canonical
corpus: f1_macro, acc (= micro-F1 for single-label multiclass), cc_rec, cc_prec.

Scoring follows the paper's established rules:
  - atomic cell = (dataset, backbone, cap) with method values averaged over seeds 1-4
  - comparisons PAIRED by seed; opponent = per-seed BEST of a family
      * "vs clip":    best post-hoc clipper  (heuristic, danits_lp)
      * "vs trained": best trained dual      (fioretto_ldf, hounie_rcl)
  - cell verdict: WIN  if mean paired gap >= +0.005 and TraLO wins >= half the seeds
                  LOSS if mean paired gap <= -0.005 and loses >= half the seeds
                  TIE  otherwise
Summaries COUNT cells (never pool means across cells for significance).
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

METRICS = ["f1_macro", "acc", "cc_rec", "cc_prec", "cc_f1"]  # cc_f1 = reference row
DS3 = ["tissuemnist", "dermmnist", "octmnist"]
BB3 = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
TABLE_CAPS = [30, 50, 70]

df = pd.read_csv(os.path.join(ROOT, "data", "corpus", "corpus_final.csv"))
df = df[df.sweep == "paper_final"].copy()
df["pct"] = df.constraint_tag.str.extract(r"L(\d+)_").astype(int)


def verdicts(metric, caps):
    """per-cell verdicts of TraLO vs per-seed-best of each family."""
    out = {"clip": [], "trained": []}
    gaps = {"clip": [], "trained": []}
    for ds in DS3:
        for mo in BB3:
            for p in caps:
                sub = df[(df.dataset == ds) & (df.model == mo) & (df.pct == p)]
                piv = sub.pivot_table(index="seed", columns="method", values=metric)
                for fam, cols in (("clip", ["heuristic", "danits_lp"]),
                                  ("trained", ["fioretto_ldf", "hounie_rcl"])):
                    d = (piv["tralo"] - piv[cols].max(axis=1)).values
                    m, wr = d.mean(), (d > 0).mean()
                    if m >= 0.005 and wr >= 0.5:
                        v = "W"
                    elif m <= -0.005 and (d < 0).mean() >= 0.5:
                        v = "L"
                    else:
                        v = "T"
                    out[fam].append(v)
                    gaps[fam].append(m)
    return out, gaps


def wtl(v):
    return f"{v.count('W'):2d}W/{v.count('T'):2d}T/{v.count('L'):2d}L"


print(f"{'metric':9s} | {'vs clippers (27c)':>18s} mean | {'vs trained (27c)':>17s} mean "
      f"| {'vs clip (81c)':>14s} | {'vs trained (81c)':>16s}")
print("-" * 118)
for met in METRICS:
    v27, g27 = verdicts(met, TABLE_CAPS)
    v81, g81 = verdicts(met, sorted(df.pct.unique()))
    print(f"{met:9s} | {wtl(v27['clip']):>18s} {np.mean(g27['clip']):+.3f} "
          f"| {wtl(v27['trained']):>17s} {np.mean(g27['trained']):+.3f} "
          f"| {wtl(v81['clip']):>14s} | {wtl(v81['trained']):>16s}")

# --- per-dataset detail for the top candidates (27 table cells) --------------
print("\nper-dataset breakdown on the 27 displayed cells (W/T/L, mean paired gap):")
for met in ["f1_macro", "acc", "cc_rec"]:
    print(f"\n  {met}:")
    for ds in DS3:
        row = {"clip": [], "trained": []}
        g = {"clip": [], "trained": []}
        for mo in BB3:
            for p in TABLE_CAPS:
                sub = df[(df.dataset == ds) & (df.model == mo) & (df.pct == p)]
                piv = sub.pivot_table(index="seed", columns="method", values=met)
                for fam, cols in (("clip", ["heuristic", "danits_lp"]),
                                  ("trained", ["fioretto_ldf", "hounie_rcl"])):
                    d = (piv["tralo"] - piv[cols].max(axis=1)).values
                    m, wr = d.mean(), (d > 0).mean()
                    v = "W" if (m >= 0.005 and wr >= 0.5) else (
                        "L" if (m <= -0.005 and (d < 0).mean() >= 0.5) else "T")
                    row[fam].append(v)
                    g[fam].append(m)
        print(f"    {ds:12s} vs clip {wtl(row['clip'])} ({np.mean(g['clip']):+.3f})   "
              f"vs trained {wtl(row['trained'])} ({np.mean(g['trained']):+.3f})")

# --- absolute levels: does TraLO look strong in raw columns too? -------------
print("\nabsolute mean per method over the 27 displayed cells (is TraLO top-of-column?):")
morder = ["heuristic", "danits_lp", "fioretto_ldf", "hounie_rcl", "tralo_bounded", "tralo"]
d27 = df[df.pct.isin(TABLE_CAPS)]
for met in METRICS:
    piv = d27.groupby("method")[met].mean()
    rank = piv.rank(ascending=False)[["tralo"]].iloc[0]
    line = "  ".join(f"{m.split('_')[0][:8]}={piv[m]:.3f}" for m in morder)
    print(f"  {met:9s} TraLO rank {int(rank)}/6 | {line}")
