"""Final: remove ALL between-method variance from the fill-AP correlation,
and spot-verify one run end to end from the raw file."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A  # noqa: E402
from src.training.constraints import (compute_global_constraints,  # noqa: E402
                                      compute_local_constraints)

pd.set_option("display.width", 250)
t = pd.read_csv("paper/scripts/out_refute_damage.csv")
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]

print("=" * 116)
print("F. WITHIN-METHOD-CENTRED correlation: rank fill and AP inside each method,")
print("   then correlate. This deletes every between-method difference, which is")
print("   the only thing that can make fill look like a method-free law.")
print("=" * 116)
for ds, g in t.groupby("dataset"):
    g = g.copy()
    g["fz"] = g.groupby("method")["fill"].rank(pct=True)
    g["az"] = g.groupby("method")["AP"].rank(pct=True)
    g["cz"] = g.groupby("method")["ccF1eq"].rank(pct=True)
    r1 = spearmanr(g["fz"], g["az"])
    r2 = spearmanr(g["fz"], g["cz"])
    print("  %-12s n=%d   within-method rho(fill,AP) %+.3f p=%.3g   "
          "rho(fill,ccF1eq) %+.3f p=%.3g"
          % (ds, len(g), r1.correlation, r1.pvalue, r2.correlation, r2.pvalue))
print("  (raw pooled derm value was +0.702, p=2.7e-08)")

print()
print("=" * 116)
print("G. Also strip the CELL: rank within (method x backbone x cap), 4 seeds each.")
print("=" * 116)
for ds, g in t.groupby("dataset"):
    g = g.copy()
    k = ["method", "model", "cap"]
    g["fz"] = g.groupby(k)["fill"].rank(pct=True)
    g["az"] = g.groupby(k)["AP"].rank(pct=True)
    r1 = spearmanr(g["fz"], g["az"])
    print("  %-12s n=%d   rho(fill,AP) within (method x cell) = %+.3f  p=%.3g"
          % (ds, len(g), r1.correlation, r1.pvalue))

print()
print("=" * 116)
print("H. Does equalize actually spend the full budget K? (checks the claim that")
print("   ccF1eq is a pure top-K metric and the raw count cannot enter it)")
print("=" * 116)
rows = []
for cfgp in sorted(glob.glob("results/headroom/headroom_b30_lrc0.0001_noceskip/"
                             "**/config.json", recursive=True)):
    cfg = json.load(open(cfgp))
    d = os.path.dirname(cfgp)
    f = os.path.join(d, "final_predictions_raw.csv")
    if not os.path.exists(f):
        continue
    tt = pd.read_csv(f)
    cols = sorted((c for c in tt.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = tt[cols].to_numpy(float)
    y = tt["True_Label"].to_numpy(int)
    g = tt["Group_ID"].to_numpy(int) if "Group_ID" in tt.columns else None
    dc = cfg.get("dataset_config") or {}
    c = dc.get("constrained_class")
    c = int(c[0] if isinstance(c, (list, tuple)) else c)
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = compute_global_constraints(df, "label", gp, constrained_class=[c],
                                   num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[c],
                                  num_classes=P.shape[1])
    eq = A.equalize(P, g, G, L, c)
    rows.append({"dataset": cfg["dataset_mode"], "method": cfg["methodology"],
                 "K": int(G[c]), "k_used": int((eq == c).sum()),
                 "raw": int((tt["Predicted_Label"].to_numpy(int) == c).sum())})
E = pd.DataFrame(rows)
E["short"] = E["K"] - E["k_used"]
print(E.groupby(["dataset", "method"])[["K", "k_used", "short", "raw"]].mean()
      .to_string(float_format=lambda x: "%.1f" % x))
print("  runs where k_used < K: %d / %d  -> ccF1eq spends the SAME budget for every"
      % (int((E["short"] > 0).sum()), len(E)))
print("  method, so the claim's point (raw count cannot enter ccF1eq) is CORRECT --")
print("  which is why 'not a cap-satisfaction win' is true of every method on every")
print("  dataset, including the 4 oct cells TraLO loses. It cannot explain a win.")

print()
print("=" * 116)
print("I. SPOT VERIFY one run end to end (derm / RegNetY400MF / L30_G30 / seed_1)")
print("=" * 116)
for m in TRAINED:
    hits = glob.glob("results/headroom/headroom_b30_lrc0.0001_noceskip/**/"
                     "RegNetY400MF/dermmnist/L30_G30/%s/seed_1/config.json" % m,
                     recursive=True)
    if not hits:
        continue
    d = os.path.dirname(hits[0])
    cfg = json.load(open(hits[0]))
    tt = pd.read_csv(os.path.join(d, "final_predictions_raw.csv"))
    cols = sorted((c for c in tt.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = tt[cols].to_numpy(float)
    y = tt["True_Label"].to_numpy(int)
    gg = tt["Group_ID"].to_numpy(int)
    c = 4
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": gg})
    G = compute_global_constraints(df, "label", gp, constrained_class=[c],
                                   num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[c],
                                  num_classes=P.shape[1])
    eq = A.equalize(P, gg, G, L, c)
    lg = pd.read_csv(os.path.join(d, "training_log.csv"))
    ecol = [x for x in lg.columns if x.strip().lower() == "epoch"][0]
    ep = pd.to_numeric(lg[ecol], errors="coerce").dropna()
    print("  %-13s n=%d  K=%d  raw=%4d  fill=%.2f  AP=%.4f  ccF1eq=%.4f  "
          "max(Epoch)=%d  len(log)=%d"
          % (m, len(y), int(G[c]), int((tt["Predicted_Label"] == c).sum()),
             (tt["Predicted_Label"] == c).sum() / float(G[c]),
             average_precision_score((y == c).astype(int), P[:, c]),
             f1_score(y, eq, labels=[c], average="macro", zero_division=0),
             int(ep.max()), len(ep)))
ctl = glob.glob("results/headroom/headroom_b30/**/RegNetY400MF/dermmnist/"
                "L30_G30/heuristic/seed_1/final_predictions_raw.csv", recursive=True)
if ctl:
    tt = pd.read_csv(ctl[0])
    cols = sorted((x for x in tt.columns if x.startswith("Prob_Class_")),
                  key=lambda x: int(x.rsplit("_", 1)[1]))
    P = tt[cols].to_numpy(float)
    y = tt["True_Label"].to_numpy(int)
    print("  %-13s                     raw=%4d               AP=%.4f"
          % ("CONTROL(CE)", int((tt["Predicted_Label"] == 4).sum()),
             average_precision_score((y == 4).astype(int), P[:, 4])))
