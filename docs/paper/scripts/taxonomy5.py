"""Fifth pass: (a) TraLO's CE-skip signature is NOT NaN -- it writes
L_CE == 0.0 exactly, because tralo/train.py:150-153 forces num_batches=1 and
epoch_ce stays 0.0, and train_acc is forced to 1.0 at line 171. The duals use
np.mean([]) -> NaN. Detect both properly.
(b) reclassify the gate-on sibling campaign with the same taxonomy so the class
shift caused by the single flag is visible.

    python paper/scripts/taxonomy5.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import taxonomy as T                                                  # noqa: E402

CELL = ["dataset", "model", "cap"]
P = lambda t: print(t.to_string(float_format=lambda x: "%.4f" % x))    # noqa: E731


def ce_stop_epoch(tr, method):
    """First constraint epoch at which the CE batch loop was skipped."""
    ce = tr["ce"].to_numpy(float)
    e = tr["e"].to_numpy()
    if method == "tralo":
        # avg_ce = 0.0/1 exactly when skip_ce is on (train.py:150-153,363)
        bad = np.where(ce == 0.0)[0]
    else:
        bad = np.where(~np.isfinite(ce))[0]      # np.mean([]) -> nan
    return int(e[bad[0]]) if len(bad) else None


print("=" * 96)
print("A. CE-SKIP DETECTION, CORRECTED FOR TraLO'S DIFFERENT SIGNATURE")
print("=" * 96)
rows = []
for tag, root in [("gate OFF for tralo", "results/headroom/headroom_b30_lrc0.0001_noceskip"),
                  ("gate ON for all", "results/headroom/headroom_b30_lrc0.0001")]:
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cp))
        m = cfg.get("methodology")
        if m not in ("tralo", "fioretto_ldf", "hounie_rcl"):
            continue
        d = os.path.dirname(cp)
        cls = cfg["dataset_config"]["constrained_class"]
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp = os.path.join(d, "training_log.csv")
        tr = (T.trace_tralo(lp, cls)[0] if m == "tralo" else T.trace_dual(lp))
        rows.append({"campaign": tag, "dataset": cfg["dataset_mode"],
                     "model": cfg["model_name"], "cap": cfg["constraint_tag"],
                     "method": m, "seed": cfg["hyperparams"]["seed"],
                     "stop_e": ce_stop_epoch(tr, m),
                     "cfg_flag": str(cfg["hyperparams"].get("enable_ce_skip", "ABSENT"))})
G = pd.DataFrame(rows)
G["stopped"] = G.stop_e.notna()
print("  runs whose CE batch loop stopped early (of 16 per dataset x method):")
P(G.pivot_table(index=["dataset", "method"], columns="campaign", values="stopped",
                aggfunc="sum"))
print("\n  median constraint epoch at which it stopped (of 29):")
P(G[G.stopped].pivot_table(index=["dataset", "method"], columns="campaign",
                           values="stop_e", aggfunc="median"))
print("\n  configured flag, by method and campaign:")
P(G.pivot_table(index="method", columns="campaign", values="cfg_flag",
                aggfunc=lambda x: x.iloc[0]))

print("\n" + "=" * 96)
print("B. RECLASSIFY THE GATE-ON SIBLING WITH THE SAME TAXONOMY")
print("=" * 96)
os.system(sys.executable + " paper/scripts/taxonomy.py --root "
          "results/headroom/headroom_b30_lrc0.0001 "
          "--out paper/scripts/out_taxonomy_gateon.csv > /dev/null 2>&1")
Aoff = pd.read_csv("paper/scripts/out_taxonomy.csv")
Aon = pd.read_csv("paper/scripts/out_taxonomy_gateon.csv")
Aoff["campaign"] = "gate OFF for tralo"
Aon["campaign"] = "gate ON for all"
Z = pd.concat([Aoff, Aon], ignore_index=True)
print("  TraLO class distribution, by dataset, in the two campaigns:")
P(pd.crosstab([Z[Z.method == "tralo"].dataset, Z[Z.method == "tralo"].campaign],
              Z[Z.method == "tralo"].klass))
print("\n  duals (unchanged by construction -- byte-identical runs):")
P(pd.crosstab([Z[Z.method != "tralo"].campaign], Z[Z.method != "tralo"].klass))

print("\n" + "=" * 96)
print("C. THE MECHANISM, PER DERM CELL: CE steps stop -> the constraint's one")
print("   step per epoch is the only force left -> the class count runs to zero")
print("=" * 96)
t = Z[(Z.method == "tralo") & (Z.dataset == "dermmnist")]
P(t.groupby(["model", "cap", "campaign"]).agg(
    n=("path", "size"), klass=("klass", lambda x: ",".join(sorted(set(x)))),
    K=("K", "mean"), count_raw=("count_raw", "mean"), ratio=("ratio", "mean"),
    n_sat=("n_sat", "mean"), first_sat=("first_sat", "mean"),
    epochs=("epochs_run", "mean"), ccF1eq=("ccF1eq", "mean"), AP=("AP", "mean")))

print("\n  and the duals in the same cells, which had the gate on in BOTH campaigns:")
u = Z[(Z.method != "tralo") & (Z.dataset == "dermmnist") &
      (Z.campaign == "gate OFF for tralo")]
P(u.groupby(["model", "cap", "method"]).agg(
    klass=("klass", lambda x: ",".join(sorted(set(x)))), K=("K", "mean"),
    count_raw=("count_raw", "mean"), ratio=("ratio", "mean"),
    ccF1eq=("ccF1eq", "mean"), AP=("AP", "mean")))
