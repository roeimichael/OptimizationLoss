"""Fourth pass: verify the CE-gate result before anyone repeats it.

Claim under test: the only configuration difference between
  results/headroom/headroom_b30_lrc0.0001_noceskip  and
  results/headroom/headroom_b30_lrc0.0001
is TraLO's `enable_ce_skip`, and that single flag moves TraLO's DermMNIST
outcome by an order of magnitude more than the TraLO-vs-duals margin the
campaign was built to measure.

    python paper/scripts/taxonomy4.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import taxonomy as T                                                  # noqa: E402

CELL = ["dataset", "model", "cap"]
A = "results/headroom/headroom_b30_lrc0.0001_noceskip"
B = "results/headroom/headroom_b30_lrc0.0001"
P = lambda t: print(t.to_string(float_format=lambda x: "%.4f" % x))    # noqa: E731


def index(root):
    out = {}
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        cfg = json.load(open(cp))
        m = cfg.get("methodology")
        if m not in ("tralo", "fioretto_ldf", "hounie_rcl"):
            continue
        k = (cfg["dataset_mode"], cfg["model_name"], cfg["constraint_tag"], m,
             cfg["hyperparams"]["seed"])
        out[k] = (os.path.dirname(cp), cfg)
    return out


IA, IB = index(A), index(B)
print("runs: noceskip=%d  gate-on-sibling=%d  shared keys=%d"
      % (len(IA), len(IB), len(set(IA) & set(IB))))

print("\n" + "=" * 96)
print("1. WHAT ACTUALLY DIFFERS BETWEEN THE TWO CAMPAIGNS' CONFIGS?")
print("=" * 96)
diffs = {}
for k in sorted(set(IA) & set(IB)):
    ha, hb = IA[k][1]["hyperparams"], IB[k][1]["hyperparams"]
    for f in set(ha) | set(hb):
        if ha.get(f, "<absent>") != hb.get(f, "<absent>"):
            diffs.setdefault((k[3], f), set()).add(
                (str(ha.get(f, "<absent>")), str(hb.get(f, "<absent>"))))
if not diffs:
    print("  no hyperparameter differs")
for (m, f), v in sorted(diffs.items()):
    print("  method=%-13s field=%-18s noceskip -> sibling : %s" % (m, f, sorted(v)))

print("\n" + "=" * 96)
print("2. ARE THE DUAL RUNS BYTE-IDENTICAL ACROSS THE TWO CAMPAIGNS?")
print("   (if yes the pipeline is deterministic and the paired delta below is")
print("    an exact single-factor contrast, not a re-run with seed noise)")
print("=" * 96)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


same = {}
for k in sorted(set(IA) & set(IB)):
    pa = os.path.join(IA[k][0], "final_predictions_raw.csv")
    pb = os.path.join(IB[k][0], "final_predictions_raw.csv")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        continue
    if IA[k][0] == IB[k][0]:
        same.setdefault(k[3], {"SAMEDIR": 0}).setdefault("SAMEDIR", 0)
        same[k[3]]["SAMEDIR"] = same[k[3]].get("SAMEDIR", 0) + 1
        continue
    eq = md5(pa) == md5(pb)
    same.setdefault(k[3], {})
    same[k[3]][eq] = same[k[3]].get(eq, 0) + 1
for m, v in sorted(same.items()):
    print("  %-13s identical-predictions counts: %s" % (m, v))
k0 = sorted([k for k in set(IA) & set(IB) if k[3] == "fioretto_ldf"])[0]
print("  example dual pair:\n    %s\n    %s" % (IA[k0][0], IB[k0][0]))

print("\n" + "=" * 96)
print("3. PAIRED EFFECT OF TraLO'S CE GATE, PER DATASET AND CELL")
print("   delta = (gate disabled) - (gate enabled), same cell, same seed")
print("=" * 96)
rows = []
for k in sorted(set(IA) & set(IB)):
    if k[3] != "tralo":
        continue
    sa = T.score_run(*IA[k])
    sb = T.score_run(*IB[k])
    if sa is None or sb is None:
        continue
    rows.append({"dataset": k[0], "model": k[1], "cap": k[2], "seed": k[4],
                 "ccF1_off": sa["ccF1eq"], "ccF1_on": sb["ccF1eq"],
                 "AP_off": sa["AP"], "AP_on": sb["AP"],
                 "ratio_off": sa["count_raw"] / sa["K"],
                 "ratio_on": sb["count_raw"] / sb["K"]})
R = pd.DataFrame(rows)
R["d_ccF1"] = R.ccF1_off - R.ccF1_on
R["d_AP"] = R.AP_off - R.AP_on
P(R.groupby("dataset").agg(n=("seed", "size"), d_ccF1=("d_ccF1", "mean"),
                           wins=("d_ccF1", lambda x: int((x > 0).sum())),
                           d_AP=("d_AP", "mean"),
                           ratio_off=("ratio_off", "mean"),
                           ratio_on=("ratio_on", "mean")))
print()
P(R.groupby(CELL).agg(d_ccF1=("d_ccF1", "mean"), d_AP=("d_AP", "mean"),
                      ratio_off=("ratio_off", "mean"), ratio_on=("ratio_on", "mean")))

print("\n" + "=" * 96)
print("4. SIZE OF THE FLAG EFFECT vs THE MARGIN THE CAMPAIGN REPORTS")
print("=" * 96)
S = []
for tag, I in [("gate OFF for tralo (noceskip)", IA), ("gate ON for all (sibling)", IB)]:
    for k, (d, cfg) in I.items():
        s = T.score_run(d, cfg)
        if s is None:
            continue
        S.append({"campaign": tag, "dataset": k[0], "model": k[1], "cap": k[2],
                  "method": k[3], "seed": k[4], "ccF1eq": s["ccF1eq"], "AP": s["AP"]})
S = pd.DataFrame(S)
for tag, g in S.groupby("campaign"):
    piv = g.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    piv["vBest"] = piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1)
    cell = piv.groupby(level=[0, 1, 2]).vBest.mean()
    print("\n  %s" % tag)
    for ds, gg in cell.groupby(level=0):
        print("    %-12s per-cell tralo-minus-best-dual ccF1eq: %s   -> wins %d of %d cells"
              % (ds, " ".join("%+.4f" % v for v in gg.values),
                 int((gg > 0).sum()), len(gg)))

print("\n" + "=" * 96)
print("5. DID THE DUALS' CE GATE FIRE IN THE SIBLING CAMPAIGN TOO?")
print("=" * 96)
rows = []
for tag, I in [("noceskip", IA), ("sibling", IB)]:
    for k, (d, cfg) in I.items():
        lp = os.path.join(d, "training_log.csv")
        if not os.path.exists(lp):
            continue
        cls = cfg["dataset_config"]["constrained_class"]
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        tr = (T.trace_tralo(lp, cls)[0] if k[3] == "tralo" else T.trace_dual(lp))
        ce = tr["ce"].to_numpy(float)
        bad = np.where(~np.isfinite(ce))[0]
        rows.append({"campaign": tag, "dataset": k[0], "method": k[3],
                     "stopped": len(bad) > 0,
                     "stop_e": tr["e"].to_numpy()[bad[0]] if len(bad) else np.nan,
                     "epochs": int(tr["e"].max())})
G = pd.DataFrame(rows)
P(G.pivot_table(index=["dataset", "method"], columns="campaign",
                values="stopped", aggfunc="sum"))
print("\n  median epoch at which CE training stopped:")
P(G[G.stopped].pivot_table(index=["dataset", "method"], columns="campaign",
                           values="stop_e", aggfunc="median"))
