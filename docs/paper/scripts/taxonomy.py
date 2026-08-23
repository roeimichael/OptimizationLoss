"""Failure taxonomy over every trained run of a headroom campaign.

WHAT THE LOGS ACTUALLY CONTAIN (read before trusting any number below)
----------------------------------------------------------------------
TraLO (src/methodologies/tralo/train.py:430) writes a CSV row iff
    (epoch + 1) % 5 == 0  or  is_satisfied  or  epoch == warmup_epochs
and the convergence break (line 426) fires BEFORE that block. Two consequences:
  * every satisfied epoch is logged, so an epoch ABSENT from the log was NOT
    satisfied. The binary satisfaction trace is therefore EXACTLY recoverable
    even though the log is sparse. Sparsity only costs excess MAGNITUDE.
  * the fifth consecutive satisfied epoch -- the one that trips the break -- is
    never logged, so a converged TraLO run executed exactly one epoch more than
    its log shows. Reconstructed here and flagged `inferred_tail`.

fioretto_ldf / hounie_rcl write one row per epoch INCLUDING the break epoch
(train.py writes the row before the break test) and record `total_excess` and
`all_satisfied`, but never a predicted count. total_excess is clipped at zero
(max(0, hard - K)), so the dual logs CANNOT show how far below the cap a run
went. Undershoot is only observable on the final restored checkpoint.

EXCESS IS THE SAME QUANTITY IN ALL THREE. Duals compute
  sum_c max(0, global_hard_c - K_c) + sum_{g,c} max(0, local_hard_{g,c} - K_{g,c})
over bounded classes only. TraLO does not write that scalar but writes every
term of it (Hard_Class*, Group*_Hard_Class*, Limit_Class*, Group*_Limit_Class*),
so it is reconstructed here term for term.

EPOCH INDEXING. TraLO's `Epoch` column is internal_epoch+1, constraint epochs
internal 1..29 -> logged 2..30. Duals' `epoch` column is internal 0..28. Both
are mapped to a common constraint-epoch index e = 1..29 (tralo: Epoch-1,
duals: epoch+1). Cross-check: for the same cell/seed, tralo Epoch 2 and dual
epoch 0 report an identical warm-up CE, confirming the alignment.

    python paper/scripts/taxonomy.py --root results/headroom/headroom_b30_lrc0.0001_noceskip
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import analyze_headroom as A                                          # noqa: E402
from src.utils.constants import UNLIMITED                             # noqa: E402
from src.training.constraints import (compute_global_constraints,     # noqa: E402
                                      compute_local_constraints)

TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
NCE = 29  # constraint epochs in this campaign


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_log(path):
    t = pd.read_csv(path, dtype=str, low_memory=False)
    # headers repeat mid-file (one per phase); drop any row that repeats them
    key = t.columns[0]
    t = t[t[key] != key]
    return t


def trace_tralo(path, cls):
    t = read_log(path)
    if "Epoch" not in t.columns:
        return None
    ep = num(t["Epoch"])
    keep = ep.notna()
    t, ep = t[keep], ep[keep]
    gsat = num(t.get("Global_Satisfied", pd.Series(index=t.index)))
    lsat = num(t.get("Local_Satisfied", pd.Series(index=t.index)))
    hard = num(t["Hard_Class%d" % cls])
    lim = num(t["Limit_Class%d" % cls])
    exc = np.maximum(0.0, hard - lim).fillna(0.0)
    ng = 0
    for c in t.columns:
        if c.startswith("Group") and c.endswith("_Hard_Class%d" % cls):
            g = c[len("Group"):-len("_Hard_Class%d" % cls)]
            lc = "Group%s_Limit_Class%d" % (g, cls)
            if lc in t.columns:
                gl = num(t[lc])
                gh = num(t[c])
                add = np.maximum(0.0, gh - gl)
                add = add.where(np.isfinite(gl), 0.0).fillna(0.0)
                exc = exc + add
                ng += 1
    sat = ((gsat == 1) & (lsat == 1)).astype(int)
    e = (ep - 1).astype(int)                 # common constraint-epoch index
    return pd.DataFrame({"e": e.values, "sat": sat.values,
                         "excess": exc.values, "count": hard.values,
                         "ce": num(t["L_CE"]).values,
                         "lam": num(t.get("Lambda_Global", pd.Series(index=t.index))).values}), ng


def trace_dual(path):
    t = read_log(path)
    ep = num(t["epoch"])
    keep = ep.notna()
    t, ep = t[keep], ep[keep]
    return pd.DataFrame({"e": (ep + 1).astype(int).values,
                         "sat": num(t["all_satisfied"]).fillna(0).astype(int).values,
                         "excess": num(t["total_excess"]).values,
                         "count": np.nan,
                         "ce": num(t["ce_loss"]).values,
                         "lam": num(t[[c for c in t.columns
                                       if c.startswith("max_lam")][0]]).values})


def score_run(d, cfg):
    raw = os.path.join(d, "final_predictions_raw.csv")
    fin = os.path.join(d, "final_predictions.csv")
    if not (os.path.exists(raw) and os.path.exists(fin)):
        return None
    t = pd.read_csv(raw)
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    P = t[cols].to_numpy(float)
    y = t["True_Label"].to_numpy(int)
    rawp = t["Predicted_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = compute_global_constraints(df, "label", gp, constrained_class=[cls],
                                   num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[cls],
                                  num_classes=P.shape[1])
    if G[cls] >= UNLIMITED:
        return None
    rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
    eq = A.equalize(P, g, G, L, cls)
    return {"cls": cls, "K": int(G[cls]), "count_raw": int((rawp == cls).sum()),
            "count_adj": int((rel == cls).sum()),
            "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
            "ccF1eq": f1_score(y, eq, labels=[cls], average="macro", zero_division=0),
            "macroEq": f1_score(y, eq, average="macro", zero_division=0)}


def evalcsv(d):
    p = os.path.join(d, "evaluation_metrics.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p)
    return dict(zip(t["Metric"], t["Value"]))


def f(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


def classify(row):
    """Named behaviour from the reconstructed satisfaction trace + final count.

    Priority order is explicit: collapse overrides the dynamics label because a
    model that satisfies by predicting almost nobody has a different failure
    mode from one that satisfies at the budget, and the outcome metric cares.
    """
    if row["n_sat"] == 0:
        return "NEVER_SAT"
    if row["count_raw"] < row["K"] / 3.0:
        return "COLLAPSE"
    if not row["held_tail"]:
        return "SAT_THEN_DRIFT"
    if row["n_down"] >= 1:
        return "OSC_THEN_LOCK"
    return "LOCK_AND_HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/headroom/headroom_b30_lrc0.0001_noceskip")
    ap.add_argument("--out", default="paper/scripts/out_taxonomy.csv")
    args = ap.parse_args()

    rows = []
    for cfg_path in sorted(glob.glob(args.root + "/**/config.json", recursive=True)):
        cfg = json.load(open(cfg_path))
        m = cfg.get("methodology")
        if m not in TRAINED:
            continue
        d = os.path.dirname(cfg_path)
        sc = score_run(d, cfg)
        if sc is None:
            continue
        lg = os.path.join(d, "training_log.csv")
        if not os.path.exists(lg):
            continue
        ng = None
        if m == "tralo":
            tr, ng = trace_tralo(lg, sc["cls"])
        else:
            tr = trace_dual(lg)
        tr = tr.dropna(subset=["e"]).sort_values("e")
        max_e = int(tr["e"].max())

        satmap = dict(zip(tr["e"].astype(int), tr["sat"].astype(int)))
        inferred_tail = False
        if m == "tralo":
            # every satisfied epoch is logged -> absent == unsatisfied
            E = max_e
            tail4 = [satmap.get(max_e - k, 0) for k in range(4)]
            if max_e < NCE and all(tail4):
                E = max_e + 1
                satmap[E] = 1
                inferred_tail = True
            s = np.array([satmap.get(e, 0) for e in range(1, E + 1)], dtype=int)
        else:
            E = max_e
            s = np.array([satmap.get(e, 0) for e in range(1, E + 1)], dtype=int)

        n_sat = int(s.sum())
        first_sat = int(np.argmax(s) + 1) if n_sat else None
        n_down = int(((s[:-1] == 1) & (s[1:] == 0)).sum()) if len(s) > 1 else 0
        n_up = int(((s[:-1] == 0) & (s[1:] == 1)).sum()) if len(s) > 1 else 0
        held_tail = bool(len(s) >= 5 and s[-5:].all())

        ex = tr["excess"].to_numpy(float)
        ev = evalcsv(d)
        hp = cfg["hyperparams"]
        # CE-saturation gate: ce_loss becomes NaN (np.mean of an empty list)
        # the first epoch the CE batch loop is skipped.
        ce = tr["ce"].to_numpy(float)
        cenan = np.where(~np.isfinite(ce))[0]
        ce_skip_e = int(tr["e"].to_numpy()[cenan[0]]) if len(cenan) else None

        r = {
            "path": d, "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
            "cap": cfg["constraint_tag"], "method": m, "seed": hp.get("seed"),
            "K": sc["K"], "count_raw": sc["count_raw"], "count_adj": sc["count_adj"],
            "ratio": sc["count_raw"] / sc["K"],
            "ccF1eq": sc["ccF1eq"], "AP": sc["AP"], "macroEq": sc["macroEq"],
            "epochs_run": E, "rows_logged": len(tr), "inferred_tail": inferred_tail,
            "n_sat": n_sat, "frac_sat": n_sat / E, "first_sat": first_sat,
            "n_down": n_down, "n_up": n_up, "held_tail": held_tail,
            "early_stop": E < NCE,
            "ex_first": ex[0], "ex_min": np.nanmin(ex), "ex_last": ex[-1],
            "ex_rises": int((np.diff(ex) > 0).sum()),
            "final_excess": f(ev.get("Raw Total Excess")),
            "final_sat": f(ev.get("Raw All Satisfied")),
            "flips": f(ev.get("Flips Required")),
            "sat_epoch_csv": f(ev.get("Satisfaction Epoch")),
            "min_excess_epoch": f(ev.get("Min Excess Epoch")),
            "min_total_excess": f(ev.get("Min Total Excess")),
            "restored_from": f(ev.get("Restored From Epoch")),
            "restore_kind": ev.get("Restore Kind"),
            "ce_skip_e": ce_skip_e,
            "ce_skip_cfg": hp.get("enable_ce_skip", "ABSENT(default True)"),
            "n_local_groups": ng,
        }
        r["klass"] = classify(r)
        rows.append(r)

    D = pd.DataFrame(rows)
    D.to_csv(args.out, index=False)
    print("runs classified: %d   -> %s" % (len(D), args.out))
    if D.empty:
        return 1

    P = lambda t: print(t.to_string(float_format=lambda x: "%.4f" % x))   # noqa: E731

    print("\n" + "=" * 96)
    print("0. SANITY: reconstruction vs what the run itself recorded")
    print("=" * 96)
    # tralo's 'Satisfaction Epoch' is internal+1 (== logged Epoch) so it is
    # first_sat+1; the duals' is internal+1 == first_sat.
    D["sat_epoch_norm"] = np.where(D.method == "tralo",
                                   D.sat_epoch_csv - 1, D.sat_epoch_csv)
    ok = D.dropna(subset=["sat_epoch_csv"])
    agree = (ok.sat_epoch_norm == ok.first_sat).sum()
    print("  runs with a recorded satisfaction epoch: %d ; reconstructed first_sat "
          "agrees on %d of them" % (len(ok), agree))
    never = D[D.sat_epoch_csv.isna()]
    print("  runs with NO recorded satisfaction epoch: %d ; of those, "
          "reconstructed n_sat==0 for %d" % (len(never), int((never.n_sat == 0).sum())))
    print("  satisfied-epoch <-> zero-excess consistency violations: %d"
          % int(((D.n_sat > 0) & (D.ex_min > 0)).sum()))
    print("\n  CE-saturation gate as configured per method (the campaign is named "
          "'noceskip'):")
    P(D.groupby("method").agg(ce_skip_cfg=("ce_skip_cfg", lambda x: x.iloc[0]),
                              runs_that_stopped_CE=("ce_skip_e",
                                                    lambda x: int(x.notna().sum())),
                              median_CE_stop_epoch=("ce_skip_e", "median")))

    print("\n" + "=" * 96)
    print("1. CLASS DEFINITIONS AND POPULATION  (144 trained runs)")
    print("=" * 96)
    print("""  NEVER_SAT      n_sat == 0 over every constraint epoch actually run
  COLLAPSE       n_sat >= 1 AND final restored model's raw count < K/3
  SAT_THEN_DRIFT n_sat >= 1, not COLLAPSE, and the last 5 run epochs are not all satisfied
  OSC_THEN_LOCK  n_sat >= 1, last 5 run epochs all satisfied, but >=1 satisfied->violated flip earlier
  LOCK_AND_HOLD  n_sat >= 1, last 5 run epochs all satisfied, zero satisfied->violated flips""")
    P(D.groupby("klass").agg(n=("klass", "size"),
                             ccF1eq=("ccF1eq", "mean"), AP=("AP", "mean"),
                             ratio=("ratio", "mean"),
                             first_sat=("first_sat", "mean"),
                             epochs=("epochs_run", "mean"),
                             flips=("flips", "mean")).sort_values("n", ascending=False))

    print("\n" + "=" * 96)
    print("2. CLASS x METHOD")
    print("=" * 96)
    P(pd.crosstab(D.method, D.klass))
    print("\n  class x method x dataset")
    P(pd.crosstab([D.dataset, D.method], D.klass))

    print("\n" + "=" * 96)
    print("3. CLASS x CELL  (dataset, backbone, cap) -- never pooled")
    print("=" * 96)
    for (ds, mo, cp), g in D.groupby(CELL):
        line = "  %-12s %-14s %-8s K=%3d | " % (ds, mo, cp, g.K.iloc[0])
        for m in TRAINED:
            h = g[g.method == m]
            ks = ",".join(sorted(set(h.klass)))
            line += "%-14s %-30s ccF1=%.3f  raw/K=%.2f | " % (
                m[:12], ks, h.ccF1eq.mean(), h.ratio.mean())
        print(line)

    print("\n" + "=" * 96)
    print("4. WITHIN-CELL OUTCOME BY CLASS")
    print("   Raw class means confound behaviour with cell difficulty, so every run's")
    print("   metric is expressed as a deviation from the mean of the 12 trained runs")
    print("   in ITS OWN cell. Cells are never pooled; the deviations are.")
    print("=" * 96)
    for c in ["ccF1eq", "AP", "macroEq"]:
        D["d_" + c] = D[c] - D.groupby(CELL)[c].transform("mean")
    P(D.groupby("klass").agg(n=("klass", "size"),
                             d_ccF1eq=("d_ccF1eq", "mean"),
                             d_AP=("d_AP", "mean"),
                             d_macroEq=("d_macroEq", "mean"),
                             cells=("path", lambda x: len(set(
                                 D.loc[x.index, CELL].agg("|".join, axis=1))))))
    print("\n  same, split by method (does the class carry signal beyond the method?)")
    P(D.groupby(["method", "klass"]).agg(n=("klass", "size"),
                                         d_ccF1eq=("d_ccF1eq", "mean"),
                                         d_AP=("d_AP", "mean")))

    print("\n" + "=" * 96)
    print("5. CONTINUOUS SIGNATURES vs OUTCOME (within-cell deviations)")
    print("=" * 96)
    num_cols = ["first_sat", "n_sat", "frac_sat", "n_down", "epochs_run",
                "ratio", "final_excess", "ex_min"]
    out = []
    for c in num_cols:
        v = pd.to_numeric(D[c], errors="coerce")
        m = v.notna()
        out.append({"signature": c, "n": int(m.sum()),
                    "corr_with_d_ccF1eq": v[m].corr(D.d_ccF1eq[m]),
                    "corr_with_d_AP": v[m].corr(D.d_AP[m])})
    P(pd.DataFrame(out).set_index("signature"))

    print("\n" + "=" * 96)
    print("6. HOW MANY EPOCHS DID EACH METHOD ACTUALLY GET?")
    print("=" * 96)
    P(D.groupby(["dataset", "method"]).agg(epochs=("epochs_run", "mean"),
                                           early_stop=("early_stop", "sum"),
                                           first_sat=("first_sat", "mean"),
                                           n_sat=("n_sat", "mean"),
                                           inferred=("inferred_tail", "sum")))

    print("\n" + "=" * 96)
    print("7. CHECKPOINT RESTORE: which model was actually evaluated?")
    print("=" * 96)
    D["restore_kind"] = D.restore_kind.fillna("none(final epoch kept)")
    P(pd.crosstab([D.method], D.restore_kind))
    P(D.groupby(["method", "restore_kind"]).agg(n=("path", "size"),
                                                d_ccF1eq=("d_ccF1eq", "mean"),
                                                ratio=("ratio", "mean")))

    print("\n" + "=" * 96)
    print("8. UNDERSHOOT: raw count of the evaluated model relative to its cap")
    print("=" * 96)
    P(D.pivot_table(index=["dataset", "model", "cap"], columns="method",
                    values="ratio", aggfunc="mean"))
    print("\n  runs with raw count < K/3 (COLLAPSE), by method x dataset")
    D["coll"] = D.count_raw < D.K / 3.0
    P(pd.crosstab([D.dataset, D.model], [D.method, D.coll]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
