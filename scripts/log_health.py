"""Read the TRAINING PROCESS out of training_log.csv, per run, before any metric.

The house rule is to validate from the training log and never from a final
number, and until now nothing executed that rule. `full_panel` scores outcomes;
this says what the optimisation actually DID, which is the only thing that says
what to change next.

Handles both log schemas. tralo/select write the canonical wide one (Epoch,
Train_Acc, per-class Hard/Soft/Limit); the three dual arms write their own
narrow one (epoch, train_acc, ce_loss, total_excess, ...). Anything the narrow
schema cannot answer is reported as "n/a (schema)", never silently skipped -- a
check that quietly does not run on 60% of a campaign is worse than no check.

    python -m scripts.log_health <campaign-root> [--full]
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

COLLAPSE_DROP = 0.02   # ~10x the epoch-to-epoch wobble of a converged run


def _col(df, *names):
    return next((n for n in names if n in df.columns), None)


def read_run(d):
    """Everything this run's log can tell us. Missing -> None, never a guess."""
    try:
        df = pd.read_csv(os.path.join(d, "training_log.csv"))
    except Exception:
        return None
    if df.empty:
        return None
    cfg = {}
    try:
        with open(os.path.join(d, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception:
        pass

    ep = _col(df, "Epoch", "epoch")
    acc = _col(df, "Train_Acc", "train_acc")
    ce = _col(df, "L_CE", "ce_loss")
    gn = _col(df, "Grad_Norm", "grad_norm")
    sat = _col(df, "Global_Satisfied", "all_satisfied")

    r = {"dir": d, "rows": len(df), "wide": "Train_Acc" in df.columns,
         "status": cfg.get("status"), "arm": cfg.get("arm")}

    # --- collapse on the FINAL epoch: the pipeline keeps it unconditionally ---
    r["collapse"] = None
    r["acc_final"] = None
    if acc:
        a = pd.to_numeric(df[acc], errors="coerce").dropna()
        if len(a) >= 2 and float(a.iloc[-1]) < float(a.iloc[-2]) - COLLAPSE_DROP:
            r["collapse"] = (float(a.iloc[-2]), float(a.iloc[-1]))
        if len(a):
            r["acc_final"] = float(a.iloc[-1])

    # --- non-finite in a column that is actually being written. A diverged run
    #     once wrote `completed`, so this matters -- but scanning every column
    #     blindly flags every run: the wide header reserves per-group columns
    #     that a run with no local cap on that class never fills, and post-hoc
    #     arms log sparsely. A column that is entirely blank was NOT LOGGED;
    #     only a NaN sitting beside real values is evidence of divergence. ---
    # The warm-up row is excluded. It is logged before the constraint object
    # exists, so its limits and group counts are legitimately blank -- and
    # including it flagged every run that happened not to load a cached warm-up
    # (exactly one NaN per constraint column), which is noise, not divergence.
    r["nonfinite"] = {}
    scan = df[pd.to_numeric(df[ep], errors="coerce") >= 2] if (ep and r["wide"]) else df
    for c in scan.select_dtypes(include=[np.number]).columns:
        v = scan[c].to_numpy(dtype=float)
        finite = np.isfinite(v)
        if finite.any() and not finite.all():  # all-blank column = not logged
            r["nonfinite"][c] = int((~finite).sum())

    # --- did the constraint ever hold? warm-up rows are excluded: their counts
    #     are zero, which registers as trivially satisfied ---
    # A post-hoc arm runs NO constraint phase, so "satisfied" is vacuously true
    # for it every epoch. Reporting 28/28 there invites exactly the comparison
    # the house rules forbid -- feasibility is not a metric, and a clipper is
    # feasible by construction because the allocator makes it so.
    # The tell is a FINITE per-class limit in the log, not the presence of the
    # count columns: the canonical header reserves those for every arm, and a
    # post-hoc arm builds no constraint object, so every limit it writes is
    # UNLIMITED. Keying on the columns reported `clip` as 28/28 satisfied.
    r["posthoc"] = not any(
        c.startswith("Limit_Class")
        and (pd.to_numeric(df[c], errors="coerce").dropna() < 1e9).any()
        for c in df.columns)
    r["sat"] = None
    if sat and ep and not r["posthoc"]:
        con = df[pd.to_numeric(df[ep], errors="coerce") >= 2] if r["wide"] else df
        v = pd.to_numeric(con[sat], errors="coerce").dropna()
        if len(v):
            r["sat"] = (int(v.sum()), len(v))

    # --- CE health ---
    r["ce"] = None
    if ce:
        c = pd.to_numeric(df[ce], errors="coerce").dropna()
        if len(c) >= 2:
            r["ce"] = (float(c.iloc[0]), float(c.iloc[-1]))

    # --- gradient norm. A norm pinned to one value on every epoch means the
    #     clip binds every step, i.e. step MAGNITUDE is a no-op there and only
    #     direction and count are live levers. That is a finding, not noise. ---
    r["gn_pinned"] = None
    if gn:
        g = pd.to_numeric(df[gn], errors="coerce").dropna()
        g = g[g > 0]
        if len(g) >= 3:
            med = float(g.median())
            r["gn_pinned"] = (med, float(g.std()) < 1e-6 * max(1.0, med))

    # --- capped-class count trajectory, wide schema only ---
    r["counts"] = {}
    for c in range(20):
        hc, lc = "Hard_Class%d" % c, "Limit_Class%d" % c
        if hc not in df.columns or lc not in df.columns:
            continue
        lim = pd.to_numeric(df[lc], errors="coerce").dropna()
        if lim.empty or float(lim.iloc[-1]) >= 1e9:
            continue                              # uncapped class
        h = pd.to_numeric(df[hc], errors="coerce").dropna()
        if r["wide"] and len(h) > 1:
            h = h.iloc[1:]                        # drop the warm-up row's zeros
        if len(h) < 3:
            continue
        K = float(lim.iloc[-1])
        y = h.to_numpy(dtype=float)
        slope = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
        r["counts"][c] = {"K": K, "first": float(y[0]), "last": float(y[-1]),
                          "mean": float(y.mean()), "min": float(y.min()),
                          "max": float(y.max()), "slope": slope}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full", action="store_true", help="one block per run")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "**", "config.json"),
                             recursive=True))
    runs = [r for r in (read_run(os.path.dirname(p)) for p in paths) if r]
    if not runs:
        print("no readable training logs under", args.root)
        return

    print("%d run(s) with a readable log  (%d wide schema, %d narrow)"
          % (len(runs), sum(r["wide"] for r in runs),
             sum(not r["wide"] for r in runs)))

    bad = [r for r in runs if r["collapse"]]
    if bad:
        print("\nTERMINAL COLLAPSE -- the pipeline keeps the last epoch, so this "
              "IS the scored model")
        for r in bad:
            print("   %-54s %.4f -> %.4f"
                  % (os.path.relpath(r["dir"], args.root),
                     r["collapse"][0], r["collapse"][1]))

    nf = [r for r in runs if r["nonfinite"]]
    if nf:
        print("\nNON-FINITE VALUES (a diverged run once wrote `completed`)")
        for r in nf:
            worst = sorted(r["nonfinite"].items(), key=lambda kv: -kv[1])[:3]
            print("   %-54s %s"
                  % (os.path.relpath(r["dir"], args.root),
                     ", ".join("%s x%d" % kv for kv in worst)))

    print("\nPER ARM")
    print("  %-14s %5s %8s %14s   %s"
          % ("arm", "runs", "acc", "satisfied", "capped-class count vs K"))
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"] or "?", []).append(r)
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        accs = [r["acc_final"] for r in rs if r["acc_final"] is not None]
        sats = [r["sat"] for r in rs if r["sat"]]
        if sats:
            satr = "%d/%d" % (sum(s[0] for s in sats), sum(s[1] for s in sats))
        elif all(r.get("posthoc") for r in rs):
            satr = "n/a (posthoc)"
        else:
            satr = "n/a (schema)"
        cs = []
        for c in sorted({c for r in rs for c in r["counts"]}):
            v = [r["counts"][c] for r in rs if c in r["counts"]]
            cs.append("c%d %.0f->%.0f (K=%.0f, slope %+.2f/ep)"
                      % (c, np.mean([x["first"] for x in v]),
                         np.mean([x["last"] for x in v]), v[0]["K"],
                         np.mean([x["slope"] for x in v])))
        print("  %-14s %5d %8s %14s   %s"
              % (arm, len(rs), "%.4f" % np.mean(accs) if accs else "n/a",
                 satr, "; ".join(cs) if cs else "n/a (schema)"))

    pinned = [r for r in runs if r["gn_pinned"] and r["gn_pinned"][1]]
    if pinned:
        print("\n%d run(s) have a gradient norm pinned to one value on every "
              "epoch." % len(pinned))
        print("   The clip binds every step, so step MAGNITUDE is a no-op there")
        print("   and only step DIRECTION and step COUNT are live levers.")

    if args.full:
        for r in runs:
            print("\n%s" % os.path.relpath(r["dir"], args.root))
            print("   rows=%d status=%s acc_final=%s sat=%s"
                  % (r["rows"], r["status"], r["acc_final"], r["sat"]))
            for c, v in sorted(r["counts"].items()):
                print("   class %d: K=%.0f  %.0f -> %.0f  mean %.0f  "
                      "range %.0f-%.0f  slope %+.2f/ep"
                      % (c, v["K"], v["first"], v["last"], v["mean"],
                         v["min"], v["max"], v["slope"]))


if __name__ == "__main__":
    main()
