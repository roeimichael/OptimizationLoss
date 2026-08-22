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
import re

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
    # Only decidable on the WIDE schema. The narrow (dual) schema carries no
    # Limit_Class columns at all, so "no finite limit" was true for every dual
    # arm and `fioretto`/`hounie`/`alm` were reported as POST-HOC -- arms that
    # run a full constraint phase, labelled as running none. The narrow schema
    # answers the question a different way, through `all_satisfied`.
    r["posthoc"] = r["wide"] and not any(
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

    # --- total excess, the only count the narrow schema writes ---
    r["excess"] = None
    exc = _col(df, "total_excess")
    if exc:
        e = pd.to_numeric(df[exc], errors="coerce").dropna()
        if len(e) >= 3:
            y = e.to_numpy(dtype=float)
            r["excess"] = {"first": float(y[0]), "last": float(y[-1]),
                           "mean": float(y.mean()),
                           "slope": float(np.polyfit(
                               np.arange(len(y), dtype=float), y, 1)[0])}

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

    # --- the LOCAL scope, per (group, class). This was unread entirely, on a
    #     protocol whose local caps are per-GROUP ceilings: with 2 capped
    #     classes and 3 groups the loss carries 2 x (1 global + 3 local) = 8
    #     terms and only the 2 global ones were being watched.
    #
    #     It is the scope where FRAMEWORK 2(a2) says the damage is: the
    #     penalty's gradient is non-monotone in the violation, so the
    #     WORST-violating group is starved by a milder one at 167:1, and the
    #     terms compete for one unit-norm clip. That is invisible in any
    #     global count and invisible in `total_excess`, which sums it away. ---
    r["group_counts"] = {}
    for col in df.columns:
        m = re.match(r"^Group(\d+)_Limit_Class(\d+)$", str(col))
        if not m:
            continue
        gid, c = int(m.group(1)), int(m.group(2))
        lim = pd.to_numeric(df[col], errors="coerce").dropna()
        if lim.empty or float(lim.iloc[-1]) >= 1e9:
            continue                              # uncapped in this group
        hc = "Group%d_Hard_Class%d" % (gid, c)
        if hc not in df.columns:
            continue
        h = pd.to_numeric(df[hc], errors="coerce").dropna()
        if r["wide"] and len(h) > 1:
            h = h.iloc[1:]                        # drop the warm-up row
        if len(h) < 3:
            continue
        K = float(lim.iloc[-1])
        y = h.to_numpy(dtype=float)
        r["group_counts"][(gid, c)] = {
            "K": K, "first": float(y[0]), "last": float(y[-1]),
            "mean": float(y.mean()), "over": float(np.maximum(y - K, 0).mean()),
            "slope": float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])}
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
        # EXIT NON-ZERO. Printing the reason and returning 0 made an empty or
        # wrong root indistinguishable from a clean campaign to anything that
        # chains on this command -- and `main()` is called bare, so a returned
        # code would have been discarded anyway. Every other campaign tool here
        # already exits 1; this one did not.
        raise SystemExit("no readable training logs under %s" % args.root)

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
        if not cs:
            ex = [r["excess"] for r in rs if r.get("excess")]
            if ex:
                cs = ["total excess %.0f->%.0f (slope %+.2f/ep)"
                      % (np.mean([e["first"] for e in ex]),
                         np.mean([e["last"] for e in ex]),
                         np.mean([e["slope"] for e in ex]))]
        print("  %-14s %5d %8s %14s   %s"
              % (arm, len(rs), "%.4f" % np.mean(accs) if accs else "n/a",
                 satr, "; ".join(cs) if cs else "n/a (schema)"))

    # --- the LOCAL scope, per arm. Reported separately from the global one
    #     because they are different constraints with different budgets, and
    #     the protocol's local caps are per-GROUP ceilings. ---
    keys = sorted({k for r in runs for k in r.get("group_counts", {})})
    if keys:
        print("")
        print("LOCAL SCOPE -- per (group, capped class), which no global count "
              "and no `total_excess` can show")
        print("  %-14s %-14s %6s %14s %10s %9s"
              % ("arm", "group/class", "K", "count first->last",
                 "mean over", "slope/ep"))
        for arm in sorted({r["arm"] or "?" for r in runs if r.get("group_counts")}):
            rs = [r for r in runs if (r["arm"] or "?") == arm and r.get("group_counts")]
            for gid, c in keys:
                v = [r["group_counts"][(gid, c)] for r in rs
                     if (gid, c) in r["group_counts"]]
                if not v:
                    continue
                print("  %-14s g%d / class%-4d %6.0f %6.0f -> %-6.0f %10.1f %+9.2f"
                      % (arm, gid, c, v[0]["K"],
                         np.mean([x["first"] for x in v]),
                         np.mean([x["last"] for x in v]),
                         np.mean([x["over"] for x in v]),
                         np.mean([x["slope"] for x in v])))

        # FRAMEWORK 2(a2): the penalty's gradient is non-monotone in the
        # violation, so the WORST-violating group is starved by a milder one
        # (167:1 measured) and the terms compete for one unit-norm clip. The
        # signature is the deepest violator improving LEAST. Checked here
        # rather than left to a reader, because it is invisible in every
        # aggregate this project prints.
        for arm in sorted({r["arm"] or "?" for r in runs if r.get("group_counts")}):
            rs = [r for r in runs if (r["arm"] or "?") == arm and r.get("group_counts")]
            per = {}
            for k in keys:
                v = [r["group_counts"][k] for r in rs if k in r["group_counts"]]
                if v:
                    per[k] = (np.mean([x["over"] for x in v]),
                              np.mean([x["slope"] for x in v]))
            live = {k: v for k, v in per.items() if v[0] > 0}
            if len(live) < 2:
                continue
            worst = max(live, key=lambda k: live[k][0])
            mildest = min(live, key=lambda k: live[k][0])
            if live[worst][1] >= live[mildest][1]:
                print("  !! %s: the WORST-violating scope g%d/class%d "
                      "(over by %.0f) is falling SLOWER than the mildest "
                      "g%d/class%d (over by %.0f): %+.2f vs %+.2f per epoch."
                      % (arm, worst[0], worst[1], live[worst][0],
                         mildest[0], mildest[1], live[mildest][0],
                         live[worst][1], live[mildest][1]))
                print("     That is FRAMEWORK 2(a2)'s starvation signature -- "
                      "the penalty's gradient is non-monotone in the violation "
                      "and the scopes compete for one unit-norm clip.")

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
