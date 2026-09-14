"""Inspect campaign training logs for nonfinite updates, dose and convergence."""

import argparse
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.constants import UNLIMITED

SATURATED_ACC = 0.93
FLAT_GAIN = 0.005
COLLAPSE_DROP = 0.02
# Columns whose value is a declared cap, where UNLIMITED (+inf) means "no cap".
# Matches the two families `logging.py` emits: the global `Limit_Class<c>` and the
# per-group `Group<g>_Limit_Class<c>`.
DECLARED_LIMIT = re.compile(r"^(Group\d+_)?Limit_Class\d+$")


def _col(df, *names):
    return next((n for n in names if n in df.columns), None)


def read_run(d):
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
    r = {
        "dir": d,
        "rows": len(df),
        "wide": "Train_Acc" in df.columns,
        "status": cfg.get("status"),
        "arm": cfg.get("arm"),
        "steps_applied": (cfg.get("results") or {}).get("constraint_steps_applied"),
    }
    r["collapse"] = None
    r["acc_final"] = None
    r["acc_first"] = None
    r["acc_gain"] = None
    if acc:
        a = pd.to_numeric(df[acc], errors="coerce").dropna()
        if len(a) >= 2 and float(a.iloc[-1]) < float(a.iloc[-2]) - COLLAPSE_DROP:
            r["collapse"] = (float(a.iloc[-2]), float(a.iloc[-1]))
        if len(a):
            r["acc_final"] = float(a.iloc[-1])
            r["acc_first"] = float(a.iloc[0])
            r["acc_gain"] = float(a.iloc[-1]) - float(a.iloc[0])
    r["nonfinite"] = {}
    scan = df[pd.to_numeric(df[ep], errors="coerce") >= 2] if ep and r["wide"] else df
    for c in scan.select_dtypes(include=[np.number]).columns:
        v = scan[c].to_numpy(dtype=float)
        w = df[c].to_numpy(dtype=float)
        # Empty warm-up constraint fields are not observations, but infinity
        # in an observed field is invalid at any epoch, including warm-up.
        infinite = np.isinf(w)
        if DECLARED_LIMIT.match(c):
            # A limit column is a DECLARATION, not an observation, and +inf is its
            # documented sentinel: `src/training/logging.py:129` writes the literal
            # 'inf' for a class whose cap is >= UNLIMITED, and lines 78/119/145
            # below read it back as exactly that. Counting the sentinel as a
            # non-finite UPDATE turned the firstrun gate RED on every campaign that
            # leaves any class uncapped -- on iwildcam, 6 of the 8. A -inf here is
            # not the sentinel and still fires, as does a cap that goes NaN after
            # being declared.
            infinite = infinite & ~np.isposinf(w)
        invalid = int(infinite.sum())
        if np.isfinite(v).any():
            invalid += int(np.isnan(v).sum())
        if invalid:
            r["nonfinite"][c] = invalid
    r["posthoc"] = r["wide"] and (
        not any(
            (
                c.startswith("Limit_Class")
                and (pd.to_numeric(df[c], errors="coerce").dropna() < UNLIMITED).any()
                for c in df.columns
            )
        )
    )
    r["sat"] = None
    if sat and ep and (not r["posthoc"]):
        con = df[pd.to_numeric(df[ep], errors="coerce") >= 2] if r["wide"] else df
        v = pd.to_numeric(con[sat], errors="coerce").dropna()
        if len(v):
            r["sat"] = (int(v.sum()), len(v))
    r["ce"] = None
    if ce:
        c = pd.to_numeric(df[ce], errors="coerce").dropna()
        if len(c) >= 2:
            r["ce"] = (float(c.iloc[0]), float(c.iloc[-1]))
    r["gn_pinned"] = None
    if gn:
        g = pd.to_numeric(df[gn], errors="coerce").dropna()
        g = g[g > 0]
        if len(g) >= 3 and np.isfinite(g).all():
            med = float(g.median())
            r["gn_pinned"] = (med, float(g.std()) < 1e-06 * max(1.0, med))
    r["excess"] = None
    exc = _col(df, "total_excess")
    if exc:
        e = pd.to_numeric(df[exc], errors="coerce").dropna()
        if len(e) >= 3:
            y = e.to_numpy(dtype=float)
            r["excess"] = {
                "first": float(y[0]),
                "last": float(y[-1]),
                "mean": float(y.mean()),
                "slope": float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]),
            }
    r["counts"] = {}
    for c in range(20):
        (hc, lc) = ("Hard_Class%d" % c, "Limit_Class%d" % c)
        if hc not in df.columns or lc not in df.columns:
            continue
        lim = pd.to_numeric(df[lc], errors="coerce").dropna()
        if lim.empty or float(lim.iloc[-1]) >= UNLIMITED:
            continue
        h = pd.to_numeric(df[hc], errors="coerce").dropna()
        if r["wide"] and len(h) > 1:
            h = h.iloc[1:]
        if len(h) < 3:
            continue
        K = float(lim.iloc[-1])
        y = h.to_numpy(dtype=float)
        slope = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
        r["counts"][c] = {
            "K": K,
            "first": float(y[0]),
            "last": float(y[-1]),
            "mean": float(y.mean()),
            "min": float(y.min()),
            "max": float(y.max()),
            "slope": slope,
        }
    r["group_counts"] = {}
    for col in df.columns:
        m = re.match("^Group(\\d+)_Limit_Class(\\d+)$", str(col))
        if not m:
            continue
        (gid, c) = (int(m.group(1)), int(m.group(2)))
        lim = pd.to_numeric(df[col], errors="coerce").dropna()
        if lim.empty or float(lim.iloc[-1]) >= UNLIMITED:
            continue
        hc = "Group%d_Hard_Class%d" % (gid, c)
        if hc not in df.columns:
            continue
        h = pd.to_numeric(df[hc], errors="coerce").dropna()
        if r["wide"] and len(h) > 1:
            h = h.iloc[1:]
        if len(h) < 3:
            continue
        K = float(lim.iloc[-1])
        y = h.to_numpy(dtype=float)
        r["group_counts"][gid, c] = {
            "K": K,
            "first": float(y[0]),
            "last": float(y[-1]),
            "mean": float(y.mean()),
            "over": float(np.maximum(y - K, 0).mean()),
            "slope": float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]),
        }
    return r


def _saturation_readout(runs):
    have = [
        r
        for r in runs
        if r.get("acc_first") is not None and r.get("acc_gain") is not None
    ]
    if not have:
        return
    first = float(np.median([r["acc_first"] for r in have]))
    gain = float(np.median([r["acc_gain"] for r in have]))
    print("")
    print("LOGGED ACCURACY OBSERVATIONS  (%d run(s))" % len(have))
    print("   first logged accuracy          %6.3f  (median)" % first)
    print("   last minus first logged value  %+6.3f  (median)" % gain)
    print(
        "   Warm-up boundary and accuracy-state comparability: unknown in untagged CSVs."
    )
    if first >= SATURATED_ACC and abs(gain) <= FLAT_GAIN:
        print(
            "   [!] HIGH FLAT LOGGED ACCURACY; this alone does not establish saturation."
        )
        print("       Inspect development allocation cuts directly.")
        print("       RUN `python -m scripts.headroom <campaign>` BEFORE reading a")
        print("       contrast: a tie here may be the saturation, not the method.")
    else:
        print(
            "   -> not a high-flat logged signature (acc >= %.2f AND |gain| <= %.3f)"
            % (SATURATED_ACC, FLAT_GAIN)
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--full", action="store_true", help="one block per run")
    args = ap.parse_args()
    paths = sorted(
        glob.glob(os.path.join(args.root, "**", "config.json"), recursive=True)
    )
    runs = [r for r in (read_run(os.path.dirname(p)) for p in paths) if r]
    if not runs:
        raise SystemExit("no readable training logs under %s" % args.root)
    print(
        "%d run(s) with a readable log  (%d wide schema, %d narrow)"
        % (
            len(runs),
            sum((r["wide"] for r in runs)),
            sum((not r["wide"] for r in runs)),
        )
    )
    bad = [r for r in runs if r["collapse"]]
    if bad:
        print(
            "\nTERMINAL COLLAPSE -- observed last-row accuracy drop; model-state attribution is unknown"
        )
        for r in bad:
            print(
                "   %-54s %.4f -> %.4f"
                % (
                    os.path.relpath(r["dir"], args.root),
                    r["collapse"][0],
                    r["collapse"][1],
                )
            )
    nf = [r for r in runs if r["nonfinite"]]
    if nf:
        print("\nNON-FINITE VALUES in observed numeric fields")
        for r in nf:
            worst = sorted(r["nonfinite"].items(), key=lambda kv: -kv[1])[:3]
            print(
                "   %-54s %s   of %d rows, %s constraint step(s) applied"
                % (
                    os.path.relpath(r["dir"], args.root),
                    ", ".join(("%s x%d" % kv for kv in worst)),
                    r["rows"],
                    "?" if r.get("steps_applied") is None else r["steps_applied"],
                )
            )
        print("   Counts alone cannot establish the cause or safety of these values.")
    _saturation_readout(runs)
    print("\nPER ARM")
    print(
        "  %-14s %5s %8s %14s   %s"
        % ("arm", "runs", "acc", "satisfied", "capped-class count vs K")
    )
    by_arm = {}
    for r in runs:
        by_arm.setdefault(r["arm"] or "?", []).append(r)
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        accs = [r["acc_final"] for r in rs if r["acc_final"] is not None]
        sats = [r["sat"] for r in rs if r["sat"]]
        if sats:
            satr = "%d/%d" % (sum((s[0] for s in sats)), sum((s[1] for s in sats)))
        elif all((r.get("posthoc") for r in rs)):
            satr = "n/a (posthoc)"
        else:
            satr = "n/a (schema)"
        cs = []
        for c in sorted({c for r in rs for c in r["counts"]}):
            v = [r["counts"][c] for r in rs if c in r["counts"]]
            cs.append(
                "c%d %.0f->%.0f (K=%.0f, slope %+.2f/ep)"
                % (
                    c,
                    np.mean([x["first"] for x in v]),
                    np.mean([x["last"] for x in v]),
                    v[0]["K"],
                    np.mean([x["slope"] for x in v]),
                )
            )
        if not cs:
            ex = [r["excess"] for r in rs if r.get("excess")]
            if ex:
                cs = [
                    "total excess %.0f->%.0f (slope %+.2f/ep)"
                    % (
                        np.mean([e["first"] for e in ex]),
                        np.mean([e["last"] for e in ex]),
                        np.mean([e["slope"] for e in ex]),
                    )
                ]
        print(
            "  %-14s %5d %8s %14s   %s"
            % (
                arm,
                len(rs),
                "%.4f" % np.mean(accs) if accs else "n/a",
                satr,
                "; ".join(cs) if cs else "n/a (schema)",
            )
        )
    keys = sorted({k for r in runs for k in r.get("group_counts", {})})
    if keys:
        print("")
        print(
            "LOCAL SCOPE -- per (group, capped class), which no global count and no `total_excess` can show"
        )
        print(
            "  %-14s %-14s %6s %14s %10s %9s"
            % ("arm", "group/class", "K", "count first->last", "mean over", "slope/ep")
        )
        for arm in sorted({r["arm"] or "?" for r in runs if r.get("group_counts")}):
            rs = [r for r in runs if (r["arm"] or "?") == arm and r.get("group_counts")]
            for gid, c in keys:
                v = [
                    r["group_counts"][gid, c]
                    for r in rs
                    if (gid, c) in r["group_counts"]
                ]
                if not v:
                    continue
                print(
                    "  %-14s g%d / class%-4d %6.0f %6.0f -> %-6.0f %10.1f %+9.2f"
                    % (
                        arm,
                        gid,
                        c,
                        v[0]["K"],
                        np.mean([x["first"] for x in v]),
                        np.mean([x["last"] for x in v]),
                        np.mean([x["over"] for x in v]),
                        np.mean([x["slope"] for x in v]),
                    )
                )
        for arm in sorted({r["arm"] or "?" for r in runs if r.get("group_counts")}):
            rs = [r for r in runs if (r["arm"] or "?") == arm and r.get("group_counts")]
            per = {}
            for k in keys:
                v = [r["group_counts"][k] for r in rs if k in r["group_counts"]]
                if v:
                    per[k] = (
                        np.mean([x["over"] for x in v]),
                        np.mean([x["slope"] for x in v]),
                    )
            live = {k: v for (k, v) in per.items() if v[0] > 0}
            if len(live) < 2:
                continue
            applied = [r.get("steps_applied") for r in rs]
            applied = [x for x in applied if isinstance(x, (int, float))]
            if applied and (not any((x > 0 for x in applied))):
                continue
            worst = max(live, key=lambda k: live[k][0])
            mildest = min(live, key=lambda k: live[k][0])
            if live[worst][1] >= live[mildest][1]:
                print(
                    "  !! %s: the WORST-violating scope g%d/class%d (over by %.0f) is falling SLOWER than the mildest g%d/class%d (over by %.0f): %+.2f vs %+.2f per epoch."
                    % (
                        arm,
                        worst[0],
                        worst[1],
                        live[worst][0],
                        mildest[0],
                        mildest[1],
                        live[mildest][0],
                        live[worst][1],
                        live[mildest][1],
                    )
                )
                print(
                    "     That is FRAMEWORK 2(a2)'s starvation signature -- the penalty's gradient is non-monotone in the violation and the scopes compete for one unit-norm clip."
                )
    pinned = [r for r in runs if r["gn_pinned"] and r["gn_pinned"][1]]
    if pinned:
        print(
            "\n%d run(s) have a gradient norm pinned to one value on every epoch."
            % len(pinned)
        )
        print(
            "   This is the raw pre-transform norm. Clipping and applied displacement are unknown."
        )
    print(
        "Task-update dose and state-tagged local/application records: unknown in this CSV audit."
    )
    if args.full:
        for r in runs:
            print("\n%s" % os.path.relpath(r["dir"], args.root))
            print(
                "   rows=%d status=%s acc_final=%s sat=%s"
                % (r["rows"], r["status"], r["acc_final"], r["sat"])
            )
            for c, v in sorted(r["counts"].items()):
                print(
                    "   class %d: K=%.0f  %.0f -> %.0f  mean %.0f  range %.0f-%.0f  slope %+.2f/ep"
                    % (
                        c,
                        v["K"],
                        v["first"],
                        v["last"],
                        v["mean"],
                        v["min"],
                        v["max"],
                        v["slope"],
                    )
                )

    if bad or nf:
        print(
            "FAIL: observed terminal accuracy collapse or non-finite numeric log values; investigate before proceeding."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
