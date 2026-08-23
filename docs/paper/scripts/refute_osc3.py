"""Part 3: (i) are the two campaigns the same files? (ii) per-run ratio distribution,
(iii) sliding length-matched window control inside the CE-ON segment,
(iv) the runs where the gate PRECEDES the turning point."""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "paper/scripts")
import hounie_dyn as H  # noqa: E402

A = "results/headroom/headroom_b30_lrc0.0001_noceskip"
B = "results/headroom/headroom_b30_lrc0.0001"
C = "results/headroom/headroom_b30_lrc5e-05"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def seg(s):
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 3:
        return None
    d = np.diff(s)
    step = float(np.mean(np.abs(d)))
    drift = float((s[-1] - s[0]) / len(d))
    return step, drift, abs(drift) / step if step > 0 else np.nan, len(s)


def key(p):
    q = p.replace("\\", "/").split("/")
    return "/".join(q[-6:-1])  # model/dataset/cap/method/seed_n


def main():
    print("=" * 110)
    print("(i) ARE THE TWO CAMPAIGNS THE SAME RUNS?  md5 of every hounie_rcl training_log.csv")
    print("=" * 110)
    for other, lbl in [(B, "headroom_b30_lrc0.0001 (gate ON label)"),
                       (C, "headroom_b30_lrc5e-05 (different LR)")]:
        ma, mb = {}, {}
        for root, store in [(A, ma), (other, mb)]:
            for p in glob.glob(root + "/**/hounie_rcl/**/training_log.csv", recursive=True):
                store[key(p)] = md5(p)
        common = set(ma) & set(mb)
        same = sum(1 for k in common if ma[k] == mb[k])
        print("  %-42s  files A=%d B=%d  matched keys=%d  BYTE-IDENTICAL=%d"
              % (lbl, len(ma), len(mb), len(common), same))
    # same question for the tralo arm, where the flag DID differ
    ma, mb = {}, {}
    for root, store in [(A, ma), (B, mb)]:
        for p in glob.glob(root + "/**/tralo/**/training_log.csv", recursive=True):
            store[key(p)] = md5(p)
    common = set(ma) & set(mb)
    print("  %-42s  files A=%d B=%d  matched keys=%d  BYTE-IDENTICAL=%d"
          % ("tralo arm (enable_ce_skip DID differ)", len(ma), len(mb), len(common),
             sum(1 for k in common if ma[k] == mb[k])))

    # ---------------------------------------------------------------
    rows = []
    for tag, root in [("lrc1e-4", A), ("lrc5e-5", C)]:
        for p in sorted(glob.glob(root + "/**/config.json", recursive=True)):
            cfg = json.load(open(p))
            if cfg.get("methodology") != "hounie_rcl":
                continue
            r = H.load_run(p)
            if r is None:
                continue
            T = H.hounie_traj(r)
            log = r["log"]
            ep = log["epoch"].to_numpy(int)
            ce = log["ce_loss"].to_numpy(float)
            exc = log["total_excess"].to_numpy(float)
            soft = T["soft"]
            off = np.isnan(ce)
            if not off.any():
                gate = None
            else:
                gate = int(ep[off][0])
            d = dict(campaign=tag, dataset=r["dataset"], model=r["model"], cap=r["cap"],
                     seed=r["seed"], gate_ep=gate, n_off=int(off.sum()))
            v = seg(soft[off]) if off.any() else None
            d["ratio_off"], d["npts_off"] = (v[2], v[3]) if v else (np.nan, np.nan)
            v = seg(soft[~off])
            d["ratio_on"], d["npts_on"] = (v[2], v[3]) if v else (np.nan, np.nan)
            # turning point: last epoch at which the reconstructed count rose
            fin = np.isfinite(soft)
            idx = np.where(fin)[0]
            ds = np.diff(soft[fin])
            up = np.where(ds > 0)[0]
            d["last_rise_ep"] = int(ep[idx[up[-1] + 1]]) if len(up) else np.nan
            de = np.diff(exc)
            ue = np.where(de > 0)[0]
            d["last_rise_ep_exc"] = int(ep[ue[-1] + 1]) if len(ue) else np.nan
            # rise AFTER the gate, in counts
            if gate is not None:
                m = ep >= gate
                s = soft[m]
                s = s[np.isfinite(s)]
                d["rise_after_gate_soft"] = float(np.nanmax(s) - s[0]) if len(s) else np.nan
                e = exc[m]
                d["rise_after_gate_exc"] = float(np.max(e) - e[0])
                # (iii) sliding length-matched window inside the CE-ON segment
                L = int(min(off.sum(), (~off).sum()))
                son = soft[~off]
                son = son[np.isfinite(son)]
                best, vals = np.nan, []
                if L >= 3 and len(son) >= L:
                    for i in range(len(son) - L + 1):
                        v = seg(son[i:i + L])
                        if v:
                            vals.append(v[2])
                    if vals:
                        best, d["win_on_mean"] = max(vals), float(np.mean(vals))
                d["win_on_best"] = best
            rows.append(d)
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 230)
    pd.set_option("display.max_columns", 40)
    f = lambda x: "%.3f" % x  # noqa: E731

    print()
    print("=" * 110)
    print("(ii) PER-RUN ratio_off DISTRIBUTION -- is 0.981 a typical run or a mean over a")
    print("     mostly-exactly-1.000 population plus outliers?")
    print("=" * 110)
    g = d[d.ratio_off.notna()]
    print(g.groupby(["campaign", "dataset"]).agg(
        runs=("ratio_off", "size"), exactly_1=("ratio_off", lambda s: int((s > 0.9999).sum())),
        below_0_9=("ratio_off", lambda s: int((s < 0.9).sum())),
        min_ratio=("ratio_off", "min"), mean_ratio=("ratio_off", "mean"),
        min_npts=("npts_off", "min"), mean_npts=("npts_off", "mean")).to_string(float_format=f))

    print()
    print("=" * 110)
    print("(iii) LENGTH-MATCHED SLIDING WINDOW inside the CE-ON segment (same length as")
    print("      that run's CE-off segment).  best = most monotone window CE-on can produce.")
    print("=" * 110)
    print(g.groupby(["campaign", "dataset"]).agg(
        runs=("seed", "size"), ratio_off=("ratio_off", "mean"),
        win_on_best=("win_on_best", "mean"), win_on_mean=("win_on_mean", "mean"),
        n_windows_beating_off=("win_on_best", lambda s: np.nan)).iloc[:, :4]
        .to_string(float_format=f))

    print()
    print("=" * 110)
    print("(iv) DID THE TRAJECTORY TURN BEFORE OR AFTER THE GATE?")
    print("     turn = last epoch the reconstructed count rose.  lag = turn - gate.")
    print("=" * 110)
    gg = d[d.gate_ep.notna()].copy()
    gg["lag"] = gg.last_rise_ep - gg.gate_ep
    gg["lag_exc"] = gg.last_rise_ep_exc - gg.gate_ep
    print("  runs with gate: %d" % len(gg))
    print("  lag distribution (soft): " + str(gg.lag.value_counts().sort_index().to_dict()))
    print("  gate fired BEFORE the turn (lag>0): %d runs -- in these the count kept RISING "
          "after CE stopped" % int((gg.lag > 0).sum()))
    print()
    print(gg[gg.lag > 0][["campaign", "dataset", "model", "cap", "seed", "gate_ep",
                          "last_rise_ep", "lag", "rise_after_gate_soft",
                          "rise_after_gate_exc", "ratio_off", "npts_off"]]
          .to_string(index=False, float_format=f))
    print()
    print("  mean rise in the reconstructed count AFTER the gate, all gated runs: %.1f counts"
          % gg.rise_after_gate_soft.mean())
    print("  mean rise in the LOGGED hard excess AFTER the gate, all gated runs: %.1f counts"
          % gg.rise_after_gate_exc.mean())
    print("  gated runs whose logged hard excess rises at all after the gate: %d / %d"
          % (int((gg.rise_after_gate_exc > 0).sum()), len(gg)))
    d.to_csv("paper/scripts/out_refute_osc3.csv", index=False)
    print("\nwrote paper/scripts/out_refute_osc3.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
