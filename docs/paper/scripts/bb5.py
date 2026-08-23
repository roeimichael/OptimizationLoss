"""Backbone interaction, step 5: the mechanism table, and two integrity checks.

Check A: lrc0.0001 and lrc0.0001_noceskip returned bit-identical tissuemnist
cells. Either they are the same files, or the CE-saturation gate they differ in
never fires on tissuemnist. Distinguish by hashing the prediction files and by
reading the max Train_Acc actually reached.

Check B: the full per-epoch hard/soft count trajectory for TraLO, so
"the constraint does not bind on RegNet" is a trajectory and not one endpoint.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]
A_ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"
B_ROOT = "results/headroom/headroom_b30_lrc0.0001"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 16), b""):
            h.update(b)
    return h.hexdigest()


def index_runs(root):
    ix = {}
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        d = os.path.dirname(cp)
        c = json.load(open(cp))
        hp = c.get("hyperparams") or {}
        ix[(c.get("dataset_mode"), c.get("model_name"), c.get("constraint_tag"),
            c.get("methodology"), hp.get("seed"))] = (d, hp)
    return ix


def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def main():
    print("=" * 120)
    print("CHECK A: are lrc0.0001 and lrc0.0001_noceskip the same runs?")
    print("=" * 120)
    ia, ib = index_runs(A_ROOT), index_runs(B_ROOT)
    keys = sorted(set(ia) & set(ib))
    print("  shared run keys: %d" % len(keys))
    same_path = same_hash = diff_hash = 0
    per_ds = {}
    for k in keys:
        da, hpa = ia[k]
        db, hpb = ib[k]
        if os.path.realpath(da) == os.path.realpath(db):
            same_path += 1
            continue
        fa = os.path.join(da, "final_predictions_raw.csv")
        fb_ = os.path.join(db, "final_predictions_raw.csv")
        if not (os.path.exists(fa) and os.path.exists(fb_)):
            continue
        eq = md5(fa) == md5(fb_)
        same_hash += eq
        diff_hash += (not eq)
        per_ds.setdefault(k[0], [0, 0])[0 if eq else 1] += 1
    print("  identical filesystem path : %d" % same_path)
    print("  distinct path, SAME bytes : %d" % same_hash)
    print("  distinct path, DIFF bytes : %d" % diff_hash)
    print("  by dataset  [same, diff]  : %s" % per_ds)
    ka = list(ia)[0]
    print("  ce_skip flag  noceskip=%s   lrc0.0001=%s"
          % (ia[ka][1].get("enable_ce_skip"), ib[ka][1].get("enable_ce_skip")))

    print("\n  max Train_Acc reached by TraLO (the CE-saturation gate fires at")
    print("  >=0.995 for 2 checks, so it can only differ where this crosses):")
    rows = []
    for k, (d, hp) in ia.items():
        if k[3] != "tralo":
            continue
        p = os.path.join(d, "training_log.csv")
        if not os.path.exists(p):
            continue
        lg = pd.read_csv(p)
        if "Train_Acc" not in lg.columns:
            continue
        v = num(lg, "Train_Acc").dropna()
        rows.append({"dataset": k[0], "model": k[1], "cap": k[2], "seed": k[4],
                     "tracc_max": float(v.max()),
                     "ge995": int((v >= 0.995).sum())})
    ta = pd.DataFrame(rows).groupby(["dataset", "model"]).agg(
        n=("seed", "size"), tracc_max=("tracc_max", "mean"),
        tracc_max_min=("tracc_max", "min"),
        rows_ge_995=("ge995", "mean")).reset_index()
    print(ta.to_string(index=False, float_format=lambda x: "%.4f" % x))

    print("\n" + "=" * 120)
    print("CHECK B: TraLO hard-count trajectory vs the cap, every logged epoch.")
    print("Each cell shows the 4-seed mean hard count at each logged constraint")
    print("epoch, then min/K -- did the argmax count EVER reach the cap?")
    print("=" * 120)
    tr = []
    for k, (d, hp) in ia.items():
        if k[3] != "tralo":
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        c = (cfg.get("dataset_config") or {}).get("constrained_class")
        c = int(c[0] if isinstance(c, (list, tuple)) else c)
        lg = pd.read_csv(os.path.join(d, "training_log.csv"))
        hc, sc = "Hard_Class%d" % c, "Soft_Class%d" % c
        if hc not in lg.columns:
            continue
        t = pd.DataFrame({"Epoch": num(lg, "Epoch"), "hard": num(lg, hc),
                          "soft": num(lg, sc),
                          "lam": num(lg, "Lambda_Global"),
                          "sat": num(lg, "Global_Satisfied"),
                          "lg": num(lg, "L_Global")}).dropna(subset=["Epoch"])
        t["cep"] = t.Epoch - 1
        t = t[t.cep >= 1]
        t["dataset"], t["model"], t["cap"], t["seed"] = k[0], k[1], k[2], k[4]
        tr.append(t)
    T = pd.concat(tr, ignore_index=True)
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    Ks = fb[fb.campaign == "lrc0.0001_noceskip"].groupby(CELL)["K"].first()

    for (ds, mo, cap), g in T.groupby(CELL):
        K = float(Ks.loc[(ds, mo, cap)])
        m = g.groupby("cep")[["hard", "soft", "lam", "lg"]].mean()
        ep = [int(e) for e in m.index]
        hv = ["%.0f" % v for v in m.hard]
        print("  %-12s %-13s %-8s K=%3d" % (ds, mo, cap, K))
        print("      cep  : %s" % " ".join("%5d" % e for e in ep))
        print("      hard : %s" % " ".join("%5s" % v for v in hv))
        print("      soft : %s" % " ".join("%5.0f" % v for v in m.soft))
        print("      minK : hard_min/K = %.2f   ever_below_K = %s"
              % (m.hard.min() / K, "YES" if (g.hard.min() < K) else "NO"))

    print("\n" + "=" * 120)
    print("MECHANISM TABLE: did TraLO's own argmax count ever reach the cap?")
    print("(per seed, using the sparse log's minimum -- a lower bound on how")
    print("close it got, since unlogged epochs are not observed)")
    print("=" * 120)
    per = T.groupby(CELL + ["seed"]).agg(hard_min=("hard", "min"),
                                         hard_last=("hard", "last")).reset_index()
    per = per.merge(Ks.reset_index(), on=CELL)
    per["below"] = per.hard_min <= per.K
    s = per.groupby(CELL).agg(n=("seed", "size"),
                              hard_min=("hard_min", "mean"),
                              K=("K", "first"),
                              seeds_reaching_cap=("below", "sum")).reset_index()
    s["hard_min_over_K"] = s.hard_min / s.K
    print(s.sort_values(["model", "dataset", "cap"])
          .to_string(index=False, float_format=lambda x: "%.3f" % x))
    print("\n  cells where TraLO's argmax count reached the cap in >=1 seed:")
    for mo, g in s.groupby("model"):
        print("    %-14s %d of %d cells   (mean hard_min/K = %.2f)"
              % (mo, int((g.seeds_reaching_cap > 0).sum()), len(g),
                 g.hard_min_over_K.mean()))
    T.to_csv("paper/scripts/out_bb_traj_full.csv", index=False)
    print("\nwrote paper/scripts/out_bb_traj_full.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
