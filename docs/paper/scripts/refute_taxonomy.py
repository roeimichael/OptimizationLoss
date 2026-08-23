"""Independent re-derivation of the taxonomy-reconstruction claim.

Does NOT import taxonomy.py. Parses every raw file itself.

Claim under test:
  A) TraLO's sparse log yields a COMPLETE per-epoch satisfaction trace (exact).
     evidence: 83 runs w/ 'Satisfaction Epoch', first_sat agrees 83/83;
               61 runs w/o, n_sat==0 61/61; sat<->zero-excess violations 0.
  B) epoch-1 excess identical across all 3 methods in 48/48 (cell,seed).

    python paper/scripts/refute_taxonomy.py
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = "results/headroom/headroom_b30_lrc0.0001_noceskip"
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 400)


def read_log(path):
    """Robust: drop repeated header rows, report how many there were."""
    with open(path) as f:
        lines = f.read().splitlines()
    hdr = lines[0]
    n_repeat = sum(1 for L in lines[1:] if L == hdr)
    t = pd.read_csv(path, dtype=str, low_memory=False)
    key = t.columns[0]
    t = t[t[key] != key]
    return t, n_repeat, len(lines) - 1


def numv(s):
    return pd.to_numeric(s, errors="coerce")


rows = []
detail = {}
for cfg_path in sorted(glob.glob(ROOT + "/**/config.json", recursive=True)):
    cfg = json.load(open(cfg_path))
    m = cfg.get("methodology")
    if m not in TRAINED:
        continue
    d = os.path.dirname(cfg_path)
    hp = cfg["hyperparams"]
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)

    lg = os.path.join(d, "training_log.csv")
    if not os.path.exists(lg):
        rows.append({"path": d, "method": m, "MISSING_LOG": True})
        continue
    t, n_repeat, n_raw_lines = read_log(lg)

    if m == "tralo":
        ep = numv(t["Epoch"])
        t2 = t[ep.notna()]
        e = (ep[ep.notna()] - 1).astype(int).to_numpy()          # internal epoch
        gsat = numv(t2.get("Global_Satisfied"))
        lsat = numv(t2.get("Local_Satisfied"))
        sat = ((gsat == 1) & (lsat == 1)).astype(int).to_numpy()
        hard = numv(t2["Hard_Class%d" % cls])
        lim = numv(t2["Limit_Class%d" % cls])
        exc = np.maximum(0.0, hard - lim).fillna(0.0)
        ngrp = 0
        for c in t2.columns:
            suf = "_Hard_Class%d" % cls
            if c.startswith("Group") and c.endswith(suf):
                g = c[len("Group"):-len(suf)]
                lc = "Group%s_Limit_Class%d" % (g, cls)
                if lc in t2.columns:
                    gl, gh = numv(t2[lc]), numv(t2[c])
                    add = np.maximum(0.0, gh - gl)
                    add = add.where(np.isfinite(gl), 0.0).fillna(0.0)
                    exc = exc + add
                    ngrp += 1
        exc = exc.to_numpy(float)
    else:
        ep = numv(t["epoch"])
        t2 = t[ep.notna()]
        e = (ep[ep.notna()]).astype(int).to_numpy() + 1          # -> common index
        sat = numv(t2["all_satisfied"]).fillna(0).astype(int).to_numpy()
        exc = numv(t2["total_excess"]).to_numpy(float)
        ngrp = None

    order = np.argsort(e)
    e, sat, exc = e[order], sat[order], exc[order]
    max_e = int(e.max())

    # --- evaluation_metrics.csv, raw ---
    ev = {}
    evp = os.path.join(d, "evaluation_metrics.csv")
    ev_keys = []
    if os.path.exists(evp):
        tv = pd.read_csv(evp)
        ev = dict(zip(tv["Metric"].astype(str), tv["Value"]))
        ev_keys = list(tv["Metric"].astype(str))

    def fnum(k):
        v = ev.get(k)
        try:
            x = float(v)
            return None if np.isnan(x) else x
        except (TypeError, ValueError):
            return None

    satmap = dict(zip(e.tolist(), sat.tolist()))
    NCE = int(hp.get("constraint_epochs", cfg.get("constraint_epochs", 29)))
    inferred = False
    if m == "tralo":
        E = max_e
        tail4 = [satmap.get(max_e - k, 0) for k in range(4)]
        if max_e < NCE and all(tail4):
            E = max_e + 1
            satmap[E] = 1
            inferred = True
    else:
        E = max_e
    s = np.array([satmap.get(x, 0) for x in range(1, E + 1)], dtype=int)
    n_sat = int(s.sum())
    first_sat = int(np.argmax(s) + 1) if n_sat else None
    n_down = int(((s[:-1] == 1) & (s[1:] == 0)).sum()) if len(s) > 1 else 0
    held_tail = bool(len(s) >= 5 and s[-5:].all())

    # ---- WRITE-CONDITION FALSIFICATION TEST (tralo only) ----
    sched = set([int(hp.get("warmup_epochs", 1))]) | {
        x for x in range(1, NCE + 1) if (x + 1) % 5 == 0}
    off_sched_unsat = off_sched_sat = missing_sched = 0
    if m == "tralo":
        for ei, si in zip(e, sat):
            if ei not in sched:
                if si == 1:
                    off_sched_sat += 1
                else:
                    off_sched_unsat += 1
        missing_sched = len([x for x in sched if x <= max_e and x not in satmap])

    rows.append({
        "path": d, "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
        "cap": cfg["constraint_tag"], "method": m, "seed": hp.get("seed"),
        "warmup": hp.get("warmup_epochs"), "cepochs": hp.get("constraint_epochs"),
        "sct": hp.get("stable_count_threshold", "ABSENT(5)"),
        "ce_skip_cfg": hp.get("enable_ce_skip", "ABSENT(True)"),
        "n_rows": len(e), "n_repeat_hdr": n_repeat, "max_e": max_e,
        "first_e": int(e.min()), "E": E, "inferred_tail": inferred,
        "n_sat": n_sat, "first_sat": first_sat, "n_down": n_down,
        "held_tail": held_tail, "n_local_groups": ngrp,
        "ex_first": exc[0], "ex_min": float(np.nanmin(exc)),
        "sat_epoch_csv": fnum("Satisfaction Epoch"),
        "sat_epoch_present": "Satisfaction Epoch" in ev_keys,
        "sat_epoch_raw": ev.get("Satisfaction Epoch", "<KEY ABSENT>"),
        "min_total_excess": fnum("Min Total Excess"),
        "min_excess_epoch": fnum("Min Excess Epoch"),
        "best_sat_epoch": fnum("Best Sat Epoch"),
        "restored_from": fnum("Restored From Epoch"),
        "off_sched_unsat": off_sched_unsat, "off_sched_sat": off_sched_sat,
        "missing_sched": missing_sched,
        "obs_frac": len(e) / float(NCE),
    })
    detail[d] = (e, sat, exc)

D = pd.DataFrame(rows)
D.to_csv("paper/scripts/out_refute_taxonomy.csv", index=False)
print("runs parsed: %d   (configs found: %d)"
      % (len(D), len(glob.glob(ROOT + "/**/config.json", recursive=True))))
print(D.groupby("method").size())

print("\n" + "=" * 100)
print("CONFIG SANITY")
print("=" * 100)
print(D.groupby("method").agg(warmup=("warmup", lambda x: sorted(set(x))),
                              cepochs=("cepochs", lambda x: sorted(set(x))),
                              sct=("sct", lambda x: sorted(set(map(str, x)))),
                              ce_skip=("ce_skip_cfg", lambda x: sorted(set(map(str, x))))))
print("  repeated header lines anywhere: %d runs" % int((D.n_repeat_hdr > 0).sum()))
print("  tralo first logged e values: %s"
      % sorted(set(D[D.method == "tralo"].first_e)))
print("  dual  first logged e values: %s"
      % sorted(set(D[D.method != "tralo"].first_e)))

print("\n" + "=" * 100)
print("A1. WRITE-CONDITION FALSIFICATION (tralo): rows off the {warmup} u {(e+1)%5==0}")
print("    schedule that are NOT satisfied would break 'absent => unsatisfied'.")
print("=" * 100)
tr = D[D.method == "tralo"]
print("  tralo runs: %d" % len(tr))
print("  off-schedule rows that are UNSATISFIED (must be 0): %d  (in %d runs)"
      % (int(tr.off_sched_unsat.sum()), int((tr.off_sched_unsat > 0).sum())))
print("  off-schedule rows that ARE satisfied            : %d" % int(tr.off_sched_sat.sum()))
print("  scheduled rows MISSING from the log (must be 0) : %d" % int(tr.missing_sched.sum()))

print("\n" + "=" * 100)
print("A2. HOW MUCH OF THE TRACE IS ACTUALLY OBSERVED vs INFERRED (tralo)")
print("=" * 100)
print(tr.groupby(["dataset", "cap"]).agg(
    runs=("path", "size"), rows_logged=("n_rows", "mean"),
    epochs_run=("E", "mean"), obs_frac=("obs_frac", "mean"),
    n_sat=("n_sat", "mean"), inferred_tail=("inferred_tail", "sum")))
print("  total tralo epoch-slots in the campaign : %d" % int(tr.E.sum()))
print("  of those, slots with an actual log row  : %d" % int(tr.n_rows.sum()))
print("  slots whose value is IMPUTED by the rule: %d (%.1f%%)"
      % (int(tr.E.sum() - tr.n_rows.sum()),
         100.0 * (tr.E.sum() - tr.n_rows.sum()) / tr.E.sum()))

print("\n" + "=" * 100)
print("A3. REPRODUCE 83 / 61 / 83 / 61 / 0")
print("=" * 100)
print("  'Satisfaction Epoch' key present in evaluation_metrics.csv:")
print(pd.crosstab(D.method, D.sat_epoch_present))
have = D.dropna(subset=["sat_epoch_csv"])
none = D[D.sat_epoch_csv.isna()]
print("\n  runs WITH a numeric Satisfaction Epoch : %d   (claim: 83)" % len(have))
print("  runs WITHOUT                           : %d   (claim: 61)" % len(none))
print(pd.crosstab(D.method, D.sat_epoch_csv.notna()))
norm = np.where(have.method == "tralo", have.sat_epoch_csv - 1, have.sat_epoch_csv)
agree = int((norm == have.first_sat).sum())
print("  first_sat agrees with recorded epoch   : %d of %d   (claim: 83/83)"
      % (agree, len(have)))
print("  of the 'no recorded epoch' runs, n_sat==0: %d of %d   (claim: 61/61)"
      % (int((none.n_sat == 0).sum()), len(none)))
print("  (n_sat>0 & ex_min>0) inconsistencies    : %d   (claim: 0)"
      % int(((D.n_sat > 0) & (D.ex_min > 0)).sum()))

print("\n" + "=" * 100)
print("A4. ARE THOSE CHECKS INDEPENDENT?  Cross-check against a DIFFERENT summary")
print("    field: 'Min Total Excess'==0  <=>  at least one satisfied epoch.")
print("=" * 100)
mm = D.dropna(subset=["min_total_excess"])
print("  runs with Min Total Excess recorded: %d" % len(mm))
if len(mm):
    a = (mm.min_total_excess == 0)
    b = (mm.n_sat > 0)
    print("  Min Total Excess==0 but reconstructed n_sat==0 : %d" % int((a & ~b).sum()))
    print("  Min Total Excess >0 but reconstructed n_sat >0 : %d" % int((~a & b).sum()))
    bad = mm[(a & ~b) | (~a & b)]
    if len(bad):
        print(bad[["dataset", "model", "cap", "method", "seed", "n_sat",
                   "min_total_excess", "min_excess_epoch", "E"]].to_string(index=False))
print("\n  Min Excess Epoch vs reconstructed E (epoch beyond the reconstructed budget?)")
mе = D.dropna(subset=["min_excess_epoch"])
over = mе[mе.min_excess_epoch > mе.E]
print("  runs where Min Excess Epoch > reconstructed epochs_run: %d" % len(over))
if len(over):
    print(over[["dataset", "model", "cap", "method", "seed", "E", "max_e",
                "min_excess_epoch", "inferred_tail", "n_sat"]].to_string(index=False))
bs = D.dropna(subset=["best_sat_epoch"])
if len(bs):
    print("\n  Best Sat Epoch present for %d runs; > reconstructed E: %d"
          % (len(bs), int((bs.best_sat_epoch > bs.E).sum())))
    print("  Best Sat Epoch recorded but reconstructed says that epoch unsatisfied:")
    nbad = 0
    for _, r in bs.iterrows():
        e_, s_, _x = detail[r.path]
        idx = int(r.best_sat_epoch) - (1 if r.method == "tralo" else 0)
        mp = dict(zip(e_.tolist(), s_.tolist()))
        if mp.get(idx, None) != 1 and not (r.inferred_tail and idx == r.E):
            nbad += 1
    print("    -> %d" % nbad)

print("\n" + "=" * 100)
print("B. EPOCH-1 EXCESS IDENTITY ACROSS METHODS  (claim: 48/48, max disagreement 0)")
print("=" * 100)
piv = D.pivot_table(index=CELL + ["seed"], columns="method", values="ex_first")
piv["n_methods"] = piv[TRAINED].notna().sum(axis=1)
piv["spread"] = piv[TRAINED].max(axis=1) - piv[TRAINED].min(axis=1)
print("  (cell,seed) rows: %d ; with all 3 methods present: %d"
      % (len(piv), int((piv.n_methods == 3).sum())))
print("  spread==0 rows: %d ; max spread: %s"
      % (int((piv.spread == 0).sum()), piv.spread.max()))
print(piv.head(12).to_string())
print("\n  per-CELL counting (never averaged): seeds with spread==0 out of 4")
per = piv.reset_index().groupby(CELL).agg(
    seeds=("spread", "size"), identical=("spread", lambda x: int((x == 0).sum())),
    max_spread=("spread", "max"))
print(per.to_string())
print("\n  worked example dermmnist/RegNetY400MF/L30_G30/seed 1:")
sub = piv.reset_index()
sub = sub[(sub.dataset == "dermmnist") & (sub.model == "RegNetY400MF")
          & (sub.cap == "L30_G30") & (sub.seed == 1)]
print(sub.to_string(index=False))

print("\n" + "=" * 100)
print("B2. does the identity extend past epoch 1?  (methods diverge -> it cannot)")
print("=" * 100)
common = {}
for k, (e_, s_, x_) in detail.items():
    r = D[D.path == k].iloc[0]
    common[(r.dataset, r.model, r.cap, r.seed, r.method)] = dict(zip(e_.tolist(), x_.tolist()))
keys = sorted({k[:4] for k in common})
agree_by_e = defaultdict(lambda: [0, 0])
for k in keys:
    for ee in range(1, 30):
        vals = []
        for m in TRAINED:
            v = common.get(k + (m,), {}).get(ee)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if len(vals) == 3:
            agree_by_e[ee][1] += 1
            if max(vals) - min(vals) == 0:
                agree_by_e[ee][0] += 1
print("  epoch : identical / comparable-triples")
for ee in sorted(agree_by_e):
    a, b = agree_by_e[ee]
    print("   e=%2d : %3d / %3d" % (ee, a, b))
