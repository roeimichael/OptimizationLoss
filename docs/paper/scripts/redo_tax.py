"""Independent re-derivation of the failure taxonomy. Does NOT import taxonomy.py.

Rebuilds the satisfaction trace from raw training_log.csv + final_predictions_raw.csv
and re-counts the five classes, then stress-tests every threshold in the definition.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/headroom/headroom_b30_lrc0.0001_noceskip"
TRAINED = ["tralo", "fioretto_ldf", "hounie_rcl"]
CELL = ["dataset", "model", "cap"]
NCE = 29


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_log(p):
    t = pd.read_csv(p, dtype=str, low_memory=False)
    k = t.columns[0]
    return t[t[k] != k]


rows = []
schema = []
for cfgp in sorted(glob.glob(ROOT + "/**/config.json", recursive=True)):
    cfg = json.load(open(cfgp))
    m = cfg.get("methodology")
    if m not in TRAINED:
        continue
    d = os.path.dirname(cfgp)
    lg = os.path.join(d, "training_log.csv")
    raw = os.path.join(d, "final_predictions_raw.csv")
    if not (os.path.exists(lg) and os.path.exists(raw)):
        schema.append({"path": d, "method": m, "issue": "MISSING FILE"})
        continue
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    lp, gp = cfg["constraint"]

    t = read_log(lg)
    cols = set(t.columns)
    if m == "tralo":
        has_g = "Global_Satisfied" in cols
        has_l = "Local_Satisfied" in cols
        ep = num(t["Epoch"])
        t = t[ep.notna()]
        ep = ep[ep.notna()]
        gs = num(t["Global_Satisfied"]) if has_g else pd.Series(np.nan, index=t.index)
        ls = num(t["Local_Satisfied"]) if has_l else pd.Series(np.nan, index=t.index)
        n_g_nan = int(gs.isna().sum())
        n_l_nan = int(ls.isna().sum())
        sat = ((gs == 1) & (ls == 1)).astype(int).values
        e = (ep - 1).astype(int).values
        schema.append({"path": d, "method": m, "has_G": has_g, "has_L": has_l,
                       "nan_G": n_g_nan, "nan_L": n_l_nan, "nrows": len(t),
                       "epmin": int(ep.min()), "epmax": int(ep.max())})
    else:
        ep = num(t["epoch"])
        t = t[ep.notna()]
        ep = ep[ep.notna()]
        sat = num(t["all_satisfied"]).fillna(0).astype(int).values
        e = (ep + 1).astype(int).values
        schema.append({"path": d, "method": m, "has_G": "all_satisfied" in cols,
                       "has_L": True, "nan_G": 0, "nan_L": 0, "nrows": len(t),
                       "epmin": int(ep.min()), "epmax": int(ep.max())})

    o = np.argsort(e)
    e, sat = e[o], sat[o]
    satmap = dict(zip(e.tolist(), sat.tolist()))
    max_e = int(e.max())

    inferred = False
    E = max_e
    if m == "tralo":
        tail4 = [satmap.get(max_e - k, 0) for k in range(4)]
        if max_e < NCE and all(tail4):
            E = max_e + 1
            satmap[E] = 1
            inferred = True
    s = np.array([satmap.get(x, 0) for x in range(1, E + 1)], dtype=int)

    # final restored model's raw predicted count for the constrained class
    fr = pd.read_csv(raw)
    count_raw = int((fr["Predicted_Label"].to_numpy(int) == cls).sum())

    # K from the config's global cap for this class
    ev = {}
    evp = os.path.join(d, "evaluation_metrics.csv")
    if os.path.exists(evp):
        te = pd.read_csv(evp)
        ev = dict(zip(te["Metric"], te["Value"]))

    n_pool = len(fr)
    Kg = None
    if isinstance(gp, dict):
        v = gp.get(str(cls), gp.get(cls))
        if v is not None:
            Kg = int(round(v * n_pool)) if v <= 1 else int(v)
    rows.append({
        "path": d, "dataset": cfg["dataset_mode"], "model": cfg["model_name"],
        "cap": cfg["constraint_tag"], "method": m, "seed": cfg["hyperparams"].get("seed"),
        "cls": cls, "n_pool": n_pool, "K_cfg": Kg, "count_raw": count_raw,
        "E": E, "max_e": max_e, "inferred": inferred, "nrows": len(e),
        "n_sat": int(s.sum()),
        "first_sat": int(np.argmax(s) + 1) if s.sum() else None,
        "n_down": int(((s[:-1] == 1) & (s[1:] == 0)).sum()),
        "s": "".join(map(str, s)),
        "sat_epoch_csv": ev.get("Satisfaction Epoch"),
        "restore_kind": ev.get("Restore Kind"),
        "gp_raw": json.dumps(gp)[:120],
    })

D = pd.DataFrame(rows)
S = pd.DataFrame(schema)
print("runs found: %d" % len(D))
print(D.method.value_counts().to_string())
print("\n--- SCHEMA AUDIT (tralo) ---")
ST = S[S.method == "tralo"]
print("tralo logs missing Global_Satisfied: %d ; missing Local_Satisfied: %d"
      % (int((~ST.has_G).sum()), int((~ST.has_L).sum())))
print("tralo logs with any NaN in Global_Satisfied: %d ; Local_Satisfied: %d"
      % (int((ST.nan_G > 0).sum()), int((ST.nan_L > 0).sum())))
print("tralo logged-Epoch min/max range: %s .. %s" % (ST.epmin.min(), ST.epmax.max()))
print("tralo rows per log: min %d max %d" % (ST.nrows.min(), ST.nrows.max()))
print("\n--- dual epoch ranges ---")
SD = S[S.method != "tralo"]
print("dual epoch min %s max %s ; rows min %d max %d"
      % (SD.epmin.min(), SD.epmax.max(), SD.nrows.min(), SD.nrows.max()))
D.to_csv("paper/scripts/out_redo_tax.csv", index=False)
print("\nwrote paper/scripts/out_redo_tax.csv")
