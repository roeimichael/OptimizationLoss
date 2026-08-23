"""CLEAN placebo: quality on the classes the constraint can never reach.

accOffRaw/macroOffRaw are contaminated -- suppressing the constrained class
mechanically stops it from stealing off-class items. So instead: delete the
constrained column entirely, argmax over the remaining classes, and evaluate
ONLY on samples whose TRUE label is not the constrained class. The count
constraint acts on P[:,cls] alone, so once that column is gone nothing about the
cap can help here. Any gap left is generic training quality.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
sys.path.insert(0, os.getcwd())
from src.utils.constants import UNLIMITED
from src.training.constraints import compute_global_constraints, compute_local_constraints

ROOT, OUT = sys.argv[1], sys.argv[2]

def topk_alloc(P, gids, K, loc, cls):
    order = np.argsort(-P[:, cls])
    room = {int(g): int(l[cls]) for g, l in loc.items()} if (gids is not None and loc) else {}
    chosen = np.zeros(len(P), dtype=bool); taken = 0
    for i in order:
        if taken >= K: break
        if room:
            g = int(gids[i])
            if room.get(g, 0) <= 0: continue
            room[g] -= 1
        chosen[i] = True; taken += 1
    o = P.copy(); o[:, cls] = -np.inf
    y = np.argmax(o, axis=1); y[chosen] = cls
    return y, taken

def ep_max(d):
    p = os.path.join(d, "training_log.csv")
    if not os.path.exists(p): return 0.0
    try: t = pd.read_csv(p, on_bad_lines="skip")
    except Exception: return np.nan
    c = "Epoch" if "Epoch" in t.columns else ("epoch" if "epoch" in t.columns else None)
    if c is None: return np.nan
    v = pd.to_numeric(t[c], errors="coerce").dropna()
    return float(v.max()) if len(v) else np.nan

rows = []
for cfg_path in glob.glob(ROOT + "/**/config.json", recursive=True):
    try: cfg = json.load(open(cfg_path))
    except Exception: continue
    d = os.path.dirname(cfg_path)
    raw, fin = os.path.join(d, "final_predictions_raw.csv"), os.path.join(d, "final_predictions.csv")
    if not (os.path.exists(raw) and os.path.exists(fin)): continue
    t = pd.read_csv(raw)
    cols = sorted((c for c in t.columns if c.startswith("Prob_Class_")), key=lambda c: int(c.rsplit("_",1)[1]))
    if not cols: continue
    P = t[cols].to_numpy(float); y = t["True_Label"].to_numpy(int)
    rawp = t["Predicted_Label"].to_numpy(int)
    g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
    dc = cfg.get("dataset_config", {}) or {}
    cls = dc.get("constrained_class")
    if cls is None: continue
    cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
    lp, gp = cfg["constraint"]
    df = pd.DataFrame({"label": y, "grp": g if g is not None else 0})
    G = compute_global_constraints(df, "label", gp, constrained_class=[cls], num_classes=P.shape[1])
    L = compute_local_constraints(df, "label", lp, "grp", constrained_class=[cls], num_classes=P.shape[1])
    if G[cls] >= UNLIMITED: continue
    K = int(G[cls])
    rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
    eq, taken = topk_alloc(P, g, K, L, cls)
    # ---- CLEAN PLACEBO ----
    Q = P.copy(); Q[:, cls] = -np.inf
    pred_off = np.argmax(Q, axis=1)
    m = y != cls
    others = [c for c in range(P.shape[1]) if c != cls]
    placeboF1 = f1_score(y[m], pred_off[m], labels=others, average="macro", zero_division=0)
    placeboAcc = accuracy_score(y[m], pred_off[m])
    hp = cfg.get("hyperparams") or {}
    rows.append({"dataset": cfg.get("dataset_mode"), "cap": cfg.get("constraint_tag"),
        "model": cfg.get("model_name"), "seed": hp.get("seed"),
        "method": cfg.get("methodology"), "ep_max": ep_max(d),
        "warmup": hp.get("warmup_epochs"), "cep": hp.get("constraint_epochs"),
        "K": K, "taken": taken, "count_raw": int((rawp == cls).sum()),
        "AP": average_precision_score((y == cls).astype(int), P[:, cls]),
        "ccF1adj": f1_score(y, rel, labels=[cls], average="macro", zero_division=0),
        "ccF1eq": f1_score(y, eq, labels=[cls], average="macro", zero_division=0),
        "macroEq": f1_score(y, eq, average="macro", zero_division=0),
        "placeboF1": placeboF1, "placeboAcc": placeboAcc, "path": d})
o = pd.DataFrame(rows); o.to_csv(OUT, index=False)
print("scored %d -> %s" % (len(o), OUT))
