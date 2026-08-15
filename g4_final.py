"""GEOM probe 4 -- the two load-bearing numbers, on CLEAN CE models (the state
the constraint phase actually starts from).

(A) EXACT structure of the entropic-OT / Sinkhorn projection onto
    {at most K of the pool assigned to class c}.  Claim:
        pi_i = sigmoid( logit(p_ic) - f ),      f the single dual potential
    so the OT target is an EXACT monotone transform of the model's own p_ic:
    it cannot reorder anything.  Verified to machine precision here (the
    previous pass used the argmax margin instead of logit(p_c) and so showed a
    0.2 discrepancy; this is the corrected identity).

(B) WHERE THE DOWN-PUSH LANDS.  Split each candidate's per-sample weight by
    (direction) x (inside/outside the budget) x (true positive or not).
    The incumbent's fatal cell is "pushed DOWN, true positive, INSIDE the
    budget": samples the cap explicitly permits and the metric rewards.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/OptimizationLoss")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.utils.constants import UNLIMITED
from src.training.constraints import compute_global_constraints


def sinkhorn_cap(P, c, K, eps=1.0, iters=300):
    logQ = np.log(np.clip(P, 1e-30, 1)) / eps
    f = 0.0
    for _ in range(iters):
        Z = logQ.copy(); Z[:, c] -= f / eps
        Z -= Z.max(1, keepdims=True)
        A = np.exp(Z); A /= A.sum(1, keepdims=True)
        s = A[:, c].sum()
        if abs(s - K) < 1e-9:
            break
        f += eps * np.log(max(s, 1e-12) / K)
    return A, f


rows = []
n_sink = 0
for root in ("results/headroom", "results/pending_runs"):
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cp))
        except Exception:
            continue
        if cfg.get("status") != "completed":
            continue
        if cfg.get("methodology") not in ("heuristic", "danits_lp"):
            continue
        raw = os.path.join(os.path.dirname(cp), "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        t = pd.read_csv(raw)
        cols = sorted((x for x in t.columns if x.startswith("Prob_Class_")),
                      key=lambda x: int(x.rsplit("_", 1)[1]))
        if not cols:
            continue
        P = t[cols].to_numpy(float)
        P = P / P.sum(1, keepdims=True)
        y = t["True_Label"].to_numpy(int)
        dc = cfg.get("dataset_config") or {}
        cl = dc.get("constrained_class")
        if cl is None:
            continue
        c = int(cl[0] if isinstance(cl, (list, tuple)) else cl)
        lp, gp = cfg["constraint"]
        G = compute_global_constraints(pd.DataFrame({"label": y}), "label", gp,
                                       constrained_class=[c], num_classes=P.shape[1])
        if G[c] >= UNLIMITED:
            continue
        K = int(G[c]); N = len(P)
        if K < 5 or K >= N - 5:
            continue
        pc = np.clip(P[:, c], 1e-12, 1 - 1e-12)
        m = np.log(pc) - np.log1p(-pc)                     # logit(p_c); ranks == p_c ranks
        oth = P.copy(); oth[:, c] = -np.inf
        mtil = np.log(np.maximum(pc, 1e-30)) - np.log(np.maximum(oth.max(1), 1e-30))
        pos = (y == c)
        order = np.argsort(-m)
        rank = np.empty(N, int); rank[order] = np.arange(N)
        inside = rank < K
        ms = np.sort(m)[::-1]
        theta = 0.5 * (ms[K - 1] + ms[K])
        s = float(np.median(np.abs(m - np.median(m)))) + 1e-9
        hp = cfg.get("hyperparams") or {}
        r = dict(dataset=cfg.get("dataset_mode"), model=cfg.get("model_name"),
                 seed=hp.get("seed"), warmup=hp.get("warmup_epochs"),
                 sweep=cfg.get("sweep_tag"), N=N, K=K, base=float(pos.mean()))

        if n_sink < 40:
            n_sink += 1
            for eps in (1.0, 0.5, 0.1):
                A, f = sinkhorn_cap(P, c, K, eps=eps)
                pi = A[:, c]
                pred = 1.0 / (1.0 + np.exp(-np.clip(m - f / eps * eps, -60, 60) + 0))
                pred = 1.0 / (1.0 + np.exp(-(m - f)))
                r["sinkerr_eps%.1f" % eps] = float(np.abs(pi - pred).max())
                r["sinkmass_eps%.1f" % eps] = float(pi.sum())
                r["sinkdiscord_eps%.1f" % eps] = float(
                    (np.sign(np.diff(pi[order])) > 0).mean())

        def landing(w, sign, tag):
            W = w.sum()
            if W <= 0:
                return
            up = sign > 0; dn = sign < 0
            r[tag + "_up"] = float(w[up].sum() / W)
            r[tag + "_dnTPin"] = float(w[dn & pos & inside].sum() / W)
            r[tag + "_dnTPout"] = float(w[dn & pos & ~inside].sum() / W)
            r[tag + "_upTPin"] = float(w[up & pos & inside].sum() / W)
            r[tag + "_dnFPin"] = float(w[dn & ~pos & inside].sum() / W)
            r[tag + "_align"] = float((w * np.sign(sign) * pos).sum() / W)
            r[tag + "_neff"] = float(W ** 2 / (w ** 2).sum() / N)

        # incumbent: dS/dz = p(1-p), everything pushed DOWN
        landing(pc * (1 - pc), -np.ones(N), "inc")
        # OT / self-labelling CE onto the budget pseudo-label
        landing(np.where(inside, 1 - pc, pc), np.where(inside, 1.0, -1.0), "ot")
        # boundary-smoothed count on the VERIFIED predicate (argmax margin)
        for tau in (1.0, 0.25):
            sg = 1 / (1 + np.exp(-np.clip(mtil / tau, -50, 50)))
            landing(sg * (1 - sg) / tau, -np.ones(N), "sig%.2f" % tau)
        # cut hinge
        for g in (0.5, 1.0):
            u = (m - theta) / s
            act = np.where(inside, u < g, -u < g)
            landing(act.astype(float), np.where(inside, 1.0, -1.0), "cut%.1f" % g)
        rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(os.path.expanduser("~/OptimizationLoss/newdirections/arm_geom/g4_final.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 80)
print("runs:", len(d))
sk = [x for x in d.columns if x.startswith("sink")]
print("\n=== (A) Sinkhorn cap projection vs sigmoid(logit(p_c) - f) ===")
print(d[sk].describe().loc[["count", "mean", "max"]].to_string())
print("\n=== (B) weight landing, clean CE models ===")
for tag in ("inc", "ot", "sig1.00", "sig0.25", "cut0.5", "cut1.0"):
    cs = [x for x in d.columns if x.startswith(tag + "_")]
    if not cs:
        continue
    print("\n-- %s --" % tag)
    print(d.groupby("dataset")[cs].mean().round(3).to_string())
