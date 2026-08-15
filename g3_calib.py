"""GEOM probe 3.

(1) STRUCTURE.  Claim: in this setting (one inequality cap on one class, single
    bag, uniform sample mass) the entropic-OT / Sinkhorn projection of the
    model's posterior onto {at most K assigned to c} is EXACTLY
        pi_i = sigmoid((m_i - theta)/eps),   m_i = log p_ic - log max_{j!=c} p_ij
    with theta the single dual potential fixed by sum_i pi_i = K.  Verified
    numerically against a Sinkhorn iteration.  If true, the whole content of the
    count constraint at the assignment level is ONE SCALAR: the location theta.

(2) CALIBRATION.  For gamma in MAD units, how many samples are active in the
    cut hinge, and what is the label composition of the two sides?

(3) LABEL ALIGNMENT of each candidate per-sample weight:
        align = sum_i w_i * s_i * 1[y_i = c] / sum_i w_i * |s_i|
    with s_i the SIGN of the push on z_ic (+1 = up).  Negative means the update
    is, on net, pushing true positives down: the loss fighting itself.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/OptimizationLoss")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.utils.constants import UNLIMITED
from src.training.constraints import compute_global_constraints

GAMMAS = (0.25, 0.5, 1.0, 2.0)


def sinkhorn_cap(P, c, K, eps=1.0, iters=400):
    """Entropic projection onto {row-stochastic, column-c mass <= K}."""
    logQ = np.log(np.clip(P, 1e-30, 1)) / eps
    f = 0.0
    for _ in range(iters):
        Z = logQ.copy()
        Z[:, c] -= f
        Z -= Z.max(1, keepdims=True)
        A = np.exp(Z)
        A /= A.sum(1, keepdims=True)
        s = A[:, c].sum()
        if s <= K + 1e-9 and f <= 1e-12:
            return A, 0.0
        f += eps * np.log(max(s, 1e-12) / K)
        if abs(s - K) < 1e-6:
            break
    return A, f


rows = []
for root in ("results/headroom", "results/pending_runs"):
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cp))
        except Exception:
            continue
        if cfg.get("status") != "completed":
            continue
        if cfg.get("methodology") not in ("heuristic", "danits_lp"):
            continue          # clean CE score field = the state the loss starts from
        d = os.path.dirname(cp)
        raw = os.path.join(d, "final_predictions_raw.csv")
        if not os.path.exists(raw):
            continue
        t = pd.read_csv(raw)
        cols = sorted((x for x in t.columns if x.startswith("Prob_Class_")),
                      key=lambda x: int(x.rsplit("_", 1)[1]))
        if not cols:
            continue
        P = t[cols].to_numpy(float)
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
        pc = P[:, c]
        oth = P.copy(); oth[:, c] = -np.inf
        m = np.log(np.maximum(pc, 1e-30)) - np.log(np.maximum(oth.max(1), 1e-30))
        pos = (y == c)
        order = np.argsort(-m)
        rank = np.empty(N, int); rank[order] = np.arange(N)
        keep = rank < K
        ms = np.sort(m)[::-1]
        theta = 0.5 * (ms[K - 1] + ms[K])
        s = float(np.median(np.abs(m - np.median(m)))) + 1e-9

        hp = cfg.get("hyperparams") or {}
        r = dict(dataset=cfg.get("dataset_mode"), model=cfg.get("model_name"),
                 cap=cfg.get("constraint_tag"), seed=hp.get("seed"),
                 warmup=hp.get("warmup_epochs"), sweep=cfg.get("sweep_tag"),
                 N=N, K=K, base=float(pos.mean()), mad=s, theta=float(theta))

        # ---- (1) Sinkhorn structure check (subsample for speed) ----
        if len(rows) < 60:
            for eps in (1.0, 0.5):
                A, f = sinkhorn_cap(P, c, K, eps=eps)
                pi = A[:, c]
                th = f            # dual potential, in logit units
                pred = 1 / (1 + np.exp(-np.clip((m - th) / eps, -50, 50)))
                r["sink_maxerr_eps%.1f" % eps] = float(np.abs(pi - pred).max())
                r["sink_mass_eps%.1f" % eps] = float(pi.sum())
                r["sink_spearman_eps%.1f" % eps] = float(
                    pd.Series(pi).corr(pd.Series(m), method="spearman"))
                r["sink_topK_jacc_eps%.1f" % eps] = float(
                    len(set(np.argsort(-pi)[:K]) & set(order[:K])) / K)

        # ---- (2) gamma calibration + (3) label alignment ----
        # incumbent soft count: all weight pushes DOWN
        w = pc * (1 - pc)
        r["align_incumbent"] = float(-(w[pos].sum()) / w.sum())
        r["neff_incumbent"] = float(w.sum() ** 2 / (w ** 2).sum() / N)
        # CE to the budget pseudo-label (the Asano/OT control): w = |1 - q|
        wce = np.where(keep, 1 - pc, pc)
        sgn = np.where(keep, 1.0, -1.0)
        r["align_otce"] = float((wce * sgn * pos).sum() / (wce.sum()))
        r["neff_otce"] = float(wce.sum() ** 2 / (wce ** 2).sum() / N)
        r["otce_wfar"] = float(wce[np.abs(m - theta) > 2 * s].sum() / wce.sum())
        for g in GAMMAS:
            u = (m - theta) / s
            act = np.where(keep, u < g, -u < g)
            n_act = int(act.sum())
            r["nact_g%.2f" % g] = n_act
            r["nactfrac_g%.2f" % g] = n_act / N
            ka = act & keep; da = act & ~keep
            r["actkeepTP_g%.2f" % g] = float(pos[ka].mean()) if ka.sum() else np.nan
            r["actdropTP_g%.2f" % g] = float(pos[da].mean()) if da.sum() else np.nan
            r["nkeep_g%.2f" % g] = int(ka.sum()); r["ndrop_g%.2f" % g] = int(da.sum())
            wc = act.astype(float)
            r["align_cut_g%.2f" % g] = float((wc * sgn * pos).sum() / max(wc.sum(), 1e-9))
        rows.append(r)

d = pd.DataFrame(rows)
d.to_csv(os.path.expanduser("~/OptimizationLoss/newdirections/arm_geom/g3_calib.csv"), index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 60)
print("runs:", len(d))
sk = [c for c in d.columns if c.startswith("sink_")]
print("\n=== (1) Sinkhorn cap projection == sigmoid((m-theta)/eps)? ===")
print(d[sk].describe().loc[["count", "mean", "max"]].round(6).to_string())
print("\n=== (2) gamma calibration (MAD units) ===")
cols = [c for c in d.columns if c.startswith(("nactfrac_", "actkeepTP_", "actdropTP_", "align_cut_"))]
print(d.groupby("dataset")[["N", "K", "base", "mad"] + cols].mean().round(3).to_string())
print("\n=== (3) label alignment of the per-sample update (+ = pushes TPs up) ===")
print(d.groupby("dataset")[["align_incumbent", "neff_incumbent", "align_otce",
                            "neff_otce", "otce_wfar", "align_cut_g1.00",
                            "nactfrac_g1.00"]].mean().round(3).to_string())
