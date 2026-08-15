"""GEOM probe 1 -- is the cut band informative, and where would a cut-supported
loss put its per-sample weight?

For every stored run with raw probabilities:
  m_i   = log p_ic - log max_{j!=c} p_ij        (margin for the constrained class)
  rank  = descending order of m
  band  = ranks [K - B/2, K + B/2)   with B = f * N   (f in {0.05, 0.10, 0.20})
Report:
  A  purity of the KEEP half (ranks just inside K) vs the DROP half (just outside)
     against the base rate -- if the two halves have the same TP rate the model's
     own cut ordering is noise and amplifying it is confirmation bias.
  B  per-sample gradient-weight landing for three candidate losses:
       incumbent  w_i = p_ic (1 - p_ic)                     (dS_c/dz_ic)
       sigcount   w_i = sigma'((m_i - 0)/tau)/tau           (boundary-smoothed count)
       cutband    w_i = 1/B on the fixed rank band          (order-statistic hinge)
     fractions on TRUE POSITIVES, on MISSED true positives, on false positives
     inside the cut, and n_eff = (sum w)^2 / sum w^2.
  C  runner-up class diversity inside the band (does the margin route the
     gradient to different competitor logits for different samples?)
  D  scale statistics needed to set gamma: MAD of m over the pool, and the
     normalised margin spread of the band.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/OptimizationLoss")
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from src.utils.constants import UNLIMITED
from src.training.constraints import compute_global_constraints


def rows_for(root):
    out = []
    for cp in glob.glob(root + "/**/config.json", recursive=True):
        try:
            cfg = json.load(open(cp))
        except Exception:
            continue
        if cfg.get("status") != "completed":
            continue
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
        K = int(G[c])
        N = len(P)
        if K < 5 or K >= N - 5:
            continue
        pc = P[:, c]
        oth = P.copy(); oth[:, c] = -np.inf
        run = oth.argmax(1)
        m = np.log(np.maximum(pc, 1e-30)) - np.log(np.maximum(oth.max(1), 1e-30))
        pos = (y == c)
        order = np.argsort(-m)
        rank = np.empty(N, int); rank[order] = np.arange(N)
        inside = rank < K                      # would be kept by a budget-K allocator
        missed_tp = pos & ~inside              # true positives outside the budget
        fp_in = ~pos & inside                  # false positives spending budget

        hp = cfg.get("hyperparams") or {}
        r = dict(method=cfg.get("methodology"), dataset=cfg.get("dataset_mode"),
                 model=cfg.get("model_name"), cap=cfg.get("constraint_tag"),
                 seed=hp.get("seed"), warmup=hp.get("warmup_epochs"),
                 lrc=hp.get("lr_constraint"), ceskip=hp.get("enable_ce_skip"),
                 sweep=cfg.get("sweep_tag"), N=N, K=K,
                 base_rate=float(pos.mean()), prec_at_K=float(pos[inside].mean()),
                 mad_m=float(np.median(np.abs(m - np.median(m)))),
                 std_m=float(m.std()),
                 theta=float(0.5 * (np.sort(m)[::-1][K - 1] + np.sort(m)[::-1][K])))

        # ---- A: purity of each half of the band, several band widths ----
        for f in (0.05, 0.10, 0.20):
            B = max(4, int(round(f * N)))
            h = B // 2
            keep = order[max(0, K - h):K]
            drop = order[K:min(N, K + h)]
            r["Bsz_f%.2f" % f] = len(keep) + len(drop)
            r["keepTP_f%.2f" % f] = float(pos[keep].mean()) if len(keep) else np.nan
            r["dropTP_f%.2f" % f] = float(pos[drop].mean()) if len(drop) else np.nan
            r["purgap_f%.2f" % f] = (float(pos[keep].mean() - pos[drop].mean())
                                     if len(keep) and len(drop) else np.nan)
            # normalised margin spread the hinge would have to close
            sc = np.median(np.abs(m - np.median(m))) + 1e-9
            band = np.concatenate([keep, drop])
            r["bandwidth_norm_f%.2f" % f] = float((m[keep].min() - m[drop].max()) / sc) \
                if len(keep) and len(drop) else np.nan
            r["band_absmarg_norm_f%.2f" % f] = float(np.mean(np.abs(m[band] - r["theta"]) / sc))
            # runner-up diversity inside the band
            vc = pd.Series(run[band]).value_counts(normalize=True)
            r["band_runup_top1_f%.2f" % f] = float(vc.iloc[0])
            r["band_runup_ndist_f%.2f" % f] = int((vc > 0.05).sum())
            # weight landing for the cutband loss
            w = np.zeros(N); w[band] = 1.0
            W = w.sum()
            r["cut_wTP_f%.2f" % f] = float(w[pos].sum() / W)
            r["cut_wMISSED_f%.2f" % f] = float(w[missed_tp].sum() / W)
            r["cut_wFPin_f%.2f" % f] = float(w[fp_in].sum() / W)
            r["cut_neff_f%.2f" % f] = float(W ** 2 / (w ** 2).sum())

        # ---- B: incumbent and boundary-smoothed count ----
        w = pc * (1 - pc); W = w.sum()
        r.update(inc_wTP=float(w[pos].sum() / W), inc_wMISSED=float(w[missed_tp].sum() / W),
                 inc_wFPin=float(w[fp_in].sum() / W),
                 inc_neff=float(W ** 2 / (w ** 2).sum()),
                 inc_neff_frac=float(W ** 2 / (w ** 2).sum() / N))
        for tau in (1.0, 0.25, 0.05):
            s = 1 / (1 + np.exp(-np.clip(m / tau, -50, 50)))
            w2 = s * (1 - s) / tau; W2 = w2.sum()
            if W2 <= 0:
                continue
            r["sig_wTP_tau%.2f" % tau] = float(w2[pos].sum() / W2)
            r["sig_wMISSED_tau%.2f" % tau] = float(w2[missed_tp].sum() / W2)
            r["sig_wFPin_tau%.2f" % tau] = float(w2[fp_in].sum() / W2)
            r["sig_neff_tau%.2f" % tau] = float(W2 ** 2 / (w2 ** 2).sum())
            r["sig_nefffrac_tau%.2f" % tau] = float(W2 ** 2 / (w2 ** 2).sum() / N)
        r["path"] = cp
        out.append(r)
    return out


if __name__ == "__main__":
    roots = sys.argv[1:-1]
    o = sys.argv[-1]
    d = pd.DataFrame(sum((rows_for(r) for r in roots), []))
    d.to_csv(o, index=False)
    print("rows", len(d), "->", o)
