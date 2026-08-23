"""INDEPENDENT re-derivation of the bb3.py overshoot claim.

Nothing here imports analyze_headroom or src/. Every number -- K, the local
caps, the equal-budget allocation, ccF1eq, count_raw -- is recomputed from
config.json + final_predictions_raw.csv. The factbase is read ONLY at the end,
to check my numbers against it.

Targets are built per SEED and then averaged inside the atomic cell
(dataset, backbone, cap). Never pooled across cells.
"""
import glob
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

TR = "results/headroom/headroom_b30_lrc0.0001_noceskip"
CL = "results/headroom/headroom_b30"
CELL = ["dataset", "model", "cap"]
DUALS = ["fioretto_ldf", "hounie_rcl"]
CLIPM = ["heuristic", "danits_lp"]
UNLIM = 10 ** 10


# ---------------------------------------------------------------- scoring ---
def K_of(count, pct):
    """round(count*pct); numpy half-to-even, same as src._round_to_K."""
    return int(np.round(count * pct))


def equal_budget(P, gids, K, local_caps, cls):
    """Assign the constrained class to the K highest-scoring rows that still
    have local room; everyone else gets the argmax over the OTHER classes.
    Re-implemented from the metric definition, not copied."""
    order = np.argsort(-P[:, cls], kind="stable")
    room = dict(local_caps) if local_caps else {}
    chosen = np.zeros(len(P), dtype=bool)
    taken = 0
    for i in order:
        if taken >= K:
            break
        if room:
            g = int(gids[i])
            if room.get(g, 0) <= 0:
                continue
            room[g] -= 1
        chosen[i] = True
        taken += 1
    other = P.copy()
    other[:, cls] = -np.inf
    y = np.argmax(other, axis=1)
    y[chosen] = cls
    return y


def f1_binary(y_true, y_pred, cls):
    tp = int(((y_true == cls) & (y_pred == cls)).sum())
    fp = int(((y_true != cls) & (y_pred == cls)).sum())
    fn = int(((y_true == cls) & (y_pred != cls)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def average_precision(y_bin, score):
    """sklearn-style step AP: sum (R_n - R_{n-1}) * P_n over the score ranking."""
    o = np.argsort(-score, kind="stable")
    yb = y_bin[o]
    tp = np.cumsum(yb)
    fp = np.cumsum(1 - yb)
    prec = tp / np.maximum(1, tp + fp)
    npos = yb.sum()
    if npos == 0:
        return float("nan")
    rec = tp / npos
    drec = np.diff(np.concatenate([[0.0], rec]))
    return float((prec * drec).sum())


def macro_f1(y_true, y_pred, n_cls):
    vals = []
    for c in range(n_cls):
        if ((y_true == c).sum() == 0) and ((y_pred == c).sum() == 0):
            continue
        vals.append(f1_binary(y_true, y_pred, c))
    return float(np.mean(vals)) if vals else float("nan")


def score_root(root):
    rows = []
    for cfg_path in sorted(glob.glob(root + "/**/config.json", recursive=True)):
        d = os.path.dirname(cfg_path)
        raw = os.path.join(d, "final_predictions_raw.csv")
        fin = os.path.join(d, "final_predictions.csv")
        if not (os.path.exists(raw) and os.path.exists(fin)):
            continue
        cfg = json.load(open(cfg_path))
        t = pd.read_csv(raw)
        pcols = sorted((c for c in t.columns if c.startswith("Prob_Class_")),
                       key=lambda c: int(c.rsplit("_", 1)[1]))
        P = t[pcols].to_numpy(float)
        y = t["True_Label"].to_numpy(int)
        g = t["Group_ID"].to_numpy(int) if "Group_ID" in t.columns else None
        dc = cfg.get("dataset_config") or {}
        cls = dc.get("constrained_class")
        cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)
        lp, gp = cfg["constraint"]
        n_true = int((y == cls).sum())
        K = K_of(n_true, gp)
        caps = {}
        if g is not None:
            for grp in np.unique(g):
                caps[int(grp)] = K_of(int(((y == cls) & (g == grp)).sum()), lp)
        # my own argmax of the stored probabilities, not the stored label
        my_argmax = np.argmax(P, axis=1)
        stored_raw = t["Predicted_Label"].to_numpy(int)
        rel = pd.read_csv(fin)["Predicted_Label"].to_numpy(int)
        eq = equal_budget(P, g, K, caps, cls)
        hp = cfg.get("hyperparams") or {}
        rows.append({
            "dataset": cfg.get("dataset_mode"), "model": cfg.get("model_name"),
            "cap": cfg.get("constraint_tag"), "seed": hp.get("seed"),
            "method": cfg.get("methodology"), "arm": cfg.get("arm"),
            "warmup": hp.get("warmup_epochs"), "cepochs": hp.get("constraint_epochs"),
            "lr": hp.get("lr"), "lr_c": hp.get("lr_constraint"),
            "ce_skip": hp.get("enable_ce_skip"),
            "n_pool": len(y), "n_true": n_true, "K": K,
            "count_raw": int((stored_raw == cls).sum()),
            "count_raw_myargmax": int((my_argmax == cls).sum()),
            "argmax_mismatch": int((my_argmax != stored_raw).sum()),
            "count_adj": int((rel == cls).sum()),
            "sat": int((stored_raw != rel).sum() == 0),
            "ccF1eq": f1_binary(y, eq, cls),
            "ccF1raw": f1_binary(y, stored_raw, cls),
            "AP": average_precision((y == cls).astype(int), P[:, cls]),
            "macroEq": macro_f1(y, eq, P.shape[1]),
            "path": d,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ correlations ---
def _rank(v):
    return pd.Series(v).rank().to_numpy(float)


def pearson(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), 0
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def tstat_p(r, n):
    if n < 3 or abs(r) >= 1:
        return float("nan"), float("nan")
    t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
    # two-sided p from the t distribution via the incomplete beta
    try:
        from math import lgamma
        df = n - 2
        x = df / (df + t * t)
        # regularized incomplete beta I_x(df/2, 1/2) by continued fraction
        def betacf(a, b, x, itmax=300, eps=3e-12):
            qab, qap, qam = a + b, a + 1.0, a - 1.0
            c, dd = 1.0, 1.0 - qab * x / qap
            if abs(dd) < 1e-30:
                dd = 1e-30
            dd = 1.0 / dd
            h = dd
            for m in range(1, itmax + 1):
                m2 = 2 * m
                aa = m * (b - m) * x / ((qam + m2) * (a + m2))
                dd = 1.0 + aa * dd
                if abs(dd) < 1e-30:
                    dd = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                dd = 1.0 / dd
                h *= dd * c
                aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
                dd = 1.0 + aa * dd
                if abs(dd) < 1e-30:
                    dd = 1e-30
                c = 1.0 + aa / c
                if abs(c) < 1e-30:
                    c = 1e-30
                dd = 1.0 / dd
                de = dd * c
                h *= de
                if abs(de - 1.0) < eps:
                    break
            return h
        def betai(a, b, x):
            if x <= 0:
                return 0.0
            if x >= 1:
                return 1.0
            bt = math.exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                          + a * math.log(x) + b * math.log(1 - x))
            if x < (a + 1) / (a + b + 2):
                return bt * betacf(a, b, x) / a
            return 1.0 - bt * betacf(b, a, 1 - x) / b
        p = betai(df / 2.0, 0.5, x)
        return t, p
    except Exception:
        return t, float("nan")


def report(x, y, lab, out):
    r, n = pearson(np.asarray(x, float), np.asarray(y, float))
    if n == 0:
        return
    rs, _ = pearson(_rank(np.asarray(x, float)), _rank(np.asarray(y, float)))
    t, p = tstat_p(r, n)
    out.append({"encoding": lab, "n": n, "r": r, "r2": r * r,
                "spearman": rs, "t": t, "p": p})
    print("    %-40s n=%2d  r=%+0.3f  r2=%.3f  rho=%+0.3f  t=%+0.2f  p=%.3f"
          % (lab, n, r, r * r, rs, t, p if p == p else float("nan")))


def mannwhitney(xw, xl):
    """exact-ish U test by rank sum, normal approx with tie correction."""
    allv = np.concatenate([xw, xl])
    rk = pd.Series(allv).rank().to_numpy(float)
    n1, n2 = len(xw), len(xl)
    R1 = rk[:n1].sum()
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z = (U1 - mu) / sd if sd > 0 else 0.0
    return U1, z


def fisher_exact_2x2(a, b, c, d):
    def C(n, k):
        return math.comb(n, k)
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c
    def pr(x):
        return C(r1, x) * C(r2, c1 - x) / C(n, c1)
    p0 = pr(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p0 + 1e-12)


def best_split(x, win):
    """Best single threshold on x separating winning cells from losing cells.
    Reports the accuracy and the Fisher exact p of the resulting 2x2."""
    best = None
    for thr in sorted(set(x)):
        for sign in (+1, -1):
            pred = (x >= thr) if sign > 0 else (x < thr)
            acc = float((pred == win).mean())
            a = int((pred & win).sum()); b = int((pred & ~win).sum())
            c = int((~pred & win).sum()); d = int((~pred & ~win).sum())
            p = fisher_exact_2x2(a, b, c, d)
            if best is None or acc > best[0]:
                best = (acc, thr, sign, (a, b, c, d), p)
    return best


def main():
    print("=" * 118)
    print("STEP 1  independent scoring of every run (no rows_for, no src import)")
    print("=" * 118)
    tr = score_root(TR)
    cl = score_root(CL)
    print("trained runs scored: %d   clip-campaign runs scored: %d" % (len(tr), len(cl)))
    print("methods (trained): %s" % tr.method.value_counts().to_dict())
    print("methods (clip campaign): %s" % cl.method.value_counts().to_dict())
    print("argmax mismatches vs stored Predicted_Label: trained %d  clip %d"
          % (tr.argmax_mismatch.sum(), cl.argmax_mismatch.sum()))
    dup = tr.duplicated(subset=CELL + ["seed", "method"]).sum()
    print("duplicate (cell,seed,method) keys in trained campaign: %d" % dup)
    print("regime check  warmup=%s cepochs=%s lr=%s lr_c=%s ce_skip=%s"
          % (sorted(tr.warmup.unique()), sorted(tr.cepochs.unique()),
             sorted(tr.lr.unique()), sorted(tr.lr_c.unique()),
             sorted(map(str, tr.ce_skip.unique()))))
    clh = cl[cl.method.isin(CLIPM)]
    print("clipper regime warmup=%s cepochs=%s"
          % (sorted(clh.warmup.unique()), sorted(clh.cepochs.unique())))
    tr.to_csv("paper/scripts/out_refute_trained.csv", index=False)
    cl.to_csv("paper/scripts/out_refute_clip.csv", index=False)

    print("\n" + "=" * 118)
    print("STEP 2  per-cell target, paired within seed, averaged over SEED ONLY")
    print("=" * 118)
    piv = tr.pivot_table(index=CELL + ["seed"], columns="method", values="ccF1eq")
    pivA = tr.pivot_table(index=CELL + ["seed"], columns="method", values="AP")
    pivM = tr.pivot_table(index=CELL + ["seed"], columns="method", values="macroEq")
    pc = cl[cl.method.isin(CLIPM)].pivot_table(index=CELL + ["seed"],
                                               columns="method", values="ccF1eq")
    s = piv.copy()
    s["clip"] = pc.max(axis=1).reindex(s.index)
    s["T_D"] = s["tralo"] - s[DUALS].max(axis=1)
    s["T_C"] = s["tralo"] - s["clip"]
    s["T_F"] = s["tralo"] - s["fioretto_ldf"]
    s["T_H"] = s["tralo"] - s["hounie_rcl"]
    s = s.reset_index()
    sA = pivA.copy(); sA["T_D"] = sA["tralo"] - sA[DUALS].max(axis=1); sA = sA.reset_index()
    sM = pivM.copy(); sM["T_D"] = sM["tralo"] - sM[DUALS].max(axis=1); sM = sM.reset_index()

    cells = s.groupby(CELL).agg(
        T_D=("T_D", "mean"), T_D_sd=("T_D", "std"), nwin=("T_D", lambda v: int((v > 0).sum())),
        T_C=("T_C", "mean"), T_F=("T_F", "mean"), T_H=("T_H", "mean"),
        tralo=("tralo", "mean"), n=("T_D", "size")).reset_index()
    cells = cells.merge(sA.groupby(CELL)["T_D"].mean().rename("T_D_AP").reset_index(), on=CELL)
    cells = cells.merge(sM.groupby(CELL)["T_D"].mean().rename("T_D_macro").reset_index(), on=CELL)

    # ---- the unconstrained model's own count, recomputed from raw preds ----
    nat = cl[cl.method == "heuristic"].groupby(CELL).agg(
        clip_raw=("count_raw", "mean"), clip_min=("count_raw", "min"),
        clip_max=("count_raw", "max"), clip_sd=("count_raw", "std")).reset_index()
    nat_lp = cl[cl.method == "danits_lp"].groupby(CELL)["count_raw"].mean().rename("lp_raw").reset_index()
    ktab = tr.groupby(CELL).agg(K=("K", "first"), n_pool=("n_pool", "first"),
                                n_true=("n_true", "first")).reset_index()
    cells = cells.merge(nat, on=CELL).merge(nat_lp, on=CELL).merge(ktab, on=CELL)
    print("heuristic vs danits_lp raw count agreement (same plain-CE model): "
          "max |diff| = %.3f" % (cells.clip_raw - cells.lp_raw).abs().max())
    print("K == round(pct*n_true)? %s" % (
        all(cells.K == [K_of(nt, 0.3 if c == "L30_G30" else 0.5)
                        for nt, c in zip(cells.n_true, cells.cap)])))
    cells = cells.sort_values(["dataset", "cap", "model"]).reset_index(drop=True)
    print(cells[["dataset", "model", "cap", "K", "n_true", "clip_raw", "clip_min",
                 "clip_max", "T_D", "T_D_sd", "nwin", "T_C"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

    print("\n  established stratified T_D, in stratify.py's sort order "
          "(dataset, cap, model):")
    for ds, g in cells.groupby("dataset"):
        print("    %-12s %s" % (ds, "  ".join("%+.4f" % v for v in g.T_D)))

    # ------------------------------------------------------------ encodings --
    c = cells
    enc = {}
    enc["overshoot_ratio  clip/K"] = c.clip_raw / c.K
    enc["overshoot_abs    clip-K"] = c.clip_raw - c.K
    enc["cut_frac         1-K/clip"] = 1 - c.K / c.clip_raw
    enc["headroom         K/clip"] = c.K / c.clip_raw
    enc["log ratio"] = np.log(c.clip_raw / c.K)
    enc["sqrt ratio"] = np.sqrt(c.clip_raw / c.K)
    enc["ratio^2"] = (c.clip_raw / c.K) ** 2
    enc["1/ratio^2"] = (c.K / c.clip_raw) ** 2
    enc["excess/pool"] = (c.clip_raw - c.K) / c.n_pool
    enc["excess/n_true"] = (c.clip_raw - c.K) / c.n_true
    enc["excess/clip"] = (c.clip_raw - c.K) / c.clip_raw
    enc["natural_rate clip/pool"] = c.clip_raw / c.n_pool
    enc["calibration clip/n_true"] = c.clip_raw / c.n_true
    enc["K/pool"] = c.K / c.n_pool
    enc["K/n_true (=cap pct)"] = c.K / c.n_true
    enc["log excess_abs"] = np.log(np.maximum(1.0, c.clip_raw - c.K))
    enc["z(ratio) within dataset"] = (c.clip_raw / c.K) - (c.clip_raw / c.K).groupby(c.dataset).transform("mean")
    enc["rank(ratio) 1..12"] = (c.clip_raw / c.K).rank()
    enc["|ratio - median|"] = ((c.clip_raw / c.K) - (c.clip_raw / c.K).median()).abs()
    enc["median split(ratio)"] = ((c.clip_raw / c.K) > (c.clip_raw / c.K).median()).astype(float)
    enc["binds margin (clip-K)/K^0.5"] = (c.clip_raw - c.K) / np.sqrt(c.K)

    results = []
    print("\n" + "=" * 118)
    print("STEP 3  target = TraLO minus best dual (ccF1eq), 12 cells, RAW")
    print("=" * 118)
    for k, v in enc.items():
        report(v.to_numpy(float), c.T_D.to_numpy(float), k, results)

    print("\n  target = TraLO minus best dual, AP")
    for k in ["overshoot_ratio  clip/K", "cut_frac         1-K/clip", "overshoot_abs    clip-K"]:
        report(enc[k].to_numpy(float), c.T_D_AP.to_numpy(float), k + " [AP]", results)
    print("\n  target = TraLO minus best dual, macroEq")
    for k in ["overshoot_ratio  clip/K", "cut_frac         1-K/clip"]:
        report(enc[k].to_numpy(float), c.T_D_macro.to_numpy(float), k + " [macro]", results)
    print("\n  target = TraLO minus plain-CE clipper, ccF1eq")
    for k in ["overshoot_ratio  clip/K", "cut_frac         1-K/clip"]:
        report(enc[k].to_numpy(float), c.T_C.to_numpy(float), k + " [T-C]", results)
    print("\n  target = TraLO minus fioretto only / hounie only, ccF1eq")
    report(enc["overshoot_ratio  clip/K"].to_numpy(float), c.T_F.to_numpy(float), "ratio [T-fioretto]", results)
    report(enc["overshoot_ratio  clip/K"].to_numpy(float), c.T_H.to_numpy(float), "ratio [T-hounie]", results)

    print("\n" + "=" * 118)
    print("STEP 4  demeaned variants (dataset / backbone / cap / two-way)")
    print("=" * 118)
    for key in ["dataset", "model", "cap"]:
        print("  -- removing %s means --" % key)
        yd = c.T_D - c.groupby(key)["T_D"].transform("mean")
        for k in ["overshoot_ratio  clip/K", "cut_frac         1-K/clip",
                  "overshoot_abs    clip-K", "natural_rate clip/pool"]:
            xv = enc[k] - enc[k].groupby(c[key]).transform("mean")
            report(xv.to_numpy(float), yd.to_numpy(float), k + " (-%s)" % key, results)
    print("  -- removing dataset AND backbone means (additive two-way) --")
    yd = c.T_D - c.groupby("dataset")["T_D"].transform("mean") \
             - c.groupby("model")["T_D"].transform("mean") + c.T_D.mean()
    for k in ["overshoot_ratio  clip/K", "cut_frac         1-K/clip"]:
        xv = enc[k] - enc[k].groupby(c.dataset).transform("mean") \
                    - enc[k].groupby(c.model).transform("mean") + enc[k].mean()
        report(xv.to_numpy(float), yd.to_numpy(float), k + " (-ds-bb)", results)

    print("\n" + "=" * 118)
    print("STEP 5  COUNT THE CELLS instead of averaging them")
    print("=" * 118)
    win = (c.T_D > 0).to_numpy()
    winsig = (c.T_D > 0.005).to_numpy()
    loss = (c.T_D < -0.005).to_numpy()
    print("  cells TraLO>0: %d/12   >+0.005: %d   <-0.005: %d"
          % (win.sum(), winsig.sum(), loss.sum()))
    for k in ["overshoot_ratio  clip/K", "overshoot_abs    clip-K",
              "cut_frac         1-K/clip", "natural_rate clip/pool",
              "calibration clip/n_true"]:
        x = enc[k].to_numpy(float)
        xw, xl = x[win], x[~win]
        U, z = mannwhitney(xw, xl)
        acc, thr, sign, tab, pf = best_split(x, win)
        pb, _ = pearson(x, win.astype(float))
        print("    %-32s win-mean %.4f  loss-mean %.4f  U=%.1f z=%+0.2f"
              "  pt-biserial r=%+0.3f r2=%.3f  bestsplit acc=%.2f (thr%s%.4g) "
              "2x2=%s fisher p=%.3f"
              % (k, xw.mean(), xl.mean(), U, z, pb, pb * pb, acc,
                 ">=" if sign > 0 else "<", thr, tab, pf))
    print("\n  per-seed win counts vs overshoot rank (48 paired seeds):")
    ps = s[CELL + ["seed", "T_D"]].copy()
    ps["ratio"] = ps.merge(c[CELL + ["clip_raw", "K"]], on=CELL, how="left").eval("clip_raw/K").to_numpy()
    ps["half"] = np.where(ps.ratio > ps.ratio.median(), "high overshoot", "low overshoot")
    tabl = ps.groupby("half")["T_D"].agg(n="size", wins=lambda v: int((v > 0).sum()),
                                         mean="mean")
    print(tabl.to_string())
    a = int(tabl.loc["high overshoot", "wins"]); b = int(tabl.loc["high overshoot", "n"]) - a
    cc_ = int(tabl.loc["low overshoot", "wins"]); d = int(tabl.loc["low overshoot", "n"]) - cc_
    print("  2x2 [high: %d win/%d loss | low: %d win/%d loss]  fisher p=%.4f"
          % (a, b, cc_, d, fisher_exact_2x2(a, b, cc_, d)))

    print("\n" + "=" * 118)
    print("STEP 6  SEED-LEVEL (n=48), per-seed clipper count as the overshoot")
    print("=" * 118)
    cr = cl[cl.method == "heuristic"][CELL + ["seed", "count_raw"]].rename(
        columns={"count_raw": "clip_raw_seed"})
    p2 = s[CELL + ["seed", "T_D", "T_C"]].merge(cr, on=CELL + ["seed"], how="left")
    p2 = p2.merge(ktab, on=CELL)
    print("  matched rows: %d of %d" % (p2.clip_raw_seed.notna().sum(), len(p2)))
    p2["ratio_seed"] = p2.clip_raw_seed / p2.K
    p2["abs_seed"] = p2.clip_raw_seed - p2.K
    report(p2.ratio_seed.to_numpy(float), p2.T_D.to_numpy(float), "seed ratio vs T-D", results)
    report(p2.abs_seed.to_numpy(float), p2.T_D.to_numpy(float), "seed abs vs T-D", results)
    for key in ["dataset", "model", "cap"]:
        xv = p2.ratio_seed - p2.groupby(key)["ratio_seed"].transform("mean")
        yv = p2.T_D - p2.groupby(key)["T_D"].transform("mean")
        report(xv.to_numpy(float), yv.to_numpy(float), "seed ratio (-%s)" % key, results)
    xv = p2.ratio_seed - p2.groupby(CELL)["ratio_seed"].transform("mean")
    yv = p2.T_D - p2.groupby(CELL)["T_D"].transform("mean")
    report(xv.to_numpy(float), yv.to_numpy(float), "seed ratio (within cell)", results)

    print("\n" + "=" * 118)
    print("STEP 7  quadratic / two-predictor fits (is it an inverted U?)")
    print("=" * 118)
    x = (c.clip_raw / c.K).to_numpy(float)
    y = c.T_D.to_numpy(float)
    for deg, lab in [(1, "linear"), (2, "quadratic"), (3, "cubic")]:
        A = np.vander(x, deg + 1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yh = A @ beta
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        n, k = len(y), deg + 1
        adj = 1 - (1 - r2) * (n - 1) / (n - k)
        print("    overshoot_ratio %-10s R2=%.3f  adjR2=%+0.3f" % (lab, r2, adj))
    # is_regnet + dataset dummies for reference
    D = pd.get_dummies(c.dataset, drop_first=True).to_numpy(float)
    reg = (c.model == "RegNetY400MF").astype(float).to_numpy()[:, None]
    for lab, A in [("dataset dummies only", np.hstack([np.ones((12, 1)), D])),
                   ("is_regnet only", np.hstack([np.ones((12, 1)), reg])),
                   ("dataset+is_regnet", np.hstack([np.ones((12, 1)), D, reg])),
                   ("dataset+is_regnet+ratio",
                    np.hstack([np.ones((12, 1)), D, reg, x[:, None]])),
                   ("ratio only", np.hstack([np.ones((12, 1)), x[:, None]]))]:
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yh = A @ beta
        r2 = 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print("    %-26s R2=%.3f" % (lab, r2))

    print("\n" + "=" * 118)
    print("STEP 8  how much of T_D is even explainable? (seed noise ceiling)")
    print("=" * 118)
    within = s.groupby(CELL)["T_D"].var(ddof=1)
    between = cells.T_D.var(ddof=1)
    se2 = float(within.mean() / 4.0)
    print("  between-cell var of the cell means : %.3e" % between)
    print("  mean within-cell var (per seed)    : %.3e  -> var of a 4-seed mean %.3e"
          % (float(within.mean()), se2))
    true_var = max(0.0, between - se2)
    rel = true_var / between if between > 0 else float("nan")
    print("  implied TRUE between-cell var      : %.3e   reliability of the cell mean = %.3f"
          % (true_var, rel))
    r_ratio, _ = pearson(x, y)
    if rel > 0:
        print("  attenuation-corrected r for overshoot_ratio: %+0.3f  (r2=%.3f)"
              % (r_ratio / math.sqrt(rel), (r_ratio / math.sqrt(rel)) ** 2))

    print("\n" + "=" * 118)
    print("STEP 9  cross-check against the prior agent's factbase")
    print("=" * 118)
    fb = pd.read_csv("paper/scripts/out_factbase.csv")
    fb = fb[fb.campaign == "lrc0.0001_noceskip"]
    m = c[CELL + ["K", "clip_raw", "T_D"]].merge(
        fb[fb.method == "tralo"][CELL + ["K", "clip_raw", "d_vs_bestdual", "ccF1eq"]],
        on=CELL, suffixes=("_mine", "_fb"))
    m["dK"] = m.K_mine - m.K_fb
    m["dclip"] = m.clip_raw_mine - m.clip_raw_fb
    m["dT"] = m.T_D - m.d_vs_bestdual
    print(m[["dataset", "model", "cap", "K_mine", "K_fb", "clip_raw_mine",
             "clip_raw_fb", "T_D", "d_vs_bestdual", "dT"]]
          .to_string(index=False, float_format=lambda x: "%.5f" % x))
    print("  max |dK|=%g  max |dclip|=%g  max |dT_D|=%.2e"
          % (m.dK.abs().max(), m.dclip.abs().max(), m.dT.abs().max()))

    R = pd.DataFrame(results).sort_values("r2", ascending=False)
    R.to_csv("paper/scripts/out_refute_overshoot.csv", index=False)
    print("\n" + "=" * 118)
    print("TOP 8 ENCODINGS BY r2 (of %d tested)" % len(R))
    print("=" * 118)
    print(R.head(8).to_string(index=False, float_format=lambda x: "%.4f" % x))
    print("\n  encodings reaching r2>=0.15 : %d" % int((R.r2 >= 0.15).sum()))
    print("  encodings with p<0.05        : %d" % int((R.p < 0.05).sum()))
    print("\nwrote paper/scripts/out_refute_overshoot.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
