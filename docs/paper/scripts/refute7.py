import numpy as np, pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
D = "paper/scripts/"
DUAL = ["fioretto_ldf", "hounie_rcl"]; CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]

def cellstat(dd, hd, hc, M):
    out = []
    for (ds, mo, cap), g in dd.groupby(CELL):
        piv = g.pivot_table(index="seed", columns="method", values=M)
        a = [m for m in hd if m in piv.columns]; b = [m for m in hc if m in piv.columns]
        if not a or not b: continue
        s = piv.dropna(subset=a+b)
        if s.empty: continue
        out.append({"dataset": ds, "model": mo, "cap": cap,
                    "delta": (s[a].max(axis=1)-s[b].max(axis=1)).mean()})
    return pd.DataFrame(out)

def fmt(t):
    v = t.delta.dropna()
    return "cells=%2d  mean %+0.4f   D%2d/C%2d/T%2d" % (len(t), v.mean(),
        int((v>.005).sum()), int((v<-.005).sum()), int((v.abs()<=.005).sum()))

pf = pd.read_csv(D+"out_ref6_pf.csv")
bb = pd.read_csv(D+"out_ref6_bb.csv")
nce = pd.read_csv(D+"out_ref6_nce.csv"); hb = pd.read_csv(D+"out_ref6_hb30.csv")
hb = hb[hb.method.isin(CLIP)]; nce = nce[nce.method.isin(["tralo"]+DUAL)]
mix = pd.concat([nce, hb], ignore_index=True)
CAPS = ["L30_G30","L50_G50"]; MODELS = ["MobileNetV3","RegNetY400MF"]
mix = mix[mix.cap.isin(CAPS) & mix.model.isin(MODELS)]

print("="*126)
print("K.  CLEAN PLACEBO -- constrained column DELETED, evaluated only on true off-class samples.")
print("    The count cap acts on P[:,cls] alone, so it cannot reach this metric.")
print("="*126)
for lbl, dd in [("paper_final  (clipper 50ep vs duals 50+N ep)", pf),
                ("paper_backbones (same asymmetry)", bb),
                ("headroom noceskip (COMPUTE MATCHED, 30ep both)", mix)]:
    print("  --- %s ---" % lbl)
    for M in ["ccF1eq", "ccF1adj", "AP", "macroEq", "placeboF1", "placeboAcc"]:
        print("      best-dual - best-clip  %-11s %s" % (M, fmt(cellstat(dd, DUAL, CLIP, M))))

print()
print("="*126)
print("L.  Is the constrained-class 'win' bigger or smaller than the placebo, per cell?")
print("="*126)
for lbl, dd in [("paper_final", pf), ("paper_backbones", bb),
                ("headroom noceskip (matched)", mix)]:
    a = cellstat(dd, DUAL, CLIP, "ccF1eq").rename(columns={"delta":"cc"})
    b = cellstat(dd, DUAL, CLIP, "placeboF1").rename(columns={"delta":"pl"})
    c = cellstat(dd, DUAL, CLIP, "placeboAcc").rename(columns={"delta":"pa"})
    m = a.merge(b, on=CELL).merge(c, on=CELL)
    print("  %-30s cells=%2d   ccF1eq %+0.4f   placeboF1 %+0.4f   placeboAcc %+0.4f"
          % (lbl, len(m), m.cc.mean(), m.pl.mean(), m.pa.mean()))
    print("      cells where placeboAcc gap > 0 : %d/%d      r(cc, placeboF1) = %+0.3f"
          % (int((m.pa > 0).sum()), len(m), np.corrcoef(m.cc, m.pl)[0,1]))

print()
print("="*126)
print("M.  ONE-SIDED EPOCH LEDGER (median epochs of ADDITIONAL optimisation past the shared warm-up)")
print("="*126)
for lbl, dd in [("paper_final", pf), ("paper_backbones", bb), ("headroom noceskip+b30", mix)]:
    g = dd.groupby("method").agg(warmup=("warmup","median"), extra=("ep_max","median"), n=("ep_max","size"))
    g["total_ish"] = g.warmup + g.extra
    print("  --- %s ---" % lbl); print(g.round(1).to_string())
