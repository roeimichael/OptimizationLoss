import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
d = pd.read_csv(sys.argv[1])
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
                    "delta": (s[a].max(axis=1) - s[b].max(axis=1)).mean()})
    return pd.DataFrame(out)

def fmt(t):
    v = t.delta.dropna()
    return "n=%2d mean %+0.4f  D%2d/C%2d/T%2d" % (len(t), v.mean(),
        int((v>.005).sum()), int((v<-.005).sum()), int((v.abs()<=.005).sum()))

print("="*128)
print("E.  IS THE ALLOCATION BUDGET REALLY EQUAL?  (taken must be method-independent)")
print("="*128)
tk = d.pivot_table(index=CELL+["seed"], columns="method", values="taken")
sp = tk.max(axis=1) - tk.min(axis=1)
print("  keys=%d  max spread of `taken` across methods = %d   (0 => ccF1eq is pure ranking@K)"
      % (len(tk), int(sp.max())))
print("  cells where taken < K (local rooms bind): %d of %d runs"
      % (int((d.taken < d.K).sum()), len(d)))

print()
print("="*128)
print("F.  DOES THE CAP EVEN BIND?  clipper raw count vs K, per cell")
print("="*128)
cl = d[d.method == "heuristic"].groupby(CELL).agg(K=("K","mean"), raw=("count_raw","mean"),
                                                   npos=("n_pos","mean")).reset_index()
cl["binds"] = cl.raw > cl.K
cl["ratio"] = cl.K / cl.raw
print("  cells where cap BINDS (clip raw > K): %d of %d" % (int(cl.binds.sum()), len(cl)))
print(cl.groupby("cap").agg(cells=("binds","size"), binding=("binds","sum"),
                            meanKoverRaw=("ratio","mean")).round(3).to_string())

print()
print("="*128)
print("G.  THE PLACEBO.  The count cap can ONLY move the constrained class.")
print("    macroOffRaw / accOffRaw = quality on the classes the constraint never touches.")
print("="*128)
for M in ["ccF1eq", "ccF1adj", "AP", "macroEq", "macroOffRaw", "accOffRaw"]:
    print("  best-dual - best-clip  %-12s  %s" % (M, fmt(cellstat(d, DUAL, CLIP, M))))

print()
print("="*128)
print("H.  DECOMPOSE THE +0.0156 BY CAP  (does it live where the cap binds?)")
print("="*128)
t = cellstat(d, DUAL, CLIP, "ccF1eq").merge(cl[CELL+["binds","ratio","K","raw"]], on=CELL)
for cap, g in t.groupby("cap"):
    v = g.delta
    print("  %-9s  binding %d/%d   K/raw %.2f   mean %+0.4f  D%d/C%d/T%d"
          % (cap, int(g.binds.sum()), len(g), g.ratio.mean(), v.mean(),
             int((v>.005).sum()), int((v<-.005).sum()), int((v.abs()<=.005).sum())))
print("  ---")
for lbl, sub in [("BINDING cells only", t[t.binds]), ("NON-binding cells only", t[~t.binds])]:
    v = sub.delta
    print("  %-24s cells=%2d  mean %+0.4f  D%d/C%d/T%d"
          % (lbl, len(sub), v.mean(), int((v>.005).sum()), int((v<-.005).sum()),
             int((v.abs()<=.005).sum())))

print()
print("="*128)
print("I.  DECOMPOSE BY DATASET AND BY BACKBONE")
print("="*128)
for key in ["dataset", "model"]:
    for k, g in t.groupby(key):
        v = g.delta
        print("  %-14s cells=%2d  mean %+0.4f  D%d/C%d/T%d"
              % (k, len(g), v.mean(), int((v>.005).sum()), int((v<-.005).sum()),
                 int((v.abs()<=.005).sum())))
    print("  ---")

print()
print("="*128)
print("J.  EPOCH ASYMMETRY: clipper trains `warmup` only; duals train warmup + N more")
print("="*128)
print(d.groupby("method").ep_max.agg(["mean","median","min","max"]).round(1).to_string())
ep = d[d.method.isin(DUAL)].groupby(CELL).ep_max.mean().rename("dual_extra_epochs")
t2 = t.merge(ep, on=CELL)
s = t2.dropna(subset=["dual_extra_epochs"])
print("  pearson r(ccF1eq gap, dual extra epochs) = %+0.3f  (n=%d cells)"
      % (np.corrcoef(s.dual_extra_epochs, s.delta)[0,1], len(s)))
lo, hi = s[s.dual_extra_epochs <= s.dual_extra_epochs.median()], s[s.dual_extra_epochs > s.dual_extra_epochs.median()]
for lbl, g in [("dual extra epochs LOW half", lo), ("dual extra epochs HIGH half", hi)]:
    v = g.delta
    print("  %-28s cells=%2d  mean %+0.4f  D%d/C%d" % (lbl, len(g), v.mean(),
          int((v>.005).sum()), int((v<-.005).sum())))
