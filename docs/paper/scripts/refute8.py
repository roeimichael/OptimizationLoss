import numpy as np, pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500)
D = "paper/scripts/"
DUAL = ["fioretto_ldf","hounie_rcl"]; CLIP = ["heuristic","danits_lp"]
CELL = ["dataset","model","cap"]

print("="*126)
print("N.  THE THIRD LEG (extra_robustness) IS A POOLED CORPUS")
print("="*126)
er = pd.read_csv(D+"out_extra_robustness.csv")
k = CELL+["seed","method"]
vc = er.groupby(k).size().rename("rows").reset_index()
for m in DUAL+CLIP+["tralo"]:
    s = vc[vc.method==m]
    print("  %-14s keys=%3d   keys with >1 row=%3d   max rows/key=%d"
          % (m, len(s), int((s.rows>1).sum()), int(s.rows.max()) if len(s) else 0))
dup = er.groupby(k).size()
dupk = dup[dup>1].index
sub = er.set_index(k).loc[dupk].reset_index()
sub["arm_dir"] = sub.path.str.split("/").str[3:5].str.join("/")
print("\n  distinct experiment ARMS silently averaged under one method label:")
print(sub.groupby(["method","arm_dir"]).size().to_string())

print()
print("="*126)
print("O.  THE '39 cells' AND '81 cells' LABELS  (final_decomp prints len(table), not the n behind the mean)")
print("="*126)
def cells_like_theirs(fs, M="ccF1eq"):
    d = pd.concat([pd.read_csv(D+f) for f in fs], ignore_index=True)
    rows=[]
    for (ds,mo,cap), g in d.groupby(CELL):
        r={"dataset":ds,"model":mo,"cap":cap}
        piv=g.pivot_table(index="seed",columns="method",values=M)
        hd=[m for m in DUAL if m in piv.columns]; hc=[m for m in CLIP if m in piv.columns]
        if hd and hc:
            s=piv.dropna(subset=hd+hc)
            if not s.empty: r["d"]=(s[hd].max(axis=1)-s[hc].max(axis=1)).mean()
        rows.append(r)
    return pd.DataFrame(rows)
for lbl, fs in [("paper_final",["out_paperfinal.csv"]),
                ("paper_backbones",["out_paper_backbones.csv"]),
                ("extra_robustness",["out_extra_robustness.csv"])]:
    t = cells_like_theirs(fs)
    print("  %-18s printed cells=%2d   cells actually behind the mean=%2d"
          % (lbl, len(t), int(t["d"].notna().sum()) if "d" in t else 0))

print()
print("="*126)
print("P.  CONSTRAINT-SPECIFIC RESIDUAL: ccF1eq gap MINUS the placebo gap, per cell")
print("="*126)
def cellstat(dd, M):
    out=[]
    for (ds,mo,cap), g in dd.groupby(CELL):
        piv=g.pivot_table(index="seed",columns="method",values=M)
        a=[m for m in DUAL if m in piv.columns]; b=[m for m in CLIP if m in piv.columns]
        if not a or not b: continue
        s=piv.dropna(subset=a+b)
        if s.empty: continue
        out.append({"dataset":ds,"model":mo,"cap":cap,"d":(s[a].max(axis=1)-s[b].max(axis=1)).mean()})
    return pd.DataFrame(out)
pf = pd.read_csv(D+"out_ref6_pf.csv")
a = cellstat(pf,"ccF1eq").rename(columns={"d":"cc"})
b = cellstat(pf,"placeboF1").rename(columns={"d":"pl"})
m = a.merge(b,on=CELL); m["resid"] = m.cc - m.pl
v = m.resid
print("  ccF1eq gap        mean %+0.4f   D%d/C%d" % (m.cc.mean(), int((m.cc>.005).sum()), int((m.cc<-.005).sum())))
print("  placebo gap       mean %+0.4f   D%d/C%d" % (m.pl.mean(), int((m.pl>.005).sum()), int((m.pl<-.005).sum())))
print("  RESIDUAL cc-pl    mean %+0.4f   D%d/C%d/T%d   <-- what is left for the constraint"
      % (v.mean(), int((v>.005).sum()), int((v<-.005).sum()), int((v.abs()<=.005).sum())))
# OLS residual too
beta = np.polyfit(m.pl, m.cc, 1)
r2 = np.corrcoef(m.pl, m.cc)[0,1]**2
m["ols"] = m.cc - (beta[0]*m.pl + beta[1])
print("  OLS ccF1eq ~ placebo: slope %+0.2f  intercept %+0.4f  R2 %.2f"
      % (beta[0], beta[1], r2))
print("  intercept = constraint-specific effect at zero generic gap: %+0.4f" % beta[1])
