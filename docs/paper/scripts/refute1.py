"""Diagnostic: is out_paperfinal.csv a clean, unpooled corpus?"""
import pandas as pd, sys
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)

for f in ["out_paperfinal.csv", "out_paper_backbones.csv", "out_extra_robustness.csv"]:
    d = pd.read_csv("paper/scripts/" + f)
    print("=" * 110)
    print(f, "  rows =", len(d))
    print("=" * 110)
    print("methods:", sorted(d.method.dropna().unique()))
    print("datasets:", sorted(d.dataset.dropna().unique()))
    print("models:", sorted(d.model.dropna().unique()))
    print("caps:", sorted(d.cap.dropna().unique()))
    print("sweeps:", d.sweep.astype(str).value_counts().to_dict())
    print("arms:", d.arm.astype(str).value_counts().to_dict())
    print()
    print("-- per-method warmup / cepochs / lr_c / ce_skip (value counts) --")
    for m in sorted(d.method.dropna().unique()):
        s = d[d.method == m]
        print("  %-14s n=%4d  warmup=%s  cepochs=%s  lr=%s  lr_c=%s  ce_skip=%s" % (
            m, len(s),
            s.warmup.value_counts().to_dict(),
            s.cepochs.value_counts().to_dict(),
            s.lr.value_counts().to_dict(),
            s.lr_c.astype(str).value_counts().to_dict(),
            s.ce_skip.astype(str).value_counts().to_dict()))
    print()
    print("-- DUPLICATES per (dataset,model,cap,seed,method) --")
    k = ["dataset", "model", "cap", "seed", "method"]
    vc = d.groupby(k).size()
    print("  keys:", len(vc), " keys with >1 row:", int((vc > 1).sum()),
          " max dupes:", int(vc.max()))
    if (vc > 1).any():
        dd = vc[vc > 1].sort_values(ascending=False)
        print(dd.head(20).to_string())
        # what varies inside a duplicated key?
        ex = dd.index[0]
        sub = d.set_index(k).loc[[ex]].reset_index()
        print("\n  EXAMPLE duplicated key", ex)
        print(sub[["warmup","cepochs","lr","lr_c","ce_skip","sweep","arm","K","count_raw","ccF1adj","ccF1eq","AP","path"]].to_string(index=False))
    print()
