import numpy as np
import pandas as pd

CELL = ["dataset", "model", "cap"]
D = pd.read_csv("paper/scripts/out_redo_tax2.csv")
T = pd.read_csv("paper/scripts/out_taxonomy.csv")
D = D.merge(T[["path", "ccF1eq", "AP", "macroEq"]], on="path", how="left")
assert D.ccF1eq.notna().all()

print("=" * 92)
print("0. RUN-KEY UNIQUENESS (lanes could duplicate a cell/seed)")
print("=" * 92)
k = D.groupby(["dataset", "model", "cap", "method", "seed"]).size()
print("distinct (ds,bb,cap,method,seed) keys: %d ; max runs per key: %d ; total %d"
      % (len(k), k.max(), k.sum()))
print("seeds present:", sorted(D.seed.unique()))
print("runs per (cell,method):", sorted(D.groupby(CELL + ["method"]).size().unique()))

print("\n" + "=" * 92)
print("1. THE CLAIM'S OWN SENSITIVITY ARITHMETIC: COLLAPSE d_ccF1eq at K/2, K/3, K/4")
print("   (within-cell deviation, exactly as taxonomy.py section 4 computes it)")
print("=" * 92)
for c in ["ccF1eq", "AP", "macroEq"]:
    D["d_" + c] = D[c] - D.groupby(CELL)[c].transform("mean")


def klass(s_str, count_raw, K, tail, cd):
    s = np.array([int(x) for x in s_str], dtype=int)
    if s.sum() == 0:
        return "NEVER_SAT"
    if cd is not None and count_raw < K / float(cd):
        return "COLLAPSE"
    if not (len(s) >= tail and s[-tail:].all()):
        return "SAT_THEN_DRIFT"
    return "OSC_THEN_LOCK" if ((s[:-1] == 1) & (s[1:] == 0)).sum() >= 1 else "LOCK_AND_HOLD"


rows = []
for cd in [2.0, 2.5, 3.0, 4.0, 5.0]:
    lab = np.array([klass(r.s, r.count_raw, r.K, 5, cd) for _, r in D.iterrows()])
    m = lab == "COLLAPSE"
    rows.append({"threshold": "K/%g" % cd, "n": int(m.sum()),
                 "d_ccF1eq": D.d_ccF1eq[m].mean(), "d_AP": D.d_AP[m].mean(),
                 "mean_ratio": (D.count_raw[m] / D.K[m]).mean()})
R = pd.DataFrame(rows)
print(R.to_string(index=False))
v = R.d_ccF1eq.tolist()
mono = all(v[i] >= v[i + 1] for i in range(len(v) - 1)) or \
       all(v[i] <= v[i + 1] for i in range(len(v) - 1))
print("\n  claim says '(monotone in depth)'. Is the d_ccF1eq sequence monotone? %s" % mono)
print("  sequence K/2 -> K/5: %s" % ["%.4f" % x for x in v])

print("\n" + "=" * 92)
print("2. PER-SEED STABILITY OF THE POPULATION (36 runs per seed)")
print("   The claim reports one population over 144 runs. Would another 4 seeds")
print("   give the same split?")
print("=" * 92)
P = pd.crosstab(D.seed, D.klass)
print(P.to_string())
print("\n  per-class range across the 4 seeds (x4 to compare with the 144-run count):")
for c in P.columns:
    print("    %-15s seed counts %s -> extrapolated 144-run count %d..%d (reported %d)"
          % (c, list(P[c]), P[c].min() * 4, P[c].max() * 4,
             int((D.klass == c).sum())))

print("\n" + "=" * 92)
print("3. LABEL HOMOGENEITY WITHIN (cell, method): is the label a run property or noise?")
print("=" * 92)
n = D.groupby(CELL + ["method"]).klass.nunique()
print(n.value_counts().sort_index().to_string())
print("  36 (cell,method) groups of 4 seeds; %d have 1 label, %d have >=2, %d have >=3"
      % (int((n == 1).sum()), int((n >= 2).sum()), int((n >= 3).sum())))

print("\n" + "=" * 92)
print("4. COUNTING CELLS, NOT RUNS: modal class of each of the 12 cells, and whether")
print("   any cell is unanimous")
print("=" * 92)
for (ds, mo, cp), g in D.groupby(CELL):
    vc = g.klass.value_counts()
    print("  %-12s %-13s %-8s modal=%-15s %d/12 runs   unanimous=%s"
          % (ds, mo, cp, vc.index[0], vc.iloc[0], len(vc) == 1))
print("\n  cells whose modal class is NEVER_SAT: %d of 12"
      % sum(1 for _, g in D.groupby(CELL) if g.klass.value_counts().index[0] == "NEVER_SAT"))

print("\n" + "=" * 92)
print("5. SIBLING CAMPAIGN (CE gate ON) -- same knobs, one unrelated flag flipped")
print("=" * 92)
try:
    Gon = pd.read_csv("paper/scripts/out_taxonomy_gateon.csv")
    print("runs: %d" % len(Gon))
    print(Gon.klass.value_counts().to_string())
except Exception as e:
    print("could not read:", e)
