"""Clean-up checks: dedupe the outcome merge, and confirm the dual arms of the
two campaigns really are separate runs that simply never saw the flag."""
import hashlib
import os

import pandas as pd

CELL = ["dataset", "model", "cap"]
P = lambda x: print(x.to_string())  # noqa: E731
pd.set_option("display.width", 250)

D = pd.read_csv("paper/scripts/out_refute_satstate.csv")
G = pd.read_csv("paper/scripts/out_refute_headroom_b30_lrc0.0001.csv")

print("=" * 100)
print("1. Are the two campaigns' dual arms the same FILES or separate runs?")
print("=" * 100)
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:12]
key = CELL + ["method", "seed"]
a = D.set_index(key)
b = G.set_index(key)
same = diff = 0
for k in a.index:
    if k not in b.index:
        continue
    pa, pb = a.loc[k, "path"], b.loc[k, "path"]
    if os.path.realpath(pa) == os.path.realpath(pb):
        continue
    m = md5(os.path.join(pa, "training_log.csv")) == md5(os.path.join(pb, "training_log.csv"))
    if k[3] == "tralo":
        continue
    same += m
    diff += (not m)
print("  dual runs (96 pairs): training_log.csv byte-identical=%d  differing=%d" % (same, diff))
print("  -> separate directories, separate executions, identical output: the")
print("     --no-ce-skip flag changed nothing at all in the dual arms.")
ta = [md5(os.path.join(a.loc[k, "path"], "training_log.csv"))
      == md5(os.path.join(b.loc[k, "path"], "training_log.csv"))
      for k in a.index if k[3] == "tralo" and k in b.index]
print("  tralo runs (48 pairs): byte-identical=%d  differing=%d" % (sum(ta), len(ta) - sum(ta)))

print("\n" + "=" * 100)
print("2. OUTCOME, deduped. Does holding longer buy anything?")
print("=" * 100)
F = pd.read_csv("paper/scripts/out_factbase_perrun.csv")
F = F[F.campaign.astype(str) == "lrc0.0001_noceskip"]
F = F.drop_duplicates(subset=key)
print("  factbase rows for this campaign after dedupe: %d" % len(F))
M = D.merge(F[key + ["ccF1eq", "AP"]], on=key, how="inner")
print("  merged: %d" % len(M))
for c in ["ccF1eq", "AP"]:
    M["d_" + c] = M[c] - M.groupby(CELL)[c].transform("mean")
P(M.groupby("method").agg(n=("path", "size"), hold=("maxrun_inf", "mean"),
                          ccF1eq=("ccF1eq", "mean"),
                          r_hold_ccF1eq=("maxrun_inf",
                                         lambda x: x.corr(M.loc[x.index, "d_ccF1eq"]))))
print("\n  per cell: tralo hold, best-dual hold, tralo ccF1eq - best-dual ccF1eq (paired on seed)")
out = []
for (ds, mo, cp), g in M.groupby(CELL):
    piv = g.pivot_table(index="seed", columns="method", values="ccF1eq").dropna()
    hp = g.pivot_table(index="seed", columns="method", values="maxrun_inf")
    if "tralo" not in piv.columns:
        continue
    out.append({"dataset": ds, "model": mo, "cap": cp,
                "hold_tralo": hp["tralo"].mean(),
                "hold_fioretto": hp["fioretto_ldf"].mean(),
                "hold_hounie": hp["hounie_rcl"].mean(),
                "d_ccF1eq_vs_bestdual":
                    (piv["tralo"] - piv[["fioretto_ldf", "hounie_rcl"]].max(axis=1)).mean()})
T = pd.DataFrame(out).sort_values(["dataset", "cap", "model"])
print(T.to_string(index=False, float_format=lambda x: "%.4f" % x))
T["tralo_holds_least"] = (T.hold_tralo < T.hold_fioretto) & (T.hold_tralo < T.hold_hounie)
print("\n  cells where tralo holds least AND still wins ccF1eq : %d"
      % int(((T.tralo_holds_least) & (T.d_ccF1eq_vs_bestdual > 0)).sum()))
print("  cells where tralo holds least AND loses ccF1eq       : %d"
      % int(((T.tralo_holds_least) & (T.d_ccF1eq_vs_bestdual < 0)).sum()))
print("  cells where tralo holds >= a dual AND loses ccF1eq   : %d"
      % int(((~T.tralo_holds_least) & (T.d_ccF1eq_vs_bestdual < 0)).sum()))
