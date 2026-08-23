import pandas as pd
pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400)
F = pd.read_csv("paper/scripts/out_factbase.csv")
R = pd.read_csv("paper/scripts/out_factbase_perrun.csv")
ff = lambda x: "%+.4f" % x

print("=" * 130)
print("TRALO ROWS: both adjudications, per cell, per campaign")
print("=" * 130)
t = F[F.method == "tralo"][["campaign", "dataset", "model", "cap", "n_seeds",
                           "ccF1eq", "d_vs_bestdual", "d_vs_clip", "AP",
                           "macroEq", "count_raw", "K", "n_collapsed",
                           "last_epoch", "sat"]]
print(t.to_string(index=False, float_format=lambda x: "%.4f" % x))

print("\n" + "=" * 130)
print("CELL-COUNT SUMMARY (count cells, never average them). |delta|>0.005 = decisive")
print("=" * 130)
for camp, g in t.groupby("campaign"):
    for lbl in ["d_vs_bestdual", "d_vs_clip"]:
        v = g[lbl].dropna()
        print("  %-24s %-14s  W %2d  L %2d  T %2d  (of %2d)"
              % (camp, lbl, int((v > 0.005).sum()), int((v < -0.005).sum()),
                 int((v.abs() <= 0.005).sum()), len(v)))

print("\n" + "=" * 130)
print("EPOCHS ACTUALLY RUN (max of the Epoch/epoch column, NOT len(df))")
print("=" * 130)
print(R.groupby(["campaign", "method"])["last_epoch"]
      .agg(["mean", "min", "max"]).round(2).to_string())

print("\n" + "=" * 130)
print("DETERMINISM / CE-SKIP: is lrc0.0001 identical to _noceskip cell by cell?")
print("=" * 130)
a = F[F.campaign == "lrc0.0001"].set_index(["dataset", "model", "cap", "method"])
b = F[F.campaign == "lrc0.0001_noceskip"].set_index(["dataset", "model", "cap", "method"])
j = a[["ccF1eq", "count_raw"]].join(b[["ccF1eq", "count_raw"]], lsuffix="_skipON",
                                    rsuffix="_skipOFF")
j["dccF1eq"] = j.ccF1eq_skipOFF - j.ccF1eq_skipON
j["identical"] = j.dccF1eq.abs() < 1e-12
print(j.to_string(float_format=lambda x: "%.4f" % x))

print("\n" + "=" * 130)
print("CLIPPER ARMS: heuristic vs danits_lp under budget equalization")
print("=" * 130)
c = F[F.method.isin(["heuristic", "danits_lp"])].pivot_table(
    index=["dataset", "model", "cap"], columns="method",
    values=["ccF1eq", "AP", "macroEq", "count_raw", "count_adj"])
print(c.round(4).to_string())
