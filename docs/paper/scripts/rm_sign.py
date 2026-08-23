"""Does 'the two metrics agree in SIGN in every regime' survive?

The claim is stated at the level of a REGIME MEAN -- one number per campaign,
averaged over cells. The project's rule is the opposite: the atomic unit is
(dataset, backbone, cap) over seeds, and summaries COUNT cells.

So: recompute both metrics from my own rescoring of the raw runs, then
  1. reproduce the regime means (and print the two regimes the claim omits)
  2. count the CELLS where the two metrics disagree in sign
  3. do the same for the comparison the paper's headline actually rests on,
     TraLO vs best-of-duals

    python paper/scripts/rm_sign.py
"""
import sys

import numpy as np
import pandas as pd

D = "paper/scripts/"
DUAL = ["fioretto_ldf", "hounie_rcl"]
CLIP = ["heuristic", "danits_lp"]
CELL = ["dataset", "model", "cap"]
EPS = 0.005


def load(files, caps=None, models=None, dedup=True):
    d = pd.concat([pd.read_csv(D + f) for f in files], ignore_index=True)
    if caps:
        d = d[d.cap.isin(caps)]
    if models:
        d = d[d.model.isin(models)]
    return d


def deltas(d, left, right, metrics=("ccF1adj", "ccF1eq", "AP")):
    """Per cell: mean over seeds of (best of `left`) - (best of `right`)."""
    rows = []
    for key, g in d.groupby(CELL):
        r = dict(zip(CELL, key))
        ok = True
        for M in metrics:
            piv = g.pivot_table(index="seed", columns="method", values=M)
            hl = [m for m in left if m in piv.columns]
            hr = [m for m in right if m in piv.columns]
            if not hl or not hr:
                ok = False
                break
            s = piv.dropna(subset=hl + hr)
            if s.empty:
                ok = False
                break
            r["d" + M] = (s[hl].max(axis=1) - s[hr].max(axis=1)).mean()
            r["n"] = len(s)
        if ok:
            rows.append(r)
    return pd.DataFrame(rows)


def sgn(v, eps=EPS):
    return 0 if abs(v) <= eps else (1 if v > 0 else -1)


def report(lbl, t):
    if t.empty:
        print("%-42s EMPTY" % lbl)
        return None
    a, e = t.dccF1adj, t.dccF1eq
    flips = t[[sgn(x) != sgn(y) for x, y in zip(a, e)]]
    hard = t[[sgn(x) * sgn(y) == -1 for x, y in zip(a, e)]]
    print("%-42s cells=%2d  ccF1adj %+0.4f [D%d/C%d]   ccF1eq %+0.4f [D%d/C%d]"
          "   MEAN-SIGN %s   cells where the two metrics disagree: %d "
          "(hard flips D<->C: %d)"
          % (lbl, len(t), a.mean(), int((a > EPS).sum()), int((a < -EPS).sum()),
             e.mean(), int((e > EPS).sum()), int((e < -EPS).sum()),
             "AGREE" if sgn(a.mean(), 0) == sgn(e.mean(), 0) else "*** DISAGREE ***",
             len(flips), len(hard)))
    return t


def main():
    CAPS = ["L30_G30", "L50_G50"]
    MODELS = ["MobileNetV3", "RegNetY400MF"]

    regimes = [
        ("OLD paper_final      (quoted)", ["rm_paperfinal.csv"], None, None),
        ("OLD paper_backbones  (OMITTED)", ["out_paper_backbones.csv"], None, None),
        ("OLD extra_robustness (OMITTED)", ["rm_extra_robustness.csv"], None, None),
        ("OLD warmup_ablation  (quoted)", ["rm_warmup_ablation.csv"], None, None),
        ("OLD w1probe          (quoted)",
         ["rm_warmup1_probe.csv", "rm_warmup1_probe_s34.csv"], None, None),
        ("NEW headroom lrc5e-6", ["rm_headroom_b30.csv"], CAPS, MODELS),
        ("NEW headroom lrc5e-5",
         ["rm_headroom_b30_lrc5e-05.csv", "rm_headroom_b30.csv"], CAPS, MODELS),
        ("NEW headroom noceskip(quoted)",
         ["rm_headroom_b30_lrc0.0001_noceskip.csv", "rm_headroom_b30.csv"],
         CAPS, MODELS),
    ]

    print("=" * 150)
    print("1. DUAL minus CLIPPER, re-derived from the raw runs.  D/C = cells "
          "won by |delta|>0.005")
    print("=" * 150)
    keep = {}
    for lbl, fs, caps, models in regimes:
        try:
            d = load(fs, caps, models)
        except Exception as ex:
            print("%-42s LOAD FAIL %s" % (lbl, ex))
            continue
        t = deltas(d, DUAL, CLIP)
        keep[lbl] = report(lbl, t)

    print()
    print("=" * 150)
    print("2. THE CELLS WHERE THE TWO METRICS DISAGREE IN SIGN")
    print("=" * 150)
    tot = flip = hardf = 0
    for lbl, t in keep.items():
        if t is None:
            continue
        m = [(sgn(x), sgn(y)) for x, y in zip(t.dccF1adj, t.dccF1eq)]
        f = [i for i, (x, y) in enumerate(m) if x != y]
        h = [i for i, (x, y) in enumerate(m) if x * y == -1]
        tot += len(t)
        flip += len(f)
        hardf += len(h)
        if f:
            sub = t.iloc[f][CELL + ["dccF1adj", "dccF1eq"]].copy()
            sub["moves"] = ["dual" if b > a else "clip"
                            for a, b in zip(sub.dccF1adj, sub.dccF1eq)]
            print("\n  %s  (%d of %d cells)" % (lbl, len(f), len(t)))
            print(sub.to_string(index=False, float_format=lambda x: "%+.4f" % x))
    print("\n  TOTAL: %d of %d cells change their verdict when the metric "
          "changes (%.0f%%);  %d are hard D<->C reversals"
          % (flip, tot, 100.0 * flip / max(tot, 1), hardf))

    print()
    print("=" * 150)
    print("3. THE COMPARISON THE PAPER ACTUALLY MAKES: TraLO vs best-of-duals, "
          "headroom noceskip")
    print("=" * 150)
    d = load(["rm_headroom_b30_lrc0.0001_noceskip.csv"], CAPS, MODELS)
    t = deltas(d, ["tralo"], DUAL)
    t = t.sort_values(["dataset", "cap", "model"])
    t["sign_adj"] = [sgn(x) for x in t.dccF1adj]
    t["sign_eq"] = [sgn(x) for x in t.dccF1eq]
    t["AGREE"] = np.where(t.sign_adj == t.sign_eq, "", "  <-- DISAGREE")
    print(t[CELL + ["n", "dccF1adj", "dccF1eq", "AGREE"]]
          .to_string(index=False, float_format=lambda x: "%+.4f" % x))
    for M in ["dccF1adj", "dccF1eq"]:
        v = t[M]
        print("  %s : TraLO WIN %d / LOSS %d / TIE %d cells   (mean %+0.4f)"
              % (M, int((v > EPS).sum()), int((v < -EPS).sum()),
                 int((v.abs() <= EPS).sum()), v.mean()))
    print("  cells whose verdict changes with the metric: %d of %d"
          % (int((t.sign_adj != t.sign_eq).sum()), len(t)))

    print()
    print("  same, TraLO vs best-of-CLIPPERS (noceskip trained arm vs b30 "
          "post-hoc arm):")
    d2 = load(["rm_headroom_b30_lrc0.0001_noceskip.csv", "rm_headroom_b30.csv"],
              CAPS, MODELS)
    d2 = d2[((d2.campaign == "headroom_b30_lrc0.0001_noceskip") & (d2.method == "tralo"))
            | ((d2.campaign == "headroom_b30") & (d2.method.isin(CLIP)))]
    t2 = deltas(d2, ["tralo"], CLIP).sort_values(["dataset", "cap", "model"])
    t2["AGREE"] = np.where([sgn(x) == sgn(y) for x, y in
                            zip(t2.dccF1adj, t2.dccF1eq)], "", "  <-- DISAGREE")
    print(t2[CELL + ["n", "dccF1adj", "dccF1eq", "AGREE"]]
          .to_string(index=False, float_format=lambda x: "%+.4f" % x))
    for M in ["dccF1adj", "dccF1eq"]:
        v = t2[M]
        print("  %s : TraLO WIN %d / LOSS %d / TIE %d cells   (mean %+0.4f)"
              % (M, int((v > EPS).sum()), int((v < -EPS).sum()),
                 int((v.abs() <= EPS).sum()), v.mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
