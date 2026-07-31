"""Leave-one-backbone-out CV for the OctMNIST tight-cap win (Sec. 5.1, check i).

Band rule: on the TRAINING backbones, a (cap) level is in the band if the mean paired
gap of TraLO over the per-seed best trained dual clears THRESH in every training backbone.
The band is then tested on the held-out backbone. Rotates over all 3 backbones.
Comparator = per-seed best of {fioretto_ldf, hounie_rcl}, matching Table 1 / Table S6.
"""
import itertools, pandas as pd

CORP = "data/corpus/corpus_final.csv"
BBS = ["MobileNetV3", "RegNetY400MF", "ViTB16"]
THRESH = 0.005


def load():
    d = pd.read_csv(CORP)
    d = d[(d.sweep == "paper_final") & (d.dataset == "octmnist") & (d.constrained_class == 2)]
    # symmetric cells only, tag L{n}_G{n}
    d = d[d.constraint_tag.str.match(r"L(\d+)_G\1$")]
    return d


def cell_gap(d, bb, tag):
    s = d[(d.model == bb) & (d.constraint_tag == tag)]
    t = s[s.method == "tralo"].set_index("seed").cc_f1
    f = s[s.method == "fioretto_ldf"].set_index("seed").cc_f1
    h = s[s.method == "hounie_rcl"].set_index("seed").cc_f1
    if len(t) == 0 or len(f) == 0 or len(h) == 0:
        return None, None
    best = pd.concat([f, h], axis=1).max(axis=1)
    diff = (t - best).dropna()
    return diff.mean(), diff


def main():
    d = load()
    tags = sorted(d.constraint_tag.unique(), key=lambda x: int(x.split("_")[0][1:]))
    rows, pooled = [], []
    for held in BBS:
        train = [b for b in BBS if b != held]
        band = [tg for tg in tags
                if all((cell_gap(d, b, tg)[0] or -9) >= THRESH for b in train)]
        for tg in band:
            m, diff = cell_gap(d, held, tg)
            if m is None:
                continue
            rows.append((held, tg, m, (diff > 0).mean()))
            pooled.extend(diff.tolist())
        print(f"held-out {held:14s} band from {train[0][:6]}+{train[1][:6]}: "
              f"{band if band else 'EMPTY'}")
    print()
    for held, tg, m, wr in rows:
        print(f"  {held:14s} {tg:9s} held-out gap {m:+.4f}  seed-winrate {wr:.2f}")
    n = len(pooled)
    gap = sum(pooled) / n
    wr = sum(x > 0 for x in pooled) / n
    print()
    print(f"POOLED held-out gap = {gap:+.4f}   winrate = {wr:.1%}   "
          f"(n={n} held-out paired runs, thresh={THRESH})")
    print(f"cell-mean of held-out gaps = {sum(r[2] for r in rows)/len(rows):+.4f}")


if __name__ == "__main__":
    main()
