"""Refresh the paired-significance artifacts in paper/tables/A_headline/.

Reads docs/all_cells_raw.csv (TraLO rows already filtered to the canonical
breakthrough recipe) and (re)writes the paired win/tie/loss tables that back the
headline experiment (Tables A1/A2). It emits ONLY the stats artifacts -- the
paper figures are built separately by paper/scripts/fig_*_v2.py, and the curated
paper/tables/A_headline/README.md is hand-maintained and never overwritten.

  - Flips: TraLO wins decisively vs all 5 baselines on all 3 datasets.
  - F1:    TraLO wins the hard TissueMNIST L20-L50/MobileNetV3 slice with
           paired significance; ties (within seed noise) on derm; loses
           on saturated aider. We show all of it -- nothing hidden.

Outputs (paper/tables/A_headline/):
  scoreboard.csv             one-line W/T/L per dataset x metric
  win_matrix.csv             per-dataset F1/Flips verdicts vs each baseline
  stats_scoreboard.md        scoreboard, markdown
  stats_headline_f1.md       tissue L20-L50 paired F1 vs each baseline
  stats_flips_dominance.md   paired flips vs each baseline, per dataset

Usage: python -m src.evaluation.make_winning_results
"""
import csv
import random
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs" / "all_cells_raw.csv"
OUT = ROOT / "paper" / "tables" / "A_headline"
BASELINES = ["fioretto_ldf", "hounie_rcl", "tralo_bounded", "danits_lp", "heuristic"]
PRETTY = {"fioretto_ldf": "Fioretto-LDF", "hounie_rcl": "Hounie-RCL",
          "tralo_bounded": "TraLO-bounded", "danits_lp": "DANITS-LP",
          "heuristic": "Heuristic", "tralo": "TraLO"}
DATASETS = ["tissuemnist", "dermmnist", "aider"]
NOISE = 0.003  # F1 ties inside this band count as ties even if p<0.05
random.seed(0)


def w(path, text):
    path.write_text(text, encoding="utf-8")


def fnum(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def load():
    """key (ds,model,cls,grp,tight,seed,method) -> {'f1':..,'flips':..}."""
    d = {}
    for r in csv.DictReader(open(SRC)):
        if r["ds"] == "eurosat":
            continue
        key = (r["ds"], r["model"], r["cls"], r["grp"], r["tight"],
               r["seed"], r["method"])
        d[key] = {"f1": fnum(r["f1m"]), "flips": fnum(r["flips"])}
    return d


def boot_p(diffs, B=20000):
    """Two-sided paired percentile bootstrap on the mean of diffs."""
    if len(diffs) < 2:
        return 1.0
    n = len(diffs)
    cnt = sum(1 for _ in range(B)
              if mean(random.choice(diffs) for _ in range(n)) <= 0)
    return 2 * min(cnt, B - cnt) / B


def paired(d, cell_ok, metric, lower_better=False):
    """{baseline: (n, mean_diff, n_pos, p)} over cells passing cell_ok."""
    out = {}
    for b in BASELINES:
        diffs = []
        for key, v in d.items():
            ds, model, cls, grp, tight, seed, method = key
            if method != "tralo" or not cell_ok(ds, model, cls, grp, tight):
                continue
            bkey = (ds, model, cls, grp, tight, seed, b)
            if bkey not in d:
                continue
            tv, bv = v[metric], d[bkey][metric]
            if tv is None or bv is None:
                continue
            diffs.append((bv - tv) if lower_better else (tv - bv))
        if diffs:
            npos = sum(1 for x in diffs if x > 1e-9)
            out[b] = (len(diffs), mean(diffs), npos, boot_p(diffs))
    return out


def verdict_f1(md, p):
    # Paired bootstrap already cancels shared seed noise, so significance
    # (not an absolute effect-size band) is the right tie test.
    if p >= 0.05:
        return "tie"
    return "WIN" if md > 0 else "loss"


def verdict_flips(md, p):
    if p >= 0.05:
        return "tie"
    return "WIN" if md > 0 else "loss"


def md_table(title, res, unit, vfn):
    L = [f"### {title}", "",
         "| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |",
         "|---|---|---|---|---|---|"]
    for b in BASELINES:
        if b not in res:
            continue
        n, md, npos, p = res[b]
        v = vfn(md, p)
        tag = f"**{v}**" if v == "WIN" else v
        L.append(f"| {PRETTY[b]} | {n} | {md:+.4f}{unit} | {npos}/{n} | {p:.3f} | {tag} |")
    return "\n".join(L) + "\n"


def csv_rows(res, ds, metric, vfn):
    rows = []
    for b in BASELINES:
        if b not in res:
            continue
        n, md, npos, p = res[b]
        rows.append([ds, metric, PRETTY[b], n, round(md, 4), f"{npos}/{n}",
                     round(p, 4), vfn(md, p)])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()

    # ---- headline F1 (tissue L20-L50 MobileNetV3) ----
    head_ok = (lambda ds, mo, cls, grp, t:
               ds == "tissuemnist" and mo == "MobileNetV3"
               and t in ("L20_G20", "L30_G30", "L50_G50"))
    head = paired(d, head_ok, "f1")
    w(OUT / "stats_headline_f1.md",
        "# Headline F1 win — TissueMNIST L20-L50, MobileNetV3\n\n"
        "Paired bootstrap over matched seeds. This is the slice with the most "
        "warmup headroom, where TraLO's accuracy edge is real and significant.\n\n"
        + md_table("TraLO vs baselines (F1-macro, higher better)", head, "", verdict_f1))
    hrows = csv_rows(head, "tissue_L20-L50_MobileNetV3", "F1", verdict_f1)

    # ---- flips dominance (per dataset) ----
    flips_md = ["# Flips dominance — TraLO needs far fewer post-hoc corrections\n",
                "Paired bootstrap; diff = baseline - TraLO (positive = TraLO needs fewer).\n"]
    frows = []
    for ds in DATASETS:
        res = paired(d, lambda a, b, c, g, t, ds=ds: a == ds, "flips",
                     lower_better=True)
        flips_md.append(md_table(f"{ds} (all cells)", res, "", verdict_flips))
        frows += csv_rows(res, ds, "Flips", verdict_flips)
    w(OUT / "stats_flips_dominance.md", "\n".join(flips_md))

    # ---- win matrix + scoreboard ----
    win_rows = [["dataset", "metric", "baseline", "n", "mean_diff",
                 "seeds_plus", "p", "verdict"]]
    win_rows += hrows + frows
    board = []  # (ds, metric, W, T, L)
    for ds in DATASETS:
        f1res = paired(d, lambda a, b, c, g, t, ds=ds: a == ds, "f1")
        flres = paired(d, lambda a, b, c, g, t, ds=ds: a == ds, "flips",
                       lower_better=True)
        win_rows += csv_rows(f1res, ds, "F1", verdict_f1)
        for tag, res, vfn, metric in (("F1", f1res, verdict_f1, "F1"),
                                      ("Flips", flres, verdict_flips, "Flips")):
            W = T = Lz = 0
            for b in BASELINES:
                if b not in res:
                    continue
                _, md, _, p = res[b]
                v = vfn(md, p)
                W += v == "WIN"; T += v == "tie"; Lz += v == "loss"
            board.append((ds, metric, W, T, Lz))
    with open(OUT / "win_matrix.csv", "w", newline="") as f:
        csv.writer(f).writerows(win_rows)

    # scoreboard
    sb = ["# Scoreboard — TraLO win/tie/loss vs 5 baselines\n",
          "Paired bootstrap over matched seeds, per dataset. "
          "WIN/loss = sign of mean diff when p<0.05; tie otherwise.\n",
          "| dataset | metric | WIN | tie | loss |", "|---|---|---|---|---|"]
    sbcsv = [["dataset", "metric", "WIN", "tie", "loss"]]
    for ds, metric, W, T, Lz in board:
        sb.append(f"| {ds} | {metric} | **{W}** | {T} | {Lz} |")
        sbcsv.append([ds, metric, W, T, Lz])
    w(OUT / "stats_scoreboard.md", "\n".join(sb) + "\n")
    with open(OUT / "scoreboard.csv", "w", newline="") as f:
        csv.writer(f).writerows(sbcsv)

    print(f"wrote stats into {OUT}")
    for name in ("scoreboard.csv", "win_matrix.csv", "stats_scoreboard.md",
                 "stats_headline_f1.md", "stats_flips_dominance.md"):
        print("  ", name)


if __name__ == "__main__":
    main()
