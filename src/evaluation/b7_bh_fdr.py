"""B7: Benjamini-Hochberg FDR correction over the family of paired tests.

The paper makes MANY paired comparisons (cells x baselines x metrics). A reviewer
will (correctly) flag multiple-comparison inflation: with dozens of tests at
alpha=0.05 some "wins" appear by chance. This script builds the BH-FDR layer on
top of the SAME paired-bootstrap p-value the headline tables use
(src/evaluation/make_winning_results.py :: boot_p), then reports which wins --
and specifically which HEADLINE claims -- survive correction.

HONESTY-FIRST (matches project house rules):
  * Matched-seed PAIRED tests only. Diff series is paired within a cell on
    (cls,grp,seed); TraLO vs baseline share every nuisance factor.
  * "cell" = (ds, model, tight). Atomic averaging is over SEED (and, within a
    cell, over the cell's constraint configs cls/grp -- e.g. dermmnist has 5).
    We NEVER pool diffs across datasets, backbones, or cap levels (tight).
  * Two-sided bootstrap p (sign tracked separately). A "win" = p_raw < 0.05 with
    positive mean diff (TraLO better). BH is applied to the raw p-values.

Families (BH applied independently within each):
  (a) per (dataset x metric)  -- the scope a reader uses per results panel
  (b) GLOBAL                  -- every (cell,baseline,metric) test at once
Both reported, at q in {0.05, 0.10}.

Metric conventions (identical to make_winning_results.py):
  f1  : macro-F1, higher better; diff = tralo.f1m - baseline.f1m
  flips: post-hoc flips, lower better; diff = baseline.flips - tralo.flips
         (positive => TraLO needs fewer flips => TraLO better)

Usage:
  python -m src.evaluation.b7_bh_fdr [--corpus docs/all_cells_raw.csv]
                                     [--B 20000] [--seed 12345]
Outputs (results/stats_trackb/b7/):
  b7_fdr_tests.csv     tidy per-test rows at both family scopes
  b7_fdr_summary.md    what survives correction, incl. headline claims
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ["fioretto_ldf", "hounie_rcl", "tralo_bounded", "danits_lp", "heuristic"]
PRETTY = {"fioretto_ldf": "Fioretto-LDF", "hounie_rcl": "Hounie-RCL",
          "tralo_bounded": "TraLO-bounded", "danits_lp": "DANITS-LP",
          "heuristic": "Heuristic"}
# (metric-key-in-record, human label, lower_better)
METRICS = [("f1", "f1m", False), ("flips", "flips", True)]
Q_LEVELS = [0.05, 0.10]


def fnum(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def load(src):
    """key (ds,model,cls,grp,tight,seed,method) -> {'f1':.., 'flips':..}."""
    d = {}
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            if r["ds"] == "eurosat":
                continue
            key = (r["ds"], r["model"], r["cls"], r["grp"], r["tight"],
                   r["seed"], r["method"])
            d[key] = {"f1": fnum(r["f1m"]), "flips": fnum(r["flips"])}
    return d


def boot_p(diffs, rng, B):
    """Two-sided paired percentile bootstrap on the mean of diffs.

    Identical definition to make_winning_results.boot_p; RNG passed in so the
    whole run is reproducible from a single seed.
    """
    if len(diffs) < 2:
        return 1.0
    n = len(diffs)
    cnt = sum(1 for _ in range(B)
              if mean(rng.choice(diffs) for _ in range(n)) <= 0)
    return 2 * min(cnt, B - cnt) / B


def build_tests(d, rng, B):
    """One paired test per (cell=(ds,model,tight), baseline, metric).

    Returns list of dicts. Diffs are matched-seed paired within the cell on
    (cls,grp,seed); positive diff = TraLO better for BOTH metrics (flips uses
    baseline - tralo). Pooling stays inside a single cell -- never across cells.
    """
    # index tralo rows by cell -> list of (cls,grp,seed, rec)
    cells = defaultdict(list)
    for key, v in d.items():
        ds, model, cls, grp, tight, seed, method = key
        if method != "tralo":
            continue
        cells[(ds, model, tight)].append((cls, grp, seed, v))

    tests = []
    for (ds, model, tight), tralo_rows in sorted(cells.items()):
        for b in BASELINES:
            for mkey, mlabel, lower in METRICS:
                diffs = []
                for cls, grp, seed, tv in tralo_rows:
                    bkey = (ds, model, cls, grp, tight, seed, b)
                    bv = d.get(bkey)
                    if bv is None:
                        continue
                    t, base = tv[mkey], bv[mkey]
                    if t is None or base is None:
                        continue
                    diffs.append((base - t) if lower else (t - base))
                if not diffs:
                    continue
                md = mean(diffs)
                p = boot_p(diffs, rng, B)
                tests.append({
                    "ds": ds, "model": model, "tight": tight, "baseline": b,
                    "metric": mlabel, "n": len(diffs), "mean_diff": md,
                    "sign": 1 if md > 0 else (-1 if md < 0 else 0),
                    "p_raw": p,
                    "win_raw": p < 0.05 and md > 0,
                })
    return tests


def bh_reject(pvals, q):
    """Benjamini-Hochberg step-up. Returns list[bool] aligned to pvals.

    Largest k with p_(k) <= k/m * q; reject all ranks <= k. Monotone by
    construction (a rejected test implies every smaller-p test is rejected).
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= rank / m * q:
            kmax = rank
    reject = [False] * m
    for rank, i in enumerate(order, start=1):
        if rank <= kmax:
            reject[i] = True
    return reject


def apply_family(tests, keyfn):
    """Attach bh reject flags for one scope. Mutates tests: sets
    bh@q keys namespaced by the scope label returned in the row 'q_scope'."""
    groups = defaultdict(list)
    for i, t in enumerate(tests):
        groups[keyfn(t)].append(i)
    flags = {}  # test-index -> {q: bool}
    for _, idxs in groups.items():
        ps = [tests[i]["p_raw"] for i in idxs]
        for q in Q_LEVELS:
            rej = bh_reject(ps, q)
            for j, i in enumerate(idxs):
                flags.setdefault(i, {})[q] = rej[j]
    return flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(ROOT / "docs" / "all_cells_raw.csv"))
    ap.add_argument("--B", type=int, default=20000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=12345, help="fixed RNG seed")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "stats_trackb" / "b7"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    d = load(args.corpus)
    tests = build_tests(d, rng, args.B)

    # Two family scopes.
    scope_a = apply_family(tests, lambda t: (t["ds"], t["metric"]))   # per ds x metric
    scope_g = apply_family(tests, lambda t: "GLOBAL")                 # global

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- tidy CSV: each test emitted once per scope ----
    csv_path = outdir / "b7_fdr_tests.csv"
    with open(csv_path, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["ds", "model", "tight", "baseline", "metric", "n",
                      "mean_diff", "p_raw", "sign", "win_raw", "q_scope",
                      "bh_reject@0.05", "bh_reject@0.10"])
        for i, t in enumerate(tests):
            for scope_label, flags in (("dataset_x_metric", scope_a),
                                       ("global", scope_g)):
                wtr.writerow([
                    t["ds"], t["model"], t["tight"], PRETTY[t["baseline"]],
                    t["metric"], t["n"], f"{t['mean_diff']:+.5f}",
                    f"{t['p_raw']:.4f}", t["sign"], int(t["win_raw"]),
                    scope_label, int(flags[i][0.05]), int(flags[i][0.10]),
                ])

    # ---- summary stats ----
    n_tests = len(tests)
    wins = [i for i, t in enumerate(tests) if t["win_raw"]]

    def survive_count(flags, q, subset=None):
        idxs = subset if subset is not None else range(len(tests))
        return sum(1 for i in idxs if tests[i]["win_raw"] and flags[i][q])

    # headline subsets (among raw wins)
    tissue_f1_wins = [i for i in wins
                      if tests[i]["ds"] == "tissuemnist" and tests[i]["metric"] == "f1m"]
    flips_wins = [i for i in wins if tests[i]["metric"] == "flips"]

    lines = []
    A = lines.append
    A("# B7 - Benjamini-Hochberg FDR correction over paired tests\n")
    A(f"Corpus: `{args.corpus}` (eurosat excluded). "
      f"Bootstrap B={args.B}, RNG seed={args.seed}.\n")
    A("Paired matched-seed bootstrap p-values (same definition as "
      "`make_winning_results.boot_p`). A **raw win** = two-sided `p_raw < 0.05` "
      "with positive mean diff (TraLO better). BH applied to raw p within each "
      "family; a claim *survives* if its raw win is still rejected (declared "
      "significant) after correction.\n")
    A(f"- Total paired tests (cell x baseline x metric): **{n_tests}**")
    A(f"- Raw wins (p<0.05, TraLO better): **{len(wins)}** "
      f"({len(flips_wins)} flips, {len(wins) - len(flips_wins)} F1)\n")

    A("## Wins surviving FDR\n")
    A("| family scope | q | wins surviving / raw wins |")
    A("|---|---|---|")
    for label, flags in (("per (dataset x metric)", scope_a), ("GLOBAL", scope_g)):
        for q in Q_LEVELS:
            A(f"| {label} | {q:.2f} | {survive_count(flags, q, wins)} / {len(wins)} |")
    A("")

    A("## Headline claim: TissueMNIST F1 win\n")
    if not tissue_f1_wins:
        A("No raw F1 wins on TissueMNIST in this corpus cut.\n")
    else:
        A("| cell | baseline | mean diff | p_raw | dsxmetric q05 | dsxmetric q10 | global q05 | global q10 |")
        A("|---|---|---|---|---|---|---|---|")
        for i in tissue_f1_wins:
            t = tests[i]
            A(f"| {t['model']}/{t['tight']} | {PRETTY[t['baseline']]} | "
              f"{t['mean_diff']:+.4f} | {t['p_raw']:.3f} | "
              f"{'Y' if scope_a[i][0.05] else 'n'} | {'Y' if scope_a[i][0.10] else 'n'} | "
              f"{'Y' if scope_g[i][0.05] else 'n'} | {'Y' if scope_g[i][0.10] else 'n'} |")
        A("")

    A("## Headline claim: Flips dominance\n")
    A(f"Raw flips wins: **{len(flips_wins)}**. Surviving FDR:\n")
    A("| family scope | q | flips wins surviving / raw |")
    A("|---|---|---|")
    for label, flags in (("per (dataset x metric)", scope_a), ("GLOBAL", scope_g)):
        for q in Q_LEVELS:
            A(f"| {label} | {q:.2f} | {survive_count(flags, q, flips_wins)} / {len(flips_wins)} |")
    A("")
    # largest-effect flips win sanity line
    if flips_wins:
        top = max(flips_wins, key=lambda i: tests[i]["mean_diff"])
        t = tests[top]
        A(f"Largest-effect flips win: {t['ds']}/{t['model']}/{t['tight']} vs "
          f"{PRETTY[t['baseline']]}, mean diff {t['mean_diff']:+.2f} flips, "
          f"p={t['p_raw']:.3f} -- survives: dsxmetric q05="
          f"{'Y' if scope_a[top][0.05] else 'n'}, global q05="
          f"{'Y' if scope_g[top][0.05] else 'n'}.\n")

    A("## Notes\n")
    A("- BH is monotone: within any family, a rejected test implies every "
      "smaller-p test in that family is also rejected.")
    A("- Two-sided p is used throughout; direction (sign) is tracked separately "
      "so a 'win' requires both significance and TraLO-favorable sign.")
    A("- dermmnist cells aggregate their 5 (cls,grp) constraint configs at a "
      "fixed cap; no aggregation crosses datasets, backbones, or cap levels.")

    md_path = outdir / "b7_fdr_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"tests={n_tests} raw_wins={len(wins)} "
          f"(flips={len(flips_wins)}, f1={len(wins) - len(flips_wins)})")
    for label, flags in (("dsxmetric", scope_a), ("global", scope_g)):
        for q in Q_LEVELS:
            print(f"  survive {label} q={q}: {survive_count(flags, q, wins)}/{len(wins)}")


if __name__ == "__main__":
    main()
