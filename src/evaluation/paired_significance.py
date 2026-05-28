"""Paired-significance sweep — find every grouping where TraLO wins.

Seeds are paired: TraLO and each baseline share the same (ds, model, cls,
grp, tight, seed) condition, so a paired test cancels between-seed variance.
For each grouping and each baseline we pool the matched per-seed differences
and compute a paired percentile-bootstrap p-value, for both F1-macro
(higher better) and post-hoc Flips (lower better -> we report baseline-tralo).

Groupings reported:
  - per dataset
  - per (dataset, backbone)
  - tissue L20-L50 (the headline winning slice)

Output: docs/paired_significance.md  (+ console)

Usage: python -m src.evaluation.paired_significance
"""
import csv, glob, json, os, random
from collections import defaultdict
from statistics import mean
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results" / "pending_runs"
OUT = ROOT / "docs" / "paired_significance.md"
BASELINES = ["fioretto_ldf", "hounie_rcl", "tralo_bounded", "danits_lp", "heuristic"]
SYM = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
random.seed(0)


def lm(p):
    o = {}
    try:
        for r in csv.reader(open(p)):
            if len(r) == 2:
                o[r[0]] = r[1]
    except Exception:
        pass
    return o


def fn(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def canon(hp):
    return (hp.get("hybrid_mode") == "undershoot_hinge"
            and hp.get("reset_optimizer_at_sat") is True
            and hp.get("alpha_kl", 0.0) == 0.0
            and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
            and hp.get("penalty_mode") == "both"
            and hp.get("enable_ce_skip") is True)


def load():
    """key (ds,model,cls,grp,tight,seed,method) -> (f1, flips)."""
    d = {}
    for f in glob.glob(str(RUNS / "*/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        method = c.get("methodology")
        if method not in (["tralo"] + BASELINES):
            continue
        hp = c.get("hyperparams", {})
        if method == "tralo" and not canon(hp):
            continue
        key = (c.get("dataset_mode"), c.get("model_name"),
               c.get("dataset_config", {}).get("constrained_class"),
               c.get("dataset_config", {}).get("group_column"),
               c.get("constraint_tag"), hp.get("seed"), method)
        if key in d:
            continue
        m = lm(ev)
        d[key] = (fn(m.get("F1 (Macro)")), fn(m.get("Flips Required")))
    return d


def boot_p(diffs, B=20000):
    if not diffs:
        return 1.0
    n = len(diffs)
    cnt = 0
    for _ in range(B):
        s = mean(random.choice(diffs) for _ in range(n))
        if s <= 0:
            cnt += 1
    return 2 * min(cnt, B - cnt) / B


def paired(d, cell_filter, metric_idx, lower_better=False):
    """Return {baseline: (n, mean_diff, n_pos, p)} for cells passing filter."""
    out = {}
    for b in BASELINES:
        diffs = []
        for key, (f1, fl) in d.items():
            ds, model, cls, grp, tight, seed, method = key
            if method != "tralo":
                continue
            if not cell_filter(ds, model, cls, grp, tight):
                continue
            bkey = (ds, model, cls, grp, tight, seed, b)
            if bkey not in d:
                continue
            tv = (f1, fl)[metric_idx]
            bv = d[bkey][metric_idx]
            if tv is None or bv is None:
                continue
            diff = (bv - tv) if lower_better else (tv - bv)
            diffs.append(diff)
        if diffs:
            npos = sum(1 for x in diffs if x > 1e-9)
            out[b] = (len(diffs), mean(diffs), npos, boot_p(diffs))
    return out


def fmt_block(title, res, unit=""):
    lines = [f"### {title}", "",
             "| vs baseline | n | mean diff | seeds + | bootstrap p | verdict |",
             "|---|---|---|---|---|---|"]
    for b in BASELINES:
        if b not in res:
            continue
        n, md, npos, p = res[b]
        sig = "**WIN**" if (md > 0 and p < 0.05) else (
            "loss" if (md < 0 and p < 0.05) else "tie")
        lines.append(f"| {b} | {n} | {md:+.4f}{unit} | {npos}/{n} | {p:.3f} | {sig} |")
    lines.append("")
    return "\n".join(lines)


def main():
    d = load()
    datasets = sorted({k[0] for k in d if k[0] not in ("eurosat", "so2sat")})
    models_by_ds = defaultdict(set)
    for k in d:
        if k[0] not in ("eurosat", "so2sat"):
            models_by_ds[k[0]].add(k[1])

    out = ["# Paired-significance sweep (TraLO vs baselines)\n",
           "Seeds paired by (ds, model, cls, grp, tight). Bootstrap p over "
           "matched per-seed differences. **WIN** = mean favors TraLO AND p<0.05.\n",
           "F1-macro: higher is better. Flips: lower is better (diff = baseline - TraLO).\n"]

    # ---- per dataset, F1 ----
    out.append("\n## F1-macro by dataset\n")
    for ds in datasets:
        res = paired(d, lambda a, b, c, g, t, ds=ds: a == ds, 0)
        out.append(fmt_block(f"{ds} (all cells)", res))

    # ---- per dataset, Flips ----
    out.append("\n## Post-hoc flips by dataset (diff = baseline - TraLO; + = TraLO needs fewer)\n")
    for ds in datasets:
        res = paired(d, lambda a, b, c, g, t, ds=ds: a == ds, 1, lower_better=True)
        out.append(fmt_block(f"{ds} flips", res))

    # ---- per (dataset, backbone), F1 ----
    out.append("\n## F1-macro by dataset x backbone\n")
    for ds in datasets:
        for model in sorted(models_by_ds[ds]):
            res = paired(d, lambda a, b, c, g, t, ds=ds, mo=model:
                         a == ds and b == mo, 0)
            if res:
                out.append(fmt_block(f"{ds} / {model}", res))

    # ---- headline slice: tissue L20-L50 ----
    out.append("\n## Headline slice: TissueMNIST L20-L50 (MobileNetV3)\n")
    res = paired(d, lambda a, b, c, g, t:
                 a == "tissuemnist" and b == "MobileNetV3"
                 and t in ("L20_G20", "L30_G30", "L50_G50"), 0)
    out.append(fmt_block("tissue L20-L50 F1", res))

    text = "\n".join(out)
    OUT.write_text(text)
    print(text)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
