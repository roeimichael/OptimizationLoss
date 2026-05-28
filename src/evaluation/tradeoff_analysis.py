"""TraLO F1-vs-flips tradeoff analysis, pooled by dataset.

Scans ALL completed cells under results/pending_runs/, applies the
canonical-TraLO filter, and for each (model, cls, grp, tight) cell:
  - computes TraLO mean+std F1m across seeds
  - computes best-baseline mean+std F1m
  - flags the cell as WIN / TIE / LOSS using a noise-aware threshold:
    a gap is a real win/loss only if |gap| exceeds the pooled seed std
    of the two methods; otherwise it is a statistical TIE.
  - records flip reduction (best-baseline flips - TraLO flips)

Then reports, per dataset:
  - N cells, win/tie/loss counts
  - mean F1m gap (TraLO - best baseline) with std across cells
  - mean flips: TraLO vs best baseline, and flips saved
  - the tradeoff: flips saved per 0.01 F1m conceded (when TraLO trails)
"""
import csv, glob, json, os
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev

METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
BASELINES = [m for m in METHODS if m != "tralo"]
RUNS = "results/pending_runs"


def load_metrics(p):
    out = {}
    try:
        with open(p) as f:
            for r in csv.reader(f):
                if len(r) == 2:
                    out[r[0]] = r[1]
    except Exception:
        pass
    return out


def fnum(v):
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def is_canonical_tralo(hp):
    return (hp.get("hybrid_mode") == "undershoot_hinge"
            and hp.get("reset_optimizer_at_sat") is True
            and hp.get("alpha_kl", 0.0) == 0.0
            and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
            and hp.get("penalty_mode") == "both"
            and hp.get("enable_ce_skip") is True)


def collect():
    rows = []
    seen = set()
    for f in glob.glob(os.path.join(RUNS, "*/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        method = c.get("methodology")
        if method not in METHODS:
            continue
        hp = c.get("hyperparams", {})
        if method == "tralo" and not is_canonical_tralo(hp):
            continue
        seed = hp.get("seed")
        ds = c.get("dataset_mode")
        key = (ds, c.get("model_name"),
               c.get("dataset_config", {}).get("constrained_class"),
               c.get("dataset_config", {}).get("group_column"),
               c.get("constraint_tag"), method, seed)
        if key in seen:
            continue
        seen.add(key)
        m = load_metrics(ev)
        rows.append({
            "ds": ds, "model": key[1], "cls": key[2], "grp": key[3],
            "tight": key[4], "method": method, "seed": seed,
            "f1m": fnum(m.get("F1 (Macro)")),
            "flips": fnum(m.get("Flips Required")),
        })
    return rows


def per_method_stats(rows):
    """(ds,model,cls,grp,tight,method) -> (f1_mean, f1_std, flips_mean, n)."""
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["ds"], r["model"], r["cls"], r["grp"], r["tight"], r["method"])
        if r["f1m"] is not None:
            g[k]["f1"].append(r["f1m"])
        if r["flips"] is not None:
            g[k]["flips"].append(r["flips"])
    out = {}
    for k, d in g.items():
        f1 = d["f1"]
        out[k] = (
            mean(f1) if f1 else None,
            stdev(f1) if len(f1) > 1 else 0.0,
            mean(d["flips"]) if d["flips"] else None,
            len(f1),
        )
    return out


def analyze():
    rows = collect()
    stats = per_method_stats(rows)
    # comparison cells = unique (ds,model,cls,grp,tight)
    cells = sorted({k[:5] for k in stats})

    per_ds = defaultdict(lambda: {
        "win": 0, "tie": 0, "loss": 0,
        "gaps": [], "tr_flips": [], "base_flips": [],
    })

    for cell in cells:
        tr = stats.get((*cell, "tralo"))
        if not tr or tr[0] is None:
            continue
        tr_f1, tr_std, tr_flips, _ = tr
        base = [(stats[(*cell, b)][0], stats[(*cell, b)][1],
                 stats[(*cell, b)][2], b)
                for b in BASELINES
                if (*cell, b) in stats and stats[(*cell, b)][0] is not None]
        if not base:
            continue
        best_f1, best_std, best_flips, best_m = max(base, key=lambda x: x[0])
        gap = tr_f1 - best_f1
        # noise-aware: pooled std of the two methods
        pooled = sqrt(tr_std ** 2 + best_std ** 2) or 0.0
        ds = cell[0]
        d = per_ds[ds]
        d["gaps"].append(gap)
        if tr_flips is not None:
            d["tr_flips"].append(tr_flips)
        if best_flips is not None:
            d["base_flips"].append(best_flips)
        if abs(gap) <= max(pooled, 0.003):
            d["tie"] += 1
        elif gap > 0:
            d["win"] += 1
        else:
            d["loss"] += 1

    print("=" * 80)
    print("TraLO vs best-baseline — F1 / flips tradeoff, pooled by dataset")
    print("(noise-aware: gap within pooled seed-std => statistical TIE)")
    print("=" * 80)
    for ds in sorted(per_ds):
        d = per_ds[ds]
        n = d["win"] + d["tie"] + d["loss"]
        if not n:
            continue
        gmean = mean(d["gaps"])
        gstd = stdev(d["gaps"]) if len(d["gaps"]) > 1 else 0.0
        trf = mean(d["tr_flips"]) if d["tr_flips"] else float("nan")
        bff = mean(d["base_flips"]) if d["base_flips"] else float("nan")
        print(f"\n## {ds}  ({n} cells)")
        print(f"  F1m: WIN {d['win']}  TIE {d['tie']}  LOSS {d['loss']}")
        print(f"  mean F1m gap (TraLO - best baseline): {gmean:+.4f} "
              f"(std across cells {gstd:.4f})")
        print(f"  mean flips: TraLO {trf:.1f}  vs  best baseline {bff:.1f}  "
              f"(saved {bff-trf:+.1f})")
        if gmean < 0 and (bff - trf) > 0:
            per_pt = (bff - trf) / (abs(gmean) / 0.01) if gmean != 0 else float("inf")
            print(f"  tradeoff: gives up {abs(gmean)*100:.2f} F1m points on average, "
                  f"saves {bff-trf:.1f} flips  => ~{per_pt:.1f} flips saved per 0.01 F1m conceded")
        elif gmean >= 0:
            print(f"  TraLO is on-average ahead or even on F1m AND saves "
                  f"{bff-trf:+.1f} flips — strict improvement, no tradeoff")
    print()


if __name__ == "__main__":
    analyze()
