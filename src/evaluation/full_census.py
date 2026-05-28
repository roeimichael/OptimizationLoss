"""Complete, transparent census + aggregation of every experiment.

Goal: account for ALL completed runs, show how raw experiments roll up into
method-comparison points, and give the full win/tie/loss verdict so nothing
is hidden.

Reconciliation buckets (every completed run lands in exactly one):
  DROPPED    — eurosat/so2sat (not in paper)
  ABLATION   — arch_validation, component_ablation, kl_ablation, *smoke
               (not method comparisons)
  NON-CANON  — tralo runs that are not the breakthrough recipe
               (variants that live inside ablation sweeps)
  IN-SCOPE   — the method-comparison grid (tissue/derm/aider, 6 methods)

IN-SCOPE runs group into comparison points keyed by
(ds, model, cls, grp, tight); each point holds up to 6 methods × 4 seeds.

Usage: python -m src.evaluation.full_census
"""
import csv, glob, json, os
from collections import defaultdict
from math import sqrt
from statistics import mean, stdev

RUNS = "results/pending_runs"
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
BASELINES = [m for m in METHODS if m != "tralo"]
DROPPED_DS = {"eurosat", "so2sat"}
ABLATION_SWEEPS = {"arch_validation", "component_ablation", "kl_ablation",
                   "aider_smoke", "expansion_smoke"}


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


def main():
    n_total = 0
    n_dropped = n_ablation = n_noncanon = n_inscope = 0
    inscope = []   # (ds, model, cls, grp, tight, method, seed, f1m, flips, sat)
    for f in glob.glob(os.path.join(RUNS, "*/**/config.json"), recursive=True):
        ev = f.replace("config.json", "evaluation_metrics.csv")
        if not os.path.exists(ev):
            continue
        try:
            c = json.load(open(f))
        except Exception:
            continue
        n_total += 1
        sweep = f.split(RUNS + "/")[1].split("/")[0]
        ds = c.get("dataset_mode")
        method = c.get("methodology")
        hp = c.get("hyperparams", {})
        if ds in DROPPED_DS:
            n_dropped += 1; continue
        if sweep in ABLATION_SWEEPS:
            n_ablation += 1; continue
        if method == "tralo" and not is_canonical_tralo(hp):
            n_noncanon += 1; continue
        if method not in METHODS:
            n_ablation += 1; continue
        m = load_metrics(ev)
        inscope.append((
            ds, c.get("model_name"),
            c.get("dataset_config", {}).get("constrained_class"),
            c.get("dataset_config", {}).get("group_column"),
            c.get("constraint_tag"), method, hp.get("seed"),
            fnum(m.get("F1 (Macro)")), fnum(m.get("Flips Required")),
            fnum(m.get("Raw All Satisfied")),
        ))
        n_inscope += 1

    print("=" * 80)
    print("RECONCILIATION — every completed run accounted for")
    print("=" * 80)
    print(f"  TOTAL completed runs:            {n_total}")
    print(f"   - DROPPED (eurosat/so2sat):     {n_dropped}")
    print(f"   - ABLATION/smoke sweeps:        {n_ablation}")
    print(f"   - NON-CANONICAL tralo variants: {n_noncanon}")
    print(f"   = IN-SCOPE comparison runs:     {n_inscope}")
    assert n_dropped + n_ablation + n_noncanon + n_inscope == n_total

    # de-dup in-scope on full key
    seen = set(); dedup = []
    for r in inscope:
        k = r[:7]
        if k in seen:
            continue
        seen.add(k); dedup.append(r)
    print(f"   (after de-dup on full key:      {len(dedup)})")

    # group into method-stats per (ds,model,cls,grp,tight,method)
    g = defaultdict(lambda: defaultdict(list))
    for (ds, model, cls, grp, tight, method, seed, f1, fl, sat) in dedup:
        key = (ds, model, cls, grp, tight, method)
        if f1 is not None:
            g[key]["f1"].append(f1)
        if fl is not None:
            g[key]["fl"].append(fl)

    points = sorted({k[:5] for k in g})
    print(f"\n  IN-SCOPE comparison points (ds,model,cls,grp,tight): {len(points)}")
    print(f"  avg raw runs per point: {len(dedup)/max(len(points),1):.1f}")

    # full verdict table
    print("\n" + "=" * 80)
    print("FULL PER-POINT VERDICT  (noise-aware: |gap| <= pooled seed std => TIE)")
    print("=" * 80)
    print(f"{'dataset':<12}{'model':<15}{'cls':<4}{'group':<11}{'tight':<9}"
          f"{'#runs':<6}{'tralo_F1':<9}{'best_base':<16}{'gap':<9}{'flips_T/base':<14}verdict")
    ds_roll = defaultdict(lambda: {"win": 0, "tie": 0, "loss": 0,
                                   "gaps": [], "trf": [], "bff": [], "runs": 0})
    for pt in points:
        ds, model, cls, grp, tight = pt
        tr = g.get((*pt, "tralo"))
        if not tr or not tr["f1"]:
            continue
        tr_f1 = mean(tr["f1"]); tr_sd = stdev(tr["f1"]) if len(tr["f1"]) > 1 else 0.0
        tr_fl = mean(tr["fl"]) if tr["fl"] else None
        bases = []
        n_runs = len(tr["f1"])
        for b in BASELINES:
            gb = g.get((*pt, b))
            if gb and gb["f1"]:
                bases.append((mean(gb["f1"]),
                              stdev(gb["f1"]) if len(gb["f1"]) > 1 else 0.0,
                              mean(gb["fl"]) if gb["fl"] else None, b))
                n_runs += len(gb["f1"])
        if not bases:
            continue
        best_f1, best_sd, best_fl, best_m = max(bases, key=lambda x: x[0])
        gap = tr_f1 - best_f1
        pooled = sqrt(tr_sd**2 + best_sd**2)
        if abs(gap) <= max(pooled, 0.003):
            verdict = "TIE"; ds_roll[ds]["tie"] += 1
        elif gap > 0:
            verdict = "WIN"; ds_roll[ds]["win"] += 1
        else:
            verdict = "LOSS"; ds_roll[ds]["loss"] += 1
        ds_roll[ds]["gaps"].append(gap)
        ds_roll[ds]["runs"] += n_runs
        if tr_fl is not None: ds_roll[ds]["trf"].append(tr_fl)
        if best_fl is not None: ds_roll[ds]["bff"].append(best_fl)
        flip_str = f"{tr_fl:.0f}/{best_fl:.0f}" if (tr_fl is not None and best_fl is not None) else "-"
        print(f"{ds:<12}{model:<15}{str(cls):<4}{str(grp):<11}{tight:<9}"
              f"{n_runs:<6}{tr_f1:<9.3f}{best_f1:.3f}({best_m[:7]:<7}) {gap:<+9.3f}"
              f"{flip_str:<14}{verdict}")

    print("\n" + "=" * 80)
    print("ROLLUP BY DATASET")
    print("=" * 80)
    tot_pts = tot_runs = 0
    for ds in sorted(ds_roll):
        d = ds_roll[ds]
        n = d["win"] + d["tie"] + d["loss"]
        tot_pts += n; tot_runs += d["runs"]
        gm = mean(d["gaps"]) if d["gaps"] else 0
        trf = mean(d["trf"]) if d["trf"] else float("nan")
        bff = mean(d["bff"]) if d["bff"] else float("nan")
        print(f"  {ds:<12} points={n:<4} runs={d['runs']:<5} "
              f"F1[W/T/L]={d['win']}/{d['tie']}/{d['loss']}  "
              f"meanF1gap={gm:+.4f}  flips TraLO={trf:.1f} vs base={bff:.1f} "
              f"(saved {bff-trf:+.1f})")
    print(f"\n  TOTAL in-scope comparison points: {tot_pts}  "
          f"backed by {tot_runs} runs")


if __name__ == "__main__":
    main()
