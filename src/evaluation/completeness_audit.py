"""Completeness audit — does every planned grid cell have all 4 seeds?

Defines the v2 paper grid (Tables A-E + tissue-backbone spillover) and checks
actual coverage cell-by-cell. Reports per-table:
  - expected (method,tight,...) cells, each needing 4 seeds
  - how many are fully covered (4 seeds)
  - which are partial (<4 seeds) or missing (0)

Usage: python -m src.evaluation.completeness_audit
"""
import glob, json, os
from collections import defaultdict

RUNS = "results/pending_runs"
METHODS = ["tralo", "tralo_bounded", "fioretto_ldf", "hounie_rcl",
           "danits_lp", "heuristic"]
SYM = ["L20_G20", "L30_G30", "L50_G50", "L70_G70", "L80_G80"]
ASYM = [f"L{l}_G{g}" for l in (20, 30, 50, 70, 80) for g in (20, 30, 50, 70, 80)]
SEEDS = {1, 2, 3, 4}


def is_canonical_tralo(hp):
    return (hp.get("hybrid_mode") == "undershoot_hinge"
            and hp.get("reset_optimizer_at_sat") is True
            and hp.get("alpha_kl", 0.0) == 0.0
            and abs(hp.get("fior_beta", 0.0) - 0.5) < 1e-6
            and hp.get("penalty_mode") == "both"
            and hp.get("enable_ce_skip") is True)


def load_seed_map():
    """(ds,model,cls,grp,tight,method) -> set(seeds completed)."""
    seedmap = defaultdict(set)
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
        key = (c.get("dataset_mode"), c.get("model_name"),
               c.get("dataset_config", {}).get("constrained_class"),
               c.get("dataset_config", {}).get("group_column"),
               c.get("constraint_tag"), method)
        seedmap[key].add(hp.get("seed"))
    return seedmap


# (table label, ds, [models], [cls], grp, [tights])
TABLES = [
    ("A.tissue (headline)", "tissuemnist", ["MobileNetV3"], [4], "synth_group", SYM),
    ("A.derm (headline)",   "dermmnist",   ["MobileNetV3"], [4], "loc_group", SYM),
    ("A.aider (headline,FROZEN)", "aider",  ["MobileNetV3"], [0], "synth_group", SYM),
    ("B.derm (asymmetric)", "dermmnist",   ["MobileNetV3"], [4], "loc_group", ASYM),
    ("C.derm (backbones)",  "dermmnist",   ["ResNet18", "EfficientNetB0"], [4], "loc_group", SYM),
    ("D.derm (multi-class)", "dermmnist",  ["MobileNetV3"], [0, 1, 2], "loc_group", SYM),
    ("E.derm (group=sex)",  "dermmnist",   ["MobileNetV3"], [4], "sex", SYM),
    ("F.tissue (backbones,spillover)", "tissuemnist", ["ResNet18", "EfficientNetB0"], [4], "synth_group", SYM),
]


def main():
    sm = load_seed_map()
    print("=" * 78)
    print("COMPLETENESS AUDIT — each cell needs 4 seeds × 6 methods")
    print("=" * 78)
    grand_full = grand_partial = grand_missing = grand_exp = 0
    for label, ds, models, classes, grp, tights in TABLES:
        full = partial = missing = 0
        partial_detail = []
        for model in models:
            for cls in classes:
                for tight in tights:
                    for method in METHODS:
                        key = (ds, model, cls, grp, tight, method)
                        seeds = sm.get(key, set())
                        n = len(seeds & SEEDS)
                        if n >= 4:
                            full += 1
                        elif n == 0:
                            missing += 1
                        else:
                            partial += 1
                            if len(partial_detail) < 6:
                                partial_detail.append(
                                    f"{model}/{cls}/{tight}/{method}={n}seed")
        exp = full + partial + missing
        grand_full += full; grand_partial += partial
        grand_missing += missing; grand_exp += exp
        status = "OK" if (partial == 0 and missing == 0) else "INCOMPLETE"
        print(f"\n{label}: {full}/{exp} cells full  "
              f"[partial {partial}, missing {missing}]  -> {status}")
        if partial_detail:
            print("   partial e.g.: " + ", ".join(partial_detail))

    print("\n" + "=" * 78)
    print(f"GRAND TOTAL: {grand_full}/{grand_exp} cells fully covered "
          f"(partial {grand_partial}, missing {grand_missing})")
    print(f"Implied runs if all full: {grand_exp*4}, currently have "
          f"~{grand_full*4 + grand_partial*2} in-grid")
    print("=" * 78)


if __name__ == "__main__":
    main()
