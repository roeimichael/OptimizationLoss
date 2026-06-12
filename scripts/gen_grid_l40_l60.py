"""Fill the symmetric tightness grid with L40_G40 and L60_G60 by CLONING existing
L50_G50 warmup=50 configs (guarantees identical HPs to the published grid).

Tightness only affects the constraint phase, not the warmup cache, so the cloned
configs reuse the same base_model_id (cached warmup) -- no warmup retraining.

Targets: active datasets x paper backbones x 6 methods x existing seeds.
Run on the server:  python scripts/gen_grid_l40_l60.py
"""
import copy
import glob
import json
import os

SRC_TAG = "L50_G50"
NEW = [("L40_G40", [0.4, 0.4]), ("L60_G60", [0.6, 0.6])]
BACKBONES = {"MobileNetV3", "MobileNetV2", "RegNetY400MF", "ShuffleNetV2"}
DATASETS = {"tissuemnist", "dermmnist", "aider"}
METHODS = {"tralo", "tralo_bounded", "fioretto_ldf",
           "hounie_rcl", "danits_lp", "heuristic"}
ROOT = "results/pending_runs/grid_l40_l60"
DEGRADED = ("contamination", "cripple", "_weak")


def main():
    seen, cfgs = set(), []
    for p in glob.glob("results/pending_runs/**/config.json", recursive=True):
        if "grid_l40_l60" in p or any(d in p for d in DEGRADED):
            continue
        try:
            c = json.load(open(p))
        except Exception:
            continue
        hp = c.get("hyperparams", {}) or {}
        if c.get("constraint_tag") != SRC_TAG or hp.get("warmup_epochs") != 50:
            continue
        ds, m = c.get("dataset_mode"), c.get("model_name")
        meth, seed = c.get("methodology"), hp.get("seed")
        if ds not in DATASETS or m not in BACKBONES or meth not in METHODS:
            continue
        key = (ds, m, meth, seed)
        if key in seen:
            continue
        seen.add(key)
        for tag, con in NEW:
            nc = copy.deepcopy(c)
            nc["constraint"] = con
            nc["constraint_tag"] = tag
            nc["experiment_path"] = f"{ROOT}/{m}/{ds}/{tag}/{meth}/seed_{seed}"
            nc.pop("results", None)
            nc["status"] = "pending"
            cfgs.append(nc)

    n_written = 0
    for nc in cfgs:
        d = nc["experiment_path"]
        os.makedirs(d, exist_ok=True)
        cp = os.path.join(d, "config.json")
        if os.path.exists(os.path.join(d, "evaluation_metrics.csv")):
            continue
        json.dump(nc, open(cp, "w"), indent=2)
        n_written += 1
    print(f"source cells (L50_G50 w50): {len(seen)}")
    print(f"wrote {n_written} configs -> {ROOT}")
    # breakdown
    from collections import Counter
    by = Counter((nc["dataset_mode"], nc["constraint_tag"]) for nc in cfgs)
    for k in sorted(by):
        print(f"  {k[0]:12s} {k[1]}: {by[k]}")


if __name__ == "__main__":
    main()
