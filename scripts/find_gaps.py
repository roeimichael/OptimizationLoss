"""Find which warmup=1/3 TraLO push-pull cells lack baseline counterparts.

For every (sweep, ds, model, tag, cls, seed) where TraLO has fsat < 0.85,
list which baselines (heuristic, danits_lp, fioretto_ldf, hounie_rcl) exist or
are missing.
"""
import csv
from collections import defaultdict

CSV_PATH = "scripts/_audit/saturation_audit_v2.csv"

BASELINES = ["fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]


def main():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                r["fsat_f"] = (float(r["frac_train_acc_sat"])
                               if r["frac_train_acc_sat"] else None)
            except ValueError:
                r["fsat_f"] = None
            try:
                r["warm_int"] = int(r["warmup_epochs"]) if r["warmup_epochs"] else None
            except ValueError:
                r["warm_int"] = None
            try:
                r["f1"] = float(r["f1_macro"]) if r["f1_macro"] else None
            except ValueError:
                r["f1"] = None
            rows.append(r)

    # Build method-presence index per (sweep, ds, model, tag, cls)
    cell_methods = defaultdict(set)
    cell_tralo_info = {}
    cell_f1 = defaultdict(dict)  # (cell_key, method) -> f1
    for r in rows:
        if not r["dataset"]:
            continue
        key = (r["sweep"], r["dataset"], r["model"], r["constraint_tag"],
               r["constrained_class"])
        cell_methods[key].add(r["method"])
        cell_f1[(key, r["method"])][r["seed"]] = r["f1"]
        if r["method"] == "tralo" and r["fsat_f"] is not None:
            if key not in cell_tralo_info:
                cell_tralo_info[key] = {
                    "fsats": [], "f1s": [], "warm": r["warm_int"],
                }
            cell_tralo_info[key]["fsats"].append(r["fsat_f"])
            if r["f1"] is not None:
                cell_tralo_info[key]["f1s"].append(r["f1"])

    # Filter: TraLO non-saturated (mean fsat < 0.85), pretrained-friendly datasets
    print("=== Non-saturated TraLO cells: which baselines exist/missing ===")
    print(f"{'sweep':28s} {'ds':12s} {'model':16s} {'tag':9s} {'cls':>3s} "
          f"{'warm':>5s} {'fsat':>5s} {'F1':>6s} {'fior':>4s} {'hou':>4s} "
          f"{'dan':>4s} {'heu':>4s}")
    print("-" * 130)

    cells_to_fill = []
    cells_all_present = []
    interesting_ds = ["dermmnist", "aider", "tissuemnist", "octmnist",
                      "bloodmnist", "retinamnist"]
    for key, info in sorted(cell_tralo_info.items()):
        if not info["fsats"]:
            continue
        mean_fsat = sum(info["fsats"]) / len(info["fsats"])
        if mean_fsat >= 0.85:
            continue
        sweep, ds, mdl, tag, cls = key
        if ds not in interesting_ds:
            continue
        if mdl == "MobileNetV3":
            continue
        methods = cell_methods[key]
        f1_mean = sum(info["f1s"]) / len(info["f1s"]) if info["f1s"] else 0
        cells = []
        for bl in BASELINES:
            mark = "[X]" if bl in methods else "[ ]"
            cells.append(mark)
        # Count missing
        missing = [bl for bl in BASELINES if bl not in methods]
        print(f"{sweep[:28]:28s} {ds:12s} {mdl:16s} {tag:9s} {cls:>3s} "
              f"{str(info['warm']):>5s} {mean_fsat:>5.2f} {f1_mean:>6.4f} "
              f"{cells[0]:>4s} {cells[1]:>4s} {cells[2]:>4s} {cells[3]:>4s}")
        if missing:
            cells_to_fill.append((key, missing, info))
        else:
            cells_all_present.append((key, info))

    print(f"\n=== Cells with ALL 5 methods present: {len(cells_all_present)} ===")
    for key, info in cells_all_present:
        sweep, ds, mdl, tag, cls = key
        mean_fsat = sum(info["fsats"]) / len(info["fsats"])
        print(f"  {sweep} | {ds} {mdl} {tag} cls{cls} | "
              f"warm={info['warm']} fsat={mean_fsat:.2f} n={len(info['fsats'])}")

    print(f"\n=== Cells with MISSING baselines: {len(cells_to_fill)} ===")
    for key, missing, info in cells_to_fill:
        sweep, ds, mdl, tag, cls = key
        print(f"  {sweep} | {ds} {mdl} {tag} cls{cls} | "
              f"warm={info['warm']} | MISSING: {missing}")


if __name__ == "__main__":
    main()
