"""Per-epoch soft_4 trajectory: confirm fix variants park near K post-sat."""
import csv
import json
from pathlib import Path

ROOT = Path("results/pending_runs/hybrid_v3")
TIGHT = "L20_G20"
SEED = "seed_1"

CELLS = [
    "baseline_tralo",
    "symquad_resetAdam",
    "symquad_sgd",
    "undershoot_b050_resetAdam",
    "undershoot_b050_sgd",
]


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


print(f"=== Post-sat trajectories  {TIGHT} {SEED}  K=34 ===\n")
print(f"{'Cell':<30} {'Sat':>4} {'soft@sat':>8} {'soft@+1':>8} {'soft@+2':>8} "
      f"{'soft@+3':>8} {'hard@final':>10} {'flips':>6}")
print("-" * 96)
for cell in CELLS:
    exp_dir = ROOT / TIGHT / cell / SEED
    rows = read_csv(exp_dir / "training_log.csv")
    sat = None
    for r in rows:
        if int(r["Global_Satisfied"]) == 1 and int(r["Local_Satisfied"]) == 1:
            sat = int(r["Epoch"])
            break
    if sat is None:
        print(f"{cell:<30} no sat")
        continue
    # Pick rows by epoch
    by_ep = {int(r["Epoch"]): r for r in rows}
    sat_row = by_ep[sat]
    softs = [float(by_ep[e]["Soft_Class4"]) if e in by_ep else None
             for e in [sat, sat+1, sat+2, sat+3]]
    final_row = rows[-1]
    hard_final = int(final_row["Hard_Class4"])
    soft_strs = [f"{s:.2f}" if s is not None else "  -  " for s in softs]
    flips = 34 - hard_final
    print(f"{cell:<30} E{sat:>3} "
          f"{soft_strs[0]:>8} {soft_strs[1]:>8} {soft_strs[2]:>8} "
          f"{soft_strs[3]:>8} {hard_final:>10} {flips:>+6d}")
