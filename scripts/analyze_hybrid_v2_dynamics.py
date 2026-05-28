"""Per-epoch dynamics: baseline_tralo vs symquad vs undershoot variants.

Pulls Soft/Hard class 4 + lambda + penalty values across the constraint
phase and reports a snapshot table at: warmup_end, first_sat, +5ep, +10ep, final.
Also reports max/min soft_4 observed AFTER first satisfaction (drift envelope).
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path("results/pending_runs/hybrid_v2")
TIGHT = "L20_G20"
SEED = "seed_1"

CELLS = [
    "baseline_tralo",
    "symquad",
    "undershoot_b020",
    "undershoot_b050",
    "undershoot_b100",
]


def read_train_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def parse_log(log_path, exp_substring):
    out = {}
    if not log_path.exists():
        return out
    pat = re.compile(
        r"src\.methodologies\.tralo_fioretto\.train INFO Epoch (\d+) \[.*?\] "
        r"ce=([\d.]+) bounded=([\d.]+) fior=([\d.]+) pen=([\d.]+) "
        r"lam_T=([\d.]+)"
    )
    in_block = False
    with open(log_path) as f:
        for line in f:
            if exp_substring in line and "Running results" in line:
                in_block = True
                continue
            if in_block and "Training complete" in line:
                in_block = False
                continue
            if not in_block:
                continue
            m = pat.search(line)
            if m:
                ep = int(m.group(1))
                out[ep] = {
                    "ce": float(m.group(2)),
                    "bounded": float(m.group(3)),
                    "fior": float(m.group(4)),
                    "pen": float(m.group(5)),
                    "lam_T": float(m.group(6)),
                }
    return out


def snapshot(rows, ep):
    for r in rows:
        if int(r["Epoch"]) == ep:
            return r
    return None


def first_sat(rows):
    for r in rows:
        if int(r["Global_Satisfied"]) == 1 and int(r["Local_Satisfied"]) == 1:
            return int(r["Epoch"])
    return None


log_path = Path("logs/hybrid_v2.log")
print(f"=== {TIGHT} {SEED}  K_global=34, K_local=17 ===\n")

for cell in CELLS:
    exp_dir = ROOT / TIGHT / cell / SEED
    cfg = json.load(open(exp_dir / "config.json"))
    rows = read_train_csv(exp_dir / "training_log.csv")
    sat = first_sat(rows)
    final = int(rows[-1]["Epoch"])
    early = 51
    mid = (sat + 5) if sat else early
    late = (sat + 10) if sat else early

    fior_data = {}
    if cfg["methodology"] == "tralo_fioretto":
        fior_data = parse_log(log_path, f"hybrid_v2/{TIGHT}/{cell}/{SEED}")

    print(f">>> {cell}  (method={cfg['methodology']}, "
          f"mode={cfg['hyperparams'].get('hybrid_mode', '-')})  "
          f"first_sat=E{sat} final=E{final}")

    hdr = (f"  {'epoch':>8} {'soft_4':>7} {'hard_4':>6} {'lam_T':>6}")
    if fior_data:
        hdr += f" {'bnded':>6} {'pen':>6}"
    print(hdr)
    print("  " + "-" * (60 if not fior_data else 78))
    for label, ep in [("warmup+1", early),
                       ("first_sat", sat),
                       ("+5ep", mid),
                       ("+10ep", late),
                       ("final", final)]:
        if ep is None:
            continue
        r = snapshot(rows, ep)
        if r is None:
            continue
        soft = float(r["Soft_Class4"])
        hard = int(r["Hard_Class4"])
        lamt = float(r["Lambda_Global"])
        line = f"  {label:<5} E{ep:<3} {soft:>7.2f} {hard:>6d} {lamt:>6.3f}"
        if fior_data:
            fd = fior_data.get(ep, {})
            line += f" {fd.get('bounded', 0):>6.3f} {fd.get('pen', 0):>6.3f}"
        print(line)

    # Post-sat drift envelope
    if sat is not None:
        post = [r for r in rows if int(r["Epoch"]) >= sat]
        softs = [float(r["Soft_Class4"]) for r in post]
        hards = [int(r["Hard_Class4"]) for r in post]
        s_min, s_max = min(softs), max(softs)
        h_min, h_max = min(hards), max(hards)
        print(f"  POST-SAT envelope ({len(post)} epochs): "
              f"soft_4 in [{s_min:.2f}, {s_max:.2f}]  "
              f"hard_4 in [{h_min}, {h_max}]  "
              f"final hard={hards[-1]} (K=34, posthoc flips +{34 - hards[-1]})")
    print()
