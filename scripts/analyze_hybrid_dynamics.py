"""Loss-component analysis: baseline TraLO vs hybrid variants.

Compares per-epoch trajectories of the constrained class:
  - soft_count_4 / hard_count_4 / K  (distance from boundary)
  - L_Global (bounded penalty value, post-lambda scaling)
  - Lambda_Global (lambda_T_mean)

For hybrid runs additionally parses the .log file to extract:
  - fior_loss (Fior linear contribution to chunk_loss)
  - lambda_F (dual_lambda mode only)

Reports a snapshot table at: warmup_end, first_sat, +10ep after sat, final.
"""
import csv
import json
import re
from pathlib import Path

ROOT = Path("results/pending_runs/hybrid_v1")
TIGHT = "L20_G20"
SEED = "seed_1"

CELLS = [
    ("baseline_tralo", "tralo"),
    ("hybrid_singleL_beta005", "single_low"),
    ("hybrid_singleL_beta020", "single_mid"),
    ("hybrid_singleL_beta050", "single_high"),
    ("hybrid_dualL_step005", "dual_mid"),
    ("hybrid_dualL_step020", "dual_aggressive"),
]


def read_train_csv(p):
    rows = []
    with open(p) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows


def parse_log_for_cell(log_path, exp_dir_substring):
    """Pull per-epoch (epoch, bounded, fior, lam_T, lam_F) for one cell from
    the dispatcher log by searching for that experiment's running prefix."""
    out = {}
    if not log_path.exists():
        return out
    pattern = re.compile(
        r"src\.methodologies\.tralo_fioretto\.train INFO Epoch (\d+) \[.*?\] "
        r"ce=([\d.]+) bounded=([\d.]+) fior=([\d.]+) "
        r"lam_T=([\d.]+) lam_F=([\d.]+)"
    )
    in_block = False
    with open(log_path) as f:
        for line in f:
            if exp_dir_substring in line and "Running results" in line:
                in_block = True
                continue
            if in_block and "Training complete" in line:
                in_block = False
                continue
            if not in_block:
                continue
            m = pattern.search(line)
            if m:
                ep = int(m.group(1))
                out[ep] = {
                    "ce": float(m.group(2)),
                    "bounded": float(m.group(3)),
                    "fior": float(m.group(4)),
                    "lam_T": float(m.group(5)),
                    "lam_F": float(m.group(6)),
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


log_path = Path("logs/hybrid_v1.log")
print(f"=== Analysis: {TIGHT} {SEED}  (K_global=34, K_local0=17, K_local1=17) ===\n")

for cell_name, _tag in CELLS:
    exp_dir = ROOT / TIGHT / cell_name / SEED
    cfg = json.load(open(exp_dir / "config.json"))
    train_log = exp_dir / "training_log.csv"
    rows = read_train_csv(train_log)
    sat_ep = first_sat(rows)
    final_ep = int(rows[-1]["Epoch"])
    early_ep = 51  # right after warmup (50 + 1)
    mid_ep = (sat_ep + 10) if sat_ep else (early_ep + 20)

    # Pull Fior-side data from .log for hybrid cells
    fior_data = {}
    if cfg["methodology"] == "tralo_fioretto":
        # Match the experiment path: hybrid_v1/L20_G20/<cell>/seed_1/config.json
        exp_substring = f"hybrid_v1/{TIGHT}/{cell_name}/{SEED}"
        fior_data = parse_log_for_cell(log_path, exp_substring)

    print(f"\n>>> {cell_name}")
    print(f"  Method: {cfg['methodology']} | first_sat=E{sat_ep} | final=E{final_ep}")

    hdr = (f"  {'Epoch':>6} {'Soft4':>7} {'Hard4':>6} {'L_Glb':>7} "
           f"{'L_Loc':>7} {'Lam_T':>6}")
    if fior_data:
        hdr += f" {'Fior':>7} {'Lam_F':>6}"
    print(hdr)
    print("  " + "-" * (76 if not fior_data else 92))

    for label, ep in [("warmup+1", early_ep),
                       ("first_sat", sat_ep),
                       ("+10ep", mid_ep),
                       ("final", final_ep)]:
        if ep is None:
            continue
        r = snapshot(rows, ep)
        if r is None:
            continue
        soft = float(r["Soft_Class4"])
        hard = int(r["Hard_Class4"])
        lglb = float(r["L_Global"])
        lloc = float(r["L_Local"])
        lamt = float(r["Lambda_Global"])
        line = (f"  {label:<6} E{ep:<3} {soft:>7.2f} {hard:>6d} {lglb:>7.4f} "
                f"{lloc:>7.4f} {lamt:>6.3f}")
        if fior_data:
            fd = fior_data.get(ep, {})
            line += f" {fd.get('fior', 0):>7.3f} {fd.get('lam_F', 0):>6.3f}"
        print(line)
    # Drift summary: soft_count between first_sat and final
    if sat_ep is not None:
        sat_r = snapshot(rows, sat_ep)
        fin_r = snapshot(rows, final_ep)
        if sat_r and fin_r:
            soft_sat = float(sat_r["Soft_Class4"])
            soft_fin = float(fin_r["Soft_Class4"])
            hard_fin = int(fin_r["Hard_Class4"])
            print(f"  >> Drift soft_4 from first_sat to final: "
                  f"{soft_sat:.2f} -> {soft_fin:.2f}  "
                  f"(K=34, final hard_4={hard_fin}, posthoc must flip "
                  f"{34 - hard_fin:+d})")
