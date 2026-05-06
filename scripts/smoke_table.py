"""Print the L50_G50 class-4 smoke head-to-head."""
import csv, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ORDER = ["tralo", "fioretto_ldf", "hounie_rcl", "heuristic", "danits_lp"]
print(f"{'method':14s} {'acc':>6s} {'F1':>6s} {'adj':>4s} {'raw_exc':>8s} "
      f"{'sat_ep':>7s} {'min_exc':>8s} {'restored':>9s} {'kind':>15s}")
for d in ORDER:
    p = Path("results/pending_runs/smoke_5way") / d
    if not (p/"config.json").exists():
        continue
    c = json.load(open(p/"config.json"))
    r = c.get("results", {})
    mfile = p/"evaluation_metrics.csv"
    m = dict(csv.reader(open(mfile))) if mfile.exists() else {}
    print(
        f"{d:14s} "
        f"{r.get('accuracy', 0):>6.4f} "
        f"{r.get('f1_macro', 0):>6.4f} "
        f"{r.get('samples_adjusted', -1):>4d} "
        f"{m.get('Raw Total Excess', '-'):>8s} "
        f"{str(m.get('Satisfaction Epoch', '-')):>7s} "
        f"{str(m.get('Min Total Excess', '-')):>8s} "
        f"{str(m.get('Restored From Epoch', '-')):>9s} "
        f"{m.get('Restore Kind', '-'):>15s}"
    )
