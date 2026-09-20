"""Is the constraint term ever SATISFIED during training, in any run?

LEDGER 2.3 claims every constraint term runs on TRAIN data, where violation is
identically zero, so the term is silent for ~80% of training. The source says
otherwise: tralo and all four duals compute their term on inputs.X_test. This
reads the actual logs and settles it.
"""
import glob, csv, collections

BASE = "/home/dsi/michaer8/optloss-rank/results"
rows = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0])   # epochs, satisfied, maxG, maxL, runs

for f in glob.glob(BASE + "/polc*/*/*/*/*/*/training_log.csv"):
    p = f.split("/")
    arm = p[-3]
    try:
        r = list(csv.DictReader(open(f)))
    except Exception:
        continue
    if not r or "Global_Satisfied" not in r[0]:
        continue
    st = rows[arm]
    st[4] += 1
    for row in r:
        st[0] += 1
        gs = str(row.get("Global_Satisfied", "0")).strip()
        ls = str(row.get("Local_Satisfied", "0")).strip()
        if gs in ("1", "True", "true") and ls in ("1", "True", "true"):
            st[1] += 1
        for k, i in (("L_Global", 2), ("L_Local", 3)):
            try:
                st[i] = max(st[i], float(row.get(k, 0) or 0))
            except ValueError:
                pass

print("%-14s %7s %9s %11s %13s %13s" % ("arm", "runs", "epochs", "satisfied", "max L_Global", "max L_Local"))
print("-" * 74)
for arm in sorted(rows):
    e, s, mg, ml, n = rows[arm]
    print("%-14s %7d %9d %11s %13.1f %13.1f"
          % (arm, n, e, "%d (%.1f%%)" % (s, 100.0 * s / max(e, 1)), mg, ml))
