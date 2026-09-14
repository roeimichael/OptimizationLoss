"""Is the boundary still MOVING while the constraint is pushing on it?

The condition candidate_gate could not test. Its eight checks are labels-only,
so none of them can see the failure that actually decides whether a constraint
phase can work: if cross-entropy has collapsed, the task gradient is ~0, and
under `constraint_grad_mode: normalize` the constraint's gradient is rescaled to
a FIXED norm no matter how small the violation is. The model then takes a
full-size step in a direction chosen entirely by the constraint with nothing
opposing it -- it is not reshaping a boundary, it is shoving a frozen one.

MEASURED, and it is the training recipe rather than any one dataset:

  train accuracy at epoch 6      fmow2/MNv3 0.985   bcn/MNv3 0.964   bcn/ViT 0.962
  live window (acc < 0.95)       3 epochs           5 epochs         5 epochs
  constraint epochs              29                 29               29

An ImageNet-pretrained backbone, lr 1e-4, no augmentation and no weight decay
memorises 8-18k images in about six epochs whatever the images are, so changing
dataset does not fix this. The knobs that do are augmentation, weight decay,
label smoothing, or running the constraint phase inside the live window.

The gate: the live window must cover at least HALF the constraint epochs.
"""
import argparse, csv, glob, os, sys, collections, statistics as st

SATURATED_ACC = 0.95


def live_window(path, saturated=SATURATED_ACC):
    """Epochs before train accuracy first reaches `saturated`, and the curve."""
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append((int(float(r["Epoch"])), float(r["L_CE"]), float(r["Train_Acc"])))
        except (KeyError, ValueError):
            continue
    if not rows:
        return None
    rows.sort()
    live = 0
    for e, _ce, acc in rows:
        if acc >= saturated:
            break
        live = e
    return live, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", required=True,
                    help="run-dir globs holding training_log.csv")
    ap.add_argument("--constraint-epochs", type=int, default=29)
    ap.add_argument("--saturated-acc", type=float, default=SATURATED_ACC)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the live window is too short to gate a launch")
    args = ap.parse_args(argv)

    per = collections.defaultdict(list)
    for g in args.glob:
        for f in glob.glob(os.path.join(g, "training_log.csv")):
            got = live_window(f, args.saturated_acc)
            if got:
                p = f.replace(os.sep, "/").split("/")
                # Key on (BACKBONE, dataset). Keying on (dataset, cap) merged
                # MobileNetV2, MobileNetV3 and ViTB16 into one row and hid
                # whether the architecture is a lever at all -- and the cap
                # cannot affect the warm-up, so it is not part of the cell.
                per[(p[-6], p[-5])].append(got)
    if not per:
        print("no training_log.csv matched")
        return 1
    need = max(1, args.constraint_epochs // 2)
    bad = 0
    print("live window = epochs before train accuracy reaches %.2f" % args.saturated_acc)
    print("required    = >= %d of %d constraint epochs" % (need, args.constraint_epochs))
    print("")
    print("  %-22s %4s %8s %10s %s" % ("cell", "n", "live", "verdict", "acc curve"))
    for k in sorted(per):
        v = per[k]
        live = st.mean([w for w, _ in v])
        ok = live >= need
        bad += 0 if ok else 1
        curve = v[0][1][:6]
        print("  %-22s %4d %8.1f %10s %s" % (
            "/".join(k), len(v), live, "ok" if ok else "SATURATED",
            " ".join("e%d:%.3f" % (e, a) for e, _c, a in curve)))
    if bad:
        print("")
        print("%d cell(s) SATURATE before half the constraint phase. The constraint" % bad)
        print("steps there act on a frozen boundary: CE is ~0, so under `normalize`")
        print("the step is full-size and opposed by nothing.")
    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
