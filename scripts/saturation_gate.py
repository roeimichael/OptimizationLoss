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
import argparse, csv, glob, json, os, sys, collections, statistics as st

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


def _live_ok(per, key):
    """Is this (backbone, dataset, constraint-epochs) cell still live at its cut?"""
    con = key[2]
    live = st.mean([w for w, _r, _c in per[key]])
    return live >= max(1, con // 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", required=True,
                    help="run-dir globs holding training_log.csv")
    ap.add_argument("--constraint-epochs", type=int, default=None,
                    help="override; by default each run's OWN constraint_epochs "
                         "is read from its config.json. The default used to be a "
                         "hardcoded 29, so a 6-epoch campaign was judged against "
                         "a 30-epoch budget and reported SATURATED while its "
                         "trained arms were live for 94%% of their constraint "
                         "phase.")
    ap.add_argument("--saturated-acc", type=float, default=SATURATED_ACC)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the live window is too short to gate a launch")
    args = ap.parse_args(argv)

    per = collections.defaultdict(list)
    for g in args.glob:
        for f in glob.glob(os.path.join(g, "training_log.csv")):
            # A post-hoc arm has NO constraint phase, so its live window says
            # nothing about whether a constraint was pushing a frozen boundary.
            # Pooling clippers into the cell dragged the mean down and reported
            # campaigns as saturated on the strength of runs the gate does not
            # apply to.
            con = args.constraint_epochs
            if con is None:
                cj = os.path.join(os.path.dirname(f), "config.json")
                try:
                    con = int(json.load(open(cj))["hyperparams"]["constraint_epochs"])
                except (OSError, KeyError, ValueError):
                    continue
            if con <= 0:
                continue
            got = live_window(f, args.saturated_acc)
            if got:
                got = (got[0], got[1], con)
                p = f.replace(os.sep, "/").split("/")
                # Key on (BACKBONE, dataset, CONSTRAINT EPOCHS). Keying on
                # (dataset, cap) merged MobileNetV2, MobileNetV3 and ViTB16 into
                # one row and hid whether the architecture is a lever at all --
                # and the cap cannot affect the warm-up, so it is not part of
                # the cell. The budget IS part of it: the criterion is a
                # FRACTION of the constraint phase, so a campaign that sweeps
                # the budget has a different threshold per arm. Judging all of
                # them against the shortest budget, as this did, slackens the
                # long arms to the short arm's bar and passes them wrongly.
                per[(p[-6], p[-5], con)].append(got)
    if not per:
        print("no training_log.csv matched")
        return 1
    budgets = sorted({k[2] for k in per})
    bad = 0
    print("live window = epochs before train accuracy reaches %.2f" % args.saturated_acc)
    print("required    = >= half of each cell's OWN constraint phase%s" % (
        "  (campaign sweeps budgets %s)" % budgets if len(budgets) > 1 else
        "  (%d epochs)" % budgets[0] if budgets else ""))
    print("")
    print("  %-22s %4s %4s %6s %8s %10s %s"
          % ("cell", "con", "n", "live", "live frac", "verdict", "acc curve"))
    for k in sorted(per, key=lambda k: (k[0], k[1], -k[2])):
        v = per[k]
        con_epochs = k[2]
        need = max(1, con_epochs // 2)
        live = st.mean([w for w, _r, _c in v])
        ok = live >= need
        bad += 0 if ok else 1
        curve = v[0][1][:6]
        print("  %-22s %4d %4d %6.1f %8.0f%% %10s %s" % (
            "/".join(k[:2]), con_epochs, len(v), live, 100.0 * live / con_epochs,
            "ok" if ok else "SATURATED",
            " ".join("e%d:%.3f" % (e, a) for e, _c, a in curve)))
    if bad:
        print("")
        print("%d cell(s) SATURATE before half the constraint phase. The constraint" % bad)
        print("steps there act on a frozen boundary: CE is ~0, so under `normalize`")
        print("the step is full-size and opposed by nothing.")

    # PROTOCOL AMENDMENT 2026-09-15, explicit and matched.
    #
    # This gate exists to stop a campaign that CANNOT ANSWER ITS QUESTION,
    # because every cell memorised before the constraint did anything. For a
    # single-budget campaign that is the same thing as "a cell saturated", and
    # the rule is unchanged.
    #
    # A campaign that deliberately SWEEPS the budget is different: the long
    # budgets are the frozen END OF THE DOSE AXIS, included on purpose as the
    # reference condition. Killing such a campaign because its reference arm is
    # frozen would delete the control, not protect the experiment. What would
    # make a swept campaign vacuous is having NO live budget at all -- then the
    # sweep has no contrast and measures the frozen regime five times over.
    #
    # So: sweeps require at least one live budget and report every verdict; a
    # single-budget campaign must itself be live. This LOOSENS nothing for the
    # campaigns the gate was written against -- `fm2_mn3`, `fm2_vit` and `gx2`
    # carry one budget each and still fail.
    live_budgets = sorted({k[2] for k in per if _live_ok(per, k)})
    if len(budgets) > 1:
        print("")
        print("swept campaign: %d of %d budgets are live (%s of %s constraint epochs)."
              % (len(live_budgets), len(budgets), live_budgets or "none", budgets))
        if not live_budgets:
            print("NO budget is live. The sweep has no contrast -- every arm would")
            print("measure the frozen regime, which is the one thing already settled.")
        else:
            print("The frozen budgets are the reference end of the dose axis, not")
            print("failures. Read each budget against its own threshold.")
        return 1 if (not live_budgets and args.strict) else 0
    return 1 if (bad and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
