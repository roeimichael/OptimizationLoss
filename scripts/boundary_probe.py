"""WHICH items did the constraint pull across the clipper's boundary, and were
they worth pulling?

Both arms deploy EXACTLY K predictions per capped class (FRAMEWORK 2(z9), 16 of
16 arms verified). So the budget cannot differ and the ONLY thing that can
differ is WHICH K items were chosen. That is the question this answers, and no
count-based diagnostic can: `flips`, `raw_over_K` and satisfaction are all
blind to it.

For each capped class c, with S_ctrl and S_trt the two deployed selections:

    evicted  = S_ctrl \\ S_trt      items the control kept and the arm dropped
    admitted = S_trt \\ S_ctrl      items the arm reached for instead

|evicted| == |admitted| exactly, because both sets hold exactly K.

The measurement that matters is WHERE the admitted items sat in the CONTROL's
own ranking. Rank <= K means the control had already selected it (impossible
here by construction). Rank > K means the arm reached BELOW the control's
boundary and pulled something the clipper had rejected. `depth = rank - K` is
how far below.

Then the only thing that decides whether any of it was worth doing:

    net = TP(admitted) - TP(evicted)      in ITEMS

\U0001f6d1 A POSITIVE `net` IS THE ONLY OUTCOME THAT COUNTS. Swapping items of
equal quality is what a pure RNG reseed does -- measured at 63 items moved for
a net of +0.38 -- so a large swap count is not evidence of anything. Quote
`net` beside the swap count, never the swap count alone.

⚠️ RANKS COME FROM THE CONTROL'S PROBABILITIES, NOT THE ARM'S. The two are
different models, so "depth" measures how deep into the CLIPPER's reject pile
the arm reached. That is the question asked; the reverse framing (depth in the
arm's own ranking) answers nothing, because the arm's top-K is its selection by
definition.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(run_dir, deployed=True):
    """(true labels, predicted labels, probability matrix, group ids)."""
    name = "final_predictions.csv" if deployed else "final_predictions_raw.csv"
    path = os.path.join(run_dir, name)
    y, p, probs, gids = [], [], [], []
    with open(path) as f:
        r = csv.DictReader(f)
        pcols = sorted([c for c in r.fieldnames if c.startswith("Prob_Class_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
        for row in r:
            y.append(int(row["True_Label"]))
            p.append(int(row["Predicted_Label"]))
            probs.append([float(row[c]) for c in pcols])
            gids.append(row.get("Group_ID"))
    return (np.array(y), np.array(p), np.array(probs, dtype=float),
            np.array(gids))


def compare(ctrl_dir, trt_dir, classes):
    """Per capped class: who was swapped, from how deep, and was it worth it."""
    y0, p0, P0, _ = load(ctrl_dir)
    y1, p1, P1, _ = load(trt_dir)
    if len(y0) != len(y1) or not np.array_equal(y0, y1):
        raise SystemExit("%s and %s are not the same test set -- refusing to "
                         "difference them" % (ctrl_dir, trt_dir))
    out = {}
    for c in classes:
        S0 = set(np.where(p0 == c)[0].tolist())
        S1 = set(np.where(p1 == c)[0].tolist())
        evicted = sorted(S0 - S1)
        admitted = sorted(S1 - S0)
        # the control's own ranking for this class, best first
        order = np.argsort(-P0[:, c])
        rank_of = np.empty(len(y0), dtype=int)
        rank_of[order] = np.arange(1, len(y0) + 1)
        K = len(S0)
        tp_adm = int(sum(1 for i in admitted if y0[i] == c))
        tp_evi = int(sum(1 for i in evicted if y0[i] == c))
        out[int(c)] = dict(
            K=K, n_pos=int((y0 == c).sum()),
            swapped=len(admitted),
            # |admitted| == |evicted| whenever both sides deploy exactly K.
            # If they differ, the budget was NOT equalized and every number
            # below is a budget measurement -- so this is reported, not assumed.
            budget_equal=(len(admitted) == len(evicted)),
            n_evicted=len(evicted),
            depths=[int(rank_of[i] - K) for i in admitted],
            admitted_correct=[bool(y0[i] == c) for i in admitted],
            evicted_correct=[bool(y0[i] == c) for i in evicted],
            tp_admitted=tp_adm, tp_evicted=tp_evi,
            net_items=tp_adm - tp_evi,
            prec_admitted=(tp_adm / len(admitted)) if admitted else float("nan"),
            prec_evicted=(tp_evi / len(evicted)) if evicted else float("nan"),
        )
    return out


def find_runs(root, model, cap, arm, seeds):
    hits = []
    for s in seeds:
        d = os.path.join(root, model, "iwildcam", cap, arm, "seed_%d" % s)
        if os.path.isdir(d) and os.path.exists(
                os.path.join(d, "final_predictions.csv")):
            hits.append((s, d))
    return hits


def self_test(out=sys.stdout):
    """A synthetic pair with a KNOWN swap must be recovered exactly."""
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-56s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    n, C, c = 20, 3, 1
    y = np.array([c if i < 8 else 0 for i in range(n)])
    # control selects items 0..3 (K=4); treated drops 3 and takes 9
    p0 = np.array([c if i < 4 else 0 for i in range(n)])
    p1 = np.array([c if i in (0, 1, 2, 9) else 0 for i in range(n)])
    P = np.zeros((n, C))
    P[:, c] = np.linspace(0.99, 0.01, n)      # rank == index + 1
    P[:, 0] = 1 - P[:, c]

    d = tempfile.mkdtemp()
    for nm, pred in (("ctrl", p0), ("trt", p1)):
        os.makedirs(os.path.join(d, nm), exist_ok=True)
        with open(os.path.join(d, nm, "final_predictions.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["True_Label", "Predicted_Label", "Correct"]
                       + ["Prob_Class_%d" % k for k in range(C)] + ["Group_ID"])
            for i in range(n):
                w.writerow([y[i], pred[i], int(y[i] == pred[i])]
                           + list(P[i]) + [0])
    r = compare(os.path.join(d, "ctrl"), os.path.join(d, "trt"), [c])[c]

    check("recovers exactly one swap", r["swapped"] == 1 and r["n_evicted"] == 1)
    check("reports the budget as equal", r["budget_equal"] is True)
    # item 9 has rank 10, K=4 -> depth 6
    check("depth is measured below the CONTROL's boundary", r["depths"] == [6])
    # evicted item 3 is a true positive; admitted item 9 is NOT (y=0 for i>=8)
    check("net is NEGATIVE when a TP is swapped for a non-TP",
          r["net_items"] == -1)
    check("admitted precision 0.0, evicted precision 1.0",
          r["prec_admitted"] == 0.0 and r["prec_evicted"] == 1.0)

    # NEGATIVE CONTROL: identical predictions must yield NO swap and net 0
    r2 = compare(os.path.join(d, "ctrl"), os.path.join(d, "ctrl"), [c])[c]
    check("negative control: identical arms swap nothing, net 0",
          r2["swapped"] == 0 and r2["net_items"] == 0)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--campaign")
    a.add_argument("--control", default="clip")
    a.add_argument("--arms", nargs="+", default=["tralo_uniform", "tralo"])
    a.add_argument("--models", nargs="+")
    a.add_argument("--caps", nargs="+")
    a.add_argument("--classes", nargs="+", type=int, default=[2, 7])
    a.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    a.add_argument("--json")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.campaign:
        raise SystemExit("--campaign is required (or --self-test)")

    models = args.models or sorted(
        d for d in os.listdir(args.campaign)
        if os.path.isdir(os.path.join(args.campaign, d)))
    rows = []
    for model in models:
        caps = args.caps or sorted(
            os.listdir(os.path.join(args.campaign, model, "iwildcam")))
        for cap in caps:
            for arm in args.arms:
                cr = find_runs(args.campaign, model, cap, args.control,
                               args.seeds)
                tr = dict(find_runs(args.campaign, model, cap, arm, args.seeds))
                for seed, cdir in cr:
                    if seed not in tr:
                        continue
                    try:
                        res = compare(cdir, tr[seed], args.classes)
                    except SystemExit as exc:
                        print("  skipped %s: %s" % (tr[seed], exc))
                        continue
                    for c, v in res.items():
                        rows.append(dict(model=model, cap=cap, arm=arm,
                                         seed=seed, cls=c, **v))
    if not rows:
        print("no comparable (control, arm) pairs found")
        return 1

    print("BOUNDARY PROBE -- which items crossed the clipper's cut, and was it "
          "worth it")
    print("control = %s   %d pair(s)" % (args.control, len(rows)))
    print("")
    print("%-13s %-9s %-14s %3s %4s %6s %7s %7s %8s %8s"
          % ("model", "cap", "arm", "cls", "seed", "swap", "precAdm",
             "precEvi", "net", "medDepth"))
    for r in rows:
        d = r["depths"]
        print("%-13s %-9s %-14s %3d %4d %6d %7.3f %7.3f %+8d %8s"
              % (r["model"], r["cap"], r["arm"], r["cls"], r["seed"],
                 r["swapped"], r["prec_admitted"], r["prec_evicted"],
                 r["net_items"],
                 "%d" % int(np.median(d)) if d else "-"))
        if not r["budget_equal"]:
            print("      !! BUDGET NOT EQUAL: %d admitted vs %d evicted. Every "
                  "number on this row is partly a budget measurement."
                  % (r["swapped"], r["n_evicted"]))

    print("")
    for arm in args.arms:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        net = sum(r["net_items"] for r in sub)
        sw = sum(r["swapped"] for r in sub)
        pa = np.mean([r["prec_admitted"] for r in sub
                      if r["swapped"]]) if sw else float("nan")
        pe = np.mean([r["prec_evicted"] for r in sub
                      if r["swapped"]]) if sw else float("nan")
        alld = [x for r in sub for x in r["depths"]]
        print("%-14s swapped %5d items, net %+5d, precision admitted %.3f vs "
              "evicted %.3f, median depth %s"
              % (arm, sw, net, pa, pe,
                 "%d" % int(np.median(alld)) if alld else "-"))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
