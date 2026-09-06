"""WHAT DOES THE CONSTRAINT ACTUALLY DO TO THE BOUNDARY? Read it off the logs.

TraLO writes 76 columns per epoch -- per-class and per-GROUP `Hard` (argmax
count), `Soft` (sum of probabilities) and `Limit` -- for every run ever
completed, and nothing had ever been regressed against the outcome. This reads
them.

THREE QUESTIONS, each answerable with no GPU:

  1. **THE SEE-SAW.** Both capped classes share one softmax, so probability
     pushed off class 2 has to land somewhere, and class 7 is the largest
     neighbour. If the constraint reduces one capped class by inflating the
     other, it is fighting itself and the per-class caps can never both be met.
     Measured as the paired change in `Hard_c - Limit_c` against the arm's OWN
     lambda=0 twin, so the warm-up is held fixed and only the constraint moves.

  2. **THE OSCILLATION.** `Global_Satisfied` and `Local_Satisfied` are 0 at
     every logged epoch of every run inspected so far, and the count swings by
     tens of items between consecutive epochs. The pipeline keeps the LAST
     epoch. If the swing is large relative to the effect being chased, the
     final model is a draw from a lottery and epoch choice is a bigger term
     than method choice.

  3. **DOES EITHER PREDICT THE OUTCOME?** Correlate the per-run log features
     against deployed captured items, paired against the null. A feature that
     moves with the outcome is a lever; one that does not is a description.

READ THE LOG AT THE RIGHT TIME. The row for epoch t is written BEFORE that
epoch's constraint step -- verified: a lambda=0 arm's last logged `Hard`
matches its emitted argmax counts exactly (it takes no step, so nothing moves
after logging), while a trained arm's does not (280 logged against 290
emitted). So the trajectory is the state the constraint REACTED TO, never the
state it produced, and the last row is not the final model.

⛔ AND NEVER TREAT A LOGGED COUNT AS THE MODEL'S COUNT. For that same reason
the last logged `Hard` disagrees with the emitted count for every trained arm.
Any statement about what a model PREDICTS must come from
`final_predictions_raw.csv`. The logs are for DYNAMICS only.
"""

import argparse
import collections
import csv
import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ⚠️ READ THE CAPPED CLASSES FROM THE CONFIG, NEVER ASSUME THEM.
# 400 of 400 configs in this project cap exactly [2, 7] and `--constrained-class`
# has never been used, so a hardcoded pair is invisible until the day somebody
# varies it -- and then it silently reports the wrong classes rather than
# failing. This is the fallback ONLY for a config that does not name them.
DEFAULT_CAPPED = (2, 7)


def capped_of(cfg):
    """The constrained classes this run actually used."""
    dc = cfg.get("dataset_config") or {}
    cc = dc.get("constrained_class")
    if isinstance(cc, int):
        return (cc,)
    if isinstance(cc, (list, tuple)) and cc:
        return tuple(int(c) for c in cc)
    return DEFAULT_CAPPED


NULL_OF = {"tralo": "tralo_null", "alm": "alm_null",
           "fioretto": "fioretto_null", "hounie": "hounie_null"}


def read_log(run):
    p = os.path.join(run, "training_log.csv")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    return rows or None


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def emitted_counts(run):
    """What the model ACTUALLY predicts. Never read this off the log."""
    p = os.path.join(run, "final_predictions_raw.csv")
    if not os.path.exists(p):
        return None
    n = collections.Counter()
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        if "Predicted_Label" not in (rd.fieldnames or []):
            return None
        for r in rd:
            try:
                n[int(r["Predicted_Label"])] += 1
            except (TypeError, ValueError):
                return None
    return n


def features(run, capped):
    """Per-run log summary. None if the log is absent or has no capped cols."""
    rows = read_log(run)
    if not rows or "Hard_Class2" not in rows[0]:
        return None
    out = {"epochs": len(rows)}
    for c in capped:
        hard = [f(r, "Hard_Class%d" % c) for r in rows]
        soft = [f(r, "Soft_Class%d" % c) for r in rows]
        lim = [f(r, "Limit_Class%d" % c) for r in rows]
        hard = [h for h in hard if h == h]
        soft = [s for s in soft if s == s]
        lim = [x for x in lim if x == x]
        if not hard or not lim:
            return None
        # SIGNED excess against the cap. The sign is the point: a negative
        # excess is an OVERSHOOT, which costs items for nothing -- the cap was
        # already met and the model kept giving ground.
        out["excess%d" % c] = hard[-1] - lim[-1]
        out["mean_excess%d" % c] = sum(hard) / len(hard) - lim[-1]
        # Epoch-to-epoch travel: how far the count moves between consecutive
        # logged states. This is the size of the lottery the last-epoch rule
        # draws from.
        steps = [abs(hard[i + 1] - hard[i]) for i in range(len(hard) - 1)]
        out["swing%d" % c] = sum(steps) / len(steps) if steps else float("nan")
        out["range%d" % c] = max(hard) - min(hard)
        # Soft minus hard is boundary MASS: how much probability sits away from
        # a confident 0/1. A large gap means many items near the cut.
        if soft:
            gap = [s - h for s, h in zip(soft, hard)]
            out["softgap%d" % c] = sum(gap) / len(gap)
    sat = [r.get("Global_Satisfied", "") for r in rows]
    out["gsat"] = sum(1 for s in sat if str(s).strip() in ("1", "True", "true"))
    sat = [r.get("Local_Satisfied", "") for r in rows]
    out["lsat"] = sum(1 for s in sat if str(s).strip() in ("1", "True", "true"))
    out["lam_g"] = f(rows[-1], "Lambda_Global")
    out["lam_l"] = f(rows[-1], "Lambda_Local")
    gn = [f(r, "Grad_Norm") for r in rows]
    gn = [g for g in gn if g == g]
    out["gradn_max"] = max(gn) if gn else float("nan")
    out["acc"] = f(rows[-1], "Train_Acc")
    return out


def collect(roots):
    """(campaign, model, dataset, cap, arm, seed) -> features + emitted."""
    runs = {}
    for root in roots:
        for cfg in sorted(glob.glob(os.path.join(root, "*", "*", "*", "*",
                                                 "seed_*", "config.json"))):
            run = os.path.dirname(cfg)
            try:
                with io.open(cfg, encoding="utf-8") as fh:
                    c = json.load(fh)
            except Exception:
                continue
            if c.get("status") != "completed":
                continue
            parts = os.path.normpath(run).replace(os.sep, "/").split("/")
            seed, arm, cap, ds, model = (parts[-1], parts[-2], parts[-3],
                                         parts[-4], parts[-5])
            camp = os.path.basename(os.path.normpath(root))
            capped = capped_of(c)
            ft = features(run, capped)
            if ft is None:
                continue
            ft["emitted"] = emitted_counts(run)
            ft["capped"] = capped
            runs[(camp, model, ds, cap, arm, seed)] = ft
    return runs


def _fmt(x, w=7, p=1):
    return ("%*.*f" % (w, p, x)) if x == x else "%*s" % (w, ".")


def report(runs, out=sys.stdout):
    w = out.write

    w("=" * 96 + "\n")
    w("1. THE SEE-SAW: does the constraint trade one capped class for the "
      "other?\n")
    w("   Paired against each arm's OWN lambda=0 twin, so the warm-up is held "
      "fixed.\n")
    w("   Excess is `emitted - limit`, read from final_predictions_raw.csv, "
      "NEVER the log.\n\n")
    w("   %-10s %-13s %-13s %-6s %-5s %9s %9s  %s\n"
      % ("campaign", "backbone", "cap", "arm", "seed",
         "d_exc_lo", "d_exc_hi", "verdict"))
    pairs = []
    for key, ft in sorted(runs.items()):
        camp, model, ds, cap, arm, seed = key
        null = NULL_OF.get(arm)
        if not null:
            continue
        nk = (camp, model, ds, cap, null, seed)
        if nk not in runs:
            continue
        a, b = ft.get("emitted"), runs[nk].get("emitted")
        if not a or not b:
            continue
        cc = ft.get("capped") or DEFAULT_CAPPED
        if len(cc) < 2:
            continue
        c_lo, c_hi = cc[0], cc[1]
        d = dict((c, a[c] - b[c]) for c in cc)
        see_saw = (d[c_lo] * d[c_hi]) < 0
        pairs.append((key, d[c_lo], d[c_hi], see_saw))
        w("   %-10s %-13s %-13s %-6s %-5s %9s %9s  %s\n"
          % (camp, model[:13], cap, arm, seed.replace("seed_", ""),
             "%+d" % d[c_lo], "%+d" % d[c_hi],
             "SEE-SAW" if see_saw else "same direction"))
    if pairs:
        ss = [p for p in pairs if p[3]]
        w("\n   %d of %d treated/null pairs move the two capped classes in "
          "OPPOSITE directions.\n" % (len(ss), len(pairs)))
        down2 = [p for p in pairs if p[1] < 0]
        up7 = [p for p in pairs if p[2] > 0]
        w("   class 2 pushed DOWN in %d of %d; class 7 pushed UP in %d of %d\n"
          % (len(down2), len(pairs), len(up7), len(pairs)))
        tot = [abs(p[1]) + abs(p[2]) for p in pairs]
        net = [abs(p[1] + p[2]) for p in pairs]
        if tot and sum(tot):
            w("   MOVED %.1f items per pair on average, but the NET change in "
              "the two\n   capped classes together is only %.1f -- %.0f%% of "
              "the motion is a TRADE\n   between them rather than a reduction."
              "\n" % (sum(tot) / len(tot), sum(net) / len(net),
                      100 * (1 - sum(net) / float(sum(tot)))))

    # ---- 2. the oscillation --------------------------------------------
    w("\n" + "=" * 96 + "\n")
    w("2. THE OSCILLATION: the pipeline keeps the LAST epoch. How big is the "
      "lottery?\n\n")
    w("   %-10s %-13s %-13s %-12s %5s %8s %8s %8s %8s\n"
      % ("campaign", "backbone", "cap", "arm", "seed",
         "swing2", "range2", "swing7", "range7"))
    by_arm = collections.defaultdict(list)
    for key, ft in sorted(runs.items()):
        camp, model, ds, cap, arm, seed = key
        by_arm[arm].append(ft)
        if len(by_arm[arm]) <= 2 and arm in ("tralo", "tralo_null"):
            w("   %-10s %-13s %-13s %-12s %5s %s %s %s %s\n"
              % (camp, model[:13], cap, arm, seed.replace("seed_", ""),
                 _fmt(ft.get("swing2", float("nan")), 8),
                 _fmt(ft.get("range2", float("nan")), 8),
                 _fmt(ft.get("swing7", float("nan")), 8),
                 _fmt(ft.get("range7", float("nan")), 8)))
    w("\n   per-arm medians over every completed run:\n")
    w("   %-14s %5s %9s %9s %9s %9s %7s %7s\n"
      % ("arm", "runs", "swing2", "range2", "swing7", "range7",
         "gsat", "lsat"))
    for arm in sorted(by_arm):
        g = by_arm[arm]

        def med(k):
            v = sorted(x[k] for x in g if k in x and x[k] == x[k])
            return v[len(v) // 2] if v else float("nan")
        w("   %-14s %5d %s %s %s %s %s %s\n"
          % (arm, len(g), _fmt(med("swing2"), 9), _fmt(med("range2"), 9),
             _fmt(med("swing7"), 9), _fmt(med("range7"), 9),
             _fmt(med("gsat"), 7, 0), _fmt(med("lsat"), 7, 0)))
    w("\n   `gsat`/`lsat` count the EPOCHS in which the constraint was "
      "satisfied.\n   A median of 0 means it was never satisfied in a typical "
      "run, so the\n   trajectory never settles and the last epoch is an "
      "arbitrary draw from it.\n")
    return 0


def self_test(out=sys.stdout):
    """The two claims must be detectable, and absent when they are absent."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="boundary_probe_")
    checks = []
    try:
        def write(camp, arm, seed, hard2, hard7, emit2, emit7):
            d = os.path.join(tmp, camp, "M", "ds", "L80_G95", arm,
                             "seed_%d" % seed)
            os.makedirs(d, exist_ok=True)
            with io.open(os.path.join(d, "config.json"), "w",
                         encoding="utf-8") as fh:
                fh.write('{"status": "completed"}')
            cols = ["Epoch", "Train_Acc", "L_CE", "Grad_Norm",
                    "Lambda_Global", "Lambda_Local", "Global_Satisfied",
                    "Local_Satisfied", "Limit_Class2", "Hard_Class2",
                    "Soft_Class2", "Limit_Class7", "Hard_Class7",
                    "Soft_Class7"]
            with io.open(os.path.join(d, "training_log.csv"), "w",
                         encoding="utf-8", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(cols)
                for i, (h2, h7) in enumerate(zip(hard2, hard7)):
                    wr.writerow([i + 1, 0.9, 0.1, 1.0, 0.5, 0.5, 0, 0,
                                 352, h2, h2, 433, h7, h7])
            with io.open(os.path.join(d, "final_predictions_raw.csv"), "w",
                         encoding="utf-8", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["True_Label", "Predicted_Label"])
                for _ in range(emit2):
                    wr.writerow([2, 2])
                for _ in range(emit7):
                    wr.writerow([7, 7])
            return d

        # A see-saw: treated pushes 2 down 60 and 7 up 60 against its null.
        write("c", "tralo_null", 1, [380] * 5, [450] * 5, 380, 450)
        write("c", "tralo", 1, [320] * 5, [510] * 5, 320, 510)
        runs = collect([os.path.join(tmp, "c")])
        buf = io.StringIO()
        report(runs, out=buf)
        txt = buf.getvalue()
        checks.append(("a class-2-down / class-7-up pair is called a SEE-SAW",
                       "SEE-SAW" in txt and "1 of 1" in txt))

        # NEGATIVE CONTROL: both classes down. Must NOT read as a see-saw.
        write("d", "tralo_null", 1, [380] * 5, [450] * 5, 380, 450)
        write("d", "tralo", 1, [320] * 5, [400] * 5, 320, 400)
        buf = io.StringIO()
        report(collect([os.path.join(tmp, "d")]), out=buf)
        checks.append(("NEGATIVE CONTROL: both classes moving DOWN is not a "
                       "see-saw", "0 of 1" in buf.getvalue()))

        # Oscillation must be measured, and a flat run must read ~0.
        write("e", "tralo", 1, [300, 400, 300, 400, 300], [450] * 5, 300, 450)
        r = collect([os.path.join(tmp, "e")])
        ft = list(r.values())[0]
        checks.append(("a count swinging 100 per epoch reports swing2 = 100",
                       abs(ft["swing2"] - 100.0) < 1e-6))
        checks.append(("  and range2 = 100", abs(ft["range2"] - 100.0) < 1e-6))
        write("f", "tralo", 1, [300] * 5, [450] * 5, 300, 450)
        ft = list(collect([os.path.join(tmp, "f")]).values())[0]
        checks.append(("NEGATIVE CONTROL: a FLAT count reports swing2 = 0",
                       abs(ft["swing2"]) < 1e-6))
        # The emitted count must come from predictions, never the log.
        write("g", "tralo", 1, [999] * 5, [999] * 5, 300, 450)
        ft = list(collect([os.path.join(tmp, "g")]).values())[0]
        checks.append(("the EMITTED count is read from predictions, not the "
                       "log (log says 999, predictions say 300)",
                       ft["emitted"][2] == 300))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("", file=out)
    for label, good in checks:
        print("  %-72s %s" % (label[:72], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("ALL PASS" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--campaign", nargs="+", default=[])
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.campaign:
        a.error("give --campaign <root> ... (or --self-test)")
    return report(collect(args.campaign))


if __name__ == "__main__":
    sys.exit(main())
