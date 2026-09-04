"""FOUR different noise numbers exist here. Quoting the wrong one is a result.

WHY THIS EXISTS. `ceiling_screen` prices a direction as `prize / sd`, and the
first two versions of that ratio were both wrong -- not in the prize, in the
`sd`. The prize is a property of K and the ranking; the noise is a property of
the CONTRAST you intend to run, and this project runs exactly one kind:
seed-paired against the arm's OWN lambda=0 twin.

Pairing normally shrinks noise, so the natural assumption is that the paired sd
is the small one and an unpaired quote is conservative. **On this design the
opposite is true**, and by a factor of 6 to 12. `tralo` and `tralo_null` share
ONE warm-up epoch and then train 29 more apart. They are two models, not two
readings of one model, so the pairing cancels almost nothing and ADDS the
variance of a second training run.

    unpaired   sd of one arm's TP@K across seeds, within cell.
               What an ABSOLUTE quality claim faces.
    reseed     sd of (reseed_arm - control). RNG stream perturbed and nothing
               else, so this is the floor under ANY paired contrast, and the
               honest bar for a new arm.
    treated    sd of (treated_arm - control). What the contrast you are
               actually running faces. Always the largest of the three here.

    (the fourth is `full_panel`'s `paired seed sd`, in `d ccF1` MACRO-averaged
     over both capped classes and converted through `(K+n)/2`. It is a
     different quantity in different units -- do not substitute it for these.)

Measured on `results/iwc3`, class 2: at K/n = 0.2 the prize is 0.42 items
against an unpaired sd of 0.80 (0.52x) but a treated sd of 7.59 (**0.05x**).
The direction that looked marginal was never close. FRAMEWORK 2(v).

    python -m scripts.paired_noise --campaign results/iwc3
    python -m scripts.paired_noise --campaign results/iwc3 --classes 2 7
    python -m scripts.paired_noise --self-test
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

COLUMNS = ["cell", "seed", "cls", "frac", "n", "k", "tp"]
DEFAULT_FRACS = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]


def load_arm(root, arm, classes, fracs):
    """TP@K for every (cell, seed, class, K/n) this arm produced.

    K comes from the LABELS and the requested fraction, not from the cap
    policy: the point here is to sweep K/n past the caps the protocol actually
    uses, so the noise curve can be read where a looser budget would put it.
    """
    rows = []
    pattern = os.path.join(root, "**", arm, "seed_*",
                           "final_predictions_raw.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        parts = path.replace(os.sep, "/").split("/")
        cell, seed = "/".join(parts[-6:-3]), parts[-2]
        df = pd.read_csv(path)
        y = df["True_Label"].values
        for c in classes:
            col = "Prob_Class_%d" % c
            if col not in df.columns:
                continue
            p = df[col].values
            n = int((y == c).sum())
            if n == 0:
                continue
            cum = np.cumsum((y[np.argsort(-p)] == c).astype(int))
            for f in fracs:
                k = max(1, min(int(round(f * n)), len(cum)))
                rows.append((cell, seed, c, f, n, k, int(cum[k - 1])))
    return pd.DataFrame(rows, columns=COLUMNS)



def _model_fingerprints(frame):
    """{(cell, seed): fingerprint of its whole TP curve}.

    Two runs with the same TP at every (class, K/n) are the same model read
    twice. That is not hypothetical here: a `_null` / `_reseed` arm has
    lambda = 0, carries no constraint term, and therefore cannot depend on the
    cap -- so running it at two cap levels produces two CELLS holding ONE
    model.
    """
    fps = {}
    if frame.empty:
        return fps
    for (cell, seed), g in frame.groupby(["cell", "seed"]):
        key = tuple(g.sort_values(["cls", "frac"])["tp"].tolist())
        fps[(cell, seed)] = key
    return fps


def _run_census(frame):
    """'3' when every run is a distinct model, '3 (2 distinct)' when not."""
    fps = _model_fingerprints(frame)
    n, d = len(fps), len(set(fps.values()))
    return "%d" % n if n == d else "%d (%d distinct)" % (n, d)


def _warn_duplicate_models(frames):
    """Name the collapsing runs, because the count alone reads as a typo.

    \U0001f6d1 A RUN IS NOT A SEED. Pooling a sd over cells that hold the same
    model double-counts it and biases the sd DOWNWARD, which makes every
    prize/noise ratio in this table look better than it is.
    """
    said = False
    for name, frame in frames.items():
        fps = _model_fingerprints(frame)
        byfp = {}
        for k, v in fps.items():
            byfp.setdefault(v, []).append(k)
        dupes = [v for v in byfp.values() if len(v) > 1]
        if not dupes:
            continue
        if not said:
            print("")
            said = True
        print("  !! %s: %d run(s) are the SAME MODEL read more than once. A "
              "lambda=0" % (name, sum(len(d) - 1 for d in dupes)))
        print("     arm has no constraint term, so its predictions cannot "
              "depend on the cap;")
        print("     two cap levels are two CELLS holding ONE model. A sd "
              "pooled over those")
        print("     cells double-counts it and biases every ratio below "
              "OPTIMISTIC.")
        for group in dupes[:3]:
            print("       %s" % "  ==  ".join("%s/%s" % g for g in group))

def paired_sd(treated, control):
    """{(cls, frac): sd of (treated - control)}, pooled over cells.

    Pooled as the root-mean-square of the WITHIN-cell sds, never as one sd over
    the flattened set: a cell-to-cell mean shift is not noise, and pooling
    across it would inflate every figure here (house rule 4).
    """
    key = ["cell", "seed", "cls", "frac"]
    m = treated.merge(control, on=key, suffixes=("_a", "_b"))
    m["d"] = m.tp_a - m.tp_b
    out = {}
    for (c, f), grp in m.groupby(["cls", "frac"]):
        sds = grp.groupby("cell").d.std(ddof=1).dropna()
        out[(c, f)] = float(np.sqrt((sds ** 2).mean())) if len(sds) else np.nan
    return out


def unpaired_sd(df):
    """{(cls, frac): sd of TP@K across seeds}, pooled over cells the same way."""
    out = {}
    for (c, f), grp in df.groupby(["cls", "frac"]):
        sds = grp.groupby("cell").tp.std(ddof=1).dropna()
        out[(c, f)] = float(np.sqrt((sds ** 2).mean())) if len(sds) else np.nan
    return out


def prizes(bar):
    """{(cls, frac): (K, p@K, prize items)} from the quality bar's own runs.

    `prize = (1 - p@K) * K` is the whole gap to a PERFECT ranking at that
    budget: no loss, dual, allocator or optimizer can win more than the items
    the current ranking has wrong inside the top K.
    """
    out = {}
    for (c, f), grp in bar.groupby(["cls", "frac"]):
        k = float(grp.k.mean())
        p = float((grp.tp / grp.k).mean())
        out[(c, f)] = (k, p, (1.0 - p) * k)
    return out


def seeds_needed(effect, sd, power_const=7.85):
    """Seeds per cell to detect `effect` against `sd` at 80% power, alpha .05.

    `8 * (sd/effect)**2` to the accuracy that matters here -- the answers span
    four orders of magnitude, so the constant is never the thing in doubt.

    This is the column that turns the table from a verdict into a decision.
    "The prize is below the noise" reads as closed everywhere; the seed count
    says WHERE it is closed. On iwc3 the FULL prize needs ~2600 seeds per cell
    at K/n = 0.2 and **~8 at K/n = 0.9** -- so the protocol's caps are hopeless
    and a loose cap is merely expensive. Quote it for the effect a method could
    plausibly capture, not for the whole prize: half the prize costs 4x the
    seeds.
    """
    if not effect or effect != effect or not sd or sd != sd:
        return float("nan")
    return power_const * (sd / effect) ** 2


def report(bar, floor_sd, treated_sd, classes, fracs, out=sys.stdout):
    """Print the table. Returns the number of (cls, frac) rows where the prize
    clears the TREATED noise, i.e. rows where a method could show something."""
    un = unpaired_sd(bar)
    pr = prizes(bar)
    worth = 0
    out.write("  %-4s %-6s %7s %8s %9s %9s %9s %9s %9s %8s\n"
              % ("cls", "K/n", "K", "prize", "unpaired", "reseed", "treated",
                 "pr/reseed", "pr/treated", "seeds"))
    for c in classes:
        for f in fracs:
            if (c, f) not in pr:
                continue
            k, p, prize = pr[(c, f)]
            u = un.get((c, f), float("nan"))
            a = floor_sd.get((c, f), float("nan"))
            b = treated_sd.get((c, f), float("nan"))
            ra = prize / a if a else float("nan")
            rb = prize / b if b else float("nan")
            if rb == rb and rb >= 1.0:
                worth += 1
            need = seeds_needed(prize, b)
            out.write("  %-4d %-5.0f%% %7.0f %8.2f %9.2f %9.2f %9.2f "
                      "%8.2fx %8.2fx %8s\n"
                      % (c, 100 * f, k, prize, u, a, b, ra, rb,
                         "-" if need != need else "%.0f" % need))
        out.write("\n")
    out.write("  reseed = RNG only, the floor under ANY paired contrast.\n"
              "  treated = the contrast actually run. If `treated` exceeds\n"
              "  `unpaired`, pairing is COSTING resolution, not buying it --\n"
              "  the two arms are two models rather than two readings of one.\n"
              "  seeds  = per cell, at 80% power, to detect the WHOLE prize.\n"
              "           A method capturing half of it costs 4x that.\n")
    if not worth:
        out.write("\n  NO row has a prize at or above the treated noise: at\n"
                  "  every K/n here, a method capturing 100%% of the gap to a\n"
                  "  perfect ranking would still not be detectable AT 4 SEEDS.\n"
                  "  Read the `seeds` column before calling any of it closed --\n"
                  "  it separates 'hopeless' from merely 'expensive', and those\n"
                  "  are different decisions.\n")
    return worth


def _synth(offsets, cells=3, seeds=4, cls=2, frac=0.2, n=100, k=20):
    """Build one arm's frame. `offsets[(cell_i, seed_i)]` gives its TP."""
    rows = []
    for ci in range(cells):
        for si in range(seeds):
            rows.append(("cell%d" % ci, "seed_%d" % si, cls, frac, n, k,
                         offsets(ci, si)))
    return pd.DataFrame(rows, columns=COLUMNS)


def self_test(out=sys.stdout):
    """The gate. A tool that can only ever report one answer is not a measurement.

    The load-bearing claim from this script is "pairing GROWS the noise here".
    That is only meaningful if the script CAN report a shrink, so the first
    case below is the liveness control: two arms differing by a constant per
    cell must come back with a paired sd of 0 against a large unpaired one.
    """
    ok = True

    # 1. LIVENESS: pairing must be able to help. Same seed-to-seed structure,
    #    offset by a constant within each cell => the difference is constant,
    #    so paired sd is exactly 0 while unpaired sd is large.
    base = _synth(lambda ci, si: 10 + 7 * si)
    same = _synth(lambda ci, si: 10 + 7 * si + 3)
    ps = paired_sd(same, base)[(2, 0.2)]
    us = unpaired_sd(base)[(2, 0.2)]
    if not (abs(ps) < 1e-9 and us > 1.0):
        out.write("SELF-TEST FAIL: pairing must be able to CANCEL shared "
                  "variation. paired=%.4f unpaired=%.4f\n" % (ps, us))
        ok = False

    # 2. The headline case: two INDEPENDENT arms. Paired sd must come back
    #    LARGER than either unpaired sd, by about sqrt(2).
    a = _synth(lambda ci, si: [10, 20, 30, 40][si])
    b = _synth(lambda ci, si: [40, 10, 30, 20][si])
    ps = paired_sd(a, b)[(2, 0.2)]
    ua = unpaired_sd(a)[(2, 0.2)]
    if not ps > ua:
        out.write("SELF-TEST FAIL: independent arms must give a paired sd "
                  "ABOVE the unpaired one. paired=%.4f unpaired=%.4f\n"
                  % (ps, ua))
        ok = False

    # 3. It must recover a KNOWN sd, not merely order two numbers.
    a = _synth(lambda ci, si: 100 + [0, 2, 4, 6][si])
    z = _synth(lambda ci, si: 100)
    want = float(np.std([0, 2, 4, 6], ddof=1))
    got = paired_sd(a, z)[(2, 0.2)]
    if abs(got - want) > 1e-9:
        out.write("SELF-TEST FAIL: known sd not recovered: got %.6f want "
                  "%.6f\n" % (got, want))
        ok = False

    # 3b. THE DUPLICATE-MODEL CENSUS. A `_null` / `_reseed` arm has lambda = 0
    #     and a post-hoc clipper trains with no constraint at all, so NEITHER
    #     can depend on the cap: run at two cap levels each produces two CELLS
    #     holding ONE model, and a sd pooled over them double-counts it and
    #     biases every ratio in the table OPTIMISTIC. Measured on
    #     vitdual1/ViTB16 2026-09-03: `clip` 4 runs / 2 models (seed_1
    #     identical across THREE cap levels), `tralo_null` and `tralo_reseed`
    #     3 / 2 -- while `tralo`, which is trained, stayed 3 / 3.
    dup = _synth(lambda ci, si: 100 + si)          # same curve in every cell
    if _run_census(dup) != "12 (4 distinct)":
        out.write("SELF-TEST FAIL: the census must collapse cells holding one "
                  "model, got %s\n" % _run_census(dup))
        ok = False
    #     NEGATIVE CONTROL, and it is the one that matters: a census that
    #     collapsed genuinely different runs would erase every real replicate
    #     while looking right on the case above. A TRAINED arm differs per
    #     cell and must stay uncollapsed.
    distinct = _synth(lambda ci, si: 100 + 10 * ci + si)
    if _run_census(distinct) != "12":
        out.write("SELF-TEST FAIL: the census collapsed runs that DIFFER, got "
                  "%s\n" % _run_census(distinct))
        ok = False

    # 4. Pooling must be WITHIN cell. A pure cell-to-cell shift is not noise
    #    and must not appear as any.
    shifted = _synth(lambda ci, si: 100 + 50 * ci)
    flat = _synth(lambda ci, si: 100)
    got = paired_sd(shifted, flat)[(2, 0.2)]
    if abs(got) > 1e-9:
        out.write("SELF-TEST FAIL: a cell-to-cell mean shift leaked into the "
                  "sd: %.6f, expected 0\n" % got)
        ok = False

    # 5. The seed count must be right, and must scale as the SQUARE of the
    #    effect -- the whole point of the column is that halving the effect
    #    quadruples the cost, which is the part people get wrong by eye.
    got = seeds_needed(10.0, 10.0)
    if abs(got - 7.85) > 1e-9:
        out.write("SELF-TEST FAIL: effect == sd must need ~8 seeds, got "
                  "%.3f\n" % got)
        ok = False
    if abs(seeds_needed(5.0, 10.0) / seeds_needed(10.0, 10.0) - 4.0) > 1e-9:
        out.write("SELF-TEST FAIL: halving the effect must QUADRUPLE the "
                  "seeds\n")
        ok = False
    if seeds_needed(0.0, 10.0) == seeds_needed(0.0, 10.0):
        out.write("SELF-TEST FAIL: a zero effect needs infinite seeds and "
                  "must report nan, not a number\n")
        ok = False

    # 6. The verdict must be able to say YES. A screen that can only refuse
    #    decides nothing.
    bar = _synth(lambda ci, si: 5 + si)            # p@K ~ 0.32 => big prize
    worth = report(bar, {(2, 0.2): 0.5}, {(2, 0.2): 0.5}, [2], [0.2],
                   out=open(os.devnull, "w"))
    if worth != 1:
        out.write("SELF-TEST FAIL: a large prize against a small noise must "
                  "count as worth running; got %d\n" % worth)
        ok = False

    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", help="campaign root, e.g. results/iwc3")
    ap.add_argument("--bar", default="clip",
                    help="arm the prize is measured off (default clip)")
    ap.add_argument("--control", default="tralo_null",
                    help="the lambda=0 twin both contrasts pair against")
    ap.add_argument("--floor", default="tralo_reseed",
                    help="RNG-only arm giving the noise floor")
    ap.add_argument("--treated", default="tralo",
                    help="the arm whose contrast is actually run")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 7])
    ap.add_argument("--fracs", type=float, nargs="+", default=DEFAULT_FRACS,
                    help="K/n levels to sweep (default 0.2 .. 0.9)")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="measure a campaign `scripts.quarantine` marked dead")
    ap.add_argument("--self-test", action="store_true",
                    help="check the tool against known-answer inputs")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.campaign:
        ap.error("give --campaign, or --self-test")
    if not os.path.isdir(args.campaign):
        print("no such campaign root: %s" % args.campaign)
        return 2
    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE,
    # so a marker on a dead campaign prevented nothing here. No fallback
    # import -- if the gate cannot load, the tool must break.
    from scripts.quarantine import gate
    blocked, dead = gate([args.campaign], args.allow_quarantined, "measure")
    if blocked:
        return 1

    frames = {}
    for name in (args.bar, args.control, args.floor, args.treated):
        frames[name] = load_arm(args.campaign, name, args.classes, args.fracs)
    per = len(args.classes) * len(args.fracs)
    print("runs: " + "  ".join(
        "%s %s" % (n, _run_census(f)) for n, f in frames.items()))
    _warn_duplicate_models(frames)

    missing = [n for n, f in frames.items() if f.empty]
    if missing:
        print("REFUSING: no runs found for %s. This tool measures the noise a\n"
              "  PAIRED contrast faces, so it cannot fall back to an unpaired\n"
              "  number -- that substitution is the defect it exists to catch."
              % ", ".join(missing))
        return 2

    print("")
    worth = report(frames[args.bar],
                   paired_sd(frames[args.floor], frames[args.control]),
                   paired_sd(frames[args.treated], frames[args.control]),
                   args.classes, args.fracs)
    return 0 if worth else 1


if __name__ == "__main__":
    sys.exit(main())
