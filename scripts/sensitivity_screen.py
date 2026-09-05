"""IS THIS PAIR SENSITIVE TO CONSTRAINT TRAINING, OR IS THE MODEL SATURATED?

Run this on the FIRST completed runs of a campaign, before it earns another
hour of GPU. It answers a question no other instrument here asks:

    Given the model this (dataset, backbone, cap) pair actually produces, can
    the constraint phase change ANYTHING -- and did the arms in fact separate?

\U0001f6d1 WHY IT EXISTS, AND WHAT IT MEASURED THE FIRST TIME IT RAN. Every arm
ties, and a grid that cannot separate its arms is a week spent measuring
nothing. Run over dom1 + taskwin2 + equaldose1 on 2026-09-04 -- 28 cells,
792 runs, the whole clean four-dual corpus -- it returned:

    SENSITIVE 0    UNDER-POWERED 27    SATURATED 1

\U0001f511 AND THE REASON IS NOT THE ONE EVERYONE REACHES FOR. The model IS
saturating globally: dom1/MobileNetV2 train accuracy runs 0.9595 at warm-up
exit to 0.9992 at the end, so CE keeps sharpening all the way through the 29
constraint epochs. But at loose caps the CUT is not in saturated territory at
all -- p@cut is 0.41-0.65 across most cells, and where it IS high (0.9943 on
dom1/MobileNetV3 class 7) the DECISION BOUNDARY is wide open at p(1-p) = 0.248.
That cell is a cut-placement result, not a frozen model, which is exactly the
distinction FRAMEWORK section 4 insists on.

What actually stops these cells is arithmetic: the typical arm-PAIR difference
is 2.0-5.0 deployed TP items and the RNG floor in the same cell is 1.5-7.5.
They are the same size. And the floor itself rests on FOUR observations,
because every campaign here carries exactly one `_null`/`_reseed` pair at four
seeds -- so the noise these comparisons are judged against is a median of four
numbers whose confidence interval is the sample range.

⛔ SO "DE-SATURATE THE MODEL" DOES NOT FOLLOW FROM THIS TABLE, and FRAMEWORK
2(j) has the counter-argument ready: post-hoc allocation is optimal for
expected TP GIVEN the probabilities, and that optimality is distribution-free,
so a worse model raises the headroom for `clip` by the same amount it raises it
for a trained arm. A bigger prize is not a bigger GAP. Any intervention aimed
at this table has to pre-register why it moves the gap.

THREE AXES, THREE DIFFERENT FAILURES, AND THEY ARE NOT INTERCHANGEABLE:

    GRADIENT   p(1-p) at the per-group cut, on the reference arm's STORED
               probabilities. This is literally the per-item scale of the
               constraint gradient at the point the metric reads. Bar is
               `GRAD_MIN`, and it is not invented -- it is exactly
               `task_window.WIGGLE_MAX` (0.99) pushed through `p(1-p)`, so the
               two instruments cannot drift apart.

               \U0001f6d1 IT IS MEASURED AT THE **END** OF THE CONSTRAINT PHASE,
               SO IT IS A LOWER BOUND. The stored probabilities are the final
               model's. `reachability` records the same quantity falling ~60x
               over the phase (0.055 at the start of L50_G30, 0.0009 at 30
               epochs), so a cell that reads SATURATED here was at least this
               dead at the end and may have been livelier at the start. Read a
               SATURATED verdict as "the constraint had nothing left to push on
               by the end", never as "it never had anything".

               \U0001f511 AND THERE ARE ALREADY TWO BARS FOR THIS QUANTITY,
               8x apart: `reachability.REACHABLE` = 0.040 and
               `cut_gap.DEAD_SLOPE` = 0.005. This one is 0.0099 -- between
               them, and derived rather than chosen, so that a cell called
               saturated here is exactly a cell `task_window` would call
               saturated. Quote which bar you mean.

    BAND       how many of the class's items sit at a probability the model is
               not already certain of. A gradient of any size reorders nothing
               if there is nothing near the cut to reorder. Bar is
               `task_window.MIN_PRIZE` items, for the same reason that bar
               exists: below it, a PERFECT method beats a reseed of itself by
               less than the reseed noise.

    SPREAD     the typical ARM-PAIR difference in deployed per-class TP against
               the RNG floor MEASURED IN THE SAME CELL (|null - reseed|). This
               is the only one of the three that can say the arms did separate,
               and it is the only one that needs more than one arm.

\U0001f6d1 SPREAD IS A PAIRWISE STATISTIC, AND THAT IS NOT A STYLE CHOICE. The
obvious version -- `max(arm means) - min(arm means)` -- is a RANGE over k arms,
and a range grows with k under pure noise: for k normal samples it runs about
`sd*sqrt(2 ln k)`, so ~3.1*sd at k=10, against `E|X-Y| = 1.13*sd` for the
two-arm floor it would be compared against. That ratio is ~2.7x BEFORE any
method does anything, so a `range >= floor` bar would certify a cell of pure
noise as sensitive. Both sides are therefore the same statistic: the median
absolute per-seed difference between TWO arms. This also makes the number
directly comparable to `deployed_h2h`, which reports exactly that pair
(|tralo - rival| against |tralo - tralo_reseed|).

\U0001f511 THE VERDICT IS FOUR-WAY, NOT TWO. "Nothing moved" and "we could not
have seen it move" are opposite conclusions from the same table, and this
project's standing rule is that they must never be collapsed:

    SENSITIVE          gradient live, band non-empty, spread over the floor
    SATURATED          gradient dead or band empty -- a MEASUREMENT that
                       nothing could have moved, independent of seed count
    UNDER-POWERED      gradient live, but the spread is under the floor and
                       there are too few seeds to call it. Says how many.
    NOT DIFFERENTIATED gradient live, seeds sufficient, spread still under the
                       floor. This is a real null about the CELL.

⚠️ WHAT IT IS NOT. It does not price a METHOD -- `paired_noise` and
`ceiling_screen` do that. It does not choose a cap -- `task_window` does, and
this tool deliberately reads the cap the cell ACTUALLY ran rather than sweeping
fractions, because the question here is about a campaign that exists. And its
GRADIENT axis is model-intrinsic: whether our particular step size can reach
the cut is `straddle_probe`'s question, not this one.

⚠️ THE TRAINING-LOG TRAJECTORY IS SINGLE-ARM, ON PURPOSE. The arms write
different log schemas (tralo* 76 columns, hounie 16, alm 15, fioretto 14) and
only the tralo family logs `Train_Acc`. Reading a count across those schemas
gave the exact opposite of the truth once already (FRAMEWORK 3(0c)), so the
trajectory is reported for ONE named arm and never compared across arms.
"""
import argparse
import csv
import glob
import json
import os
import sys
from io import StringIO as _StringIO

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imported, never reimplemented. A screen that rounds the budget differently
# from the trainer screens a cap the campaign does not run, and a screen with
# its own power formula disagrees with the tables it is supposed to gate.
from src.training.constraints import (cap_fraction_for,  # noqa: E402
                                      normalize_constrained_classes)
from scripts.task_window import (MIN_PRIZE, WIGGLE_MAX,  # noqa: E402
                                 load, select_local)
from scripts.paired_noise import seeds_needed  # noqa: E402
from scripts import quarantine  # noqa: E402

# The gradient bar is WIGGLE_MAX expressed in the units that matter. p(1-p) is
# what multiplies every per-item constraint gradient, so a cut at the existing
# saturation bar and this bar are the same statement about the same cell.
GRAD_MIN = WIGGLE_MAX * (1.0 - WIGGLE_MAX)      # 0.0099 at WIGGLE_MAX = 0.99

# An item counts as contestable when the model is not already certain of it.
# The band is deliberately WIDER than the cut neighbourhood: it measures the
# model's total uncertainty in this class, which is what saturation destroys.
BAND_LO, BAND_HI = 0.05, 0.95
BAND_MIN = MIN_PRIZE                            # 3.0 items, same bar, same reason

# One source of truth, shared with `deployed_h2h`; see scripts/floors.py.
from scripts.floors import MIN_FLOOR_OBS  # noqa: E402

# Arms that carry NO constraint term, in the order we prefer them as the
# unconstrained reference. A trained arm must never be used: its probabilities
# are the thing under test.
REFERENCE_ARMS = ("tralo_null", "alm_null", "fioretto_null", "hounie_null",
                  "clip", "lp")
# The RNG floor is `<fam>_null` vs `<fam>_reseed` -- BOTH lambda=0, differing
# in the RNG stream and nothing else.
#
# \U0001f6d1 NOT `tralo` vs `tralo_reseed`, which is the mistake
# `deployed_h2h.rng_floor` was corrected for on 2026-09-02. `tralo_reseed` is
# built from `[constraint_phase, tralo_null, tralo_reseed]` and so INHERITS
# `lambda_step: 0.0`; pairing it with the TREATED arm puts the very effect
# being measured inside the floor it is measured against. The error runs the
# unintuitive way -- the contaminated floor is too LOW (median 4.0 vs 6.5 on
# dom1), so it refuses too seldom. Do not "fix" it back.
FLOOR_FAMILIES = ("tralo", "alm", "fioretto", "hounie")
# Arms excluded from the cross-arm comparison: a `_reseed` arm DEFINES the
# floor, so letting it widen the spread it is compared against would make
# every cell look differentiated.
def _is_floor_control(arm):
    return arm.endswith("_reseed")

VERDICTS = ("SENSITIVE", "SATURATED", "UNDER-POWERED", "NOT DIFFERENTIATED",
            "NO DATA")


# ---------------------------------------------------------------- discovery
def cell_key(run_dir, root):
    """(campaign, backbone, dataset, cap, arm, seed) from the run's path.

    The layout is <root>/<Backbone>/<dataset>/<cap>/<arm>/seed_<n>. Parsed
    from the path rather than the config because a campaign is identified by
    where it lives; the config is then read for the values that matter.
    """
    rel = os.path.relpath(run_dir, root).replace("\\", "/").split("/")
    if len(rel) < 5:
        return None
    backbone, dataset, cap, arm, seed = rel[:5]
    return (os.path.basename(root.rstrip("/\\")), backbone, dataset, cap,
            arm, seed)


def completed_runs(root):
    """Every run under `root` whose config says `completed`, with its config."""
    out = []
    for cfg_path in sorted(glob.glob(os.path.join(
            root, "*", "*", "*", "*", "seed_*", "config.json"))):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        if cfg.get("status") != "completed":
            continue
        run_dir = os.path.dirname(cfg_path)
        key = cell_key(run_dir, root)
        if key is not None:
            out.append((key, run_dir, cfg))
    return out


def capped_classes(cfg):
    """The classes this run caps, from its own config. No default.

    `src/pipeline/data.py` raises rather than defaulting here, for the reason
    given there: a silent cap on whichever class happens to be last looks
    completely normal in every log. The screen inherits that refusal.
    """
    ds = cfg.get("dataset_config") or {}
    if "constrained_class" not in ds:
        raise KeyError(
            "dataset_config.constrained_class missing from the run config; "
            "the screen will not guess which class was capped")
    return normalize_constrained_classes(ds["constrained_class"])


def local_fraction(cfg, cls, classes):
    """This cell's ACTUAL local cap fraction for one class.

    `constraint` is `[local, global]`, where local is a scalar or one value
    per capped class. `cap_fraction_for` is the trainer's own mapping.
    """
    con = cfg.get("constraint")
    if not con:
        raise KeyError("run config carries no `constraint`")
    return cap_fraction_for(con[0], cls, classes)


# -------------------------------------------------------------- measurement
def gradient_at_cut(run_dir, cls, frac):
    """p(1-p) at the cut, and how many items are still contestable.

    The cut is the per-GROUP one the real allocator uses. A global top-K
    overstated this project's prize by 4.25x once already (`select_local`), so
    the same selection is used here.

    Returns None when the predictions carry no `Group_ID`: a one-group reading
    IS the global top-K, and printing it under a local heading is the defect
    that mistake consisted of.
    """
    y, pr, groups, hard = load(run_dir, cls, raw=True)
    if groups is None:
        return None
    sel, cuts, budgets, n_zero = select_local(y, pr, groups, cls, frac)
    if not cuts:
        # Every group's ceiling rounded to zero: the cap emits nothing, so
        # there is no cut and no gradient to measure. Reported, not skipped.
        return dict(p_cut=None, grad=0.0, band=0, hard=hard, n_sel=0,
                    n_zero=n_zero, p_bd=None, grad_bd=None,
                    reason="every per-group budget is zero")
    # Budget-weighted, matching how task_window summarises the same quantity:
    # the mean cut an emitted ITEM sits at, not the mean over groups.
    w = np.asarray(budgets, dtype=float)
    p_cut = float(np.average(np.asarray(cuts, dtype=float), weights=w))
    band = int(np.count_nonzero((pr >= BAND_LO) & (pr <= BAND_HI)))
    # The DECISION BOUNDARY, which is a different item from the cut and must
    # be reported beside it -- see `classify`. It is rank `hard` in this
    # class's ranking: the last item the argmax still assigns to `cls`.
    p_bd = grad_bd = None
    if 0 < hard <= len(pr):
        p_bd = float(np.sort(pr)[::-1][hard - 1])
        grad_bd = p_bd * (1.0 - p_bd)
    return dict(p_cut=p_cut, grad=p_cut * (1.0 - p_cut), band=band,
                hard=hard, n_sel=int(len(sel)), n_zero=n_zero, reason=None,
                p_bd=p_bd, grad_bd=grad_bd)


def deployed_tp(run_dir, cls):
    """True positives for `cls` in what the run would actually DEPLOY.

    `final_predictions.csv` is the as-deployed allocation. The panel re-derives
    its own equal-budget allocation and is allocator-blind by construction, so
    it cannot answer a question about what the arms emitted.
    """
    path = os.path.join(run_dir, "final_predictions.csv")
    if not os.path.exists(path):
        return None
    tp = 0
    with open(path) as f:
        for r in csv.DictReader(f):
            tp += int(int(r["Predicted_Label"]) == cls
                      and int(r["True_Label"]) == cls)
    return tp


def train_trajectory(run_dir):
    """(first, last) training accuracy over the constraint phase, or None.

    Single-arm by construction -- see the module docstring. Returns None when
    this arm's log schema has no `Train_Acc`, which is the honest answer for
    every dual arm, rather than a zero that would read as collapse.
    """
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("Train_Acc") not in ("", None)]
    if len(rows) < 2:
        return None
    return float(rows[0]["Train_Acc"]), float(rows[-1]["Train_Acc"])


# ------------------------------------------------------------------ verdict
def classify(grad, band, spread, floor, n_seeds, reason=None, n_floor=None,
             grad_bd=None):
    """Four-way verdict plus the criterion that decided it.

    ORDER MATTERS. Saturation is checked FIRST and is independent of the seed
    count: if nothing could have moved, more seeds measure the same nothing
    more precisely. Only once the cell is capable of responding does the
    question "did the arms separate, and could we have seen it" arise.
    """
    if grad is None:
        return "NO DATA", reason or "no per-group cut could be computed"
    if grad < GRAD_MIN:
        # \U0001f6d1 SAY WHICH POINT. FRAMEWORK 4 is explicit that rank K and the
        # decision boundary are different items and that "the gradient cannot
        # reach the cut" is wrong when stated without one. With a hard count of
        # 300 against K=44 the boundary is at item 300 and rank 44 is buried
        # inside the class, where p(1-p) is near its MAXIMUM. So a dead CUT is
        # only half the statement, and the boundary figure travels with it.
        bd = ("" if grad_bd is None else
              "; at the DECISION BOUNDARY it is %.5f%s"
              % (grad_bd, " -- live there, so this is a CUT-PLACEMENT "
                          "result, not a saturated model"
                 if grad_bd >= GRAD_MIN else " -- dead there too"))
        return "SATURATED", ("p(1-p) at the cut is %.5f, under the %.5f bar%s"
                             % (grad, GRAD_MIN, bd))
    if band < BAND_MIN:
        return "SATURATED", ("only %d item(s) lie in [%.2f, %.2f]; below the "
                             "%.1f-item bar nothing is contestable"
                             % (band, BAND_LO, BAND_HI, BAND_MIN))
    if floor is None or spread is None:
        return "NO DATA", ("the cell carries no <family>_null / <family>_reseed "
                           "pair, so the RNG floor is unmeasured and the "
                           "spread cannot be judged")
    if n_floor is not None and n_floor < MIN_FLOOR_OBS:
        return "UNDER-POWERED", (
            "the RNG floor itself rests on %d observation(s), under the %d "
            "bar; its median could be off by the width of the sample, so "
            "spread %.1f vs floor %.1f decides nothing. Buy observations with "
            "SEEDS, or with a distinct RNG STREAM (`tralo_reseed2`) -- NOT "
            "with another family's `_reseed`, which is byte-identical because "
            "lambda=0 makes every family plain CE (FRAMEWORK 2(z41))."
            % (n_floor, MIN_FLOOR_OBS, spread, floor))
    if spread >= floor:
        return "SENSITIVE", ("cross-arm spread %.1f items over an RNG floor of "
                             "%.1f (%.2fx)" % (spread, floor,
                                               spread / floor if floor else
                                               float("inf")))
    need = seeds_needed(spread, floor)
    if need != need or need > n_seeds:
        return "UNDER-POWERED", (
            "spread %.1f items under a floor of %.1f; %s seeds/cell would be "
            "needed at 80%% power and %d are present"
            % (spread, floor,
               "?" if need != need else "%.0f" % need, n_seeds))
    return "NOT DIFFERENTIATED", (
        "spread %.1f items under a floor of %.1f with %d seeds, which is "
        "enough to have seen it" % (spread, floor, n_seeds))


def screen_cell(runs, cls):
    """Measure one (campaign, backbone, dataset, cap) cell for one class.

    `runs` maps arm -> {seed: (run_dir, cfg)}.
    """
    classes_by_arm = {}
    ref_arm = ref_dir = None
    for arm in REFERENCE_ARMS:
        if arm in runs and runs[arm]:
            ref_arm = arm
            ref_dir = sorted(runs[arm].items())[0][1][0]
            break
    if ref_dir is None:
        return dict(verdict="NO DATA", why="no unconstrained reference arm "
                    "(%s) in this cell" % "/".join(REFERENCE_ARMS[:3]),
                    ref_arm=None)

    cfg = sorted(runs[ref_arm].items())[0][1][1]
    classes = capped_classes(cfg)
    if cls not in classes:
        return dict(verdict="NO DATA",
                    why="class %d is not capped in this cell (capped: %s)"
                        % (cls, classes), ref_arm=ref_arm)
    frac = local_fraction(cfg, cls, classes)
    classes_by_arm[ref_arm] = classes

    g = gradient_at_cut(ref_dir, cls, frac)
    if g is None:
        return dict(verdict="NO DATA", ref_arm=ref_arm, frac=frac,
                    why="predictions carry no Group_ID, so the per-group cut "
                        "this allocator actually uses cannot be reconstructed")

    # --- deployed TP per arm per seed, read once
    tp = {}
    for arm, seeds in runs.items():
        per = {}
        for seed, (d, _cfg) in seeds.items():
            v = deployed_tp(d, cls)
            if v is not None:
                per[seed] = v
        if per:
            tp[arm] = per

    def pair_gaps(a, b):
        """Every seed-paired |a - b|, one per SHARED seed.

        Seed-paired, because the two arms share a warm-up and the RNG stream
        is the axis being controlled. An unpaired difference would mix the
        seed effect into both the floor and the spread.

        Returns the raw values, NOT a summary -- the two sides of this
        comparison are estimated from very different numbers of observations,
        and collapsing each to a median first hides that.
        """
        shared = sorted(set(tp.get(a, {})) & set(tp.get(b, {})))
        return [abs(tp[a][s] - tp[b][s]) for s in shared]

    # --- the RNG floor, measured in THIS cell and no other, per FAMILY
    floor_vals = [v for fam in FLOOR_FAMILIES
                  for v in pair_gaps(fam + "_null", fam + "_reseed")]
    floor = float(np.median(floor_vals)) if floor_vals else None

    # --- the typical ARM-PAIR difference. Same statistic as the floor: the
    #     median absolute seed-paired difference between TWO arms. See the
    #     module docstring for why a max-min RANGE is not comparable to it.
    arms = sorted(a for a in tp if not _is_floor_control(a))
    pair_vals = [v for i, a in enumerate(arms) for b_ in arms[i + 1:]
                 for v in pair_gaps(a, b_)]
    spread = float(np.median(pair_vals)) if pair_vals else None
    means = {a: float(np.mean(list(tp[a].values()))) for a in arms}
    n_seeds = max((len(tp[a]) for a in arms), default=0)
    rng_range = (max(means.values()) - min(means.values())) if len(means) > 1 \
        else None

    verdict, why = classify(g["grad"], g["band"], spread, floor, n_seeds,
                            reason=g.get("reason"), n_floor=len(floor_vals),
                            grad_bd=g.get("grad_bd"))
    traj = train_trajectory(ref_dir)
    return dict(verdict=verdict, why=why, ref_arm=ref_arm, frac=frac,
                p_cut=g["p_cut"], grad=g["grad"], band=g["band"],
                hard=g["hard"], n_sel=g["n_sel"], zero_budget=g["n_zero"],
                spread=spread, floor=floor, n_seeds=n_seeds,
                n_arms=len(means), trajectory=traj, arm_means=means,
                rng_range=rng_range, n_floor=len(floor_vals),
                p_bd=g.get("p_bd"), grad_bd=g.get("grad_bd"))


def screen(roots, classes=None, dead=()):
    """Every cell under every root, one row per (cell, capped class).

    `dead` is the arm set a PARTIAL quarantine marker disqualifies. The
    SPREAD axis is a typical ARM-PAIR difference, so leaving a dead arm in
    silently prices every verdict off a contrast the marker calls not
    comparable.
    """
    rows = []
    for root in roots:
        cells = {}
        here = completed_runs(root)
        keep = set(quarantine.drop_dead_runs(
            [d for _k, d, _c in here], dead, label="completed run"))
        for (camp, backbone, dataset, cap, arm, seed), run_dir, cfg in here:
            if run_dir not in keep:
                continue
            cells.setdefault((camp, backbone, dataset, cap), {}) \
                 .setdefault(arm, {})[seed] = (run_dir, cfg)
        for key, runs in sorted(cells.items()):
            any_cfg = next(iter(next(iter(runs.values())).values()))[1]
            try:
                cls_list = classes or capped_classes(any_cfg)
            except KeyError as exc:
                rows.append(dict(cell=key, cls=None, verdict="NO DATA",
                                 why=str(exc)))
                continue
            for cls in cls_list:
                r = screen_cell(runs, cls)
                r["cell"], r["cls"] = key, cls
                rows.append(r)
    return rows


# ------------------------------------------------------------------- report
def report(rows, out=sys.stdout):
    def w(s=""):
        out.write(s + "\n")

    w("SENSITIVITY SCREEN -- can the constraint phase change anything here?")
    w("  gradient bar p(1-p) >= %.5f   (= WIGGLE_MAX %.2f pushed through p(1-p))"
      % (GRAD_MIN, WIGGLE_MAX))
    w("  band bar     >= %.1f items in [%.2f, %.2f]   (= task_window MIN_PRIZE)"
      % (BAND_MIN, BAND_LO, BAND_HI))
    w()
    w("  spread and floor are BOTH the median |seed-paired difference| between")
    w("  TWO arms, in deployed TP items. `range` is max-min over arm means and")
    w("  is INFORMATION ONLY -- a range grows with the number of arms under")
    w("  pure noise, so it is not comparable to a two-arm floor.")
    w()
    hdr = ("%-10s %-13s %-12s %3s  %6s %8s %6s  %6s %6s %5s %5s  %s"
           % ("campaign", "backbone", "cap", "cls", "p@cut", "p(1-p)",
              "band", "spread", "floor", "range", "seeds", "verdict"))
    w(hdr)
    w("-" * len(hdr))
    tally = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        camp, backbone, _ds, cap = r["cell"]
        def num(key, f, wid):
            v = r.get(key)
            return (f % v) if isinstance(v, float) else ("%%%ds" % wid) % "-"
        w("%-10s %-13s %-12s %3s  %s %s %6s  %s %s %s %5s  %s"
          % (camp[:10], backbone[:13], cap[:12],
             r["cls"] if r["cls"] is not None else "-",
             num("p_cut", "%6.4f", 6), num("grad", "%8.5f", 8),
             ("%6d" % r["band"]) if isinstance(r.get("band"), int)
             else "%6s" % "-",
             num("spread", "%6.1f", 6), num("floor", "%6.1f", 6),
             num("rng_range", "%5.1f", 5),
             r.get("n_seeds", "-"), r["verdict"]))
    w()
    for r in rows:
        if r["verdict"] != "SENSITIVE":
            camp, backbone, _ds, cap = r["cell"]
            w("  %s/%s/%s class %s: %s" % (camp, backbone, cap, r["cls"],
                                           r.get("why", "")))
    w()
    w("TALLY: " + ", ".join("%s %d" % (k, tally[k])
                            for k in VERDICTS if k in tally))
    ok = tally.get("SENSITIVE", 0)
    if not ok:
        w()
        w("NOT ONE CELL IS SENSITIVE.")
        # 🛑 AND THE REASON DECIDES THE ACTION. This block used to say "change
        # the pair" whatever the reason, which is the exact collapse this tool
        # exists to prevent: SATURATED means the cell cannot show an effect and
        # the pair is wrong; UNDER-POWERED means we could not SEE one and the
        # pair may be perfectly good. On vitdual2 -- task cells, live gradient
        # p(1-p) 0.019-0.114, full dose -- the honest advice is `add seeds`,
        # and `change the pair` would have thrown away the best cells in the
        # project.
        sat = tally.get("SATURATED", 0)
        und = tally.get("UNDER-POWERED", 0)
        nod = tally.get("NOT DIFFERENTIATED", 0)
        if sat and not und:
            w("Every cell is SATURATED: the cut sits where p(1-p) is too small "
              "for the penalty")
            w("to move anything. CHANGE THE PAIR -- more seeds cannot buy a "
              "gradient that")
            w("is not there.")
        elif und and not sat:
            w("Every cell is UNDER-POWERED, which is NOT the same finding. The "
              "cells may be")
            w("fine; we could not have SEEN an effect at this many "
              "observations. Buy seeds")
            w("(`scripts.add_seeds`) and re-run this before changing anything "
              "about the design.")
        elif nod and not (sat or und):
            w("Every cell is NOT DIFFERENTIATED: adequately powered and the "
              "arms still agree.")
            w("That is a real null about the METHODS, not about the cells.")
        else:
            w("A full grid on this pair cannot distinguish two methods as it "
              "stands, but the")
            w("cells do not agree on WHY (%d saturated, %d under-powered, %d "
              "not differentiated)." % (sat, und, nod))
            w("Read the per-cell reasons above -- they call for opposite "
              "actions.")
    for r in rows:
        if r.get("trajectory"):
            camp, backbone, _ds, cap = r["cell"]
            f, l = r["trajectory"]
            w("  train acc over the constraint phase (%s, %s/%s/%s): "
              "%.4f -> %.4f%s" % (r["ref_arm"], camp, backbone, cap, f, l,
                                  "   <- CE is still sharpening"
                                  if l - f > 0.005 else ""))
            break
    return tally


# ----------------------------------------------------------------- self test
def self_test(out=sys.stdout):
    """Every bar is exercised in BOTH directions. A gate that has never
    failed has never been shown to work."""
    import shutil
    import tempfile

    fails = []

    def check(name, cond):
        out.write("  [%s] %s\n" % ("PASS" if cond else "FAIL", name))
        if not cond:
            fails.append(name)

    base = tempfile.mkdtemp(prefix="senscreen_")
    try:
        def write_run(root, backbone, cap, arm, seed, pr, y, groups, pred,
                      frac=0.9, cls=2, n_classes=3, train_acc=None):
            d = os.path.join(root, backbone, "iwildcam", cap, arm,
                             "seed_%d" % seed)
            os.makedirs(d, exist_ok=True)
            for name in ("final_predictions_raw.csv", "final_predictions.csv"):
                with open(os.path.join(d, name), "w", newline="") as f:
                    cols = (["True_Label", "Predicted_Label", "Group_ID"]
                            + ["Prob_Class_%d" % c for c in range(n_classes)])
                    wr = csv.writer(f)
                    wr.writerow(cols)
                    for i in range(len(y)):
                        probs = [0.0] * n_classes
                        probs[cls] = float(pr[i])
                        probs[(cls + 1) % n_classes] = 1.0 - float(pr[i])
                        wr.writerow([int(y[i]), int(pred[i]), groups[i]]
                                    + ["%.10f" % p for p in probs])
            with open(os.path.join(d, "config.json"), "w") as f:
                json.dump(dict(status="completed", constraint=[frac, 0.95],
                               methodology=arm, model_name=backbone,
                               dataset_config=dict(constrained_class=[cls])),
                          f)
            if train_acc is not None:
                with open(os.path.join(d, "training_log.csv"), "w",
                          newline="") as f:
                    wr = csv.writer(f)
                    wr.writerow(["Epoch", "Train_Acc"])
                    for e, a in enumerate(train_acc):
                        wr.writerow([e + 1, a])
            return d

        # -- fixture A: SATURATED. Every probability is pinned at the extremes,
        #    so the cut sits at p ~ 1 and nothing is contestable.
        n = 60
        y = np.array([2] * 30 + [0] * 30)
        groups = np.array(["g%d" % (i % 3) for i in range(n)])
        pred = np.where(y == 2, 2, 0)
        sat_pr = np.where(y == 2, 0.99999, 1e-5)
        rootA = os.path.join(base, "sat")
        for arm in ("tralo_null", "tralo_reseed", "tralo", "clip"):
            for s in (1, 2, 3, 4):
                write_run(rootA, "MobileNetV3", "L90_G95", arm, s,
                          sat_pr, y, groups, pred,
                          train_acc=[0.97, 0.99, 0.998])
        rows = screen([rootA])
        rA = [r for r in rows if r["cls"] == 2][0]
        check("a pinned-at-1.0 model is SATURATED (p@cut %.5f, grad %.2e)"
              % (rA["p_cut"], rA["grad"]), rA["verdict"] == "SATURATED")
        check("  and it says WHICH bar failed",
              "p(1-p)" in rA["why"] or "contestable" in rA["why"])
        check("  and it reports the training trajectory it read",
              rA["trajectory"] == (0.97, 0.998))

        # -- NEGATIVE CONTROL for the gradient bar: same shape, same counts,
        #    same everything -- only the confidence is pulled off the rail.
        #    If this also came back SATURATED the bar would be reading
        #    something other than saturation.
        live_pr = np.where(y == 2,
                           np.linspace(0.80, 0.55, n),
                           np.linspace(0.45, 0.10, n))
        rootB = os.path.join(base, "live")
        rng = np.random.RandomState(0)
        for arm in ("tralo_null", "tralo_reseed", "tralo", "clip"):
            for s in (1, 2, 3, 4):
                # arms differ by a real margin; the reseed differs by ~1 item
                shift = {"tralo": 0.06, "clip": -0.05,
                         "tralo_reseed": 0.002}.get(arm, 0.0)
                p = np.clip(live_pr + shift + rng.normal(0, 1e-4, n), 0.01, 0.99)
                pr_pred = np.where(p >= 0.5, 2, 0)
                write_run(rootB, "MobileNetV3", "L90_G95", arm, s,
                          p, y, groups, pr_pred)
        rows = screen([rootB])
        rB = [r for r in rows if r["cls"] == 2][0]
        check("NEGATIVE CONTROL: an unsaturated model is NOT saturated "
              "(p@cut %.4f, grad %.4f, band %d)"
              % (rB["p_cut"], rB["grad"], rB["band"]),
              rB["verdict"] != "SATURATED")

        # -- the two fixtures differ ONLY in confidence, so the verdict must
        #    have come from the confidence and not from the counts.
        check("  the two fixtures share their labels, groups and cap",
              rA["frac"] == rB["frac"] and rA["hard"] == rB["hard"])

        # -- fixture C: UNDER-POWERED vs NOT DIFFERENTIATED. Same tiny spread,
        #    different seed counts -- the verdict MUST change, or the tool is
        #    collapsing "no effect" into "not enough seeds".
        vC = classify(grad=0.2, band=50, spread=0.5, floor=4.0, n_seeds=4)
        vD = classify(grad=0.2, band=50, spread=0.5, floor=4.0, n_seeds=600)
        check("a small spread at 4 seeds is UNDER-POWERED (%s)" % vC[0],
              vC[0] == "UNDER-POWERED")
        check("  the SAME spread at 600 seeds is NOT DIFFERENTIATED (%s)"
              % vD[0], vD[0] == "NOT DIFFERENTIATED")
        check("  and the under-powered branch prices the seeds",
              "seeds/cell" in vC[1])

        # -- saturation OUTRANKS the seed count: no number of seeds rescues a
        #    cell where nothing could have moved.
        vE = classify(grad=1e-4, band=50, spread=0.5, floor=4.0, n_seeds=10000)
        check("saturation beats any seed count (%s)" % vE[0],
              vE[0] == "SATURATED")

        # -- the band bar fires on its own, with the gradient healthy.
        vF = classify(grad=0.2, band=1, spread=99.0, floor=1.0, n_seeds=4)
        check("an empty contestable band is SATURATED even at a wide spread "
              "(%s)" % vF[0], vF[0] == "SATURATED")

        # -- a live cell with a real spread must come back SENSITIVE, or the
        #    tool can only ever say no.
        vG = classify(grad=0.2, band=50, spread=12.0, floor=4.0, n_seeds=4)
        check("LIVENESS: the tool CAN say SENSITIVE (%s)" % vG[0],
              vG[0] == "SENSITIVE")

        # -- the floor control must not be allowed to widen the spread it
        #    defines. Removing it from the arm set is load-bearing.
        check("the reseed control is excluded from the cross-arm spread",
              "tralo_reseed" not in rB["arm_means"]
              and "tralo_null" in rB["arm_means"])

        # -- THE RANGE TRAP, and this is the control for the statistic itself.
        #    Ten arms drawn from ONE noise distribution: no arm differs from
        #    any other by anything but the RNG. A `max - min` range over ten
        #    such arms runs ~3.1*sd against a two-arm floor's ~1.13*sd, so a
        #    `range >= floor` bar CERTIFIES THIS CELL AS SENSITIVE. The
        #    pairwise statistic must refuse it -- and the test asserts BOTH
        #    halves, so it fails if the trap stops being a trap.
        rootD = os.path.join(base, "noise")
        rng2 = np.random.RandomState(7)
        noise_arms = ["tralo_null", "tralo_reseed", "tralo", "clip",
                      "focal_clip", "lp", "alm", "fioretto", "hounie",
                      "tralo_uniform"]
        for arm in noise_arms:
            for s in (1, 2, 3, 4):
                pred_n = np.zeros(n, dtype=int)
                pos = np.flatnonzero(y == 2)
                ntp = 25 + rng2.randint(-2, 3)   # identical law for EVERY arm
                pred_n[rng2.choice(pos, ntp, replace=False)] = 2
                write_run(rootD, "MobileNetV3", "L90_G95", arm, s,
                          live_pr, y, groups, pred_n)
        rD = [r for r in screen([rootD]) if r["cls"] == 2][0]
        check("TRAP HALF 1: with 10 iid-noise arms the RANGE does exceed the "
              "floor (range %.1f > floor %.1f)"
              % (rD["rng_range"], rD["floor"]), rD["rng_range"] > rD["floor"])
        check("TRAP HALF 2: the PAIRWISE spread does not, so pure noise is "
              "NOT called SENSITIVE (spread %.1f vs floor %.1f -> %s)"
              % (rD["spread"], rD["floor"], rD["verdict"]),
              rD["verdict"] != "SENSITIVE")
        check("  and the cell is live, so it was refused on the spread and "
              "not on saturation (grad %.4f)" % rD["grad"],
              rD["grad"] >= GRAD_MIN and rD["band"] >= BAND_MIN)

        # -- a cell with no floor pair must say so rather than inventing one.
        rootC = os.path.join(base, "nofloor")
        for arm in ("tralo_null", "tralo"):
            for s in (1, 2):
                write_run(rootC, "MobileNetV3", "L90_G95", arm, s,
                          live_pr, y, groups, pred)
        rC = [r for r in screen([rootC]) if r["cls"] == 2][0]
        check("no reseed twin -> NO DATA, not a fabricated floor (%s)"
              % rC["verdict"], rC["verdict"] == "NO DATA"
              and "RNG floor" in rC["why"])

        # -- the bars are DERIVED, not invented; if task_window moves, so does
        #    this, and the test says so.
        check("GRAD_MIN is WIGGLE_MAX pushed through p(1-p) (%.5f)" % GRAD_MIN,
              abs(GRAD_MIN - WIGGLE_MAX * (1 - WIGGLE_MAX)) < 1e-12)
        check("BAND_MIN is task_window's MIN_PRIZE (%.1f)" % BAND_MIN,
              BAND_MIN == MIN_PRIZE)

        # -- THE SUMMARY MUST GIVE OPPOSITE ADVICE FOR OPPOSITE REASONS.
        #    It said "change the pair" for every no-sensitive tally until
        #    2026-09-04, which is this tool's own four-way distinction
        #    collapsed in its own last line: on `vitdual2` -- verified task
        #    cells, live gradient, full dose -- that advice would have thrown
        #    away the best cells in the project for want of seeds.
        def advice(verdicts):
            rows = [dict(cell=("c", "b", "iwildcam", "L80_G95"), cls=2,
                         verdict=v, why="w", n_seeds=4) for v in verdicts]
            buf = _StringIO()
            report(rows, out=buf)
            return buf.getvalue()

        sat_txt = advice(["SATURATED", "SATURATED"])
        und_txt = advice(["UNDER-POWERED", "UNDER-POWERED"])
        mix_txt = advice(["SATURATED", "UNDER-POWERED"])
        check("all SATURATED -> CHANGE THE PAIR",
              "CHANGE THE PAIR" in sat_txt and "Buy seeds" not in sat_txt)
        check("all UNDER-POWERED -> buy seeds, and NOT change the pair",
              "Buy seeds" in und_txt and "CHANGE THE PAIR" not in und_txt)
        check("a MIXED tally says the cells disagree on why, rather than "
              "picking one action",
              "do not agree on WHY" in mix_txt
              and "CHANGE THE PAIR" not in mix_txt)
        check("and a SENSITIVE cell suppresses the whole block",
              "NOT ONE CELL IS SENSITIVE"
              not in advice(["SENSITIVE", "UNDER-POWERED"]))
    finally:
        shutil.rmtree(base, ignore_errors=True)

    out.write("\n%d check(s) failed\n" % len(fails) if fails
              else "\nALL PASS\n")
    return 1 if fails else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", nargs="+", default=[],
                    help="one or more campaign roots")
    ap.add_argument("--classes", nargs="+", type=int, default=None,
                    help="override the capped classes read from the configs")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="screen a campaign `scripts.quarantine` marked dead")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.campaign:
        ap.error("--campaign is required (or --self-test)")
    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE,
    # so a marker on a dead campaign prevented nothing here. No fallback
    # import -- if the gate cannot load, the tool must break.
    from scripts import quarantine
    from scripts.quarantine import gate
    blocked, dead = gate(a.campaign, a.allow_quarantined, "screen")
    if blocked:
        return 1
    rows = screen(a.campaign, classes=a.classes, dead=dead)
    if not rows:
        print("no completed runs under %s" % ", ".join(a.campaign))
        return 1
    tally = report(rows)
    # Exit non-zero when NOTHING is sensitive, so this drops into a gate.
    return 0 if tally.get("SENSITIVE", 0) else 2


if __name__ == "__main__":
    sys.exit(main())
