"""STAGE 2 -- THE CAP ARITHMETIC, before a single config is written.

Five questions answerable with labels, a cap policy and a calculator, each
answered WRONG here at least once at a cost of a campaign: does the cap pose a
QUESTION at all (2(z16), 2(z17)); which SEEDS does it pose it to (2(z24),
2(z24b)); which SCOPE binds (FRAMEWORK 1); is there anything to WIN once it
binds (2(v)); and where does the penalty PUSH versus where does the metric
READ (2(y), cut_gap.py).

The gates drive the repo's OWN predicates -- `task_window.verdict`,
`task_cells.classify`, `full_panel.effective_budget`,
`verify_caps.duplicate_budget_tags` -- because a gate that reimplements the
thing it checks only ever agrees with itself.
"""
import os

import pytest

from configs.task_cells import (cap_pair, classify, in_window, load_windows,
                                tolerance)
from scripts.ceiling_screen import IWILDCAM_CURVE
from scripts.full_panel import effective_budget
from scripts.task_window import MIN_FORCED, WIGGLE_MAX, verdict
from scripts.verify_caps import duplicate_budget_tags

from .conftest import CAPPED_CLASSES, items_from_f1, report

pytestmark = pytest.mark.stage2_budget

# The three tags every campaign before 2026-09-01 ran. FRAMEWORK 2(z17).
NON_TASK_TAGS = ("L20_G50", "L30_G50", "L50_G30")
BACKBONES = ("ViTB16", "MobileNetV3", "MobileNetV2", "RegNetY400MF")
ALL_TAGS = NON_TASK_TAGS + ("L30_G30", "L50_G50", "L70-90_G95",
                            "L80-100_G95", "L90_G95")


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


@pytest.fixture(scope="session")
def scopes(slice_dir, protocol):
    """{tag: {cls: (K_global, K_local_sum, K_eff, n_true, n_zero_ceilings)}},
    from the pipeline's OWN constraint code. A cap tag is a percentage; what
    binds the model is the integer it rounds to on THIS test set."""
    import pandas as pd
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)
    dc = protocol["datasets"]["iwildcam"]
    te = pd.read_csv(os.path.join(slice_dir, "test_meta.csv"))
    y = te["label"].to_numpy()
    out = {}
    for tag in ALL_TAGS:
        lp, gp = cap_pair(tag)
        G = compute_global_constraints(te, "label", gp, CAPPED_CLASSES,
                                       dc["num_classes"])
        L = compute_local_constraints(te, "label", lp, dc["group_column"],
                                      CAPPED_CLASSES, dc["num_classes"])
        out[tag] = {c: (int(G[c]), sum(int(b[c]) for b in L.values()),
                        effective_budget(G, L, c), int((y == c).sum()),
                        sum(1 for b in L.values() if int(b[c]) == 0))
                    for c in CAPPED_CLASSES}
    return out


def test_a_cap_is_a_task_only_when_all_three_conditions_hold():
    """BINDS and PRIZE and WIGGLE -- drop one and the cell measures nothing.
    FRAMEWORK 2(z16): 24 of 24 cells at L20/L30/L50 fail at least one, the best
    single explanation on record for why so many arms tied. The thresholds live
    in `task_window` and are quoted in `task_windows.yml`; if those drift the
    yml documents a gate nobody runs. NEGATIVE CONTROL: the all-conditions row
    must read TASK, and BINDS must stay a COUNT -- `hard > K` is passed by a
    cap evicting one item."""
    crit = load_windows()["meta"]["criteria"]
    cases = [
        ("LIVENESS all three hold", 15.0, 0.718, [40] * 4, "** TASK **"),
        ("BINDS fails: L90 evicts 3", 27.8, 0.586, [3] * 4, "barely binds"),
        ("BINDS fails: K above the count", 46.0, 0.287, [-5] * 4, "cap slack"),
        ("PRIZE fails: top-K perfect", 0.0, 0.500, [40] * 4, "no prize"),
        ("WIGGLE fails: p@K = 0.9999", 15.0, 0.9999, [40] * 4, "saturated"),
        ("no reference runs at all", 15.0, 0.718, [], "no data"),
    ]
    fails = ["%s: verdict=%r, expected %r" % (lbl, verdict(e, p, f), w)
             for lbl, e, p, f, w in cases if verdict(e, p, f) != w]
    if MIN_FORCED != crit["min_forced"]:
        fails.append("MIN_FORCED=%s but the yml gate says %s"
                     % (MIN_FORCED, crit["min_forced"]))
    if WIGGLE_MAX != crit["wiggle_max"]:
        fails.append("WIGGLE_MAX=%s but the yml gate says %s"
                     % (WIGGLE_MAX, crit["wiggle_max"]))
    if verdict(15.0, 0.5, [MIN_FORCED - 1] * 4) == "** TASK **":
        fails.append("a cap evicting %d items reads TASK; BINDS is a boolean "
                     "again" % (MIN_FORCED - 1))
    if verdict(15.0, 0.5, [MIN_FORCED] * 4) != "** TASK **":
        fails.append("exactly MIN_FORCED=%d is refused; threshold off by one"
                     % MIN_FORCED)
    report(fails, "task-condition failures")


def test_binds_is_a_per_seed_count_and_never_a_mean():
    """FRAMEWORK 2(z24): `task_window` applied MIN_FORCED to the MEAN.
    iwildcam/MobileNetV3 class 2, four lambda=0 seeds: 278/329/354/383, spread
    105 items. At `L90_G95` (K=333) the mean says `forced = 3` and the cap
    reads "barely binds" while it evicts 50 in one seed and is slack in two.
    No seed resembles the mean. NEGATIVE CONTROL both ways: the same numbers as
    their own mean must read DIFFERENTLY, or this passes by refusing all."""
    K = 333
    per_seed = [c - K for c in (278, 329, 354, 383)]     # -55, -4, 21, 50
    mean_only = [sum(per_seed) / float(len(per_seed))]   # +3.0
    cases = [
        ("2(z24) L90_G95 c2, per seed", per_seed, "** PARTIAL 2/4 **"),
        ("2(z24) L90_G95 c2, MEAN (the defect)", mean_only, "barely binds"),
        ("LIVENESS mostly-binding, per seed", [5, 60, 60, 60],
         "** PARTIAL 3/4 **"),
        ("LIVENESS the same numbers as their mean", [33.75], "** TASK **"),
    ]
    fails = ["%s: verdict=%r, expected %r" % (lbl, verdict(27.8, 0.586, f), w)
             for lbl, f, w in cases if verdict(27.8, 0.586, f) != w]
    if verdict(27.8, 0.586, per_seed) == verdict(27.8, 0.586, mean_only):
        fails.append("per-seed and mean-collapsed `forced` give the SAME "
                     "verdict -- the 2(z24) fix is inert")
    report(fails, "per-seed BINDS failures")


def test_classify_keeps_every_status_distinct_and_the_24_of_24_census(
        protocol, windows, scopes):
    """`classify` must never collapse its statuses. FRAMEWORK 2(z24b).
    Six outcomes meaning opposite things: `non_task` is about the experiment;
    `partial` = binds in only some seeds, so the effective n is below the seed
    count; `unmeasured` = the K/n fell in the GAP between the strict and
    partial bands, off the 0.1 grid; `no_window` = unmeasured backbone;
    `no_data` = missing instrument. Reporting `unmeasured` as `non_task` claims
    a measurement nobody took. NEGATIVE CONTROLS: an unmeasured backbone and an
    absent slice must not read `non_task`, and a loose per-class tag must read
    `task` somewhere -- a classifier that only refuses decides nothing."""
    # 🛑 A SEVENTH STATUS EXISTS AS OF 2026-09-02: `no_strict_band`,
    # for a class whose strict window is measured and EMPTY. And the (model,
    # tag) that produces each status MOVED when the windows were rebuilt with
    # the per-group prize -- MobileNetV3 `L70-90_G95` was the `task` example
    # and is now the `no_strict_band` one. What this gate protects is that the
    # statuses stay DISTINGUISHABLE, so the pairs below are chosen to hit each
    # one and the assertion is on distinctness, not on any particular label.
    fails, seen = [], {}
    for model, tag, want in (("MobileNetV2", "L80_G95", "task"),
                             ("MobileNetV3", "L90_G95", "partial"),
                             ("MobileNetV2", "L85_G95", "unmeasured"),
                             ("MobileNetV2", "L30_G50", "non_task"),
                             ("MobileNetV3", "L70-90_G95", "no_strict_band")):
        got = classify(protocol, windows, "iwildcam", model, tag)
        seen[want] = got["status"]
        if got["status"] != want:
            fails.append("%s %s: status=%r, expected %r"
                         % (model, tag, got["status"], want))
        if want in ("task", "partial") and not got.get("provenance"):
            fails.append("%s carries no provenance; a window row is measured "
                         "from ONE campaign's model and does not transfer" % tag)
    no_win = classify(protocol, windows, "iwildcam", "NoSuchNet", "L30_G50")
    no_data = classify({"datasets": {"iwildcam": {"data_dir": "no/such/dir"}}},
                       windows, "iwildcam", "MobileNetV3", "L70-90_G95")
    seen["no_window"], seen["no_data"] = no_win["status"], no_data["status"]
    if no_win["status"] != "no_window":
        fails.append("an unmeasured backbone reads %r" % no_win["status"])
    if no_data["status"] != "no_data":
        fails.append("an absent slice reads %r" % no_data["status"])
    if len(set(seen.values())) != len(seen):
        fails.append("statuses collapsed: %r" % (seen,))

    cells = 0        # 4 backbones x 2 classes x 3 tags, every one outside
    for model in BACKBONES:
        for tag in NON_TASK_TAGS:
            r = classify(protocol, windows, "iwildcam", model, tag)
            # THE CENSUS ASSERTS THE FINDING, NOT ITS LABEL. 2(z17) measured
            # that L20/L30/L50 pose no question on any backbone; after the
            # 2026-09-02 rebuild some of those cells say so as `no_strict_band`
            # (the class has no strict window at all) rather than `non_task`
            # (this K/n is outside one). Both mean "do not run a campaign
            # here"; pinning the string would have failed the gate for a
            # relabel while the measurement was unchanged.
            if r["status"] in ("task", "partial"):
                fails.append("%s %s reads %r; 2(z17) measured that it poses "
                             "NO question" % (model, tag, r["status"]))
            for c, v in r["classes"].items():
                cells += 1
                if v["band"] not in ("outside", "no_strict"):
                    fails.append("%s %s c%d band=%r at K/n=%.4f"
                                 % (model, tag, c, v["band"], v["ratio"]))
    if cells != 24:
        fails.append("the census covered %d cells, not the 24 of 2(z17)" % cells)
    if not any(classify(protocol, windows, "iwildcam", m,
                        "L70-90_G95")["status"] == "task" for m in BACKBONES):
        fails.append("LIVENESS: no backbone reads `task` at L70-90_G95")
    report(fails, "classify status failures")


def test_effective_K_is_min_of_global_and_the_local_sum(scopes):
    """FRAMEWORK 1 and `full_panel.effective_budget`. Local caps are per-GROUP
    ceilings, so their sum already bounds the count: the global is REDUNDANT
    where it equals that sum and INERT above it, and reading `int(G[c])` alone
    inflated the iwildcam prize 30x (L30_G50 class 2: global 185 against a
    reachable 111). Also gated: a cap TAG does not produce a round K/n, which
    is why the window file needs a grid-snapping tolerance at all. NEGATIVE
    CONTROLS: the global-alone bug must stay observable, and one UNCAPPED group
    must send the budget back to the global -- no ceiling bounds nothing."""
    fails = []
    for tag, per in sorted(scopes.items()):
        for c, (kg, kl, keff, n, _) in sorted(per.items()):
            if keff != min(kg, kl):
                fails.append("%s c%d: eff=%d but min(global %d, local sum %d)"
                             "=%d" % (tag, c, keff, kg, kl, min(kg, kl)))
            if keff > n * 1.2:
                fails.append("%s c%d: K=%d exceeds 1.2*n=%d, off the grid"
                             % (tag, c, keff, n))
    for tag, want in (("L30_G30", "redundant"), ("L30_G50", "inert"),
                      ("L50_G30", "binds")):
        kg, kl = scopes[tag][2][0], scopes[tag][2][1]
        got = "redundant" if kg == kl else "inert" if kg > kl else "binds"
        if got != want:
            fails.append("%s c2 global scope reads %r (global %d vs local sum "
                         "%d), expected %r" % (tag, got, kg, kl, want))
    kg, keff = scopes["L30_G50"][2][0], scopes["L30_G50"][2][2]
    if kg == keff:
        fails.append("L30_G50 c2 global %d equals the effective %d, so the 30x "
                     "inert-global bug is not observable" % (kg, keff))
    from src.utils.constants import UNLIMITED
    G = [UNLIMITED] * 8
    G[2] = 100
    L = {0: [UNLIMITED] * 8, 1: [UNLIMITED] * 8}
    L[0][2], L[1][2] = 10, 10
    if effective_budget(G, L, 2) != 20:
        fails.append("all groups capped: expected the local sum 20, got %d"
                     % effective_budget(G, L, 2))
    L[1][2] = UNLIMITED
    if effective_budget(G, L, 2) != 100:
        fails.append("one UNCAPPED group still bounds the total; expected the "
                     "global 100, got %d" % effective_budget(G, L, 2))
    ratios = [p[c][2] / float(p[c][3]) for p in scopes.values() for c in p]
    if all(abs(r * 100 - round(r * 100)) < 1e-9 for r in ratios):
        fails.append("every cap tag produced a round K/n; the integer rounding "
                     "this gate exists for has stopped happening")
    report(fails, "effective-budget failures")


def test_two_cap_tags_are_not_two_cap_levels_unless_the_budget_differs(scopes):
    """House rule 4, via `verify_caps.duplicate_budget_tags`. `gen_campaign`
    refuses a single-cap campaign by comparing TAG STRINGS, which any two
    spellings satisfy. On iwildcam `L30_G30`, `L30_G50` and `L50_G30` all land
    on K=111/137 -- one budget level wearing three tags, and a single-cap claim
    has been retracted three times. NEGATIVE CONTROL: genuinely distinct
    budgets must NOT be reported, or the detector refuses every campaign."""
    fails = []
    same = {c: {t: scopes[t][c][2] for t in ("L30_G30", "L30_G50", "L50_G30")}
            for c in CAPPED_CLASSES}
    dups = duplicate_budget_tags(same)
    for c in CAPPED_CLASSES:
        if not [d for d in dups if d[0] == c and len(d[2]) == 3]:
            fails.append("class %d: L30_G30/L30_G50/L50_G30 were not reported "
                         "as one budget level (budgets %r)" % (c, same[c]))
    distinct = {c: {t: scopes[t][c][2]
                    for t in ("L20_G50", "L30_G50", "L50_G50")}
                for c in CAPPED_CLASSES}
    if duplicate_budget_tags(distinct):
        fails.append("three DISTINCT budget levels reported as duplicates: %r"
                     % (duplicate_budget_tags(distinct),))
    # 7 of 14 local ceilings are K=0 on iwildcam: a zero ceiling binds however
    # slack the class TOTAL is, which is why a slack seed is NOT a free null
    # (FRAMEWORK 2(z24) point 2).
    if sum(scopes["L90_G95"][c][4] for c in CAPPED_CLASSES) == 0:
        fails.append("no local ceiling is K=0; the structural reason a PARTIAL "
                     "seed still differs from its null has disappeared")
    report(fails, "duplicate-budget failures")


def test_the_ceiling_and_the_prize_are_arithmetic_bounds():
    """`ceiling = 2K/(K+n)`, `prize_items = (1-p)*K`. FRAMEWORK 2(v). Emitting
    only K predictions for a class with n true instances caps cc-F1 at
    precision 1, recall K/n; no loss, dual, allocator or optimizer changes that
    bound, and `ceiling_screen` reproduces `headroom`'s numbers from labels
    alone. NEGATIVE CONTROLS: the inert-global K (185) inflates the class 2
    ceiling from the reachable 0.4615 to 0.6667 -- the 30x bug's signature --
    and the screen must still be able to say WORTH RUNNING."""
    fails = []
    table = [("L20_G50", 2, 370, 74, 0.3333), ("L20_G50", 7, 456, 92, 0.3358),
             ("L30_G50", 2, 370, 111, 0.4615),
             ("L30_G50", 7, 456, 137, 0.4621),
             ("L50_G30", 2, 370, 111, 0.4615),
             ("L50_G30", 7, 456, 137, 0.4621)]
    for tag, c, n, K, want in table:
        ceil = 2.0 * K / (K + n)
        if abs(ceil - want) > 5e-5:
            fails.append("%s c%d: 2K/(K+n)=%.4f, 2(v) publishes %.4f"
                         % (tag, c, ceil, want))
        if not 0.0 < ceil <= 1.0:
            fails.append("%s c%d: ceiling %.4f is not a probability"
                         % (tag, c, ceil))
    inert, reachable = 2.0 * 185 / 555, 2.0 * 111 / 481
    if inert <= reachable + 0.2:
        fails.append("the inert-global ceiling %.4f no longer separates from "
                     "the reachable %.4f" % (inert, reachable))
    for ratio, p, sd in IWILDCAM_CURVE:
        prize = (1.0 - p) * ratio * 370
        if ratio <= 0.80 and prize >= sd:
            fails.append("K/n=%.2f: prize %.2f items is no longer below the "
                         "paired sd %.2f -- 2(v) says it is"
                         % (ratio, prize, sd))
    if (1.0 - 0.95) * 300 <= 2.11:
        fails.append("LIVENESS: the arithmetic cannot say WORTH RUNNING even "
                     "at p=0.95, K=300")
    report(fails, "ceiling/prize failures")


def test_a_ccf1_delta_is_quantised_and_converts_to_items():
    """`items = dF1 * (K+n)/2`, and it must land on an INTEGER. With exactly K
    predictions emitted `F1 = 2TP/(K+n)`, so the only cc-F1 deltas that can
    occur are integer multiples of `2/(K+n)`; anything else is an arithmetic
    bug -- a budget mismatch, or arms emitting different counts. The whole gap
    from `clip` to a PERFECT allocator is 1.9-9.9 items, so 0.02 is not a small
    effect. NEGATIVE CONTROL: 0.02 is not legal at K=111, n=370."""
    fails = []
    for K, n in ((74, 370), (111, 370), (137, 456), (259, 370), (411, 456)):
        quantum = 2.0 / (K + n)
        for tp in (1, 2, 5, 17):
            got = items_from_f1(tp * quantum, K, n)
            if abs(got - tp) > 1e-9:
                fails.append("K=%d n=%d: %d TP round-tripped to %.6f items"
                             % (K, n, tp, got))
        # CLAUDE.md states the quantum as 1/(K+n); the sharp value is 2/(K+n),
        # so one 1/(K+n) is HALF an item and cannot occur.
        half = items_from_f1(1.0 / (K + n), K, n)
        if abs(half - round(half)) < 1e-9:
            fails.append("K=%d n=%d: 1/(K+n) reads as %.4f whole items; the "
                         "quantum is not 2/(K+n)" % (K, n, half))
    illegal = items_from_f1(0.02, 111, 370)
    if abs(illegal - round(illegal)) < 1e-6:
        fails.append("dF1=0.02 at K=111,n=370 reads as the whole %.4f items, "
                     "so quantisation can reject nothing" % illegal)
    # A plausibility floor on the CONVERSION, not a claim about the effect
    # space: 0.02 * (111+370)/2 = 4.81 items, so anything under ~2 means the
    # scale factor collapsed. The number 1.9 originally came from a dermmnist
    # headroom band (removed, 38.7%-leaking dataset -- FRAMEWORK 2(z32)d); it
    # survives here only as an arithmetic sanity bar, and the message must not
    # re-assert it as the effect space.
    if illegal < 1.9:
        fails.append("dF1=0.02 at K=111,n=370 converts to %.2f items; the "
                     "arithmetic gives 4.81, so a value this small means the "
                     "(K+n)/2 scale factor is wrong" % illegal)
    report(fails, "quantisation failures")


def test_the_tolerance_is_grid_snapping_not_window_widening():
    """0.005 is 1/20th of the 0.1 measurement grid step. FRAMEWORK 2(z24b). It
    exists so `L90_G95`'s class 7 K/n of 0.9013 reads against the MEASURED grid
    point 0.90 -- reading the measurement, not extrapolating past it -- and it
    is load-bearing (46% of task-cell runs qualify only through it). NEGATIVE
    CONTROL: a tolerance ten times larger WOULD rescue `L80-100_G95`'s 0.950,
    the fraction nobody measured, and breaks the grid relation -- so raising it
    fails here instead of silently widening every window in the file."""
    TW = load_windows()
    tol = tolerance(TW)
    grid = TW["meta"]["fraction_grid"]
    step = min(round(b - a, 6) for a, b in zip(grid, grid[1:]))
    fails = []
    if tol > step / 10.0:
        fails.append("tolerance %.4f exceeds a tenth of the %.2f grid step"
                     % (tol, step))
    if tol >= step / 2.0:
        fails.append("tolerance %.4f reaches past the midpoint of the %.2f "
                     "grid, so it widens rather than snaps" % (tol, step))
    lo, hi = TW["windows"]["iwildcam"]["MobileNetV3"]["class"][7]
    if not in_window(0.9013, lo, hi, tol):
        fails.append("L90_G95 c7 K/n=0.9013 does not snap onto %.2f-%.2f"
                     % (lo, hi))
    if in_window(0.9496, lo, hi, tol):
        fails.append("L80-100_G95 c7 K/n=0.9496 snapped onto %.2f-%.2f; that "
                     "fraction was never measured" % (lo, hi))
    if not in_window(0.9496, lo, hi, 0.05):
        fails.append("LIVENESS: a 0.05 tolerance does NOT rescue 0.9496, so "
                     "the negative control proves nothing")
    if 0.05 <= step / 10.0:
        fails.append("LIVENESS: the 0.05 counter-example passes the grid "
                     "relation, so the relation gates nothing")
    if not (in_window(lo, lo, hi, tol) and in_window(hi, lo, hi, tol)):
        fails.append("a window EDGE reads as outside its own window")
    report(fails, "tolerance failures")


def test_rank_K_is_not_the_decision_boundary():
    """The penalty pushes at the BOUNDARY; the metric reads at the CUT, and
    `gap = hard_count - K` is the distance between them (cut_gap.py, FRAMEWORK
    2(y)). The per-item gradient scales with `p(1-p)`, maximal at p=0.5; at a
    tight cap the CUT sits at p=0.9999 where `p(1-p)`=0.0001, so quoting the
    boundary's weight as reachable at the cut overstates it by orders of
    magnitude. NEGATIVE CONTROLS: at a loose cap the two nearly coincide, so
    the gate is not vacuous; and with the hard count FIXED `gap` is an exact
    affine function of K (rho = -1.0000), i.e. K in another costume and never a
    cause -- letting it vary is the only way to pry them apart."""
    # cut_gap.py header table: tag, K/n, min gap in items, p at the cut
    measured = [("L20_G50", 0.20, 235, 0.9990), ("L30/L50", 0.30, 198, 0.9970),
                ("L80_G95", 0.80, 40, 0.9990), ("L90_G95", 0.90, 3, 0.5870)]
    fails = []
    for tag, ratio, gap, p_cut in measured:
        if ratio > 0.30:
            continue
        at_cut = p_cut * (1.0 - p_cut)
        if gap < 100:
            fails.append("%s: gap %d items -- the cut and the boundary have "
                         "stopped being far apart at a tight cap" % (tag, gap))
        if 0.25 / at_cut < 10.0:
            fails.append("%s: p(1-p) at the cut is %.5f, only %.1fx below the "
                         "boundary maximum" % (tag, at_cut, 0.25 / at_cut))
    loose = 0.587 * (1.0 - 0.587)
    if 0.25 / loose > 1.5:
        fails.append("even at K/n=0.90 the cut is %.1fx below the boundary; "
                     "the regime contrast has vanished" % (0.25 / loose))
    weights = [p * (1.0 - p) for _, p, _ in IWILDCAM_CURVE]
    if any(b <= a for a, b in zip(weights, weights[1:])):
        fails.append("p(1-p) at the cut is not monotone in K/n: %r"
                     % [round(w, 5) for w in weights])
    if all(b <= a for a, b in zip(weights, weights[1:])):
        fails.append("LIVENESS: the reversed order also passes, so "
                     "monotonicity gates nothing")
    Ks = [round(f * 370) for f in (0.2, 0.3, 0.5, 0.7, 0.9)]
    fixed = [336 - K for K in Ks]
    if abs(_pearson(Ks, fixed) + 1.0) > 1e-9:
        fails.append("with the hard count fixed, rho(gap, K) = %.4f, not "
                     "-1.0000" % _pearson(Ks, fixed))
    varying = [h - K for h, K in zip((278, 329, 354, 383, 500), Ks)]
    if abs(_pearson(Ks, varying) + 1.0) < 1e-3:
        fails.append("LIVENESS: a VARYING hard count still gives rho=-1, so "
                     "the degeneracy check detects nothing")
    report(fails, "cut-vs-boundary failures")
