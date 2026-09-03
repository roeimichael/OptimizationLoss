"""STAGE 6 -- RESULT INTEGRITY: what may be read off the output, before a claim.

Nine measured incidents. Each gate below carries its own receipt; this is the
index.

  1  inert flags -- `cb_lp` == `clip` in 24/24 with its OWN base_model_id and
     every config audit green; only the raw md5 finds it.     FRAMEWORK 2(x1)
  2  the panel is ALLOCATOR-BLIND -- `lp` vs `clip` reads +0.0000 p=1.000
     while the DEPLOYED predictions differ in 23/24.          FRAMEWORK 2(x1)
  3  rule 4 -- the cell is (dataset, backbone, cap, method), seed the only
     collapsed axis; count cells, never runs.                 FRAMEWORK 3(0)
  4  a cell is NOT an independent unit -- `dom1` and `loose1` are ONE model in
     8/8 pairs, so 4/4 p=0.0625 is 3/3 p=0.125.              FRAMEWORK 2(z24)
  5  `flips`, raw count over K and proximity to feasibility are not metrics;
     post-hoc filling is free.                     PLAYBOOK 1.6, CLAUDE.md 5
  6  items and quantisation -- per class F1 = 2TP/(K+n) with TP an integer,
     and clip -> a PERFECT allocator is the whole 1.9-9.9 items.        2(v)
  7  a tie is "no effect" OR "too few seeds" -- `dualbar2` reads +0.36 items
     at ~174 seeds per cell.                    full_panel's RESOLUTION block
  8  four noise numbers differing up to 12x, and pairing GROWS this one. 2(v)
  9  `d capF1` goes beside `d macroF1` AND `uncF1`.       PLAYBOOK 1.3, 3(0c)
"""
import hashlib
import io
import math
import os

import pytest

from gates.conftest import items_from_f1, report

pytestmark = pytest.mark.stage6_results

Z2 = (1.959963985 + 0.8416212336) ** 2          # alpha .05 two-sided, power .80
HEADROOM_ITEMS = 9.9                            # clip -> PERFECT allocator, 2(v)
CELL_KEY = ("dataset", "model", "cap", "arm")   # seed is NOT in it


# ------------------------------------------------------------- pure helpers
def write_pred(path, rows):
    """A minimal predictions file; the digest is over the FILE, so only the
    shape has to be stable across arms."""
    io.open(path, "w", encoding="utf-8", newline="").write(
        "True_Label,Predicted_Label\n" + "".join("%d,%d\n" % r for r in rows))
    return path

def md5_file(path):
    return hashlib.md5(io.open(path, "rb").read()).hexdigest()[:12]

def inert_pairs(md5_by_arm):
    """Arm pairs whose digests agree on EVERY shared cell. Configs get no vote
    on purpose: `cb_lp` carries its own `base_model_id` and is still inert."""
    arms, out = sorted(md5_by_arm), []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            shared = sorted(set(md5_by_arm[a]) & set(md5_by_arm[b]))
            same = sum(md5_by_arm[a][c] == md5_by_arm[b][c] for c in shared)
            if shared and same == len(shared):
                out.append((a, b, len(shared)))
    return out

def f1_of(y, pred, c):
    tp = sum(1 for a, b in zip(y, pred) if a == c and b == c)
    kn = sum(1 for b in pred if b == c) + sum(1 for a in y if a == c)
    return 0.0 if not kn else 2.0 * tp / kn

def equalized(probs, K, c):
    """What the panel does: re-derive top-K from the RAW probabilities,
    discarding whatever the arm actually deployed."""
    sel = set(sorted(range(len(probs)), key=lambda i: -probs[i])[:K])
    return [c if i in sel else -1 for i in range(len(probs))]

def allocator_claim_ok(source):
    return source == "final_predictions.csv"     # as-deployed, never the panel

def pooled_axes(rows, collapsed):
    return sorted(a for a in ("dataset", "model", "cap")
                  if a in collapsed and len({r[a] for r in rows}) > 1)

def n_cells(rows):
    return len({tuple(r[k] for k in CELL_KEY) for r in rows})

def independent_units(model_md5_by_cell):
    """Byte-identical warm-ups are ONE unit however many cells they appear in
    -- across cap tags AND across campaigns."""
    return len(set(model_md5_by_cell.values()))

def sign_p(n_units):
    return 0.5 ** n_units                        # one-sided, all units agreeing

def seeds_needed(effect, sd):
    if not (effect > 0 and sd > 0):
        return float("inf")
    return int(math.ceil(Z2 * sd * sd / (effect * effect)))

def detectable_at(sd, n):
    return math.sqrt(Z2) * sd / math.sqrt(n)

def items_are_quantised(d_f1, K, n, tol=1e-6):
    it = items_from_f1(d_f1, K, n)               # TP is an integer, so this is
    return abs(it - round(it)) <= tol            # an integer or it is a bug

def tie_report(effect, sd, have):
    """(verdict, line) -- a tie without its power is not reportable."""
    if sd is None:
        return "NOT REPORTABLE", "no seed sd -- a tie here says nothing"
    if effect == 0.0:
        return ("BOUNDED NULL", "0.00 items; anything above %.2f items would "
                "have been seen at %d seeds" % (detectable_at(sd, have), have))
    need = seeds_needed(abs(effect), sd)
    return ("POWERED" if have >= need else "UNDERPOWERED",
            "%+.2f items, sd %.2f, %d seeds, needs ~%d per cell"
            % (effect, sd, have, need))

NOISE_KINDS = ("unpaired", "reseed", "treated", "panel_macro_ccf1")

def price(kind, contrast_is_paired, unit="tp_items"):
    """None if this noise may price this prize, else why not."""
    if kind not in NOISE_KINDS:
        return "unknown noise %r -- name which of the four" % kind
    if kind == "panel_macro_ccf1" and unit == "tp_items":
        return "panel sd is macro-averaged d ccF1, not per-class TP items"
    if kind == "unpaired" and contrast_is_paired:
        return "unpaired sd priced against a PAIRED contrast"
    return None

DOC_ALIAS = {"sat": "raw_feasible", "native satisfaction": "raw_feasible",
             "cnt/K": "cnt_over_K", "raw count over K": "raw_over_K",
             "proximity to feasibility": "raw_over_K"}

def scorable(name):
    from scripts.full_panel import NON_SCORING
    return DOC_ALIAS.get(name, name) not in NON_SCORING

REQUIRED_BESIDE = {"ccF1": ("macroF1", "uncF1"), "macroF1": ("uncF1",)}

def table_ok(metrics):
    missing = []
    for m, need in REQUIRED_BESIDE.items():
        if m in metrics:
            missing += [x for x in need if x not in metrics]
    return sorted(set(missing))


# -------------------------------------------------------------------- gates
def test_inert_flags_are_caught_by_hashing_raw_predictions(tmp_path):
    """GATE 1 -- FRAMEWORK 2(x1). Hash `final_predictions_raw.csv` across arms
    BEFORE any metric, and let no config field rescue an identical pair."""
    same = [(2, 2), (2, 7), (7, 7), (0, 2)]
    diff = [(2, 7), (2, 7), (7, 7), (0, 2)]
    cases = [
        ("cb_lp vs clip, identical on every cell (2(x1): 24/24)",
         {"clip": [same] * 4, "cb_lp": [same] * 4}, True),
        ("la_lp vs clip, 0/4 identical -- LIVE (negative control)",
         {"clip": [same] * 4, "la_lp": [diff] * 4}, False),
        ("tralo vs tralo_null, distinct in 4/4 slack seeds (2(z24).2)",
         {"tralo_null": [same] * 4, "tralo": [diff] * 4}, False),
        ("partial 2 of 4 -- NOT a dead flag (negative control)",
         {"clip": [same] * 4, "focal_clip": [same, same, diff, diff]}, False),
    ]
    fails = []
    for j, (label, payload, expect_inert) in enumerate(cases):
        md5s = {}
        for arm, per_cell in payload.items():
            md5s[arm] = {("iwildcam", "MobileNetV3", "L80_G95", "2-7", i):
                         md5_file(write_pred(os.path.join(
                             str(tmp_path), "%d_%s_%d.csv" % (j, arm, i)), rows))
                         for i, rows in enumerate(per_cell)}
        if bool(inert_pairs(md5s)) != expect_inert:
            fails.append("%s: gate said inert=%s" % (label, not expect_inert))
    cfgs = {"clip": "067715022594", "cb_lp": "7e92e1b76bc5"}   # 2(x1)'s real ids
    if len(set(cfgs.values())) != 2:
        fails.append("negative control broken: base_model_ids not distinct")
    if not inert_pairs({a: {("c", 0): "deadbeefcafe"} for a in cfgs}):
        fails.append("a distinct base_model_id rescued a byte-identical pair")
    report(fails, "inert-flag gate failures")


def test_allocators_are_compared_on_deployed_not_on_the_panel():
    """GATE 2 -- FRAMEWORK 2(x1). The panel re-derives its own top-K from the
    RAW probabilities, so two arms sharing a warm-up score exactly +0.0000
    however differently they allocate."""
    y = [2, 2, 2, 7, 7, 0, 0, 0]
    raw_clip = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    raw_lp = list(raw_clip)                       # ONE warm-up, two allocators
    raw_la = [0.2, 0.3, 0.9, 0.4, 0.8, 0.5, 0.7, 0.6]      # a different model
    dep_clip = [2, 2, 2, -1, -1, -1, -1, -1]      # greedy
    dep_lp = [2, 2, -1, -1, -1, 2, -1, -1]        # LP-LG, SAME budget K=3
    fails = []
    if f1_of(y, equalized(raw_clip, 3, 2), 2) != f1_of(y, equalized(raw_lp, 3, 2), 2):
        fails.append("panel moved on a shared warm-up -- fixture is wrong")
    if f1_of(y, dep_clip, 2) == f1_of(y, dep_lp, 2):
        fails.append("as-deployed contrast is blind too -- fixture is wrong")
    if sum(x == 2 for x in dep_clip) != sum(x == 2 for x in dep_lp):
        fails.append("budgets differ, so the deployed delta is free fill")
    # NEGATIVE CONTROL: on genuinely different models the panel DOES move, so it
    # is the right instrument for a MODEL claim and the wrong one for allocators.
    if f1_of(y, equalized(raw_clip, 3, 2), 2) == f1_of(y, equalized(raw_la, 3, 2), 2):
        fails.append("panel blind to a real model difference -- gate vacuous")
    for src, ok in [("final_predictions_raw.csv", False),
                    ("full_panel budget-equalized", False),
                    ("final_predictions.csv", True)]:
        if allocator_claim_ok(src) != ok:
            fails.append("allocator claim from %r: expected ok=%s" % (src, ok))
    report(fails, "allocator-blindness gate failures")


def test_only_seed_may_be_collapsed_and_cells_are_counted_not_runs():
    """GATE 3 -- house rule 4 / FRAMEWORK 3(0). Pooling across cap, backbone or
    dataset has retracted a claim three times, and a sign test over runs is a
    sign test over seeds."""
    grid = [dict(dataset=d, model=m, cap=c, arm="tralo", seed=s)
            for d in ("iwildcam",) for m in ("ViTB16", "MobileNetV3")
            for c in ("L80_G95", "L90_G95") for s in range(4)]
    cases = [
        ("seed only -- legal (negative control)", grid, {"seed"}, []),
        ("pooled over cap", grid, {"seed", "cap"}, ["cap"]),
        ("pooled over backbone", grid, {"seed", "model"}, ["model"]),
        ("pooled over cap AND backbone", grid, {"seed", "cap", "model"},
         ["cap", "model"]),
        ("'pooled' over an axis with ONE level -- nothing pooled "
         "(negative control)",
         [r for r in grid if r["cap"] == "L80_G95"], {"seed", "cap"}, []),
    ]
    fails = ["%s: named %s, expected %s" % (l, pooled_axes(r, c), e)
             for l, r, c, e in cases if pooled_axes(r, c) != e]
    if n_cells(grid) != 4 or len(grid) != 16:
        fails.append("cell counter: %d cells over %d runs, expected 4 over 16"
                     % (n_cells(grid), len(grid)))
    if n_cells(grid) == len(grid):
        fails.append("cell count equals run count -- the key lost an axis")
    report(fails, "pooling-axis gate failures")


def test_independent_units_collapse_byte_identical_warm_up_models():
    """GATE 4 -- FRAMEWORK 2(z24).4 / PLAYBOOK 1.2. `dom1` and `loose1` are ONE
    model in 8/8 (cap, seed) pairs and two cap tags share a warm-up, so 8 cells
    can be 4 units and 4/4 is p=0.0625, not 0.0039."""
    digests = ["7f1ff13ebc", "1df6ab42f8", "b51c30725d", "7ab05f80c4"]   # 2(z24)
    shared = {"%s|L80_G95|s%d" % (camp, s): digests[s]
              for camp in ("dom1", "loose1") for s in range(4)}
    within = {"dom1|%s|s%d" % (cap, s): digests[s]
              for cap in ("L80_G95", "L90_G95") for s in range(4)}
    distinct = {"c%d" % i: "%010d" % i for i in range(8)}
    cases = [("dom1 + loose1, byte-identical 8/8", shared, 8, 4, 0.0625),
             ("two cap tags, one warm-up", within, 8, 4, 0.0625),
             ("8 distinct models (negative control)", distinct, 8, 8, 2 ** -8),
             ("the 2(z24) recount, 4 units -> 3",
              dict(list(distinct.items())[:3]), 3, 3, 0.125)]
    fails = []
    for label, table, n_c, n_u, p in cases:
        u = independent_units(table)
        if len(table) != n_c or u != n_u or abs(sign_p(u) - p) > 1e-9:
            fails.append("%s: %d cells / %d units / p %.6f, expected %d / %d / "
                         "%.6f" % (label, len(table), u, sign_p(u), n_c, n_u, p))
    if sign_p(independent_units(shared)) <= sign_p(len(shared)):
        fails.append("collapsing units did not WEAKEN the p-value")
    report(fails, "independent-unit gate failures")


def test_flips_and_feasibility_are_never_a_headline():
    """GATE 5 -- PLAYBOOK 1.6, CLAUDE.md rule 5. Post-hoc filling is free, so
    an arm that halves the flip count and ties on quality produced nothing.
    Read against `full_panel.NON_SCORING` itself, not a copy of it."""
    banned = ("flips", "flips_over_K", "raw_over_K", "count_eq_K", "cnt_over_K",
              "raw_feasible", "sat", "native satisfaction",
              "raw count over K", "proximity to feasibility")
    # negative control -- the metrics a verdict MAY rest on
    allowed = ("ccF1", "macroF1", "uncF1", "AP", "AUROC", "ECE")
    fails = ["%s: scorable=%s, expected %s" % (n, scorable(n), ok)
             for n, ok in ([(x, False) for x in banned] +
                           [(x, True) for x in allowed]) if scorable(n) != ok]
    from scripts.full_panel import NON_SCORING
    if not {"flips", "raw_over_K"} <= set(NON_SCORING):
        fails.append("full_panel.NON_SCORING lost flips/raw_over_K")
    if any(scorable(n) for n in NON_SCORING):
        fails.append("a NON_SCORING metric reads as scorable")
    report(fails, "non-metric headline gate failures")


def test_deltas_convert_to_items_and_are_quantised_per_class():
    """GATE 6 -- CLAUDE.md rule 2 / FRAMEWORK 2(v). Per class F1 = 2TP/(K+n)
    with TP an integer, so `items` is an integer or the arithmetic is wrong,
    and the whole clip -> PERFECT gap is 1.9-9.9 items."""
    K, n = 333, 418                              # iwildcam class 2 at K/n = 0.90
    cases = [  # label, dF1, K, n, quantised?, items, within the 9.9 headroom?
        ("4 TP items (negative control)", 2.0 * 4 / (K + n), K, n, 1, 4.0, 1),
        ("1 TP item (negative control)", 2.0 * 1 / (K + n), K, n, 1, 1.0, 1),
        ("0.02 at K=74 -- 4.92 items, not an integer", 0.02, 74, 418, 0, 4.92, 1),
        ("40 items -- 4x the headroom", 2.0 * 40 / (K + n), K, n, 1, 40.0, 0),
        ("a sub-item re-allocation", 2.0 * 0.4 / (K + n), K, n, 0, 0.4, 1)]
    fails = []
    for label, d_f1, kk, nn, quant, want_items, plausible in cases:
        it = items_from_f1(d_f1, kk, nn)
        if abs(it - want_items) > 1e-6:
            fails.append("%s: %.4f items, expected %.4f"
                         % (label, it, want_items))
        if items_are_quantised(d_f1, kk, nn) != bool(quant):
            fails.append("%s: quantisation gate said %s" % (label, not quant))
        if (abs(it) <= HEADROOM_ITEMS) != bool(plausible):
            fails.append("%s: plausibility gate disagrees" % label)
    # The panel's `ccF1` is MACRO-AVERAGED over both capped classes, whose
    # (K+n) differ (full_panel.py:404, average="macro"), so no single quantum
    # exists for it -- the exact items conversion is PER CLASS.
    macro = 0.5 * (2.0 * 3 / (333 + 418) + 2.0 * 3 / (74 + 418))
    if items_are_quantised(macro, 333, 418):
        fails.append("a two-class macro average passed a single-class quantum")
    report(fails, "items/quantisation gate failures")


def test_a_tie_is_never_reported_without_its_power():
    """GATE 7 -- `full_panel`'s RESOLUTION block. "No effect" and "not enough
    seeds" are opposite conclusions from one table; `dualbar2` reads +0.36
    items and needs ~174 seeds per cell against the 4 the protocol runs."""
    cases = [("dualbar2 +0.36 items", 0.36, 1.695, 4, "UNDERPOWERED", 174),
             ("exact zero -- bound it, never call it no effect", 0.0, 2.7, 4,
              "BOUNDED NULL", None),
             ("a real effect at 4 seeds (negative control)", 5.0, 2.7, 4,
              "POWERED", 3),
             ("no sd estimable (negative control)", 0.36, None, 1,
              "NOT REPORTABLE", None)]
    fails = []
    for label, eff, sd, have, want, need in cases:
        verdict, line = tie_report(eff, sd, have)
        if verdict != want:
            fails.append("%s: %s, expected %s" % (label, verdict, want))
        if need is not None and seeds_needed(abs(eff), sd) != need:
            fails.append("%s: needs %d seeds, expected %d"
                         % (label, seeds_needed(abs(eff), sd), need))
        if sd is not None and eff and ("sd" not in line or "seeds" not in line):
            fails.append("%s: report line omits the sd or the seed count"
                         % label)
    if not math.isfinite(detectable_at(2.7, 4)):
        fails.append("the zero-effect bound is not finite -- nothing to state")
    report(fails, "tie/power gate failures")


def test_the_four_noise_numbers_are_not_interchangeable():
    """GATE 8 -- FRAMEWORK 2(v). unpaired / reseed / treated / the panel's
    macro-averaged paired sd differ up to 12x, and on THIS design pairing GROWS
    the noise: `tralo` and `tralo_null` are two models, not two readings."""
    prize, unpaired, reseed, treated, panel_sd = 0.42, 0.80, 6.17, 7.59, 2.11
    cases = [("unpaired quoted for the paired contrast", "unpaired", True, False),
             ("panel sd quoted in TP items", "panel_macro_ccf1", True, False),
             ("treated -- the contrast run (negative control)",
              "treated", True, True),
             ("reseed floor -- the honest bar (negative control)",
              "reseed", True, True),
             ("unpaired for an ABSOLUTE quality claim (negative control)",
              "unpaired", False, True)]
    fails = ["%s: price() said %r" % (l, price(k, p))
             for l, k, p, ok in cases if (price(k, p) is None) != ok]
    if price("paired", True) is None:
        fails.append("an unnamed noise kind was accepted")
    if seeds_needed(prize, treated) / float(seeds_needed(prize, unpaired)) < 10:
        fails.append("substituting the unpaired sd did not change the seed cost")
    if treated <= unpaired:
        fails.append("fixture lost 2(v)'s finding that pairing GROWS the noise")
    if reseed <= prize:
        fails.append("the RNG-only floor no longer exceeds the whole prize")
    if panel_sd == treated:
        fails.append("the panel sd was substituted for the treated sd")
    # closed by the CAP CHOICE, not by physics: L20 hopeless, K/n=0.9 affordable
    if seeds_needed(0.42, 7.59) < 1000 or seeds_needed(29.83, 29.07) > 10:
        fails.append("the seeds column no longer separates hopeless from cheap")
    report(fails, "noise-substitution gate failures")


def test_arms_are_compared_at_EQUAL_SPEND_not_merely_under_the_cap():
    """GATE 10 -- FRAMEWORK 2(z31)d. Every feasibility check in the pipeline
    tests `count > limit`. NOTHING tested `count < K_eff`, so `lp` emitting 319
    against a budget of 333 logged as OK -- and `2TP/(K+n)` then handed it a
    SMALLER denominator for the 12 true items it forfeited.

    The detector needs no labels and no budget re-derivation: arms in one cell
    face the same K_eff, so a difference in emitted counts at a fixed seed IS
    unequal spend.
    """
    from scripts.deployed_h2h import ccf1, spend_audit

    fails = []

    # the real shape, dom1/MobileNetV2/L90_G95/seed_1, class 2
    short = {"clip": {1: dict(TP=319., per={2: dict(TP=319, K=333, n=370)})},
             "lp":   {1: dict(TP=307., per={2: dict(TP=307, K=319, n=370)})}}
    flag = spend_audit(short, (2,))
    if len(flag) != 1 or flag[0][3] != 14:
        fails.append("14 unspent slots not flagged: %r" % (flag,))

    # NEGATIVE CONTROL 1: equal spend must NOT fire, or the gate is vacuous.
    equal = {"clip":  {1: dict(TP=300., per={2: dict(TP=300, K=333, n=370)})},
             "tralo": {1: dict(TP=305., per={2: dict(TP=305, K=333, n=370)})}}
    if spend_audit(equal, (2,)):
        fails.append("fired on arms that spent the same budget")

    # NEGATIVE CONTROL 2: one arm alone cannot be unequal to anything.
    if spend_audit({"lp": short["lp"]}, (2,)):
        fails.append("fired on a single arm, which has nothing to differ from")

    # the DIRECTION of the harm, which is why the metric cannot police itself:
    # at FIXED TP, spending less RAISES cc-F1.
    lo = ccf1({2: dict(TP=307, K=319, n=370)}, (2,))
    hi = ccf1({2: dict(TP=307, K=333, n=370)}, (2,))
    if not lo > hi:
        fails.append("under-spend did not raise ccF1 at fixed TP (%.4f vs %.4f)"
                     % (lo, hi))

    # and the over-emission direction is still someone else's job, but must not
    # be silently swallowed here either.
    over = {"clip": {1: dict(TP=300., per={2: dict(TP=300, K=333, n=370)})},
            "bad":  {1: dict(TP=310., per={2: dict(TP=310, K=352, n=370)})}}
    if not spend_audit(over, (2,)):
        fails.append("an arm emitting MORE than its peers was not flagged")

    report(fails, "equal-spend gate failures")


def test_capped_class_deltas_carry_macro_and_uncapped_beside_them():
    """GATE 9 -- PLAYBOOK 1.3 / FRAMEWORK 3(0c). `dom1` reads ccF1 +0.0141 (6/6)
    and macroF1 -0.0022 (2/6); macroF1 is carried by the 6 of 8 UNCAPPED
    classes, so it is unattributable until `uncF1` sits beside it."""
    cases = [("dom1 ccF1 alone", {"ccF1"}, ["macroF1", "uncF1"]),
             ("ccF1 + macroF1, no uncF1", {"ccF1", "macroF1"}, ["uncF1"]),
             ("macroF1 alone", {"macroF1"}, ["uncF1"]),
             ("the full row (negative control)",
              {"ccF1", "macroF1", "uncF1"}, []),
             ("allocation-free row, no capped metric (negative control)",
              {"AP", "AUROC"}, [])]
    fails = ["%s: missing %s, expected %s" % (l, table_ok(m), want)
             for l, m, want in cases if table_ok(m) != want]
    # uncF1 must actually be PRINTED, not merely computed: it sat in the frame
    # and reached no reader until 2026-08-30.
    from scripts.full_panel import EQ_RESOLUTION, GROUPS
    printed = {m for _h, ms in GROUPS for m in ms}
    for m in ("ccF1", "macroF1", "uncF1"):
        if m not in printed:
            fails.append("%s is computed but not in any printed family" % m)
        if m not in EQ_RESOLUTION:
            fails.append("%s is printed with no power statement" % m)
    report(fails, "ccF1-beside-macroF1 gate failures")
