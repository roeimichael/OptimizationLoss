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

from gates.conftest import items_from_f1, rel, report

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


# ==========================================================================
#   THE QUARANTINE REACHES EVERY SCORER. Source: scripts/quarantine.py, and
#   the 2026-09-04 audit that found five of seven scorers checked nothing.
# ==========================================================================
SCORERS = ("full_panel", "cell_table", "deployed_h2h", "score_scan",
           "paired_noise", "sensitivity_screen", "paper_rows",
           # Added 2026-09-04. A second audit found SIX more tools under
           # `scripts/` that enumerate runs and produce a per-arm contrast
           # while calling no quarantine symbol at all. Each is on this list
           # because BOTH halves of the verdict are checkable in it: it takes
           # campaign roots, so `gate(paths)` applies, and it names arms, so
           # the dead-arm half has somewhere to go.
           #
           # `family_split` was the worst of them: `--families` DEFAULTS to
           # `[tralo, fioretto, hounie]` and two of those three are dead arms
           # in `dom1`, `dom1b` and `equaldose1`, so the BARE invocation
           # printed a compute-vs-constraint split for disqualified arms with
           # no banner whatsoever. `arm_identity_check --pairs` defaults to
           # `alm:fioretto`, the same shape.
           "family_split", "arm_identity_check", "straddle_probe",
           "order_probe", "paired_seeds", "collateral_probe")

# Tools that must REFUSE a wholly-quarantined campaign but have NO dead-arm
# half to enforce, and are therefore held to the `blocked` check only.
#
# 🛑 THIS LIST IS NOT AN EXEMPTION HATCH, and it has exactly one member for a
# reason. `headroom` prints no arm-vs-arm contrast at all: `ceiling`,
# `achieved`, `headroom`, `excess` and `binds` are read from `--control`
# alone, so a dead arm cannot reach a column of it. Requiring `dead_ok` there
# would force a filter that filters nothing, and a gate that lists a tool it
# cannot check is worse than not listing it -- but so is a tool with no gate
# at all, which is what leaving `headroom` off both lists would mean.
#
# Anything that ranks, pairs or contrasts arms belongs in SCORERS instead.
GATE_ONLY_SCORERS = ("headroom",)


def _quarantine_symbols(src):
    """(imported gate symbols, called ones). AST, never grep.

    A name in a docstring is not a call, and an import is not a gate.
    """
    import ast
    tree = ast.parse(src)
    wanted = {"gate", "by_name", "is_quarantined", "refuses_scoring"}
    imported, called = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and \
                "quarantine" in n.module:
            imported |= {a.name for a in n.names} & wanted
        if isinstance(n, ast.Call):
            f = n.func
            name = (f.id if isinstance(f, ast.Name)
                    else f.attr if isinstance(f, ast.Attribute) else None)
            if name in wanted:
                called.add(name)
    return imported, called


def _gate_verdict_used(src):
    """(blocked_acted_on, dead_used) for a module that calls `gate`.

    CALLING the gate is not OBEYING it, and this test used to check only the
    call. Two separate defects hid behind that: a `blocked` that is bound and
    never tested, and a `dead` that is bound and never used -- the second was
    REAL in six of seven scorers on 2026-09-04, which is how `deployed_h2h`
    printed a PARTIAL banner and then ranked a dead arm #1.

    Both halves are traced to a use:
      * `blocked` must appear in the test of an `If`;
      * `dead` must be LOADED somewhere -- passed to `drop_dead_runs`, or
        intersected with the arm names, or filtered against.
    """
    import ast
    tree = ast.parse(src)

    # 🛑 SCOPE THE SEARCH TO THE FUNCTION THAT OBTAINED THE VERDICT. Walking
    # the whole module made this vacuous: `deployed_h2h`'s SELF-TEST has an
    # unrelated local also called `dead`, so deleting the real enforcement
    # left the module-wide load count unchanged and the mutation survived.
    host = None
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                f = n.value.func
                nm = (f.id if isinstance(f, ast.Name)
                      else f.attr if isinstance(f, ast.Attribute) else None)
                if nm == "gate":
                    host = fn
                    break
        if host is not None:
            break
    if host is None:
        return False, False

    blocked_names, dead_names = set(), set()
    for n in ast.walk(host):
        if not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Call):
            continue
        f = n.value.func
        nm = (f.id if isinstance(f, ast.Name)
              else f.attr if isinstance(f, ast.Attribute) else None)
        if nm != "gate":
            continue
        for tgt in n.targets:
            if isinstance(tgt, ast.Tuple) and len(tgt.elts) == 2:
                a, b = tgt.elts
                if isinstance(a, ast.Name):
                    blocked_names.add(a.id)
                if isinstance(b, ast.Name):
                    dead_names.add(b.id)
    if not blocked_names and not dead_names:
        return False, False

    tested = set()
    for n in ast.walk(host):
        if isinstance(n, ast.If):
            for sub in ast.walk(n.test):
                if isinstance(sub, ast.Name) and sub.id in blocked_names:
                    tested.add(sub.id)
    blocked_ok = bool(blocked_names) and blocked_names <= tested

    # The dead arms must reach a CALL or an operator -- passed to a filter,
    # intersected with the arm names, compared. A load into a print is an
    # announcement, and announcing is the defect, not the fix.
    used = set()
    for n in ast.walk(host):
        if isinstance(n, ast.Call):
            fname = (n.func.id if isinstance(n.func, ast.Name)
                     else n.func.attr if isinstance(n.func, ast.Attribute)
                     else None)
            if fname in ("print", "format", "join", "write"):
                continue
            operands = list(n.args) + [k.value for k in n.keywords]
        elif isinstance(n, (ast.Compare, ast.BinOp, ast.BoolOp)):
            operands = [n]
        else:
            continue
        for o in operands:
            for sub in ast.walk(o):
                if isinstance(sub, ast.Name) and sub.id in dead_names and                         isinstance(sub.ctx, ast.Load):
                    used.add(sub.id)
    dead_ok = bool(dead_names) and dead_names <= used
    return blocked_ok, dead_ok


def _explains_gate_only(src):
    """Does this module say, in words, why it enforces no dead-arm filter?

    The one thing a reader cannot recover from the code is INTENT: a scorer
    that forgot its filter and a control-only tool that needs none look
    identical at the call site. Requiring the sentence is what keeps
    GATE_ONLY_SCORERS from becoming a place to park an oversight.
    """
    return "no dead-arm filter" in src.lower().replace("_", "-")


def _namegate_verdict_used(src):
    """(refuses, filters) for a scorer gated by campaign NAME, not by path.

    `paper_rows` reads a `cell_table` CSV and has no directory to walk, so it
    cannot call `gate(paths)`; it resolves each campaign with `by_name` and
    DROPS rows. That is a different shape, not a weaker one -- it is the tool
    that decides what may be WRITTEN -- so it gets its own recogniser rather
    than an exemption, because an exemption by module name would silently
    cover a future regression in the same file.

    Four independent behaviours, all required:
      * `by_name` is called;
      * `dead_arms` is READ off the registry entry;
      * an `If` refuses with a non-zero return;
      * a comprehension FILTERS rows -- announcing is not dropping.
    """
    import ast
    tree = ast.parse(src)
    called = any(isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", None))
                 == "by_name" for n in ast.walk(tree))
    reads = any(isinstance(n, ast.Constant) and n.value == "dead_arms"
                for n in ast.walk(tree))
    refuses = any(
        isinstance(n, ast.If) and any(
            isinstance(s, ast.Return) and isinstance(s.value, ast.Constant)
            and s.value.value not in (0, None)
            for s in ast.walk(n))
        for n in ast.walk(tree))
    # `filters` must be the comprehension that references the DEAD-ARM
    # mapping, not merely SOME comprehension: paper_rows has a dozen, so the
    # loose version stayed True with the drop deleted and gated nothing.
    carriers = set()
    for n in ast.walk(tree):
        if not isinstance(n, ast.Assign):
            continue
        if not any(isinstance(v, ast.Constant) and v.value == "dead_arms"
                   for v in ast.walk(n.value)):
            continue
        for tgt in n.targets:
            for sub in ast.walk(tgt):
                if isinstance(sub, ast.Name):
                    carriers.add(sub.id)
    filters = any(
        isinstance(n, (ast.ListComp, ast.GeneratorExp))
        and any(isinstance(v, ast.Name) and v.id in carriers
                for c in n.generators for f in c.ifs for v in ast.walk(f))
        for n in ast.walk(tree))
    return (called and reads and refuses), filters


def test_every_scorer_actually_calls_the_quarantine_gate():
    """A marker only prevents a mistake in the tools that READ it.

    Audited 2026-09-04: `full_panel` and `cell_table` carried a private copy
    of the refusal each, and `deployed_h2h`, `paper_rows`, `score_scan`,
    `paired_noise` and `sensitivity_screen` checked NOTHING. Five of seven.
    `paper_rows` is the tool whose entire job is deciding what may be WRITTEN,
    so it was the worst one to leave open.

    AND CALLING THE GATE IS NOT OBEYING IT. Re-audited the same day: six of
    the seven bound the dead-arm half of the verdict and never referenced it
    again, so the PARTIAL banner printed and the dead arms were ranked anyway.
    Both halves are traced to a use here.

    AND THE LIST WAS TOO SHORT. Audited a third time the same day: SEVEN more
    tools under `scripts/` enumerate runs and produce a per-arm contrast while
    calling no quarantine symbol at all. `family_split` is the worst -- its
    `--families` defaults to `[tralo, fioretto, hounie]`, two of which are
    dead arms in `dom1`, `dom1b` and `equaldose1`, so the BARE invocation
    printed a compute-vs-constraint split on disqualified arms with no banner.
    Six of the seven joined SCORERS; `headroom` prints no arm-vs-arm contrast
    and is held to the `blocked` half in GATE_ONLY_SCORERS instead.

    NEGATIVE CONTROLS, all built from the REAL source so they cannot drift
    from what is checked: the call site stripped; the verdict bound and
    discarded; and the positive direction, so the checker is not one that
    simply rejects everything.
    """
    bad = []
    for mod in SCORERS:
        path = rel("scripts", "%s.py" % mod)
        if not os.path.exists(path):
            bad.append("%s: missing entirely" % mod)
            continue
        src = io.open(path, encoding="utf-8").read()
        imported, called = _quarantine_symbols(src)
        if not imported:
            bad.append("%s imports NO quarantine symbol, so a marker on a "
                       "dead campaign prevents nothing here" % mod)
            continue
        if not called:
            bad.append("%s imports %s but never CALLS it; an import is not a "
                       "gate" % (mod, sorted(imported)))
            continue
        blocked_ok, dead_ok = _gate_verdict_used(src)
        if not (blocked_ok or dead_ok):
            # No `gate(paths)` call. The only legitimate reason is that the
            # tool has no path to walk, which is `paper_rows` -- it must then
            # enforce by NAME, to the same standard.
            blocked_ok, dead_ok = _namegate_verdict_used(src)
        if not blocked_ok:
            bad.append("%s calls gate() but never TESTS the blocked half of "
                       "the verdict; it announces a refusal and proceeds"
                       % mod)
        if not dead_ok:
            bad.append("%s binds the DEAD-ARM half of the verdict and never "
                       "uses it: the PARTIAL banner prints and the dead arms "
                       "are scored anyway" % mod)

    # The gate-only tools. They are held to the `blocked` half ONLY -- see
    # GATE_ONLY_SCORERS for why that is not an exemption -- but they are held
    # to it, because the alternative is a tool nothing checks at all.
    for mod in GATE_ONLY_SCORERS:
        path = rel("scripts", "%s.py" % mod)
        if not os.path.exists(path):
            bad.append("%s: missing entirely" % mod)
            continue
        src = io.open(path, encoding="utf-8").read()
        imported, called = _quarantine_symbols(src)
        if not imported:
            bad.append("%s imports NO quarantine symbol, so a marker on a "
                       "dead campaign prevents nothing here" % mod)
            continue
        if not called:
            bad.append("%s imports %s but never CALLS it; an import is not a "
                       "gate" % (mod, sorted(imported)))
            continue
        if not _gate_verdict_used(src)[0]:
            bad.append("%s calls gate() but never TESTS the blocked half of "
                       "the verdict; it announces a refusal and proceeds"
                       % mod)
        # A gate-only tool must say WHY it has no dead-arm filter, at the call
        # site. Without that line the next reader cannot tell a deliberate
        # control-only tool from a scorer whose filter was forgotten -- which
        # is the whole distinction this second list encodes.
        if not _explains_gate_only(src):
            bad.append("%s is on the gate-only list and does not say at the "
                       "call site why it has no dead-arm filter; that makes "
                       "it indistinguishable from a forgotten one" % mod)

    real = io.open(rel("scripts", "full_panel.py"), encoding="utf-8").read()

    # CONTROL 1: no call at all.
    stripped = real.replace("blocked, DEAD_ARMS = gate(",
                            "blocked, DEAD_ARMS = (lambda *a, **k: (0, 0))(")
    _imp, called = _quarantine_symbols(stripped)
    if called:
        bad.append("control 1: the ungated build still reads as gated, so "
                   "this test cannot detect an ungated scorer")

    # CONTROL 2: the verdict is bound and BOTH halves discarded -- the exact
    # shape six scorers shipped in, and the shape this test used to PASS.
    discard = real.replace("blocked, DEAD_ARMS = gate(",
                           "_unused_b, _unused_d = gate(")
    b_ok, d_ok = _gate_verdict_used(discard)
    if b_ok:
        bad.append("control 2: a scorer that never tests `blocked` still "
                   "reads as obeying the gate")
    if d_ok:
        bad.append("control 2: a scorer that binds the dead arms and never "
                   "uses them still reads as enforcing the partial marker")

    # CONTROL 3: the positive direction. Without it, controls 1 and 2 would
    # also pass for a checker that rejects every input.
    b_ok, d_ok = _gate_verdict_used(real)
    if not (b_ok and d_ok):
        bad.append("control 3: the real full_panel does not satisfy the "
                   "checker, so this test rejects everything and gates "
                   "nothing (blocked=%s dead=%s)" % (b_ok, d_ok))

    # CONTROL 4: the name-gated shape must fail when it ANNOUNCES the dead
    # arms and does not DROP them -- the exact defect this whole audit found.
    pr = io.open(rel("scripts", "paper_rows.py"), encoding="utf-8").read()
    ok_ref, ok_filt = _namegate_verdict_used(pr)
    if not (ok_ref and ok_filt):
        bad.append("control 4: the real paper_rows fails its own recogniser "
                   "(refuses=%s filters=%s), so that recogniser gates nothing"
                   % (ok_ref, ok_filt))
    # Build the announce-only variant LINE-WISE: an embedded multi-line
    # literal here is what a heredoc mangles, and a control that does not
    # parse is not a control.
    pr_lines = pr.split(chr(10))
    hit = [k for k, L in enumerate(pr_lines)
           if L.strip().startswith('rows = [r for r in rows')]
    if not hit:
        bad.append('control 4: the row FILTER in paper_rows was not found, '
                   'so the announce-only control proves nothing')
    else:
        k = hit[0]
        announce_only = chr(10).join(
            pr_lines[:k] + ['        pass  # announce only'] + pr_lines[k + 2:])
        if _namegate_verdict_used(announce_only)[1]:
            bad.append('control 4: a paper_rows that only ANNOUNCES the '
                       'dead arms still reads as enforcing them')

    # CONTROL 5: the GATE-ONLY branch, in both directions. It checks one half
    # of the verdict instead of two, so it is the branch most likely to be
    # silently vacuous -- a check that passes for a tool with no gate at all
    # would make the second list strictly worse than no list.
    hr = io.open(rel("scripts", "headroom.py"), encoding="utf-8").read()
    hr_call = "blocked, _ = gate("
    if hr_call not in hr:
        bad.append("control 5: the gate call site in headroom was not found, "
                   "so the gate-only controls prove nothing")
    else:
        # 5a: the call removed. It must read as CALLING nothing.
        if _quarantine_symbols(
                hr.replace(hr_call, "blocked, _ = (lambda *a, **k: (0, 0))("))[1]:
            bad.append("control 5a: an ungated headroom still reads as "
                       "calling the gate")
        # 5b: the verdict bound and discarded -- announce and proceed.
        if _gate_verdict_used(hr.replace(hr_call, "_unused_b, _d = gate("))[0]:
            bad.append("control 5b: a headroom that never tests `blocked` "
                       "still reads as obeying the gate")
        # 5c: the positive direction, stated rather than inferred.
        if not _gate_verdict_used(hr)[0]:
            bad.append("control 5c: the real headroom fails the blocked-half "
                       "recogniser, so the gate-only branch rejects "
                       "everything and gates nothing")
    # 5d: the "say why" requirement, both directions, through the SAME
    # predicate the loop uses so the two cannot drift.
    if not _explains_gate_only(hr):
        bad.append("control 5d: the real headroom carries no explanation, so "
                   "the requirement above rejects everything")
    if _explains_gate_only(hr.replace("NO DEAD-ARM FILTER", "NO FILTER")):
        bad.append("control 5d: a headroom with the explanation deleted still "
                   "reads as explaining itself")

    report(bad, "ungated scorers")


def test_a_partial_quarantine_drops_arms_without_killing_the_campaign():
    """`scorable=False` blocks everything; a PARTIAL marker blocks arms.

    Three campaigns (`dom1`, `dom1b`, `equaldose1`, 792 runs) ran `fioretto`
    and `hounie` at 28.00 attempted constraint steps against `tralo`'s 29.00,
    which is the same defect that quarantined `vitdual1`. But they also carry
    the independent units behind the headline `tralo` vs `clip` claim, which
    is at equal dose and untouched. A blanket marker would delete the evidence
    for a live claim in order to describe a defect touching two arms, so the
    registry grew a third state.

    NEGATIVE CONTROLS, in both directions:
      * a partial marker must NOT hard-block, or the headline evidence dies;
      * it must still RETURN the dead arms, or it is a marker that does
        nothing while the table still looks complete;
      * `scorable=False` must still hard-block, or the third state has
        quietly disabled the second;
      * an UNREGISTERED campaign must still read clean, or the registry
        fallback is matching everything.
    """
    from scripts.quarantine import (REGISTRY, dead_arms, gate,
                                    is_quarantined, refuses_scoring)

    bad = []
    PARTIAL = {"dom1": {"fioretto", "hounie"},
               "dom1b": {"fioretto", "hounie"},
               "equaldose1": {"fioretto", "hounie", "tralo_lam0"}}
    for camp, want in sorted(PARTIAL.items()):
        e = REGISTRY.get(camp)
        if not e:
            bad.append("%s is not in the registry; the 29-vs-28 dose gap "
                       "measured there is unmarked and will be scored" % camp)
            continue
        if e.get("scorable") is not True:
            bad.append("%s is marked wholly unscorable, which deletes the "
                       "equal-dose evidence it also carries" % camp)
        if set(e.get("dead_arms") or ()) != want:
            bad.append("%s dead_arms %s, expected %s"
                       % (camp, sorted(e.get("dead_arms") or ()), sorted(want)))
        root = os.path.join("results", camp)
        if refuses_scoring(root) is not None:
            bad.append("%s hard-blocks; a partial marker must not" % camp)
        if dead_arms(root) != want:
            bad.append("%s: dead_arms(root) did not resolve to %s"
                       % (camp, sorted(want)))
        blocked, dead = gate([root], out=io.StringIO())
        # 🛑 RESOLVE PER CAMPAIGN. `gate` returns a `DeadArms` mapping
        # {campaign -> arms}, not a flat set, because a union over several
        # roots deleted `fioretto` and `hounie` from `taskwin2` -- which
        # carries no marker at all -- while printing "everything else in this
        # campaign is unaffected". This assertion compared the MAPPING to a
        # set of arm names and had been failing since that change landed;
        # `.for_path` is the API the six path-based scorers call, so checking
        # it here checks what they actually get.
        if blocked or dead.for_path(root) != want:
            bad.append("%s: gate() returned blocked=%s dead=%s"
                       % (camp, blocked, sorted(dead.for_path(root))))
        # And the per-campaign resolution must be a RESTRICTION, not a
        # relabelling: a campaign with no marker must come back empty even
        # when a marked one was gated in the same call.
        _b, both = gate([root, os.path.join("results", "taskwin2")],
                        out=io.StringIO())
        if both.for_path(os.path.join("results", "taskwin2")):
            bad.append("%s: gating it alongside an UNMARKED campaign killed "
                       "arms in the unmarked one" % camp)

    # The REGISTRY is the source of truth and the marker is only its on-disk
    # copy. These paths do not exist on this machine at all, and must still be
    # caught: markers are written on ONE host while scoring happens in
    # fourteen worktrees and on a laptop.
    if is_quarantined(os.path.join("results", "dom1")) is None:
        bad.append("a registry entry with no marker on disk reads as clean")
    if is_quarantined(os.path.join("results", "no_such_campaign_xyz")):
        bad.append("an unregistered campaign reads as quarantined; the "
                   "registry fallback is matching too eagerly")

    hard = sorted(k for k, v in REGISTRY.items() if v.get("scorable") is False)
    if not hard:
        bad.append("no fully-unscorable entries remain, so the hard refusal "
                   "is untested by any real registry row")
    for k in hard[:3]:
        blocked, _d = gate([os.path.join("results", k)], out=io.StringIO())
        if not blocked:
            bad.append("%s has scorable=False but gate() let it through" % k)

    report(bad, "partial-quarantine failures")


def test_the_cross_arm_dose_asymmetry_is_detectable_from_configs_alone():
    """29/29/28/28 with every arm reading 100%. The shape that got through.

    Each arm's percentage is applied/attempted WITHIN that arm, so an arm that
    never ATTEMPTS a step it should have attempted reads a clean 100.0%. Only
    the DENOMINATORS differ, and nothing but a cross-arm comparison sees it.
    Four campaigns carried this shape before anyone looked.

    NEGATIVE CONTROL: an equal-dose campaign must print NOTHING, or the
    detector fires on everything and gets ignored. And a ONE-step gap must
    still fire, because the real defect was exactly one epoch in 29.
    """
    from scripts.dose_landed import cross_arm_attempts

    def render(per):
        buf = io.StringIO()
        cross_arm_attempts(per, buf)
        return buf.getvalue()

    bad = []
    # (applied, attempted, runs) -- the shape dose_landed builds per arm.
    GAP = {"tralo": (696, 696, 24), "alm": (696, 696, 24),
           "fioretto": (672, 672, 24), "hounie": (672, 672, 24)}
    out = render(GAP)
    if "CROSS-ARM" not in out:
        bad.append("the 29-vs-28 shape printed nothing; this is exactly what "
                   "dom1, dom1b, equaldose1 and vitdual1 all carried")
    for arm in ("fioretto", "hounie"):
        if arm not in out:
            bad.append("%s is not named in the asymmetry report" % arm)
    if "3.4%" not in out:
        bad.append("the gap SIZE is not quoted, and '28 against 29' is the "
                   "whole finding: %r" % out)

    EQUAL = {"tralo": (696, 696, 24), "alm": (696, 696, 24),
             "fioretto": (696, 696, 24), "hounie": (696, 696, 24)}
    if render(EQUAL).strip():
        bad.append("an EQUAL-dose campaign printed an asymmetry, so the "
                   "detector fires always and will be ignored")

    NEAR = dict(EQUAL, hounie=(695, 695, 24))
    if "CROSS-ARM" not in render(NEAR):
        bad.append("a one-step gap is invisible, and the real defect was "
                   "exactly one epoch per run")

    report(bad, "cross-arm dose failures")


def test_the_gate_announces_cells_that_do_not_pose_the_cap_question(tmp_path):
    """A campaign can be mechanically PERFECT and still measure nothing.

    `uniform1` ran 252 runs at 1044/1044 constraint steps with zero collapse,
    zero non-finite values, clean parity and one code version -- and all NINE
    of its cells sit outside the measured task window (L20/L30/L50, the regime
    2(z16) closed). `vittask1` is the same shape at 13 runs: both its ViTB16
    cells are `non_task` because class 2 sits at K/n 0.600 and 0.700 against a
    measured strict band of [0.80, 0.90]. Nothing went wrong mechanically in
    either, so no health check could fire, and every scorer printed a full
    plausible panel.

    So the gate classifies the CELLS it is about to read. Hand-listing the two
    campaigns we happen to know about catches those two and goes stale on the
    next one.

    FOUR NEGATIVE CONTROLS, because each failure mode is silent in a different
    way:
      * a campaign whose cells are all `task` must produce NO banner, or the
        warning is noise and will be ignored;
      * a campaign with a non-task cell MUST produce one naming that cell;
      * the four not-a-task statuses must stay DISTINGUISHABLE in the output
        -- `unmeasured` is an absence of measurement and `non_task` is a
        measured verdict, and collapsing them is 2(z25)'s inversion;
      * a checkout with no task-window instrument must announce UNVERIFIABLE,
        never silence. A PINNED campaign worktree genuinely predates
        `task_windows.yml`, and reading that as "no problems found" is how a
        gate becomes decoration.
    """
    import json
    from scripts import quarantine as Q

    bad = []

    def tree(root, cells):
        """Build the standard <root>/<model>/<ds>/<cap>/<arm>/seed_N layout."""
        for (ds, model, cap) in cells:
            d = os.path.join(str(root), model, ds, cap, "tralo", "seed_1")
            os.makedirs(d)
            io.open(os.path.join(d, "config.json"), "w",
                    encoding="utf-8").write(json.dumps({"arm": "tralo"}))
        return str(root)

    CELLS = [("iwildcam", "ViTB16", "L80-80_G95"),
             ("iwildcam", "ViTB16", "L60-90_G95")]
    root = tree(tmp_path / "camp", CELLS)

    # the layout must actually be parsed -- if cell_status returns {} the whole
    # gate is vacuous and every control below passes for the wrong reason.
    got = Q.cell_status(root)
    if got is None:
        bad.append("cell_status returned None in a checkout that HAS the "
                   "instrument; UNVERIFIABLE is being reported as the normal "
                   "case and no cell will ever be checked")
    elif set(got) != set(CELLS):
        bad.append("cell_status parsed %s from the standard layout, not %s -- "
                   "the path walk is broken, so the gate reads zero cells and "
                   "silently passes everything" % (sorted(got), sorted(CELLS)))

    def banner(cells_map):
        """Run the announcer against a forced status map."""
        real, Q.cell_status = Q.cell_status, lambda r: cells_map
        buf = io.StringIO()
        try:
            Q._announce_cells([root], buf)
        finally:
            Q.cell_status = real
        return buf.getvalue()

    # 1. all task -> silence
    out = banner({c: "task" for c in CELLS})
    if out.strip():
        bad.append("a campaign whose cells are ALL `task` still printed a "
                   "warning, so the banner is noise:\n%s" % out)

    # 2. one non-task -> named
    out = banner({CELLS[0]: "task", CELLS[1]: "non_task"})
    if "L60-90_G95" not in out or "non_task" not in out:
        bad.append("a non_task cell was NOT named by the gate:\n%s" % out)
    if "L80-80_G95" in out:
        bad.append("the healthy cell was named too, so the banner does not "
                   "say WHICH cell is the problem:\n%s" % out)

    # 3. every status that is not `task` must be caught, and stay distinct.
    # 🛑 ASSERT ON THE PER-CELL ROW, NOT ON THE WHOLE OUTPUT. The banner ends
    # with a static footer that NAMES all five statuses, so `st in out` was
    # true for every status whatever the rows said -- a control that passes on
    # its own legend. The row is `<model> <dataset> <cap> <status>`, so the
    # status has to land on the line carrying the cap.
    for st in ("non_task", "no_strict_band", "unmeasured", "no_window",
               "no_data"):
        out = banner({CELLS[0]: st, CELLS[1]: st})
        rows = [L for L in out.split(chr(10))
                if "L70-90_G95" in L or "L80-80_G95" in L]
        if not rows:
            bad.append("status %r produced no per-cell row at all, so the "
                       "banner names no cell" % st)
        elif not any(st in L for L in rows):
            bad.append(
                ("status %r is not on the CELL's own row -- it is being "
                 "collapsed into another verdict, and only the static "
                 "footer still names it:" + chr(10) + out) % st)
        if "ALL 2 OF 2" not in out:
            bad.append("a campaign with NO usable cell did not say so for "
                       "%r; `some cells` and `no cells` are different "
                       "conclusions:\n%s" % (st, out))
    if "task" not in Q.NOT_A_TASK and "partial" in Q.NOT_A_TASK:
        bad.append("`partial` is being treated as not-a-task; it means the "
                   "cap binds in SOME seeds, which is conservative, not empty")
    if "task" in Q.NOT_A_TASK:
        bad.append("`task` is in NOT_A_TASK, so every cell reads as a defect")

    # 4. no instrument -> UNVERIFIABLE, never silence. Break the REAL import
    # rather than stubbing cell_status, or this control never runs the code it
    # is about: a pinned worktree fails exactly here, inside the import.
    import configs.task_cells as TC
    realw, TC.load_windows = TC.load_windows, _raise
    try:
        if Q.cell_status(root) is not None:
            bad.append("with the task-window instrument broken, cell_status "
                       "returned a dict instead of None; the empty dict reads "
                       "as `no problems found`, and a PINNED campaign worktree "
                       "takes this path routinely")
        buf = io.StringIO()
        Q._announce_cells([root], buf)
        if "UNVERIFIABLE" not in buf.getvalue():
            bad.append("a checkout with no task-window instrument printed %r "
                       "instead of announcing that nothing was verified"
                       % buf.getvalue())
    finally:
        TC.load_windows = realw

    # 4b. and the banner must reach the caller through gate(), not merely
    # exist. Five scorers once imported a refusal nobody called.
    real, Q.cell_status = Q.cell_status, lambda r: {CELLS[0]: "non_task"}
    buf = io.StringIO()
    try:
        blocked, _dead = Q.gate([root], out=buf)
    finally:
        Q.cell_status = real
    if "DO NOT POSE THE CAP QUESTION" not in buf.getvalue():
        bad.append("gate() did not announce the non-task cell; the announcer "
                   "exists but nothing calls it, so every scorer stays "
                   "blind: %s" % buf.getvalue())
    if blocked:
        bad.append("a non-task cell HARD-BLOCKED the campaign; it is an "
                   "announcement, not a refusal -- `dom1` has 2 task and 4 "
                   "partial cells and must stay scorable")

    # 5. and the two campaigns that motivated this are marked
    for camp in ("uniform1", "vittask1"):
        e = Q.REGISTRY.get(camp)
        if not e:
            bad.append("%s measured no cap question at all and is not in the "
                       "registry" % camp)
        elif e.get("scorable") is not False:
            bad.append("%s is registered but still scorable" % camp)
        elif not e.get("keep_for"):
            bad.append("%s is marked without saying what it is still a "
                       "receipt for; dead and worthless are different" % camp)

    report(bad, "cell-status gate defects")


def _raise(*a, **k):
    """Stand-in for an instrument that is absent, not one that returns empty."""
    raise ImportError("configs/task_windows.yml does not exist in this "
                      "checkout (simulated pinned worktree)")
