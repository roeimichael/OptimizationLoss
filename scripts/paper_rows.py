"""PAPER-LEVEL ROWS -- one line per (cell, contrast), and NOTHING averaged.

WHY THIS EXISTS, BESIDE `cell_table` AND `full_panel`.

  `cell_table`  is the SURVEY: the absolute level each arm reached, per cell.
  `full_panel`  prints CONTRASTS, but macro-averaged and campaign-wide.
  neither       says what a paper row needs: THIS cell, THIS contrast, in
                ITEMS, against the noise THAT contrast actually faces, with
                the seeds it would take to resolve it, and whether the cell
                posed a question at all.

A number that survives averaging across cells is not a result here. Rule 4:
the atomic cell is (dataset, backbone, cap, method) over 4 seeds, and SEED IS
THE ONLY AXIS THAT MAY BE COLLAPSED. This file collapses nothing else, and
refuses to print a mean over cells even when asked -- the mean over cells is
how three claims in this project were retracted.

THREE THINGS IT ADDS THAT NOTHING ELSE PRINTS TOGETHER:

1. **THE CELL'S OWN STATUS.** A contrast measured where the cap poses no
   question is not a null, it is an absence of measurement. Every row carries
   `task` / `partial` / `unmeasured` / `non_task` from the measured windows
   (FRAMEWORK 2(z16), 2(z24b)), so a reader cannot quote a non-task row.

2. **THE INDEPENDENT UNIT.** Cells are not replicates. `dom1` and `loose1`
   share ONE lambda=0 model byte-identically in 8/8 (cap, seed) pairs, and
   within one campaign two cap levels share one warm-up. Eight cells can be
   four units, and a 4/4 sign test is p=0.0625, not p=0.0039. Rows are
   labelled with the md5-derived unit, so the sign test is over units.

3. **THE CONTRAST'S OWN NOISE.** Four different noise numbers exist here and
   they differ up to 12x (FRAMEWORK 2(v)). Each row quotes the within-cell
   seed sd of ITS OWN contrast, in items, and the seeds per cell needed at 80%
   power -- so a tie is never printed without saying whether it is "no effect"
   or "not enough seeds", which are opposite conclusions from the same table.

    python -m scripts.paper_rows --cells cells.csv --out paper_rows.csv
    python -m scripts.paper_rows --self-test
"""
import argparse
import collections
import csv
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from scripts import quarantine  # noqa: E402  (path set above)

# MEASURED INDEPENDENT UNITS, and THE UNIT IS (backbone, HOST).
#
# Measured 2026-09-01 by md5'ing `final_predictions_raw.csv` of every
# `tralo_null` on iwildcam across all 14 worktrees. The result is not one model
# per campaign -- it is EXACTLY TWO per (backbone, seed), however many
# campaigns exist. Nine MobileNetV3 campaigns share two models. And the two
# groups are the two HOSTS:
#
#   group a  RTX PRO 6000 (dsisco02)  bfloat16  grad_scaler False
#            dom1  loose1  uniform1  xfam1
#   group B  Quadro RTX 6000 (dsisco01)  float16  grad_scaler True
#            equaldose1  iwc1  iwc3  iwc4  taskwin2
#
# `base_model_id` is IDENTICAL across both groups (`MobileNetV3_iwildcam_
# f598484ecba1`), so the id cannot separate them and only the md5 can. What
# differs is the numerics of the 29 lambda=0 epochs, not the warm-up.
#
# 🛑 CONSEQUENCE: A NEW CAMPAIGN ON AN ALREADY-USED (backbone, host) BUYS NO
# UNIT. There are 4 backbones x 2 hosts = 8 possible units on iwildcam and four
# are spent. `taskwin2` (MobileNetV3 x dsisco01) and `vittask1` (ViTB16 x
# dsisco01) are units 5 and 6 because they are new BACKBONES, not new
# campaigns.
#
# ⚠️ AND SAY WHAT THE AXIS IS. These units are independent MODELS. They are
# not independent datasets, splits or tasks -- all four share one iwildcam
# slice. A sign test over them supports "the sign is stable across backbones
# and numerics", never "across datasets".
#
# An entry ABSENT here is UNVERIFIED, not independent -- the default must not
# be the flattering one.
# ARCHIVED 2026-09-02: `loose1` ran `constraint_grad_mode: clip`, not the
# current `normalize`, so it is not the same method. It supplied the old "B2"
# and half of "A1". Removing it LOSES NOTHING on MobileNetV2 -- loose1's
# `tralo` there is byte-identical to dom1's in 4/4 seeds, because `clip` scales
# by min(raw_norm, 1.0) and IS `normalize` wherever the raw norm is >= 1 -- and
# it REMOVES the one dissenting unit, which was measured on the wrong recipe.
MEASURED_UNITS = {
    ("dom1", "MobileNetV2"): "A1",          # a / dsisco02
    ("equaldose1", "MobileNetV2"): "A2",    # B / dsisco01
    ("dom1b", "RegNetY400MF"): "B1",        # B / dsisco01
    # UNIT 5, md5-verified 2026-09-02. `taskwin2`'s MobileNetV3 null hashes to
    # 1aa30e5a25 -- the B/dsisco01 model it shares with equaldose1, iwc1, iwc3
    # and iwc4. None of those contributed a strict-task unit (their caps are
    # L20/L30/L50 or non_task), and MobileNetV3 is a backbone absent from
    # A1/A2/B1/B2 entirely, so this is genuinely the fifth.
    ("taskwin2", "MobileNetV3"): "C1",      # B / dsisco01
    # 🛑 AND `equaldose1`'s MobileNetV3 IS THE SAME UNIT, ADDED 2026-09-04.
    # It was absent, so its rows read `UNVERIFIED` while `docs/COVERAGE.md`'s
    # per-unit table aggregated them into C1 -- the label `C1` meant two
    # different things in two places, and the ledger's version was the one that
    # could hand out a free replicate. Verified rather than assumed: the
    # MobileNetV3 `tralo_null` of the two campaigns is BYTE-IDENTICAL in 4 of 4
    # seeds (1aa30e5a25, c14fea0ac3, 212ef19b0e, f5e73dd7cd). One model, one
    # unit. This does not change the unit COUNT -- it stops the same model
    # being counted twice.
    ("equaldose1", "MobileNetV3"): "C1",    # B / dsisco01, same model as above
    # 🛑 AND FOUR MORE CAMPAIGNS ARE THE SAME UNITS AGAIN (2026-09-04). Every
    # campaign missing from this table reads `UNVERIFIED`, which is the
    # cautious label -- but a reader counting labels counted EIGHT units on the
    # live corpus where there are FIVE, and `paper_rows` printed "6 of 8 carry
    # a task cell" against a truth of 3 of 5. UNVERIFIED protects a SIGN test
    # from a free replicate; it does not stop a COUNT being read off the line.
    # Verified by md5 of `final_predictions_raw.csv`, not assumed:
    #   coin1 / RegNetY400MF  == dom1b      8669c02f59  -> B1
    #   coin2 / MobileNetV2   == equaldose1 e7be738bc8  -> A2
    #   seed58a               is dom1b's backbone+host at SEEDS 5-8. New seeds
    #                         are new models and NOT a new unit -- that is the
    #                         whole point of the (backbone, host) key.
    ("coin1", "RegNetY400MF"): "B1",        # B / dsisco01, == dom1b
    ("coin2", "MobileNetV2"): "A2",         # B / dsisco01, == equaldose1
    ("seed58a", "RegNetY400MF"): "B1",      # B / dsisco01, dom1b seeds 5-8
    # 🛑 AND `price1` IS A2 AS WELL, ADDED 2026-09-11 THE DAY IT COMPLETED.
    # It is the campaign built to answer the noise question -- the first in the
    # corpus with THREE lambda=0 streams, so its floor rests on 12 observations
    # and clears MIN_FLOOR_OBS. That made it tempting to read as a NINTH unit
    # and it is not one: its MobileNetV2 `tralo_null` is BYTE-IDENTICAL to
    # `equaldose1`'s and `coin2`'s in 4 of 4 seeds (e7be738bc8, 7758aef831,
    # d77a1c47be, 0cf8acc779), while `dom1`'s MobileNetV2 differs at every seed
    # (7f1ff13ebc, ...) -- which is the A1-vs-A2 host split doing exactly what
    # the (backbone, host) key says it does.
    #
    # ⚠️ THIS ENTRY COSTS TraLO A UNIT, WHICH IS WHY IT IS HERE. Left
    # UNVERIFIED, price1's two cells sit in their own bucket and A2 reads
    # `2 of 3 cells TRALO`. Folded in where the md5 says they belong, A2 is
    # `2 of 5` and flips to `rival`. A ledger that only ever ADDS replicates
    # when they agree is not a ledger. FRAMEWORK 2(z97).
    ("price1", "MobileNetV2"): "A2",        # B / dsisco01, == equaldose1
    # !! C2 IS LICENSED AND CAN NEVER CARRY A `task` CELL (2026-09-10).
    # `configs/task_windows.yml` gives iwildcam/MobileNetV3 `strict class 2:
    # []` -- a band measured EMPTY, because the row is the INTERSECTION of
    # dom1's [0.60, 0.70] with equaldose1's, and they do not overlap.
    # `task_cells.classify` checks strict before partial, so every cap
    # fraction on the grid reads `partial` here and the restricted sign test
    # below (`cell_status == "task"`) can never see this unit. Reading C2
    # moves the LICENSED tally 4->5 and the TASK-RESTRICTED tally not at all:
    # C2 + D1 together give 6/6 p=0.0156 unrestricted but 4/4 p=0.0625
    # restricted, and six documents advertised only the first number.
    # Gated in tests/gates/test_g2_budget.py. FRAMEWORK 2(z75).
    ("dom1", "MobileNetV3"): "C2",          # A / dsisco02, the other MNv3 host
    # 🛑 UNIT D1: THE FIRST NON-iwildcam UNIT, AND IT WAS SITTING UNREAD
    # (2026-09-09). `bcn1mn3` is COMPLETE -- 228 runs, 4 seeds, and L80 and
    # L90 are verified task cells -- and it was absent from this table, so
    # every one of its rows read `UNVERIFIED` and it contributed nothing to
    # any sign test. Same defect class as the `add_seeds` pooling bug and
    # `shape1`'s third stream: the runs were bought, executed, and then not
    # read.
    #
    # 🔑 ITS INDEPENDENCE IS PROVED FROM CODE, NOT SAMPLED BY md5. Every other
    # entry here needed a hash comparison because two iwildcam campaigns CAN
    # share a warm-up. This one cannot, by construction:
    # `gen_campaign.compute_base_model_id` returns
    # `"%s_%s_%s" % (model_name, dataset_mode, h)` and additionally puts
    # `dataset_mode`, `data_dir` and `num_classes` INSIDE `h`. A bcn model's
    # id begins `MobileNetV3_bcn_` and an iwildcam model's `MobileNetV3_
    # iwildcam_`, so the caches are disjoint and the warm-ups were trained
    # separately on different data. md5 could only ever have sampled that;
    # the prefix settles it. Gated in `tests/test_lessons_learned.py`.
    #
    # ⚠️ THE UNIT IS LICENSED; ITS SIGN IS NOT YET READ. What this entry
    # asserts is independence. Whether TraLO clears its own lambda=0 floor
    # here is a separate question that needs `paper_rows` run against the
    # campaign, and SSH was down the day this was added.
    ("bcn1mn3", "MobileNetV3"): "D1",        # bcn / dsisco02 bf16, COMPLETE
    # UNITS 7 AND 8, added 2026-09-10 when `fmow1` completed at 304/304.
    # fmow is a THIRD dataset, so neither backbone can share a warm-up with
    # anything already in this table -- `base_model_id` hashes the dataset, and
    # no other campaign has ever run on fmow. No md5 comparison is needed to
    # establish independence here, which is the ONE case in this ledger where
    # that is true; every other entry required one.
    # `fmow1` staged BOTH backbones in one campaign, so the campaign spans two
    # units and the (campaign, backbone) key is what separates them -- the same
    # shape as `dom1` carrying A1 and C2.
    # ViTB16 is THE HEADLINE BACKBONE (FRAMEWORK 1-pre), and E2 is the first
    # unit that puts `tralo` against all three rival duals at EQUAL DOSE on it:
    # `dose_landed` reads 29.00 attempted steps/run for alm, fioretto, hounie
    # and every tralo variant, with only the retired `tralo_lam0` at 28.00.
    # That is what `vitdual2` was staged to provide and has not finished.
    ("fmow1", "MobileNetV3"): "E1",          # fmow / dsisco02 bf16, COMPLETE
    ("fmow1", "ViTB16"): "E2",               # fmow / dsisco02 bf16, COMPLETE
}

# The contrasts a paper row may carry, and what each one licenses.
CONTRASTS = [
    ("vs_clip", "clip",
     "the quality bar. `clip` is the stronger clipper; a win here is the "
     "headline claim"),
    ("vs_null", None,
     "the arm minus its OWN lambda=0 twin: the only contrast that attributes "
     "an effect to the CONSTRAINT rather than to the regime"),
    ("vs_reseed", "tralo_reseed",
     "the RNG noise floor -- same null with the RNG stream perturbed and "
     "nothing else. An effect below this is not an effect"),
]


def seeds_needed(mean, sd, power_const=7.85):
    """Seeds per cell for 80% power at alpha=0.05, two-sided."""
    if not sd or not mean or not (mean == mean) or not (sd == sd):
        return None
    return int(math.ceil(power_const * (sd / abs(mean)) ** 2))


_PROTO_CACHE = None


def _protocol_arms():
    """`arms:` from configs/protocol.yml, or {} if it cannot be read.

    Read, not restated: the family list used to be a literal
    `("tralo", "alm", "fioretto", "hounie")` here, which is FOUR of the FIVE
    roots the protocol declares -- `select` has its own `select_null` and was
    missing. A hardcoded copy of an authority drifts from it the day the
    authority changes, and nothing goes red.
    """
    global _PROTO_CACHE
    if _PROTO_CACHE is None:
        try:
            import yaml
            P = yaml.safe_load(io.open(os.path.join("configs", "protocol.yml"),
                                       encoding="utf-8"))
            _PROTO_CACHE = (P or {}).get("arms") or {}
        except Exception as exc:
            # SAY SO. An unreadable protocol means every twin resolves by
            # concatenation, which is the defect this function exists to
            # remove -- and it would degrade silently, on the tool that says
            # what may be WRITTEN. `--self-test` fails on this too.
            sys.stderr.write(
                "!! paper_rows: configs/protocol.yml unreadable (%s: %s). "
                "The lambda=0 twin cannot be resolved from the authority; "
                "rival-dual `vs_null` rows will be DROPPED.%s"
                % (type(exc).__name__, exc, chr(10)))
            _PROTO_CACHE = {}
    return _PROTO_CACHE


def _family_root(arm, arms):
    """The longest declared root `f` with `f + "_null"` an arm, or None."""
    best = None
    for f in {a[:-len("_null")] for a in arms if a.endswith("_null")}:
        if (arm == f or arm.startswith(f + "_")) and (
                best is None or len(f) > len(best)):
            best = f
    return best


def null_of(arm, present=None):
    """The lambda=0 twin an arm must be attributed against.

    NOT a fixed `tralo_null`: that is right for `tralo`, `tralo_cut`,
    `tralo_uniform` and `tralo_head`, which share one twin, and quietly WRONG
    for `alm`/`fioretto`/`hounie` on a campaign that really ran
    `<family>_null`. Returning the wrong twin attributes one arm's effect to
    another's model.

    🛑 BUT CONCATENATION ALONE DROPPED THE CONTRAST ENTIRELY FOR EVERY RIVAL
    DUAL, ON EVERY CAMPAIGN IN THE CORPUS (found 2026-09-10). This returned
    `alm_null` unconditionally; `build()` skips a contrast whose reference arm
    is not in the cell (`if not ref or ref not in arms: continue`); and NO
    current campaign runs `alm_null` / `fioretto_null` / `hounie_null` --
    `fmow1` carries 19 arms and none of the three. So `vs_null`, the contrast
    CONTRASTS itself calls "the only contrast that attributes an effect to the
    CONSTRAINT rather than to the regime", was emitted for `tralo` and every
    `tralo_*` variant and silently omitted for `alm`, `fioretto` and `hounie`.
    Not a wrong number -- a MISSING ROW, in the tool that says what may be
    written, which is why no gate saw it. The self-test PINNED the broken
    expectation (`("alm", "alm_null")`) and passed green.

    `scripts/family_split.py` had already solved this and its docstring names
    the trap in as many words: "Concatenation alone invented
    `tralo_uniform_null`, which exists nowhere". This is the same two rules,
    in the same order, reading the same authority:

      1. a DEDICATED `<family>_null` IF THE CELL RAN ONE -- xfam1's design,
         where the byte-identity of `fioretto_null` with `tralo_null` is a
         MEASUREMENT and resolving it away would discard a positive control;
      2. otherwise `null_sibling` from protocol.yml, which points the dual
         families and every `tralo_*` variant at the SHARED `tralo_null`,
         because at lambda = 0 they are all the same run -- FRAMEWORK's four
         byte-identical `_null` arms.

    `present` is the cell's arm set. Passing None means "not known", and then
    the protocol answers alone; it is never a licence to concatenate.
    """
    if arm.endswith(("_null", "_reseed", "_lam0")):
        return None
    arms = _protocol_arms()
    fam = _family_root(arm, arms)
    dedicated = (fam + "_null") if fam else None
    if dedicated and present is not None and dedicated in present:
        return dedicated
    sib = (arms.get(arm) or {}).get("null_sibling")
    if sib:
        return sib
    return dedicated


def load_cells(path):
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    if not rows:
        raise SystemExit("%s is empty" % path)
    need = {"campaign", "dataset", "model", "cap", "arm", "n_seeds",
            "items_per_001", "ccF1", "ccF1_sd"}
    missing = need - set(rows[0])
    if missing:
        raise SystemExit("%s is not a cell_table CSV: missing %s"
                         % (path, sorted(missing)))
    return rows


def build(rows, status_of=None, unit_of=None):
    """One record per (cell, arm, contrast). Nothing is averaged."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["campaign"], r["dataset"], r["model"], r["cap"])][r["arm"]] = r

    out = []
    for cell, arms in sorted(by.items()):
        camp, ds, model, cap = cell
        for arm, r in sorted(arms.items()):
            if arm.endswith(("_null", "_reseed")):
                continue
            scale = float(r["items_per_001"]) * 100.0   # ccF1 delta -> items
            for name, fixed, _why in CONTRASTS:
                ref = (fixed if fixed is not None
                       else null_of(arm, present=arms))
                if not ref or ref not in arms:
                    continue
                # 🛑 THE TWO MEANS MUST REST ON THE SAME SEEDS.
                # `mean(a) - mean(b) == mean(a - b)` only when the seed sets
                # are equal; on a ragged cell the difference is between two
                # POPULATIONS and is not a paired delta at all. This row is
                # what says what may be WRITTEN, so it refuses rather than
                # emits a number that looks like every other number in the
                # table. 2(z52). A COMPLETE cell is unaffected -- and that is
                # gated as a negative control, because a fix that silently
                # restated the corpus would be worse than the defect.
                sa_seeds = _seedset(r)
                sb_seeds = _seedset(arms[ref])
                ragged = (sa_seeds is not None and sb_seeds is not None
                          and sa_seeds != sb_seeds)
                shared = (len(sa_seeds & sb_seeds)
                          if sa_seeds is not None and sb_seeds is not None
                          else int(r["n_seeds"]))
                d = (float(r["ccF1"]) - float(arms[ref]["ccF1"])) * scale
                # sqrt(sa^2 + sb^2) is the sd of the per-seed DIFFERENCE at
                # rho = 0. The arms are correlated -- `tralo` and its null share
                # ONE warm-up epoch then train 29 apart -- so the true sd is
                # sqrt(sa^2 + sb^2 - 2*rho*sa*sb), which this may over- or
                # under-state depending on the sign of rho.
                # 🛑 IT IS BOUNDED EITHER WAY, AND THE OLD COMMENT HERE WAS NOT.
                # It claimed a "6-12x LOWER BOUND" off FRAMEWORK 2(v). That is
                # algebraically impossible: sd(A-B) <= sa + sb <= sqrt(2) *
                # sqrt(sa^2 + sb^2), so the WORST underestimate is 41%, and
                # positive correlation makes it an OVER-estimate instead.
                # 2(v)'s "0.80 vs 7.59" compares the paired difference sd to
                # ONE ARM's sd (`paired_noise`: unpaired = sd of one arm's TP@K
                # across seeds) -- a different quantity, and the treated arm's
                # own inflated variance is ALREADY inside sa here.
                # => `seeds_needed` below is accurate to within a factor of two,
                #    not an order of magnitude. Do not inflate it by hand.
                #    `scripts.paired_noise` still gives the directly measured
                #    per-class sd, which is the better number when available.
                sa = float(r["ccF1_sd"] or 0.0) * scale
                sb = float(arms[ref]["ccF1_sd"] or 0.0) * scale
                sd = math.sqrt(sa * sa + sb * sb)
                out.append(dict(
                    campaign=camp, dataset=ds, model=model, cap=cap, arm=arm,
                    contrast=name, ref=ref,
                    # the seeds behind THIS contrast, not the treated arm's own
                    n_seeds=shared,
                    items=d, sd_items=sd,
                    seeds_needed=seeds_needed(d, sd),
                    resolved=("RAGGED" if ragged else
                              "yes" if (sd and abs(d) >= 2.0 * sd) else "no"),
                    cell_status=(status_of or {}).get((ds, model, cap), "?"),
                    unit=(unit_of or {}).get(
                        (camp, model), "UNVERIFIED:" + camp + "/" + model),
                ))
    return out


def _seedset(row):
    """The seeds behind a `cell_table` row, or None if it did not say.

    `cell_table` grew the `seeds` column on 2026-09-07. A corpus CSV written
    before that has only `n_seeds`, and the honest answer there is "unknown",
    never "assume they match" -- an older file must not silently read as a
    clean paired contrast.
    """
    raw = row.get("seeds")
    if raw is None or (isinstance(raw, float)) or str(raw).strip() == "":
        return None
    return set(str(raw).split("|"))


def write(recs, path, out=sys.stdout):
    cols = ["campaign", "dataset", "model", "cap", "arm", "contrast", "ref",
            "cell_status", "unit", "n_seeds", "items", "sd_items",
            "seeds_needed", "resolved"]
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({c: r[c] for c in cols})
    out.write("wrote %d row(s) to %s%s" % (len(recs), path, chr(10)))


def render(recs, out=sys.stdout):
    """The table a paper row is read off. One line per (cell, contrast)."""
    w = out.write
    w("PAPER ROWS -- one line per (cell, contrast). NOTHING is averaged over "
      "cells." + chr(10))
    w("  `items` = d ccF1 * (K+n)/2. A sub-item delta is a re-allocation, "
      "not a" + chr(10))
    w("  difference. (The `1.9-9.9 items` band this line used to quote is a "
      "dermmnist" + chr(10))
    w("  number on a removed, leaking dataset -- see FRAMEWORK 2(z32)d -- and "
      "the" + chr(10))
    w("  iwildcam prize is regime-dependent, ~0.2-0.7 items/class at the "
      "retired" + chr(10))
    w("  tight caps against 7-31 at the live loose ones.)" + chr(10))
    w("  `items` IS APPROXIMATE: `full_panel` macro-averages ccF1 over BOTH "
      "capped" + chr(10))
    w("  classes, whose (K+n) differ, so the macro delta has no single "
      "quantum and one" + chr(10))
    w("  scale cannot be exact for both. Exact only PER CLASS. Signs and "
      "orders of" + chr(10))
    w("  magnitude are safe; do not quote an items figure to two decimals as "
      "exact." + chr(10))
    w("  `sd` and `seeds` are LOWER BOUNDS: they assume the two arms are "
      "independent," + chr(10))
    w("  and they are not -- an arm and its null are two MODELS sharing one "
      "warm-up," + chr(10))
    w("  measured at 6-12x this sd. `res?` = does |items| clear 2 sd. "
      "FRAMEWORK 2(v)." + chr(10) + chr(10))
    w("  %-10s %-13s %-12s %-12s %-11s %-10s %8s %8s %7s %5s%s"
      % ("campaign", "model", "cap", "arm", "contrast", "cell", "items",
         "sd", "seeds", "res?", chr(10)))
    for r in recs:
        w("  %-10s %-13s %-12s %-12s %-11s %-10s %+8.2f %8.2f %7s %5s%s"
          % (r["campaign"][:10], r["model"][:13], r["cap"][:12], r["arm"][:12],
             r["contrast"], r["cell_status"][:10], r["items"], r["sd_items"],
             (r["seeds_needed"] if r["seeds_needed"] is not None else "-"),
             r["resolved"], chr(10)))
    # ONE list, imported, not restated. This read
    # `("non_task", "unmeasured")` until 2026-09-04 and therefore stayed SILENT
    # on `no_strict_band` -- the status of `taskwin2`/`L70-90_G95`, which is
    # the only cell unit C1 contributes and the one that took the headline
    # sign test from three units to four. The tool whose whole job is saying
    # what may be WRITTEN printed that row with no warning at all.
    non = [r for r in recs if r["cell_status"] in quarantine.NOT_A_TASK]
    if non:
        by = {}
        for r in non:
            by[r["cell_status"]] = by.get(r["cell_status"], 0) + 1
        w(chr(10) + "  *** %d row(s) sit in a cell that poses NO measured "
          "question: %s." % (len(non),
                             ", ".join("%d %s" % (n, s)
                                       for s, n in sorted(by.items())))
          + chr(10))
        w("      A contrast there is an ABSENCE of measurement, not a null. "
          "Do not" + chr(10) + "      quote it. FRAMEWORK 2(z16), 2(z24b)."
          + chr(10))
    units = sorted({r["unit"] for r in recs})
    w(chr(10) + "  SIGN TESTS ARE OVER UNITS, NOT ROWS. Cells sharing a "
      "lambda=0 model are" + chr(10))
    w("  ONE unit: `dom1` and `loose1` are byte-identical in 8/8, and two cap "
      "levels" + chr(10))
    w("  in one campaign share a warm-up. %d row(s) over %d unit(s): %s"
      % (len(recs), len(units), ", ".join(units)) + chr(10))

    # 🛑 AND A UNIT WITH NO TASK CELL IS NOT A UNIT OF EVIDENCE ABOUT THE CAP.
    # Counted here rather than stated in a doc, because it was stated in a doc
    # and went stale: `MEASURED_UNITS` licensed FOUR units and the headline
    # read 4/4, p=0.0625, while unit C1 (`taskwin2`/MobileNetV3) contributes
    # only `no_strict_band` and `unmeasured` cells -- MobileNetV3 class 2's
    # strict band was re-measured EMPTY on 2026-09-02 with the per-group prize.
    # Restricted to units carrying at least one verified `task` cell the tally
    # is 3, and a sign test over 3 floors at 0.125. The signs themselves do not
    # change; a unit does.
    task_units = sorted({r["unit"] for r in recs if r["cell_status"] == "task"})
    empty = [u for u in units if u not in task_units]
    w(chr(10) + "  UNITS CARRYING AT LEAST ONE VERIFIED `task` CELL: %d of %d"
      % (len(task_units), len(units)) + chr(10))
    if empty:
        w("  *** %d unit(s) contribute NO task cell: %s"
          % (len(empty), ", ".join(empty)) + chr(10))
        w("      Their signs are real but they are not evidence that the CAP "
          "did anything," + chr(10) + "      because at those cells the cap "
          "poses no measured question. A sign test over" + chr(10))
        if task_units:
            w("      the remaining %d unit(s) floors at %.4f."
              % (len(task_units), 0.5 ** len(task_units)) + chr(10))
        else:
            # 0.5**0 is 1.0, which would print as a p-value and read as a
            # measured null. There is no sign test here at all.
            w("      NO unit carries a task cell, so there is no sign test to "
              "restrict -- not" + chr(10) + "      a p of 1.0. Nothing here "
              "is evidence about the cap." + chr(10))
    unver = sorted({u for u in units if u.startswith("UNVERIFIED")})
    if unver:
        w(chr(10) + "  *** %d unit label(s) are UNVERIFIED: %s"
          % (len(unver), ", ".join(unver)) + chr(10))
        w("      Nobody has md5'd their lambda=0 twin against the others, so "
          "they may" + chr(10) + "      be ONE model wearing several campaign "
          "names. Do NOT run a sign test" + chr(10) + "      over them until "
          "`scripts.flag_live` says they differ." + chr(10))
    return recs


def self_test(out=sys.stdout):
    """The gate. Both directions on every derived quantity."""
    ok = True
    w = out.write
    w("SELF-TEST -- does a paper row say what it claims?" + chr(10) + chr(10))

    # 1. the twin resolver must be per FAMILY, not a fixed tralo_null --
    #    AND it must fall back to the protocol when the campaign ran no
    #    dedicated twin, or the contrast is silently DROPPED.
    #
    # ⛔ THE EXPECTATIONS BELOW WERE MOVED DELIBERATELY ON 2026-09-10 AND THE
    # OLD ONES WERE WRONG. This list used to read ("alm", "alm_null"),
    # ("fioretto", "fioretto_null"), ("hounie", "hounie_null") with no
    # `present` argument at all, so it asserted exactly the behaviour that
    # dropped `vs_null` for all three rival duals on every campaign in the
    # corpus. A green self-test pinning a defect is worse than no self-test.
    #
    # `cell` is a campaign that ran the SHARED twin only, which is every
    # current campaign (`fmow1`: 19 arms, no `alm_null`).
    cell = {"clip", "tralo", "tralo_null", "tralo_reseed", "alm",
            "fioretto", "hounie"}
    # `xfam` is xfam1's design, where the dedicated twins were run on purpose
    # so that their byte-identity with `tralo_null` is a MEASUREMENT.
    xfam = cell | {"alm_null", "fioretto_null", "hounie_null"}
    cases = [
        # (arm, present, expected)
        ("tralo", cell, "tralo_null"),
        ("tralo_cut", cell, "tralo_null"),
        ("tralo_uniform", cell, "tralo_null"),
        # THE FIX: the shared twin, because no dedicated one was run.
        ("alm", cell, "tralo_null"),
        ("fioretto", cell, "tralo_null"),
        ("hounie", cell, "tralo_null"),
        # NEGATIVE CONTROL: a dedicated twin that WAS run must still win, or
        # the fix would silently delete xfam1's positive control.
        ("alm", xfam, "alm_null"),
        ("fioretto", xfam, "fioretto_null"),
        ("hounie", xfam, "hounie_null"),
        # `select` is the fifth root the old hardcoded 4-list did not know
        # about; the protocol gives it its OWN twin, not `tralo_null`.
        ("select", cell, "select_null"),
        # present=None means "not known" -- the protocol answers alone, and
        # it must never fall back to bare concatenation for a dual.
        ("alm", None, "tralo_null"),
        # arms that have no twin at all
        ("clip", cell, None),
        ("tralo_null", cell, None),
        ("tralo_reseed", cell, None),
    ]
    bad = [(a, null_of(a, present=p), e)
           for a, p, e in cases if null_of(a, present=p) != e]

    # 1a. THE ROOT LIST MUST BE DERIVED FROM protocol.yml, NOT RESTATED.
    # `null_of`'s protocol fallback rescues most arms even when the roots are
    # wrong, so testing only through `null_of` lets a hardcoded 4-list pass:
    # measured 2026-09-10, that exact mutation survived the case table above.
    # This asserts the derivation itself. `select` is the fifth root and the
    # one the old literal omitted.
    _pa = _protocol_arms()
    _roots = {a[:-len("_null")] for a in _pa if a.endswith("_null")}
    _rootbad = [r for r in _roots if _family_root(r, _pa) != r]
    if not _pa:
        w("  FAIL  protocol.yml unreadable -- the root list cannot be derived"
          + chr(10))
        ok = False
    elif "select" not in _roots or _rootbad:
        w("  FAIL  _family_root is not reading protocol.yml: roots=%s bad=%s%s"
          % (sorted(_roots), _rootbad, chr(10)))
        ok = False
    if bad:
        w("  FAIL  null_of: %s%s" % (bad, chr(10)))
        ok = False
    else:
        w("  PASS  the lambda=0 twin prefers a DEDICATED `<fam>_null` the "
          "cell actually ran," + chr(10) + "        and otherwise takes "
          "protocol.yml's `null_sibling` -- concatenation" + chr(10)
          + "        alone named an arm no campaign runs and DROPPED the "
            "contrast" + chr(10))

    # 1b. END TO END, because the resolver test above cannot see the drop.
    # `build()` skips any contrast whose reference arm is missing from the
    # cell, so a resolver returning a plausible-but-absent name loses the row
    # silently. This is the shape every corpus campaign has: the shared twin
    # only. Before the fix `alm` produced NO `vs_null` row here.
    _cellrows = []
    for _arm, _f1 in (("clip", 0.500), ("tralo", 0.520), ("alm", 0.530),
                      ("tralo_null", 0.510), ("tralo_reseed", 0.511)):
        _cellrows.append({"campaign": "zz", "dataset": "iwildcam",
                          "model": "MobileNetV2", "cap": "L80-80_G95",
                          "arm": _arm, "n_seeds": "4", "seeds": "1|2|3|4",
                          "items_per_001": "1.0", "ccF1": "%.3f" % _f1,
                          "ccF1_sd": "0.010"})
    _got = {(r["arm"], r["contrast"]) for r in build(_cellrows)}
    if ("alm", "vs_null") not in _got:
        w("  FAIL  build() drops ('alm', 'vs_null') -- the rival dual has no "
          "own-twin row" + chr(10))
        ok = False
    elif ("tralo", "vs_null") not in _got:
        w("  FAIL  build() lost tralo's own-twin row (regression)" + chr(10))
        ok = False
    else:
        w("  PASS  END TO END: a cell carrying the SHARED twin only still "
          "emits `vs_null`" + chr(10) + "        for a RIVAL DUAL -- the row "
          "that was silently missing corpus-wide" + chr(10))

    # 2. items conversion and the power formula, both directions
    if seeds_needed(1.0, 0.0) is not None or seeds_needed(0.0, 1.0) is not None:
        w("  FAIL  seeds_needed must refuse a zero sd or a zero effect"
          + chr(10))
        ok = False
    elif not (seeds_needed(1.0, 2.0) > seeds_needed(2.0, 2.0) > 0):
        w("  FAIL  seeds_needed must FALL as the effect grows" + chr(10))
        ok = False
    else:
        w("  PASS  seeds_needed: %d at 1 item vs %d at 2 items against sd 2"
          % (seeds_needed(1.0, 2.0), seeds_needed(2.0, 2.0)) + chr(10))

    # 3. end to end on a synthetic cell table, including the NEGATIVE control
    #    that a non-task cell is named rather than folded in.
    rows = []
    for arm, f1, sd in (("clip", 0.500, 0.004), ("tralo", 0.520, 0.004),
                        ("tralo_null", 0.505, 0.004),
                        ("tralo_reseed", 0.522, 0.004)):
        rows.append(dict(campaign="c1", dataset="iwildcam", model="MobileNetV2",
                         cap="L80_G95", arm=arm, n_seeds="4",
                         items_per_001="0.50", ccF1="%.3f" % f1,
                         ccF1_sd="%.3f" % sd))
    # 3b. THE RAGGED CELL. This is the fixture 2(z52) found missing from every
    #     self-test in the project: they all give every arm all four seeds, so
    #     a whole defect class was unreachable. `mean(a) - mean(b)` equals
    #     `mean(a - b)` ONLY on a shared seed set; when it does not, this row
    #     is a difference between two POPULATIONS and must not be written.
    ragged_rows = []
    for arm, f1, seeds in (("clip", 0.500, "1|2|3|4"),
                           ("tralo", 0.520, "1|2"),
                           ("tralo_null", 0.505, "1|3")):
        ragged_rows.append(dict(
            campaign="c1", dataset="iwildcam", model="MobileNetV2",
            cap="L80_G95", arm=arm, n_seeds=str(len(seeds.split("|"))),
            seeds=seeds, items_per_001="0.50", ccF1="%.3f" % f1,
            ccF1_sd="0.004"))
    rr = {r["contrast"]: r for r in build(
        ragged_rows,
        status_of={("iwildcam", "MobileNetV2", "L80_G95"): "task"})}
    if rr.get("vs_clip", {}).get("resolved") != "RAGGED":
        w("  FAIL  a contrast over DIFFERENT seed sets must read RAGGED, got "
          "%r%s" % (rr.get("vs_clip", {}).get("resolved"), chr(10)))
        ok = False
    elif rr["vs_clip"]["n_seeds"] != 2:
        w("  FAIL  n_seeds must be the SHARED count (2), got %r%s"
          % (rr["vs_clip"]["n_seeds"], chr(10)))
        ok = False
    elif rr.get("vs_null", {}).get("n_seeds") != 1:
        w("  FAIL  tralo{1,2} vs tralo_null{1,3} share ONE seed, got %r%s"
          % (rr.get("vs_null", {}).get("n_seeds"), chr(10)))
        ok = False
    else:
        w("  PASS  a contrast whose two arms ran DIFFERENT seeds reads RAGGED "
          "and" + chr(10) + "        reports the SHARED seed count, not the "
          "treated arm's own" + chr(10))

    # 3c. NEGATIVE CONTROL, and it is the one that matters most: the fix must
    #     be a NO-OP on every complete cell. A change that silently restated
    #     the corpus would be worse than the defect it repairs.
    square = []
    for arm, f1 in (("clip", 0.500), ("tralo", 0.520), ("tralo_null", 0.505)):
        square.append(dict(campaign="c1", dataset="iwildcam",
                           model="MobileNetV2", cap="L80_G95", arm=arm,
                           n_seeds="4", seeds="1|2|3|4", items_per_001="0.50",
                           ccF1="%.3f" % f1, ccF1_sd="0.004"))
    sq = {r["contrast"]: r for r in build(
        square, status_of={("iwildcam", "MobileNetV2", "L80_G95"): "task"})}
    if any(v["resolved"] == "RAGGED" for v in sq.values()):
        w("  FAIL  a COMPLETE cell must never read RAGGED" + chr(10))
        ok = False
    elif sq["vs_clip"]["n_seeds"] != 4 or round(sq["vs_clip"]["items"], 3) != 1.0:
        w("  FAIL  a complete cell must be byte-unchanged, got n=%r items=%r%s"
          % (sq["vs_clip"]["n_seeds"], sq["vs_clip"]["items"], chr(10)))
        ok = False
    else:
        w("  PASS  NEGATIVE CONTROL: a COMPLETE cell is unchanged -- same "
          "items, same" + chr(10) + "        seed count, never RAGGED"
          + chr(10))

    # 3d. And a corpus CSV written BEFORE the `seeds` column existed must not
    #     silently read as a clean paired contrast just because it cannot say.
    legacy = [dict(campaign="c1", dataset="iwildcam", model="MobileNetV2",
                   cap="L80_G95", arm=a, n_seeds="4", items_per_001="0.50",
                   ccF1="%.3f" % f, ccF1_sd="0.004")
              for a, f in (("clip", 0.500), ("tralo", 0.520))]
    lg = {r["contrast"]: r for r in build(legacy)}
    if lg["vs_clip"]["resolved"] == "RAGGED":
        w("  FAIL  a pre-`seeds` CSV cannot be PROVEN ragged and must not be "
          "marked so" + chr(10))
        ok = False
    else:
        w("  PASS  a CSV predating the `seeds` column abstains rather than "
          "asserting" + chr(10) + "        either way" + chr(10))

    recs = build(rows, status_of={("iwildcam", "MobileNetV2", "L80_G95"): "task"})
    got = {r["contrast"]: round(r["items"], 3) for r in recs}
    want = {"vs_clip": 1.0, "vs_null": 0.75, "vs_reseed": -0.1}
    if got != want:
        w("  FAIL  contrast arithmetic: got %s want %s%s" % (got, want, chr(10)))
        ok = False
    else:
        w("  PASS  all three contrasts computed in items, and `vs_reseed` is "
          "NEGATIVE" + chr(10) + "        here -- the arm does not clear its "
          "own RNG floor, which a" + chr(10) + "        `vs_clip` row alone "
          "would have hidden" + chr(10))

    buf = io.StringIO()
    render(build(rows, status_of={("iwildcam", "MobileNetV2", "L80_G95"):
                                  "non_task"}), out=buf)
    if "poses NO measured question" not in buf.getvalue():
        w("  FAIL  a non-task cell must be named, not folded in" + chr(10))
        ok = False
    else:
        w("  PASS  a non_task cell is named and disqualified" + chr(10))

    # A UNIT WITH NO TASK CELL, in both directions. This is the shape that
    # went unnoticed: `taskwin2`/MobileNetV3 is licensed as unit C1 and every
    # cell it contributes is `no_strict_band` or `unmeasured`, so the tally
    # read 4 units when 3 carry a verified task cell.
    for st, want_flag, label in (
            ("task", False, "a unit whose cell IS a task cell is not flagged"),
            ("no_strict_band", True,
             "a unit whose ONLY cell has an empty strict band is flagged, and "
             "the restricted sign-test floor is printed"),
            ("unmeasured", True,
             "and so is one whose only cell is an unmeasured K/n")):
        b = io.StringIO()
        render(build(rows, status_of={("iwildcam", "MobileNetV2", "L80_G95"):
                                      st}), out=b)
        txt = b.getvalue()
        flagged = "contribute NO task cell" in txt
        counted = "UNITS CARRYING AT LEAST ONE VERIFIED `task` CELL" in txt
        good = flagged == want_flag and counted
        if want_flag and good:
            # zero units survive here, and 0.5**0 = 1.0 must NOT print as a
            # p-value: "no sign test" and "p = 1.0" are opposite conclusions.
            good = ("no sign test to restrict" in txt
                    and "floors at 1.0000" not in txt)
        w("  %-4s %s%s" % ("PASS" if good else "FAIL", label, chr(10)))
        ok = ok and good

    # 4. NEGATIVE CONTROL on independence: two campaigns given the same unit
    #    must not be counted twice.
    u = {("c1", "MobileNetV2"): "A", ("c2", "MobileNetV2"): "A"}
    rows2 = [dict(r, campaign="c2") for r in rows]
    recs2 = build(rows + rows2,
                  status_of={("iwildcam", "MobileNetV2", "L80_G95"): "task"},
                  unit_of=u)
    if len({r["unit"] for r in recs2}) != 1:
        w("  FAIL  two campaigns sharing a model must collapse to ONE unit"
          + chr(10))
        ok = False
    else:
        w("  PASS  byte-identical campaigns collapse to one unit, so a 2-cell "
          "agreement" + chr(10) + "        cannot be sold as two "
          "replicates" + chr(10))

    # 5. NEGATIVE CONTROL on the DEFAULT: an unmeasured pair must read
    #    UNVERIFIED, never quietly become its own independent unit. The
    #    flattering default is the whole defect class of FRAMEWORK 2(z25).
    buf = io.StringIO()
    render(build(rows + rows2,
                 status_of={("iwildcam", "MobileNetV2", "L80_G95"): "task"}),
           out=buf)
    txt = buf.getvalue()
    if "UNVERIFIED" not in txt or "Do NOT run a sign test" not in txt:
        w("  FAIL  an un-md5'd campaign must read UNVERIFIED and disable the "
          "sign test" + chr(10))
        ok = False
    else:
        w("  PASS  an un-md5'd campaign reads UNVERIFIED -- the default is the "
          "cautious" + chr(10) + "        one, not two free replicates"
          + chr(10))

    w(chr(10) + "SELF-TEST %s%s" % ("PASSED" if ok else "FAILED", chr(10)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    ap.add_argument("--cells", help="a cell_table CSV")
    ap.add_argument("--out", help="write the paper rows here")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="emit rows for campaigns `scripts.quarantine` marked dead, or for arms a PARTIAL marker names")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.cells:
        ap.error("give --cells <cell_table.csv>, or --self-test")

    rows = load_cells(args.cells)

    # 🛑 THE QUARANTINE GATE, BY CAMPAIGN NAME. This tool reads a
    # `cell_table` CSV and never touches a campaign tree, so it has no path to
    # walk -- and until 2026-09-04 it was therefore ungated entirely, which
    # made THE tool that says what may be WRITTEN the one place a marker did
    # not reach. No fallback import: if the gate cannot load, this must break.
    from scripts.quarantine import by_name
    hard, partial = [], {}
    for camp in sorted({r.get("campaign") for r in rows if r.get("campaign")}):
        e = by_name(camp)
        if not e:
            continue
        if e.get("scorable") is False:
            hard.append((camp, e))
        elif e.get("dead_arms"):
            partial[camp] = set(e["dead_arms"])
    if hard and not args.allow_quarantined:
        for camp, e in hard:
            print("REFUSING to emit paper rows for %s" % camp)
            print("  reason   : %s" % e.get("reason"))
            print("  keep for : %s" % e.get("keep_for"))
        print("")
        print("These rows are in the input CSV but must not reach a paper.")
        print("Pass --allow-quarantined only to report them AS quarantined.")
        return 1
    dropped = 0
    # 🛑 THE FLAG GOVERNS THE HARD MARKER, NOT THIS DROP. It used to
    # read `if partial and not args.allow_quarantined`, so a user
    # passing it to report a hard-quarantined campaign AS quarantined
    # also, silently, re-admitted every PARTIAL campaign's dead-arm
    # rows -- in the one tool whose job is saying what may be WRITTEN.
    # The six path-based scorers cannot be overridden this way either:
    # `gate()` returns the dead arms regardless of `allow`. One rule.
    if partial:
        for camp, arms in sorted(partial.items()):
            print("!! PARTIAL QUARANTINE: %s -- dropping rows for %s"
                  % (camp, ", ".join(sorted(arms))))
            print("   %s" % (by_name(camp) or {}).get("reason"))
        before = len(rows)
        rows = [r for r in rows
                if r.get("arm") not in partial.get(r.get("campaign"), ())]
        dropped = before - len(rows)
        print("   dropped %d of %d row(s); every other contrast is untouched"
              % (dropped, before))
        print("")
        if not rows:
            print("nothing left to emit after dropping quarantined arms")
            return 1

    status, unit = {}, dict(MEASURED_UNITS)
    try:
        import yaml
        from configs.task_cells import classify, load_windows
        P = yaml.safe_load(io.open(os.path.join("configs", "protocol.yml"),
                                   encoding="utf-8"))
        TW = load_windows()
        for r in rows:
            k = (r["dataset"], r["model"], r["cap"])
            if k not in status:
                # classify() narrates every K=0 local budget it meets, and
                # iwildcam has 7 of 14. That is the right thing on a launch
                # gate and pure noise on a 234-row table, so swallow the
                # narration -- but NOT the exception.
                keep, sys.stdout = sys.stdout, io.StringIO()
                try:
                    status[k] = classify(P, TW, *k)["status"]
                except Exception:
                    status[k] = "no_data"
                finally:
                    sys.stdout = keep
    except Exception as exc:
        # NOT silent. Without the windows every row would read `?`, which is
        # indistinguishable from a measured non-task.
        print("  !! could not classify cells (%s): every row will read `?`, "
              "which is NOT the same as `task`." % exc)

    recs = build(rows, status_of=status, unit_of=unit)
    render(recs)
    if args.out:
        write(recs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
