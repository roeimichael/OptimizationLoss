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
MEASURED_UNITS = {
    ("dom1", "MobileNetV2"): "A1",          # a / dsisco02
    ("loose1", "MobileNetV2"): "A1",        # a -- md5-identical to dom1, 4/4
    ("equaldose1", "MobileNetV2"): "A2",    # B / dsisco01
    ("dom1b", "RegNetY400MF"): "B1",        # B / dsisco01
    ("loose1", "RegNetY400MF"): "B2",       # a / dsisco02
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


def null_of(arm):
    """The lambda=0 twin an arm must be attributed against.

    NOT a fixed `tralo_null`: that is right for `tralo`, `tralo_cut`,
    `tralo_uniform` and `tralo_head`, which share one twin, and quietly WRONG
    for `alm`/`fioretto`/`hounie` on a cross-family campaign, where the twin
    actually run is `<family>_null`. Returning the wrong twin silently
    attributes one arm's effect to another's model.
    """
    if arm.endswith(("_null", "_reseed", "_lam0")):
        return None
    for fam in ("tralo", "alm", "fioretto", "hounie"):
        if arm == fam or arm.startswith(fam + "_"):
            return fam + "_null"
    return None


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
                ref = fixed if fixed is not None else null_of(arm)
                if not ref or ref not in arms:
                    continue
                d = (float(r["ccF1"]) - float(arms[ref]["ccF1"])) * scale
                # sqrt(sa^2 + sb^2) is the sd of the per-seed DIFFERENCE if the
                # two arms were independent. They are not, and NOT in the
                # direction that helps: `tralo` and its null share ONE warm-up
                # epoch then train 29 apart, so they are two MODELS, and the
                # measured treated sd runs 6-12x the unpaired one (FRAMEWORK
                # 2(v): 0.80 unpaired vs 7.59 treated on iwc3).
                # => this sd is a LOWER BOUND on the noise the contrast faces,
                #    and `seeds_needed` is a LOWER BOUND on the seeds needed.
                # Get the real one from `scripts.paired_noise` on the campaign.
                sa = float(r["ccF1_sd"] or 0.0) * scale
                sb = float(arms[ref]["ccF1_sd"] or 0.0) * scale
                sd = math.sqrt(sa * sa + sb * sb)
                out.append(dict(
                    campaign=camp, dataset=ds, model=model, cap=cap, arm=arm,
                    contrast=name, ref=ref,
                    n_seeds=int(r["n_seeds"]),
                    items=d, sd_items=sd,
                    seeds_needed=seeds_needed(d, sd),
                    resolved=("yes" if (sd and abs(d) >= 2.0 * sd) else "no"),
                    cell_status=(status_of or {}).get((ds, model, cap), "?"),
                    unit=(unit_of or {}).get(
                        (camp, model), "UNVERIFIED:" + camp + "/" + model),
                ))
    return out


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
    w("  `items` = d ccF1 * (K+n)/2. The WHOLE gap from `clip` to a PERFECT "
      "allocator" + chr(10))
    w("  is 1.9-9.9 items, so a sub-item delta is a re-allocation, not a "
      "difference." + chr(10))
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
    non = [r for r in recs if r["cell_status"] in ("non_task", "unmeasured")]
    if non:
        w(chr(10) + "  *** %d row(s) sit in a cell that poses NO measured "
          "question." % len(non) + chr(10))
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

    # 1. the twin resolver must be per FAMILY, not a fixed tralo_null
    cases = [("tralo", "tralo_null"), ("tralo_cut", "tralo_null"),
             ("alm", "alm_null"), ("fioretto", "fioretto_null"),
             ("hounie", "hounie_null"), ("clip", None), ("tralo_null", None)]
    bad = [(a, null_of(a), e) for a, e in cases if null_of(a) != e]
    if bad:
        w("  FAIL  null_of: %s%s" % (bad, chr(10)))
        ok = False
    else:
        w("  PASS  the lambda=0 twin resolves per FAMILY -- a fixed "
          "`tralo_null` would" + chr(10) + "        attribute alm's effect to "
          "tralo's model" + chr(10))

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
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        sys.exit(self_test())
    if not args.cells:
        ap.error("give --cells <cell_table.csv>, or --self-test")

    rows = load_cells(args.cells)
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
