"""AS-DEPLOYED head-to-head between the duals, in EXACT captured items.

WHY THIS EXISTS, and it is not a duplicate of `full_panel`.

`full_panel` scores the budget-equalized family on `eq` -- its OWN re-derived
equal-budget allocation, rebuilt from `final_predictions_raw.csv`. That is the
right instrument for "whose learned RANKING is better", and it is deliberately
allocator-blind. It is the WRONG instrument for "which arm would you deploy",
and the two disagree in RANK ORDER, not just in magnitude.

Measured 2026-09-02 on dom1 / MobileNetV2 / L80_G95:

    arm        panel d capF1 -> items     as-deployed captured TP (4 seeds)
    tralo          +0.00582     +5.77                  2602
    alm            +0.00617     +5.49                  2602

Identical items captured. The panel orders them anyway, and the ordering comes
entirely from cc-F1 being MACRO-averaged over two classes whose (K+n) differ
(666 vs 820): `alm` trades 5 items from class 7 into class 2, and class 2's
smaller denominator makes those 5 items worth more F1. Nothing was gained.

So this tool reports both scales side by side and says when they disagree.

THE SECOND REASON, and it is the one that changes the conclusion. An arm-vs-arm
delta is only a result if it exceeds the noise the contrast actually faces, and
on this design that noise is NOT the seed sd -- it is `<family>_reseed`, the
same arm with the RNG stream perturbed and nothing else. Measured over the
clean corpus, 17 task cells:

    |tralo - rival |  median 4.0 items   (n=180 paired seed-comparisons)
    |tralo - reseed|  median 4.0 items   (n=70)   <- SAME ARM, different RNG

Ratio 1.00x. So a #1 arm named off a 4-item lead is naming the RNG, and 11 of
17 cells duly change their #1 when a single seed is dropped. This tool REFUSES
to name a winner in that situation rather than printing one.

Usage:
    python -m scripts.deployed_h2h --campaign results/dom1 results/dom1b
    python -m scripts.deployed_h2h --campaign <roots> --control clip --json out.json
    python -m scripts.deployed_h2h --self-test
"""

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

# The recipe boundary. A campaign outside it is a DIFFERENT METHOD and pooling
# it silently is how the corpus got five TraLO configurations. Post-hoc arms
# take no constraint step, so they carry neither key and are exempt.
CURRENT_RECIPE = {"constraint_fp32": True, "constraint_grad_mode": "normalize"}

DUALS = ("tralo", "alm", "fioretto", "hounie")
FAMILIES = ("tralo", "alm", "fioretto", "hounie")

# 7.85 = (z_{0.975} + z_{0.80})^2, the paired-t sample-size constant at 80%
# power / alpha 0.05. Same constant as `paper_rows.seeds_needed`; kept local so
# the two tools cannot drift apart silently.
POWER_CONST = 7.85


def reseed_of(arm):
    """The RNG-only twin of `arm`, resolved per FAMILY.

    `alm`'s noise floor is `alm_reseed` when one exists, never `tralo_reseed` --
    attributing one family's RNG spread to another is the same class of error
    as `paper_rows.null_of` was written to prevent.
    """
    if arm.endswith(("_null", "_reseed", "_lam0")):
        return None
    for fam in FAMILIES:
        if arm == fam or arm.startswith(fam + "_"):
            return fam + "_reseed"
    return None


def on_recipe(cfg):
    """True if this run is on the current recipe (or is a post-hoc arm)."""
    hp = cfg.get("hyperparams") or {}
    if int(hp.get("constraint_epochs") or 0) <= 0:
        return True
    return (hp.get("constraint_fp32") is CURRENT_RECIPE["constraint_fp32"]
            and hp.get("constraint_grad_mode")
            == CURRENT_RECIPE["constraint_grad_mode"])


def capped_classes(cfg):
    cls = (cfg.get("dataset_config") or {}).get("constrained_class")
    if isinstance(cls, int):
        cls = [cls]
    return tuple(sorted(cls or []))


def read_run(run_dir):
    """(TP, K, n) per capped class from the AS-DEPLOYED predictions.

    Returns None when the run is unfinished, off-recipe, or has no capped class.
    """
    fin = os.path.join(run_dir, "final_predictions.csv")
    cj = os.path.join(run_dir, "config.json")
    if not (os.path.exists(fin) and os.path.exists(cj)):
        return None
    try:
        cfg = json.load(open(cj))
    except Exception:
        return None
    if not on_recipe(cfg):
        return None
    classes = capped_classes(cfg)
    if not classes:
        return None
    import pandas as pd
    df = pd.read_csv(fin)
    if "Predicted_Label" not in df.columns or "True_Label" not in df.columns:
        return None
    p, y = df["Predicted_Label"], df["True_Label"]
    per = {}
    for c in classes:
        per[c] = dict(TP=int(((p == c) & (y == c)).sum()),
                      K=int((p == c).sum()),
                      n=int((y == c).sum()))
    return dict(cfg=cfg, classes=classes, per=per,
                TP=sum(per[c]["TP"] for c in classes))


def ccf1(per, classes):
    """Macro cc-F1 on the deployed predictions.

    `K` here is what the arm actually EMITTED, not the budget it was given.
    `F1 = 2TP/(K+n)` is then exact per class by construction -- but two arms
    with different emitted counts are being scored on different denominators,
    which is a comparison at unequal SPEND. `spend_audit` is what catches that;
    this function cannot, and must not be read as if it did.
    """
    vals = []
    for c in classes:
        d = per[c]
        den = d["K"] + d["n"]
        vals.append(2.0 * d["TP"] / den if den else float("nan"))
    return sum(vals) / len(vals) if vals else float("nan")


def collect(roots):
    """cell key -> arm -> seed -> record."""
    cells = {}
    for root in roots:
        for fin in sorted(glob.glob(os.path.join(
                root, "*", "*", "*", "*", "seed_*", "final_predictions.csv"))):
            run = os.path.dirname(fin)
            rec = read_run(run)
            if rec is None:
                continue
            cfg = rec["cfg"]
            key = (os.path.basename(root.rstrip(os.sep)),
                   cfg.get("model_name"), cfg.get("dataset_mode"),
                   cfg.get("constraint_tag"),
                   "-".join(str(c) for c in rec["classes"]))
            arm = cfg.get("arm") or os.path.basename(os.path.dirname(run))
            seed = (cfg.get("hyperparams") or {}).get("seed")
            cells.setdefault(key, {}).setdefault(arm, {})[seed] = rec
    return cells


def paired(a_map, b_map, get):
    """Per-seed paired differences over the seeds BOTH arms have."""
    seeds = sorted(set(a_map) & set(b_map))
    return [get(a_map[s]) - get(b_map[s]) for s in seeds], seeds


def spend_audit(cell, classes):
    """Arms in this cell that did not emit the SAME number of predictions.

    Every arm in a cell faces the same integer budget `K_eff`, so an emitted
    count differing between arms at the SAME seed is unequal SPEND, not
    allocator quality. The budget `K = round(f * n_true)` is label-informed
    side information the problem grants every arm equally; an arm that leaves
    slots unfilled has declined part of it, and `2TP/(K+n)` then rewards it
    with a smaller denominator for the items it forfeited.

    Measured cause, FRAMEWORK 2(z31)d: `danits_lp` and the imbalanced arms set
    `skip_targeted_correction`, so they bypass `targeted_correction(
    force_exact=True)` -- which exists precisely to make cross-method
    comparisons apples-to-apples -- and the LP then correctly declines the last
    slots, because filling them raises expected 0-1 cost.

    This needs NO budget re-derivation and no labels: the comparison is
    arm-vs-arm at a fixed seed, so it is exact wherever two arms are present.

    Returns [(cls, seed, {arm: emitted}, spread)], worst spread first.
    """
    out = []
    seeds = sorted({s for m in cell.values() for s in m},
                   key=lambda x: (x is None, x))
    for c in classes:
        for s in seeds:
            got = {a: m[s]["per"][c]["K"] for a, m in cell.items()
                   if s in m and c in (m[s].get("per") or {})}
            if len(got) < 2:
                continue
            spread = max(got.values()) - min(got.values())
            if spread:
                out.append((c, s, got, spread))
    out.sort(key=lambda t: -t[3])
    return out


def seeds_needed(diffs):
    """Seeds per cell for this paired difference at 80% power."""
    if len(diffs) < 2:
        return None
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    if not m or not sd:
        return None
    return int(math.ceil(POWER_CONST * (sd / abs(m)) ** 2))


def rng_floor(cell, get):
    """Median |NULL - its own reseed twin|, per family. The RNG-only noise.

    🛑 CORRECTED 2026-09-02, and the old version was measuring the treatment.
    It paired `tralo` with `tralo_reseed` and asserted they "differ in the RNG
    stream and in nothing else". They do not. `configs/protocol.yml` builds
    `tralo_reseed` from blocks `[constraint_phase, tralo_null, tralo_reseed]`,
    so it INHERITS `tralo_null`'s `lambda_step: 0.0` and adds only
    `rng_reseed: True`. `tralo_reseed` is a lambda=0 arm. So
    `|tralo - tralo_reseed|` is a TREATED-vs-UNTREATED contrast carrying the
    very effect the floor is supposed to be measured against.

    The RNG-only pair is `fam_null` vs `fam_reseed` -- both lambda=0, differing
    in the RNG stream and nothing else. That is what this now computes.

    ⚠️ THE DIRECTION OF THE ERROR IS THE OPPOSITE OF THE OBVIOUS ONE, so do not
    "fix" it back. Measured on dom1 over 24 paired cell-seeds in captured TP
    items: the old contaminated floor is median **4.0**, the true RNG-only floor
    is median **6.5**, and the treatment itself is median +7.5 (19/24 positive).
    The contaminated floor was too LOW, not too high, so the tool was
    REFUSING TOO SELDOM -- it named a #1 in cells where the true noise already
    covered the spread. Correcting it makes this tool MORE conservative and
    makes "the head-to-head is inside the noise" a stronger statement, not a
    weaker one.
    """
    gaps = []
    for fam in FAMILIES:
        a, b = fam + "_null", fam + "_reseed"
        if a in cell and b in cell:
            d, _ = paired(cell[a], cell[b], get)
            gaps += [abs(x) for x in d]
    return (st.median(gaps), len(gaps)) if gaps else (None, 0)


def rank_cell(cell, control, get, arms=DUALS):
    """(ordered [(arm, mean delta vs control)], jackknife #1 set)."""
    if control not in cell:
        return [], set()
    ctrl = cell[control]
    out = []
    for a in arms:
        if a not in cell:
            continue
        d, seeds = paired(cell[a], ctrl, get)
        if d:
            out.append((a, st.mean(d), d, seeds))
    out.sort(key=lambda t: -t[1])
    # jackknife: drop one seed at a time and see whether #1 survives
    firsts = set()
    if out:
        allseeds = sorted(set.intersection(*[set(t[3]) for t in out]))
        for drop in allseeds:
            keep = [s for s in allseeds if s != drop]
            if not keep:
                continue
            sub = {a: st.mean(get(cell[a][s]) - get(ctrl[s]) for s in keep)
                   for a, _, _, _ in out}
            firsts.add(max(sub, key=lambda a: sub[a]))
    return out, firsts


def report(cells, control, w=sys.stdout.write):
    """Print one block per cell. Returns the machine-readable rows."""
    rows = []
    n_named = n_refused = n_unstable = n_disagree = n_unequal = 0
    for key in sorted(cells):
        cell = cells[key]
        root, model, ds, cap, capped = key
        classes = tuple(int(x) for x in capped.split("-") if x != "")
        g_tp = lambda r: float(r["TP"])
        g_f1 = lambda r: ccf1(r["per"], classes)

        if control not in cell:
            continue
        order_tp, first_tp = rank_cell(cell, control, g_tp)
        order_f1, first_f1 = rank_cell(cell, control, g_f1)
        if not order_tp:
            continue
        floor, nfloor = rng_floor(cell, g_tp)
        base = st.mean(g_tp(r) for r in cell[control].values())
        nseed = len(order_tp[0][3])

        w("%s\n" % ("-" * 78))
        w("%s / %s / %s / %s   capped %s   %d seeds\n"
          % (root, model, ds, cap, capped, nseed))
        w("  control `%s` captures %.1f items\n" % (control, base))

        unequal = spend_audit(cell, classes)
        if unequal:
            n_unequal += 1
            short = {}
            for c, sd, got, spr in unequal:
                full = max(got.values())
                for a, k in got.items():
                    if k < full:
                        short[a] = max(short.get(a, 0), full - k)
            w("  !! UNEQUAL SPEND: %d (class, seed) pairs where the arms\n"
              % len(unequal))
            w("     emit DIFFERENT counts. Worst: class %d seed %s, "
              "spread %d slots.\n"
              % (unequal[0][0], unequal[0][1], unequal[0][3]))
            w("     under-spending arms: %s\n"
              % ", ".join("%s -%d" % (a, d)
                          for a, d in sorted(short.items(),
                                             key=lambda t: -t[1])))
            w("     `2TP/(K+n)` rewards the smaller denominator, so part\n")
            w("     of any gap involving these arms is BUDGET, not\n")
            w("     allocator quality. FRAMEWORK 2(z31)d.\n")

        w("  %-10s %10s %10s %8s\n"
          % ("arm", "d items", "d ccF1", "seeds@80%"))
        f1_rank = [a for a, _, _, _ in order_f1]
        for a, m, d, _ in order_tp:
            f1m = next((x[1] for x in order_f1 if x[0] == a), float("nan"))
            sn = seeds_needed(d)
            w("  %-10s %+10.2f %+10.5f %8s\n"
              % (a, m, f1m, "-" if sn is None else sn))
        tp_rank = [a for a, _, _, _ in order_tp]
        if tp_rank != f1_rank:
            n_disagree += 1
            w("  !! ITEMS AND ccF1 DISAGREE ON THE ORDER\n")
            w("     items: %s\n" % " > ".join(tp_rank))
            w("     ccF1 : %s\n" % " > ".join(f1_rank))
            w("     cc-F1 is MACRO-averaged over classes with different (K+n),\n")
            w("     so trading an item between them moves it with NO item won.\n")

        spread = order_tp[0][1] - order_tp[-1][1] if len(order_tp) > 1 else 0.0
        verdict = None
        if floor is None:
            verdict = "NO FLOOR: no `_reseed` twin in this cell, so the spread is unpriced"
        elif len(order_tp) < 2:
            verdict = "ONE ARM: nothing to rank"
        elif spread <= floor:
            verdict = ("REFUSED: spread %.1f items <= RNG floor %.1f (n=%d). "
                       "Naming a #1 here names the RNG." % (spread, floor, nfloor))
        if verdict:
            n_refused += 1
            w("  #1: %s\n" % verdict)
        else:
            n_named += 1
            w("  #1: %s   (spread %.1f items > RNG floor %.1f)\n"
              % (order_tp[0][0], spread, floor))
        if len(first_tp) > 1:
            n_unstable += 1
            w("  !! JACKKNIFE UNSTABLE: dropping ONE seed makes #1 any of {%s}\n"
              % ", ".join(sorted(first_tp)))
        rows.append(dict(campaign=root, model=model, dataset=ds, cap=cap,
                         capped=capped, seeds=nseed, control=control,
                         unequal_spend=[dict(cls=c, seed=sd, emitted=got,
                                             spread=spr)
                                        for c, sd, got, spr in unequal],
                         base_items=base, rng_floor=floor, spread=spread,
                         refused=bool(verdict), jackknife=sorted(first_tp),
                         order=[dict(arm=a, d_items=m, seeds_needed=seeds_needed(d))
                                for a, m, d, _ in order_tp]))
    w("%s\n" % ("=" * 78))
    w("%d cells: #1 NAMED in %d, REFUSED in %d (spread under the RNG floor)\n"
      % (len(rows), n_named, n_refused))
    w("%d cells are JACKKNIFE-UNSTABLE (one dropped seed changes #1)\n" % n_unstable)
    w("%d cells have items and ccF1 disagreeing on the order\n" % n_disagree)
    w("%d cells compare arms at UNEQUAL SPEND -- see the !! blocks\n"
      % n_unequal)
    if n_named:
        tally = {}
        for r in rows:
            if not r["refused"]:
                tally[r["order"][0]["arm"]] = tally.get(r["order"][0]["arm"], 0) + 1
        w("of the %d named: %s\n"
          % (n_named, "  ".join("%s %d" % (a, n)
                                for a, n in sorted(tally.items(), key=lambda t: -t[1]))))
    return rows


# --------------------------------------------------------------------------
# self-test: the tool must NAME a real separation and REFUSE a fake one.

def _cell(spec, K=300, n=370):
    """spec: arm -> list of TP per seed. Builds a one-class cell."""
    out = {}
    for arm, tps in spec.items():
        out[arm] = {i + 1: dict(TP=float(t), classes=(2,),
                                per={2: dict(TP=t, K=K, n=n)})
                    for i, t in enumerate(tps)}
    return out


def self_test(w=sys.stdout.write):
    ok = True

    def check(good, label):
        nonlocal ok
        w("  %-4s %s\n" % ("PASS" if good else "FAIL", label))
        ok = ok and good

    g = lambda r: float(r["TP"])

    # 1. a REAL separation, with a tight RNG floor, must be NAMED.
    # The floor is now |null - reseed|, both lambda=0. A TIGHT floor means
    # those two agree; the treated arm's own level is irrelevant to it.
    live = _cell({"clip":         [600, 601, 599, 600],
                  "tralo":        [640, 641, 639, 640],
                  "alm":          [610, 611, 609, 610],
                  "tralo_null":   [600, 601, 599, 600],
                  "tralo_reseed": [600, 601, 599, 600]})
    order, first = rank_cell(live, "clip", g)
    floor, _ = rng_floor(live, g)
    spread = order[0][1] - order[-1][1]
    check(order[0][0] == "tralo" and spread > floor,
          "a 30-item lead over a 0-item RNG floor is NAMED (tralo)")
    check(len(first) == 1, "  and it survives the jackknife")

    # 2. the REAL corpus situation: lead == floor. Must be REFUSED.
    dead = _cell({"clip":         [600, 601, 599, 600],
                  "tralo":        [604, 606, 601, 605],
                  "alm":          [605, 602, 606, 601],
                  "tralo_null":   [600, 601, 599, 600],
                  "tralo_reseed": [604, 597, 603, 596]})
    order, first = rank_cell(dead, "clip", g)
    floor, nf = rng_floor(dead, g)
    spread = order[0][1] - order[-1][1]
    check(spread <= floor,
          "a lead the size of the RNG floor is REFUSED (%.1f vs %.1f, n=%d)"
          % (spread, floor, nf))

    # 3. NEGATIVE CONTROL on the floor itself: remove the reseed twin and the
    #    tool must say the spread is UNPRICED, never fall back to naming one.
    noflow = {k: v for k, v in dead.items() if k != "tralo_reseed"}
    fl, _ = rng_floor(noflow, g)
    check(fl is None, "with no `_reseed` twin the floor is None, not 0")

    # 4. the jackknife must FIRE on a cell decided by one seed.
    # tralo leads on the full set (+4.75 vs +4.00) but ONLY because of seed 1.
    fragile = _cell({"clip":  [600, 600, 600, 600],
                     "tralo": [616, 601, 601, 601],
                     "alm":   [604, 604, 604, 604]})
    _, first = rank_cell(fragile, "clip", g)
    check(len(first) > 1,
          "a #1 held up by ONE seed is flagged jackknife-unstable {%s}"
          % ", ".join(sorted(first)))

    # 5. the items-vs-ccF1 disagreement that started this: equal items, split
    #    differently across two classes with different (K+n), must reorder.
    two = {"clip":  {1: dict(TP=1000., per={2: dict(TP=500, K=296, n=370),
                                            7: dict(TP=500, K=364, n=456)})},
           "tralo": {1: dict(TP=1010., per={2: dict(TP=505, K=296, n=370),
                                            7: dict(TP=505, K=364, n=456)})},
           "alm":   {1: dict(TP=1010., per={2: dict(TP=515, K=296, n=370),
                                            7: dict(TP=495, K=364, n=456)})}}
    f1 = lambda r: ccf1(r["per"], (2, 7))
    o_tp, _ = rank_cell(two, "clip", lambda r: float(r["TP"]))
    o_f1, _ = rank_cell(two, "clip", f1)
    tie = abs(o_tp[0][1] - o_tp[1][1]) < 1e-9
    check(tie and o_f1[0][0] == "alm",
          "equal items but a class-2-heavy split ranks HIGHER on ccF1 (alm)")

    # 6. the recipe gate, both directions.
    check(on_recipe({"hyperparams": {"constraint_epochs": 29,
                                     "constraint_fp32": True,
                                     "constraint_grad_mode": "normalize"}}),
          "the current recipe is accepted")
    check(not on_recipe({"hyperparams": {"constraint_epochs": 29,
                                         "constraint_fp32": True,
                                         "constraint_grad_mode": "clip"}}),
          "grad_mode=clip is refused -- a different method")
    check(on_recipe({"hyperparams": {"constraint_epochs": 0}}),
          "a post-hoc arm is EXEMPT, not a violation")

    # 7. UNEQUAL SPEND, both directions. The real shape: on
    #    dom1/MobileNetV2/L90_G95/seed_1 the LP emitted 319 for class 2 while
    #    every force_exact arm emitted 333, and no gate anywhere looked, because
    #    `verify_allocation` and the eval-time raise both test `count > limit`.
    equal = {"clip":  {1: dict(TP=300., per={2: dict(TP=300, K=333, n=370)})},
             "tralo": {1: dict(TP=305., per={2: dict(TP=305, K=333, n=370)})}}
    check(not spend_audit(equal, (2,)),
          "arms emitting the SAME count are not flagged")

    short = {"clip": {1: dict(TP=319., per={2: dict(TP=319, K=333, n=370)})},
             "lp":   {1: dict(TP=307., per={2: dict(TP=307, K=319, n=370)})}}
    flag = spend_audit(short, (2,))
    check(len(flag) == 1 and flag[0][3] == 14,
          "an arm emitting 14 fewer slots is flagged with the right spread")

    # and the reason it matters: the under-spender's SMALLER denominator hands
    # it credit it did not earn. 2*307/(319+370) = 0.8912 against
    # 2*319/(333+370) = 0.9075 -- so the raw ccF1 gap UNDERSTATES the 12 items
    # the LP actually forfeited, and on a closer cell the sign can flip.
    f1_lp = ccf1({2: dict(TP=307, K=319, n=370)}, (2,))
    f1_eq = ccf1({2: dict(TP=307, K=333, n=370)}, (2,))
    check(f1_lp > f1_eq,
          "under-spending RAISES ccF1 at fixed TP (%.4f > %.4f), which is why "
          "the audit cannot be left to the metric" % (f1_lp, f1_eq))

    # 8. reseed twins resolve per FAMILY.
    check(reseed_of("alm") == "alm_reseed" and reseed_of("tralo_cut") == "tralo_reseed"
          and reseed_of("tralo_reseed") is None,
          "reseed twin resolves per family, and a twin has no twin")

    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", nargs="+", default=[],
                    help="campaign root(s), e.g. results/dom1")
    ap.add_argument("--control", default="clip",
                    help="the quality bar every arm is measured against")
    ap.add_argument("--json", default=None, help="write the rows here")
    ap.add_argument("--allow-quarantined", action="store_true",
                    help="compare arms in a campaign `scripts.quarantine` marked dead")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    # 🛑 THE QUARANTINE GATE. Audited 2026-09-04: this tool had NONE,
    # so a marker on a dead campaign prevented nothing here. No fallback
    # import -- if the gate cannot load, the tool must break.
    from scripts.quarantine import gate
    if not args.campaign:
        ap.error("--campaign is required (or --self-test)")
    blocked, dead = gate(args.campaign, args.allow_quarantined, "compare")
    if blocked:
        return 1
    cells = collect(args.campaign)
    if not cells:
        print("no runs on the current recipe under %s" % " ".join(args.campaign))
        return 1
    rows = report(cells, args.control)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1, default=str)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
