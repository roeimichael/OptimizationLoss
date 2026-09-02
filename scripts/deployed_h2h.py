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
    """Macro cc-F1 on the deployed predictions. F1 = 2TP/(K+n) exactly, because
    the allocator emits exactly K per capped class."""
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
    """Median |arm - its own reseed twin| over every arm that has one.

    This is the noise an arm-vs-arm contrast actually faces on this design:
    `tralo` and `tralo_reseed` differ in the RNG stream and in nothing else, so
    whatever separates them is not a method.
    """
    gaps = []
    for arm, seedmap in cell.items():
        twin = reseed_of(arm)
        if not twin or twin not in cell:
            continue
        d, _ = paired(seedmap, cell[twin], get)
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
    n_named = n_refused = n_unstable = n_disagree = 0
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
                         base_items=base, rng_floor=floor, spread=spread,
                         refused=bool(verdict), jackknife=sorted(first_tp),
                         order=[dict(arm=a, d_items=m, seeds_needed=seeds_needed(d))
                                for a, m, d, _ in order_tp]))
    w("%s\n" % ("=" * 78))
    w("%d cells: #1 NAMED in %d, REFUSED in %d (spread under the RNG floor)\n"
      % (len(rows), n_named, n_refused))
    w("%d cells are JACKKNIFE-UNSTABLE (one dropped seed changes #1)\n" % n_unstable)
    w("%d cells have items and ccF1 disagreeing on the order\n" % n_disagree)
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
    live = _cell({"clip":         [600, 601, 599, 600],
                  "tralo":        [640, 641, 639, 640],
                  "alm":          [610, 611, 609, 610],
                  "tralo_reseed": [640, 641, 639, 640]})
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
                  "tralo_reseed": [600, 605, 603, 604]})
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

    # 7. reseed twins resolve per FAMILY.
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
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.campaign:
        ap.error("--campaign is required (or --self-test)")
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
