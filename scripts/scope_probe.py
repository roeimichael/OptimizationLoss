"""At a MATCHED TOTAL, does pinning the per-group split change the output?

WHY THIS EXISTS. FRAMEWORK 2(l) found that the local cap has never bound the
OUTPUT: `lp_fallback_used` is False with 0 candidates on every completed run,
because the allocator imposes the tighter GLOBAL total first and the split it
lands on has been inside every local ceiling every time. The proposed fix is a
campaign at `L20_G50`, where the local ceilings become the binding ones.

That campaign is ~120 runs and ~10 GPU-hours, so price it first.

THE MEASUREMENT IS EXACT AND NEEDS NO GPU. Per-group `L%` sums to `L%` of the
total, so `L20_G50` and `L50_G20` impose the SAME TOTAL BUDGET -- 41 for class 2
and 44 for class 4 on dermmnist -- and differ ONLY in whether the split across
groups is pinned (14/12/15 and 19/19/6) or free. That makes the scope contrast
answerable from stored probabilities alone:

    allocate the SAME probabilities under a free split and under a pinned one,
    at the same total, and read the endpoint.

WHAT A NULL HERE WOULD AND WOULD NOT CLOSE. This holds the MODEL fixed, so it
measures the allocator half only. Training under a binding local cap could in
principle produce different probabilities. But the standing finding is that the
constraint prunes and never re-ranks, so if pinning the split cannot move the
endpoint even with the probabilities held identical, the campaign's only
remaining route is "a local cap yields better probabilities than a global one at
the same total" -- and nothing in the corpus supports that. Read a null here as
a strong prior against the campaign, not as a proof.

TWO NUMBERS, NEVER ONE. A scope change can move many items and score the same,
which is a RE-ALLOCATION and not a difference (FRAMEWORK: a sub-item delta is a
re-allocation). So every row reports both `d items` (the endpoint, converted by
the scorer's own scale) and `moved`, the number of items whose assignment
actually changed. `moved = 0` means the pinned split is literally inert;
`moved > 0` with `d items ~ 0` means it reshuffles without consequence.

LIVENESS CONTROLS, both at the SAME total, because a null is only a measurement
if a wrong answer would have shown up:

  C1  ROTATED ceilings -- the same three numbers assigned to the wrong groups.
  C2  REVERSED ceilings -- which on class 4 puts the punitive ceiling (6) on a
      large group and a loose one (19) on the small dense group, the worst
      split of the right size.

If neither control moves the endpoint either, the probe cannot resolve anything
on this data and the run says so instead of reporting a null.

    python -m scripts.scope_probe --campaign results/dualbar2
    python -m scripts.scope_probe <run-dir> [<run-dir> ...] --tight L20_G50
"""
import argparse
import glob
import itertools
import os

import numpy as np

from scripts.frozen_head_probe import (allocate, budgets, cc_f1, items_per_001,
                                       load_real, paired, seeds_needed)
from src.utils.constants import UNLIMITED
from configs.gen_campaign import cap_pair

TIGHT_DEFAULT = "L20_G50"      # local binds, global slack
LOOSE_DEFAULT = "L50_G20"      # global binds, local slack -- same total


def _free_local(groups, n_classes):
    """Local ceilings that constrain nothing, so only the global acts."""
    return {int(g): [UNLIMITED] * n_classes for g in np.unique(groups)}


def _free_global(n_classes):
    return [UNLIMITED] * n_classes


def _permute_ceilings(L, classes, order):
    """Reassign each class's per-group ceilings to different groups.

    The multiset of ceilings -- and therefore the total budget -- is preserved
    exactly; only which group receives which ceiling changes. That is the
    control a null needs: a split of the right size but the wrong shape.
    """
    gids = sorted(L)
    out = {g: list(L[g]) for g in gids}
    for c in classes:
        vals = [L[g][c] for g in gids]
        for slot, g in enumerate(gids):
            out[g][c] = vals[order[slot]]
    return out


def score(run_dir, tight, loose):
    d = load_real(run_dir)
    y, g, classes, n = d.y, d.groups, d.classes, d.n_classes
    tight_l, tight_g = cap_pair(tight)
    loose_l, loose_g = cap_pair(loose)

    G_tight, L_tight = budgets(y, g, classes, tight_l, tight_g, n)
    G_loose, L_loose = budgets(y, g, classes, loose_l, loose_g, n)

    totals = {}
    for c in classes:
        summed = sum(L_tight[gg][c] for gg in L_tight)
        totals[c] = (G_loose[c], summed)
        if G_loose[c] != summed:
            raise SystemExit(
                "%s: class %d totals disagree -- global %d under %s vs local "
                "sum %d under %s. The contrast is only controlled when they "
                "match; see FRAMEWORK 2(l)."
                % (run_dir, c, G_loose[c], loose, summed, tight))

    gids = sorted(L_tight)
    rot = list(range(1, len(gids))) + [0]
    rev = list(range(len(gids)))[::-1]

    regimes = {
        # the total is pinned, the split is free -- what every campaign has run
        "global_only": (G_loose, _free_local(g, n)),
        # the same total, imposed as per-group ceilings -- never yet run
        "local_only": (_free_global(n), L_tight),
        "C1_rotated": (_free_global(n), _permute_ceilings(L_tight, classes, rot)),
        "C2_reversed": (_free_global(n), _permute_ceilings(L_tight, classes, rev)),
    }

    base_alloc = allocate(d.ref_probs, g, *regimes["global_only"], classes)
    base_f1 = cc_f1(y, base_alloc, classes)
    scale = items_per_001(y, base_alloc, classes)

    out = {"_base_ccF1": base_f1, "_items_per_001": scale, "_totals": totals}
    for name, (G, L) in regimes.items():
        alloc = allocate(d.ref_probs, g, G, L, classes)
        out[name] = (cc_f1(y, alloc, classes) - base_f1) / 0.01 * scale
        out[name + "__moved"] = int((alloc != base_alloc).sum())
    return out


def _splits(total, n_groups):
    """Every way to divide `total` across `n_groups`, as tuples."""
    if n_groups == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _splits(total - first, n_groups - 1):
            yield (first,) + rest


def oracle_split(d, classes, totals, rounds=3):
    """The BEST per-group split at the matched total, found with true labels.

    A HEADROOM measure, not a method: it reads `y` to score, so no allocator
    could achieve it without labels. Its purpose is to decompose the known
    1.9-9.9 item gap between `clip` and a perfect allocator into the part
    reachable by choosing the SPLIT and the part that needs re-ranking WITHIN
    groups. If the oracle split is worth ~0, then no local-cap method can win
    however it is trained, and the whole scope direction closes rather than
    just the one campaign.

    Coordinate ascent over classes: the allocator is joint across capped
    classes, so their splits interact and cannot be optimised in one
    independent pass. Ascent from the free split is monotone in the objective
    and stops when a full sweep improves nothing.
    """
    y, g, n = d.y, d.groups, d.n_classes
    gids = sorted(np.unique(g).tolist())
    P = d.ref_probs

    def endpoint(per_class):
        L = {int(gg): [UNLIMITED] * n for gg in gids}
        for c, split in per_class.items():
            for slot, gg in enumerate(gids):
                L[int(gg)][c] = split[slot]
        return cc_f1(y, allocate(P, g, _free_global(n), L, classes), classes)

    G_free = [UNLIMITED] * n
    for c in classes:
        G_free[c] = totals[c][0]
    base_alloc = allocate(P, g, G_free, _free_local(g, n), classes)
    base_f1 = cc_f1(y, base_alloc, classes)
    scale = items_per_001(y, base_alloc, classes)
    free = {c: tuple(int(((base_alloc == c) & (g == gg)).sum()) for gg in gids)
            for c in classes}

    cur = dict(free)
    best_f1 = endpoint(cur)
    for _ in range(rounds):
        improved = False
        for c in classes:
            for cand in _splits(totals[c][0], len(gids)):
                trial = dict(cur)
                trial[c] = cand
                f1 = endpoint(trial)
                if f1 > best_f1 + 1e-12:
                    best_f1, cur, improved = f1, trial, True
        if not improved:
            break
    return {"free_ccF1": base_f1, "oracle_ccF1": best_f1,
            "d_items": (best_f1 - base_f1) / 0.01 * scale,
            "free_split": free, "oracle_split": cur}


def group_calibrate(P, g, classes, targets, factor_map=None, group_key=None):
    """Rescale each class within each GROUP toward a known per-group prevalence.

    WHY THIS IS NOT THE PRIOR SHIFT FRAMEWORK 2(j) CLOSED. That measurement
    applied ONE multiplier to a class across the whole test set. A single
    positive multiplier is a monotone transform of that class's scores, so it
    cannot reorder any two items and top-K is invariant to it -- which is
    exactly what 2(j) found, and why a 1000x correction moved fewer items than
    an RNG reseed. A PER-GROUP multiplier is not monotone over the full set: it
    raises one group's items relative to another's, and reordering ACROSS groups
    is precisely what determines the split. 2(j) does not apply here, and the
    oracle-split measurement says the split is worth ~6 items.

    `targets[gg][c]` is the true count of class c in group gg. `group_key`
    permutes which group's items receive which correction, for the controls.
    """
    Q = np.array(P, dtype=float)
    gids = sorted(np.unique(g).tolist())
    key = dict(zip(gids, gids if group_key is None else group_key))
    for c in classes:
        for gg in gids:
            mask = g == gg
            n_g = int(mask.sum())
            if not n_g:
                continue
            model_prior = float(Q[mask, c].mean())
            src = key[gg]
            target = float(targets[src][c]) / float((g == src).sum())
            if factor_map is not None:
                factor = factor_map[gg][c]
            elif model_prior <= 0:
                continue
            else:
                factor = target / model_prior
            Q[mask, c] = Q[mask, c] * factor
    return Q / np.clip(Q.sum(axis=1, keepdims=True), 1e-12, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--campaign")
    ap.add_argument("--tight", default=TIGHT_DEFAULT,
                    help="cap tag whose LOCAL scope binds (default %s)"
                         % TIGHT_DEFAULT)
    ap.add_argument("--loose", default=LOOSE_DEFAULT,
                    help="cap tag whose GLOBAL scope binds at the SAME total "
                         "(default %s)" % LOOSE_DEFAULT)
    ap.add_argument("--oracle-split", action="store_true",
                    help="also search for the BEST split at the "
                         "matched total, using true labels. A "
                         "headroom measure, not a method.")
    ap.add_argument("--oracle-runs", type=int, default=8,
                    help="how many runs to search (exhaustive per "
                         "class, so keep it small)")
    ap.add_argument("--calibrate", action="store_true",
                    help="rescale scores per GROUP toward the known "
                         "per-group prevalence, then allocate with a "
                         "FREE split. A method candidate, not a "
                         "headroom measure.")
    args = ap.parse_args()

    runs = list(args.runs)
    if args.campaign:
        runs += [os.path.dirname(p) for p in sorted(glob.glob(
            os.path.join(args.campaign, "**", "final_predictions_raw.csv"),
            recursive=True))]
    if not runs:
        raise SystemExit("no run directories given")

    print("SCOPE PROBE -- %s (local binds) vs %s (global binds), %d run(s)"
          % (args.tight, args.loose, len(runs)))
    print("The MODEL is held fixed. Only the allocator's budget changes, and")
    print("the TOTAL is identical in both, so this isolates the SCOPE.\n")

    rows, names, totals = {}, [], None
    for r in runs:
        try:
            s = score(r, args.tight, args.loose)
        except SystemExit as exc:
            print("  skipped %s: %s" % (r, exc))
            continue
        totals = s.pop("_totals")
        names.append(os.path.relpath(r, args.campaign or os.path.dirname(r)))
        for key, v in s.items():
            rows.setdefault(key, []).append(v)

    if not names:
        raise SystemExit("no probeable runs")

    print("  matched totals per capped class (global under %s == local sum "
          "under %s):" % (args.loose, args.tight))
    for c, (gk, lk) in sorted(totals.items()):
        print("    class %d: %d == %d" % (c, gk, lk))

    keys = ("local_only", "C1_rotated", "C2_reversed")
    print("\n  PAIRED OVER %d RUN(S), each against its OWN free-split "
          "allocation at the same total" % len(names))
    print("  %-16s %9s %7s %7s %8s %9s %9s"
          % ("regime", "d items", "sd", "sem", "sign", "sign p", "moved"))
    stats = {}
    for key in keys:
        st = paired(np.asarray(rows[key]))
        stats[key] = st
        print("  %-16s %+9.2f %7.2f %7.2f %5d/%-2d %9.4f %9.1f"
              % (key, st["mean"], st["sd"], st["sem"], st["pos"], st["n"],
                 st["sign_p"], float(np.mean(rows[key + "__moved"]))))

    real = stats["local_only"]
    ctrl = max(abs(stats["C1_rotated"]["mean"]), abs(stats["C2_reversed"]["mean"]))
    moved = float(np.mean(rows["local_only__moved"]))
    print("")
    if ctrl < 1.0:
        print("  ⚠️  PROBE CANNOT RESOLVE THIS. Both wrong-shape controls moved the")
        print("     endpoint less than one item (max %.2f), so a null in the real" % ctrl)
        print("     row is silence, not a measurement. Do not price the campaign on it.")
    elif abs(real["mean"]) < 1.0:
        print("  → NULL, and it is a MEASUREMENT: the controls move %.2f items, so a"
              % ctrl)
        print("     real effect of that size would have shown. Pinning the split")
        print("     changes the endpoint by %+.2f items while re-assigning %.0f items."
              % (real["mean"], moved))
        print("     That is a RE-ALLOCATION, not a difference.")
    else:
        print("  → the pinned split moves the endpoint %+.2f items (%d/%d runs)."
              % (real["mean"], real["pos"], real["n"]))
        if np.isfinite(real["sd"]) and real["sd"] > 0 and real["mean"] > 0:
            print("     a GPU campaign would need ~%d seeds per cell to see it, "
                  "vs the standard 4" % seeds_needed(real["mean"], real["sd"]))
    print("\n  This holds the MODEL fixed -- it prices the allocator half only.")
    print("  See the module docstring for what a null does and does not close.")

    if args.calibrate:
        print("")
        print("  PER-GROUP CALIBRATION -- rescale each class within each group")
        print("  toward its known per-group prevalence, then allocate with a "
              "FREE")
        print("  split under the same total. Uses NO labels beyond the counts "
              "the")
        print("  cap already states. This is a METHOD candidate, so the "
              "controls")
        print("  decide it: a permuted assignment of the same factors must NOT "
              "help.")
        got = {"calibrated": [], "C1_permuted_targets": [],
               "C2_shuffled_groups": []}
        for r in runs[:args.oracle_runs]:
            try:
                d = load_real(r)
            except SystemExit:
                continue
            y, g, cls, n = d.y, d.groups, d.classes, d.n_classes
            ll, lg = cap_pair(args.loose)
            G_loose, _ = budgets(y, g, cls, ll, lg, n)
            G_free = [UNLIMITED] * n
            for c in cls:
                G_free[c] = G_loose[c]
            gids = sorted(np.unique(g).tolist())
            targets = {gg: [int(((y == c) & (g == gg)).sum())
                            for c in range(n)] for gg in gids}
            base = allocate(d.ref_probs, g, G_free, _free_local(g, n), cls)
            f0 = cc_f1(y, base, cls)
            scale = items_per_001(y, base, cls)
            rng = np.random.default_rng(0)
            rot = gids[1:] + gids[:1]
            variants = {
                "calibrated": group_calibrate(d.ref_probs, g, cls, targets),
                "C1_permuted_targets": group_calibrate(
                    d.ref_probs, g, cls, targets, group_key=rot),
                "C2_shuffled_groups": group_calibrate(
                    d.ref_probs, rng.permutation(g), cls, targets),
            }
            for name, Q in variants.items():
                a = allocate(Q, g, G_free, _free_local(g, n), cls)
                got[name].append((cc_f1(y, a, cls) - f0) / 0.01 * scale)
        print("")
        print("  %-22s %9s %7s %8s %9s"
              % ("variant", "d items", "sd", "sign", "sign p"))
        for name in ("calibrated", "C1_permuted_targets", "C2_shuffled_groups"):
            if not got[name]:
                continue
            st = paired(np.asarray(got[name]))
            print("  %-22s %+9.2f %7.2f %5d/%-2d %9.4f"
                  % (name, st["mean"], st["sd"], st["pos"], st["n"],
                     st["sign_p"]))

    if args.oracle_split:
        print("")
        print("  ORACLE SPLIT -- the best division of the SAME total across "
              "groups,")
        print("  found WITH true labels. Not achievable by any method; it "
              "bounds what")
        print("  choosing the split could ever be worth, so a small number "
              "here closes")
        print("  the direction itself and not merely one campaign.")
        gains, searched = [], {}
        for r in runs[:args.oracle_runs]:
            try:
                d = load_real(r)
            except SystemExit:
                continue
            tl, tg = cap_pair(args.tight)
            ll, lg = cap_pair(args.loose)
            G_loose, _ = budgets(d.y, d.groups, d.classes, ll, lg, d.n_classes)
            _, L_tight = budgets(d.y, d.groups, d.classes, tl, tg, d.n_classes)
            tot = {c: (G_loose[c], sum(L_tight[gg][c] for gg in L_tight))
                   for c in d.classes}
            o = oracle_split(d, d.classes, tot)
            searched[r] = {"d": d, "tot": tot, "o": o}
            gains.append(o["d_items"])
            print("    %-34s %+7.2f items  free %s -> oracle %s"
                  % (os.path.relpath(r, args.campaign
                                     or os.path.dirname(r))[-34:],
                     o["d_items"],
                     {c: list(v) for c, v in o["free_split"].items()},
                     {c: list(v) for c, v in o["oracle_split"].items()}))
        if gains:
            st = paired(np.asarray(gains))
            print("    %-34s %+7.2f items  sd %.2f  over %d run(s)"
                  % ("MEAN own-oracle", st["mean"], st["sd"], st["n"]))

        # THE TRANSFER IS NOT OPTIONAL. Choosing the best of ~900 splits on the
        # same 2014 labels that then score it is a selection, and with a seed sd
        # of ~2.7 items the selection alone buys several items. An own-oracle
        # number quoted without its transfer is a headroom claim built out of
        # noise, so this prints unconditionally beside it.
        xs = []
        for a, b in itertools.permutations(sorted(searched), 2):
            db = searched[b]["d"]
            y, g, cls, n = db.y, db.groups, db.classes, db.n_classes
            G_free = [UNLIMITED] * n
            for c in cls:
                G_free[c] = searched[b]["tot"][c][0]
            base = allocate(db.ref_probs, g, G_free, _free_local(g, n), cls)
            f0 = cc_f1(y, base, cls)
            scale = items_per_001(y, base, cls)
            gids = sorted(np.unique(g).tolist())
            L = {int(gg): [UNLIMITED] * n for gg in gids}
            for c, sp in searched[a]["o"]["oracle_split"].items():
                for slot, gg in enumerate(gids):
                    L[int(gg)][c] = sp[slot]
            f1 = cc_f1(y, allocate(db.ref_probs, g, [UNLIMITED] * n, L, cls), cls)
            xs.append((f1 - f0) / 0.01 * scale)
        if xs:
            stx = paired(np.asarray(xs))
            print("    %-34s %+7.2f items  sd %.2f  %d/%d positive"
                  % ("TRANSFERRED to a different run", stx["mean"], stx["sd"],
                     stx["pos"], stx["n"]))
            print("")
            if stx["mean"] <= 0 or stx["pos"] * 2 <= stx["n"]:
                print("    -> the own-oracle gain does NOT transfer. It is "
                      "SELECTION NOISE,")
                print("       not headroom, and no split-based method can "
                      "collect it.")
            else:
                print("    -> part of the gain survives transfer; that part is "
                      "real headroom.")


if __name__ == "__main__":
    main()
