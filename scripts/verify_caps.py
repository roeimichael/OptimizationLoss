"""Compute integer global and local budgets on actual dataset arrays."""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.gen_campaign import cap_pair, load_protocol
from src.training.constraints import (
    compute_global_constraints,
    compute_local_constraints,
)


def duplicate_budget_tags(eff_by_class):
    out = []
    for c, per_tag in sorted(eff_by_class.items()):
        same = {}
        for tag, eff in per_tag.items():
            same.setdefault(eff, []).append(tag)
        for eff, tags in sorted(same.items()):
            if len(tags) > 1:
                out.append((c, eff, sorted(tags)))
    return out


def load_test(dc):
    y = np.load(os.path.join(dc["data_dir"], "test_labels.npy")).ravel()
    meta = pd.read_csv(os.path.join(dc["data_dir"], "test_meta.csv"))
    # The training path factorises a non-integer group column via
    # `_encode_groups`; this read did a bare `.astype(np.int64)` and raised
    # "invalid literal for int(): 'anterior torso|40s'" on bcn, so the cap
    # verifier could not be run on the second dataset at all. Use the SAME
    # encoder, or the group ids here would not be the ids the budgets use.
    from src.utils.data_loader import _encode_groups

    groups = _encode_groups(meta[dc["group_column"]], dc["group_column"])
    return pd.DataFrame({"label": y, dc["group_column"]: groups})


def main():
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("--datasets", nargs="+", default=sorted(P["datasets"]))
    a.add_argument("--caps", nargs="+", default=["L30_G30", "L30_G50", "L50_G50"])
    a.add_argument("--constrained-class", nargs="+", type=int, default=None)
    a.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if any cap is inert/redundant or any group is uninformative, so this can gate a launch. Off by default: an inert global cap is a real fact about the campaign, not necessarily a mistake -- but it must never be a SILENT one.",
    )
    a.add_argument(
        "--campaign",
        help="a staged campaign root. Takes the datasets, cap tags and constrained "
             "classes FROM ITS CONFIGS instead of from the defaults. Without this "
             "the gate ran on `--caps L30_G30 L30_G50 L50_G50` across every "
             "protocol dataset -- caps no campaign has used since they were the "
             "defaults, so it verified a configuration nobody launches while "
             "failing on any dataset whose slice happens to be absent.",
    )
    args = a.parse_args()
    if args.campaign:
        import glob, json

        (ds_seen, cap_seen, cls_seen) = (set(), set(), set())
        for f in glob.glob(os.path.join(args.campaign, "*/*/*/*/seed_*/config.json")):
            cfg = json.load(open(f))
            ds_seen.add(cfg["dataset_mode"])
            cap_seen.add(cfg["constraint_tag"])
            cc = cfg["dataset_config"]["constrained_class"]
            cls_seen.add(tuple(cc) if isinstance(cc, list) else (cc,))
        if not ds_seen:
            raise SystemExit("REFUSED: no configs under %s" % args.campaign)
        if len(cls_seen) > 1:
            raise SystemExit(
                "REFUSED: campaign mixes constrained_class sets %s; verify each "
                "separately rather than checking one against the other's budgets"
                % sorted(cls_seen))
        args.datasets = sorted(ds_seen)
        args.caps = sorted(cap_seen)
        args.constrained_class = list(next(iter(cls_seen)))
        print("scoped to campaign %s: datasets %s, caps %s, classes %s"
              % (args.campaign, args.datasets, args.caps, args.constrained_class))
    (fails, inert) = ([], [])
    for ds in args.datasets:
        dc = dict(P["datasets"][ds])
        cls = (
            args.constrained_class
            if args.constrained_class is not None
            else dc["constrained_class"]
        )
        cls = cls if isinstance(cls, list) else [cls]
        (gcol, n_cls) = (dc["group_column"], dc["num_classes"])
        try:
            df = load_test(dc)
        except (OSError, KeyError) as e:
            print("%-12s FAIL -- could not read the slice: %s" % (ds, e))
            fails.append(
                "%s: slice unreadable (%s). This gate proves nothing about a dataset it never opened."
                % (ds, e)
            )
            continue
        n = len(df)
        counts = df["label"].value_counts().sort_index()
        print("=" * 78)
        print(
            "%s   %d test items, %d classes, %d groups (%s)"
            % (ds, n, n_cls, df[gcol].nunique(), gcol)
        )
        print(
            "  class counts: %s"
            % "  ".join(("%d:%d" % (c, counts.get(c, 0)) for c in range(n_cls)))
        )
        bad = [c for c in cls if not 0 <= c < n_cls]
        if bad:
            fails.append(
                "%s: class %s out of range for num_classes=%d" % (ds, bad, n_cls)
            )
            print("  FAIL: constrained_class %s out of range" % bad)
            continue
        print(
            "  constrained: %s  (%s of test)"
            % (cls, ", ".join(("%.1f%%" % (100.0 * counts.get(c, 0) / n) for c in cls)))
        )
        overall = np.array([counts.get(c, 0) for c in range(n_cls)], dtype=float) / n
        tvs = []
        for gid in sorted(df[gcol].unique()):
            sub = df[df[gcol] == gid]
            share = np.array(
                [(sub["label"] == c).sum() for c in range(n_cls)], dtype=float
            ) / max(1, len(sub))
            tvs.append(0.5 * np.abs(share - overall).sum())
        worst = max(tvs) if tvs else 0.0
        print(
            "  groups: %d, class-mix distance from the whole test set (total variation) %s"
            % (len(tvs), ["%.3f" % t for t in tvs])
        )
        if worst < 0.05:
            print(
                "     UNINFORMATIVE GROUPS: every group has the same class mix (max %.3f), so each local budget is essentially global/%d and the local scope adds no constraint the global one does not."
                % (worst, len(tvs))
            )
            inert.append(
                "%s: groups carry no class information (max TV %.3f)" % (ds, worst)
            )
        eff_by_class = {}
        for tag in args.caps:
            (local_pct, global_pct) = cap_pair(tag)
            try:
                gcon = compute_global_constraints(
                    df, "label", global_pct, constrained_class=cls, num_classes=n_cls
                )
                lcon = compute_local_constraints(
                    df,
                    "label",
                    local_pct,
                    gcol,
                    constrained_class=cls,
                    num_classes=n_cls,
                )
            except ValueError as e:
                fails.append("%s %s: %s" % (ds, tag, e))
                print("    %-9s FAIL -- %s" % (tag, e))
                continue
            for c in cls:
                K_g = gcon[c]
                per_group = sorted((v[c] for v in lcon.values()))
                eff = min(K_g, sum(per_group))
                eff_by_class.setdefault(c, {})[tag] = eff
                print(
                    "    %-9s class %d: global K=%-5d local K per group=%s sum=%d -> effective %d (%.1f%% of the %d true)"
                    % (
                        tag,
                        c,
                        K_g,
                        per_group,
                        sum(per_group),
                        eff,
                        100.0 * eff / max(1, counts.get(c, 0)),
                        counts.get(c, 0),
                    )
                )
                lsum = sum(per_group)
                if K_g > lsum:
                    print(
                        "              INERT GLOBAL: K=%d is above the local sum %d, so it can never bind -- this tag runs the same experiment as L%02d_G%02d."
                        % (K_g, lsum, int(local_pct * 100), int(local_pct * 100))
                    )
                    inert.append(
                        "%s %s class %d: global (slack by %d)"
                        % (ds, tag, c, K_g - lsum)
                    )
                elif K_g == lsum:
                    print(
                        "              REDUNDANT GLOBAL: K=%d equals the local sum, so it binds only when every group is already at its own cap -- it adds no constraint of its own."
                        % K_g
                    )
                    inert.append("%s %s class %d: global (redundant)" % (ds, tag, c))
                true_by_group = df[df["label"] == c].groupby(gcol).size()
                zero_k = sorted((g for g in lcon if lcon[g][c] == 0))
                slack = [
                    g
                    for g in lcon
                    if lcon[g][c] > 0 and lcon[g][c] >= int(true_by_group.get(g, 0))
                ]
                if zero_k:
                    print(
                        "              K=0 on group(s) %s -- no true instance of the class there, so the budget is zero. Real and binding; verify the loss is driving it to zero."
                        % zero_k
                    )
                if len(slack) + len(zero_k) == len(lcon) and (not zero_k):
                    print(
                        "              INERT LOCAL: every group cap >= that group's true count, so no local cap can bind"
                    )
                    inert.append("%s %s class %d: local" % (ds, tag, c))
        for c, eff, tags in duplicate_budget_tags(eff_by_class):
            print(
                "    *** SAME BUDGET, DIFFERENT TAGS: class %d gets K=%d under %s."
                % (c, eff, " and ".join(tags))
            )
            print(
                "        Those are ONE cap level, not %d. The binding scope is the same in each, so any per-cell count over them double-counts a single measurement."
                % len(tags)
            )
            inert.append(
                "%s class %d: %s all give K=%d -- one cap level"
                % (ds, c, "/".join(tags), eff)
            )
        print()
    print("=" * 78)
    if fails:
        print("CAP CHECK FAILED:")
        for f in fails:
            print("  - %s" % f)
        return 1
    print(
        "CAP CHECK OK -- every cap tag produces a real integer budget on every dataset."
    )
    if inert:
        print()
        print(
            "%d INERT cap(s) -- these bind nothing, so the arm runs unconstrained"
            % len(inert)
        )
        print("on that scope and the tag overstates what was tested:")
        for i in inert:
            print("  - %s" % i)
        print("A global cap only adds a constraint when it is strictly BELOW the")
        print("sum of the local caps, so G>=L never does. Sweep G<L to make the")
        print("global scope the thing under test.")
        if args.strict:
            print()
            print(
                "--strict: failing because the campaign would not exercise what its tags claim."
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
