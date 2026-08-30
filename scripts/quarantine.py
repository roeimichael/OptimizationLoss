"""Make dead results READ as dead, and refuse to score them.

WHY THIS EXISTS AND WHY IT IS NOT `rm -rf`. Disk is not the problem here: the
NFS home is 31% used with 588T free, and `results/` is gitignored, so anything
deleted is unrecoverable. The problem is ANALYTICAL. A campaign that died,
or that ran at a fraction of its dose, or that targets a dataset which no
longer exists, looks from the outside exactly like one that is merely
unfinished -- `status: pending` on every remaining run. This project has been
bitten by that specific confusion more than once ("a dead arm reads as
pending"), and the cost is not wasted disk, it is a number quoted from a
campaign that should never have been scored.

So the rule here is:

  * NEVER delete a `completed` run. Completed runs are receipts. `iwc2` is the
    only evidence that ViTB16 under fp16 without `--constraint-fp32` loses
    25.4% of its dose; `selectrun` is the only evidence that the selection
    head costs 22 items. Deleting either would make a documented claim
    unverifiable, and the corpus already cannot be rebuilt.
  * DO delete a run that can never execute -- a `pending` config whose dataset
    has been removed from disk. It contributes nothing but a false "unfinished"
    reading.
  * DO correct a `running` status with no process behind it. That one is not
    clutter, it is false.
  * MARK the rest, in a file both a human and a tool can read, and make the
    tools refuse.

A quarantined campaign keeps its data and gains a `QUARANTINE.json` at its
root. `is_quarantined()` is what the scorers call; `--check` is what a launch
or a test calls.

    python -m scripts.quarantine --list
    python -m scripts.quarantine --apply            # writes markers, dry by default
    python -m scripts.quarantine --apply --execute  # actually writes/removes
    python -m scripts.quarantine --check <root>     # exit 1 if quarantined
    python -m scripts.quarantine --self-test
"""

import argparse
import glob
import json
import os
import sys
import time

MARKER = "QUARANTINE.json"

# The registry. Each entry names a campaign, why it is dead, and what it is
# still good for -- because "dead" and "worthless" are different, and every
# entry below is still the receipt for something.
REGISTRY = {
    "uniform1_VOID_dose3.4pct_2026-08-25": dict(
        reason="ran at 3.4% of its constraint dose; superseded by `uniform1`",
        keep_for="the receipt for FRAMEWORK 2(u): an arm can land 1 of 29 steps "
                 "and still write `status: completed`",
        scorable=False),
    "iwc2": dict(
        reason="ViTB16 under fp16+GradScaler WITHOUT --constraint-fp32: landed "
               "173/232 constraint steps, 74.6%. `check_parity` is GREEN on it "
               "anyway -- only `dose_landed` sees this",
        keep_for="the only evidence that --constraint-fp32 is load-bearing on "
                 "ViTB16; superseded for results by `vitu1` at 100% dose",
        scorable=False),
    "iwc3": dict(
        reason="fp16+GradScaler with `constraint_fp32: False`: landed 716/1044 "
               "constraint steps, 68.6% -- a LOWER landing rate than `iwc2`, "
               "which was quarantined for exactly this. Superseded by `iwc4` "
               "(same design, 1044/1044, fp32 True)",
        keep_for="the receipt that fp16 without --constraint-fp32 silently "
                 "drops a third of the dose on a CNN",
        scorable=False),
    "iwc1": dict(
        reason="fp16+GradScaler with `constraint_fp32: False`: alm landed "
               "51.7%, fioretto 57.1%, tralo 66.8%, hounie 100% -- an "
               "arm-DEPENDENT dose spread, so its cross-arm ordering is a "
               "measurement of the GradScaler. It also carries no per-family "
               "nulls, the only live campaign with a trained arm lacking its "
               "own twin",
        keep_for="the receipt for the representation-channel finding, and for "
                 "how far an fp16 dose can diverge BETWEEN arms",
        scorable=False),
    "xfam1": dict(
        reason="fails `check_parity`: `run_code_version` splits by SEED, "
               "`9b89ce26d6bb` x142 against `9b89ce26d6bb-dirty` x182 (seed 1 "
               "clean, seeds 3-4 dirty). Its commit predates the "
               "`TRAINING_PATHS` fix `ca373f4e` by 53 minutes, so the dirty "
               "flag still diffed the WHOLE tree -- most likely a `scripts/` "
               "deploy, but unprovable after the fact",
        keep_for="the receipt that all four constraint terms are negative, "
                 "and the only TIGHT-cap source for the rival duals",
        scorable=False),
    "mc_sgd": dict(reason="dermmnist: leaked test set AND removed from disk. "
                          "32 pending, 0 completed -- nothing ever ran",
                   keep_for="nothing", scorable=False),
    "vit_diag": dict(reason="dermmnist: leaked and removed from disk; 40 of 49 "
                            "pending and unrunnable",
                     keep_for="8 completed diagnostic runs", scorable=False),
    "vit_ceskip": dict(reason="dermmnist: leaked and removed from disk; 46 of 48 "
                              "pending and unrunnable",
                       keep_for="1 completed run", scorable=False),
    "mnv3bar": dict(reason="dermmnist: leaked and removed from disk; 62 of 80 "
                           "pending and unrunnable",
                    keep_for="17 completed runs", scorable=False),
    "mc29": dict(reason="dermmnist: leaked test set (38.7% of test, 67.3% of "
                        "melanoma). Completed, but on data whose split is invalid",
                 keep_for="the receipt for the mc29 dose finding", scorable=False),
    "dosefix": dict(reason="dermmnist: leaked test set", keep_for="dose receipts",
                    scorable=False),
    "dualbar2": dict(reason="dermmnist: leaked test set", keep_for="the dual-bar "
                     "comparison, on invalid data", scorable=False),
    "selectrun": dict(reason="dermmnist: leaked test set",
                      keep_for="the receipt that `select` costs 22 items vs clip",
                      scorable=False),
}

# A dataset that is gone from disk. A `pending` run against one can never
# execute, so it is the one category safe to remove.
DEAD_DATASETS = ("dermmnist", "octmnist", "tissuemnist")


def campaign_roots(home=None):
    home = home or os.path.expanduser("~")
    out = []
    for t in sorted(glob.glob(os.path.join(home, "optloss-*"))
                    + [os.path.join(home, "OptimizationLoss")]):
        for c in sorted(glob.glob(os.path.join(t, "results", "*"))):
            if os.path.isdir(c):
                out.append(c)
    return out


def is_quarantined(root):
    """What the scorers call. Returns the marker dict, or None."""
    p = os.path.join(root, MARKER)
    if not os.path.exists(p):
        name = os.path.basename(os.path.normpath(root))
        return REGISTRY.get(name)
    try:
        return json.load(open(p))
    except Exception:
        return dict(reason="unreadable %s -- treat as quarantined" % MARKER,
                    scorable=False)


def scan(root):
    """Status counts, dataset set, and the runs that can never execute."""
    counts, datasets, unrunnable, stale = {}, set(), [], []
    quarantined = is_quarantined(root) is not None
    for f in glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*",
                                    "config.json")):
        try:
            d = json.load(open(f))
        except Exception:
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue
        st = d.get("status", "?")
        counts[st] = counts.get(st, 0) + 1
        ds = d.get("dataset_mode") or ""
        datasets.add(ds)
        # A pending run is removable when it can never execute (its dataset is
        # gone) OR when it sits inside a quarantined campaign. The second case
        # is the live hazard: `uniform1_VOID` holds 240 pending runs on
        # iwildcam, which EXISTS, so `main.py` would dispatch all 240 of a
        # campaign that is void by name. A marker file does not stop the
        # dispatcher; an absent config does.
        if st == "pending" and (ds in DEAD_DATASETS or quarantined):
            unrunnable.append(f)
        if st == "running" and (time.time() - os.path.getmtime(f)) > 2 * 86400:
            stale.append(f)
    return counts, datasets, unrunnable, stale


def cmd_list(home=None, out=sys.stdout):
    print("%-36s %-9s %-28s %s"
          % ("campaign", "state", "status counts", "why"), file=out)
    print("-" * 118, file=out)
    for root in campaign_roots(home):
        name = os.path.basename(root)
        counts, _, unrunnable, stale = scan(root)
        if not counts:
            continue
        q = is_quarantined(root)
        state = "QUARANTINE" if q else "live"
        note = (q or {}).get("reason", "")
        if len(note) > 60:
            note = note[:57] + "..."
        print("%-36s %-9s %-28s %s" % (name[:36], state, str(counts)[:28], note),
              file=out)
        if unrunnable:
            print("%-36s %-9s   %d pending run(s) that must never execute "
                  "(dead dataset, or a quarantined campaign the dispatcher "
                  "would still pick up)" % ("", "", len(unrunnable)), file=out)
        if stale:
            print("%-36s %-9s   %d run(s) claim `running` with no process behind "
                  "them" % ("", "", len(stale)), file=out)
    return 0


def cmd_apply(execute=False, home=None, out=sys.stdout):
    """Write markers, drop unrunnable configs, correct stale statuses."""
    wrote = removed = fixed = 0
    for root in campaign_roots(home):
        name = os.path.basename(root)
        counts, _, unrunnable, stale = scan(root)
        if not counts:
            continue
        entry = REGISTRY.get(name)

        if entry and not os.path.exists(os.path.join(root, MARKER)):
            payload = dict(entry, campaign=name, quarantined_by="scripts.quarantine")
            print("  marker  %s" % name, file=out)
            if execute:
                with open(os.path.join(root, MARKER), "w") as fh:
                    json.dump(payload, fh, indent=2)
            wrote += 1

        for f in unrunnable:
            if removed < 3:
                print("  remove  %s" % f.replace(os.path.expanduser("~/"), ""),
                      file=out)
            if execute:
                os.remove(f)
            removed += 1

        for f in stale:
            if fixed < 6:
                print("  correct %s  running -> crashed"
                      % f.replace(os.path.expanduser("~/"), ""), file=out)
            if execute:
                d = json.load(open(f))
                d["status"] = "crashed"
                d["status_corrected_by"] = "scripts.quarantine: claimed `running` " \
                                           "with no process behind it"
                json.dump(d, open(f, "w"), indent=2)
            fixed += 1

    print("\n%s: %d marker(s), %d unrunnable config(s) removed, %d stale status(es) "
          "corrected" % ("APPLIED" if execute else "DRY RUN (pass --execute)",
                         wrote, removed, fixed), file=out)
    if not execute:
        print("Nothing was changed.", file=out)
    return 0


def cmd_check(root, out=sys.stdout):
    q = is_quarantined(root)
    if not q:
        print("OK -- %s is not quarantined" % root, file=out)
        return 0
    print("QUARANTINED: %s" % root, file=out)
    print("  reason   : %s" % q.get("reason"), file=out)
    print("  keep for : %s" % q.get("keep_for"), file=out)
    print("  Do not score it. Do not pool it. It is kept as a receipt.", file=out)
    return 1


def self_test(out=sys.stdout):
    """Can the gate say NO, and does it say YES for anything else?

    A quarantine that never fires is decoration. A quarantine that fires on
    everything stops the project. Both directions are checked.
    """
    import shutil
    import tempfile

    ok = True
    tmp = tempfile.mkdtemp()
    try:
        live = os.path.join(tmp, "results", "a_live_campaign")
        dead = os.path.join(tmp, "results", "iwc2")
        for d in (live, dead):
            os.makedirs(d)

        checks = [
            ("a campaign in the registry is refused BY NAME, with no marker file",
             is_quarantined(dead) is not None),
            ("a campaign not in the registry is allowed",
             is_quarantined(live) is None),
            ("--check exits 1 on the dead one", cmd_check(dead, open(os.devnull, "w")) == 1),
            ("--check exits 0 on the live one", cmd_check(live, open(os.devnull, "w")) == 0),
        ]

        # a written marker must override the name, in BOTH directions
        with open(os.path.join(live, MARKER), "w") as fh:
            json.dump(dict(reason="marked by hand", scorable=False), fh)
        checks.append(("a hand-written marker quarantines a campaign not in the "
                       "registry", is_quarantined(live) is not None))

        # an unreadable marker must fail CLOSED, never open
        with open(os.path.join(live, MARKER), "w") as fh:
            fh.write("{ this is not json")
        q = is_quarantined(live)
        checks.append(("an unreadable marker fails CLOSED",
                       q is not None and q.get("scorable") is False))

        # every registry entry must carry a reason and a keep_for
        checks.append(("every registry entry states a reason and what it is kept for",
                       all(e.get("reason") and e.get("keep_for")
                           for e in REGISTRY.values())))
        # and none may be marked scorable -- that would be a contradiction
        checks.append(("no registry entry claims to be scorable",
                       not any(e.get("scorable") for e in REGISTRY.values())))

        print("SELF-TEST\n", file=out)
        for label, good in checks:
            print("  %-4s %s" % ("OK" if good else "FAIL", label), file=out)
            ok = ok and good
        print("\nSELF-TEST %s" % ("PASSED" if ok else "FAILED"), file=out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--execute", action="store_true",
                    help="with --apply, actually write and remove")
    ap.add_argument("--check", metavar="ROOT")
    ap.add_argument("--home", help="scan under this home instead of ~")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()
    if a.check:
        return cmd_check(a.check)
    if a.apply:
        return cmd_apply(execute=a.execute, home=a.home)
    if a.list:
        return cmd_list(home=a.home)
    ap.error("give --list, --apply, --check <root> or --self-test")


if __name__ == "__main__":
    sys.exit(main())
