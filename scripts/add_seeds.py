"""Add SEEDS to a campaign that already exists, at its own code version.

WHY THIS EXISTS. Seeds are the only axis the scorer may collapse, and every
verdict this project has reached lately is limited by how few of them there
are: the RNG floor under `vitdual2`/`vitcoin1` rests on TWO observations
against a `MIN_FLOOR_OBS` of 8, and the observed `tralo` effect at those cells
is 50-100% of the whole available prize yet still sits under the minimum
detectable effect at 4 seeds. That is "could not have seen it", not "no
effect", and the fix is seeds, not a new idea.

But `gen_campaign` reads its seed list from `configs/protocol.yml`, and
`configs/` is FROZEN on the server while a campaign runs -- `code_version` is a
git hash, so editing it to bump the seed list would split the very campaign the
new seeds are meant to extend. The seed is not even a config field: it is baked
into `base_model_id`, so the configs cannot be produced by copying a sibling
and editing a number either.

So this reads the campaign's OWN protocol and emits the additional seeds the
generator would have emitted, writing only into `results/`. Nothing under
`configs/`, `src/` or `main.py` is touched or even opened for writing.

🛑 THE FIDELITY CHECK IS THE POINT, NOT A NICETY. Before writing anything, it
REGENERATES every config already on disk and requires a byte-level match on
every field that defines the experiment. If this tool and `gen_campaign`
disagree by so much as a default, the new seeds are not replicates of the old
ones -- they are a second arm wearing the first one's name, and the pooled
"8 seeds" would be two populations of four. It refuses rather than warn: a
silent split is exactly the failure mode that quarantined four campaigns.

    python -m scripts.add_seeds --root results/vitdual2 --seeds 5 6 7 8
    python -m scripts.add_seeds --root results/vitdual2 --seeds 5 6 7 8 \
        --arms clip focal_clip tralo tralo_null tralo_reseed --execute

Dry run unless `--execute`. Never overwrites an existing config, and never
touches a `completed` run.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.gen_campaign import (build_hyperparams, cap_pair,  # noqa: E402
                                  code_version, compute_base_model_id,
                                  load_protocol, _cls_tag)

# Fields a config carries that are written by the RUNNER, not the generator, so
# they cannot take part in a fidelity comparison.
RUNTIME_KEYS = ("status", "run_code_version", "results", "reordering",
                "data_fingerprint")


def cells(root):
    """Every (model, dataset, cap, arm, seed) already staged under `root`."""
    out = []
    for dirpath, _dirs, files in os.walk(root):
        if "config.json" not in files:
            continue
        parts = os.path.normpath(dirpath).split(os.sep)
        if len(parts) < 5 or not parts[-1].startswith("seed_"):
            continue
        try:
            seed = int(parts[-1].split("_", 1)[1])
        except ValueError:
            continue
        out.append((parts[-5], parts[-4], parts[-3], parts[-2], seed,
                    os.path.join(dirpath, "config.json")))
    return sorted(out)


_MISSING = object()


def emit(P, model, ds, cap, arm, seed, dc, version, overrides=None):
    """The config `gen_campaign` would write for this cell. Kept in one place
    so the fidelity check and the writer cannot drift apart."""
    hp = build_hyperparams(P, P["arms"][arm], seed)
    if overrides:
        # ONLY where the key already exists. `clip` and `focal_clip` take no
        # constraint step, so `gen_campaign` gives them no constraint keys at
        # all -- adding `constraint_grad_mode` to a post-hoc arm would invent a
        # setting the generator never wrote and change its base_model_id.
        for k, v in overrides.items():
            if k in hp:
                hp[k] = v
    return {"methodology": P["arms"][arm]["methodology"], "model_name": model,
            "constraint": cap_pair(cap), "constraint_tag": cap,
            "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
            "base_model_id": compute_base_model_id(P, model, hp, ds, dc),
            "arm": arm,
            "exp_name": "%s_%s_%s_%s_c%s_seed%d"
                        % (model, ds, arm, cap, _cls_tag(dc), seed),
            "status": "pending", "code_version": version}


def campaign_recipe(P, existing, version, out=sys.stdout):
    """The hyperparameters this campaign was generated with that the PROTOCOL
    alone does not produce. Returns (overrides, failures).

    🛑 THE PROTOCOL DOES NOT DETERMINE A CAMPAIGN, AND THIS IS THE PROJECT'S
    MOST EXPENSIVE FOOTGUN. `--constraint-fp32` and `--constraint-grad-mode`
    are COMMAND-LINE flags whose protocol defaults are `False` and `clip`,
    while the recipe that defines the corpus is `constraint_fp32: True` +
    `constraint_grad_mode: normalize`. Regenerating from `protocol.yml` alone
    silently produces a DIFFERENT METHOD -- five distinct TraLO configurations
    existed across 277 runs for exactly this reason, and `taskwin1` was killed
    for landing 69% of its dose after being staged without the fp32 flag.

    So the recipe is READ BACK OFF THE CAMPAIGN rather than restated on a
    command line. A flag that must be typed correctly every time is a flag that
    will eventually be typed wrong.

    An override is adopted ONLY if it is consistent across every arm, cap and
    seed in the campaign. A value that differs from the protocol in some runs
    and not others is not a recipe -- it is a split campaign, and this refuses
    rather than picking a side.
    """
    seen, present, fails = {}, {}, []
    for model, ds, cap, arm, seed, path in existing:
        try:
            have = json.load(open(path))
        except (ValueError, OSError):
            continue
        if arm not in P["arms"]:
            continue
        want = emit(P, model, ds, cap, arm, seed, have.get("dataset_config"),
                    version)["hyperparams"]
        for k in want:
            present[k] = present.get(k, 0) + 1
        for k, v in (have.get("hyperparams") or {}).items():
            if want.get(k, _MISSING) != v:
                seen.setdefault(k, {}).setdefault(repr(v), []).append(
                    "%s/%s/seed_%d" % (cap, arm, seed))

    overrides = {}
    for k, byval in sorted(seen.items()):
        if len(byval) != 1:
            fails.append(
                "hyperparameter %r takes %d different values across this "
                "campaign (%s). That is not a recipe, it is a SPLIT campaign, "
                "and adding seeds to it would deepen the split"
                % (k, len(byval),
                   "; ".join("%s in %d run(s)" % (val, len(w))
                             for val, w in sorted(byval.items()))))
            continue
        val_repr, where = list(byval.items())[0]
        # The denominator is the runs whose regenerated hyperparameters CARRY
        # this key, not every run in the campaign. A post-hoc arm has no
        # constraint keys, so `constraint_fp32` legitimately overrides 72 of
        # 88 runs in an 11-arm campaign with two clippers -- demanding all 88
        # would reject every real campaign.
        if len(where) != present.get(k, 0):
            fails.append(
                "hyperparameter %r overrides the protocol in %d of the %d "
                "run(s) that carry it (e.g. %s) but not the rest. A "
                "campaign-wide flag touches every run that has the key; this "
                "looks like a hand edit"
                % (k, len(where), present.get(k, 0), where[0]))
            continue
        # take the real value, not its repr
        for _m, _d, _c, _a, _s, path in existing:
            hp = (json.load(open(path)).get("hyperparams") or {})
            if k in hp:
                overrides[k] = hp[k]
                break

    if overrides:
        print("  recipe (read off the campaign, not the protocol):", file=out)
        for k in sorted(overrides):
            print("      %-28s = %r" % (k, overrides[k]), file=out)
    return overrides, fails


def fidelity(P, existing, version, overrides=None, out=sys.stdout):
    """Regenerate what is ALREADY on disk and demand it match.

    Returns a list of failures. A non-empty list must stop the run: it means
    this emitter and `gen_campaign` disagree, so the seeds about to be written
    would not be replicates of the seeds already there.
    """
    fails, checked = [], 0
    for model, ds, cap, arm, seed, path in existing:
        try:
            have = json.load(open(path))
        except (ValueError, OSError) as exc:
            fails.append("%s is unreadable (%s)" % (path, exc))
            continue
        if arm not in P["arms"]:
            fails.append("%s holds arm %r, which this protocol does not "
                         "declare -- the campaign was generated from a "
                         "different protocol" % (path, arm))
            continue
        want = emit(P, model, ds, cap, arm, seed, have.get("dataset_config"),
                    version, overrides)
        for k in sorted(set(want) | set(have)):
            if k in RUNTIME_KEYS:
                continue
            if want.get(k) != have.get(k):
                fails.append(
                    "%s %s seed %d: field %r would be regenerated as %r but is "
                    "%r on disk" % (cap, arm, seed, k,
                                    want.get(k), have.get(k)))
        checked += 1
    print("  fidelity: regenerated %d existing config(s), %d mismatch(es)"
          % (checked, len(fails)), file=out)
    if not checked:
        fails.append("NOTHING was checked: no existing config could be "
                     "regenerated, so the fidelity guarantee is vacuous and "
                     "the new seeds are unverified")
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restrict to these arms; default is every arm "
                         "already present in the campaign")
    ap.add_argument("--out", default=None,
                    help="write the new seeds HERE instead of into --root. "
                         "Use this when --root is a live campaign: adding "
                         "seeds to only some of its arms makes its coverage "
                         "ragged and turns `check_parity` red. The extension "
                         "carries the same protocol and the same "
                         "code_version, so the two roots pool.")
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--execute", action="store_true",
                    help="write the configs; without it this is a dry run")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.root or not args.seeds:
        ap.error("--root and --seeds are required")

    P = load_protocol(args.protocol) if args.protocol else load_protocol()
    existing = cells(args.root)
    if not existing:
        print("no configs under %s -- nothing to extend. This tool ADDS seeds "
              "to a campaign that exists; use gen_campaign to create one."
              % args.root)
        return 1

    # 1. one code version, and it must be THIS tree's.
    stamps, unread = set(), []
    for _m, _d, _c, _a, _s, path in existing:
        try:
            stamps.add(json.load(open(path)).get("code_version"))
        except (ValueError, OSError) as exc:
            # NOT swallowed: an unreadable config is a config whose code
            # version we do not know, and "one stamp" would then be a claim
            # about a set we could not see.
            unread.append("%s (%s)" % (path, exc))
    version = code_version()
    if unread:
        print("REFUSED: %d config(s) could not be read, so this campaign "
              "cannot be shown to carry ONE code version:" % len(unread))
        for u in unread[:10]:
            print("   - %s" % u)
        return 1
    print("campaign %s" % args.root)
    print("  existing: %d config(s), code_version %s"
          % (len(existing), " ".join(sorted(map(str, stamps)))))
    print("  this tree: %s" % version)
    if len(stamps) != 1:
        print("REFUSED: the campaign already carries %d code versions. Adding "
              "seeds cannot fix that and would deepen it." % len(stamps))
        return 1
    if stamps != {version}:
        print("REFUSED: this tree is at %s but the campaign was generated at "
              "%s. Seeds written here would not be replicates of the seeds "
              "already there -- they would be a second experiment in the same "
              "directory. Check out the campaign's commit, or add the seeds "
              "from the worktree that holds it." % (version, stamps.pop()))
        return 1

    # 1b. THIS TOOL CREATES RUNS, so it must refuse what the scorers refuse.
    # It is the only script in the repo that stages new GPU work, and it had
    # neither check: it would happily extend a quarantined campaign, and it
    # would faithfully reproduce an OFF-RECIPE campaign, staging GPU-days of
    # runs that `rig_status` refuses and no scorer may pool.
    from scripts import quarantine
    from scripts.rig_status import recipe_of, recipe_verdict
    blocked, dead = quarantine.gate([args.root], verb="extend")
    if blocked:
        return 1
    here = dead.for_path(args.root) if hasattr(dead, "for_path") else dead
    if here:
        # ⚠️ REFUSE THE DEAD ARMS, NOT THE CAMPAIGN. A blanket refusal here was
        # wrong and blocked the documented remedy: `dom1`, `dom1b` and
        # `equaldose1` are all PARTIAL, and their `keep_for` says `tralo` vs
        # `clip`/`tralo_null`/`tralo_reseed` is at EQUAL dose and unaffected --
        # while `sensitivity_screen` prices "seeds 5-8 on the existing pair"
        # as the fix for the 4-observation noise floor. The over-broad gate
        # blocked exactly the campaigns that needed it.
        want = set(args.arms or [])
        clash = sorted(want & set(here)) if want else []
        if clash:
            print("REFUSED: %s is a DEAD arm of %s (partial quarantine), so "
                  "more seeds of it buy nothing -- the contrast is "
                  "disqualified at any n." % (", ".join(clash), args.root))
            return 1
        print("!! PARTIAL QUARANTINE on %s: dead arm(s) %s will NOT be "
              "extended." % (args.root, ", ".join(sorted(here))))
        print("   Every other arm here is at equal dose and is extended "
              "normally.")
        print("")
    cfgs, unreadable = [], []
    for _m, _d, _c, _a, _s, path in existing:
        try:
            cfgs.append(json.load(open(path)))
        except (ValueError, OSError) as exc:
            unreadable.append("%s (%s)" % (path, exc))
    if unreadable:
        # A config this tool cannot read is one whose recipe it cannot check,
        # and staging seeds off a partially-read campaign is exactly how a
        # mixed-recipe extension gets written. Refuse, naming every file.
        print("REFUSED: %d config(s) under %s could not be read, so the "
              "campaign's recipe cannot be established:"
              % (len(unreadable), args.root))
        for u in unreadable:
            print("   %s" % u)
        return 1
    verdict, why = recipe_verdict(recipe_of(cfgs))
    if verdict == "fail":
        print("REFUSED: this campaign is OFF-RECIPE -- %s" % why)
        print("  Reproducing it faithfully would stage more runs of a "
              "DIFFERENT METHOD.")
        return 1

    # 2. the recipe is a property of the CAMPAIGN, not of protocol.yml.
    overrides, fails = campaign_recipe(P, existing, version)
    if fails:
        print("")
        print("REFUSED: this campaign does not carry one consistent recipe:")
        for f in fails[:20]:
            print("   - %s" % f)
        return 1

    # 3. the emitter must agree with the generator, on THIS campaign.
    fails = fidelity(P, existing, version, overrides)
    if fails:
        print("")
        print("REFUSED: %d fidelity mismatch(es). This tool would not "
              "reproduce the runs already on disk, so the new seeds would not "
              "be replicates of them:" % len(fails))
        for f in fails[:20]:
            print("   - %s" % f)
        if len(fails) > 20:
            print("   ... and %d more" % (len(fails) - 20))
        return 1

    # 4. what to write
    have = {(m, d, c, a, s) for m, d, c, a, s, _p in existing}
    arms = sorted({a for _m, _d, _c, a, _s, _p in existing}
                  if args.arms is None else set(args.arms))
    unknown = [a for a in arms
               if a not in {x[3] for x in existing}]
    if unknown:
        print("REFUSED: %s not present in this campaign. Adding an ARM is a "
              "different experiment, not another seed of this one; that needs "
              "gen_campaign and its protocol assertions."
              % ", ".join(unknown))
        return 1
    grid = sorted({(m, d, c) for m, d, c, _a, _s, _p in existing})
    dcs = {}
    for m, d, c, _a, _s, path in existing:
        dcs.setdefault((m, d, c), json.load(open(path)).get("dataset_config"))

    # With --out the template root is NOT the destination, so a seed present
    # in the template says nothing about the destination. Skipping on it would
    # silently drop every seed the two roots happen to share.
    blocked = have if args.out is None else {
        (m, d, c, a, s) for m, d, c, a, s, _p in cells(args.out)}
    # 🛑 (cell, arm) PAIRS THAT EXIST, NOT `grid x arms`. The cross product
    # RESURRECTS combinations the campaign never had: a ragged grid -- which is
    # the normal state after `quarantine --execute` drops pending runs, 34 from
    # `vittask1` and 84 from `vitdual1` -- would gain those cells back at ONE
    # seed, and `cell_table` would then emit them with n_seeds=1 and an sd from
    # a single observation. The arm-membership check above is campaign-global
    # and does not catch it.
    arms = [a for a in arms if a not in here]
    have_pairs = sorted({(m, d, c, a)
                         for m, d, c, a, _s, _p in existing if a in arms})
    todo = [(m, d, c, a, s) for (m, d, c, a) in have_pairs
            for s in sorted(args.seeds) if (m, d, c, a, s) not in blocked]
    dup = [(m, d, c, a, s) for (m, d, c, a) in have_pairs
           for s in sorted(args.seeds) if (m, d, c, a, s) in blocked]

    print("  arms   : %s" % " ".join(arms))
    print("  cells  : %d" % len(grid))
    print("  seeds  : %s" % " ".join(map(str, sorted(args.seeds))))
    print("  to write: %d   already present (skipped): %d"
          % (len(todo), len(dup)))
    if not todo:
        print("nothing to do")
        return 0

    dest_root = args.out or args.root
    if args.out:
        print("  writing to: %s (extension root; --root is the template)"
              % args.out)
    written = 0
    for m, d, c, a, s in todo:
        cfg = emit(P, m, d, c, a, s, dcs[(m, d, c)], version, overrides)
        path = os.path.join(dest_root, m, d, c, a, "seed_%d" % s)
        dest = os.path.join(path, "config.json")
        if os.path.exists(dest):
            continue
        if args.execute:
            os.makedirs(path, exist_ok=True)
            json.dump(cfg, open(dest, "w"), indent=2)
        written += 1

    if args.execute:
        print("WROTE %d config(s) -> %s" % (written, dest_root))
        if args.out:
            # Without this marker the extension scores as its OWN cell and
            # the seeds it was bought for never reach the parent. The
            # docstring above promised the two roots pool; until
            # 2026-09-05 nothing implemented it, and `vitseed1` plus
            # `seed58a` both sat unpooled. `write_extends_marker` verifies
            # before it writes, so a mis-staged extension refuses here
            # rather than at scoring time.
            quarantine.write_extends_marker(
                args.out, os.path.basename(os.path.normpath(args.root)))
        print("  run `python -m scripts.check_parity %s` before launching."
              % dest_root)
    else:
        print("DRY RUN (pass --execute): %d config(s) would be written"
              % written)
        print("Nothing was changed.")
    return 0


def self_test():
    """Gate it in BOTH directions: it must extend a healthy campaign and
    REFUSE a mismatched one."""
    import shutil
    import tempfile

    P = load_protocol()
    version = code_version()
    tmp = tempfile.mkdtemp(prefix="add_seeds_")
    checks = []
    try:
        root = os.path.join(tmp, "camp")
        dc = {"data_dir": "data/iwildcam/oodslice", "num_classes": 8,
              "group_column": "location", "constrained_class": [2, 7],
              "disjoint_groups": True}
        # THE RECIPE. `gen_campaign` DEFAULTS these off, so a fixture built
        # from protocol defaults alone is a DIFFERENT METHOD and the recipe
        # gate refuses it -- correctly. Every fixture below is on-recipe, so
        # the controls exercise the path they name rather than tripping this.
        REC = {"constraint_fp32": True, "constraint_grad_mode": "normalize"}
        base = []
        for arm in ("clip", "tralo", "tralo_null", "tralo_reseed"):
            for seed in (1, 2):
                cfg = emit(P, "ViTB16", "iwildcam", "L80-80_G95", arm, seed,
                           dc, version, REC)
                d = os.path.join(root, "ViTB16", "iwildcam", "L80-80_G95",
                                 arm, "seed_%d" % seed)
                os.makedirs(d)
                json.dump(cfg, open(os.path.join(d, "config.json"), "w"),
                          indent=2)
                base.append(os.path.join(d, "config.json"))

        rc = main(["--root", root, "--seeds", "3", "4", "--execute"])
        n = len(cells(root))
        checks.append(("a healthy campaign is extended (rc=0, 8 -> 16)",
                       rc == 0 and n == 16))

        # every NEW seed must differ from every old one, or the extra runs are
        # duplicates and buy no observations at all.
        ids = {}
        for m, d_, c, a, s, path in cells(root):
            ids.setdefault(a, {})[s] = json.load(open(path))["base_model_id"]
        distinct = all(len(set(v.values())) == len(v) for v in ids.values())
        checks.append(("each added seed gets its OWN base_model_id, so it is "
                       "a new model and not a re-run of seed 1", distinct))

        # rerunning is a no-op, never an overwrite
        rc = main(["--root", root, "--seeds", "3", "4", "--execute"])
        checks.append(("re-running writes nothing (idempotent)",
                       rc == 0 and len(cells(root)) == 16))

        # NEGATIVE CONTROL 1: a campaign whose configs this tool would NOT
        # reproduce must be REFUSED, not extended.
        bad = json.load(open(base[0]))
        bad["hyperparams"] = dict(bad["hyperparams"])
        bad["hyperparams"]["lr"] = 0.123456
        json.dump(bad, open(base[0], "w"), indent=2)
        rc = main(["--root", root, "--seeds", "5", "--execute"])
        checks.append(("REFUSES a campaign it cannot reproduce (fidelity)",
                       rc == 1 and len(cells(root)) == 16))
        json.dump(emit(P, "ViTB16", "iwildcam", "L80-80_G95", "clip", 1, dc,
                       version, REC), open(base[0], "w"), indent=2)

        # NEGATIVE CONTROL 2: a foreign code version must be REFUSED, or the
        # added seeds silently split the campaign in two.
        other = json.load(open(base[1]))
        other["code_version"] = "deadbeefcafe"
        json.dump(other, open(base[1], "w"), indent=2)
        rc = main(["--root", root, "--seeds", "5", "--execute"])
        checks.append(("REFUSES a campaign carrying a foreign code_version",
                       rc == 1 and len(cells(root)) == 16))
        json.dump(emit(P, "ViTB16", "iwildcam", "L80-80_G95", "clip", 2, dc,
                       version, REC), open(base[1], "w"), indent=2)

        # NEGATIVE CONTROL 3: adding an ARM is a different experiment.
        rc = main(["--root", root, "--seeds", "5", "--arms", "alm",
                   "--execute"])
        checks.append(("REFUSES to add an ARM (that is a new experiment, not "
                       "another seed)", rc == 1))

        # NEGATIVE CONTROL 4: an empty root is not silently 'nothing to do'.
        rc = main(["--root", os.path.join(tmp, "empty"), "--seeds", "1"])
        checks.append(("REFUSES an empty root rather than reporting success",
                       rc == 1))

        # THE RECIPE. A campaign generated with `--constraint-grad-mode
        # normalize --constraint-fp32` must hand those to its new seeds. If it
        # does not, the extension is a DIFFERENT METHOD wearing the same arm
        # names, and the pooled seeds are two populations.
        rec = os.path.join(tmp, "recipe")
        RECIPE = {"constraint_grad_mode": "normalize", "constraint_fp32": True}
        for arm in ("clip", "tralo", "tralo_null", "tralo_reseed"):
            for seed in (1, 2):
                cfg = emit(P, "ViTB16", "iwildcam", "L80-80_G95", arm, seed,
                           dc, version, RECIPE)
                d = os.path.join(rec, "ViTB16", "iwildcam", "L80-80_G95",
                                 arm, "seed_%d" % seed)
                os.makedirs(d)
                json.dump(cfg, open(os.path.join(d, "config.json"), "w"),
                          indent=2)
        rc = main(["--root", rec, "--seeds", "3", "--execute"])
        by_arm = {}
        for _m, _d, _c, a, s, p in cells(rec):
            by_arm.setdefault(a, {})[s] = json.load(open(p))["hyperparams"]
        new = [a for a in by_arm if 3 in by_arm[a]]
        # Compared against seed 1 OF THE SAME ARM, not against RECIPE: a
        # post-hoc arm carries no constraint keys at all, so demanding it hold
        # `constraint_grad_mode` would require inventing a setting the
        # generator never wrote.
        kept = all(by_arm[a][3].get(k) == by_arm[a][1].get(k)
                   for a in new for k in RECIPE)
        touched = any(by_arm[a][1].get(k) == v
                      for a in new for k, v in RECIPE.items())
        checks.append(("the campaign's RECIPE (a CLI flag, not a protocol "
                       "default) is carried into the new seeds",
                       rc == 0 and len(new) == 4 and kept and touched))

        # NEGATIVE CONTROL 5: a campaign that is NOT on one recipe must be
        # refused, not silently resolved toward one side.
        victim = [p for _m, _d, _c, a, s, p in cells(rec)
                  if s == 3 and a == "tralo"][0]
        split = json.load(open(victim))
        split["hyperparams"]["constraint_grad_mode"] = "clip"
        json.dump(split, open(victim, "w"), indent=2)
        rc = main(["--root", rec, "--seeds", "4", "--execute"])
        checks.append(("REFUSES a campaign that mixes two recipes internally",
                       rc == 1))

        # --out writes the extension elsewhere and leaves --root alone, and
        # it must NOT skip a seed merely because the template has it.
        ext = os.path.join(tmp, "ext")
        rc = main(["--root", root, "--seeds", "1", "2", "--out", ext,
                   "--arms", "tralo", "tralo_null", "--execute"])
        checks.append(("--out writes the extension to its own root and "
                       "does NOT skip seeds the template already has",
                       rc == 0 and len(cells(ext)) == 4
                       and len(cells(root)) == 16))

        # NEGATIVE CONTROL 6: a RAGGED grid must not be filled in. The cross
        # product would resurrect (cell, arm) pairs the campaign never had --
        # at ONE seed, with an sd from a single observation.
        rag = os.path.join(tmp, "ragged")
        for arm, caps in (("clip", ("L80-80_G95", "L90-90_G95")),
                          ("tralo", ("L80-80_G95",)),
                          ("tralo_null", ("L80-80_G95",)),
                          ("tralo_reseed", ("L80-80_G95",))):
            for cap in caps:
                for seed in (1, 2):
                    cfg = emit(P, "ViTB16", "iwildcam", cap, arm, seed, dc,
                               version, REC)
                    d = os.path.join(rag, "ViTB16", "iwildcam", cap, arm,
                                     "seed_%d" % seed)
                    os.makedirs(d)
                    json.dump(cfg, open(os.path.join(d, "config.json"), "w"),
                              indent=2)
        before = {(m, d_, c, a) for m, d_, c, a, _s, _p in cells(rag)}
        rc = main(["--root", rag, "--seeds", "3", "--execute"])
        after = {(m, d_, c, a) for m, d_, c, a, _s, _p in cells(rag)}
        checks.append(("a RAGGED grid gains SEEDS but never a (cell, arm) it "
                       "did not already have", rc == 0 and after == before))

        # NEGATIVE CONTROLS 7-9: THE TWO GATES THIS TOOL GAINED MUST FIRE.
        # It is the only script in the repo that CREATES runs, and it had
        # neither check -- so it would extend a campaign no scorer may read,
        # and reproduce an off-recipe one faithfully into more GPU-days of a
        # different method. A gate never shown to fail has not been shown to
        # work, so each is driven here from the real registry, not a stub.
        def _clone(name, mutate=None):
            dst = os.path.join(tmp, name)
            shutil.copytree(root, dst)
            if mutate:
                for _m, _d, _c, _a, _s, path in cells(dst):
                    cfg = json.load(open(path))
                    if mutate(cfg):
                        json.dump(cfg, open(path, "w"), indent=2)
            return dst, len(cells(dst))

        q, nq = _clone("uniform1")          # scorable=False in the registry
        rc = main(["--root", q, "--seeds", "9", "--execute"])
        checks.append(("REFUSES to extend a QUARANTINED campaign (registry, "
                       "not a stub)", rc == 1 and len(cells(q)) == nq))

        # PARTIAL marker: `dom1`'s dead arms are `fioretto` and `hounie`, and
        # its `keep_for` says every OTHER contrast is at equal dose. So the
        # right behaviour is not a refusal -- it is extending the live arms
        # and leaving the dead ones alone. Both directions are checked.
        d1, _nd = _clone("dom1")
        for seed in (1, 2):
            cfg = emit(P, "ViTB16", "iwildcam", "L80-80_G95", "fioretto",
                       seed, dc, version, REC)
            dd = os.path.join(d1, "ViTB16", "iwildcam", "L80-80_G95",
                              "fioretto", "seed_%d" % seed)
            os.makedirs(dd)
            json.dump(cfg, open(os.path.join(dd, "config.json"), "w"),
                      indent=2)
        rc = main(["--root", d1, "--seeds", "9", "--execute"])
        by_arm = {}
        for _m, _d, _c, a, s, _p in cells(d1):
            by_arm.setdefault(a, set()).add(s)
        checks.append(("a PARTIAL campaign is extended on its LIVE arms and "
                       "the DEAD ones gain nothing",
                       rc == 0
                       and 9 not in by_arm.get("fioretto", set())
                       and all(9 in by_arm.get(a, set())
                               for a in ("clip", "tralo", "tralo_null",
                                         "tralo_reseed"))))
        rc = main(["--root", d1, "--seeds", "10", "--arms", "fioretto",
                   "--execute"])
        after = {s for _m, _d, _c, a, s, _p in cells(d1) if a == "fioretto"}
        checks.append(("...but ASKING for a dead arm by name is REFUSED",
                       rc == 1 and 10 not in after))

        def _offrecipe(cfg):
            hp = cfg.get("hyperparams") or {}
            if "constraint_grad_mode" not in hp:
                return False
            hp["constraint_grad_mode"] = "clip"
            hp["constraint_fp32"] = False
            return True

        orc, no = _clone("offrecipe", _offrecipe)
        rc = main(["--root", orc, "--seeds", "9", "--execute"])
        checks.append(("REFUSES an OFF-RECIPE campaign rather than "
                       "reproducing a DIFFERENT METHOD faithfully",
                       rc == 1 and len(cells(orc)) == no))

        # POSITIVE CONTROL for the three above: the SAME clone, unmutated and
        # under a name the registry does not know, must still be extended --
        # or the three refusals prove only that copytree broke something.
        live, nl = _clone("livecamp")
        rc = main(["--root", live, "--seeds", "9", "--execute"])
        checks.append(("...and the same campaign under a live name, on "
                       "recipe, IS extended", rc == 0 and len(cells(live)) > nl))

        # a dry run must change nothing
        before = len(cells(root))
        rc = main(["--root", root, "--seeds", "9"])
        checks.append(("a DRY RUN writes nothing",
                       rc == 0 and len(cells(root)) == before))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    for name, ok in checks:
        print("  %-72s %s" % (name[:72], "PASS" if ok else "FAIL"))
    bad = [n for n, ok in checks if not ok]
    print("")
    print("ALL PASS" if not bad else "FAILED: %d" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
