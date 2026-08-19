"""Prove a campaign compares apples to apples, BEFORE it runs.

Every retraction in this project traces to two arms differing in something other
than the thing under test: unequal compute (worth 7-9 pp), an unequal
lr_constraint (worth 16 pp), a cap level present for one arm and not another, or
two arms silently sharing a cached warm-up so one of them was never really run.
This checks all four on the generated configs.

    python -m scripts.check_parity results/<campaign>

Exit code 1 if anything fails, so it can gate a launch.
"""
import collections
import glob
import json
import os
import sys

# knobs that must be IDENTICAL across every arm; if they differ, the comparison
# measures the knob and not the method
SHARED_KEYS = ["lr", "lr_constraint", "dropout", "batch_size", "pretrained",
               "class_weighted_ce", "constraint_chunk_size", "stable_count_threshold",
               # measured at -0.0351 AP within-run: if one arm restores its best
               # feasible checkpoint and another keeps its final model, the delta
               # between them is the restore, not the method
               "enable_checkpoint_restore"]


def load(root):
    runs = []
    for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = json.load(open(p))
        runs.append(c)
    return runs


def _report_passes(runs, arms):
    """What each arm actually executes, beside the epoch count.

    Optimizer epochs are equal by construction. FORWARD AND BACKWARD PASSES are
    not: every constraint epoch runs a full CE epoch over the train set AND two
    passes over the test set (pass 1 FP32 no-grad for the counts, pass 2
    carrying the constraint gradient). A post-hoc arm runs neither test pass.

    The excess therefore scales with n_test/n_train and DIFFERS PER DATASET,
    which makes any cross-dataset reading of a trained arm's margin a
    confounded one. It is printed rather than gated because the direction
    favours the trained arms -- it cannot manufacture the negative result the
    project currently reports -- but it belongs on screen, not in a reviewer's
    first question.
    """
    print()
    print("   passes per run (train-set epochs are shared; TEST-set passes are not):")
    print("   %-12s %14s %14s %16s" % ("arm", "train epochs", "test passes",
                                       "test passes are"))
    for arm in arms:
        hp = [r["hyperparams"] for r in runs if r["arm"] == arm]
        if not hp:
            continue
        wu = sorted({h["warmup_epochs"] for h in hp})[0]
        ce = sorted({h["constraint_epochs"] for h in hp})[0]
        # every epoch, warm-up or constraint, runs one CE pass over the train set
        train_epochs = wu + ce
        # only constraint epochs touch the test set, twice
        test_passes = 2 * ce
        kind = "-" if not ce else "1 no-grad + 1 with-grad, per constraint epoch"
        print("   %-12s %14d %14d   %s" % (arm, train_epochs, test_passes, kind))
    trained = [a for a in arms
               if any(r["hyperparams"]["constraint_epochs"] for r in runs
                      if r["arm"] == a)]
    if trained and len(trained) != len(arms):
        print("   ^ the trained arms do %d extra full passes over the TEST set "
              "that the" % (2 * max(
                  r["hyperparams"]["constraint_epochs"] for r in runs)))
        print("     post-hoc arms do not. Equal OPTIMIZER EPOCHS is not equal FLOPs,")
        print("     and the gap scales with n_test/n_train so it differs per dataset.")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    runs = load(root)
    if not runs:
        sys.exit("no configs under %s" % root)
    arms = sorted({r["arm"] for r in runs})
    fails = []

    print("campaign: %s" % root)
    print("%d runs, %d arms: %s\n" % (len(runs), len(arms), " ".join(arms)))

    # ---- 1. equal compute ---------------------------------------------------
    print("1. OPTIMIZER EPOCHS  (warm-up + constraint must total the same "
          "for every arm)")
    print("   NOTE: this is compute parity under ONE definition of compute.")
    totals = set()
    for arm in arms:
        hp = [r["hyperparams"] for r in runs if r["arm"] == arm]
        wu = {h["warmup_epochs"] for h in hp}
        ce = {h["constraint_epochs"] for h in hp}
        tot = {a + b for a in wu for b in ce}
        totals |= tot
        print("   %-12s warm-up %-6s constraint %-6s total %s"
              % (arm, sorted(wu), sorted(ce), sorted(tot)))
    if len(totals) == 1:
        print("   OK -- every arm gets %d optimizer epochs" % sorted(totals)[0])
    else:
        fails.append("UNEQUAL COMPUTE: totals %s" % sorted(totals))
        print("   FAIL -- differing totals: %s" % sorted(totals))
    _report_passes(runs, arms)
    print()

    # ---- 2. shared knobs ----------------------------------------------------
    print("2. SHARED KNOBS  (identical wherever present, or the delta is the knob)")
    print("   Absence is not a mismatch: a post-hoc arm has no constraint phase,")
    print("   so it carries no lr_constraint. WHICH arms carry a key is checked by")
    print("   scripts.audit_config; here we check they AGREE on the value.")
    for k in SHARED_KEYS:
        carriers = sorted({r["arm"] for r in runs if k in r["hyperparams"]})
        vals = {json.dumps(r["hyperparams"][k]) for r in runs if k in r["hyperparams"]}
        if not carriers:
            print("   --   %-24s carried by no arm" % k)
            continue
        status = "OK  " if len(vals) == 1 else "FAIL"
        if len(vals) != 1:
            fails.append("%s differs across the arms that carry it: %s"
                         % (k, sorted(vals)))
        missing = sorted(set(arms) - set(carriers))
        note = "" if not missing else "   (absent on: %s)" % " ".join(missing)
        print("   %s %-24s %s%s" % (status, k, sorted(vals), note))
    print()

    # ---- 3. cell coverage ---------------------------------------------------
    print("3. COVERAGE  (every arm must cover the same cells and seeds)")
    cells = collections.defaultdict(set)
    for r in runs:
        # Same omission as the scorer had: without the capped class, two roots
        # that cap different classes look like one cell and coverage passes on
        # a campaign that covers neither properly.
        key = (r["dataset_mode"], r["model_name"], r["constraint_tag"],
               str(r.get("dataset_config", {}).get("constrained_class")),
               r["hyperparams"]["seed"])
        cells[r["arm"]].add(key)
    ref = cells[arms[0]]
    for arm in arms:
        d = cells[arm]
        mark = "OK  " if d == ref else "FAIL"
        if d != ref:
            fails.append("%s covers %d cells, %s covers %d"
                         % (arm, len(d), arms[0], len(ref)))
        print("   %s %-12s %d cell-seeds" % (mark, arm, len(d)))
    caps = sorted({r["constraint_tag"] for r in runs})
    print("   cap levels: %s%s" % (caps, "" if len(caps) > 1 else "   <-- FAIL: need >=2"))
    if len(caps) < 2:
        fails.append("single cap level: %s" % caps)
    print()

    # ---- 4. warm-up cache sharing ------------------------------------------
    print("4. WARM-UP CACHE  (arms sharing a base_model_id share a trained model)")
    byid = collections.defaultdict(set)
    for r in runs:
        byid[r["base_model_id"]].add(r["arm"])
    groups = collections.defaultdict(set)
    for _bid, a in byid.items():
        groups[frozenset(a)] |= a
    for grp in sorted(groups, key=lambda g: sorted(g)):
        shared = sorted(grp)
        note = "share a warm-up" if len(shared) > 1 else "own warm-up"
        print("   %-46s %s" % (" + ".join(shared), note))
    # This section used to print the groups and ask a human to eyeball them --
    # narration, not an assertion, guarding occurrence 5 of the inert-flag
    # failure (clip and focal_clip hashing identically, so focal_clip silently
    # became a second clip).
    warmup_loss = collections.defaultdict(lambda: collections.defaultdict(set))
    for cfg in runs:
        warmup_loss[cfg["base_model_id"]][cfg["arm"]].add(
            json.dumps(cfg["hyperparams"].get("warmup_loss", "ce")))
    for bid, per_arm in sorted(warmup_loss.items()):
        losses = {v for vals in per_arm.values() for v in vals}
        if len(losses) > 1:
            fails.append(
                "base_model_id %s is shared by arms with DIFFERENT warm-up "
                "objectives %s -- one of them would silently load the other's "
                "trained model" % (bid, sorted(losses)))
    if any("base_model_id" in f for f in fails):
        print("   FAIL -- an arm shares a warm-up with a different objective")
    else:
        print("   OK -- every shared warm-up has one training objective")
    print("   (arms that share a warm-up must differ ONLY in the allocator)\n")

    # ---- 4b. one dataset definition ----------------------------------------
    print("4b. DATASET  (every arm must cap the same class(es) on the same data)")
    seen = collections.defaultdict(set)
    for cfg in runs:
        dcfg = cfg.get("dataset_config", {})
        for k in ("constrained_class", "data_dir", "num_classes", "group_column"):
            seen[(cfg["dataset_mode"], k)].add(json.dumps(dcfg.get(k)))
    for (ds, k), vals in sorted(seen.items()):
        if len(vals) > 1:
            fails.append("%s: %s differs across runs: %s" % (ds, k, sorted(vals)))
            print("   FAIL %s %-18s %s" % (ds, k, sorted(vals)))
        else:
            print("   OK   %s %-18s %s" % (ds, k, list(vals)[0]))
    print()

    # ---- 4c. one numeric regime --------------------------------------------
    rt = collections.defaultdict(set)
    for cfg in runs:
        r = (cfg.get("results") or {}).get("runtime") or {}
        if r:
            rt[(r.get("gpu_name"), r.get("amp_dtype"), r.get("grad_scaler"))
               ].add(cfg["arm"])
    if rt:
        print("4c. NUMERIC REGIME  (FP16+scaler SKIPS an overflowing optimizer")
        print("    step and BF16 does not, so the same config applies a")
        print("    different number of steps on the two servers)")
        for k, v in sorted(rt.items(), key=lambda kv: str(kv[0])):
            print("   %-44s %s" % ("gpu=%s amp=%s scaler=%s" % k,
                                   " ".join(sorted(v))))
        if len(rt) > 1:
            fails.append("MIXED NUMERIC REGIMES: %s" % sorted(map(str, rt)))
            print("   FAIL -- arms were run under different AMP regimes")
        else:
            print("   OK -- one regime across every run that has finished")
        print()

    # ---- 5. one code version ------------------------------------------------
    print("5. CODE VERSION  (arms generated by different code are not comparable)")
    versions = collections.defaultdict(set)
    for cfg in runs:
        versions[cfg.get("code_version", "unknown")].add(cfg["arm"])
    for v in sorted(versions):
        print("   %-24s %s" % (v, " ".join(sorted(versions[v]))))
    if len(versions) > 1:
        fails.append("MIXED CODE VERSIONS: %s -- some arms were generated by "
                     "different code than others" % sorted(versions))
        print("   FAIL -- regenerate the whole campaign from one commit")
    elif "unknown" in versions or "-dirty" in list(versions)[0]:
        print("   WARN -- uncommitted changes; the campaign is not reproducible "
              "from a SHA")
    else:
        print("   OK -- every arm generated by one commit")
    print()

    if fails:
        print("PARITY FAILED:")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("PARITY OK -- this campaign is a fair comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
