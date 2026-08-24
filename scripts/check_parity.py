"""Prove a campaign compares apples to apples, BEFORE it runs.

Every retraction in this project traces to two arms differing in something other
than the thing under test: unequal compute (worth 7-9 pp), an unequal
lr_constraint (worth 16 pp), a cap level present for one arm and not another, or
two arms silently sharing a cached warm-up so one of them was never really run.
This checks all four on the generated configs.

Two of them are NOT questions about whether the arms agree, and reading them
that way is how both stayed invisible for months. `lr_constraint` agreeing at
5e-6 on every arm IS the LR trap (2b), and `constraint_grad_mode` agreeing at
`clip` on every arm is exactly the ~20x delivered-dose gap of FRAMEWORK 1b-pre
finding (2), because the clip DELIVERS min(raw, clip) over natural gradient
scales that are orders of magnitude apart (2c). Both are checked against the
protocol, not against the other arms.

    python -m scripts.check_parity results/<campaign>

Exit code 1 if anything fails, so it can gate a launch.
"""
import collections
import glob
import io
import json
import os
import sys

# knobs that must be IDENTICAL across every arm; if they differ, the comparison
# measures the knob and not the method
SHARED_KEYS = ["lr", "lr_constraint", "dropout", "batch_size", "pretrained",
               "class_weighted_ce", "constraint_chunk_size", "inference_chunk_size",
               "stable_count_threshold",
               # measured at -0.0351 AP within-run: if one arm restores its best
               # feasible checkpoint and another keeps its final model, the delta
               # between them is the restore, not the method
               "enable_checkpoint_restore",
               # protocol.yml calls this "THE treatment dose... the only dose
               # axis the protocol admits", and it was the one knob this gate
               # did not check. Dormant today -- gen_campaign has no flag to
               # vary it, so every block gets 1.0 -- but the sweep protocol.yml
               # recommends (0.3 / 1.0 / 3.0) would have gone ungated.
               "constraint_grad_clip",
               # ...and the three knobs that decide what actually HAPPENS to
               # that gradient, which were not checked at all.
               # `constraint_grad_clip` alone says nothing: `clip` vs
               # `normalize` decides whether the clip's ceiling is also its
               # floor, `sgd` vs `shared` sends the same-norm gradient through
               # two different optimizers, and fp32 decides whether an epoch can
               # be dropped to a non-finite gradient. All three live in the
               # SHARED `constraint_phase` block, so no arm may override them.
               #
               # `constraint_random_direction` is deliberately NOT here: it is
               # ARM-DEFINING. `tralo_coin` IS the arm whose constraint step is
               # a random vector of the same norm, so requiring it to agree
               # across arms would refuse every campaign carrying the coin
               # control -- the one arm that answers "did the direction matter".
               "constraint_grad_mode", "constraint_step_rule",
               "constraint_fp32"]

# Trained methodologies whose constraint gradients live on natively different
# scales. See _check_dose below.
TRAINED_METHODOLOGIES = {"tralo", "fioretto_ldf", "hounie_rcl", "fioretto_alm",
                         "select"}


def _check_lr_trap(runs, fails):
    """`lr_constraint` must EQUAL `lr`, not merely agree across the arms.

    THE GATE THAT WAS MISSING, and the one this file's own docstring already
    claimed. Section 2 checks that each key holds one value across the arms --
    which an LR-trapped campaign satisfies perfectly: every arm carries
    lr 1e-4 and every trained arm carries lr_constraint 5e-6. That campaign
    printed "PARITY OK -- this campaign is a fair comparison".

    It is not a small knob. The trained arms build their constraint-phase
    optimizer with `lr_constraint` and force every param group onto it, so ALL
    29 cross-entropy epochs of a trained arm run at `lr_constraint` while the
    clipper's 30 warm-up epochs run at `lr`. Unequal, the comparison is one
    epoch of matched training against 29 of detuned training -- which is what
    fabricated a -16.7 pp finding that was -1.7 pp once equalized, and why the
    protocol says lr_constraint MUST equal lr.

    `gen_campaign` refuses this at generation, but the 2,972 trapped pairs in
    the provenance archive were never written by today's generator, and a
    hand-edited or resumed config never passes through it at all.
    """
    print("2b. LEARNING RATE  (lr_constraint MUST EQUAL lr -- not merely agree)")
    bad = collections.defaultdict(set)
    seen = 0
    for cfg in runs:
        hp = cfg["hyperparams"]
        if "lr_constraint" not in hp or "lr" not in hp:
            continue
        seen += 1
        if hp["lr_constraint"] != hp["lr"]:
            bad[(json.dumps(hp["lr"]), json.dumps(hp["lr_constraint"]))].add(
                cfg["arm"])
    if not seen:
        print("   --   no arm carries both keys (post-hoc-only campaign)")
    elif not bad:
        print("   OK   every trained arm trains at one learning rate")
    else:
        for (lr, lrc), arms_ in sorted(bad.items()):
            fails.append("THE LR TRAP: lr=%s but lr_constraint=%s on %s -- the "
                         "trained arms run 29 of their 30 epochs at the wrong "
                         "learning rate" % (lr, lrc, " ".join(sorted(arms_))))
            print("   FAIL lr=%s vs lr_constraint=%s on %s"
                  % (lr, lrc, " ".join(sorted(arms_))))
    print()


def _check_dose(runs, fails):
    """Under `constraint_grad_mode: clip` the arms are NOT at the same dose.

    `finish_constraint_step` delivers `min(raw_norm, constraint_grad_clip)`, and
    the trained arms' natural gradient scales are orders of magnitude apart by
    construction: `hounie_rcl` divides its primal violation by n_test / N_g to
    match its own dual, `fioretto_ldf` and `fioretto_alm` sum it, and `tralo`
    weights a bounded penalty. Measured on one warm-up model with every config
    saying `constraint_grad_clip: 1.0`, the raw norms ran 0.005-0.11 (hounie),
    0.64-1826 (tralo) and 17,667-80,827 (fioretto) -- so hounie delivered its
    raw ~0.05-norm step while the others delivered a unit one.

    That is a ~20x dose difference between arms which no config gate could see,
    because every config says 1.0. `normalize` rescales instead of capping, so
    the step size becomes a protocol constant and what differs between arms is
    DIRECTION -- which is what the comparison is supposed to be about.

    Scoped to campaigns holding MORE THAN ONE trained methodology: with a single
    trained family the delivered dose is whatever that family produces, and it
    is constant across everything being compared.
    """
    modes = collections.defaultdict(set)
    fams = collections.defaultdict(set)
    for cfg in runs:
        hp = cfg["hyperparams"]
        meth = cfg.get("methodology")
        if meth not in TRAINED_METHODOLOGIES or not hp.get("constraint_epochs"):
            continue
        # The zero-dose siblings never form a constraint gradient, so they are
        # at the same (zero) dose whatever the mode is, and a campaign that
        # holds only nulls has nothing to be unmatched. Same `_null` convention
        # gen_campaign's `all` uses.
        if cfg["arm"].endswith("_null"):
            continue
        modes[hp.get("constraint_grad_mode", "clip")].add(cfg["arm"])
        fams[meth].add(cfg["arm"])
    if len(fams) < 2:
        return
    print("2c. CONSTRAINT DOSE  (%d trained methodologies in one campaign)"
          % len(fams))
    for meth in sorted(fams):
        print("   %-14s %s" % (meth, " ".join(sorted(fams[meth]))))
    clipped = modes.get("clip", set())
    if clipped:
        fails.append(
            "UNMATCHED CONSTRAINT DOSE: constraint_grad_mode=clip with %d "
            "trained methodologies (%s). The clip DELIVERS min(raw, %s) and "
            "the arms' natural gradient scales differ by orders of magnitude "
            "-- measured ~20x apart with every config saying the same clip. "
            "Regenerate with --constraint-grad-mode normalize, or run one "
            "family per campaign."
            % (len(fams), " ".join(sorted(clipped)),
               sorted({json.dumps(c["hyperparams"].get("constraint_grad_clip"))
                       for c in runs
                       if "constraint_grad_clip" in c["hyperparams"]})))
        print("   FAIL constraint_grad_mode=clip -- delivered step is "
              "min(raw, clip), and")
        print("        raw differs by orders of magnitude between these "
              "families, so the")
        print("        arms differ in DOSE as well as direction. Use "
              "--constraint-grad-mode normalize.")
    else:
        print("   OK   constraint_grad_mode=%s -- every arm delivers the same "
              "step norm" % " ".join(sorted(modes)))
    print()


def load(root):
    runs = []
    for p in glob.glob(os.path.join(root, "**", "config.json"), recursive=True):
        c = json.load(open(p))
        # Whether this run has actually produced anything. A campaign is
        # normally read while it is still running, and without this the
        # coverage check below reports PARITY FAILED on every campaign until
        # its very last run lands -- which makes a real coverage failure
        # indistinguishable from "not finished yet", and a gate that cries
        # wolf on every invocation is a gate that stops being read.
        c["_done"] = os.path.exists(
            os.path.join(os.path.dirname(p), "final_predictions.csv"))
        runs.append(c)
    return runs


def _identity_keys():
    """The warm-up-identity keys, from protocol.yml -- the same list
    compute_base_model_id and audit_config use, so adding a key there extends
    this gate too rather than leaving it behind."""
    import yaml
    try:
        proto = yaml.safe_load(io.open(os.path.join("configs", "protocol.yml"),
                                       encoding="utf-8"))
        keys = proto.get("warmup_identity_keys")
        if keys:
            return list(keys)
    except Exception as exc:
        # NOT `pass`. The narrowed fallback below is deliberate; announcing it
        # is mandatory. A parity gate that quietly checks ONE key instead of the
        # protocol's full list still prints PARITY OK, and this project has been
        # burned four times by a check that passed because it stopped looking.
        sys.stderr.write(
            "WARNING check_parity: configs/protocol.yml unreadable (%s: %s). "
            "Falling back to checking ONLY 'warmup_loss' -- this gate is now "
            "much weaker than PARITY OK implies.\n"
            % (type(exc).__name__, exc))
    # If the protocol cannot be read, check the one key that is known to have
    # collided rather than silently checking nothing.
    return ["warmup_loss"]


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

    # ---- 2b/2c. the two knobs whose EQUALITY ACROSS ARMS is not the point ----
    # Section 2 asks "do the arms agree on this value". These two ask "is the
    # agreed value itself a fair comparison" -- lr_constraint agreeing at 5e-6
    # on every arm IS the LR trap, and constraint_grad_mode agreeing at `clip`
    # on every arm is exactly how the ~20x dose gap stayed invisible.
    _check_lr_trap(runs, fails)
    _check_dose(runs, fails)

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
    # The PLAN is every config on disk; DONE is the subset with predictions.
    # An unbalanced plan is a real parity failure. A balanced plan that is
    # unevenly finished is just a campaign in flight, and is reported as such.
    done = collections.defaultdict(set)
    for r in runs:
        if r.get("_done"):
            key = (r["dataset_mode"], r["model_name"], r["constraint_tag"],
                   str(r.get("dataset_config", {}).get("constrained_class")),
                   r["hyperparams"]["seed"])
            done[r["arm"]].add(key)
    ref = cells[arms[0]]
    in_flight = sum(1 for r in runs if not r.get("_done"))
    for arm in arms:
        d = cells[arm]
        mark = "OK  " if d == ref else "FAIL"
        if d != ref:
            fails.append("%s plans %d cell-seeds, %s plans %d"
                         % (arm, len(d), arms[0], len(ref)))
        print("   %s %-12s %d planned, %d finished"
              % (mark, arm, len(d), len(done[arm])))
    if in_flight:
        print("   IN FLIGHT: %d of %d runs have no predictions yet. Coverage is"
              % (in_flight, len(runs)))
        print("      judged on the PLAN (configs on disk); uneven FINISHED")
        print("      counts above are progress, not a parity failure.")
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
    # EVERY warm-up-identity key, read from protocol.yml -- not just
    # warmup_loss, which was a point-fix for the one collision that happened
    # (clip and focal_clip hashed identically, so focal_clip silently loaded
    # clip's model and became a second clip). Two arms differing only in
    # dropout or focal_alpha would have shared a model and passed this gate.
    keys = _identity_keys()
    shared = collections.defaultdict(lambda: collections.defaultdict(set))
    for cfg in runs:
        hp = cfg["hyperparams"]
        for k in keys:
            shared[cfg["base_model_id"]][k].add(json.dumps(hp.get(k, None)))
    collided = False
    for bid, per_key in sorted(shared.items()):
        for k, vals in sorted(per_key.items()):
            if len(vals) > 1:
                collided = True
                fails.append(
                    "base_model_id %s is shared by arms with DIFFERENT %s "
                    "%s -- one of them would silently load the other's trained "
                    "model" % (bid, k, sorted(vals)))
    if collided:
        print("   FAIL -- an arm shares a warm-up that another arm trained "
              "differently")
    else:
        print("   OK -- every shared warm-up agrees on all %d identity key(s)"
              % len(keys))
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
    # `run_code_version` is stamped by src/experiments/runner at EXECUTION time
    # and describes the commit that produced the weights; `code_version` is
    # stamped by gen_campaign when the config is CREATED and is never revisited,
    # so it describes the generator. Prefer the runner's wherever a run has
    # finished -- a campaign generated once and resumed after a training-file
    # change carries ONE code_version across both halves, which is this gate
    # passing on the drift it exists to catch.
    versions = collections.defaultdict(set)
    unstamped = set()
    for cfg in runs:
        rcv = cfg.get("run_code_version")
        if not rcv:
            unstamped.add(cfg["arm"])
        versions[rcv or cfg.get("code_version", "unknown")].add(cfg["arm"])
    for v in sorted(versions):
        print("   %-24s %s" % (v, " ".join(sorted(versions[v]))))
    if unstamped:
        print("   NOTE -- no run_code_version on: %s"
              % " ".join(sorted(unstamped)))
        print("           those arms are compared on the GENERATOR's commit, "
              "which is")
        print("           written once and cannot see a code change landed "
              "mid-campaign.")
        print("           Runs that have not started yet always read this way; "
              "for a")
        print("           FINISHED campaign, full_panel repeats the check on "
              "the stamp")
        print("           the runner writes.")
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
