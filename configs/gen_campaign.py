"""THE campaign generator. It reads `configs/protocol.yml` and contains no
experimental constants of its own.

If a number decides what an experiment does, it is in the YAML. This file only
assembles: it picks each arm's blocks, splits the epoch budget, hashes the
warm-up identity, and refuses to emit a campaign that violates the protocol.

    python -m configs.gen_campaign --root results/<name> \\
        --datasets dermmnist tissuemnist --models MobileNetV3 \\
        --caps L30_G30 L50_G50 --arms all

Caps
----
`L30_G50` sets the LOCAL (per-group) cap to 30% and the GLOBAL cap to 50% of the
constrained class's true test-set count. The two are independent, so an
asymmetric sweep is just a list of tags. Both are turned into integer budgets by
`src/training/constraints.py` against the actual test labels -- the percentage
only standardizes how hard the cap binds across datasets.

Constrained classes
-------------------
`constrained_class` may be a single index or a list, per dataset in the YAML, and
`--constrained-class` overrides it for one run. Indices are validated against the
dataset's `num_classes`, because a cap on a class that does not exist is silently
skipped by the loss.
"""
import argparse
import hashlib
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.gitver import git_version   # stdlib-only: no torch here  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "protocol.yml")


def load_protocol(path=PROTOCOL_PATH):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _null_of(P, arm):
    """The zero-dose sibling for an arm.

    Usually `<arm>_null`, but an arm may name another arm's null when the two
    differ only in something the null zeroes out anyway -- `tralo_margin` and
    `tralo` share `tralo_null` because at lambda 0 no constraint gradient is
    formed and the choice of soft count is inert. Without this lookup the gate
    below silently skips any arm whose name is not `<something>_null` away from
    its control, i.e. exactly the new arms that most need one.
    """
    return P["arms"][arm].get("null_sibling", arm + "_null")


def count_control_arms(P):
    """Arms flagged `count_control` in the YAML: the RESEED floor.

    A trained arm writes a per-epoch capped-class count, and that trajectory is
    what every "the constraint moved the count by N" claim is read out of.
    Measured 2026-08-22 and independently verified, turning the constraint on
    moves that count by RMS 75-95 items while merely re-randomising the RNG
    stream of two pure-CE runs moves it 83-95 -- 0.90-1.00x. So the trajectory
    is not readable without the reseed arm beside it, and `validate` refuses a
    campaign that has one without the other.

    Flagged in the YAML rather than matched on a name, for the same reason
    `null_sibling` exists: `_null_of` found its control by appending "_null",
    and `tralo_margin` -- the arm that most needed one -- silently resolved to
    an arm that does not exist while the gate said nothing.
    """
    return {a for a, spec in P["arms"].items() if spec.get("count_control")}


def resolve_block(P, name):
    """A block name is either a top-level section or an entry under `blocks`."""
    if name in P.get("blocks", {}):
        return P["blocks"][name]
    if name in P:
        return P[name]
    raise KeyError("protocol.yml: unknown block %r" % name)


def build_hyperparams(P, arm_spec, seed):
    """Assemble exactly the keys this arm's methodology reads, plus the contract
    keys that scripts/check_parity.py verifies on every arm."""
    total = P["protocol"]["total_epochs"]
    trained_warmup = P["protocol"]["trained_warmup"]
    posthoc = arm_spec["phase"] == "posthoc"

    hp = dict(P["core"])
    for name in arm_spec.get("blocks") or []:
        hp.update(resolve_block(P, name))
    hp["seed"] = seed
    hp["warmup_epochs"] = total if posthoc else trained_warmup
    hp["constraint_epochs"] = 0 if posthoc else total - trained_warmup
    return hp


def compute_base_model_id(P, model_name, hp, dataset_mode, dc):
    """Identity of the WARM-UP-trained model, so arms that share a warm-up share
    its cache and arms that do not, do not.

    The key is built from `warmup_identity_keys` in the YAML plus the dataset
    identity. Anything that changes what the warm-up optimizes must be listed
    there, or a second arm silently loads the first one's model -- which has
    happened four times in this project.
    """
    key = {"model_name": model_name,
           "dataset_mode": dataset_mode,
           "data_dir": dc["data_dir"],
           "num_classes": dc["num_classes"]}
    for k in P["warmup_identity_keys"]:
        if k in hp:
            key[k] = hp[k]
    h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return "%s_%s_%s" % (model_name, dataset_mode, h)


def code_version():
    """The commit that WRITES this config -- not the one that runs it.

    A config is stamped once, here, and `main()` never revisits a config it has
    already written. So this records generation time and nothing else: land a
    change to a training file half way through a campaign and every config
    still carries this value. The commit that produced the WEIGHTS is stamped
    by `src/experiments/runner.py` as `run_code_version`, and that is the one
    the provenance gates read when it is present.
    """
    return git_version()


def _cls_tag(dc):
    """Capped class(es) as a filename-safe tag: 4 -> "4", [4, 5] -> "4-5"."""
    c = dc["constrained_class"]
    return "-".join(str(x) for x in (c if isinstance(c, list) else [c]))


def _zero_ceilings(P, dataset, local_pct):
    """(zero per-group ceilings, total cells) for a dataset's capped classes.

    Returns (0, 0) when the slice is not present on this machine -- campaigns
    are generated on laptops as well as on the server, and a missing dataset
    must not crash generation. It reports nothing rather than guessing.
    """
    try:
        import pandas as pd
        from src.training.constraints import (compute_local_constraints,
                                              normalize_constrained_classes)
        dc = P["datasets"][dataset]
        meta = os.path.join(dc["data_dir"], "test_meta.csv")
        if not os.path.exists(meta):
            return 0, 0
        te = pd.read_csv(meta)
        classes = normalize_constrained_classes(dc["constrained_class"])
        L = compute_local_constraints(te, "label", local_pct,
                                      dc["group_column"],
                                      constrained_class=classes,
                                      num_classes=dc["num_classes"])
        cells = [L[g][c] for g in L for c in classes]
        return sum(1 for v in cells if v == 0), len(cells)
    except Exception:
        return 0, 0


TASK_WINDOWS_PATH = os.path.join(HERE, "task_windows.yml")


def load_task_windows(path=TASK_WINDOWS_PATH):
    """The MEASURED K/n windows in which a cap poses a question, or None."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def in_window(ratio, lo, hi, tol=0.0):
    """Is this K/n inside a measured task window? Pure, so it is testable with
    no dataset on disk."""
    return (lo - tol) <= ratio <= (hi + tol)


def _effective_budgets(P, dataset, lp, gp):
    """{class: (K_eff, n_true)} for one cap, or None if the slice is absent.

    K_eff = min(global budget, SUM of the per-group local budgets) -- what the
    allocator can actually emit. Reading one scope alone has twice called a cap
    binding when the other scope was the one that bound.

    Returns None ONLY when the slice is genuinely not on this machine, because
    campaigns are generated on laptops as well as on the server. A slice that
    IS present and cannot be read raises: the first version of this function
    read labels from `test_labels.npy`, which does not exist on the laptop
    where the meta does, and a bare `except` turned the whole gate into a
    silent no-op that still printed nothing and refused nothing.
    """
    import numpy as np
    import pandas as pd
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints,
                                          normalize_constrained_classes)
    dc = P["datasets"][dataset]
    meta = os.path.join(dc["data_dir"], "test_meta.csv")
    if not os.path.exists(meta):
        return None
    te = pd.read_csv(meta)
    if "label" not in te.columns:
        npy = os.path.join(dc["data_dir"], "test_labels.npy")
        if not os.path.exists(npy):
            sys.exit("REFUSED: %s has a test_meta.csv with no `label` column "
                     "and no test_labels.npy beside it, so the task window "
                     "cannot be checked. Fix the slice rather than "
                     "generating an ungated campaign." % dataset)
        te["label"] = np.load(npy)
    y = te["label"].to_numpy()
    classes = normalize_constrained_classes(dc["constrained_class"])
    G = compute_global_constraints(te, "label", gp,
                                   constrained_class=classes,
                                   num_classes=dc["num_classes"])
    L = compute_local_constraints(te, "label", lp, dc["group_column"],
                                  constrained_class=classes,
                                  num_classes=dc["num_classes"])
    out = {}
    for c in classes:
        c = int(c)
        lsum = sum(int(L[g][c]) for g in L)
        out[c] = (min(int(G[c]), lsum), int((y == c).sum()))
    return out


def task_window_gate(P, args, resolved):
    """REFUSE a campaign whose caps pose no question. FRAMEWORK 2(z16), 2(z17).

    A cap outside the measured window cannot distinguish any two methods: the
    top-K is already perfect, or the cut sits at p ~ 1, or the cap evicts almost
    nothing. 24 of 24 (backbone x class x cap) cells at L20/L30/L50 on iwildcam
    are outside it, on every backbone the paper claims -- which is the best
    single explanation on record for why so many arms tied.

    Silent when the slice is absent (nothing to measure) and a WARNING, not a
    refusal, for a (dataset, backbone) with no measured row: an unmeasured
    backbone is an unknown, not a known non-task.
    """
    TW = load_task_windows()
    if not TW:
        return
    tol = float(TW.get("meta", {}).get("tolerance", 0.0))
    rows, bad, unknown, measured = [], [], set(), False
    for ds in args.datasets:
        wds = (TW.get("windows") or {}).get(ds) or {}
        for model in args.models:
            w = wds.get(model)
            for tag in args.caps:
                lp, gp = cap_pair(tag)
                eff = _effective_budgets(P, ds, lp, gp)
                if eff is None:
                    continue
                if not w:
                    unknown.add("%s/%s" % (ds, model))
                    continue
                measured = True
                for c, (K, n) in sorted(eff.items()):
                    ratio = (K / float(n)) if n else 0.0
                    lo, hi = w["class"][c]
                    ok = in_window(ratio, lo, hi, tol)
                    rows.append((model, tag, c, K, n, ratio, lo, hi, ok))
                    if not ok:
                        bad.append((model, tag, c, ratio, lo, hi))
    for u in sorted(unknown):
        print("  !! NO MEASURED TASK WINDOW for %s, so it is NOT gated. "
              "Measure it with" % u)
        print("     python -m scripts.task_window --glob <unconstrained runs>")
        print("     before trusting any null from it. An unmeasured backbone "
              "is an unknown,")
        print("     not a known task.")
    if not measured:
        return
    print("  TASK WINDOW (FRAMEWORK 2(z16)/2(z17)) -- does each cap pose a "
          "question?")
    print("    %-13s %-13s %4s %6s %6s %7s %12s  %s"
          % ("model", "cap", "cls", "K", "n", "K/n", "window", ""))
    for model, tag, c, K, n, ratio, lo, hi, ok in rows:
        print("    %-13s %-13s %4d %6d %6d %7.3f   %4.2f-%4.2f  %s"
              % (model, tag, c, K, n, ratio, lo, hi,
                 "in" if ok else "** OUTSIDE **"))
    if bad and not getattr(args, "allow_nontask", False):
        lines = ["REFUSED: %d of %d (model, cap, class) cell(s) sit OUTSIDE "
                 "the measured task window." % (len(bad), len(rows))]
        for model, tag, c, ratio, lo, hi in bad:
            lines.append("  %s %s class %d: K/n=%.3f, window %.2f-%.2f"
                         % (model, tag, c, ratio, lo, hi))
        lines += [
            "",
            "  Outside the window the cap either forces out almost nothing, "
            "leaves NO errors",
            "  inside K, or cuts at p@K ~ 1. None of those can distinguish "
            "two methods, so the",
            "  cell measures the ABSENCE of a question and its null is not "
            "evidence about any",
            "  method. Measured on all four backbones: 24 of 24 cells at "
            "L20/L30/L50 are outside.",
            "",
            "  Pick caps inside the window, PER CLASS where the classes' "
            "windows do not overlap:",
            "    --caps L80-100_G95 L70-90_G95",
            "  or pass --allow-nontask and say in the write-up that the "
            "campaign cannot",
            "  distinguish its arms by construction.",
        ]
        sys.exit("\n".join(lines))
    if bad:
        print("  !! --allow-nontask: %d cell(s) pose NO question and are "
              "generated anyway." % len(bad))
        print("     Their nulls are the absence of a measurement, not a "
              "result. Say so.")


def _gate_self_test():
    """Gate the gate, in BOTH directions. Never claims a pass it did not run."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-66s %s" % (name, "PASS" if cond else "FAIL"))

    TW = load_task_windows()
    check("configs/task_windows.yml loads", bool(TW))
    if not TW:
        return 1
    tol = float(TW["meta"]["tolerance"])
    w = TW["windows"]["iwildcam"]["MobileNetV3"]["class"]
    lo2, hi2 = w[2]
    lo7, hi7 = w[7]
    check("L30 class 2 (K/n=0.30) is OUTSIDE MobileNetV3 window %.2f-%.2f"
          % (lo2, hi2), not in_window(0.30, lo2, hi2, tol))
    check("L20 class 7 (K/n=0.20) is OUTSIDE MobileNetV3 window %.2f-%.2f"
          % (lo7, hi7), not in_window(0.20, lo7, hi7, tol))
    check("LIVENESS: taskwin1 class 2 (K/n=0.800) is INSIDE",
          in_window(0.800, lo2, hi2, tol))
    check("LIVENESS: taskwin1 class 7 (K/n=0.950) is INSIDE",
          in_window(0.950, lo7, hi7, tol))
    check("a window EDGE is inside, not excluded by float noise",
          in_window(hi2, lo2, hi2, tol) and in_window(lo7, lo7, hi7, tol))
    check("every backbone the paper claims has a measured row",
          set(TW["windows"]["iwildcam"]) >=
          {"ViTB16", "MobileNetV3", "MobileNetV2", "RegNetY400MF"})

    P = load_protocol()
    dc = P["datasets"]["iwildcam"]
    have = os.path.exists(os.path.join(dc["data_dir"], "test_meta.csv"))
    if not have:
        print("  %-66s %s" % ("end-to-end refusal (needs the iwildcam slice)",
                              "SKIPPED -- slice not on this machine"))
    else:
        class A(object):
            datasets = ["iwildcam"]
            models = ["MobileNetV3"]
            allow_nontask = False
            caps = ["L20_G50", "L30_G50"]
        try:
            task_window_gate(P, A, None)
            check("end to end: an L20/L30 campaign is REFUSED", False)
        except SystemExit:
            check("end to end: an L20/L30 campaign is REFUSED", True)
        A.caps = ["L80-100_G95", "L70-90_G95"]
        try:
            task_window_gate(P, A, None)
            check("LIVENESS end to end: taskwin1 caps are ALLOWED", True)
        except SystemExit:
            check("LIVENESS end to end: taskwin1 caps are ALLOWED", False)
    print("")
    print("ALL PASS" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


def cap_pair(tag):
    """'L30_G50' -> [0.30, 0.50]: (local, global), independent by construction.

    PER-CLASS LOCAL CAPS: 'L80-100_G95' -> [[0.80, 1.00], 0.95]. The list is
    read POSITIONALLY against `constrained_class`, so L80-100 with classes
    [2, 7] caps class 2 at 80% and class 7 at 100%.

    WHY (FRAMEWORK 2(z16)). A cap poses a question only where it forces out
    >= 10 predictions, leaves errors inside K, and sits at p@K < 0.99. On
    iwildcam those windows are class 2 at K/n 0.70-0.80 and class 7 at
    0.90-1.00, and on MobileNetV3 they DO NOT OVERLAP -- so a single fraction
    could not express a valid two-class experiment at all, and every
    L20/L30/L50 campaign ran on a NON-TASK.

    A percentage ABOVE 100 is legal and is not degenerate: class 7's model
    predicts 490 against 456 true, so K/n = 1.00 still evicts 34.
    """
    try:
        local, glob = tag.split("_")
        if local[0] != "L" or glob[0] != "G":
            raise ValueError
        parts = local[1:].split("-")
        loc = ([int(x) / 100.0 for x in parts] if len(parts) > 1
               else int(parts[0]) / 100.0)
        return [loc, int(glob[1:]) / 100.0]
    except (ValueError, IndexError):
        sys.exit("bad cap tag %r -- expected L<pct>_G<pct> (e.g. L30_G50) or "
                 "per-class L<pct>-<pct>_G<pct> (e.g. L80-100_G95)" % tag)


def resolve_datasets(P, args):
    """Dataset config per dataset with `--constrained-class` already applied.

    Resolved BEFORE validation, not inside the emit loop: validating the YAML
    default while emitting the override let `--constrained-class 9` through on a
    7-class dataset, where the loss would have skipped the cap silently.
    """
    out = {}
    for ds in args.datasets:
        dc = dict(P["datasets"][ds])
        if args.constrained_class is not None:
            dc["constrained_class"] = (args.constrained_class[0]
                                       if len(args.constrained_class) == 1
                                       else list(args.constrained_class))
        out[ds] = dc
    return out


def validate(P, args, resolved, arms):
    if len(set(args.caps)) < 2:
        sys.exit("REFUSED: at least two cap levels are required. A claim from cells "
                 "sharing one cap level has been retracted three times.")
    # Two distinct TAGS can be the same EXPERIMENT. The binding budget is
    # min(global_K, sum of local_K), so L30_G30 and L30_G50 are identical
    # wherever the global cap is already slack -- and duplicate runs manufacture
    # significance: 8 pairs where 4 duplicate the other 4 gave p=0.0078 when the
    # honest n=4 has an exact floor of 0.125.
    locals_ = {}
    for tag in args.caps:
        lp, gp = cap_pair(tag)
        # A per-class local cap is a LIST and lists are unhashable. Key on a
        # tuple so the duplicate-experiment check below works for both forms
        # rather than crashing on the per-class one.
        key = tuple(lp) if isinstance(lp, (list, tuple)) else lp
        locals_.setdefault(key, []).append((tag, gp))
    for lp, tags in locals_.items():
        if len({gp for _t, gp in tags}) > 1 and len(tags) > 1:
            print("NOTE: caps %s share local %d%%. They are the SAME experiment "
                  "wherever the global cap is slack (it can only bind BELOW the "
                  "sum of the local caps). Run `python -m scripts.verify_caps "
                  "--caps %s` on the real slices before trusting them as two "
                  "levels."
                  % ([t for t, _g in tags],
                     ("/".join(str(int(x * 100)) for x in lp)
                      if isinstance(lp, tuple) else int(lp * 100)),
                     " ".join(t for t, _g in tags)))
    # WHICH SCOPE ACTUALLY BINDS, stated per tag. Local caps are per-GROUP
    # ceilings, so their sum is `L * total_true` against the global's
    # `G * total_true` -- the comparison is L vs G and nothing else.
    #
    # This exists because the project has now made the SAME mistake in both
    # directions. Until 2026-08-18 the global cap had never bound (`G >= L`
    # throughout) and every result was a local-cap result. The fix was "sweep
    # `G < L`", which worked -- and silently made the LOCAL scope inert:
    # `results/dualbar2` ran L50_G20 and L50_G40, both `G < L`, and
    # `lp_fallback_used` came back False on all 50 completed runs with 0
    # candidates, i.e. a local ceiling was never once the binding constraint.
    # Neither direction was noticed at generation time, twice, because nothing
    # printed this line.
    binds = {}
    for tag in args.caps:
        lp, gp = cap_pair(tag)
        # A per-class local cap has one fraction per capped class, so the
        # L-vs-G comparison is per class. Report the TIGHTEST, which is the
        # one that decides whether the local scope can bind at all.
        lp_list = list(lp) if isinstance(lp, (list, tuple)) else [lp]
        lp_cmp = max(lp_list)
        which = ("GLOBAL (local sum is %.1fx slack)" % (lp_cmp / gp) if gp < lp_cmp
                 else "LOCAL (global is %.1fx slack)" % (gp / lp_cmp) if gp > lp_cmp
                 else "IDENTICAL -- global exactly equals the local sum")
        binds.setdefault(which.split()[0], []).append(tag)
        print("  cap %-12s L=%s G=%d%%  ->  binding scope: %s"
              % (tag, "/".join("%d%%" % int(x * 100) for x in lp_list),
                 int(gp * 100), which))
        if len(lp_list) > 1:
            print("     ^ PER-CLASS local caps, read positionally against "
                  "constrained_class. FRAMEWORK 2(z16): the two capped classes "
                  "have task windows that do not overlap, so one fraction "
                  "cannot pose a question for both.")
        # 🛑 SUM-SLACKNESS DOES NOT IMPLY NON-BINDING. The line above is
        # pure arithmetic on the two percentages and was written against
        # dermmnist, where every per-group ceiling is positive. A ceiling of
        # ZERO binds absolutely, whatever the sum does -- and on a held-out-
        # camera dataset most cells are zero, because a species simply is not
        # at that camera. Reporting "local sum is 2.5x slack" there would call
        # the local scope inert in the one campaign where it does the most
        # work. So this reads the ACTUAL budgets rather than inferring them.
        for ds in args.datasets:
            zeros, total = _zero_ceilings(P, ds, lp)   # accepts either form
            if zeros:
                print("     ^ but %s has %d of %d per-group ceiling(s) at "
                      "K=0 for the capped class(es)." % (ds, zeros, total))
                print("       A ZERO CEILING BINDS regardless of sum slack, so "
                      "the LOCAL scope")
                print("       constrains the output at this cap too.")
    if len(binds) == 1 and len(args.caps) > 1:
        only = list(binds)[0]
        other = "L20_G50" if only == "GLOBAL" else "L50_G20"
        print("  !! EVERY cap in this campaign binds the %s scope. The other "
              "scope is carried" % only)
        print("     but slack, so nothing here tests it -- which is how the "
              "global cap went")
        print("     unmeasured until 2026-08-18 and the local one until "
              "2026-08-22.")
        print("     A binding LOCAL cap is not the same constraint: it fixes "
              "the DISTRIBUTION")
        print("     across groups, where a global cap fixes only the TOTAL. "
              "Add e.g. %s" % other)
        print("     to test the other scope, or say plainly that this campaign "
              "tests one.")

    # A SPEC refusal, so it comes before the hygiene ones below: a campaign
    # whose caps pose no question is not a campaign with a missing control,
    # it is a campaign with nothing to measure.
    task_window_gate(P, args, resolved)

    lr = P["core"]["lr"]
    lr_c = P["constraint_phase"]["lr_constraint"]
    if lr != lr_c:
        sys.exit("REFUSED: lr (%s) != lr_constraint (%s). Unequal learning rates "
                 "fabricated a -16.7pp finding that was -1.7pp once equalized." % (lr, lr_c))
    for ds, dc in resolved.items():
        classes = dc["constrained_class"]
        classes = classes if isinstance(classes, list) else [classes]
        if not classes:
            sys.exit("REFUSED: %s has no constrained class." % ds)
        if len(set(classes)) != len(classes):
            sys.exit("REFUSED: %s constrained_class %s repeats a class; the second "
                     "cap would overwrite the first." % (ds, classes))
        for c in classes:
            if not 0 <= int(c) < int(dc["num_classes"]):
                sys.exit("REFUSED: %s constrained_class %s is out of range for "
                         "num_classes=%s. A cap on a nonexistent class is silently "
                         "skipped by the loss." % (ds, c, dc["num_classes"]))
    # LAST, deliberately. The three refusals above name a defect in the campaign
    # SPEC; this one names a missing control, and a spec error must be reported
    # before a hygiene error or the user fixes the wrong thing.
    controls = count_control_arms(P)
    trained = sorted(a for a in arms if P["arms"][a].get("phase") == "trained")
    if trained and not controls:
        # No arm carries the flag at all. Refusing with an empty "Add: --arms"
        # would be a puzzle, and it is a different defect from a campaign that
        # merely forgot the control: the PROTOCOL has lost it.
        sys.exit(
            "REFUSED: this campaign holds trained arm(s) %s and "
            "configs/protocol.yml declares\n"
            "  no `count_control` arm at all, so there is nothing to bound "
            "their count trajectories\n"
            "  against. Restore the reseed control (FRAMEWORK section 13) "
            "rather than generating a\n"
            "  campaign whose counts cannot be read."
            % " ".join(trained))
    if trained and not (controls & set(arms)):
        sys.exit(
            "REFUSED: this campaign holds trained arm(s) %s and no reseed "
            "control.\n"
            "  A trained arm writes a per-epoch capped-class count, and that "
            "trajectory is what\n"
            "  'the constraint moved the count by N items' is read out of. "
            "Measured 2026-08-22\n"
            "  and independently verified: turning the constraint ON moves "
            "that count by RMS\n"
            "  75-95 items, and RESEEDING two pure-CE runs moves it 83-95. "
            "The constraint's\n"
            "  whole measurable footprint on the count is 0.90-1.00x a "
            "reseed, so without the\n"
            "  floor in THIS campaign no count trajectory is attributable -- "
            "the same argument\n"
            "  that puts both clippers in every campaign.\n"
            "  Add: --arms ... %s"
            % (" ".join(trained), " ".join(sorted(controls))))


def _apply_constraint_step(P, args):
    """Constraint-step knobs, into the SHARED block for the same reason.

    `normalize` exists because the arms were NOT getting the same dose: one
    absolute clip over natural gradient scales six orders of magnitude apart
    left hounie taking a ~0.05-norm step while tralo and fioretto took unit
    ones. Per-arm would recreate exactly the asymmetry it fixes.
    """
    if args.constraint_grad_mode is not None:
        P["constraint_phase"]["constraint_grad_mode"] = args.constraint_grad_mode
    if args.constraint_fp32 is not None:
        P["constraint_phase"]["constraint_fp32"] = bool(args.constraint_fp32)
    if args.constraint_step_rule is not None:
        P["constraint_phase"]["constraint_step_rule"] = args.constraint_step_rule
    if args.penalty_shape is not None:
        P["blocks"]["tralo"]["penalty_shape"] = args.penalty_shape
    if args.soft_count_mode is not None:
        P["blocks"]["tralo"]["soft_count_mode"] = args.soft_count_mode
    if args.cut_window_items is not None:
        P["blocks"]["tralo"]["cut_window_items"] = int(args.cut_window_items)
    if args.constraint_random_direction:
        P["constraint_phase"]["constraint_random_direction"] = True
        print("  CONSTRAINT DIRECTION RANDOMISED: this campaign is a CONTROL, "
              "not a method run.")
        print("      Same step norm, no information. If it scores like the real "
              "arm, the penalty")
        print("      contributed nothing a coin could not have.")
    if P["constraint_phase"].get("constraint_grad_mode") == "normalize":
        print("  CONSTRAINT GRAD NORMALIZED to %s for EVERY trained arm: the "
              "step size is now a" % P["constraint_phase"]["constraint_grad_clip"])
        print("      protocol constant, so what differs between arms is "
              "direction, not dose.")
    else:
        # Say it HERE too, so the generator and check_parity agree. `clip` is
        # the default only because it keeps every stored result reproducible;
        # it is not the mode a cross-family comparison may be read out of, and
        # `scripts.check_parity` REFUSES a multi-family campaign that carries
        # it. A generator that emits silently what the gate then rejects is the
        # tool contradicting itself, which this file already has a rule about.
        print("  CONSTRAINT GRAD MODE = clip: the delivered step is "
              "min(raw_norm, %s), and the"
              % P["constraint_phase"]["constraint_grad_clip"])
        print("      trained arms' natural gradient scales differ by orders of "
              "magnitude -- hounie")
        print("      divides its primal by n_test, fioretto and alm sum it. "
              "Measured on one")
        print("      warm-up model with every config saying the same clip: "
              "hounie 0.005-0.11,")
        print("      tralo 0.64-1826, fioretto 17,667-80,827. So under `clip` "
              "the arms differ in")
        print("      DOSE as well as direction. Fine for a ONE-FAMILY "
              "campaign; for more than")
        print("      one, add --constraint-grad-mode normalize or "
              "check_parity will refuse it.")
    shape = P["blocks"]["tralo"].get("penalty_shape", "rational_bounded")
    if shape != "rational_bounded":
        print("  PENALTY SHAPE = %s for every trained arm: this is NOT the "
              "manuscript's Eq. 4," % shape)
        print("      so results from it are not comparable to the stored "
              "corpus without saying so.")
    if P["constraint_phase"].get("constraint_step_rule") == "sgd":
        print("  CONSTRAINT STEP = PLAIN SGD for every trained arm: the step no "
              "longer inherits CE's")
        print("      Adam moments, so it points where the constraint points.")
    if P["constraint_phase"].get("constraint_fp32"):
        print("  CONSTRAINT PASS IN FP32: no loss scaler on the constraint "
              "step, so an epoch cannot be")
        print("      silently dropped to a non-finite gradient.")


def main():
    # Before the parser: --root and --datasets are required for a real
    # generation, and the self-test generates nothing.
    if "--self-test" in sys.argv:
        sys.exit(_gate_self_test())
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True)
    a.add_argument("--datasets", nargs="+", required=True, choices=sorted(P["datasets"]))
    a.add_argument("--models", nargs="+", default=[P["models"][0]], choices=P["models"])
    a.add_argument("--caps", nargs="+", default=["L30_G30", "L50_G50"],
                   help="L<local>_G<global>, independent; e.g. L30_G50")
    a.add_argument("--arms", nargs="+", default=["tralo"],
                   choices=sorted(P["arms"]) + ["all", "all+null"],
                   help="'all' runs the full panel: every baseline the paper claims")
    a.add_argument("--allow-nontask", action="store_true",
                   help="generate even where a cap sits OUTSIDE the "
                        "measured task window "
                        "(configs/task_windows.yml). The cells it lets "
                        "through cannot distinguish two methods by "
                        "construction, so their nulls are the absence "
                        "of a measurement -- say so in the write-up.")
    a.add_argument("--self-test", action="store_true",
                   help="gate the task-window gate in both directions and exit")
    a.add_argument("--constrained-class", nargs="+", type=int, default=None,
                   help="override the YAML's capped class(es) for every dataset; "
                        "one index or several for the coupled multi-class setting")
    a.add_argument("--constraint-grad-mode", choices=["clip", "normalize"],
                   default=None,
                   help="clip: cap the constraint gradient at "
                        "constraint_grad_clip (historical). normalize: rescale "
                        "it to EXACTLY that value, so every trained arm takes "
                        "the same-size constraint step. Measured: with one "
                        "absolute clip, hounie's gradient never reached it "
                        "(max 0.11 of 1.0) while fioretto's exceeded it by "
                        "80,000x -- a ~20x dose gap invisible to every gate.")
    a.add_argument("--soft-count-mode", choices=["sum", "margin", "uniform", "cut"],
                   default=None,
                   help="WHERE the count puts its gradient. `sum` is the "
                        "manuscript's count, whose per-item derivative p(1-p) "
                        "is largest where the model is unsure and ~zero at the "
                        "cut -- the only place a prediction can change. "
                        "`margin` keeps the count's VALUE and moves its "
                        "WEIGHT onto the decision boundary by softening the "
                        "ARGMAX. tralo only; the duals do not form this "
                        "count.")
    a.add_argument("--cut-window-items", type=int, default=None,
                   help="how many items sit inside the margin window, for "
                        "--soft-count-mode margin. The sigmoid width T is "
                        "DERIVED from it per class per epoch, because a fixed "
                        "T is not a fixed dose: measured on the stored "
                        "dermmnist evidence the T holding ~20 items spans "
                        "0.182 to 0.502 across seeds of ONE cell, and margins "
                        "grow through the constraint phase as CE converges. "
                        "This knob is dimensionless and cannot produce an "
                        "empty window.")
    a.add_argument("--penalty-shape",
                   choices=["rational_bounded", "linear", "squared"],
                   default=None,
                   help="rational_bounded is the manuscript's Eq. 4 and the "
                        "default. Its gradient VANISHES on the worst "
                        "violations, so with several capped scopes the "
                        "deepest violator gets the weakest pull. Measured "
                        "multi-class against a lambda=0 control, that shows "
                        "up as a SEE-SAW: every shape pushes one capped "
                        "class down and the other up, because the softmax "
                        "makes them compete and the starved class cannot "
                        "resist. Shape sets the see-saw SIZE -- class 2 "
                        "moved +197 under rational_bounded, +112 squared, "
                        "+86 linear -- but no shape reduced total excess. "
                        "It is a dial on the coupling, not a fix.")
    a.add_argument("--constraint-random-direction", action="store_true",
                   default=None,
                   help="THE CONTROL FOR WHETHER THE DIRECTION MATTERS. "
                        "Replaces the constraint gradient with a random "
                        "vector of the SAME norm, holding the dose and "
                        "removing only the information. Measured: the "
                        "constraint costs exactly 4 correct capped-class "
                        "predictions out of 89 at every one of three seeds "
                        "while the count trajectories behind them end at "
                        "57, 201 and 439 -- a constant loss from wildly "
                        "different paths. If a coin costs the same 4, no "
                        "shape or dose tuning will help.")
    a.add_argument("--constraint-step-rule", choices=["shared", "sgd"],
                   default=None,
                   help="shared: the constraint step goes through the same "
                        "Adam as CE, which retains only 0.009-0.017 of its "
                        "direction. sgd: p -= lr_constraint * g, recovering "
                        "the direction at the smallest step in the dose sweep. "
                        "Distinct from the rejected dedicated-Adam arm, whose "
                        "step was ~8,900x larger.")
    a.add_argument("--constraint-fp32", action="store_true", default=None,
                   help="evaluate the constraint pass in fp32 and bypass the "
                        "loss scaler. fioretto lost 10 of 29 constraint epochs "
                        "to non-finite gradients while reporting completed.")
    a.add_argument("--protocol", default=PROTOCOL_PATH, help="alternate protocol.yml")
    args = a.parse_args()
    # Reload FIRST: applying overrides and then replacing P discarded them.
    if args.protocol != PROTOCOL_PATH:
        P = load_protocol(args.protocol)
    _apply_constraint_step(P, args)

    # `all` deliberately EXCLUDES the zero-dose siblings. Adding them is a
    # compute decision -- four more trained arms is +27% on the canonical
    # campaign -- and protocol.yml says the same thing. Silently growing what
    # `all` costs because new arms were defined is the scope expansion this
    # project has a rule against. Name them to get them; the warning below
    # fires every time they are missing.
    # `all` also EXCLUDES arms the framework has rejected. `select` was
    # measured on 2026-08-22 (docs/FRAMEWORK.md section 12) at -22 items
    # against `clip`, 0 of 2 cells on every metric, with 2 of 8 runs collapsing
    # on their final epoch -- and FRAMEWORK says do not re-run it. Until this
    # subtraction existed, `--arms all` ran it anyway: the generator was
    # spending GPU on a closed question and putting a known-unstable arm into
    # every campaign, which is the generator contradicting the law.
    # Naming a rejected arm explicitly still works, so `results/selectrun`
    # stays reproducible; it just cannot arrive by default any more.
    # And `all` excludes the reseed control for the SAME reason it excludes the
    # zero-dose siblings: it is a trained arm, so auto-adding it would grow what
    # `all` costs without anyone deciding to spend that. It is not left to a
    # warning, though -- `validate` REFUSES when a trained arm is present
    # without it, because a count trajectory read without its reseed floor is
    # not a measurement. `all+null` includes it.
    rejected = set(P.get("rejected_arms", {}))
    controls = count_control_arms(P)
    # NAMED ARMS ARE ADDED TO `all`, NOT DISCARDED BY IT. `all` used to REPLACE
    # args.arms outright, so `--arms all tralo_null` produced a campaign with no
    # tralo_null in it -- while this same function printed "Add: --arms ...
    # tralo_null" and the refusal below printed "Add: --arms ... tralo_reseed".
    # The tool was instructing the user in a form the tool ignored, and the
    # result looks exactly like a campaign that was generated correctly.
    # Explicit naming also beats the rejected-arm subtraction: `all` must not
    # schedule a rejected arm, but asking for one by name has to keep working
    # or the campaign that produced its verdict stops being reproducible.
    named = {a for a in args.arms if a not in ("all", "all+null")}
    if "all+null" in args.arms:
        # `+null` means "and the null each arm is READ AGAINST", resolved
        # through `null_sibling` and deduplicated -- NOT "every arm whose name
        # ends in _null". It used to be the latter, which scheduled one
        # bit-identical zero-dose run per FAMILY: 32 of `results/dualbar2`'s 88
        # runs computed a single control four times over, 2.6 h of a 7 h
        # campaign, verified by one shared md5 at both cap levels. Resolving
        # through the sibling map collapses them automatically, and keeps the
        # per-family nulls available by name for re-verifying the equivalence.
        base = ({a for a in P["arms"] if not a.endswith("_null")}
                - rejected) | named
        requested = base | {_null_of(P, a) for a in base
                            if _null_of(P, a) in P["arms"]}
    elif "all" in args.arms:
        requested = ({a for a in P["arms"]
                      if not a.endswith("_null") and a not in controls}
                     - rejected) | named
    else:
        requested = named
    for arm in sorted(requested & rejected):
        print("!! %s IS REJECTED and you named it explicitly: %s"
              % (arm, P["rejected_arms"][arm]))
    if not (requested & rejected):
        skipped = sorted(rejected & set(P["arms"])) if (
            "all" in args.arms or "all+null" in args.arms) else []
        if skipped:
            print("NOTE: 'all' skips the rejected arm(s) ->", " ".join(skipped))
    mandatory = set(P["mandatory_arms"])
    arms = sorted(requested | mandatory)
    added = sorted(mandatory - requested)
    if added:
        print("NOTE: added the mandatory clippers ->", " ".join(added))
    if "tralo_margin" in arms or             P["blocks"]["tralo"].get("soft_count_mode") == "margin":
        n = P["blocks"]["tralo"].get("cut_window_items", 5)
        print("  SOFT COUNT = MARGIN-CENTRED (%d items in the window): the "
              "count keeps its value" % n)
        print("      -- the penalty reads the HARD count -- and moves its "
              "gradient onto the decision")
        print("      boundary, where a prediction can actually flip. A "
              "DIFFERENT ESTIMATOR of the same")
        print("      constraint, not a different constraint, so it is scored "
              "against `tralo` (same")
        print("      constraint, other estimator), `tralo_null` (same "
              "estimator, no dose) and the")
        print("      clippers -- all of which must be in THIS campaign.")
        if "tralo_st" in arms:
            print("      DECOMPOSED: `tralo_st` is the same placement as "
                  "`tralo` with only the count")
            print("      VALUE fixed (hard, not sum_i p_ic). tralo -> "
                  "tralo_st isolates the value fix,")
            print("      tralo_st -> tralo_margin isolates the window. "
                  "Bundled they are unattributable.")
        elif "tralo_margin" in arms:
            print("      *** NOT DECOMPOSED: `tralo_margin` changes the count "
                  "VALUE (to the hard count)")
            print("      *** and the gradient PLACEMENT at once. Without "
                  "`tralo_st` a win cannot be")
            print("      *** attributed to either. Add --arms ... tralo_st.")
        if "tralo_coin" not in arms:
            print("      *** NO COIN: the pre-registered kill condition is that "
                  "the arm must beat a")
            print("      *** RANDOM step of the same norm. Add --arms ... "
                  "tralo_coin. The campaign-wide")
            print("      *** --constraint-random-direction flag cannot serve: "
                  "it randomises EVERY arm.")
        if "tralo" not in arms:
            print("      *** `tralo` is NOT in this campaign. Without it the "
                  "margin arm has nothing")
            print("      *** to be an estimator OF: add --arms ... tralo.")

    resolved = resolve_datasets(P, args)
    validate(P, args, resolved, arms)

    todo = [(ds, mdl, tag, arm, seed)
            for seed in P["protocol"]["seeds"] for ds in args.datasets
            for mdl in args.models for tag in args.caps for arm in arms]

    version, written, skipped = code_version(), 0, 0
    total = P["protocol"]["total_epochs"]
    for ds, mdl, tag, arm, seed in todo:
        dc = resolved[ds]
        spec = P["arms"][arm]
        hp = build_hyperparams(P, spec, seed)
        assert hp["warmup_epochs"] + hp["constraint_epochs"] == total, "equal compute"

        path = "%s/%s/%s/%s/%s/seed_%d" % (args.root, mdl, ds, tag, arm, seed)
        cfg = {"methodology": spec["methodology"], "model_name": mdl,
               "constraint": cap_pair(tag), "constraint_tag": tag,
               "dataset_mode": ds, "dataset_config": dc, "hyperparams": hp,
               "base_model_id": compute_base_model_id(P, mdl, hp, ds, dc),
               "arm": arm,
               "exp_name": "%s_%s_%s_%s_c%s_seed%d"
                           % (mdl, ds, arm, tag, _cls_tag(dc), seed),
               "status": "pending", "code_version": version}
        dest = os.path.join(path, "config.json")
        if os.path.exists(dest):
            try:
                prev = json.load(open(dest))
            except (ValueError, OSError):
                prev = None
            if prev is not None:
                prev_cls = prev.get("dataset_config", {}).get("constrained_class")
                if prev_cls is not None and prev_cls != dc["constrained_class"]:
                    sys.exit(
                        "REFUSED: %s already holds a run with constrained_class "
                        "%s, but this campaign asks for %s. Writing here would "
                        "leave two different capped classes in one cell -- a "
                        "completed run is never reset, so the old one would "
                        "survive and be pooled with the new. Use a different "
                        "--root." % (path, prev_cls, dc["constrained_class"]))
                if prev.get("status") == "completed":
                    skipped += 1
                    continue          # never reset a finished run back to pending
        os.makedirs(path, exist_ok=True)
        json.dump(cfg, open(dest, "w"), indent=2)
        written += 1

    cells = len(args.datasets) * len(args.models) * len(args.caps)
    print("%d written, %d already completed (skipped) -> %s"
          % (written, skipped, args.root))
    print("  %d cells (dataset x model x cap) x %d arms x %d seeds"
          % (cells, len(arms), len(P["protocol"]["seeds"])))
    print("  arms:", " ".join(arms))
    print("  trained arms: warm-up %d + constraint %d | post-hoc arms: warm-up %d + 0"
          % (P["protocol"]["trained_warmup"],
             total - P["protocol"]["trained_warmup"], total))
    # POWER, before the GPU time is spent rather than after.
    # The scorer's atomic unit is the CELL, and the exact two-sided Wilcoxon
    # floor at n non-zero pairs is 2^(1-n). With 11 metrics in the BH family, a
    # metric that is the only mover gets q = p * 11, so q < 0.05 needs
    # p < 0.00455 and therefore n >= 9 cells. Below that, NO isolated metric can
    # ever print *** WIN or *** LOSS however large the effect.
    #
    # This guard used to live only in full_panel, i.e. after the campaign ran.
    # The recorded failure is exactly that shape: "at n=2 Wilcoxon floors at
    # p=0.5 so in-flight campaigns ALWAYS read as ties -- arms were abandoned on
    # that."
    floor = 2.0 ** (1 - cells)
    n_family = 11
    print("  POWER: %d cells -> exact Wilcoxon floor p=%.5f; a lone mover needs"
          % (cells, floor))
    print("         q = p x %d < 0.05, i.e. p < %.5f" % (n_family, 0.05 / n_family))
    if floor > 0.05 / n_family:
        need = 9
        print("  *** UNDERPOWERED: with %d cells NO single metric can reach a "
              "*** verdict," % cells)
        print("      whatever the effect size. %d cells is the minimum for one. "
              "Add a" % need)
        print("      backbone, a dataset or a cap level -- or accept that this "
              "campaign can")
        print("      only report DIRECTION and per-cell consistency, never "
              "significance.")
    # Those three are NOT interchangeable, and saying so plainly matters more
    # than the floor above. Every cell inside one dataset shares that dataset's
    # fixed test set and the K derived from it, so a new backbone or a new cap
    # level buys RESOLUTION on the cells we ran; only a new dataset buys an
    # independent test set. full_panel prints a dataset-clustered readout beside
    # the per-cell one for exactly this reason.
    n_ds = len(args.datasets)
    ds_floor = 2.0 ** (1 - n_ds)
    print("  GENERALIZATION: %d dataset(s) -> exact sign-flip floor p=%.3f "
          "on the" % (n_ds, ds_floor))
    print("         clustered unit. Cells within a dataset are NOT independent "
          "draws:")
    print("         a backbone or a cap level adds resolution, a DATASET adds "
          "independence.")
    if ds_floor > 0.05:
        print("         *** at %d dataset(s) no clustered result can reach "
              "p<0.05 --" % n_ds)
        print("         *** with all three it is still 0.25. Generality here is "
              "a DIRECTION")
        print("         *** claim across datasets, never a significant one.")
    # Not auto-added: four more arms DOUBLES the trained half of a campaign, and
    # that is a compute decision. But silently omitting the control is how a
    # delta gets attributed to the constraint when the regime produced it, so it
    # is said out loud, before anything launches.
    orphaned = [a for a in arms
                if P["arms"].get(a, {}).get("phase") == "trained"
                and not a.endswith("_null")
                and _null_of(P, a) in P["arms"]
                and _null_of(P, a) not in arms]
    if orphaned:
        print("  *** NO ZERO-DOSE CONTROL for: %s" % " ".join(sorted(orphaned)))
        print("      Each has a `_null` sibling -- same code path, same warm-up,")
        print("      same 29 transductive epochs, same optimizer restart, same")
        print("      allocator, treatment zeroed. Without it in THIS campaign, a")
        print("      delta vs clip cannot be attributed to the constraint rather")
        print("      than to the regime. Add: --arms ... %s"
              % " ".join(sorted({_null_of(P, a) for a in orphaned})))
    # Say what it is FOR, not just that it is present. The reseed arm is the
    # only control in this campaign that bounds the count trajectory, and a
    # reader who does not know that will report the constraint's count movement
    # as if the alternative were zero.
    present = sorted(count_control_arms(P) & set(arms))
    if present:
        print("  RESEED FLOOR in campaign -> %s" % " ".join(present))
        print("      lambda = 0 and one extra draw from the global generator, "
              "so it is `tralo_null`")
        print("      re-randomised. Read every count trajectory against it: "
              "the constraint moves")
        print("      the capped count RMS 75-95 items and a reseed moves it "
              "83-95, so a count")
        print("      movement is only a result once it is stated as a RATIO to "
              "this arm's.")
    print("  protocol: %s" % os.path.relpath(args.protocol))
    print("  code_version:", version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
