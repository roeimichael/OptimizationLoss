"""Run required software and campaign checks at each lifecycle stage."""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
STEPS = [
    (
        "stage",
        "before a config exists -- is this cell even a question?",
        [
            ("gate:data", ["-m", "scripts.preflight", "--stage", "data"], True, "gate"),
            (
                "gate:budget",
                ["-m", "scripts.preflight", "--stage", "budget"],
                True,
                "gate",
            ),
            # --campaign or this checks the DEFAULT caps (L30_G30 L30_G50
            # L50_G50) on every protocol dataset, which is not what is being
            # launched and fails on any dataset whose slice is absent.
            ("verify_caps",
             ["-m", "scripts.verify_caps", "--campaign", "{root}"],
             True, "instrument"),
        ],
    ),
    (
        "verify",
        "after generating, before launching -- is the grid fair and alive?",
        [
            (
                "gate:model",
                ["-m", "scripts.preflight", "--stage", "model"],
                True,
                "gate",
            ),
            ("gate:grid", ["-m", "scripts.preflight", "--stage", "grid"], True, "gate"),
            (
                "audit_config",
                ["-m", "scripts.audit_config", "{root}"],
                True,
                "instrument",
            ),
            (
                "check_parity",
                ["-m", "scripts.check_parity", "{root}"],
                True,
                "instrument",
            ),
            ("smoke_arms", ["-m", "scripts.smoke_arms"], False, "instrument"),
        ],
    ),
    (
        "launch",
        "immediately before dispatch -- is the RIG healthy?",
        [
            (
                "data_present",
                ["-m", "scripts.data_present", "{root}"],
                True,
                "instrument",
            ),
            ("rig_status", ["-m", "scripts.rig_status", "--campaign", "{root}"], True, "instrument"),
        ],
    ),
    (
        "firstrun",
        "on the FIRST completed runs -- kill a bad campaign here, not at hour 19",
        [
            (
                "gate:trainlog",
                ["-m", "scripts.preflight", "--stage", "trainlog"],
                True,
                "gate",
            ),
            # HARD GATE. If the model memorises the train set in the first few
            # epochs, cross-entropy is ~0 for the rest of the run and -- under
            # `constraint_grad_mode: normalize`, which rescales the constraint
            # gradient to a FIXED norm however small the violation -- every
            # remaining constraint step is full-size and opposed by nothing.
            # The constraint is shoving a frozen boundary rather than reshaping
            # it, and no arm comparison made in that regime means anything.
            # Measured on the whole corpus: fmow2 is live for 3.2 of 29
            # constraint epochs and bcn for 4.5, on MobileNetV2, MobileNetV3
            # and ViTB16 alike -- so this is the TRAINING RECIPE, not one bad
            # dataset, and it must stop a campaign rather than be noted.
            (
                "gate:saturation",
                ["-m", "scripts.saturation_gate", "--glob",
                 "{root}/*/*/*/*/seed_*", "--strict"],
                True,
                "gate",
            ),
            (
                "dose_landed",
                ["-m", "scripts.dose_landed", "{root}"],
                True,
                "instrument",
            ),
            ("log_health", ["-m", "scripts.log_health", "{root}"], True, "instrument"),
            (
                "pred_integrity",
                ["-m", "scripts.pred_integrity", "{root}"],
                True,
                "instrument",
            ),
            (
                "feasibility_check",
                ["-m", "scripts.feasibility_check", "{root}"],
                True,
                "instrument",
            ),
        ],
    ),
    (
        "score",
        "before any number is quoted",
        [
            (
                "pred_integrity",
                ["-m", "scripts.pred_integrity", "{root}"],
                True,
                "instrument",
            ),
            (
                "feasibility_check",
                ["-m", "scripts.feasibility_check", "{root}"],
                True,
                "instrument",
            ),
            (
                "gate:results",
                ["-m", "scripts.preflight", "--stage", "results"],
                True,
                "gate",
            ),
            (
                "check_parity",
                ["-m", "scripts.check_parity", "{root}"],
                True,
                "instrument",
            ),
            (
                "dose_landed",
                ["-m", "scripts.dose_landed", "{root}"],
                True,
                "instrument",
            ),
            (
                "deployed_h2h",
                ["-m", "scripts.deployed_h2h", "--campaign", "{root}"],
                True,
                "instrument",
            ),
        ],
    ),
]
BY_NAME = {s: (b, c) for (s, b, c) in STEPS}
NEEDS_ROOT = {
    s
    for (s, _, checks) in STEPS
    if any(("{root}" in a for (_, argv, _, _) in checks for a in argv))
}


def run_check(name, argv, root, verbose):
    cmd = [PY] + [a.format(root=root) if "{root}" in a else a for a in argv]
    p = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (p.stdout or "") + (p.stderr or "")
    if verbose:
        print(out.rstrip())
    tail = [
        l
        for l in out.splitlines()
        if l.strip()
        and (
            "FAIL" in l or "ERROR" in l or "!!" in l or ("REFUS" in l) or ("Error" in l)
        )
    ][:3]
    return (p.returncode, tail)


PYTEST_COULD_NOT_RUN = (2, 3, 4, 5)


def run_step(step, root, skip, verbose, out=print, prior_incomplete=False):
    (blurb, checks) = BY_NAME[step]
    out("")
    out("=" * 74)
    out("STEP %-10s %s" % (step, blurb))
    out("=" * 74)
    npass = nfail = nskip = nunrun = 0
    (failures, unrunnable) = ([], [])
    for name, argv, required, kind in checks:
        if name == "deployed_h2h" and (prior_incomplete or nfail or nunrun or nskip):
            out("  BLOCKED deployed_h2h: required checks did not all pass")
            nskip += 1
            continue
        if step in skip or name in skip:
            out("  SKIP   %-16s <-- DISABLED by --skip, nothing was verified" % name)
            nskip += 1
            continue
        (rc, tail) = run_check(name, argv, root, verbose)
        if rc == 0:
            out("  ok     %s" % name)
            npass += 1
        elif kind == "gate" and rc in PYTEST_COULD_NOT_RUN:
            out(
                "  N/A    %-16s (exit %d) -- the GATE could not run here, so it verified NOTHING"
                % (name, rc)
            )
            for t in tail:
                out("           %s" % t.strip()[:96])
            nunrun += 1
            unrunnable.append("%s/%s" % (step, name))
        elif required:
            out("  FAIL   %-16s (exit %d)" % (name, rc))
            for t in tail:
                out("           %s" % t.strip()[:96])
            nfail += 1
            failures.append("%s/%s" % (step, name))
        else:
            out("  warn   %-16s (exit %d, advisory -- not a blocker)" % (name, rc))
            npass += 1
    return (npass, nfail, nskip, nunrun, failures, unrunnable)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="campaign root, e.g. results/dom1b")
    ap.add_argument(
        "--step",
        nargs="*",
        default=None,
        help="one or more of: %s" % ", ".join(BY_NAME),
    )
    ap.add_argument("--all", action="store_true", help="every step, in order")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="step or check names to disable; every one is named loudly in the output and the exit banner",
    )
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="print each check's full output, not just its verdict",
    )
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.list:
        print("STEP        WHAT IT GATES")
        for s, b, checks in STEPS:
            print("  %-10s %s" % (s, b))
            for n, _, req, kind in checks:
                print(
                    "       %-16s %-9s %s"
                    % (n, kind, "required" if req else "advisory")
                )
        return 0
    bad = [n for n in a.step or [] if n not in BY_NAME]
    if bad:
        ap.error("unknown step(s): %s" % ", ".join(bad))
    if a.all:
        names = [s for (s, _, _) in STEPS]
    elif a.step:
        names = a.step
    else:
        ap.error("give --step <name> ... or --all (see --list)")
    bad = [n for n in names if n not in BY_NAME]
    if bad:
        ap.error(
            "unknown step(s): %s. Known: %s" % (", ".join(bad), ", ".join(BY_NAME))
        )
    need = [n for n in names if n in NEEDS_ROOT]
    if need and (not a.root):
        ap.error(
            "step(s) %s check THIS campaign, so --root is required" % ", ".join(need)
        )
    if a.root:
        from src.pipeline.campaign import validate_campaign, safe_path
        try:
            a.root = str(safe_path(os.path.join(REPO, a.root)))
            report_only = all(name in ('score', 'firstrun') for name in names)
            validate_campaign(a.root, check_data=not report_only, check_runtime=not report_only)
        except (ValueError, OSError, KeyError) as exc:
            ap.error(str(exc))
    known = set(BY_NAME) | {n for (_, _, cs) in STEPS for (n, _, _, _) in cs}
    unknown = [s for s in a.skip if s not in known]
    if unknown:
        ap.error(
            "--skip names nothing that exists: %s. Known: %s"
            % (", ".join(unknown), ", ".join(sorted(known)))
        )
    tp = tf = ts = tu = 0
    (all_failures, all_unrunnable) = ([], [])
    for step in names:
        (p, f, sk, u, fails, unrun) = run_step(
            step, a.root, set(a.skip), a.verbose, prior_incomplete=bool(tf or ts or tu)
        )
        (tp, tf, ts, tu) = (tp + p, tf + f, ts + sk, tu + u)
        all_failures += fails
        all_unrunnable += unrun
    print("")
    print("=" * 74)
    if tf:
        print("RED -- %d required check(s) FAILED: %s" % (tf, ", ".join(all_failures)))
        print(
            "Do NOT move to the next step. Each of these is cheaper to fix here than at any later stage."
        )
    elif tu or ts:
        print("INCOMPLETE -- required checks were unavailable or skipped")
    else:
        print("GREEN -- %d check(s) passed" % tp)
    if tu:
        print("")
        print(
            "!! %d GATE(S) COULD NOT RUN and verified NOTHING: %s"
            % (tu, ", ".join(all_unrunnable))
        )
        print(
            "!! This is version skew, not a campaign defect. A campaign worktree is PINNED at the commit its configs were generated"
        )
        print(
            "!! from, and the gate buckets import training-path modules that may postdate it. `configs/` is frozen while a campaign"
        )
        print(
            "!! runs, so the fix is to run these buckets in a checkout whose src/ and configs/ are current -- not to unpin this one."
        )
    if ts:
        print("")
        print(
            "!! %d CHECK(S) WERE SKIPPED and verified NOTHING: %s"
            % (ts, ", ".join(sorted(set(a.skip))))
        )
        print("!! This run does not attest to what those checks cover.")
    print("=" * 74)
    return 1 if tf or tu or ts else 0


def self_test(w=sys.stdout.write):
    ok = True

    def check(good, label):
        nonlocal ok
        w("  %-4s %s\n" % ("PASS" if good else "FAIL", label))
        ok = ok and good

    def exits(argv):
        try:
            return main(argv)
        except SystemExit as e:
            return e.code if e.code is not None else 0

    check(exits(["--list"]) == 0, "--list works with no root")
    check(
        exits(["--step", "nosuchstep", "--root", "results"]) == 2,
        "an unknown STEP errors rather than running nothing",
    )
    check(
        exits(["--step", "verify"]) == 2,
        "a step that checks THIS campaign refuses without --root",
    )
    check(
        exits(["--step", "verify", "--root", "no/such/dir"]) == 2,
        "a --root that does not exist is refused",
    )
    check(
        exits(["--step", "verify", "--root", "results", "--skip", "nosuchcheck"]) == 2,
        "an unknown --skip name errors rather than disabling nothing",
    )
    check(exits([]) == 2, "no --step and no --all errors")
    missing = []
    for _, _, checks in STEPS:
        for name, argv, _, _ in checks:
            if argv[0] == "-m":
                mod = argv[1].replace(".", os.sep) + ".py"
                if not os.path.exists(os.path.join(REPO, mod)):
                    missing.append("%s -> %s" % (name, argv[1]))
    check(
        not missing,
        "every declared check names a real module (%s)"
        % (", ".join(missing) or "all present"),
    )
    lines = []
    (p, f, sk, u, _, _) = run_step(
        "launch", "results", {"rig_status", "data_present"}, False, out=lines.append
    )
    text = "\n".join(lines)
    check(
        sk == 2
        and f == 0
        and ("DISABLED" in text)
        and ("nothing was verified" in text),
        "a skipped check ANNOUNCES itself instead of reading as a pass",
    )
    g = globals()
    real = g["run_check"]
    try:
        g["run_check"] = lambda n, a_, r, v: (
            (2, ["ImportError: no module"]) if n.startswith("gate:") else (0, [])
        )
        lines = []
        (p, f, sk, u, fails, unrun) = run_step(
            "firstrun", "results", set(), False, out=lines.append
        )
        check(
            u >= 1 and (not fails),
            "a gate that CANNOT BE COLLECTED reads as UNRUNNABLE, not failed",
        )
        check(
            "verified NOTHING" in "\n".join(lines),
            "an unrunnable gate says it verified nothing",
        )
        g["run_check"] = lambda n, a_, r, v: (
            (1, ["assert False"]) if n.startswith("gate:") else (0, [])
        )
        lines = []
        (p, f, sk, u, fails, unrun) = run_step(
            "firstrun", "results", set(), False, out=lines.append
        )
        check(
            f >= 1 and (not unrun),
            "a gate that RAN AND FAILED reads as FAILED, not unrunnable",
        )
    finally:
        g["run_check"] = real
    check(
        "stage" not in NEEDS_ROOT and "verify" in NEEDS_ROOT,
        "NEEDS_ROOT is derived from the check table, not hardcoded",
    )
    check(
        "launch" in NEEDS_ROOT,
        "adding a {root} check to `launch` puts it in NEEDS_ROOT by itself",
    )
    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
