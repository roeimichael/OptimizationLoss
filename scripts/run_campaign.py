"""THE PIPELINE STEP GATE -- run the right checks at the right step, and refuse
to move on when they are red.

`tests/` gates the CODE. `tests/gates/` gates the EXPERIMENT in six buckets.
This runs BOTH at the point in a campaign's life where each is answerable, and
that pairing is the whole point:

  * a GATE bucket proves the DETECTOR works (synthetic inputs, negative
    controls, runs in CI with no data on the machine)
  * an INSTRUMENT runs that detector against THIS campaign

Either alone is a half measure. `tests/gates` passing says the dose detector is
correct; it says nothing about whether THIS campaign landed its dose. And
`dose_landed` printing a number says nothing about whether the tool that
printed it still works. So every step below runs the bucket and the instrument.

THE STEPS, in the order a campaign lives them:

    1 stage      before a config exists      data + budget buckets
    2 verify     after generating, before launching
    3 launch     immediately before dispatch
    4 firstrun   on the FIRST completed runs -- the cheapest place to kill a
                 bad campaign, and the one this project keeps skipping
    5 score      before any number is quoted

THREE OUTCOMES, NOT TWO. A check can pass, it can FAIL, or it can turn out to
be UNRUNNABLE here -- and the third is reported separately from both. A
campaign worktree is PINNED at the commit its configs were generated from, and
the gate buckets import training-path modules that may postdate it
(`configs.task_cells` is the live example on `optloss-domb`). `configs/` is
frozen while a campaign runs, so that gate genuinely cannot execute there.
Reporting it as RED would blame a healthy campaign for version skew, and a gate
that cries wolf is a gate that gets switched off -- which is how this project
lost the `--constraint-fp32` dose for a whole campaign.

WHY `firstrun` IS THE ONE THAT MATTERS. `tralo_uniform` ran at 1/29 of its dose
beside `tralo` at 29/29 in the SAME campaign, and still wrote
`status: completed` with plausible metrics on every other axis. `iwc3` lost 328
of 1044 steps. `taskwin1` landed 20/29 and had to be killed at 3/48. Every one
of those was visible in the first finished run and cost a night or a week
because nobody looked until the end.

TOGGLES. `--skip` turns off a named check or a whole step. A skipped check
ANNOUNCES itself, loudly, in the summary and in the exit banner -- this repo
has paid three times for a check that could not run and said nothing (the
and-chained cache guard, `full_panel`'s allocator blindness, `graph_probe
--dump`). Silence and a pass must never look the same.

    python -m scripts.run_campaign --root results/foo --step verify
    python -m scripts.run_campaign --root results/foo --step firstrun
    python -m scripts.run_campaign --root results/foo --all
    python -m scripts.run_campaign --list
    python -m scripts.run_campaign --self-test

Exit code is 0 only if every REQUIRED check that ran passed. A skipped required
check makes the exit code 0 but the banner says SKIPPED and names it, because
the tool must not lie about what it verified.
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# (step, blurb, [(check-name, argv-template, required, kind)])
# `{root}` is substituted with the campaign root. A check with no `{root}` is
# campaign-independent and runs the same everywhere.
#
# `kind` decides how a NON-ZERO exit is read, and the distinction is
# load-bearing (see THREE OUTCOMES in the module docstring):
#
#   "gate"        a pytest bucket. Exit 1 means the gate RAN and the campaign
#                 failed it. Exit 2/3/4 (collection, internal, usage) and 5
#                 (nothing collected) mean the gate COULD NOT RUN here.
#   "instrument"  an ordinary script. Any non-zero exit is a failure.
STEPS = [
    ("stage",
     "before a config exists -- is this cell even a question?",
     [("gate:data", ["-m", "scripts.preflight", "--stage", "data"], True, "gate"),
      ("gate:budget", ["-m", "scripts.preflight", "--stage", "budget"], True, "gate"),
      ("verify_caps", ["-m", "scripts.verify_caps"], True, "instrument")]),

    ("verify",
     "after generating, before launching -- is the grid fair and alive?",
     [("gate:model", ["-m", "scripts.preflight", "--stage", "model"], True, "gate"),
      ("gate:grid", ["-m", "scripts.preflight", "--stage", "grid"], True, "gate"),
      ("audit_config", ["-m", "scripts.audit_config"], True, "instrument"),
      ("check_parity", ["-m", "scripts.check_parity", "{root}"], True, "instrument"),
      ("quarantine", ["-m", "scripts.quarantine", "--list"], False, "instrument"),
      ("smoke_arms", ["-m", "scripts.smoke_arms"], False, "instrument")]),

    ("launch",
     "immediately before dispatch -- is the RIG healthy?",
     [("rig_status", ["-m", "scripts.rig_status"], True, "instrument")]),

    ("firstrun",
     "on the FIRST completed runs -- kill a bad campaign here, not at hour 19",
     [("gate:trainlog", ["-m", "scripts.preflight", "--stage", "trainlog"], True, "gate"),
      ("dose_landed", ["-m", "scripts.dose_landed", "{root}"], True, "instrument"),
      ("log_health", ["-m", "scripts.log_health", "{root}"], False, "instrument"),
      # ADVISORY, and deliberately so. It exits 2 when no cell is SENSITIVE,
      # which on the corpus of 2026-09-04 is EVERY cell in dom1, taskwin2 and
      # equaldose1 -- so required=True would block every campaign this project
      # currently knows how to run. Advisory puts the number in front of the
      # person deciding whether hour 19 is worth buying, which is the job.
      ("sensitivity_screen",
       ["-m", "scripts.sensitivity_screen", "--campaign", "{root}"],
       False, "instrument")]),

    ("score",
     "before any number is quoted",
     [("gate:results", ["-m", "scripts.preflight", "--stage", "results"], True, "gate"),
      ("check_parity", ["-m", "scripts.check_parity", "{root}"], True, "instrument"),
      ("dose_landed", ["-m", "scripts.dose_landed", "{root}"], True, "instrument")]),
]
BY_NAME = {s: (b, c) for s, b, c in STEPS}
NEEDS_ROOT = {s for s, _, checks in STEPS
              if any("{root}" in a for _, argv, _, _ in checks for a in argv)}


def run_check(name, argv, root, verbose):
    """-> (name, returncode, tail-of-output)"""
    cmd = [PY] + [a.format(root=root) if "{root}" in a else a for a in argv]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    if verbose:
        print(out.rstrip())
    tail = [l for l in out.splitlines()
            if l.strip() and ("FAIL" in l or "ERROR" in l or "!!" in l
                              or "REFUS" in l or "Error" in l)][:3]
    return p.returncode, tail


# pytest's exit codes. 1 is the only one that means "it ran and something
# failed"; the rest mean the bucket never executed. Treating 2 as a failure
# reports RED on a healthy campaign, and a gate that cries wolf gets ignored.
PYTEST_RAN_AND_FAILED = 1
PYTEST_COULD_NOT_RUN = (2, 3, 4, 5)


def run_step(step, root, skip, verbose, out=print):
    """-> (n_pass, n_fail, n_skip, n_unrunnable, [failures], [unrunnable])"""
    blurb, checks = BY_NAME[step]
    out("")
    out("=" * 74)
    out("STEP %-10s %s" % (step, blurb))
    out("=" * 74)
    npass = nfail = nskip = nunrun = 0
    failures, unrunnable = [], []
    for name, argv, required, kind in checks:
        if step in skip or name in skip:
            # LOUD. A check that did not run must never read as one that
            # passed -- three defects in this repo were exactly that shape.
            out("  SKIP   %-16s <-- DISABLED by --skip, nothing was verified"
                % name)
            nskip += 1
            continue
        rc, tail = run_check(name, argv, root, verbose)
        if rc == 0:
            out("  ok     %s" % name)
            npass += 1
        elif kind == "gate" and rc in PYTEST_COULD_NOT_RUN:
            # THE THIRD OUTCOME. The bucket never executed -- almost always
            # because this worktree is PINNED at the commit its configs were
            # generated from and the gates import a training-path module that
            # did not exist yet (`configs.task_cells` is the live example).
            # `configs/` is frozen while a campaign runs, so this cannot be
            # fixed here and must not be reported as the campaign's fault.
            out("  N/A    %-16s (exit %d) -- the GATE could not run here, so "
                "it verified NOTHING" % (name, rc))
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
            out("  warn   %-16s (exit %d, advisory -- not a blocker)"
                % (name, rc))
            npass += 1
    return npass, nfail, nskip, nunrun, failures, unrunnable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="campaign root, e.g. results/dom1b")
    ap.add_argument("--step", nargs="*", default=None,
                    help="one or more of: %s" % ", ".join(BY_NAME))
    ap.add_argument("--all", action="store_true", help="every step, in order")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="step or check names to disable; every one is named "
                         "loudly in the output and the exit banner")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="print each check's full output, not just its verdict")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return self_test()

    if a.list:
        print("STEP        WHAT IT GATES")
        for s, b, checks in STEPS:
            print("  %-10s %s" % (s, b))
            for n, _, req, kind in checks:
                print("       %-16s %-9s %s"
                      % (n, kind, "required" if req else "advisory"))
        return 0

    if a.all:
        names = [s for s, _, _ in STEPS]
    elif a.step:
        names = a.step
    else:
        ap.error("give --step <name> ... or --all (see --list)")

    bad = [n for n in names if n not in BY_NAME]
    if bad:
        # NOT a silent skip. A typo'd step that runs nothing and exits 0 is the
        # same shape as the defect this whole file exists to prevent.
        ap.error("unknown step(s): %s. Known: %s"
                 % (", ".join(bad), ", ".join(BY_NAME)))

    need = [n for n in names if n in NEEDS_ROOT]
    if need and not a.root:
        ap.error("step(s) %s check THIS campaign, so --root is required"
                 % ", ".join(need))
    if a.root and not os.path.isdir(os.path.join(REPO, a.root)):
        ap.error("--root %s does not exist" % a.root)

    # An unknown --skip name would silently disable nothing while reading as if
    # it had. Validate against the real check and step names.
    known = set(BY_NAME) | {n for _, _, cs in STEPS for n, _, _, _ in cs}
    unknown = [s for s in a.skip if s not in known]
    if unknown:
        ap.error("--skip names nothing that exists: %s. Known: %s"
                 % (", ".join(unknown), ", ".join(sorted(known))))

    tp = tf = ts = tu = 0
    all_failures, all_unrunnable = [], []
    for step in names:
        p, f, sk, u, fails, unrun = run_step(step, a.root, set(a.skip),
                                             a.verbose)
        tp, tf, ts, tu = tp + p, tf + f, ts + sk, tu + u
        all_failures += fails
        all_unrunnable += unrun

    print("")
    print("=" * 74)
    if tf:
        print("RED -- %d required check(s) FAILED: %s"
              % (tf, ", ".join(all_failures)))
        print("Do NOT move to the next step. Each of these is cheaper to fix "
              "here than at any later stage.")
    else:
        print("GREEN -- %d check(s) passed" % tp)
    if tu:
        print("")
        print("!! %d GATE(S) COULD NOT RUN and verified NOTHING: %s"
              % (tu, ", ".join(all_unrunnable)))
        print("!! This is version skew, not a campaign defect. A campaign "
              "worktree is PINNED at the commit its configs were generated")
        print("!! from, and the gate buckets import training-path modules "
              "that may postdate it. `configs/` is frozen while a campaign")
        print("!! runs, so the fix is to run these buckets in a checkout whose "
              "src/ and configs/ are current -- not to unpin this one.")
    if ts:
        print("")
        print("!! %d CHECK(S) WERE SKIPPED and verified NOTHING: %s"
              % (ts, ", ".join(sorted(set(a.skip)))))
        print("!! This run does not attest to what those checks cover.")
    print("=" * 74)
    return 1 if tf else 0


def self_test(w=sys.stdout.write):
    """Both directions: the driver must REFUSE bad input and ALLOW good."""
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
    check(exits(["--step", "nosuchstep", "--root", "results"]) == 2,
          "an unknown STEP errors rather than running nothing")
    check(exits(["--step", "verify"]) == 2,
          "a step that checks THIS campaign refuses without --root")
    check(exits(["--step", "verify", "--root", "no/such/dir"]) == 2,
          "a --root that does not exist is refused")
    check(exits(["--step", "verify", "--root", "results",
                 "--skip", "nosuchcheck"]) == 2,
          "an unknown --skip name errors rather than disabling nothing")
    check(exits([]) == 2, "no --step and no --all errors")

    # every declared check must name a module that actually exists
    missing = []
    for _, _, checks in STEPS:
        for name, argv, _, _ in checks:
            if argv[0] == "-m":
                mod = argv[1].replace(".", os.sep) + ".py"
                if not os.path.exists(os.path.join(REPO, mod)):
                    missing.append("%s -> %s" % (name, argv[1]))
    check(not missing, "every declared check names a real module (%s)"
          % (", ".join(missing) or "all present"))

    # the LOUD-SKIP contract: a skipped check must be reported, not swallowed
    lines = []
    p, f, sk, u, _, _ = run_step("launch", "results", {"rig_status"}, False,
                                 out=lines.append)
    text = "\n".join(lines)
    check(sk == 1 and f == 0 and "DISABLED" in text and "nothing was verified"
          in text,
          "a skipped check ANNOUNCES itself instead of reading as a pass")

    # THE THREE OUTCOMES, both directions. A gate bucket that cannot even be
    # collected must read as UNRUNNABLE, and one that runs and fails must read
    # as FAILED -- conflating them is what makes a gate get switched off.
    import types
    g = globals()
    real = g["run_check"]
    try:
        # stub ONLY the gate -- the instruments beside it are healthy, which
        # is exactly the real situation on a pinned worktree
        g["run_check"] = lambda n, a_, r, v: (
            (2, ["ImportError: no module"]) if n.startswith("gate:") else (0, []))
        lines = []
        p, f, sk, u, fails, unrun = run_step("firstrun", "results", set(),
                                             False, out=lines.append)
        check(u >= 1 and not fails,
              "a gate that CANNOT BE COLLECTED reads as UNRUNNABLE, not failed")
        check("verified NOTHING" in "\n".join(lines),
              "an unrunnable gate says it verified nothing")

        g["run_check"] = lambda n, a_, r, v: (
            (1, ["assert False"]) if n.startswith("gate:") else (0, []))
        lines = []
        p, f, sk, u, fails, unrun = run_step("firstrun", "results", set(),
                                             False, out=lines.append)
        check(f >= 1 and not unrun,
              "a gate that RAN AND FAILED reads as FAILED, not unrunnable")
    finally:
        g["run_check"] = real

    # NEEDS_ROOT must be derived, not hand-listed, or it goes stale
    check("launch" not in NEEDS_ROOT and "verify" in NEEDS_ROOT,
          "NEEDS_ROOT is derived from the check table, not hardcoded")

    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
