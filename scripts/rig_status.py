"""Is the rig healthy? One command, and every check is a trap this project hit.

WHY THIS EXISTS. Every failure in this project's operations has been SILENT.
A launch ran 40 runs on CPU because `bash -c` re-sourced `.bashrc` and flipped
conda back to base, and the only evidence was one `Device: CPU` line. Killing a
dispatcher left three runner children alive writing into the directory a fresh
dispatcher had just claimed. A sibling checkout turned out to share the live
campaign's git object store. A GPU picked up a second user. None of those raise;
all of them corrupt a campaign or waste a night.

So this reports STATE, and every row is a check that has already failed once.
Run it before launching, after launching, and whenever something looks odd:

    python -m scripts.rig_status                      # everything
    python -m scripts.rig_status --campaign results/iwc2

The predicates are pure functions over plain data (`orphaned_runners`,
`code_version_uniform`, `shared_gpus`, `interpreter_is_env`) so they are tested
on fixtures locally, where there is no server, no GPU and no NFS. `main()` only
gathers.

EXIT CODE is 1 if any check is FAIL, so it can gate a launch script.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

OK, WARN, FAIL = "OK", "WARN", "FAIL"

# The runner is what actually trains. It is a CHILD of the dispatcher, and it
# does not die when the dispatcher does -- that is the orphan bug.
RUNNER_PAT = re.compile(r"src\.experiments\.runner")
DISPATCH_PAT = re.compile(r"python\s+main\.py")   # kept for the self-test only


def is_dispatcher(args):
    r"""Is this command line a DISPATCHER, or something that merely MENTIONS one?

    The substring `python\s+main\.py` matched three things that are not
    dispatchers, and produced a FAIL on a perfectly healthy rig on 2026-09-01:

      * the wrapper shell -- `bash -c '... && python main.py'` -- which is the
        dispatcher's own PARENT and always present when it was launched detached
      * `setsid ... python main.py`, same story
      * the `pgrep -f "python main.py"` that goes looking for it

    A FAIL that cries wolf is worse than no check at all: it is the one people
    learn to scroll past, and then the real one scrolls past too.

    So: the EXECUTABLE must be a python, and the first non-flag argument must be
    `main.py`. `-m` and `-c` disqualify -- those are the runner
    (`python -u -m src.experiments.runner ...`) and inline code, not the
    dispatcher.
    """
    toks = args.split()
    if not toks or not os.path.basename(toks[0]).startswith("python"):
        return False
    rest = toks[1:]
    i = 0
    while i < len(rest) and rest[i].startswith("-"):
        if rest[i] in ("-m", "-c"):
            return False
        i += 1
    return i < len(rest) and os.path.basename(rest[i]) == "main.py"


# --------------------------------------------------------------------------
# pure predicates -- these are the tested surface
# --------------------------------------------------------------------------
def orphaned_runners(procs):
    """Runner processes whose dispatcher is gone.

    `procs` is a list of dicts: {"pid", "ppid", "args"}. A runner is orphaned
    when no LIVE process in the list is a dispatcher AND the runner's parent is
    not itself a live runner (dataloader workers fork from the runner and
    inherit its command line, so they must not be counted as orphans).

    This is the exact shape of the bug that nearly corrupted `results/iwc2`:
    killing PID 806447 left three `anaconda3/bin/python -m src.experiments.runner`
    children alive, writing into the same run directory as the relaunch.
    """
    live = {p["pid"] for p in procs}
    dispatchers = [p for p in procs if is_dispatcher(p["args"])]
    runners = [p for p in procs if RUNNER_PAT.search(p["args"])]
    if dispatchers:
        return []
    # No dispatcher at all: every runner whose parent is also dead is an orphan.
    # A forked dataloader worker has a live runner parent, so it is excluded.
    runner_pids = {p["pid"] for p in runners}
    return [p for p in runners
            if p["ppid"] not in runner_pids and p["ppid"] not in live]


def stale_running(campaign, running_count, owned_roots):
    """Is a `running` status a lie?

    A run is marked `running` by the dispatcher that started it, and the status
    is only reset to `pending` when a dispatcher next starts on THAT campaign
    root. So a campaign showing `running > 0` with no dispatcher pointed at it
    holds a run that DIED -- and it reads as alive, which is worse than reading
    as pending, because nothing prompts anyone to look.

    Found on first execution: mc29, mnv3bar, vit_ceskip and vit_diag each held
    a stale `running` while the only live dispatcher was on iwc2.

    `owned_roots` is the set of EXPERIMENT_DIR values of live dispatchers.
    """
    if running_count <= 0:
        return False
    return not any(campaign == _root_name(r) for r in owned_roots)


def _root_name(path):
    return os.path.basename(str(path).rstrip("/").rstrip(os.sep))


def code_version_uniform(configs):
    """Every completed run of a campaign must carry ONE code_version.

    `configs` is a list of dicts already loaded from config.json. Returns
    (uniform, mapping version -> count). A campaign split across two commits is
    not a fair comparison -- `code_version` is a git hash, which is why the
    training path is frozen while a campaign runs.
    """
    seen = {}
    for c in configs:
        v = c.get("run_code_version") or c.get("code_version")
        if v is None:
            continue
        seen[v] = seen.get(v, 0) + 1
    return (len(seen) <= 1), seen


def shared_gpus(apps, me):
    """GPUs carrying a process from someone other than `me`, plus one of mine.

    `apps` is a list of dicts: {"gpu", "pid", "user"}. The house rule is never
    to share a GPU with another user; this reports where that is happening
    rather than leaving it to a manual read of `nvidia-smi`.
    """
    by_gpu = {}
    for a in apps:
        by_gpu.setdefault(a["gpu"], set()).add(a["user"])
    return sorted(g for g, users in by_gpu.items()
                  if me in users and len(users) > 1)


def interpreter_is_env(executable, env_marker="envs/optloss"):
    """Is this the optloss interpreter, or did we fall back to base conda?

    Base conda carries a CPU-only torch here. A campaign launched under it runs
    to completion and produces plausible numbers -- on CPU, at ~1/50 the speed
    and with a different numerical path.
    """
    return env_marker in executable.replace(os.sep, "/")


# --------------------------------------------------------------------------
# gathering
# --------------------------------------------------------------------------
def _sh(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
        return out.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def read_procs():
    txt = _sh("ps -u $(whoami) -o pid=,ppid=,args=")
    procs = []
    for line in txt.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            procs.append({"pid": int(parts[0]), "ppid": int(parts[1]),
                          "args": parts[2]})
        except ValueError:
            continue
    return procs


def read_gpu_apps():
    txt = _sh("nvidia-smi --query-compute-apps=gpu_bus_id,pid "
              "--format=csv,noheader")
    apps = []
    for line in txt.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        user = _sh("ps -o user= -p %s" % parts[1]).strip()
        apps.append({"gpu": parts[0], "pid": parts[1], "user": user or "?"})
    return apps


def experiment_dir(pid):
    """The EXPERIMENT_DIR one dispatcher was launched with, or None.

    None means UNKNOWN, never "no root": off Linux, or when /proc is
    unreadable, the caller must fail closed rather than assume two dispatchers
    are on different campaigns.
    """
    try:
        with open("/proc/%d/environ" % pid, "rb") as fh:
            blob = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    for item in blob.split("\0"):
        if item.startswith("EXPERIMENT_DIR="):
            return item.split("=", 1)[1]
    return None


def dispatcher_verdict(pairs):
    """(status, detail) for a list of (pid, EXPERIMENT_DIR-or-None).

    🛑 TWO DISPATCHERS ARE NOT A DEFECT. THEY ARE THE DOCUMENTED MULTI-GPU
    PATTERN. This check used to FAIL on `len(disp) > 1` regardless of what
    those dispatchers were pointed at, so the normal way of running two
    campaigns on two GPUs -- one `EXPERIMENT_DIR` each, which is exactly how
    the project's own dispatch note says to do it -- lit a red FAIL on a
    healthy rig every time. A gate that fires on correct operation is a gate
    that gets ignored, and this one sits directly above `orphaned runners`,
    which is a real and silent failure.

    The defect it exists to catch is two dispatchers CLAIMING THE SAME RUN
    DIRECTORIES: both reset `pending`, both pick the same config, both write
    into it. That is a collision of roots, not a count of processes.

    Fails CLOSED on an unreadable root: if two dispatchers are up and we
    cannot prove they are on different campaigns, that is not an OK.
    """
    if not pairs:
        return WARN, "none running"
    if len(pairs) == 1:
        pid, root = pairs[0]
        return OK, "1 running (pid %d%s)" % (
            pid, "" if not root else " on %s" % root)

    byroot, unknown = {}, []
    for pid, root in pairs:
        if root is None:
            unknown.append(pid)
        else:
            byroot.setdefault(os.path.normpath(root), []).append(pid)
    clash = {r: p for r, p in byroot.items() if len(p) > 1}
    if clash:
        return FAIL, "%d dispatcher(s) share a run root: %s -- they race on " \
                     "the same run dirs, both resetting `pending` and both " \
                     "claiming the same config" % (
                         sum(len(p) for p in clash.values()),
                         "; ".join("%s <- pids %s"
                                   % (r, ", ".join(map(str, sorted(p))))
                                   for r, p in sorted(clash.items())))
    if unknown:
        return FAIL, "%d dispatchers running and the EXPERIMENT_DIR of %d of " \
                     "them could not be read (pids %s), so they cannot be " \
                     "shown to be on different campaigns" % (
                         len(pairs), len(unknown),
                         ", ".join(map(str, sorted(unknown))))
    return OK, "%d running, one per campaign root: %s" % (
        len(pairs), "; ".join("%s (pid %d)" % (r, p[0])
                              for r, p in sorted(byroot.items())))


def dispatcher_roots(procs):
    """EXPERIMENT_DIR of every live dispatcher, read from /proc.

    Falls back to an empty set off Linux, where the stale check is then skipped
    rather than reported wrongly.
    """
    roots = set()
    for p in procs:
        if not is_dispatcher(p["args"]):
            continue
        try:
            with open("/proc/%d/environ" % p["pid"], "rb") as fh:
                blob = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        for item in blob.split("\0"):
            if item.startswith("EXPERIMENT_DIR="):
                roots.add(item.split("=", 1)[1])
    return roots


# THE CURRENT TraLO RECIPE. Anything else is a DIFFERENT METHOD, not a variant.
#
# Found 2026-09-02 by walking every config.json in all 14 worktrees: FIVE
# distinct TraLO configurations existed across 277 completed `tralo` runs, and
# only 106 were the current one. A corpus assembled by campaign NAME rather
# than by RECIPE silently mixes methods, and it did -- the single unit that
# dissented on all three contrasts turned out to be the one campaign running
# `constraint_grad_mode: clip`.
#
# `results/` is now one recipe. This row is what keeps it that way.
CURRENT_RECIPE = {"constraint_fp32": True, "constraint_grad_mode": "normalize"}

# Arms that take a constraint step. A post-hoc arm has no dose and no grad
# mode, so it is exempt: reading its absent keys as a violation would make this
# row fire on every healthy campaign.
def _is_trained(cfg):
    hp = cfg.get("hyperparams") or {}
    return int(hp.get("constraint_epochs") or 0) > 0


def recipe_of(cfgs):
    """The distinct (fp32, grad_mode) pairs among a campaign's TRAINED arms.

    Returns a sorted list. Empty means the campaign is post-hoc only, which is
    not a violation. More than one entry means the campaign mixes recipes
    INTERNALLY, which is worse than being on the wrong one.
    """
    out = set()
    for c in cfgs:
        if not _is_trained(c):
            continue
        hp = c.get("hyperparams") or {}
        out.add((hp.get("constraint_fp32"), hp.get("constraint_grad_mode")))
    return sorted(out, key=lambda t: (str(t[0]), str(t[1])))


def recipe_verdict(pairs):
    """(status, message) for a campaign's recipe. `pairs` from recipe_of."""
    want = (CURRENT_RECIPE["constraint_fp32"],
            CURRENT_RECIPE["constraint_grad_mode"])
    if not pairs:
        return "ok", "post-hoc only, no constraint step to mis-dose"
    if len(pairs) > 1:
        return "fail", ("MIXES %d recipes internally: %s -- the arms are not "
                        "the same method" % (len(pairs),
                        " ".join("fp32=%s/%s" % p for p in pairs)))
    got = pairs[0]
    if got == want:
        return "ok", "fp32=True/normalize"
    return "fail", ("fp32=%s/%s, not %s/%s -- a DIFFERENT METHOD. Archive it or "
                    "score it separately; do NOT pool it with the current "
                    "corpus" % (got[0], got[1], want[0], want[1]))


def campaign_configs(root):
    out = []
    for f in glob.glob(os.path.join(root, "*", "*", "*", "*", "seed_*",
                                    "config.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def worktree_topology(repo):
    """A `.git` FILE (not directory) means this checkout SHARES an object store.

    Then `git gc` / `prune` / `repack` run in ANY sibling reaches into this
    campaign's objects, so the file-level freeze is not sufficient.
    """
    dot = os.path.join(repo, ".git")
    if os.path.isdir(dot):
        return None
    try:
        with open(dot, encoding="utf-8") as fh:
            line = fh.read().strip()
    except OSError:
        return None
    return line[len("gitdir:"):].strip() if line.startswith("gitdir:") else line


# --------------------------------------------------------------------------
def _row(rows, status, name, detail):
    rows.append((status, name, detail))


def self_test(out=sys.stdout):
    """Gate `is_dispatcher` in BOTH directions on real command lines.

    Every REJECT below was observed on dsisco01 on 2026-09-01 and every one of
    them made the old substring match report `2 running at once -- two
    dispatchers race on the same run dirs` against a single healthy dispatcher.
    """
    w = out.write
    ACCEPT = [
        "/home/dsi/michaer8/anaconda3/envs/optloss/bin/python main.py",
        "python main.py",
        "python -u main.py --filter results/taskwin2",
        "/usr/bin/python3.10 main.py",
    ]
    REJECT = [
        # the wrapper shell -- the dispatcher's own parent, always present
        "bash -c cd ~/optloss-cutwin && python main.py",
        "/bin/sh -c setsid python main.py > log 2>&1",
        # the search that goes looking for it
        "grep --color=auto python main.py",
        "ps -u michaer8 -o pid,ppid,etime,cmd",
        # a RUNNER is not a dispatcher, and it is a child of one
        "python -u -m src.experiments.runner results/taskwin2/.../config.json",
        # a scorer that merely names the file
        "python -m scripts.dose_landed results/taskwin2 main.py",
        "",
    ]
    bad = ([("ACCEPT", a) for a in ACCEPT if not is_dispatcher(a)] +
           [("REJECT", r) for r in REJECT if is_dispatcher(r)])
    ok = not bad
    for kind, line in bad:
        w("  FAIL  should %s: %r%s" % (kind, line, chr(10)))
    if ok:
        w("  PASS  %d dispatcher command lines accepted, %d non-dispatchers "
          "rejected" % (len(ACCEPT), len(REJECT)) + chr(10))
        w("        including the wrapper shell and the pgrep that looks for "
          "it -- both of" + chr(10) + "        which the old substring match "
          "counted as dispatchers" + chr(10))

    # the old pattern MUST have been broken, or this fix is inert
    fooled = [r for r in REJECT if r and DISPATCH_PAT.search(r)]
    if not fooled:
        w("  FAIL  the OLD substring match rejects everything too, so this "
          "change is inert" + chr(10))
        ok = False
    else:
        w("  PASS  liveness: the old substring match was fooled by %d of these "
          "(%s)" % (len(fooled), ", ".join(r.split()[0] for r in fooled))
          + chr(10))

    # and the count that actually gets reported
    procs = [{"pid": 1, "ppid": 0, "args": "bash -c ... python main.py"},
             {"pid": 2, "ppid": 1, "args": "python main.py"},
             {"pid": 3, "ppid": 2, "args": "python -u -m src.experiments.runner c.json"}]
    n = len([q for q in procs if is_dispatcher(q["args"])])
    if n != 1:
        w("  FAIL  a wrapper + dispatcher + runner must count as ONE "
          "dispatcher, got %d%s" % (n, chr(10)))
        ok = False
    else:
        w("  PASS  wrapper + dispatcher + runner counts as ONE, so a healthy "
          "detached" + chr(10) + "        launch no longer reports a race"
          + chr(10))

    # TWO DISPATCHERS: a count is not a collision. Measured on dsisco01
    # 2026-09-04, where the documented one-EXPERIMENT_DIR-per-GPU pattern was
    # reported FAIL on a rig where nothing was wrong.
    disp_cases = [
        ([(1, "results/vitdual2")], OK, "one dispatcher is fine"),
        ([], WARN, "no dispatcher is a WARN, not a FAIL"),
        ([(1, "results/vitdual2"), (2, "results/vitcoin1")], OK,
         "TWO dispatchers on DIFFERENT roots is the multi-GPU pattern, not a "
         "race"),
        ([(1, "results/vitdual2"), (2, "results/vitdual2")], FAIL,
         "two dispatchers on the SAME root is the real defect"),
        ([(1, "results/x"), (2, "results/x/")], FAIL,
         "and a trailing slash does not hide it"),
        ([(1, "results/a"), (2, None)], FAIL,
         "an unreadable EXPERIMENT_DIR fails CLOSED, never OK"),
        ([(1, "results/a"), (2, "results/b"), (3, "results/c")], OK,
         "three dispatchers on three roots is still not a race"),
    ]
    for pairs, want, label in disp_cases:
        got = dispatcher_verdict(pairs)[0]
        good = got == want
        w("  %-4s %s%s" % ("PASS" if good else "FAIL", label, chr(10)))
        ok = ok and good

    # the RECIPE row, both directions
    def cfg(fp32, mode, trained=True):
        return {"hyperparams": {"constraint_fp32": fp32,
                                "constraint_grad_mode": mode,
                                "constraint_epochs": 29 if trained else 0}}
    cases = [
        ([cfg(True, "normalize"), cfg(True, "normalize")], "ok",
         "the current recipe passes"),
        ([cfg(True, "clip")], "fail",
         "grad_mode=clip is refused -- it is a different method"),
        ([cfg(False, "normalize")], "fail",
         "fp32=False is refused -- it lands 69-87% of the dose"),
        ([cfg(True, "normalize"), cfg(True, "clip")], "fail",
         "a campaign MIXING two recipes internally is refused"),
        ([cfg(None, None, trained=False)], "ok",
         "a post-hoc-only campaign is EXEMPT, not a violation"),
    ]
    for cfgs, want, label in cases:
        got = recipe_verdict(recipe_of(cfgs))[0]
        good = got == want
        w("  %-4s %s%s" % ("PASS" if good else "FAIL", label, chr(10)))
        ok = ok and good

    w(chr(10) + "SELF-TEST %s%s" % ("PASSED" if ok else "FAILED", chr(10)))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", action="append", default=None,
                    help="campaign root to check; repeatable. "
                         "Default: every results/* holding runs.")
    ap.add_argument("--repo", default=".", help="repo root to inspect")
    ap.add_argument("--self-test", action="store_true",
                    help="gate the predicates and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = []

    # 1. interpreter -------------------------------------------------------
    if interpreter_is_env(sys.executable):
        _row(rows, OK, "interpreter", sys.executable)
    else:
        _row(rows, WARN, "interpreter",
             "%s -- NOT the optloss env. Base conda is CPU-only torch here; a "
             "campaign launched under it trains on CPU silently." % sys.executable)

    # 2. worktree topology -------------------------------------------------
    shared = worktree_topology(args.repo)
    if shared:
        _row(rows, WARN, "git topology",
             "WORKTREE, object store shared at %s -- never run gc/prune/repack "
             "in ANY sibling while a campaign runs" % shared)
    else:
        _row(rows, OK, "git topology", "standalone .git directory")

    # 3. training path frozen ----------------------------------------------
    dirty = _sh("git -C %s status --porcelain src/ configs/ main.py"
                % args.repo)
    n_dirty = len([x for x in dirty.splitlines() if x.strip()])
    head = _sh("git -C %s rev-parse --short HEAD" % args.repo)
    if n_dirty:
        _row(rows, FAIL, "training path",
             "%d modified file(s) under src/ configs/ main.py at %s -- this "
             "splits code_version" % (n_dirty, head))
    else:
        _row(rows, OK, "training path", "clean at %s" % (head or "?"))

    # 4. processes ---------------------------------------------------------
    procs = read_procs()
    disp = [p for p in procs if is_dispatcher(p["args"])]
    status, detail = dispatcher_verdict(
        [(p["pid"], experiment_dir(p["pid"])) for p in disp])
    _row(rows, status, "dispatchers", detail)

    orphans = orphaned_runners(procs)
    if orphans:
        _row(rows, FAIL, "orphaned runners",
             "%d runner(s) alive with no dispatcher: %s -- they keep writing "
             "into run directories a relaunch will claim"
             % (len(orphans), ", ".join(str(p["pid"]) for p in orphans)))
    else:
        _row(rows, OK, "orphaned runners", "none")

    # 5. GPUs --------------------------------------------------------------
    me = _sh("whoami").strip()
    apps = read_gpu_apps()
    if apps:
        bad = shared_gpus(apps, me)
        mine = sorted({a["gpu"] for a in apps if a["user"] == me})
        if bad:
            _row(rows, FAIL, "gpu sharing",
                 "sharing with another user on: %s" % ", ".join(bad))
        else:
            _row(rows, OK, "gpu sharing",
                 "%d gpu(s) mine, none shared" % len(mine))
        if len(mine) > 2:
            _row(rows, FAIL, "gpu count",
                 "%d in use, house limit is 2" % len(mine))
    else:
        _row(rows, WARN, "gpu", "no compute apps visible")

    # 6. campaigns ---------------------------------------------------------
    owned = dispatcher_roots(procs)
    roots = args.campaign
    if not roots:
        roots = sorted(d for d in glob.glob(os.path.join(args.repo, "results",
                                                         "*"))
                       if os.path.isdir(d))
    for root in roots:
        cfgs = campaign_configs(root)
        if not cfgs:
            continue
        name = os.path.basename(root.rstrip(os.sep))
        counts = {}
        for c in cfgs:
            counts[c.get("status", "?")] = counts.get(c.get("status", "?"), 0) + 1
        done = [c for c in cfgs if c.get("status") == "completed"]
        uniform, seen = code_version_uniform(done)
        summary = " ".join("%s=%d" % kv for kv in sorted(counts.items()))
        n_running = counts.get("running", 0)
        if not uniform:
            _row(rows, FAIL, "campaign %s" % name,
                 "%s -- SPLIT ACROSS %d commits %s; arms are not comparable"
                 % (summary, len(seen), sorted(seen)))
        elif owned and stale_running(name, n_running, owned):
            _row(rows, WARN, "campaign %s" % name,
                 "%s -- %d run(s) marked RUNNING with no dispatcher on this "
                 "root. They died; the status lies until a dispatcher starts "
                 "here and resets them to pending" % (summary, n_running))
        else:
            _row(rows, OK, "campaign %s" % name, summary)
        rst, rmsg = recipe_verdict(recipe_of(cfgs))
        if rst != "ok":
            _row(rows, FAIL, "recipe %s" % name, rmsg)

    # 7. disk --------------------------------------------------------------
    df = _sh("df -h ~ | tail -1")
    if df:
        parts = df.split()
        avail = parts[3] if len(parts) > 3 else "?"
        pct = parts[4].rstrip("%") if len(parts) > 4 else "0"
        try:
            bad_disk = int(pct) >= 90
        except ValueError:
            bad_disk = False
        _row(rows, FAIL if bad_disk else OK, "disk",
             "%s free (%s%% used)" % (avail, pct))

    # ---------------------------------------------------------------- print
    width = max(len(r[1]) for r in rows) if rows else 10
    print("RIG STATUS -- every row here is a failure mode this project has hit")
    print("=" * 78)
    for status, name, detail in rows:
        print("  %-4s %-*s  %s" % (status, width, name, detail))
    print("=" * 78)
    n_fail = sum(1 for r in rows if r[0] == FAIL)
    n_warn = sum(1 for r in rows if r[0] == WARN)
    print("  %d FAIL, %d WARN, %d OK"
          % (n_fail, n_warn, sum(1 for r in rows if r[0] == OK)))
    if n_fail:
        print("  A FAIL means do not launch, and do not read a number off a "
              "campaign, until it is cleared.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
