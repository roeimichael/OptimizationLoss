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
DISPATCH_PAT = re.compile(r"python\s+main\.py")


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
    dispatchers = [p for p in procs if DISPATCH_PAT.search(p["args"])]
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


def dispatcher_roots(procs):
    """EXPERIMENT_DIR of every live dispatcher, read from /proc.

    Falls back to an empty set off Linux, where the stale check is then skipped
    rather than reported wrongly.
    """
    roots = set()
    for p in procs:
        if not DISPATCH_PAT.search(p["args"]):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", action="append", default=None,
                    help="campaign root to check; repeatable. "
                         "Default: every results/* holding runs.")
    ap.add_argument("--repo", default=".", help="repo root to inspect")
    args = ap.parse_args()

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
    disp = [p for p in procs if DISPATCH_PAT.search(p["args"])]
    if len(disp) > 1:
        _row(rows, FAIL, "dispatchers",
             "%d running at once -- two dispatchers race on the same run dirs"
             % len(disp))
    elif disp:
        _row(rows, OK, "dispatchers", "1 running (pid %d)" % disp[0]["pid"])
    else:
        _row(rows, WARN, "dispatchers", "none running")

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
