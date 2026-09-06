"""CAN THIS CAMPAIGN'S RUNS ACTUALLY READ THEIR DATASET, FROM THIS TREE?

The last check before dispatch, and it exists because on 2026-09-06 a freshly
created worktree passed `run_campaign --step verify` AND `--step launch` with
every gate GREEN, then failed 24 runs in 120 seconds on

    FileNotFoundError: data/iwildcam/oodslice/train_images.npy

The `.npy` files are gitignored -- they are 3.0 GB and 443 MB -- so
`git worktree add` produces a tree carrying only the tracked `*_meta.csv`. Every
one of the fourteen pre-existing worktrees had the arrays copied or symlinked in
by hand at creation, so nobody had ever created a new one and this failure mode
had never happened. It cost nothing that time only because the dispatcher was
watched; unwatched it would have burned the whole grid to `failed` and come back
looking merely unfinished, which is `smoke_arms`'s founding failure mode wearing
different clothes.

WHY THE EXISTING GATES DID NOT CATCH IT. `gate:data` is real and would have
caught it, but it lives in `run_campaign`'s FIRST step, `stage` -- the one that
runs "before a config exists". A campaign generated in one sitting and launched
in the next skips straight to `verify`, and neither `verify` nor `launch` looks
at the dataset at all. `launch` is the last thing standing between a config and
a GPU-hour, so the check belongs there too. Duplicating it is the point: a gate
that only fires in a step people skip is not a gate.

WHY NOT REUSE `src.utils.data_loader`. A campaign worktree is PINNED at the
commit its configs were generated from, and a scorer that imports `src/` cannot
run in a tree whose `src/` predates the names it needs -- the same reason
`floors` exists as its own module. The filenames are restated here, with the
loader's line numbers, and `--self-test` asserts the list is non-empty.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# What `src/utils/data_loader.py:154-158,194` actually opens. Restated rather
# than imported, deliberately -- see the module docstring.
REQUIRED = ("train_images.npy", "train_labels.npy",
            "test_images.npy", "test_labels.npy",
            "train_meta.csv", "test_meta.csv")


def data_dirs(root):
    """Every distinct data_dir the configs under `root` will try to read."""
    out = {}
    for cfg in sorted(glob.glob(os.path.join(root, "*/*/*/*/*/config.json"))):
        try:
            c = json.load(open(cfg))
        except (ValueError, OSError):
            continue
        d = (c.get("dataset_config") or {}).get("data_dir")
        if d:
            out.setdefault(d, []).append(cfg)
    return out


def check_dir(data_dir, base):
    """(missing, empty) filenames under `base`/`data_dir`.

    Sizes are taken with `os.stat`, which FOLLOWS symlinks -- the arrays are
    normally symlinked between worktrees, and a dangling symlink is exactly the
    case that must read as missing rather than present.
    """
    missing, empty = [], []
    for name in REQUIRED:
        p = os.path.join(base, data_dir, name)
        if not os.path.exists(p):          # False for a dangling symlink
            missing.append(name)
            continue
        try:
            if os.stat(p).st_size == 0:
                empty.append(name)
        except OSError:
            missing.append(name)
    return missing, empty


def analyse(root, base=".", out=sys.stdout):
    w = out.write
    dirs = data_dirs(root)
    if not dirs:
        w("no config under %s names a data_dir -- nothing to check, and that\n"
          "is itself suspicious for a staged campaign.\n" % root)
        return 1
    bad = 0
    for d, cfgs in sorted(dirs.items()):
        missing, empty = check_dir(d, base)
        if not missing and not empty:
            w("  ok       %-42s %d config(s)\n" % (d, len(cfgs)))
            continue
        bad += 1
        w("  MISSING  %-42s %d config(s) would fail\n" % (d, len(cfgs)))
        for name in missing:
            w("             absent or dangling: %s\n" % name)
        for name in empty:
            w("             present but ZERO bytes: %s\n" % name)
    if bad:
        w("\n  %d data_dir(s) unreadable from %s.\n"
          % (bad, os.path.abspath(base)))
        w("  The arrays are gitignored, so a FRESH WORKTREE has only the\n"
          "  tracked *_meta.csv. Symlink them from a worktree that has them,\n"
          "  pointing at the REAL file rather than at another symlink:\n")
        w("      SRC=<a worktree>/%s\n" % list(dirs)[0])
        w("      for f in $(readlink -f $SRC/*.npy); do \\\n"
          "          ln -s $f <this worktree>/%s/; done\n" % list(dirs)[0])
        return 1
    w("\n  every data_dir readable: %d dir(s), %d file(s) each.\n"
      % (len(dirs), len(REQUIRED)))
    return 0


def self_test(out=sys.stdout):
    import shutil
    import tempfile
    checks = []
    tmp = tempfile.mkdtemp(prefix="datapresent_")
    try:
        base = os.path.join(tmp, "tree")
        dd = "data/ds/slice"
        os.makedirs(os.path.join(base, dd))

        # 1. NEGATIVE CONTROL: nothing present at all must FAIL. Without this
        #    the check could be vacuous and still pass the happy path below.
        missing, empty = check_dir(dd, base)
        checks.append(("an empty data dir reports every required file missing",
                       set(missing) == set(REQUIRED) and not empty))

        # 2. The exact shape of the real failure: the tracked CSVs are there
        #    and the gitignored arrays are not. This must FAIL, not pass.
        for n in ("train_meta.csv", "test_meta.csv"):
            open(os.path.join(base, dd, n), "w").write("x\n")
        missing, _ = check_dir(dd, base)
        checks.append(("THE REAL CASE: meta CSVs present, .npy absent -> still "
                       "missing %d" % len(missing),
                       len(missing) == 4 and all(m.endswith(".npy")
                                                 for m in missing)))

        # 3. All present and non-empty must PASS -- the positive control that
        #    stops this from being a check that refuses everything.
        for n in REQUIRED:
            open(os.path.join(base, dd, n), "w").write("x\n")
        missing, empty = check_dir(dd, base)
        checks.append(("a complete data dir passes", not missing and not empty))

        # 4. NEGATIVE CONTROL: a ZERO-BYTE file is present but unusable, and
        #    `os.path.exists` alone would call it fine.
        open(os.path.join(base, dd, "train_images.npy"), "w").close()
        missing, empty = check_dir(dd, base)
        checks.append(("a zero-byte array is caught as empty, not present",
                       empty == ["train_images.npy"] and not missing))

        # 5. NEGATIVE CONTROL: a DANGLING symlink. This is the likely form of
        #    the bug in practice -- the arrays are symlinked between worktrees,
        #    and `os.path.lexists` would call a broken link present.
        open(os.path.join(base, dd, "train_images.npy"), "w").write("x\n")
        gone = os.path.join(tmp, "gone.npy")
        open(gone, "w").write("x\n")
        link = os.path.join(base, dd, "test_images.npy")
        os.remove(link)
        try:
            os.symlink(gone, link)
            os.remove(gone)
            missing, _ = check_dir(dd, base)
            checks.append(("a DANGLING symlink reads as missing, not present",
                           "test_images.npy" in missing))
        except (OSError, NotImplementedError):
            checks.append(("dangling-symlink check skipped (no symlink "
                           "privilege on this host)", True))

        checks.append(("the required-file list is not empty", len(REQUIRED) > 0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("", file=out)
    for label, good in checks:
        print("  %-66s %s" % (label[:66], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("SELF-TEST PASSED" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("root", nargs="?")
    a.add_argument("--base", default=".",
                   help="tree the data_dir is resolved against (default cwd, "
                        "which is what the runner itself uses)")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.root:
        a.error("give a campaign root (or --self-test)")
    return analyse(args.root, args.base)


if __name__ == "__main__":
    sys.exit(main())
