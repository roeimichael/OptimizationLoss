"""The per-cell rule-4 report, in ONE place. Nothing else belongs here.

WHY THIS FILE EXISTS (2026-09-11). `_cell_of` and `_per_cell_report` were
duplicated BYTE-IDENTICALLY -- 84 lines -- in `graph_probe` and `scope_probe`,
and the duplication was managed rather than removed: `_cell_of`'s own docstring
said *"the two must stay byte-identical"* and
`tests/gates/test_g6_results.py` asserted it with `inspect.getsource`. That
gate was written for a good reason, which the docstring also recorded: a
predicate copied into two files is how the lambda=0 stream test drifted.

🔑 **BUT A TEST THAT ENFORCES A DUPLICATION IS A COMMENT WITH AN ASSERT
ATTACHED.** It makes drift loud; it does not make drift impossible, and it
costs a check on every run forever. One definition makes the failure mode
unreachable instead of detectable. Same class as FRAMEWORK 2(z81) (an exemption
whose reason is a ticket), 2(z94) (a DEFERRED list), 2(z95) (a correction
recorded in a data file with no reader) and 2(z96) (a fix living on one disk):
the artefact looks like diligence, and the thing it describes is still there.

⚠️ THIS MODULE FOLLOWS `floors.py`'s RULE: it must not import anything that
reaches `src/`, because the scorers that use it have to run in a campaign
worktree PINNED at an older commit. Its only module-level import is `os`.
`seeds_needed` stays a function-local import inside a `try`, exactly as it was,
so a checkout where `frozen_head_probe` will not import still prints the table
with a `-` in the seeds column rather than dying.
"""

import os


def cell_of(run_dir):
    """(backbone, dataset, cap, ARM) for a run, from its path.

    <root>/<Backbone>/<dataset>/<cap>/<arm>/<seed>. Returns None when the path
    is too shallow to say, which is honest: an unknown cell must not silently
    join a known one.

    🛑 THE ARM IS PART OF THE CELL, AND IT USED TO BE DROPPED (2026-09-07).
    This returned `parts[-5:-2]`, i.e. (backbone, dataset, cap), so every arm's
    runs at one cap collapsed into ONE key -- and the rule-4 guard below then
    printed "ONE CELL, so the pooled block above is a legal aggregate" over a
    mixture of six methods. It certified precisely the thing it exists to
    catch. Rule 4 is explicit: the atomic cell is (dataset, backbone, cap,
    METHOD) over seeds. FRAMEWORK 2(z52).
    """
    # NOT `abspath`. It expands a relative path against the CWD, so the
    # "too shallow to say" branch below was unreachable for any relative
    # input: `cell_of("seed_1")` returned a cell built out of whatever
    # directories happened to be above the working directory. The docstring
    # promised abstention and the code could not deliver it. Found 2026-09-07
    # by the gate written for the arm-in-the-key fix.
    parts = os.path.normpath(run_dir).split(os.sep)
    return tuple(parts[-5:-1]) if len(parts) >= 5 else None


def per_cell_report(names, rows, keys):
    """RULE 4: never pool across backbones, cap levels or datasets.

    The pooled block above keys on the REGIME NAME only, so a `--campaign`
    spanning three backbones and two cap levels produced ONE line per regime
    and ran a sign test over it. That is the aggregation this project has
    retracted a result over three times, and a direction-closing verdict was
    published off it. The pooled line stays so the published number remains
    reproducible; this block is what says whether it was legal.
    """
    cells = {}
    for i, nm in enumerate(names):
        cells.setdefault(cell_of(nm) or ("?", "?", "?", "?"), []).append(i)
    if len(cells) <= 1:
        print("")
        print("  ONE CELL (%s) -- backbone, dataset, cap AND arm -- so the "
              "pooled block above is a legal aggregate."
              % ("/".join(sorted(cells)[0]) if cells else "none"))
        return cells
    print("")
    print("  *** THE BLOCK ABOVE POOLS %d CELLS, AND RULE 4 FORBIDS THAT."
          % len(cells))
    print("      A backbone or a cap level is not a replicate: the")
    print("      unconstrained count, the ranking quality and K all move with")
    print("      both. Count CELLS, never runs.")
    try:
        from scripts.frozen_head_probe import seeds_needed
    except Exception:
        seeds_needed = None
    print("  %-46s %4s %s %8s %6s %7s"
          % ("cell", "n", "  ".join("%12s" % k[:12] for k in keys),
             "sd", "sign", "seeds"))
    for c in sorted(cells):
        idx = cells[c]
        v0 = [rows[keys[0]][i] for i in idx]
        m0 = sum(v0) / float(len(v0))
        sd0 = (sum((x - m0) ** 2 for x in v0) / max(1, len(v0) - 1)) ** 0.5
        pos = sum(1 for x in v0 if x > 0)
        need = ("%7s" % (seeds_needed(m0, sd0)
                         if seeds_needed and m0 > 0 and sd0 > 0 else "-"))
        vals = ["%+12.2f" % (sum(rows[k][i] for i in idx) / float(len(idx)))
                for k in keys]
        print("  %-46s %4d %s %8.2f %3d/%-2d %s"
              % ("/".join(c)[-46:], len(idx), "  ".join(vals), sd0, pos,
                 len(idx), need))
    n_pos = sum(1 for c in cells
                if sum(rows[keys[0]][i] for i in cells[c]) > 0)
    print("      CELL sign test on `%s`: %d of %d positive. That is the sample"
          % (keys[0], n_pos, len(cells)))
    print("      size, not %d run(s), and `seeds` is per cell at 80%% power."
          % len(names))
    return cells
