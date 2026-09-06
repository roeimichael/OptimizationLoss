# CLEANUP_AUDIT.md

Autonomous repository decontamination, run under `CLEANUP_PROMPT.md`.
Universe: `git ls-files` = **266 tracked files**. Everything else on disk is
gitignored and therefore read-only (a deletion there is unrecoverable).

**Headline: this repository is already clean.** One stale path was fixed. Nothing
was deleted. The evidence for every "keep" decision is recorded below, because
under this rulebook a KEEP with no evidence is indistinguishable from a KEEP with
no inspection.

---

## 0. WHY THE HARVEST IS THIN, AND WHY THAT IS THE CORRECT RESULT

Commit `e7d9e893` ("cleanup: delete 94 stale files, and generalise the gate that
caught the overreach", 2026-09-02) already ran this exact sweep. It took the
tracked set from 338 to 244 files and docs from 32,096 to 26,200 lines. Its own
commit message records the finding this run reproduces independently:

> `scripts/dead_code.py` finds THREE dead symbols in configs+src+scripts, and an
> orphan audit (named in no live doc, imported by nothing, exercised by no test)
> finds zero orphaned scripts -- so the tracked Python is not where the
> redundancy is.

It also records that its FIRST attempt over-reached: it deleted all eleven launch
scripts and **eight gates went red**, and four launchers had to be restored. That
is the failure mode this audit is built to avoid.

---

## 1. FILE CLASSIFICATION (266 tracked)

| Bucket | Count | Basis |
|---|---|---|
| PROTECTED -- training path (`src/ configs/ main.py tests/`) | 142 | 2a. `code_version` is a git hash; a comment splits a campaign |
| PROTECTED -- operational docs + README/CLAUDE | 7 | 2b. Repetitive *on purpose* |
| PROTECTED -- the paper (`docs/paper/**`) | 83 | 2c. `main_edited_by_roei.tex` is the paper of record |
| PROTECTED -- failure record (`docs/archive/**`) | 17 | 2e + "history: leave it in place" |
| ACTIVE -- `scripts/**` | 62 | 2d, all four tests run per module |
| ACTIVE -- root config/infra | 9 | `.gitignore`, `.gitattributes`, `pytest.ini`, `requirements.txt`, `run_autoclean.sh`, `.claudeignore`, `.github/workflows/preflight.yml`, `data/*_meta.csv` |
| CANDIDATE | **0** | see sections 2-6 |

---

## 2. `scripts/` -- 62 MODULES, DEAD LIST **EMPTY**

The 2d test is conjunctive: a module is dead only if ALL FOUR hold. Result:

- **Test 1 (named in `CLAUDE.md` / `docs/` / `tests/` / `scripts/` / `.github/`)
  fails to clear for 62 of 62.** Minimum observed 1 hit; every minimum-hit case
  (`arm_identity_check`, `boundary_probe`, `tralo_wins`) is a live reference in
  `CLAUDE.md` or `docs/FRAMEWORK.md`, not a self-mention.
- **Test 3 (no entry point) clears for only 2 of 62**: `scripts/floors.py` and
  `scripts/score_arm.py` are the only modules with no `if __name__ == "__main__"`.
  Both are pure library helpers, and both fail tests 1 and 2 decisively:

```
scripts/floors.py      imported by scripts/deployed_h2h.py:58,
                       scripts/sensitivity_screen.py:146, scripts/tralo_wins.py:44
                       named in tests/test_lessons_learned.py:1769,1778,1802,1810
                       7 hits in docs/FRAMEWORK.md
scripts/score_arm.py   imported by scripts/frozen_head_probe.py:112,
                       scripts/full_panel.py:50, tests/test_baseline_fidelity.py:1010
                       2 hits in docs/FRAMEWORK.md
```

**Verdict: 62/62 KEEP.** This is the outcome 2d predicts ("Expect this list to be
SHORT or EMPTY") and it must not be stretched past.

### 2b. Dead SYMBOLS (report only -- NOT removed)

`python -m scripts.dead_code --paths configs src scripts` -- 823 definitions
scanned, 4 unreferenced:

| Symbol | Location | Why it is KEPT |
|---|---|---|
| `MEASURED_CCP = 0.9954` | `scripts/ceiling_screen.py:72` | **A MEASUREMENT** (iwc3, `tralo_null` vs `clip`), sitting directly above the comment explaining why holding `p` fixed was wrong. The mission says keep every negative result; this is one |
| `SEED_AXIS = "seed"` | `scripts/cell_table.py:44` | Documentary pair with `CELL_KEY` on the next line -- names the one collapsible axis |
| `PYTEST_RAN_AND_FAILED = 1` | `scripts/run_campaign.py:152` | Half of a documented pair with `PYTEST_COULD_NOT_RUN`; the comment explains that treating exit 2 as failure "reports RED on a healthy campaign" |
| `print_status_summary` | `src/utils/filesystem_manager.py:111` | In `src/` -- **REPORT ONLY** per 2a, must not be touched |

Deleting a named constant whose value is a measured number is exactly the
"manufacture work" failure section 3 warns against.

---

## 3. `docs/` -- NOTHING REMOVABLE

22 tracked non-paper docs. 5 are the operational law (2b), 17 are `docs/archive/`.

**The four `docs/archive/launchers/*.sh` are LOAD-BEARING, not decorative.**
`tests/test_pipeline.py:8206` globs `docs/archive/launchers/launch_*.sh`, extracts
each `gen_campaign` `--caps`/`--models` invocation, classifies every (model, cap)
cell against `configs/task_windows.yml`, and asserts the banner discipline.
`tests/test_baseline_fidelity.py:4615` globs the same set. Deleting any one of
them turns the suite red -- this is precisely the over-reach that `e7d9e893` had
to undo. KEEP is required for test correctness, not merely conservative.

Every other `docs/archive/` file carries a measurement preserved nowhere else --
`AUDIT_2026-07-31.md` (the budget-fill artifact: `+0.0377` as-run vs `-0.0016`
budget-equalised), `RESULTS_SUMMARY.md` (per-baseline paired-bootstrap p-values),
`WARMUP_ABLATION.md` (the w50 to w1 cc-F1 gap, whose generating scripts are gone,
so this is the only surviving copy). `PROFESSOR_REVIEW.md` is quoted verbatim by
`docs/FRAMEWORK.md:10741`. **All KEEP.**

---

## 4. `docs/paper/` -- BUILD GRAPH INTACT, NOTHING REMOVABLE

All four manuscripts parse clean: **no broken `\input`, `\includegraphics` or
`\bibliography` target in any of them.** All 6 figure PDFs are reached by at least
one manuscript. The three table directories are genuinely cross-wired --
`tab_ccf1.tex` is pulled from three *different* directories by three different
manuscripts and the three copies differ by 6-95 lines. Confirmed 2c is right that
they are parallel versions, not duplication.

Confirmed `CLAUDE.md`'s generator claim statically (generators NOT executed -- they
write tracked files): 8 tables regenerate via 4 `make_*.py`, and
`tab_ablation_complete` / `tab_deploy` / `tab_oct_backbone` have no generator, as
documented. Also newly noted: **every generator writes only into `tables/`** --
`tables_rev/` and `tables_clean/` are hand-maintained forks, so the
"regenerates byte-for-byte" guarantee applies to `tables/` alone.

---

## 5. DUPLICATES -- ONE PAIR, AND IT IS PROTECTED

Across all 266 tracked files there are exactly two byte-identical groups:

1. **10 empty `__init__.py`** -- package markers. Removing any breaks an import.
2. **`docs/paper/tables_rev/tab_granular_asym.tex` = 
   `docs/paper/tables_clean/tab_granular_asym.tex`** (3646 bytes). Neither exact
   path is `\input` by any manuscript; the only live reference is
   `main.tex:1548 \input{tables/tab_granular_asym}`, a *different* file.

Pair 2 is genuinely unreferenced -- but it sits inside `tables_rev/` and
`tables_clean/`, which 2c names in the preserve list **wholesale**. Rulebook 0.1
requires the conservative branch on a genuine conflict. **KEPT.** See `## DEFERRED`.

---

## 6. ACTION TAKEN -- ONE STALE PATH FIXED

`docs/launch_uniform.sh` was **renamed** (not deleted) to
`docs/archive/launchers/launch_uniform.sh`, confirmed as a git rename with 100%
similarity:

```
$ git log --diff-filter=R --oneline --name-status --all | grep launch_uniform
R100    docs/launch_uniform.sh    docs/archive/launchers/launch_uniform.sh
```

`docs/FRAMEWORK.md` still pointed four readers at the old path, including the
pointer `Launch: docs/launch_uniform.sh` -- a link a reader would follow and not
find. Fixed at `docs/FRAMEWORK.md:3915, 5839, 6148, 6438`. This is section 3
bullet 4 ("broken intra-repo links and paths in docs pointing at files that no
longer exist"), the referent still exists, and the repo has already adopted the
new location as canonical (`tests/test_pipeline.py:59`: "The launchers were
archived to `docs/archive/launchers/`").

Not touched: the same string inside `tests/` (read-only, 2a) and inside
`docs/archive/launchers/launch_uniform.sh:428` (the file referring to itself;
`tests/test_baseline_fidelity.py` parses that comment block).

---

## DEFERRED -- for a human, with evidence

1. **Five stale references to files that were DELETED, not moved.** The fix is an
   annotation, and annotating the law is a scientific judgement, not a janitorial
   one -- especially as this project's documented style is to leave the old claim
   and add a correction beside it, which `docs/MISSION.md:1196` already does for
   `margin2` ("NOT staged -- checked 2026-09-02, no `margin2` exists on disk").
   - `docs/MISSION.md:1216, 1460` -> `docs/launch_margin2.sh`
   - `docs/MISSION.md:1453` -> `docs/launch_vitdom1.sh`
   - `docs/FRAMEWORK.md:5811, 6745` -> `docs/launch_margin1.sh`
   - `docs/FRAMEWORK.md:6758` -> `docs/launch_iwc4.sh`
   All recoverable via `git show e7d9e893^:<path>` if the knob is still wanted.

2. **`docs/paper/data/PROVENANCE.md:5`** calls `docs/PAPER_REVISION_TRACKER.md` a
   live "companion ... which tracks the open work". That path is gone, but the
   content is on disk at `docs/archive/PAPER_REVISION_TRACKER.md` -- which is
   **gitignored**, so repointing a tracked doc at it would give a fresh clone a
   link to a file it does not have. Genuinely ambiguous; left alone. Neighbouring
   dead references in the same file *are* annotated ("SCRIPT DOES NOT EXIST"), so
   the consistent fix is that annotation.

3. **The `tab_granular_asym.tex` twin pair** (section 5) -- unreferenced by any
   build, but inside a wholesale-preserved directory. A human should decide
   whether 2c's preserve list licenses deleting dead files *within* those
   directories.

4. **`.hypothesis/unicode_data/13.0.0/charmap.json.gz`** (21 KB) is a Hypothesis
   library cache blob, tracked, regenerable, and **not** in `.gitignore`. It was
   swept in by `7f9181b3` ("tests: lock in every invariant..."). It is a textbook
   section 3 "committed build artifact" -- but `.hypothesis/` is on section 1's
   explicit "READ-ONLY, NO EXCEPTIONS" list. Rules conflict; conservative branch
   taken. Suggested human fix: `git rm --cached` it and add `.hypothesis/` to
   `.gitignore`.

5. **`.gitignore:123` is a bare `archive/`, which matches at ANY depth** -- so
   `docs/archive/` is ignored too. 17 files there are tracked only because they
   predate the rule; 22 more (including `docs/archive/README.md`, the index of the
   whole archive) are silently invisible to git. Nothing is broken today, and the
   invisible content is stale, so the rule is arguably doing useful quarantine
   work. But any NEW file added to `docs/archive/` will vanish silently.
   Root-anchoring it (`/archive/`) would fix the surprise and pull 22 stale files
   into the tracked set -- which is why this is a human call, not a cleanup call.

6. **Duplicated helpers inside `scripts/` (unification candidates; section 3
   permits but does not require).** Reported, not merged, because each merge is a
   behaviour risk against a live scorer:
   - `_cell_of()` and `_per_cell_report()` are **byte-identical** between
     `scripts/graph_probe.py:153-163+` and `scripts/scope_probe.py:250-260+`.
   - `seeds_needed(...)` implements the same `7.85 * (sd/effect)**2` in both
     `scripts/paired_noise.py` and `scripts/paper_rows.py`, with different
     signatures -- a drift risk.
   - `null_of` in `scripts/family_split.py` reads `protocol.yml`'s `null_sibling`;
     `scripts/paper_rows.py` has a hardcoded 4-family list for the same job. If a
     fifth family is added, one of them silently goes wrong.

7. **Duplicated dual-method math in `src/`** -- 2a makes this a REPORT, and this
   audit did not open it, because `src/` may not be edited and a proposal to
   refactor the training path mid-campaign is not actionable while campaigns run.

---

## PRE-EXISTING

None. The clean-tree baseline was captured BEFORE any change:

```
$ python -m pytest tests -q          # on the untouched tree
583 passed, 1 skipped, 7 warnings in 241.17s (0:04:01)
```
