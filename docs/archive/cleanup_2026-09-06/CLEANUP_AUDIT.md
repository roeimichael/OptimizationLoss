> 🛑 **ARCHIVED -- HISTORY, NOT INSTRUCTIONS.** (banner added 2026-09-11)
> The record of ONE cleanup pass, 2026-09-06. A finding here was true of the
> tree on that date and says nothing about today's.
> `docs/FRAMEWORK.md` is the ONLY operational document. Where this file
> disagrees with it, FRAMEWORK wins and this file is wrong. Do not run
> anything, and do not quote any figure, on the strength of this page.

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

## 7. VERIFICATION (Phase 4) -- ALL FOUR GATES GREEN

Run on the tree AFTER the change, in the order the rulebook prescribes:

```
$ python -m compileall -q src scripts configs tests main.py
(silent -- success)

$ python -m scripts.audit_config
  OK -- every unresolvable read is in audit/scoring code
  that iterates a declared key list, not in src/.
==============================================================================
No hallucinated keys: every emitted value has a reader.

$ python -m pytest tests -q
583 passed, 1 skipped, 7 warnings in 233.78s (0:03:53)

$ python -m scripts.preflight --before-launch
39 passed, 24 deselected, 3 warnings in 11.51s
PRE-FLIGHT -- 4 stage(s): data, budget, model, grid
```

The suite is **identical to the pre-change baseline** (583 passed, 1 skipped),
so the one edit is provably inert with respect to every gate the project owns.

---

## 8. INDEPENDENT PRESERVATION AUDIT

A fresh agent that performed none of the editing re-derived the outcome from git
alone, instructed to look adversarially for a violation. Verdict: **CLEAN.**

| Check | Result |
|---|---|
| Files deleted | **none** -- `git diff --diff-filter=D` empty |
| Files renamed | **none** -- `git diff --diff-filter=R` empty |
| Training path (`src/ configs/ main.py tests/`) | **untouched**, diff and status both empty |
| 17 protected paths still tracked | **all present** |
| Protected directory counts vs `2af31614` | **all identical** (tables 11, tables_rev 11, tables_clean 11, figures 6, paper data 22, paper scripts 11, launchers 4, scripts 62) |
| `docs/FRAMEWORK.md` diff | **exactly 4 hunks**, each a pure path substitution; no prose, number or claim altered |
| Removed lines anywhere in the session | **4**, all the same substitution -- no measurement, p-value or rejected-idea entry lost |
| Tracked file count | 266 -> 267, the +1 being this file |
| `scripts/` dead-list spot-check (`verify_caps`, `scope_probe`, `reset_crashed`) | all fail tests 1 and 3 independently -- "dead list empty" corroborated |

It also caught one real process violation: `CLEANUP_STATE.md` was still at its
pristine seed value while Phases 1-4 had demonstrably run. That was deliberate --
the state file was held until this audit returned, so that `STATUS: COMPLETE`
would never be written on an unverified tree -- but the auditor is right that the
window existed, and rulebook 0.4 names an un-updated state file as the one way
the outer loop can spin forever. Closed in the final commit.

---

## PRE-EXISTING

None. The clean-tree baseline was captured BEFORE any change:

```
$ python -m pytest tests -q          # on the untouched tree
583 passed, 1 skipped, 7 warnings in 241.17s (0:04:01)
```

---

## DISPOSITION OF THE DEFERRED LIST -- 2026-09-10

🛑 **A DEFERRED LIST IS A DEFECT WITH A COMMENT ATTACHED, AND THIS ONE PROVED
IT.** Written 2026-09-06, read by nobody for four days, and item 5 was a live
data-loss hazard the whole time. The list was correct; the mechanism that was
supposed to bring it back to a human did not exist. That is the same shape as
the exemption-whose-reason-is-a-ticket rule already in CLAUDE.md.

Every item is now discharged or has a named owner. **Five of the seven were
closed the day the list was finally read**, which is the argument for reading
one rather than growing one.

| # | item | disposition |
|---|---|---|
| 1 | five stale `docs/launch_*.sh` refs | ✅ **DISCHARGED.** Registry in MISSION `0-LAUNCH`, plus a gate in `tests/test_lessons_learned.py` that fails on any doc-named launcher which neither exists nor appears there. Mutation-tested: a NEW dead path and a deleted registry row both turn it red. |
| 2 | `PROVENANCE.md:5` companion path | ✅ **DISCHARGED.** Repointed to `docs/archive/PAPER_REVISION_TRACKER.md` with a note that it is history, not a queue. This was only possible BECAUSE item 5 was fixed first -- the note itself said repointing would hand a clone a link to a file it did not have. |
| 3 | `tab_granular_asym.tex` twin pair | ⬜ **STILL OPEN.** Unreferenced by any build, inside a wholesale-preserved directory. Unchanged: still a judgement about whether the preserve list licenses deleting dead files within it. |
| 4 | tracked Hypothesis cache blob | ✅ **DISCHARGED.** `git rm --cached` and `.hypothesis/` added to `.gitignore`. The file stays on disk. The rule that blocked this was the 2026-09-06 run's OWN rulebook, not a standing one. |
| 5 | bare `archive/` matches at any depth | ✅ **DISCHARGED, and it was the serious one.** `!docs/archive/` added -- naming the DIRECTORY, because a file-level negation cannot re-include anything under an excluded directory, which is why the pre-existing `!docs/**/*.tex` and `!docs/**/*.pdf` never worked. **22 files, 1.7 MB, recovered into git**, the note's own count exactly: `main.tex` (150 KB), `PAPER_REVISION_TRACKER.md` (41 KB), `BLUE_REVISION_BRIEFING.md` (36 KB while its own `.tex` and `.pdf` were tracked), both `review/` rounds, all six `track_b/` files, `MEETING_BRIEF.tex`/`.pdf`. "Archive, don't delete" had been silently deleting. |
| 6 | duplicated helpers in `scripts/` | 🟡 **PARTLY, and one bullet was a LIVE DEFECT, not a tidiness item.** See below. |
| 7 | duplicated dual math in `src/` | ⬜ **STILL OPEN, correctly.** `src/` is frozen while campaigns run; `code_version` is a git hash. Not actionable today. |

### Item 6, expanded -- the bullet that was not cosmetic

* **`null_of`'s hardcoded family list (was: "`paper_rows` has a hardcoded
  4-family list ... if a fifth family is added, one of them silently goes
  wrong").** The fifth family already existed: `select`, with its own
  `select_null`. But the real defect was worse and the note did not reach it.
  `null_of` returned `<fam>_null` **by string construction**, `build()` skips a
  contrast whose reference arm is absent from the cell, and **no campaign in
  the corpus runs `alm_null` / `fioretto_null` / `hounie_null`** -- `fmow1`
  carries 19 arms and none of the three. So `vs_null`, which `CONTRASTS` itself
  calls "the only contrast that attributes an effect to the CONSTRAINT rather
  than to the regime", was emitted for `tralo` and every `tralo_*` variant and
  **silently omitted for all three rival duals, on every campaign.** Not a wrong
  number -- a missing row, in the tool that says what may be written. The
  self-test PINNED the broken expectation and passed green.
  ✅ Fixed to `family_split`'s two rules (dedicated twin if the cell ran one,
  else `protocol.yml`'s `null_sibling`), roots derived from the protocol rather
  than restated, mutation-tested 4/4 with an end-to-end check the resolver test
  could not make.
* **`seeds_needed`.** The note said two implementations; there are **FOUR** --
  `paper_rows`, `paired_noise`, `deployed_h2h`, `frozen_head_probe` -- with
  three different signatures. Audited on identical inputs: the CONSTANT agrees
  (7.85 against the exact `(z_{a/2}+z_b)^2` = 7.848880, 0.0143% apart) and the
  FORMULA agrees. ✅ The one divergence was ROUNDING -- `paired_noise` returned a
  raw float where the other three ceil -- and it lands precisely on the figures
  that decide something: at 2607 and 546 seeds rounding is noise, at "7-8 at
  K/n = 0.9" it is the whole answer. Now ceils, gated, and the affected figures
  were already withdrawn as UNVERIFIED (`iwc3`, `scorable=False`).
* **`_cell_of` / `_per_cell_report`** remain byte-identical between
  `graph_probe.py` and `scope_probe.py` (28 and 52 lines, verified 2026-09-10).
  ⬜ **STILL OPEN** -- a merge here is a behaviour risk against two live probes
  and buys nothing but line count.
