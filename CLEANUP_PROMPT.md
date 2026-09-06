# MISSION: Autonomous Repository Decontamination (OptimizationLoss)

You are running headless in a loop. There is no human to ask. Make deterministic
engineering decisions, record them, and exit cleanly so the next invocation resumes.

---

## 0. OPERATIONAL DIRECTIVES (ZERO HUMAN INTERACTION)

1. **Never ask a question.** No confirmations, no choices, no "would you like me to".
   If a call is genuinely ambiguous, take the CONSERVATIVE branch (leave the file
   alone), log the ambiguity in `CLEANUP_AUDIT.md` under `## DEFERRED`, and continue.
2. **Read `CLEANUP_STATE.md` FIRST.** It names the phase and the exact next step.
   Resume there. Do not restart from Phase 1.
3. **Never classify a file by its name.** Read it, or grep for its references,
   before moving, merging or deleting it.
4. **Before exiting for ANY reason** (work done, context pressure, rate limit),
   update `CLEANUP_STATE.md` with the phase and a concrete next step, and commit.
   An un-updated state file is the only way this loop can spin forever.
5. **Commit after every discrete change.** Small commits on the working branch.
   Never `git push`, never touch `main`, never `git reset --hard`, never
   `git clean`, never `git gc` / `prune` / `repack` / `worktree prune` — fourteen
   worktrees share one object store and a repack there can corrupt a live campaign.

---

## 1. THE UNIVERSE: TRACKED FILES ONLY

**`git ls-files` is the complete set of files you may modify, move or delete.**
That is ~260 files. Everything else on disk is READ-ONLY, and this is not a style
preference: those trees are gitignored, so a deletion there is **unrecoverable**.

**READ-ONLY, NO EXCEPTIONS — do not move, rename, delete, or write into:**

```
results/     the live experiment corpus (dom1 dom1b equaldose1 uniform1 taskwin2
             vittask1 vitdual1 ...). Gitignored. Irreplaceable. GPU-months.
evidence/    provenance for 14,524 runs + predictions for 128, as tarballs
data/        the .npy arrays (3.0 GB + 443 MB), gitignored
archive/     already-archived history, gitignored
.venv/  .git/  .claude/  .agents/  .hypothesis/  .pytest_cache/  __pycache__/
*.pdf *.aux *.bbl *.blg *.log *.out    LaTeX build artifacts: regenerable, ignore them
```

Before ANY destructive operation run `git ls-files --error-unmatch <path>`.
If it fails, the file is untracked: **leave it alone.**

---

## 2. STRICT PRESERVATION LIST (tracked, but still protected)

### 2a. The training path — READ, NEVER WRITE

```
src/    configs/    main.py    tests/
```

`code_version` is `git rev-parse HEAD`, and any edit to `src/`, `configs/` or
`main.py` splits a campaign into two non-comparable halves and turns
`check_parity` red. **A comment change counts.** You may READ them, and you may
write FINDINGS about them into `CLEANUP_AUDIT.md` as a proposal for a human. You
may not edit, reformat, deduplicate, or unify helpers across them. If Phase 3
finds duplicated math across `tralo` / `alm` / `fioretto` / `hounie`, that is a
**report**, not a refactor.

`tests/` is likewise read-only: every test in it encodes a failure already paid for.

### 2b. The operational docs — the law, not clutter

```
docs/FRAMEWORK.md    the only operational document (protocol + rejected ideas)
docs/MISSION.md      the resume point
docs/COVERAGE.md     what we have vs what the paper needs
docs/PLAYBOOK.md     what to do when a campaign lands
docs/THEORY.md
CLAUDE.md    README.md
```

These are long and repetitive **on purpose** — each repetition is a failure that
cost a week. Do not "consolidate" them, do not merge them into a `CONTRIBUTING.md`,
do not trim them for concision. `docs/archive/` is history: leave it in place.

### 2c. The paper — and THE TRAP IN THIS BRIEF

⛔ **DO NOT create `archive/legacy_medical/`. There is no legacy medical paper.**

Every manuscript in `docs/paper/` is written on the MedMNIST corpus
(`dermmnist` / `octmnist` / `tissuemnist`). That includes
**`main_edited_by_roei.tex`, which is the paper of record.** A rule that archives
"old medical research paper drafts" deletes the live submission. The current
experiments (`iwildcam`) and the manuscripts are disjoint generations by design;
see `docs/paper/WHICH_CORPUS.md`.

Preserve all of: `docs/paper/*.tex`, `references.bib`, `*.sty`, `*.bst`,
`math_commands.tex`, `figures/`, `tables/`, `tables_rev/`, `tables_clean/`,
`data/`, `scripts/`. `main.tex` is the professor's file — never edit it.

### 2d. The scripts — "zero imports" DOES NOT MEAN DEAD

`scripts/` holds ~62 modules. **Almost none of them are imported by anything.**
They are invoked as `python -m scripts.<name>` from `CLAUDE.md`,
`docs/FRAMEWORK.md` and `docs/PLAYBOOK.md`. An import-graph dead-code pass will
report nearly all of them as orphans, and it will be wrong every time.

A `scripts/*.py` file is dead ONLY if ALL of the following hold, each verified:

1. `grep -rn "scripts\.<name>\|scripts/<name>" CLAUDE.md docs/ tests/ scripts/ .github/`
   returns nothing, AND
2. nothing imports it (`python -m scripts.dead_code --paths scripts`), AND
3. it defines no `--self-test` and no `if __name__ == "__main__"` entry point, AND
4. it is not named in `docs/FRAMEWORK.md` section 2 as the receipt for a closed
   direction.

If any one is unmet: **KEEP**. Log it as kept, and why.

---

## 3. WHAT YOU ARE ACTUALLY ALLOWED TO CHANGE

The legitimate targets, within the tracked set:

- Genuinely orphaned scratch `.md` / `.txt` at repo root or in `docs/` that no
  other tracked file references and that record no measurement.
- Duplicated files: byte-identical or near-identical copies of the same doc or
  script in two tracked locations — keep the referenced one, delete the other.
- Committed build/junk artifacts that are regenerable and belong in `.gitignore`
  instead (`.aux`, `.bbl`, `.log`, `__pycache__`, editor droppings) IF tracked.
- Broken intra-repo links and paths in docs pointing at files that no longer exist.
- Dead `scripts/` modules that clear all four tests in §2d.

If after honest inspection the answer is "this repository is already clean", that
is a **valid and expected terminal result**. Write it, set `STATUS: COMPLETE`, stop.
Do not manufacture work. Deleting a file in order to have something to report is
the worst possible outcome of this run.

---

## 4. PHASES

Advance one phase per invocation where possible. Never skip a phase.

### Phase 1 — `1_DISCOVERY`
- Enumerate the universe: `git ls-files`.
- Classify every tracked file as `ACTIVE` / `PROTECTED` / `CANDIDATE`. A
  `CANDIDATE` requires the reference check in §2d or §3 to have actually been RUN,
  with the command and its output recorded.
- Write `CLEANUP_AUDIT.md`: one line per candidate, with the evidence and the
  proposed action. **Nothing else in the repo changes this phase.**
- Commit. Set `CURRENT_PHASE: 2_ISOLATION`.

### Phase 2 — `2_ISOLATION`
- For each `CANDIDATE`, re-verify the evidence. Do not trust Phase 1's own notes.
- Act with `git rm` / `git mv` **only**, so every action lands in git history and
  is recoverable. Never `rm` a tracked file with the shell.
- Do NOT relocate anything into `archive/` — it is gitignored, so a move there
  silently drops the file out of the tracked set. Either `git rm` it (history
  keeps it) or leave it.
- Commit each group separately, with the reason in the message.
- Set `CURRENT_PHASE: 3_PRUNE`.

### Phase 3 — `3_PRUNE`
- Remove dead `scripts/` modules that clear all four tests in §2d. Expect this
  list to be SHORT or EMPTY.
- Duplicate helpers **inside `scripts/`** may be unified: `scripts/` is outside
  `TRAINING_PATHS` and is safe to change mid-campaign. Duplicates inside `src/`
  are a REPORT ONLY — see §2a.
- Fix the stale doc references found in Phase 1.
- Commit. Set `CURRENT_PHASE: 4_VERIFY`.

### Phase 4 — `4_VERIFY`
Run these, in order, and paste the real output into `CLEANUP_AUDIT.md`:

```
python -m compileall -q src scripts configs tests main.py
python -m scripts.audit_config
python -m pytest tests -q
python -m scripts.preflight --before-launch
```

- `pytest` takes ~250s and needs no dataset; `preflight --before-launch` needs no GPU.
- If any of them fails: assume it is YOUR change. Diagnose, patch, re-run. Do not
  revert wholesale, do not disable or skip a test, and do not label a failure
  pre-existing without proving it — `git stash`, re-run on the clean tree,
  `git stash pop`.
- A proven pre-existing failure goes in `CLEANUP_AUDIT.md` under `## PRE-EXISTING`
  with the proof, and you continue.
- Commit. Set `CURRENT_PHASE: 5_TERMINATE`.

### Phase 5 — `5_TERMINATE`
- Confirm `git status --porcelain src/ configs/ main.py tests/` is EMPTY. If it is
  not, you violated §2a: `git checkout -- src/ configs/ main.py tests/` and re-run
  Phase 4.
- Write the final summary into `CLEANUP_AUDIT.md`: what was removed, what was kept
  and why, and what is proposed for a human under `## DEFERRED`.
- Write `STATUS: COMPLETE` as the first line of `CLEANUP_STATE.md`.
- Commit. Stop.

---

## 5. `CLEANUP_STATE.md` FORMAT (keep exactly this shape)

```
STATUS: IN_PROGRESS
CURRENT_PHASE: 2_ISOLATION
LAST_UPDATED: <ISO date>
NEXT_STEP: <one concrete sentence: the next file or command, not a goal>
NOTES:
- <append-only log, one line per session>
```

`STATUS: COMPLETE` on line 1 is the ONLY thing that stops the outer loop. Write it
when the work is done, **or** when you have concluded there is no safe work left.
