> 🛑 **ARCHIVED -- HISTORY, NOT INSTRUCTIONS.** (banner added 2026-09-11)
> Scratch state from that one cleanup pass.
> `docs/FRAMEWORK.md` is the ONLY operational document. Where this file
> disagrees with it, FRAMEWORK wins and this file is wrong. Do not run
> anything, and do not quote any figure, on the strength of this page.

STATUS: COMPLETE
CURRENT_PHASE: 5_TERMINATE
LAST_UPDATED: 2026-09-06
NEXT_STEP: None. The tracked set was audited end to end and holds no safe removal; see CLEANUP_AUDIT.md, and its DEFERRED section for the seven items that need a human.
NOTES:
- 2026-09-06 session 1: Phases 1-5 in one pass. Universe = 266 tracked files.
- Five parallel read-only recon passes (docs redundancy / scripts census /
  cross-reference integrity / duplicate detection / paper asset reachability),
  every load-bearing claim re-verified by hand before acting on it.
- REMOVED: nothing. scripts/ dead list is EMPTY 62/62 -- test 1 of the four
  conjunctive tests in section 2d fails to clear for every single module.
- CHANGED: one stale path, 4 lines in docs/FRAMEWORK.md. `docs/launch_uniform.sh`
  was renamed (git R100) to `docs/archive/launchers/launch_uniform.sh` and the
  doc still pointed readers at the old path.
- ADDED: CLEANUP_AUDIT.md. Tracked set 266 -> 267.
- Gates green, and the baseline was taken BEFORE the edit so a pre-existing
  failure could not be confused with a caused one: 583 passed / 1 skipped both
  times. compileall silent, audit_config clean, preflight 39 passed.
- Independent preservation audit (a fresh agent that did no editing) re-derived
  all of the above from git: zero deletions, zero renames, training path
  untouched, all 17 protected paths tracked, every protected directory at its
  original file count, and the FRAMEWORK.md diff exactly 4 pure path
  substitutions with no measurement or claim altered. Verdict CLEAN.
- Prior art, not re-litigated: e7d9e893 (2026-09-02) already swept this repo
  338 -> 244 files and recorded that the tracked Python is not where the
  redundancy is. This run reproduces that independently. "Already clean" is the
  correct terminal answer here, not a failure to find work.
