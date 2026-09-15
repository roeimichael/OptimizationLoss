# RULESET -- point me at this file when I have lost the thread

**Purpose.** One file the user can paste to force a re-read and re-validation.
It is an INDEX plus the rules that exist only as spoken instructions. It does
not restate `docs/FRAMEWORK.md`; that file is the law and wins every conflict.

**Order of authority.** The user's spoken instruction > `docs/FRAMEWORK.md` >
this file > memory > my own recollection. If this file disagrees with FRAMEWORK,
FRAMEWORK is right and this file is the bug -- fix it here, do not work around
it. When the user overrides FRAMEWORK, record the override here WITH ITS DATE so
it is neither mistaken for an invention nor silently re-tightened later.

**The three files.** `RULESET.md` is how to work. `docs/MISSION.md` is where we
are now. `docs/LEDGER.md` is what is proved, measured and closed.
`docs/FRAMEWORK.md` is the protocol and outranks all three.

---

## 0. The re-entry checklist

Run this, in order, when the user says I am out of context:

1. Read `docs/FRAMEWORK.md` **in full**. Not a summary of it. The five
   validation gates are the part I skip under pressure.
2. Read `docs/MISSION.md` section "THE COURSE", then `docs/LEDGER.md` -- the
   ledger is what stops a direction being tried twice.
3. SSH `dsisco01` and `dsisco02` -- what is actually running, on BOTH hosts, by
   reading `/proc/<pid>/environ` for `EXPERIMENT_DIR`. Never assume.
4. State plainly: what is **verified**, what is **unverified**, what is **next**.

---

## 1. The question and the bar

- **One question:** can training-time TraLO beat a post-hoc clipper on a
  capped-class task, at matched data, compute, allocator and precision?
- **The bar, set by the user:** leading group on **cc-F1** AND not dominated on
  the rest of the profile. A win on accuracy while losing cc-F1 is a **TRADE**
  and is reported as a trade.
- A valid negative result is a real deliverable. Do not start a new research
  task to manufacture an advantage.
- Stay inside dual-constraint training. Do not pivot the research question.

## 2. Validation is not optional -- this is where I actually fail

FRAMEWORK defines five gates. I have skipped 2 and 3 repeatedly. Before any
campaign is read as evidence:

- **Software** -- tests, autograd vs analytic, AMP/nonfinite paths, deterministic
  repeats, crash/restart.
- **Data** -- real array paths, hashes, shapes, label alignment, class supports,
  grouping semantics, split overlap, duplicates, subject identities.
  STOP: **"Training accuracy alone cannot diagnose test-cut saturation."** My
  `gate:saturation` reads train accuracy. It is a screen, NOT a diagnosis of the
  regime. Pair it with the test-side cut (gate 3) before claiming "frozen".
- **Regime** -- `scripts/headroom.py` at the ACTUAL per-group allocation cut:
  is the cap binding (`filled == slots`), how many slots are wasted, how much
  correctable headroom sits outside, and where the marginal probability lands.
  **Screen every backbone separately.** Never weaken a model or pick a slice
  because TraLO wins there.
- **Logs** -- verify each logged quantity against the computation it names:
  phase/epoch, planned/attempted/applied/skipped, raw gradient vs applied
  displacement, soft vs hard counts, per-scope budget/residual/multiplier/rho.
  Log actual values, never inferred percentages or cached predictions.
- **Release** -- commit tested source, sync local/remote by SHA-256, run tests on
  the target host. **Never change campaign code after launch.**

## 3. Statistics -- the rules I keep violating

- **Atomic cell = (dataset, backbone, cap, method) over matched SEEDS.**
  Average over seed ONLY.
- **A cap level is not a seed. A copied warm-up is not a seed. A re-run is not a
  seed.** Training is bit-deterministic given `(config, seed)`, so an
  overlapping arm adds zero information -- de-duplicate by prediction hash
  before counting n.
- **`tralo_null` has no cap dependence** (lambda=0), so the L80 and L90 nulls are
  the SAME run. Share it across caps; do not count it twice.
- **4 seeds is a pilot, not evidence.** Effects here are ~0.01 against a seed sd
  of ~0.011, so n=4 carries roughly 15% power. Use **12 seeds** for anything
  meant to settle something; `--seeds` is per-campaign.
- Report **mean, seed SD, every seed delta, n, and a two-sided 95% Student-t CI
  in native metric units.** Not a bare p-value. A reseed spread is a diagnostic,
  not a confidence interval and not proof of equivalence.
- Do not pick the inferential method after seeing which one declares a win.
- **`scripts/deployed_h2h.py` is the maintained reporter.** Do not hand-roll
  cc-F1 with an ad-hoc sklearn call; that has produced three scorer bugs before.
- Flips, raw counts, and proximity to a cap are **not** classification quality.

## 4. Experimental design

- Every trained arm needs its **own matched zero-constraint control** sharing the
  warm-up identity AND the schedule. `tralo_null` controls the plain column
  only; `aug_tralo` needs `aug_tralo_null`.
- Report all prespecified comparisons: both clippers, all rival duals, the null.
- The 30-epoch budget is a **comparability convention, not a measured optimum**.
  The user confirmed he made the number up and that every hyperparameter is
  movable. Change it only in an **explicit matched protocol amendment**, never
  implicitly per arm without saying so.
- **Rejected by the user as cheating:** weight decay, label smoothing.
- Do not repeat a failed setting without naming what invalidated its test.
- Pre-register the reading of an experiment BEFORE the seeds land.

## 5. Server and dispatch

- **Never run experiments locally.** SSH `dsisco01` / `dsisco02`, env `optloss`.
- **GPUs: the user authorised ALL FOUR dsisco02 cards on 2026-09-15**
  ("let's take advantage of all four of them", and later "we have enough GPU
  power to simulate both ideas at the same time"). This SUPERSEDES the earlier
  same-day three-GPU note and FRAMEWORK gate 6's "total ceiling is three".
  Four campaigns ran concurrently on 2026-09-15 under it. The gate-6 procedure
  still applies: inspect the first runs before trusting any of them.
- **Never share a GPU with another user.** Check owners first.
- Both hosts share NFS -- check processes on BOTH before dispatch or recovery.
- One campaign per card: partition by `EXPERIMENT_DIR`, never by threads. Keep
  each campaign on **one host and one precision regime** -- dsisco01 is fp16 on
  older cards, dsisco02 is bf16 on Blackwell.
- Launch detached: `CUDA_VISIBLE_DEVICES=<n> EXPERIMENT_DIR=<root> setsid nohup
  python -u main.py < /dev/null > logs/<name>.log 2>&1 &`.
  **Never through a nested `bash -c`** -- it silently drops CUDA and runs on CPU
  with a healthy-looking log.
- **Stop by explicit PID**, scoped by reading `/proc/<pid>/environ`. Never
  `pkill`, never `grep main.py`.
- **Freeze the running source.** Do not touch `src/`, `configs/`, `scripts/`,
  `main.py` on the server while a campaign is live -- they are hashed into
  `source_inventory()` and editing them splits `code_version`.
- Pin each campaign at the commit its configs were generated from
  (`freeze_campaign`), and run the staged gates: stage, verify, launch,
  firstrun, score. **No bypass to keep GPUs busy.**
- **Kill a bad campaign at firstrun, not at hour 19.**

## 6. Evidence handling

- **Preserve evidence by recoverable archival.** Never delete predictions,
  checkpoints, data, or Git objects as a cleanup shortcut. Protect unrelated
  dirty changes.
- 🛑 **`docs/archive/` IS QUARANTINED. Never source a claim from it.** Everything
  under it is historical and much of it is contaminated with retired datasets
  (dermMNIST leaked, BCN blocked, iwildcam retired) -- the archived paper sources
  carry 78-115 references each. Every archived markdown carries an `ARCHIVED --
  NOT CURRENT EVIDENCE` banner in its first lines; if I am reading one, I am
  already off course. `tests/test_archive_quarantine.py` enforces this.
- **The live record is exactly eight files** and the same test enforces that too.
  A new document is a deliberate decision, not something that accretes.
- Never `git gc` / `prune` / `repack` / `reflog expire` / `worktree prune`.
  Pass `-c gc.auto=0` on every git call.
- Never edit `docs/archive/paper/main.tex` -- it is the professor's file, and
  it is preserved byte-identical through the archival move.
- Retract in place, in the same document. Do not leave a stale claim standing
  and correct it further down.
- **A gate is not done until a mutation makes it FAIL**, and the restore is
  verified by EXECUTING it -- stale bytecode has faked a pass.
- Claims about what code reads come from AST or reading it, never from grep.
- Fix defects; do not write a note describing them.

## 7. Working style

- **Never idle while a GPU is free.** Price a direction offline before spending
  the machine on it. But **no bypass of the staged checks to keep GPUs busy** --
  an unvalidated campaign is worse than an idle card.
- **Keep active instructions short.** History goes in `docs/archive/`, not into
  repeated warnings spread across source, skills and operational documents.
- **Verify behaviour with executable tests**, never with an assertion that some
  document contains a particular sentence, test count, or verdict.
- **`keep working` runs the loop.** The `keepworking` skill is the hourly
  procedure: READ the four live documents from disk, OBSERVE what is actually
  running on both hosts, ASSESS anything that finished, WORK the highest-value
  item, **WRITE BACK** what was learned into MISSION/LEDGER/RULESET, then REPORT
  verified / unverified / running / next. It holds no rules or results of its
  own -- everything it needs is in the four files, which is why they must be
  kept current. A quiet hold is a valid outcome.
- Small changes, fast iteration. Do not inflate the codebase.
- **Ask** when a decision changes the scientific question, data access, held-out
  evaluation, or compute budget. Ordinary validated cleanup is authorised.
- Standing approval: push and merge freely on `cleanup/consolidate-pipeline`.
- Prefix shell commands with `rtk`; `rtk proxy` for unfiltered commands.
- Date run-state when CHECKED, not when written.
- Report **what is verified, what is unverified, what is next.**

## 8. Cosmetic (lowest priority -- never at the cost of the above)

- No em-dashes; use ` -- `. No emoji or non-ASCII in Python prints.

*These are style rules for the eventual paper. If I am spending attention here
while a validation gate is unrun, I have my priorities inverted.*
