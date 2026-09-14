> ARCHIVED — historical skill, not current instructions. Superseded by the installed keepworking skill and docs/FRAMEWORK.md after the evidence reset.

---
name: keepworking
description: Use when resuming work on OptimizationLoss, when a long job is in flight, when the next step is unclear, or before launching any experiment - carries the target, the bar a result must clear, the rules of engagement, and the minefield of failures this project has already paid for
---

# Keep Working

Invoked because work stopped, is about to, or is about to go somewhere expensive.

Read this in order. Do not skip to the task list.

---

## 1. THE TARGET

> **Train a network under transductive prediction-count constraints (TraLO) and
> show it beats (a) post-hoc clipping at equal compute, and (b) the rival
> constrained-optimisation methods `fioretto`, `hounie` and `alm`.**
>
> **(b) is the pre-registered goal. It is the one that makes this a paper.**

(a) alone is not the contribution. A post-hoc clipper is given the same budget
K and allocates optimally from the model's own probabilities, and that
optimality is distribution-free. **So the ONLY way TraLO can win is by
producing a better RANKING.** Every other channel is closed by arithmetic.
Write that sentence out before proposing any new arm; most proposals die on it.

**Status as of 2026-09-04:** (a) holds at 4 of 4 units, which floors at
p=0.0625. (b) is **unanswerable, not lost** -- every `fioretto`/`hounie`
comparison ever run was at unequal dose, and against the one validly-dosed
rival (`alm`) TraLO does not lead.

---

## 2. THE BAR -- what a result must clear to be real

Nothing below is negotiable, and each line is here because it was violated once.

* **Atomic cell = (dataset, backbone, cap, method) over >= 4 seeds.** Seed is
  the ONLY axis you may average over. Never pool across caps, backbones or
  datasets. Count cells.
* **The independent unit is (backbone, HOST), not the campaign.** Campaigns
  sharing a warm-up model are ONE unit -- verify by md5, never assume. A
  campaign pair is not a free replicate.
* **Sign tests run over UNITS and floor at `0.5^n`.** 3 units cannot beat
  p=0.125; 4 cannot beat 0.0625. **More seeds can never make a sign test
  significant. Only more units can.** If the plan is "add seeds to reach
  significance", the plan is arithmetically impossible -- say so.
* **An effect below the RNG reseed floor is not an effect.** Carry
  `tralo_reseed` always. The floor currently rests on too few observations;
  check `MIN_FLOOR_OBS` before quoting it.
* **Both clippers (`clip` and `focal_clip`) inside the same campaign.** An
  arm-vs-arm delta is not a result until the bar is in the same campaign.
* **Equal dose, equal compute, one recipe, one host.**
* **`flips`, raw count over K, and "proximity to feasibility" are NOT metrics.**
  Post-hoc filling is free. When quality ties, the honest report is "this arm
  produced nothing."

---

## 3. RULES OF ENGAGEMENT

Set by the user 2026-09-04, after a session that spiralled.

1. **One thing at a time. Report between.** No parallel fronts, no fleets of
   subagents running while you edit something else. Finish, report, wait.
2. **Lead with the science, not the plumbing.** Default to answering *is TraLO
   winning*. Fix tooling only when it blocks that answer, and say explicitly
   that you are doing so and why.
3. **~~Ask before any new GPU campaign.~~ SUPERSEDED 2026-09-05 by a standing
   mandate: run, analyse, upgrade, re-run and fine-tune until TraLO beats the
   rival duals, in a long autonomous session.** The rule is recorded rather
   than deleted because it was right for the situation that produced it and
   may be reinstated.
   **What the mandate does NOT suspend:** rules 4-6 below and the whole
   minefield. A campaign still has to be shown capable of answering something
   BEFORE it is launched. Autonomy is permission to proceed without asking,
   not permission to run experiments that cannot separate methods -- that is
   the failure the mandate exists to escape, not a shortcut it licenses.
   **Still ask** when a choice would change the goal, discard completed runs,
   or spend days on an unvalidated direction.
4. **Validate the training logs after an experiment lands**, before any metric
   is read. Dose, collapse, non-finite, and the actual accuracy curve.
5. **Apples to apples across methods.** Same environment, same host, same AMP,
   same warm-up, same optimizer budget. If two arms differ in anything except
   the thing under test, the comparison is void.
6. **Verify the regime BEFORE committing to a grid.** See rule 4 of the
   minefield -- this is the one that has cost the most.

### The honest-null clause

**The goal is to beat the rival duals. The goal is NOT to report that we did.**
If the measurements say TraLO ties or loses, that is the result, and it gets
reported in those words. A manufactured win is worth less than nothing here:
every number in this project is checked against a noise floor, a dose ledger
and a unit count precisely so that a real one can be trusted. Push hard on the
science; never push on the reporting.

### An idle GPU is cheaper than a wrong number

**This replaces the old "never idle, never wait" rule, which is what produced
the spiral.** Running something because the GPU is free is how this project
accumulated thousands of runs that measured nothing: 265 runs that were
mechanically perfect and posed no question, 24 of 24 cells outside the task
window, 792 runs with a silent dose gap.

**STOP is a valid state.** Stop when:
* the experiment cannot answer the question even if it succeeds;
* the numbers on hand are known-wrong and would be quoted;
* the next step needs a decision only the user can make.

When genuinely blocked on one thing, do the parts that do not depend on it --
but do not invent work, and do not launch compute to look busy.

---

## 4. THE MINEFIELD

Every entry is a failure this project actually paid for. Check each before a
launch, not after.

**1. CE SATURATION -- the biggest one, and the one most recently missed.**
The dataset must not be solved during warm-up. Measured on iwildcam/ViTB16:
**95.8% accuracy after ONE warm-up epoch**, 98.6% by epoch 5, and `tralo`
tracks `tralo_null` to four decimals throughout. Precision at the cut is
0.9948-0.9972, so the entire prize is 0-1 items on tight caps. **A saturated
model cannot separate methods, because the ranking is already right where the
cap cuts.**
⚠️ **But de-saturating by weakening the model is NOT the fix**: post-hoc
allocation is optimal given the probabilities and that is distribution-free,
so a worse model raises the prize for the CLIPPER too. **A bigger prize is not
a bigger gap.** What is needed is a dataset whose *ranking is genuinely
uncertain at the cut*, not a handicapped model.
🛑 **Verify a candidate lands in the non-saturating region BEFORE any grid.**

**2. THE CAP MUST POSE A QUESTION.** A cap is a task only if it evicts >= 10
items, leaves errors inside K, and cuts at p@K < 0.99. 24 of 24 cells at
L20/L30/L50 failed this. Ask `configs.task_cells.classify` -- never quote a
K/n range from prose, it has been stated three incompatible ways.

**3. AND THE TASK WINDOW IS A TRAP OF ITS OWN.** Where the constraint binds,
nothing is measurable; where something is measurable, the constraint hardly
binds. Half the prize costs 4x the seeds. Say which side you are on.

**4. DOSE GAPS ARE INVISIBLE IN PERCENTAGES.** Read `attempted/run`, never the
percentage. `672/672` and `696/696` are both exactly 100.0%, and 672 = 24x28
against 696 = 24x29. **A percentage computed WITHIN an arm cannot see a gap
BETWEEN arms.** That one line was quoted as proof of parity and was the defect.

**5. INERT FLAGS -- five occurrences and counting.** md5 across arms is
**ONE-SIDED**: identical predictions prove inertness, different predictions
prove NOTHING (`logit_adjust` differs in 24/24 and is mathematically plain CE).
To clear a loss variant, compare its GRADIENT against CE on the real prior.

**6. CHECK BOTH HOSTS.** dsisco01 and dsisco02 share one NFS `/home`, so
`results/` is byte-identical from either and only `ps` differs. Concluding a
dispatcher died from one host's process list put four dispatchers on one tree
and split a campaign across fp16 and bf16 -- and host changes the model.

**7. ANNOUNCING IS NOT ENFORCING.** A quarantine marker, a warning banner, a
doc note: none of them prevent anything. Six of seven scorers printed
"DEAD ARMS: fioretto, hounie" and then ranked fioretto #1. If a rule matters,
put it in code that refuses, and mutation-test that it refuses.

**8. A GATE THAT HAS NEVER FAILED HAS NEVER BEEN SHOWN TO WORK.** Every check
needs a negative control that makes it go red. A control that only ever runs
through a monkeypatch is not a control.

**9. FREEZE `src/`, `configs/`, `main.py` on the server during a campaign.**
`code_version` is a git hash; even a comment splits the campaign. `scripts/` is
exempt. Never run `git gc/prune/repack/worktree prune` anywhere in the
worktree family -- they share one object store.

**10. DO NOT RE-RUN THE REJECTED.** `docs/FRAMEWORK.md` section 2 is the ledger
of everything already measured and already worse: penalty-shape variants, more
constraint steps, a dedicated constraint optimizer, the joint objective, the
undershoot hinge, finer granularity, KL. Read it before proposing anything.

---

## 5. PROCEDURE

**On resume:** read `docs/MISSION.md` in full, then run its RESUME PROTOCOL to
find what is actually running. Never re-derive state from memory or a summary;
the files win. If `MISSION.md` is stale, updating it is the first task.

**Before spending a GPU**, price the direction offline. These run on CPU in
minutes against artefacts that already exist, and each closed a direction that
would otherwise have cost a campaign: `dataset_screen`, `task_window`,
`ceiling_screen`, `paired_noise`, `sensitivity_screen`, `frozen_head_probe`,
`straddle_probe`, `scope_probe`, `step_direction_probe`, `ortho_survival`.
**A cheap probe that closes a direction is worth more than the run it saves.**

**Before launching:** `pytest tests -q`, `scripts.preflight --before-launch`,
`scripts.run_campaign --step stage|verify|launch`, `scripts.rig_status`.

**On the first completed run** -- not at the end -- `scripts.dose_landed` and
`scripts.sensitivity_screen`. Both are readable while the campaign is 1% done,
and both have killed campaigns that looked healthy from every other angle.

**Before quoting any number:** `scripts.quarantine --list`, then
`scripts.preflight --stage results`.

**Iterate the goal, not the task list.** A finished task is where you ask what
the result changed:
* **worked** -> record it, ask what the next increment is;
* **failed** -> write it into the rejected ledger with the evidence, so it is
  never proposed again;
* **inconclusive** -> say WHICH: *not enough measurement* or *no effect*. They
  are opposite conclusions requiring opposite actions, and conflating them is
  this project's most expensive habit.

**Update `docs/MISSION.md` before the session ends.** It is the only thing that
survives you.

## 6. REPORTING

Every report states, in this order: **what the result was** (science first),
**what it changed in the state file**, **what is running now** (or plainly:
nothing, and why that is right), and **what needs a decision from the user**.

Never end a turn by asking what to do next when you could have found out
yourself -- but equally, never launch compute to avoid asking a question that
is genuinely the user's to answer.
