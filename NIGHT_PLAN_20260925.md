# Night plan, 2026-09-25 -> morning 2026-09-26 (Claude, branch claude/rebuild-validation-20260925)

The user is asleep and will not answer. Work until morning. This file is the state
machine the heartbeat reads; update the STATUS lines as stages finish.

Worktree: C:\Users\roeym\Desktop\projects\OL-rebuild  (forked from codex/tralo-rebuild-20260922 @ eac8db3c)
Local run archives: C:\Users\roeym\.codex\rebuild-audit-20260922 (1.1G); extracted copies under the session
scratchpad/runs (sweep 384 fits, ALM 48, anchor/far-error 40 -- all with per-sample probabilities).

## Stages, in order

1. CODE AUDIT -- workflow wf_1c6808f3-0cf (9 reviewers, 2 skeptics per medium+ finding, tiebreak).
   STATUS: DONE -- analysis/CODE_AUDIT_20260925.md. Arithmetic correct by execution; design
   findings D1-D7 (D1 separate Adam scale-invariant, D2 soft/hard gap, D3 knee caps never bind,
   D4 frozen head, D5 momentum confound, D6 oscillating endpoint, D7 test-label path).
2. FIX every confirmed finding on this branch, each with a test that FAILS before the fix
   (mutation-proven), full suite green, commit + push. Low-severity findings: fix if cheap.
   STATUS: DONE for D1 (tralo/constraint_optimizers.py CalibratedSGD), sham control, D3 (cap 76
   binds on hard count), D4 (end-to-end ResNet18), D7 (knee_data drops path/label). 127 tests
   green. D2/D5/D6 open, not blocking the study. Release ff2e8c7c.
3. LOG ANALYSIS on stored per-sample probabilities: does the constraint change WHICH items
   occupy the capped slots beyond what a reseed changes? Slot-turnover vs reseed floor, TraLO
   vs Null item-level agreement, where along the cut the moved items sit, correct-in/correct-out.
   STATUS: A1-A3 DONE (analysis/FINDINGS_20260925.md). Below the reseed floor the constraint is
   indistinguishable from noise; above it, it evicts correct items. fmow2 reseed floor is 50% of slots.
   Missing control identified: a SHAM constraint of equal step norm.
4. DECIDE the night's GPU experiments from 2+3. Pre-register each in experiments/ BEFORE launch.
   Priority candidates (to be confirmed by 2+3, not assumed):
   a. local/group constraints -- the user's actual problem; the rebuild never ran them.
   b. whatever 3 shows is the regime where the constraint moves items at all.
   STATUS: DONE -- experiments/claude_controller_sham_protocol_20260925.md (C1 sgd-null, C2 sgd-sham,
   Holm; n=24 seeds 1701-1724; MDE ~4.5 F1 pts). Scorer analysis/score_sham.py.
5. RUN on the servers (ssh dsisco01/dsisco02). Check GPU owners first; never share a card.
   Kill bad runs at the first integrity check. Score against the pre-registration.
   STATUS (checked 2026-09-26 01:46):
   v1 (controller/sham): STOPPED at its pilot gate. Pilot showed the published dose overshoots
   ~10x (experiments/claude_controller_sham_protocol_20260925_result.md).
   v3 (targeted step, cap 76, seeds 1801-1824): DONE, READING 2 -- bounded null.
   C1 target-null +0.96 [-0.96,+2.88]; C2 target-sham +0.14 [-1.21,+1.48]; target-adam
   +2.98 [+1.24,+4.71] (the published damage is the dose).
   (experiments/claude_targeted_step_protocol_20260925_result.md)
   v3b (cap 50, seeds 1901-1924): PREREGISTERED (experiments/claude_targeted_step_cap50_protocol_20260926.md),
   release cd6a205a. Pilot 1901 PASSED 02:08 (23.6 min). Seeds LAUNCHED 02:09 via claude_target50_queue.sh:
   1902-1906 gpu0, 1907-1912 gpu1, 1913-1918 gpu2, 1919-1924 gpu3. ETA ~04:35.
6. ALSO: score the unread augfin_a/augfin_b campaign on the MAIN branch
   (~/optloss-rank/augfin_*, 168 runs, pre-registered in docs/MISSION.md) once ssh is back.
   STATUS: DONE -- outcome 2 AMBIGUOUS (decisive +0.0011/+0.0033, t<1). Recorded in MISSION, pushed.
7. MORNING REPORT for the user: verified / unverified / what ran / what is next.

## Rules that bite tonight
- Never touch codex's worktree or its releases; never update a release in place.
- Never score the knee TEST split.
- A failed ssh means monitoring is unavailable, not that a job stopped.
- Do not extend a grid until a favourable p-value appears.
