# Night plan, 2026-09-25 -> morning 2026-09-26 (Claude, branch claude/rebuild-validation-20260925)

The user is asleep and will not answer. Work until morning. This file is the state
machine the heartbeat reads; update the STATUS lines as stages finish.

Worktree: C:\Users\roeym\Desktop\projects\OL-rebuild  (forked from codex/tralo-rebuild-20260922 @ eac8db3c)
Local run archives: C:\Users\roeym\.codex\rebuild-audit-20260922 (1.1G); extracted copies under the session
scratchpad/runs (sweep 384 fits, ALM 48, anchor/far-error 40 -- all with per-sample probabilities).

## Stages, in order

1. CODE AUDIT -- workflow wf_1c6808f3-0cf (9 reviewers, 2 skeptics per medium+ finding, tiebreak).
   STATUS: running -- 31/38 agents returned at heartbeat 1; confirmed so far include the
   separate-Adam controller being scale-invariant (lambda/rho inert) and the knee soft/hard gap.
2. FIX every confirmed finding on this branch, each with a test that FAILS before the fix
   (mutation-proven), full suite green, commit + push. Low-severity findings: fix if cheap.
   STATUS: pending
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
   STATUS: pending
5. RUN on the servers (ssh dsisco01/dsisco02). Check GPU owners first; never share a card.
   Kill bad runs at the first integrity check. Score against the pre-registration.
   STATUS: dsisco01 UP (4 GPUs free), dsisco02 banner-hang. Shared NFS, so 02's files are readable via 01.
6. ALSO: score the unread augfin_a/augfin_b campaign on the MAIN branch
   (~/optloss-rank/augfin_*, 168 runs, pre-registered in docs/MISSION.md) once ssh is back.
   STATUS: DONE -- outcome 2 AMBIGUOUS (decisive +0.0011/+0.0033, t<1). Recorded in MISSION, pushed.
7. MORNING REPORT for the user: verified / unverified / what ran / what is next.

## Rules that bite tonight
- Never touch codex's worktree or its releases; never update a release in place.
- Never score the knee TEST split.
- A failed ssh means monitoring is unavailable, not that a job stopped.
- Do not extend a grid until a favourable p-value appears.
