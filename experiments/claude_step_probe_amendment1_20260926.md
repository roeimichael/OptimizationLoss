# Step probe amendment 1 -- 2026-09-26

Written while seeds 2601-2624 were still running and before any probe output was scored. It amends
`claude_step_probe_20260926.md` and follows an independent code review of the scorer.

## Why

The original readings had three gaps:
- reading (a) ("CIs above 0 at some f") had no multiplicity rule over the five fractions;
- "grows with depth" had no computed quantity;
- the CIs used a normal quantile.

## Fixed now

1. **Primary 1:** tralo - sham in correct capped slots, per fraction f in {1.0, 0.8, 0.6, 0.4, 0.2},
   one-sample t over seeds, two-sided, Holm over the 5 fractions.
2. **Primary 2:** the per-seed OLS slope of (tralo - sham) on depth = 1 - f, with a t CI over seeds.
3. **Readings**, replacing (a)-(c):
   - **(a')** Primary 2 > 0 (CI above 0): the direction's WHO information grows with push depth.
   - **(a'')** Primary 1 is Holm-significant > 0 at one or more f, with Primary 2 not > 0: information
     is present but does not grow with depth.
   - **(b')** Neither: no measurable information at this state (the epoch-6 +1.12 was chance, or
     specific to that state).
   - **(c)** Descriptive: tralo - S at deep f is reported with its CI; a negative value means deep
     pushes damage the ranking.
4. **Integrity:**
   - A seed is excluded, and listed, if any tralo step is not applied or lands off its target hard
     count.
   - A missing seed is listed. A sham whose radius differs from its tralo step is a hard error.
5. **CIs:** use the t quantile.

## Operational note

`tools/claude_probe_queue.sh` logs `exit $?` after a `$(date)` substitution, so it always logs exit 0.
A crashed seed therefore shows only as a missing probe.json. The scorer's seed-set check covers this.
The deployed copy is not edited while it runs.
