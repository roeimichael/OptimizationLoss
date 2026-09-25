# Targeted step at a deeper cut (grade-3 cap 50) -- PREREGISTERED 2026-09-26

**Written before any seed of this study ran.** Author: Claude, branch
`claude/rebuild-validation-20260925`.

Disclosure: the author has seen the full cap-76 result
(`claude_targeted_step_protocol_20260925_result.md`: reading 2; C1 +0.96 [-0.96, +2.88],
C2 +0.14 [-1.21, +1.48]). Seeds 1901-1924 are fresh.

## Why

At cap 76 a targeted step evicts ~10 items (null hard calls are typically 80-120). Few evictions
leave little room for the constraint direction's information about WHICH items to matter.
Cap 50 forces ~30-70 evictions per step, so the same question is asked where "who" has the most
leverage.

The cap is a policy choice. It was set from the cap-76 run's label-free prediction counts
(the median null call is ~90; 50 forces a deep cut on every seed). No development label informed it.

## Design (fixed)

Identical to `claude_targeted_step_protocol_20260925.md` in every respect except:
- grade-3 cap **50**;
- seeds **1901-1924** (the runner pins cap 50 to this block and cap 76 to 1801-1824);
- release: the commit that adds this file.

Same five arms (clipper, tralo_null, tralo_adam, tralo_target, sham_target), same schedule, same
host and thread setting.

## Endpoint and contrasts (fixed)

- **Primary endpoint:** development grade-3 F1 under `capped_first`, exactly 50 slots.
  F1 = 2TP/156; also reported in correct slots.
- **Primary family** (Holm, paired, two-sided alpha 0.05):
  - **C1** `tralo_target` - `tralo_null`;
  - **C2** `tralo_target` - `sham_target`.
- **Secondaries:** as in the cap-76 protocol.
- **MDE:** the cap-76 run's paired SD for C2 was 3.2 points, giving ~1.9 points (~1.5 correct
  slots) at n = 24 and 80% power. For C1 it was 4.5 points, giving ~2.6 points.

## Readings, fixed before the numbers

1. **C1 > 0 and C2 > 0, both Holm-significant:** the constraint's information matters once
   the cut is deep. The cap-76 null is then an operating-point limit, not a property of the method.
2. **Both non-significant:** the negative result holds at a deep cut as well.
3. **C2 < 0 significant:** the direction evicts the wrong items.
4. **C1 < 0 with C2 non-significant:** the harm comes from the perturbation.

## Integrity gate for the pilot (seed 1901)

Same items as the cap-76 gate, with 50 in place of 76. Any failure stops the study and is recorded.
