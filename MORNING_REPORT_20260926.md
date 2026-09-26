# Morning report -- 2026-09-26

Branch `claude/rebuild-validation-20260925` (pushed). The live record is updated on
`cleanup/consolidate-pipeline`: LEDGER settled #9 and MISSION THE COURSE.

## The answer to your question

You asked whether TraLO really works, whether the constraint pushes us forward or it is
luck, and whether the code is wrong or we tested the wrong things.

- **The code's arithmetic is correct.** The audit verified it by execution.
- **The design was wrong in seven places (D1-D7),** and together they made the old experiments
  unable to answer the question.
- **I fixed them, ran the experiment that can answer it, and replicated it at a second
  operating point.** Answer: **it is not luck, and it does not help.**
  - At the correct step size, TraLO's constraint direction knows exactly HOW MANY patients to
    remove from the capped class.
  - It removes the same patients the post-hoc clipper would remove (83-87% overlap).
  - It chooses no better than a random move of the same size.

## What is verified

**1. Code audit** (38-agent workflow, `analysis/CODE_AUDIT_20260925.md`).

The arithmetic is correct. The design findings:

| # | finding |
|---|---|
| D1 | The separate Adam optimizer is scale-invariant, so lambda, rho and violation depth all cancel. |
| D2 | The trigger uses the soft count, not the hard count. |
| D3 | The knee caps never bound. |
| D4 | The frozen head could only re-rank by feature norm. |
| D5 | Momentum confound. |
| D6 | Oscillating endpoint. |
| D7 | The Chen data path encoded test labels. |

D1, D3, D4 and D7 are fixed with mutation-proven tests; D2 is fixed by the targeted step.

**2. The published dose overshoots ~10x.** Found by the v1 pilot, then replicated over 24 seeds.
- One constraint step moved the grade-3 soft count from 82.6 to 8.2, against a cap of 76.
- The controller froze at its first check in every arm.
- v1 was stopped at its preregistered gate, as its own rule required
  (`experiments/claude_controller_sham_protocol_20260925_result.md`).

**3. v3 study** (preregistered; n = 24; knee; trainable ResNet18; grade-3 cap 76).

The design: TraLO's direction, stepped to the smallest radius that meets the HARD cap, compared
against a sham that moves the same radius in a random direction.

| contrast | capped grade-3 F1 [95% CI] |
|---|---|
| target - null | +0.96 [-0.96, +2.88] |
| **target - sham** | **+0.14 [-1.21, +1.48]** |
| target - published adam | **+2.98 [+1.24, +4.71]**, p = 0.002 |

Preregistered reading 2. It is a *bounded* null: any gain from the constraint's information is
below +1.5 F1 points (~1.3 of 76 slots).

**4. v3b replication at a deep cut (cap 50, n = 24):** target - null +0.96 [-0.87, +2.79],
**target - sham -0.75 [-2.45, +0.96]**. Reading 2 again.

**5. Mechanism** (post hoc, development labels offline). The step evicts 83% (cap 76) and 87%
(cap 50) the same items as the post-hoc cut on the same probabilities:

| cap | step precision | post-hoc cut precision |
|---|---|---|
| 76 | 57.3% | 58.6% |
| 50 | 49.3% | 48.4% |

The soft-count gradient ranks patients by the same probabilities the allocator cuts on, so it
cannot beat the allocator. This is the end-to-end counterpart of settled result #3 and of M1.

**6. augfin (fmow2)** was scored earlier this session: outcome 2, AMBIGUOUS
(decisive +0.0011 / +0.0033, t < 1).

## What is unverified, and why

- **One dataset cell for the new studies:** knee grade 3, ResNet18, 5 + 5 epochs. The fmow2 corpus
  agrees, but under the older design.
- **Local/group caps,** your actual deployment setting, were never run with the fixed design
  (targeted step plus sham). This is **not measured**, which is different from "no effect".
- **D5 and D6** (momentum confound, oscillating endpoint) remain open. Neither can manufacture
  or hide an effect of the size the intervals allow.

## Incidents overnight (all fixed, all recorded)

1. **Sham crashed on CUDA:** CPU noise met a CUDA norm tensor. Fixed with a CUDA test that
   fails on the old code.
2. **Pilot ran 10x slow:** another user's ~60 CPU jobs plus our uncapped threads.
   OMP_NUM_THREADS=8 now; outputs proven byte-identical across thread counts.
3. **Queue script got CRLF line endings.** Fixed; `.gitattributes` now pins `tools/*.sh` to LF.

## What ran

| study | seeds | GPU time | outcome |
|---|---|---|---|
| v1 pilot | 1701 (3 attempts) | ~1.5 h | stopped at gate |
| v3, cap 76 | 1801-1824 | ~9.5 h over 4 GPUs | reading 2 |
| v3b, cap 50 | 1901-1924 | ~9.5 h over 4 GPUs | reading 2 |

GPUs have been idle since 04:35, deliberately: both preregistered questions are answered, and
the next step changes the scientific question, which is your call.

## What is next (your decision)

1. **Write it up.** The thesis now has a clean, preregistered, replicated negative result with a
   mechanism. "Training-time count constraints reproduce the post-hoc allocator through the
   weights; at the right dose they are exactly as good, and at the published dose they are
   worse." That is submittable.
2. **If you want one more test before writing,** run the targeted-step and sham design under
   **local/group caps** on fmow2. The code is ~80% there; it needs a multi-class targeted step.
   Cost: roughly 4 GPU-hours.
3. **The only direction the theory still permits** is one that improves the SCORE itself (Woodworth
   et al.), for example the augmentation-plus-clipper line. That is not a constraint method.
