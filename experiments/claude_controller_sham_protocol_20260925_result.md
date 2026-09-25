# Live controller and sham control -- RESULT: STOPPED AT THE PILOT GATE (2026-09-25)

Protocol: `claude_controller_sham_protocol_20260925.md`. Its gate says "any failure stops the
study and is recorded, not patched silently". Seed 1701 failed one gate item, so seeds
1702-1724 were never launched. Everything below is n = 1 and descriptive only.

## Pilot attempts

| attempt | release | outcome |
|---|---|---|
| 1 | ff2e8c7c | died in `sham_sgd`: seeded CPU noise times a CUDA norm tensor. Fixed in 8337a39e with a CUDA test that fails on the old code. |
| 2 | 8337a39e | ran 10x slow (4 min/epoch): another user's ~60 niced CPU jobs plus our uncapped ~44 torch threads. Stopped by PID. The queue now sets OMP_NUM_THREADS=8; the clipper output is byte-identical at 8 threads, uncapped, and under both releases (sha256 7ac40fa3...). |
| 3 | 8337a39e | completed. Gate below. Output: `runs/claude-sham-20260925/seed1701` on dsisco01. |

Arms 1-4 of attempt 1 are byte-identical to attempt 3 (final_probabilities sha256), consistent
with bit-deterministic training.

## Gate, attempt 3

| item | result |
|---|---|
| all five arms complete | pass |
| warm-up and batch hashes identical | pass (1 identity) |
| 1810 task updates in every arm | pass |
| constraint steps finite | pass |
| sham first displacement = sgd first displacement (1e-4) | pass (0.0999807 both) |
| sgd first displacement = adam first displacement (1e-3) | pass |
| **cap binds on the hard count at the first constraint epoch** | **FAIL**: 74 grade-3 calls vs cap 76 (soft count 82.6) |

The protocol's claim that the cap binds "in every seed of the 24 Sept run (null raw calls 95.75)"
used the FINAL raw count, not the first-epoch count the runner records. Here the null's hard
counts at epochs 6-10 were 74, 120, 77, 83, 94: the cap binds in 4 of 5 epochs and at the endpoint.

## What the pilot showed (n = 1, descriptive)

Grade-3 counts on the 826 development images (no labels involved):

| arm | epoch-6 soft count before -> after ONE constraint step |
|---|---|
| tralo_adam (published) | 82.6 -> **8.2** |
| tralo_sgd | 82.6 -> **0.2** |
| sham_sgd (same norm, random direction) | 82.6 -> 87.3 |

1. **The published dose overshoots about 10x.** One step of displacement 0.10 removes ~74 grade-3
   calls when the cap asks for ~7. Every later constrained epoch repeats this: the task epoch
   restores the class, the constraint step wipes it out again.
2. **The direction carries the information; the size is wrong.** At an identical norm, a random
   direction barely moves the count.
3. **The controller never acted in v1 either.** The first hard count (74) met the cap, so
   `advance_controller` froze at the first check in every arm; lambda stayed 0.01 all run. With
   one capped class, lambda and rho only rescale the gradient, so no controller setting could
   choose the right size.
4. Endpoint, capped_first grade-3 F1: clipper 0.637, null 0.670, adam 0.539, sgd 0.275, sham 0.615.
   The ~40-point sgd loss is an overshoot artefact of the dose, not a verdict on the direction.

## Consequence

The question "does the constraint's information help when it is applied at the right size" was
not askable with this design. A successor study that sizes the step by bisection to land on the
hard cap is preregistered as `claude_targeted_step_protocol_20260925.md`, on fresh seeds.
