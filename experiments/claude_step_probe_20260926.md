# Targeted-step dose-response probe -- EXPLORATORY, specified before it ran (2026-09-26)

## Why

The wash-out analysis (`analysis/washout.py`; post hoc, development labels offline) looked at
the cap-50 targeted-step study.

- The FIRST targeted step starts from a state identical to the null's.
- It adds **+1.12 correct capped slots [+0.62, +1.63]** (n = 24); the sham at the same radius adds +0.00.
- The next CE epoch erases the gain: the difference at epoch 7 before its step is +0.00.
- Later steps add +0.3 to +0.7 each, and are erased the same way.
- At cap 76 the first step adds only +0.17 [-0.14, +0.47].

**Hypothesis.** The constraint direction carries a small amount of WHO information, and it grows
with how deep the step must cut. The endpoint never shows it because CE training overwrites it.

## Design (fixed)

- **Runner:** `tralo/step_probe.py`, seeds **2601-2624** (fresh), cap 50, on dsisco01.
- **State S:** train the tralo_null schedule for 5 CE warm-up epochs, then 1 post-reset epoch. S is
  the epoch-6 pre-step state of every arm.
- **Steps:** from copies of S, apply `targeted_step` to hard grade-3 targets t = round(f x 50), for
  f in {1.0, 0.8, 0.6, 0.4, 0.2}. Each step goes along TraLO's direction and along a seeded sham
  at the same radius.
- **No training follows.** Each stepped model is scored by capped_first at K = 50, offline.
- **Scorer:** `analysis/score_step_probe.py`.

Exploratory. The readouts are fixed now:
- (tralo - S) and (tralo - sham), in correct slots, per f, with 95% CIs over 24 seeds;
- the eviction precision of each TraLO step.

**Readings:**
- (a) tralo - sham grows with push depth, with CIs above 0 at some f: the direction carries WHO
  information proportional to depth, and wash-out, not the direction, is what kills the endpoint.
- (b) tralo - sham stays about 0 at every f: the epoch-6 +1.12 was chance, or specific to that state.
- (c) tralo - S turns negative at deep f: deep pushes damage the ranking.

Development labels are read only by the offline scorer.
