# CUTPAIR amendment 1: reading P1 when the shift arm's dose frequency differs -- 2026-09-26

Written after the pilot (seed 2700) was launched and before its results were read. No study seed
(2701-2724) has run. It amends `claude_cutpair_protocol_20260926.md`. It comes from the independent
code review of v5, finding A.

## The confound

The dose rule gives every batch with at least one active item exactly 10% of the CE gradient norm,
whatever the hinge magnitude. An arm's total hinge dose is therefore set by how many batches are dosed.

The shift arm's anchors change that count, not just the location of the anchor:
- rank 25 has a high tau, so few negatives sit above it;
- rank 228 has a low tau, so many do.

P1 (`cutpair_aug` - `cutpair_aug_shift`) therefore mixes anchor location with dose frequency and
active-set composition.

## Fixed reading rules

1. **Report per rank.** For `cutpair_aug` and, per drawn rank, for `cutpair_aug_shift`: mean N_act,
   mean P_act, the active-batch fraction, the dosed-batch fraction, and per-seed side counts.
2. **Dose-matched reading.** Let D_cut and D_shift be the mean dosed-batch fractions over epochs 6-10.
   - If they differ by at most a factor of 2 (0.5 <= D_shift / D_cut <= 2): P1 is read as preregistered.
   - Otherwise: P1 is reported, but a positive P1 is NOT claimed as attributable to the anchor's
     location. The location claim then rests on P2 (`cutpair_aug` - `aug_clip`), together with the
     per-rank breakdown. It is worded "a cut-anchored hinge helps; location attribution confounded by
     dose frequency".
3. **Unchanged:** the pilot gate, which applies to `cutpair_aug`, the endpoint, the Holm family and the
   readings. The pilot seed 2700 is excluded from the study scorer.
4. **Code changes.** Log-only changes land before the study (`dosed_batches`, `anchor_p3`, and two
   regression tests) and must leave every trained output bit-identical. So the pilot's release
   (c0ab1344) and the study release train identically. The existing bit-identity tests verify this.
