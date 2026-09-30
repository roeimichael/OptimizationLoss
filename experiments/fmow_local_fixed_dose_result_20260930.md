# fmow2 fixed-dose country-aware step: completed negative result

The prespecified 12-seed block (6200–6211) completed on dsisco02 at
2026-09-30 01:12:10 UTC. Each seed used an independently trained PTO fit and
the frozen MobileNetV3-Large FP32 recipe. All 12 jobs exited 0, with one start
and one end per seed and no missing run artifacts. Independent offline scoring
from scorer release `88bf362e37df1257a25d39b3d80d35f47a172e09` passed
source/config/data/split/artifact, training-trajectory, fixed-dose, quota,
allocator, and complete-denominator checks. The original training release was
`aae5f599ef9b4e8c124de536da92259e979a8640`; no run was repeated or
modified. The 12-seed report, including every seed, arm, cap, secondary metric,
raw-count diagnostic, and artifact hash, is in
[`fmow_local_fixed_dose_result_20260930.json`](fmow_local_fixed_dose_result_20260930.json)
(SHA-256 `1b2e935f720f6d2ec19019bf082ef74deb59392ae793cdb5d57570e239b4bf21`).

The primary metric is deployed class-1 cc-F1 on the repeatedly viewed 1,673
development images. The fixed final country-plus-pooled allocator is shared by
every arm. Means below are across 12 seeds; differences are paired, with
two-sided 95% Student-t intervals and Holm-adjusted p values for the four
prespecified primary contrasts.

| Cap | PTO cc-F1 | Joint local step cc-F1 | Pooled-only dose cc-F1 | Joint − PTO | Joint − pooled dose |
| --- | ---: | ---: | ---: | ---: | ---: |
| G=167 (`/10`) | 0.4961 | 0.4095 | 0.4204 | −0.0866 [−0.0947, −0.0785], Holm p=3.52e−10 | −0.0109 [−0.0194, −0.0023], Holm p=0.0344 |
| G=83 (`/20`) | 0.4052 | 0.3356 | 0.3322 | −0.0697 [−0.0854, −0.0539], Holm p=2.88e−6 | +0.0033 [−0.0042, +0.0109], Holm p=0.3524 |

The local step fails the prespecified positive-signal rule at both caps: it
substantially harms cc-F1 against untouched PTO; it is also worse than the
matched pooled-only dose at G=167 and inconclusive against it at G=83.
Accuracy, macro-F1, and weighted-F1 are lower for the joint step than **both**
comparators at both caps, each with a wholly negative paired 95% interval.
For example, joint-minus-PTO accuracy is −0.1364 at G=167 and −0.1307 at
G=83. The negative result is retained; no radius, cap, epoch, or country rule
was selected from these development scores.

All 72 side-step opportunities per cap were active. The raw calls after the
joint step remained infeasible in 7/72 epoch-cap observations at G=167 and
30/72 at G=83; these are diagnostics, not failures of the final allocator.
Across seeds, the joint step made 244 correct class-1 entries but 443 correct
exits relative to PTO at G=167, and 222 entries versus 347 exits at G=83.
This is direct evidence that the fixed gradient direction changed the ranking
of allocated examples adversely at this 0.1 L2 dose, not evidence that local
constraints in general are impossible to optimize.

The separate seed-6199 pilot passed a label-blind integrity/cost gate after a
provenance-scorer correction recorded in
[`fmow_fixed_gate_provenance_amendment_20260930.md`](fmow_fixed_gate_provenance_amendment_20260930.md).
Its projected total cost was 1.498 GPU-hours, below the 72 GPU-hour ceiling;
the pilot's exploratory score was also negative and did not change the fixed
block. The original five reserved fmow2 test countries remain unscored.
These repeatedly viewed development countries cannot provide independent
confirmation. Any different local-gradient rule or dose is a new, separately
specified study, not a continuation or favorable subset of this block.
