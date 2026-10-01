# Knee TraLO: consolidated research findings, 23 September 2026

**Decision so far: no demonstrated TraLO advantage.** We completed60 head fits and4 backbone adaptations. Reducing the constraint step fixed much of the severe suppression, and supervised knee fine-tuning produced a stronger classifier. Neither established an additional benefit from the constraint term.

## What was compared

Dataset: Pingjun Chen's original OAI knee grading release, DOI10.17632/56rmx5bjcr.1, CC BY4.0. Original splits:5778training/826validation/1656test images. Filename-based patient identities and decoded-image hashes are disjoint across splits. This does not independently verify clinical metadata or exclude every near-duplicate. Test images were checked for overlap but not scored.

One backbone: ImageNet ResNet18. Four seeds901-904. Global upper caps: grade3<=82 andgrade4<=16 on826validation samples. These are synthetic capacities, not Yuval's verified settings. This is an independent mimic of the problem, not an exact paper reproduction.

Clipper trains with ordinary cross-entropy. Null shares TraLO's phase boundary and supervised objective, but omits constraint updates. TraLO adds the constraint updates. Margin and false-positive auxiliaries are included in both TraLO and their corresponding null. All arms receive equal supervised updates. Constraints use unlabeled validation features; validation labels are used for metrics only.

## Results

ccF1 means the average F1 of grades3 and4, shown on a0–100 scale. Every row averages four seeds. A positive delta favors TraLO.95% intervals use seed-paired Student-t differences; they are unadjusted exploratory intervals, not multiple-testing-corrected discoveries. Repeated Clipper runs are controls, not extra independent evidence.

### Only repair predictions exceeding a cap

These are distinct allocation rules applied to the same saved probabilities; do not pool their results.

| Recipe | Clipper ccF1 | Null ccF1 | TraLO ccF1 | TraLO minus null [95% interval] |
|---|---:|---:|---:|---|
| none | 35.83 | 38.00 | 23.63 | -14.36 [-22.82, -5.91] |
| margin | 35.83 | 38.25 | 23.91 | -14.34 [-23.94, -4.75] |
| false_positive | 35.83 | 37.13 | 22.80 | -14.33 [-22.36, -6.30] |
| smaller constraint step | 35.83 | 38.00 | 37.24 | -0.76 [-2.67, +1.16] |
| adapted ResNet18 + smaller step | 68.56 | 68.56 | 68.69 | +0.13 [-0.29, +0.56] |

### Allocate constrained slots first

These are distinct allocation rules applied to the same saved probabilities; do not pool their results.

| Recipe | Clipper ccF1 | Null ccF1 | TraLO ccF1 | TraLO minus null [95% interval] |
|---|---:|---:|---:|---|
| none | 39.65 | 41.08 | 40.13 | -0.95 [-5.25, +3.35] |
| margin | 39.65 | 41.48 | 40.98 | -0.50 [-3.47, +2.47] |
| false_positive | 39.65 | 41.21 | 40.85 | -0.36 [-2.56, +1.83] |
| smaller constraint step | 39.65 | 41.08 | 42.19 | +1.11 [-0.74, +2.96] |
| adapted ResNet18 + smaller step | 68.56 | 68.69 | 68.83 | +0.13 [-0.29, +0.56] |

## What we can explain from the evidence

1. **The constraint step could be too large.** For the original knee seed901, its final update reduced grade3 predictions from95 to29 when the cap was82. With a tenfold smaller constraint learning rate, the corresponding trajectory ended67. Across four seeds, raw constrained-class F1 rose from23.63 to37.24. This intervention changes update size; it does not prove the best size or establish superiority.

2. **Soft probability totals are not hard prediction counts.** In the smaller-step seed901, grade3 had73 hard predictions before the last update, already under82, while its summed probabilities were105.19. The penalty was therefore still active and reduced the hard count to67. This measured mismatch can cause pressure on an already feasible classifier. It is not proof that all soft surrogates fail.

3. **Better features helped every method.** Five training-only epochs of ResNet18 adaptation improved raw accuracy to about61%. That is shared supervised training, not a TraLO contribution. On adapted features, clipped TraLO remains statistically indistinguishable from null in this four-seed experiment.

4. **Sample-aware losses were actually tested.** The margin term rewards a larger correct-class logit gap. The false-positive term penalizes probability assigned to a constrained class when the training label is another class. Both use sample labels, not only counts. At the tested weights with frozen features, neither established a benefit from the additional constraint update. We have not shown that those losses can never work.

## Limits and next decision

The evidence does not support a positive method claim today. It also does not establish mathematical impossibility. Constraints affected the linear head only; the backbone adaptation was ordinary supervised learning. Only one dataset, one backbone, one pair of synthetic caps and four seeds were studied, with development outcomes inspected along the way.

A justified next research question is whether an update rule aligned with the actual global allocation decision can improve sample ordering without suppressing already-feasible predictions. That requires a separately specified objective and stronger evidence; renaming TraLO or selecting one favorable seed would not resolve the current result.

## Evidence records

- [First three recipes](knee_first_stage_20260923.md)
- [Adapted-backbone results and every seed](knee_adapted_result_20260923.md)
- [Adaptation protocol](knee_adaptation_protocol_20260923.md)
- [Full numerical summaries for all recipes](knee_consolidated_20260923.json)

Remote evidence roots: /home/dsi/michaer8/tralo-rebuild/runs/knee-chen-20260923 and knee-adapted-20260923. Immutable experiment releases: fb688418,be4cf796,e101c5bd. Source hashes, data hashes, run receipts, saved predictions, matched warmup/batch hashes, finite-update counts and independent sklearn metric checks were verified.