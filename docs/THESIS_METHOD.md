# What TraLO changes, and what it has not yet shown

## The decision problem

An image classifier returns probabilities `p[i,c]` for item `i` and class `c`. If only `K[c]` items may receive an expensive class-`c` decision, ordinary prediction followed by a **Clipper** can satisfy the upper bound by changing deployed labels after training. This is the minimum baseline: a feasible count alone says nothing about whether the limited slots went to the right items. We therefore score the *deployed* predictions with constrained-class F1 (cc-F1), accuracy, macro-F1, weighted-F1, precision, recall, per-class supports, and a confusion matrix. All arms use the same label-free allocator.

## Kassif and Singer's method, and what we reuse

[Kassif and Singer's paper](https://www.sciencedirect.com/science/article/pii/S0952197626022736) proposes an adaptive cost-sensitive training loop. The [author's code](https://github.com/YuvalKassif/ConstrainedClassification) and our [source audit](../experiments/claude_yuval_repo_audit_20260927.md) establish the specific comparison implemented here. A pretrained image backbone with a fresh classification head is trained on labeled knee images. Their PTO reference trains once. Their PAO loop observes how many unlabeled deployment images are predicted in the capped class, adjusts a class-dependent cost matrix in the supervised loss, and retrains. The knee port retains the augmentation, class-balanced sampler, optimizer schedule, stopping rule, and PAO arithmetic. We changed the stopping split to a subject-disjoint carve from the training set, fixed caps from training-only information, and evaluate every arm through the same allocator; these are comparison safeguards, not TraLO contributions. The tested backbones are EfficientNet-B5, MobileNetV3-Large, and ViT-B/16. No backbone architecture was invented by TraLO.

## The TraLO mechanism in this repository

Let `S[c] = sum_i p[i,c]` be the *soft* predicted count on an unlabeled deployment cohort, and let `K[c]` be its declared upper bound. The original global TraLO primitive uses `e[c] = max(S[c]-K[c], 0) / max(K[c], 1)` and adds a separate constraint penalty:

```text
L_count = sum over capped c of lambda[c] *
          ( e[c] / (1 + e[c]) + rho * e[c]^2 / (1 + e[c]^2) ).
```

Its exact cohort gradient is accumulated in two passes: first obtain all probabilities at fixed weights, then replay the same images to backpropagate the analytic logit gradient. No development labels enter this update. For several class or group constraints, different residuals can give different directions. An ALM control uses its own projected-dual residual update, and a local quota uses a supplied group attribute to form `S[g,c] = sum_{i:attribute[i]=g} p[i,c]` with its own cap `K[g,c]`. A group attribute such as sex is separate metadata, **not inferred from the image or its diagnosis**. No patient-tabular knee comparison has yet been validated.

The present persistent knee study tests a **calibrated variant**. After each supervised epoch, when the raw grade-3 count exceeds the cap, it computes the negative gradient of the grade-3 soft count, normalizes that direction, and searches for the smallest parameter displacement along it that meets the hard count. To obtain this direction, the runner temporarily sets the soft-count gradient cap to zero. It may therefore act when the hard count is over the declared cap even if the original soft-count penalty at that cap has no gradient. Training then continues from the altered model. The sham arm receives the same per-tensor displacement magnitudes in a random direction. A no-change replay is checked in the pilot, while PTO plus the same Clipper and Kassif's PAO are rival outcomes. PAO can spend multiple full retrainings, so its compute must be reported alongside quality.

**Scope of the novelty claim:** with only one capped class, the derivative of the original TraLO penalty is a positive scalar times the gradient of `S[c]` whenever that soft constraint is active. Normalization cancels the scalar; the boundary search chooses the step length. The zero-cap direction and hard-count trigger mean the tested knee variant is not literal gradient descent on the original penalty at its declared cap. Therefore a knee win would support this count-informed *direction and update procedure*, not prove the particular saturating penalty curve, multiplier schedule, or a new backbone is superior. Those stronger claims need separate multi-constraint or matched-loss tests. Both Kassif's method and TraLO already use predictions on unlabeled deployment images, so that access alone is not unique. The defensible distinction relative to his method is **direct differentiation of cohort capacity through the model**, versus repeated supervised cost-matrix retraining.

## What would justify a paper claim

The fixed knee comparison must first pass source, data, split, gradient, dose, artifact, and no-label-leakage gates. A quality gain must be paired over prespecified seeds, survive comparison with both PTO/Clipper and the dose-matched sham, and not be dominated on secondary metrics. A full ALM baseline and a genuine local group-constraint study remain separate requirements for a broad dual-constraint claim. The repeatedly viewed knee and fMoW development cohorts cannot be called held-out confirmation. The sealed Chen test requires one fixed method and a separate evaluation decision; a negative result must be reported just as clearly as a positive one.
