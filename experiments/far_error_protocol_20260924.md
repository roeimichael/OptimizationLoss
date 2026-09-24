# Training-label far-error TraLO experiment, registered 24 September 2026

## Question

Can a term that distinguishes confidently wrong samples from correct samples
improve the ranking of constrained predictions, while the existing TraLO count
term continues to control the number of predictions? This tests a different
loss shape. It does not assert a theorem that every seed must improve.

## Loss and controls

For training image i with true class y, let eligible competing classes be every
class if y is capped, or only capped classes if y is uncapped. Let
`d_i = max_eligible(z_ic - z_i,y)`. Define
`L_far = mean_i [max(0,d_i)]^2`, with zero for an image having no eligible rival.
The square gives an error twice as far across a logit decision boundary four
times the loss. Correct predictions with positive separation get zero from this
term. This is a logit-gap proxy, **not a normalized geometric distance** in
feature or image space. Cross-entropy remains active on every training image.

After five warm-up epochs, task updates use `CE + 0.1 L_far` for both new Null
and new TraLO. The new TraLO additionally uses the existing separate Adam count
update after each epoch, with constraint learning rate 0.0001. This isolates the
count update against a null that receives exactly the same sample information.
Ordinary Clipper uses CE only. Existing no-auxiliary arms on the same four seeds
are separate reference trajectories, never counted as additional seeds.

The 0.1 auxiliary coefficient is the previously registered coefficient for the
margin and false-positive terms. The zero margin focuses on confidently wrong
examples. These choices are fixed before outcomes; there is no score-based
selection of alternate weights or rates in this campaign. The count penalty
still uses only unlabeled development features; training labels enter only the
task minibatch. Development labels enter metrics after training, never a
gradient, quota or selection decision.

## Fixed comparison

- Datasets: Chen knee OA v1, 5,778 train / 826 development; official CIFAR-100
  train subset, fixed 10,000 train / 2,000 development. No test split scoring.
- Frozen ImageNet ResNet18 features, existing hashes and synthetic global caps.
  Knee grades 3/4 have caps 82/16; CIFAR classes 0-9 have cap 10 each.
- Seeds 1001-1004, chosen as the first four of the completed independent
  12-seed sweep. New Clipper, new Null and new TraLO are matched within seed.
  Compare existing no-auxiliary results only as contextual matched references.
- Twenty head epochs, five warm-up, batch 256, task Adam 0.001, separate count
  Adam 0.0001, initial lambda 0.01, increment 0.05, fixed rho 0.5. FP32 and
  one server architecture for all new runs. No checkpoint choice on labels.
- Primary output: constrained-class F1 after named `capped_first` allocation
  with exact identical constrained-class slots. Report accuracy, macro-F1,
  per-class confusion/counts, raw and upper-correction policies, feasibility,
  actual update dose, runtime and all seed-paired differences as well.

This is an exploratory, repeatedly inspected development comparison; four
seeds cannot establish generalization. Report a null gain separately from the
incremental TraLO-minus-new-Null effect. Negative results remain evidence.

## Launch gate

Check mathematical gradient and masking fixtures, no-cap zero, full local and
both-host native tests, immutable committed source and byte parity, both hosts'
actual GPU PID ownership, and first-seed finite/matched-update/artifact checks
before dispatching remaining seeds. Preserve exclusive launch receipts and all
predictions. Do not combine Quadro and Blackwell trajectories as paired data.
