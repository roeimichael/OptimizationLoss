# Knee hard-pair ranking: fixed follow-up to the inactive cutoff loss

**Status: specified and implemented, not yet run.** The completed seeds
1401–1404 showed that the previous wrong-occupant ranking term had an active
pair in only 2/20 rank-only and 1/20 rank-plus-count opportunities. Most
training top-532 sets were entirely true grade 3, so the loss was exactly zero
although roughly 225 true grade-3 training images remained outside the quota.
The four-seed capped-first grade-3 F1 means were Clipper 63.74%, phase Null
64.29%, count-only 61.54%, rank-only 63.46%, and rank-plus-count 62.36%.
Rank-plus-count minus rank-only was −1.10 points, exploratory 95% paired t
interval [−7.48,+5.29]. This closes that *specific* ranking formulation; it
does not test a consistently active supervised ranking term.

## New objective and reason

For each training image, use the same grade-3 log odds
`m_i = z_i,3 − logsumexp(z_i,c≠3)`. Among training true grade-3 images take the
`q` **weakest** margins, and among other training images take the `q` **strongest**
margins, where `q = min(K_train, n_positive, n_negative)` and the already fixed
`K_train=532`. Break ties by stable training ID. Hold these sets fixed during one
auxiliary update and minimize

`L_hard = (1/q²) Σ_{i∈weak true grade3} Σ_{j∈hard non-grade3} softplus(m_j−m_i)`.

This keeps a label-informed gradient when the training top-532 selection is
already pure: near-cutoff non-grade-3 competitors still exist just below it.
Its logistic derivative automatically emphasizes pairs with smaller or reversed
separation. The term is zero only when a class is absent, not merely when a
finite top-K is pure. It tries to make the correct ranking *more robust*, while
the existing count term still pressures the unlabeled development soft count.
The unit coefficient is fixed; no new margin, weight or development-tuned rate
is introduced. The analytic derivative and two-pass parameter update must match
an independent unchunked autograd/Adam oracle before the GPU pilot.

Prediction: the hard-pair gradient should be nonzero at most or all of the five
post-warmup opportunities. If that ranking transfers, rank-only should increase
correct grade-3 membership among the exact 76 development slots over phase
Null. Rank-plus-count should beat rank-only on that same measure if count
pressure helps rather than counteracts the learned ordering. Raw counts and
upper-bound correction are reported separately, because count suppression can
look different after forced slot filling.

## Matched experiment and limits

Use the same audited Chen knee split (5778 train, 826 development, 1656 test
unscored), ImageNet-initialized ResNet18, deterministic 224-pixel transform,
five CE warmup plus five intervention epochs, batch 32, task Adam 0.0001,
separate auxiliary Adam 0.00003, grade-3 development cap 76, and training
capacity 532. Fresh fixed seeds 1501–1504 have five arms: ordinary Clipper,
phase-matched CE-only Null, count-only, hard-rank-only, and hard-rank-plus-count.
The three auxiliary arms each have at most one auxiliary step per post-warmup
epoch; combined rank+count accumulates both gradients before one Adam step.
Training labels may select rank pairs; development images used by the count
term have no labels. Development labels appear only in offline scoring after
all five fits. No test scoring, validation checkpoint selection, altered cap,
or favorable-only seed reporting.

First run seed 1501 as an integrity pilot. Expand seeds 1502–1504 only after
checking source/data identity, all artifact and probability hashes, exact task
and auxiliary update dose, finite gradients/weights, matched initialization,
warmup and batch order, component gradient norms, independent raw and both
allocated metrics, and correct/wrong quota-slot entries/exits. Failure evidence
is retained. All four seeds and paired differences get reported even if negative.

This is a mechanism-motivated exploratory follow-up **on a development split
already inspected repeatedly**. A favorable result would need independent
confirmation on untouched data; it would not by itself establish a paper claim.
