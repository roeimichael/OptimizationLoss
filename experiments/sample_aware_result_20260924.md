# Sample-aware TraLO: knee and CIFAR-100, 24 September 2026

## Question and experiment

Can TraLO learn **which** constrained predictions to retain, in addition to
responding to the total predicted counts? All numbers here are exploratory
development results. There was no test-set scoring or checkpoint selection.

The first new term uses **training labels only**. For image `i`, let `d_i` be
the largest logit advantage of an eligible wrong class over the true class.
Wrong capped classes are eligible; if the true class is capped, every wrong
class is eligible. `L_far = mean_i max(0,d_i)^2`. An error twice as far across
the logit boundary incurs four times the loss. Correctly separated images get
no added loss; ordinary cross-entropy still trains on every image. This is a
logit-gap proxy, not geometric distance in image/feature space.

After five warm-up epochs, Sample Null and Sample TraLO both train with
`CE + 0.1 L_far`. Sample TraLO alone also receives the original unlabeled
soft-count penalty through one separate Adam step per epoch. The follow-up
tests that step with a training-label anchor: Anchored Null gets an additional
`0.1(CE_train + 0.1 L_far_train)` step; Anchored TraLO gets the same step plus
the soft-count penalty. Thus the anchored pair has equal extra training-label
dose. It was designed **after** the first results and is labelled adaptive.

All methods use the same frozen ImageNet ResNet18 feature cache per dataset,
head initialization and minibatch order per seed, 20 head epochs, batch 256,
task Adam 0.001, and four seeds 1001-1004. Knee has 5,778 training / 826
development images and caps grade 3 ≤82, grade 4 ≤16. CIFAR-100 uses a fixed
10,000/2,000 subset of its official training split, with caps 10 for each of
classes 0-9. Other classes are unrestricted. Caps were set before outcomes.
FP32 Quadro on dsisco01 for both studies. The source releases are
`f01cbeb9f2c35df6809232192be3860bbe9896bd` and
`f411d4b91af00b648781a22fc289a2be650b4d4c`, respectively.

## Original-cap results

The table reports **constrained-class F1 points after capped-first allocation**.
Every arm receives exactly the same constrained-class output slots, so a score
gain cannot come from predicting more constrained samples. Clipper has CE only;
Sample Null adds the new sample term; Sample TraLO additionally applies the
count term. The anchored pair adds matched extra supervised steps as described
above. Higher F1 is better. These are four-seed means on each fixed development
split, not patient/population confidence claims.

| Dataset | Clipper | Sample Null | Sample TraLO | Anchored Null | Anchored TraLO |
|---|---:|---:|---:|---:|---:|
| Knee | 38.04 | 43.35 | 43.17 | **43.94** | 43.17 |
| CIFAR-100 subset | **53.52** | 52.73 | 52.94 | 52.73 | 52.94 |

Bold identifies the largest observed mean in each row. For knee it is Anchored
Null; for CIFAR it is Clipper. Bold is
descriptive, not a significance claim. The full seed values and raw/upper-bound
policies are in [the sample analysis](far_error_result_20260924.json) and
[the anchor analysis](anchor_result_20260924.json).

| Dataset | Sample TraLO − Sample Null, seed deltas | Mean, paired 95% interval | Anchored TraLO − Anchored Null, seed deltas | Mean, paired 95% interval |
|---|---|---|---|---|
| Knee | −1.79, −2.33, +2.33, +1.06 | −0.18 [−3.74, +3.38] | −1.79, −2.33, 0, +1.06 | −0.76 [−3.27, +1.74] |
| CIFAR-100 subset | 0, +0.87, 0, 0 | +0.22 [−0.47, +0.91] | 0, +0.87, 0, 0 | +0.22 [−0.47, +0.91] |

Intervals are ordinary Student-t intervals over four paired training seeds on
the same already inspected development split; they are uncorrected for these
and earlier comparisons. Neither paired difference establishes an incremental
count-term benefit. Raw constrained-class F1 deltas for Sample TraLO minus
Sample Null were −1.42 knee and −0.71 CIFAR; the allocator changes that view.

Relative to the **old** no-auxiliary Null on the same four seeds, Sample Null
changed knee allocated constrained F1 by +3.17 points, paired interval
[−3.32, +9.67]; CIFAR by −0.34 [−1.50, +0.82]. The knee improvement is
interesting but uncertain and belongs to the sample term/pipeline, not to
TraLO's count term. The matched Clipper replay was byte-identical in score to
the old Clipper for every seed.

The anchor changed probabilities, so the intervention was active: maximum
per-score change in the first seed was up to 0.0107 for knee Null and 0.0121
for CIFAR Null. Yet anchored TraLO had exactly the same final allocated F1 as
Sample TraLO in all eight dataset/seed cases. Its final allocated labels changed
on 0, 1, 1, and 3 knee images and no CIFAR images. This particular extra
supervised anchor did not alter the constrained outcome enough to help.

## Professor's achieved-count comparison

For each Sample TraLO seed, its **raw** constrained-class counts were copied as
new per-class caps, then Clipper, Sample Null and Sample TraLO were reallocated
at those identical new caps. This is offline evaluation of saved probabilities,
not another training run. No labels determine the new caps or allocation.

On knee, TraLO's achieved counts for grades 3/4 were 22/14, 81/9, 79/5 and
41/8, versus the original caps 82/16. All four raw TraLO outputs therefore
satisfied the original *upper bounds*, often by leaving substantial capacity
unused. At these TraLO-derived caps, mean capped-first constrained F1 was
32.79 Clipper, **38.61 Sample Null**, 37.67 Sample TraLO. This does not show
that TraLO selected better images at its own counts.

On CIFAR, many TraLO-derived caps exceeded the original cap of 10. The net
changes summed across capped classes were +65, +74, +91 and +98 in seeds
1001-1004. Under these **relaxed** TraLO-derived caps, means were 61.28
Clipper, **61.98 Sample Null**, 61.98 Sample TraLO. Those scores cannot be
presented as satisfaction of the original CIFAR constraint. A fixed 20-epoch
run is not a convergence-to-satisfaction experiment.

## Checks, interpretation and next direction

The new releases passed 91 and 93 native tests respectively on both DSI hosts,
source hash parity, CLI and GPU smoke checks. The completed studies contain
24 sample-stage and 16 anchor-stage head fits. For each arm, saved artifact
hashes, finite updates, matched warm-up/batch hashes, 15/0 count steps as
declared, 15 matched anchor steps in the follow-up, exact capped-first slots,
and independent scikit-learn metrics passed. Full predictions, checkpoints,
configs and logs remain under the two exclusive server roots
`/home/dsi/michaer8/tralo-rebuild/runs/far-error-20260924` and
`/home/dsi/michaer8/tralo-rebuild/runs/anchor-20260924`.
Both roots were also archived locally at
`C:/Users/roeym/.codex/rebuild-audit-20260922/far-error-and-anchor-20260924.tar.gz`
(45,665,143 bytes; SHA-256
`ef3d94a5187802efebc586d6ccef3dcba761607263cf22c0aadbc690010abaed`).

These experiments support a narrower result than either “TraLO works” or
“constraints cannot help”: adding a stronger training-label error signal did
not make this soft-count penalty reliably useful, and placing that signal in
the count step also failed here. They do not test end-to-end backbone training
or a learned quota cutoff.

The next distinct hypothesis is **quota-aware ranking**: learn to put true
constrained examples ahead of false ones at the actual K-th allocation cutoff,
while retaining the original cap. A count loss sees the total; a ranking loss
would train the quality of the selected slots directly. This requires a
training-label-only objective and a null with the same supervised ranking
term. It is a proposed experiment, not an observed gain or a novelty claim.
Related primary work includes
[constrained classification via quantile thresholds](https://research.google/pubs/constrained-classification-and-ranking-via-quantiles/)
and [transductive top-K precision](https://arxiv.org/abs/1510.05976).
Our multiclass greedy allocator and F1 target differ from those papers, so
their reported results do not transfer numerically to this experiment.
