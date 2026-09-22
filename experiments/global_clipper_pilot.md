# First global-only Clipper pilot

## Question and scope

Can the new pipeline load real images, fit a classifier, save probabilities,
enforce declared global quotas, and produce independently inspectable results?
This is one development pilot, not a TraLO comparison, a tuned baseline, or proof
that a strong converged model finds this dataset difficult. No primary winner
metric is selected. Accuracy, fixed-class macro-F1, constrained-class F1, per-class
counts, feasibility and changed predictions are descriptive diagnostics.

## Data and model

CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html . The official training
split supplies 10,000 training and 2,000 disjoint development examples selected
by the configured seed. The official test split is not used. Integrity, selected
sample IDs, image-byte duplicate checks, file hashes and class supports must be
saved before interpreting results. These checks do not prove absence of semantic
duplicates or establish ImageNet pretraining data independence.

Frozen ImageNet-pretrained ResNet-18 features plus a trained linear classifier
is a modest transfer-learning baseline. It isolates the data/prediction/allocation
path while keeping the first run short; it is not a claim of a strong optimized
CIFAR-100 model. Image size 224 follows the pretrained model's intended scale.
The configured 10 head epochs and learning rate 0.1 are explicit pilot choices,
not fixed project rules. Only supervised labels of the training subset are used
to fit the classifier. Development labels are scored afterward, not used to
select epochs or set caps. FP32 on dsisco01 is an initial numerical reference;
it does not validate future FP16 training or BF16 runs on Blackwell.

## Global policy

Before scores are seen, assign cap 10 to class indices 0 through 9 and leave all
other classes uncapped, on the 2,000-example development pool. This is a synthetic
quota policy for pipeline testing, not a domain-motivated deployment requirement.
No cap reads development labels, their class counts, or predicted scores.

For class c, feasibility means sum_i 1[prediction_i = c] <= K_c.
It is an upper bound, not a required equality. Ties use sample ID then class
index. Raw argmax ties use class index. No label reaches either allocator.

- **upper_bound_correction:** retain raw assignments for classes within their
  caps. For an overfull class, retain its K highest-probability raw assignments;
  release the rest. Assign released items by descending feasible item/class
  probabilities. Retained items are not reconsidered.
- **capped_first:** start empty, assign descending item/capped-class pairs while
  capacity exists; then assign remaining items from all feasible class pairs.
  This can assign a constrained class even when an uncapped class had a higher
  probability. It is a distinct algorithm, not another name for correction.

Both are greedy, neither claims to maximize total probability or F1. Global-only
assignment is feasible when total class capacity covers the pool (an uncapped
class has capacity at least the pool size). Each item may use any class. More
complex eligibility constraints would require a different feasibility argument.
Record both methods, even if one is worse. One seed provides no uncertainty claim.

## Commands (from an immutable DSI release)

```text
CUDA_VISIBLE_DEVICES=<verified-free-index> /home/dsi/michaer8/anaconda3/envs/optloss/bin/python -m tralo.image_baseline examples/cifar100_pilot.json <new-run-directory>
/home/dsi/michaer8/anaconda3/envs/optloss/bin/python -m tralo.global_report <new-run-directory>/development_probabilities.json examples/cifar100_global_caps.json <new-report-directory>
```

The server skill requires ownership checks, source/config hashes and exclusive
paths before these commands. Read `events.jsonl`, probability/checkpoint hashes,
per-class counts, and raw/allocated metrics. A zero exit code alone is insufficient.
If a validation step fails, preserve that attempt and fix it in a new release.
