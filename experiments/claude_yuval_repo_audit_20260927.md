# Audit: our knee pipeline vs Yuval Kassif's ConstrainedClassification (read-only, 2026-09-27)

Scope:
- Ours: OL-bandcons @ c7887054, `tralo/knee_e2e_v3.py` and its helpers.
- Yuval: `yuval-ConstrainedClassification` @ 413d96c (2026-09-25). The repo tracks 24 files, all `.py`, with no results committed.

Checks script and raw output: `scratchpad/audit/audit_checks.py`, `scratchpad/audit/audit_checks.json`. They were run locally on CPU against stored v3 probabilities (seeds 1801-1824, 72 arm-runs) and synthetic matrices.

## (A) Bugs on our side: none found in data, labels, allocator, metrics or model mode

| Check | Result | Evidence |
|---|---|---|
| Train loader | Shuffled every epoch; every item exactly once per epoch; last partial batch kept; dose asserted. | v3:141 `randperm`, v3:145, v3:131/202 planned==applied |
| Label/row alignment | Dev pool, `val_ids` and labels come from the same manifest filter in the same order. Recomputing capped_first from the manifest reproduces the stored report exactly (seed 1801 clipper: cc-F1 0.692308, acc 0.600484). | knee_end_to_end.py:55; v3 `run()` `val_rows`; audit_checks.json `report_crosscheck` |
| Dev pool order | The pool is class-sorted: manifest paths are `val/GRADE/id.png`, and `labels == sorted(labels)` is True. Harmless, because tie-breaks use `sample_id` (the patient id), never the index. | global_clipper.py:48-52 |
| Grayscale, resize, normalization | `convert('RGB')` replicates the channel; Resize 224 on 224-px images; ImageNet mean/std. He uses the same RGB conversion with dataset mean/std 0.6613/0.2123. An input affine shift only. | knee_end_to_end.py:46-49, supervised_adaptation.py:51; his load_data.py:119 |
| BatchNorm mode | `infer`, `streamed_step` (hence the count gradient) and `targeted_step` all run in eval mode and restore the previous mode. The step direction is the exact gradient of the *deployed* soft count. Correct, not a defect. | knee_end_to_end.py:72-84; streamed_constraint.py:49-50,82-83 |
| Allocator vs his `constrained_classification` | 300/300 random matrices identical at integer caps. **72/72 stored knee runs identical, 0 items differ.** | audit_checks.json |
| Metrics vs sklearn | Our macro-F1 equals sklearn macro to 1e-12 on all 72 runs (all 5 classes present in dev: 328/153/212/106/27). The F1 formula is the same. | metrics.py:37-43 |

Design weaknesses (not code bugs) that cost statistical power:
- **A-w1.** The deployed model is a noisy draw. The final-epoch raw grade-3 count swings ±30 from one epoch to the next in a memorised model; seed 1801 clipper ran 91, 94, 67, 84, 122 over epochs 6-10.
- **A-w2.** The cap binds on the raw argmax count in only 21/24 v3 clipper seeds (raw counts 49-125). In the other 3 seeds, any count-triggered treatment (TraLO step, PAO) is a no-op, which dilutes the effect.

## (B) Pipeline choices of his that plausibly improve the base model or the constraint's effect (ranked)

**B1. The mechanism difference: PAO is label-aware, TraLO is not.** `CustomLoss` equals C_{y}·CE on training rows whose argmax is k, and plain CE elsewhere (C_k = 1). Checked numerically: loss matches the weighted CE, gradient max diff 7e-4 (losses.py:14-52). It upweights **training false positives of k**, which tells the model *which* items are not k. TraLO's pool count only knows *how many*.
- Its signal needs training false positives to exist. Our recipe memorises the training set: seed 1801 training CE was 0.15 at epoch 5 and 0.04-0.10 over epochs 6-10.
- So in our recipe PAO's signal would be mostly dead, as the CUTPAIR gate found (N_act 1.2/13.4).
- What to do: run PAO only in a recipe that does not memorise (B2+B3), with a matched arm; do not port it into v3.

**B2. Augmentation on every epoch.** hflip 0.5, rotation 3°, RandomAffine translate 0.1 / scale 0.9-1.1, ColorJitter 0.2/0.2/0.2 (load_data.py:64-84); present before the 2026-09-25 commit too. We have none: v3:245 "no augmentation". It delays memorisation and keeps training false positives alive; our CUTPAIR study already measured about +3 F1 from augmentation alone.
- What to do: adopt his train transform for all arms, drawing its parameters from a per-arm reseeded RNG.

**B3. Early stopping with a best-val-loss checkpoint.** Patience 5, up to 75 epochs (train.py:130-146; config.py:38,43). Caveats:
- Before 2026-09-25, `best_model = model` was a reference, so his returned model was the last epoch, about 5 epochs past the minimum.
- In PAO iterations the val loss is the C-weighted criterion (train.py:113), so the stopping point moves with C.
- We deploy epoch 10 of a memorised, count-swinging model (A-w1).
- What to do: early-stop on a patient-disjoint split carved from TRAIN, never on the dev pool.

**B4. Class-balanced WeightedRandomSampler.** Weights are 1/count (load_data.py:56-61). It is wired in and on by default **only since 413d96c (2026-09-25)** (config.py:45); earlier results used uniform shuffling.
- It over-samples grade 3 about ×1.5 and grade 4 about ×6.7, which raises raw grade-3 predictions.
- The cap then binds in every seed, fixing A-w2, and PAO's F > 0 in every seed.
- What to do: add it to the shared recipe; report the binding rate.

**B5. Optimiser recipe.** Adam 1e-4 with WD 1e-4, LR ×0.8 every 5 epochs, and in `full_experiment.py` a grid over lr {1e-3, 1e-4, 5e-5} × epochs {30, 50, 75} by val accuracy (train.py:7-22; config.py:34-43). Ours is constant Adam 1e-4 for 10 epochs with no WD (v3:122).
- What to do: the schedule is cheap to adopt. WD only equally across all arms, as "his pipeline".

**B6. Backbone.** timm EfficientNet-B5 at 224 px (config.py:58), about 30M parameters versus ResNet18's 11.7M. His hard-coded unconstrained accuracies are B0 66.18, B5 68.78, R50 67.03, R101 65.94 (utils.py:235-240). Our ResNet18 clipper reaches 58.9% dev accuracy under capped_first; this is not the same split, but his base model is clearly stronger.
- What to do: this is a secondary factor. Weights are not cached on the server; torchvision EfficientNet weights would have to be uploaded.

**B7. Dynamic LR.** lr = (1−t)·base/mean(C) + t·base per batch (train.py:59-76). It reduces the LR when many samples in the batch are predicted k and mean(C) > 1. It is part of his method, and also a confound (C4).

## (C) Artefacts in his evaluation that could manufacture a PAO > PTO gap (ranked)

**C1. No reseed inside `full_experiment.py`'s outer loop.** `set_seed` is called only at :358, while `get_model` at :235 runs every iteration. Each PAO retrain therefore gets a fresh head, batch order and augmentation stream, so PAO − PTO contains pure retrain noise. Our measured arm-vs-arm floor is on the order of the seed sd (~0.011 cc-F1). `main.py` does reseed (:157), so the two scripts are not the same experiment.
- What to do: pair iterations with common random numbers, as in main.py.

**C2. One seed, no interval, no held-out confirmation.**
- PTO is a single draw per percent; full_experiment retrains it per percent (:222/:260).
- The same test set sets the cap, stops the loop and reports the result.
- What to do: many seeds, paired t CI, with PAO − PTO as a preregistered contrast.

**C3. Test labels and test predictions steer the method.**
- N_K = true test count × percent (main.py:138; full_experiment.py:228).
- The loop stops when the TEST argmax count ≤ N_K (main.py:190,259; full_experiment.py:263,337), so the reported PAO model is chosen by a test-set statistic.
- What to do: derive the cap from training prevalence; count on the unlabeled deployment pool only.

**C4. Three changes bundled into "PAO", with no control.**
1. C-weighting of predicted-k rows;
2. LR scaled down by mean(C) on those batches (train.py:67-73);
3. an early-stop criterion that is itself C-weighted (train.py:113).

   An accuracy gap could come from (2)/(3), a lower effective LR or a later stop, rather than from the false-positive weighting.
   - What to do: add a control retrain with the same effective LR and stopping but uniform C.

**C5. Outcome filtering in `weights_impact.py:46-54`.** `good_ones` keeps only runs with PAO accuracy > PTO accuracy and |count − N_K| < 20 (class index 2 only). `optimization_analysis.py:23-43` also plots **every iteration's** results.json as "PAO", including non-converged ones.
- What to do: report every seed and only the converged final iteration; never filter on the outcome.

**C6. The metric is not the constrained class.**
- results.json has accuracy plus macro/weighted P/R/F1 over all classes (optimization.py:94-111).
- The constrained-class F1 is only printed, never stored.
- With exact-fill allocation, a retrain can gain accuracy on unconstrained classes (noise, LR) and read as a constraint win.
- What to do: report cc-F1 next to accuracy/macro/weighted.

**C7. Pre-2026-09-25 early-stopping reference bug** (git show 413d96c -- train.py). Both arms used last-epoch models, which is symmetric but noisier. Because the C-weighted val loss stops PAO at a different epoch, the effective training length differs between arms.

**C8. Loss numerics, and a verification that never verified anything.**
- `CE_verification.py:8-22` applies softmax twice and uses C = [1, 1, 30, 1, 1] while labelling it "C=1".
- My check: CustomLoss(C=1) ≈ CE, except that the 1e-7 inside the log caps per-sample loss at about 16.1 nats (a zero gradient when p_y < 1e-7).
- A near-tie (argmax ≠ k by about 1e-8 in probability) gives a gradient norm of 14,349 versus 0.50 normally, from the tanh(5e7·gap) gate. These are sporadic spikes; not a gap artefact by themselves.

**C9. PAO equals PTO exactly whenever PTO's argmax count ≤ N_K.**
- In that case F = 0 at iteration 1: dF = 0 for N_K_p ≤ 76, and dF = 4·(N_K_p − N_K) above it (a C increment of +2.35 per iteration at 120 vs 76).
- Gaps can only appear at tight caps where PTO over-predicts, and only the balanced sampler (after 09-25) makes that universal.

Neutral, both sides symmetric:
- For non-integer N_K his loop fills ceil(N_K): 156.1 gives 157, 66.9 gives 67 (`count < N_K`, optimization.py:68).
- `row[k] = 0` mutates `all_probabilities` in place and corrupts the saved CSV's p_k column for unselected rows (optimization.py:72-73), but not the predictions.
- Exact ties at the cut change the selected set (53/200 synthetic trials: his argsort is unstable and the test order is class-sorted ImageFolder order), but real knee outputs have **0/72** ties straddling rank 76 (at most 1 duplicate p3, at most 1 p3 == 1.0).

## Our existing v3 study read on his metrics

Capped_first, cap 76, n = 24 paired seeds, t 95% CI:

| Contrast | Accuracy | cc-F1 | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| tralo_target − clipper | +0.0019 [−0.0104, +0.0141] | +0.0119 [−0.0034, +0.0272] | +0.0010 [−0.0075, +0.0095] | −0.0007 [−0.0087, +0.0074] |
| tralo_target − tralo_null | +0.0045 [−0.0056, +0.0145] | +0.0096 [−0.0096, +0.0288] | +0.0036 [−0.0067, +0.0139] | −0.0011 [−0.0095, +0.0074] |

On his metrics, TraLO in our recipe is a null. The metric switch does not rescue it.

## Bottom line

- No code bug on our side explains the gap.
- The likely difference is (B1) a label-aware false-positive-weighting signal, which only lives in a non-memorising recipe (B2-B4).
- His reported PAO > PTO rests on single-seed, test-steered, partly outcome-filtered evidence with bundled confounds (C1-C6).
- The fair test is PAO − PTO paired with common random numbers, over many seeds, in his recipe with an early-stop split carved from train, plus a uniform-C control retrain.
