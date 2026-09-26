# Why the count constraint cannot push the capped-class boundary in the correct direction

Synthesis of three investigations (lab_mech, lab_losslab, lab_lit), 2026-09-26.
Real data: knee dev set, 826 items, 106 true grade-3. Two finished targeted-step studies: cap 76 (claude-target-20260925) and cap 50 (claude-target50-20260926), each 5 arms x 24 seeds.
All CIs are 95% seed-bootstrap intervals with n = 24 seeds unless stated otherwise. Dev labels were used only to score finished runs offline. Nothing that trains was chosen on them. The test split and the live study were not touched.

---

## 1. The one-sentence answer

A penalty on the count alone has a logit-space direction, p_i3 (e_3 - p_i), that only ever demotes. Within the slots it keeps the p3 order, so it reproduces the post-hoc cut. Its weight peaks at the argmax boundary (p3 ≈ 0.5, rank ≈ 89), not at the capped boundary (rank 76 or 50), and it has no promote component, so it cannot lift the correct items that sit just outside the cap. The only who-information that does get in comes from parameter sharing. That information follows where the uncertain mass is dense, not where the truth lies, and on a memorised training set the next CE epoch erases it.

## 2. Proof sketch

**2.1 Bayes target.** Under a cardinality constraint ("select exactly K"), the Bayes-optimal selector is the top-K by η_3(x) = P(y=3 | x). The same holds for linear-fractional metrics and for threshold-quasi-concave metrics, where the optimum is a threshold on η (Koyejo et al. 2014; Narasimhan et al. 2014; Kar, Narasimhan & Jain 2015; Yan et al. 2018; Boyd et al. 2012 for the precision-at-K form). `capped_first` (top-cap by p3) is therefore the exact plug-in form of the optimal rule. A training-time constraint can beat it only by making p3 a better estimate of η_3 near the cut. The fairness literature says the same thing: post-processing a Bayes-quality score is optimal under the constraint (Hardt et al. 2016; Menon & Williamson 2018), and the post-processing Pareto frontier contains every in-processing method tested (Cruz & Hardt, ICLR 2024).

**2.2 A single count cannot identify who.** A cap is one linear moment of the labels, and all C(N, K) labelings with K positives satisfy it equally. From one unlabelled set, the risk of an arbitrary classifier cannot be estimated without bias. Two sets with different class priors are the minimum supervision (Lu, Niu, Menon & Sugiyama, ICLR 2019). The same holds across learning from label proportions (Quadrianto et al. 2008; Patrini et al. 2014; Yu et al. 2014; Scott & Zhang 2020). This is LEDGER #9 in formal terms: the count knows how many, not who.

**2.3 The exact projection preserves order.** The KL projection of the predictions onto a count constraint (posterior regularisation, Ganchev et al. 2010; expectation regularisation, Mann & McCallum 2007) is an exponential tilt of class 3. The tilt is strictly monotone in p3, so the top-K set cannot change. Prior-shift correction (Saerens et al. 2002) and logit adjustment are the same fact. Check: across 200 random pools (N = 826, C = 5, K = 76, λ ~ U(0, 10)), the top-76 set changed in 0 of 200.

**2.4 The gradient step demotes only and is ordered by p3.** The count-penalty logit gradient is g_i = a · p_i3 (e_3 - p_i) with a > 0 (tralo/streamed_constraint.py, `count_logit_gradient`). Descending along it changes item i's grade-3 log-odds L_i by

  dL_i = -a · w_i · dt,  w_i = p_i3(1 - p_i3) · c_i,  c_i = |e_3 - p_i|² / (1 - p_i3)² ∈ [1.25, 2] for K = 5.

- *Sign.* w_i ≥ 0, so no item's grade-3 log-odds can rise. There is no promote term.
- *Order.* If c_i were the same for every item, dL/dt = -a σ(L)(1 - σ(L)) c would be one autonomous 1-D ODE, and the flow of such an ODE preserves order. Items can swap places only through differences in c_i, which measure how concentrated the rival-class mass is.
- *Location.* p(1 - p) is largest at p3 = 0.5. The weight therefore falls on the argmax boundary, while the capped boundary sits at p3 ≈ 0.85-0.99, where p(1 - p) is 10-100x smaller.
- *Closed form.* One free-logit step changes p3 by -η κ p3² [(1 - p3)² + Σ_{k≠3} p_k²]. This is a fixed function of each item's own probability vector, so a post-hoc rule can reproduce it exactly without training (finite-difference check: max abs error 7.4e-7).

**2.5 With shared parameters, reordering has to come through the kernel.** In a network, Δz_j = -η Σ_i Θ(x_j, x_i) g_i. Reordering beyond a score transform therefore needs the tangent kernel Θ to predict false positives better than p3 does. That is a cluster, smoothness or expansion assumption (Joachims 1999; Zhu, Ghahramani & Lafferty 2003; Grandvalet & Bengio 2005; FixMatch; ReMixMatch). With a random-feature kernel, the kernel-routed step moved no more items across the cut than the free step: 1.25 ± 0.54 vs 1.7 ± 0.95 items leaving the top-76 at η = 10, n = 20 pools. In-processing is known to beat post-processing only when the hypothesis class is restricted, the constraint attribute is unavailable at prediction time, or representations are learned under base-rate conflict (Woodworth et al. 2017; Zhao & Gordon 2019). None of these applies cleanly to a count on the same score that the cut thresholds.

## 3. Measurements confirming it on real data

**3.1 The sign and formula hold exactly.** On every tralo_null pre-constraint snapshot (epochs 6-10; 120 snapshots per cap, 99,120 item-snapshots per cap), `count_logit_gradient` matched a · p3 (e_3 - p) to a relative error of at most 7.4e-16, and the first-order dL formula matched to at most 2.1e-6. a_min = 3.0e-5 > 0. Items whose log-odds rose: **0 of 99,120** at each cap. Observed c range: [1.251, 2.000].

**3.2 The weight lands on the wrong boundary.**
- The largest w_i sits at rank 88.7 [85.5, 92.3] at cap 76 and 89.6 [85.9, 93.6] at cap 50, where p3 = 0.498 [0.491, 0.506]. That is the argmax boundary (hard count ≈ 92-93).
- The top slot gets 0.0002 [0.0001, 0.0003] of the last slot's weight at cap 76, and 0.0003 at cap 50.
- Share of total w, cap 76: slots 0.250 [0.221, 0.279]; wrong occupants 0.089 [0.080, 0.100]; **true grade-3 items outside the slots 0.241 [0.232, 0.250]**, i.e. the items that should be promoted are being pushed down.
- Share of total w, cap 50: slots 0.050 [0.037, 0.064]; wrong occupants 0.012 [0.008, 0.016]; true grade-3 outside 0.345 [0.332, 0.359]; far items 0.154 [0.132, 0.176].
- Items with p3 > 0.99 carry 0.47% of the logit-gradient norm. The uncertain band carries 78% (cap 76) and 60% (cap 50).

**3.3 The direction adds nothing beyond p3.** AUC for separating wrong from correct occupants among the top-cap-by-p3 slots (pooled over epochs 6-10):

| | cap 76 | cap 50 |
|---|---|---|
| AUC(-p3) | 0.736 [0.725, 0.747] | 0.685 [0.660, 0.711] |
| AUC(w) | 0.726 [0.713, 0.739] | 0.684 [0.660, 0.710] |
| paired AUC(w) - AUC(-p3) | **-0.010 [-0.016, -0.006]** | -0.001 [-0.003, 0.001] |
| residual AUC(c \| p3) | **0.458 [0.443, 0.472]** | 0.488 [0.462, 0.515] |
| mean wrong occupants | 16.5 of 76 | 6.4 of 50 |

c_i, the only term that can reorder, carries no information once p3 is controlled. At cap 76 it points the wrong way.

**3.4 The idealised flow is the post-hoc cut.** Gradient flow on free per-item logits along p3(e_3 - p), from each binding tralo_null snapshot, until the hard count reaches the cap:
- Cap 76 (95 snapshots): items admitted **0**; eviction overlap with the post-hoc cut 0.861 [0.817, 0.898]; top-cap set overlap 0.992 [0.989, 0.994]; correct slots -0.21 [-0.31, -0.11].
- Cap 50 (119 snapshots): admitted 0; overlap 0.958 [0.936, 0.974]; top-cap overlap 0.987; correct slots -0.04 [-0.14, +0.05].
- An independent reimplementation (lab_losslab real.py, every applied-step snapshot) gives the same picture: top-cap overlap 0.991 [0.989, 0.994] at cap 76 and 0.987 [0.985, 0.990] at cap 50, with correct slots per step -0.17 [-0.26, -0.07] and -0.11 [-0.19, -0.03].

So the targeted step's 83-87% eviction overlap with the post-hoc cut is exactly what the count direction does on its own.

**3.5 The direction is anti-aligned with the oracle.** Cosine in the 826x5 centred-logit space between the count direction and the oracle direction (demote wrong-inside, promote correct-outside):
- Cap 76: **-0.157 [-0.176, -0.136]**. Split into demote part +0.156 [0.140, 0.172] and promote part **-0.313 [-0.325, -0.301]**. Restricted to 20 ranks of the cut: -0.062 [-0.074, -0.051]. Random baseline: 0.000, sd 0.017 (200 draws per snapshot). The cosine falls from -0.10 at epoch 6 to -0.20 at epoch 10.
- Cap 50: -0.067 [-0.080, -0.053]; 20-rank window -0.039 [-0.048, -0.030].
- The post-hoc cut written as a logit direction is more anti-aligned still: -0.321 [-0.352, -0.289] at cap 76 and -0.284 [-0.302, -0.265] at cap 50.
- No demote-only direction can exceed cos 0.7071.

**3.6 The real parameter-space step does carry who-information, but only locally.** Measured on before/after snapshots at every applied step (cap 76: 96 target and 94 sham steps; cap 50: 120 and 120):

| | cap 76 | cap 50 |
|---|---|---|
| AUC(step dL) among occupants, target vs sham | 0.671 [0.655, 0.688] vs 0.507 | 0.734 [0.712, 0.756] |
| residual AUC \| p3, target vs sham | 0.601 [0.582, 0.620] vs 0.501 | 0.696 [0.666, 0.726] |
| AUC(-p3) before -> after step | 0.736 -> 0.754 | 0.686 -> 0.763 |
| target - sham on that AUC | +0.022 [0.008, 0.035] | +0.073 [0.049, 0.095] |
| correct slots per step | +0.11 [-0.16, +0.36] | **+0.62 [0.35, 0.86]** (paired vs sham +0.62 [0.37, 0.86]) |
| cos(real step, oracle), target vs sham | +0.0079 [0.0039, 0.0115] vs 0.0010 | +0.015 [0.012, 0.018] |

- At cap 50, 5.5 items enter the slots per step, 4.26 of them true grade 3.
- An independent WHO-AUC scoring (lab_losslab real2.py) agrees: 0.545 [0.524, 0.565] at cap 76 and 0.602 [0.583, 0.621] at cap 50, vs sham 0.503 / 0.507. Target - sham: +0.042 [+0.018, +0.065] (20/24 seeds positive) and +0.095 [+0.073, +0.117] (23/24).
- The cos(real step, count direction) is 0.368, so about 86% (1 - 0.368²) of the step's squared norm lies outside the count direction. By elimination, the information beyond p3 has to be in that shared-representation component.
- Transfer evidence at cap 76: 0.960 [0.944, 0.972] of far items (rank ≥ 3·cap) move down, vs 0.523 for sham. Coherence is 1.81 vs 0.30, and Kendall τ on far items is 0.901 [0.880, 0.918] vs 0.998. Spearman(dL, -w) on far items is only 0.22 [0.19, 0.25]. The step acts through the representation, not locally at the boundary.

**3.7 The local gain does not persist.**
- At cap 50, +0.62 [+0.32, +0.91] correct slots are gained per step, and the next CE epoch changes that by -0.72 [-1.32, -0.12] (n = 96 step-epoch pairs).
- Cap 50, target - sham: +1.125 [+0.59, +1.66] after the epoch-6 step (16 better / 1 worse / 7 tied); -0.21 [-1.09, +0.67] one epoch later; -0.58 [-1.91, +0.75] at the final checkpoint.
- The endpoint therefore carries at most about one step's worth. That is why target - sham is null at both caps at the endpoint (n = 24).
- The observed endpoint scatter implies a between-arm SD of about 3.1 slots. Detecting a 0.6-slot endpoint effect would need roughly 200 seeds.

**3.8 The prize is small.** Wrong occupants among tralo_null slots (epochs 6-10):
- Cap 76: 16.5 [15.7, 17.3]. By grade: 2 = 8.1, 1 = 4.0, 4 = 2.7, 0 = 1.7. Median p3: wrong 0.868 vs correct 0.984.
- Cap 50: 6.4 [5.9, 6.9]. By grade: 2 = 2.7, 1 = 1.9, 4 = 1.3, 0 = 0.4. Median p3: wrong 0.972 vs correct 0.995.

Slots recoverable by swaps within reach of the cut:

| | within 10 ranks | within 20 ranks |
|---|---|---|
| cap 76 | 3.8 [3.6, 4.1] slots (21.6 inversion pairs) | 7.3 [7.0, 7.6] slots (73.2 pairs) |
| cap 50 | 2.5 slots (17.1 pairs) | 4.0 slots (51.2 pairs) |

A third of the errors are grades 0-1, which are not neighbours of grade 3. Wrong occupants sit at p3 ≈ 0.87-0.97, far above the p3 = 0.5 where the count's weight peaks.

## 4. Synthetic map: when a constraint does carry who-information

Synthetic lab (lab_losslab lab.py), 24 seeds per scenario. Each arm is paired against the clipper and against a sham random direction of equal norm. WHO-AUC is the AUC of the margin drop for "not class 3" inside ranks cap ± 60; 0.5 means no information.

| Scenario | WHO-AUC (target) | Precision, target - clipper | Reading |
|---|---|---|---|
| S1, well-specified | 0.418 [0.378, 0.458] | -0.0056 [-0.0119, +0.0006] | anti-informative |
| S2, memorised (train acc 1.000) | 0.393 [0.356, 0.430] | -0.0128 [-0.0192, -0.0063] | anti-informative |
| S2_noise | 0.363 [0.321, 0.406] | -0.0109 [-0.0168, -0.0050] | anti-informative |
| S3a, prior shift on class 3 only | 0.410 [0.365, 0.454] | -0.0034 [-0.0053, -0.0015] | top-K invariant; nothing helps (EM +0.0008 [-0.0010, +0.0025]) |
| **S5a, dense wrong cluster at the cut** | **0.648 [0.587, 0.709]** | **+0.0180 [+0.0049, +0.0310]** (17/5/2) | **informative**; vs sham +0.0159 [+0.0034, +0.0284] |
| S5b_cap, same geometry, cluster truly class 3 | 0.271 [0.214, 0.328] | -0.0261 [-0.0342, -0.0180] (1/23/0) | same push, wrong way |

- **The information follows density, not truth.** The push correlates +0.336 with a label-free density measure in both S5a and S5b. Density correlates -0.83 with the Bayes posterior in S5a and +0.82 in S5b.
- **Most of it is the class-level bias channel**, which acts like a prior shift. A bias-only step gives S5a +0.0139 [+0.0086, +0.0191] and S5b -0.0078 [-0.0105, -0.0050].
- **The damage grows with step size.** S1 dose-response by cap / true count: 0.9 gives -0.0043 [-0.0078, -0.0008]; 0.7 gives -0.0056; 0.5 gives -0.0184 [-0.0284, -0.0084]; 0.3 gives -0.0342 [-0.0534, -0.0150]. The anti-information is already present in a head-only step (WHO-AUC 0.421 [0.376, 0.466]). Sham - clipper stays within ±0.003 everywhere.
- **Uneven label shift (S3b; pool priors [.02, .02, .50, .20, .26], train uniform).** Top-K is no longer invariant here.

  | S3b arm | minus clipper | minus EM |
  |---|---|---|
  | Bayes | +0.1005 [+0.070, +0.131] | |
  | Post-hoc SLD EM | +0.0135 [+0.0061, +0.0208] | |
  | Targeted step | +0.0001 [-0.0084, +0.0085] | |
  | Joint penalty w1 | +0.0177 [+0.0060, +0.0294] | +0.0042 [-0.0064, +0.0148] |
  | Joint penalty w10 | +0.0299 [+0.0106, +0.0493] | +0.0165 [+0.0007, +0.0322] |

- **Joint training persists, but mostly as regularisation.** The same penalty applied to unlabelled draws from the training distribution (jtrain) matches the pool penalty in S1, S2, S2_noise and S3a. Transductive gain (joint - jtrain) with CI above 0 appears only in: S3b (w10 +0.023 [+0.011, +0.035]), S5a (w10 +0.042 [+0.009, +0.075]), S5b_cap (w0.1 +0.017 [+0.006, +0.028]) and S2 (w0.1 +0.013 [+0.004, +0.023]). It is negative in S3a w1 (-0.009 [-0.016, -0.003]) and S2_noise w1 (-0.019 [-0.035, -0.003]). The controller overshoots badly: in S1 the final hard count is 66 [58, 75] (w1) and 43 [35, 52] (w10) against cap 112.
- **Group caps (S4).** The post-hoc local allocator dominates: local - global = +0.091 [+0.075, +0.107] (24/0) with covariate shift and +0.019 [+0.012, +0.027] without. A training-time local constraint loses to it: -0.117 [-0.146, -0.088] (0/24) with shift. The targeted local step loses -0.053 [-0.066, -0.041] (0/24), and per-group sequential steps lose -0.051 [-0.064, -0.039].

**Map.** A label-free count carries who-information only when the class structure of the confusion at the cut makes the dense uncertain mass there disproportionately wrong (S5a; the knee per step, WHO-AUC 0.55-0.60), or when non-capped classes shift unevenly (S3b; joint penalty only, EM recovers part). It is anti-informative in well-specified and memorised cells (WHO-AUC 0.36-0.42), and when the dense mass is correct (S5b). The effect is never better than the density-truth correlation of the particular cell, and it cannot be diagnosed without labels.

## 5. Redesign implications, ranked

1. **Stop tuning dose, schedule, controller or targeting within the per-item count family.**
   - *Grounds:* §2.3-2.4 and §3.1-3.5. The direction is demote-only and ordered by p3. c_i carries nothing given p3 (residual AUC 0.46-0.49). The flow admits 0 items and keeps 99% of the top-cap set.
   - *Consequence:* at best this family ties `capped_first`, matching target - sham being null at both caps (n = 24).

2. **Measure the one channel that works as a within-trajectory contrast, not an endpoint.**
   - *Grounds:* §3.6-3.7. Parameter coupling gives +0.62 [0.35, 0.86] slots per step at cap 50, and the next CE epoch erases it (-0.72 [-1.32, -0.12]).
   - *Options:* (i) score the step's before/after contrast at fixed weights, paired within a trajectory; this has power at n = 24, whereas an endpoint test needs about 200 seeds; (ii) put the step after all CE; (iii) keep the penalty in the objective (joint) so CE does not undo it.
   - *Expected size:* bounded by one step, about 0.6 slots at cap 50 and not significant at cap 76.

3. **Before building anything, run a label-free gate on finished runs.**
   - *Grounds:* §2.5 and §4. The gain exists only where a label-free pool property predicts near-cut false positives beyond p3.
   - *Test:* partial AUC controlling for p3, n = 24 seeds, dev rows only, offline. Candidate properties: augmentation stability of p3, kNN agreement in backbone features, density of the uncertain mass. S5a vs S5b shows density alone can point either way.
   - *Decision:* if no property clears the gate, the thesis should state that the post-hoc cut is optimal given the score.

4. **If a training-time term is kept, aim it at the capped boundary and give it a promote side.**
   - *Grounds:* §3.2 (the weight peaks at rank ≈ 89 / p3 = 0.5, while the errors sit at p3 0.87-0.97) and §3.5 (the promote part of the cosine is -0.31).
   - *Design:* a pairwise rank or margin loss between ranks cap - k and cap + k, with k ≈ 10-20. Its supervision must come from a source outside the count (point 3). Without one it is a sharper version of the same tilt.
   - *Ceiling:* 3.8 / 7.3 slots (cap 76) and 2.5 / 4.0 (cap 50) within 10 / 20 ranks.

5. **For any prior-shift claim, beat post-hoc SLD EM, not the clipper.**
   - *Grounds:* §4, S3b. EM recovers +0.0135; the joint penalty clears EM only at w10 (+0.0165 [+0.0007, +0.0322]).
   - *Controls:* control non-transductive regularisation with a jtrain arm, and use a controller that stops at the cap (the current one ends at 40-60% of it).

6. **For group caps, keep the post-hoc local allocator.**
   - *Grounds:* §4, S4. Every training-time local variant lost to it: -0.117, -0.053, -0.051, all with CIs excluding 0.

7. **Loss hygiene, independent of the direction question** (edge.py):
   - Trigger on the hard count, or sharpen the count. In the D2 region (soft < cap < hard) the gradient is exactly 0 while λ and ρ both climb from 1 to 26 over 50 controller steps without freezing.
   - The bounded penalty's coefficient decays as 1/(1+e)²: 0.25 at e = 1, 0.0083 at e = 10, 4.3e-5 at cap 0. Large violations are therefore nearly gradient-free.
   - Items with p3 > 0.999 carry under 0.1% of the direction.
   - Ties at the cut make the bisection undershoot (by 2 with 6 tied rows).
   - Improving the score itself (calibration, ensembling) helps the post-hoc cut equally, so it cannot be credited to the loss.

## 6. What remains untested

1. **Which channel makes the knee step informative per step.** Candidates: the bias/competitor channel (grade 2 vs 3, like S5a's bias-only step) or kernel coupling. This needs bias-only and head-only targeted steps on real checkpoints, which needs a GPU or saved weights. Not run under the CPU-only rule.
2. **A coupled-only step.** Would a step that removes the per-item logit component and keeps only the coupled part do better than the full step? Needs compute.
3. **The label-free partial-AUC gate (§5 point 3).** Not yet run on the finished knee runs.
4. **Persistence under a fixed schedule.** Does a step-last schedule, or a joint penalty with a controller that lands exactly at the cap, turn the per-step gain into an endpoint gain? The joint arms' overshoot confounds the synthetic answer.
5. **Mechanism of the anti-information in well-specified cells.** After controlling for margin in S1, the push correlates +0.22 with distance to the class-3 mean, -0.27 with uncertain-mass density and +0.24 with the Bayes posterior. It is present in a head-only step. A feature-norm or NTK effect is suspected but not identified.
6. **Erasure rate on the knee vs synthetic.** The knee erases in about one minibatch epoch; the synthetic lab uses full-batch Adam over a 20-step block. The two rates have not been matched.
7. **Stability and definitions.**
   - The oracle direction is one definition (unit rows, m lowest-p3 wrong-inside paired with m highest-p3 correct-outside). Other weightings could change its magnitude; the sign of the promote part is fixed by §2.4.
   - The S1 effect moved between runs (-0.0097 vs -0.0056) when the data draw changed. It is not stable to ±0.005 at n = 24.
   - Cap-50 per-epoch AUC CIs are about ±0.05 (6.4 wrong occupants per snapshot).
   - The two labs' per-step slot numbers at cap 76 differ in step count (96 vs 120 steps) and point estimate (+0.11 vs +0.10). Both CIs include 0.
8. **Row alignment.** The analyses assume the 826 stored rows follow the manifest's val-row order. Every `capped_first` set check passed under that assumption, but this does not prove the label alignment.
9. **Citation gaps.** Scott & Zhang's (2020) exact identifiability conditions and the Lee (2013) record were not verified. The claims attributed to Agarwal et al. 2018, Celis et al. 2019, Lipton et al. 2018, Singh et al. 2008 and Wei et al. 2021 come from memory against verified records.
10. **Woodworth et al.'s restricted-class exception.** It could apply if the dev scores at the constraint epoch are far from Bayes quality. Train accuracy above 99% shows the training set is memorised, not that the dev scores are calibrated.
