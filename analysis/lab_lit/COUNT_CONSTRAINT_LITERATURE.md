# Why a count constraint on the deployment pool is post-hoc thresholding, and what a training-time method must add to beat it

Lab: `analysis/lab_lit/` (literature role). Date 2026-09-26.
Checks: `order_check.py`, `order_check2.py` in this folder, run on the local CPU with synthetic pools only (N=826, 5 classes, cap K=76). No run data, no dev labels, no test split, and no GPU were used.

Verification key: **[V]** means the bibliographic record (title, authors, venue, year) was confirmed through Semantic Scholar or the publisher page, and the claim used here matches the abstract or proceedings summary I retrieved. **[V-bib]** means the record is confirmed but the specific claim comes from my reading of the paper and was not re-checked against the text in this session. **[UNVERIFIED]** means neither was confirmed.

---

## 1. The argument in six steps

**Step 1. Under a cardinality constraint, the optimal selector thresholds the score.**
Suppose the objective is to pick exactly K items from a pool so that the expected number of true members of class c is as large as possible. The solution is to take the K items with the largest η_c(x) = P(y=c | x). This is the rearrangement argument, and it is the discrete form of the Neyman–Pearson lemma: a test with a fixed selection rate is most powerful when it thresholds the likelihood ratio. The same result holds for a much larger family of metrics:
- Koyejo et al. (2014) **[V]**: for every linear-fractional metric (F-measure, Jaccard, weighted accuracy and so on), the Bayes-optimal classifier is sign(η(x) − δ*), where the threshold δ* depends on the metric. A plug-in estimate with a tuned threshold is consistent.
- Narasimhan, Vaish & Agarwal (2014) **[V-bib]**: plug-in classifiers, meaning a class-probability estimate plus an empirically tuned threshold, are statistically consistent for non-decomposable measures.
- Yan et al. (2018) **[V]**: for any metric with the "karmic" and "threshold-quasi-concave" properties, the Bayes-optimal classifier is a threshold on η(x).
- Precision at a quantile or at K: Boyd et al. (2012) **[V]** define accuracy at the top τ-quantile, Clémençon & Vayatis (2007) **[V-bib]** treat ranking of the best instances, and Kar, Narasimhan & Jain (2015) **[V]** give surrogates for precision at the top. In every case the target is the top of an η-ordering.

*Consequence for this project:* `capped_first` (top-cap items by p_3 get grade 3) is exactly the plug-in form of the Bayes-optimal rule. It can lose only because its score p_3 is a poor estimate of η_3, and never because of how the cut is placed.

**Step 2. Adding a group constraint still leaves a threshold rule, and post-processing reaches it when the scores are good.**
- Hardt, Price & Srebro (2016) **[V]**: equalized odds and equal opportunity can be reached by post-processing any score. With Bayes-optimal scores, the post-processed rule is optimal among rules that satisfy the constraint.
- Corbett-Davies et al. (2017) **[V-bib]**: under demographic parity and related constraints, the optimal rules are group-specific thresholds on the risk score.
- Menon & Williamson (2018) **[V]**: for cost-sensitive fairness measures, the optimal classifier is an instance-dependent thresholding of the class-probability function. They propose thresholding class-probability estimates as the practical method.
- Xian, Yin & Zhao (2023, ICML) **[V-bib]**: under demographic parity, post-processing the Bayes score is optimal.
- Chzhen et al. (2019) **[V-bib]**: consistent fair plug-in classification uses **unlabeled data only to set the thresholds**. This is exactly the role a deployment-pool count plays.
- Cruz & Hardt (2024, ICLR) **[V]**: across thousands of model evaluations, the fairness–accuracy Pareto frontier of post-processing contained every in-processing method they could evaluate. Earlier "wins" came from comparing different base models or different levels of constraint relaxation.

**Step 3. The count penalty's gradient depends on the pool only through one scalar, and its closed form is a score transform.**
The penalty depends on the pool only through S_c = Σ_i p_ic. The gradient of that penalty with respect to item i's logits is κ · p_ic (e_c − p_i), where κ > 0 is a single scalar shared by all items (`tralo/streamed_constraint.py:count_logit_gradient`, line 37). Two cases:

- **Exact projection.** The KL projection of the model's predictions onto the set {q : Σ_i q_ic ≤ K} is the posterior-regularization / expectation-regularization E-step (Ganchev et al. 2010 **[V-bib]**; Mann & McCallum 2007 **[V]**). Its solution is an exponential tilt: q_ic ∝ p_ic e^(−λ). The map p_ic → p_ic e^(−λ) / (1 − p_ic + p_ic e^(−λ)) is strictly increasing, so the ordering by class c is unchanged. This is the same fact as prior-shift correction (Saerens et al. 2002 **[V]**; Lipton et al. 2018 **[V-bib]**) and as logit adjustment (Menon et al. 2021 **[V-bib]**). *Check:* over 200 random pools with λ ~ U(0,10), the top-76 set changed in **0 of 200** pools (`order_check.py`).
- **One gradient step with free logits.** To first order the change is dp_ic = −η κ p_ic² [(1−p_ic)² + Σ_{k≠c} p_ik²]. The closed form matches autograd-free finite differences to a maximum absolute error of 7.4e−7 (`order_check2.py`). This is a fixed function of item i's own probability vector. Its Spearman correlation with (p_c(1−p_c))² is 0.9987 (one pool, N=826), and with p_c alone it is 0.850. The step can therefore reorder items, but only according to how the remaining probability mass is spread over rival classes. The information it uses is information the score already contains. A post-hoc rule on s(p) = p_c − η κ p_c²[(1−p_c)² + Σ p_k²] reproduces the step exactly, without training.
  *Magnitude on synthetic pools (n=20 pools per η, K=76):* the number of items that leave the top 76 is 0.15 ± 0.36 at η=0.3, 0.6 ± 0.58 at η=1, 1.6 ± 1.0 at η=3 and 1.7 ± 0.95 at η=10.

**Step 4. Any change beyond a score transform must come through parameter sharing, and so from features.**
With a network, the update to item j is Δz_j ≈ −η Σ_i Θ(x_j, x_i) g_i, where Θ is the tangent kernel. Two items j and k change relative order only through Σ_i [Θ(x_j,x_i) − Θ(x_k,x_i)] κ p_ic(e_c − p_i). In words: an item is pushed down more when it resembles the pool items with high class-c mass. That helps precision at the cut only if feature similarity to "high p_c" items predicts being a false positive better than p_c already does. This is the cluster, smoothness or expansion assumption of semi-supervised learning, and no count supplies it. *Check:* with features unrelated to the labels (row-normalized Gaussian kernel), the kernel-routed step moved 0.0, 0.2 ± 0.40, 1.2 ± 0.75 and 1.25 ± 0.54 items at η = 0.3, 1, 3 and 10 (n=20). That is no more than the free step. Coupling alone adds no information.

**Step 5. One global proportion does not identify which items are which.**
- Lu, Niu, Menon & Sugiyama (2019, ICLR) **[V]**: from a single set of unlabeled data, the risk of an arbitrary binary classifier cannot be estimated without bias. **Two sets with different class priors** make it possible, and that is the minimal supervision required.
- The literature on learning from label proportions (LLP) needs many bags whose proportions differ: Kück & de Freitas (2005) **[V-bib]**; Quadrianto et al. (2008 ICML / 2009 JMLR) **[V]**; Patrini et al. (2014) **[V-bib]**; Yu et al. (2014) **[V-bib]**; Scott & Zhang (2020) **[V]**, which reduces LLP to mutual contamination models; Zhang, Wang & Scott (2022) **[V-bib]**. The exact identifiability conditions in Scott & Zhang (pairs of bags with distinct proportions) are **[UNVERIFIED]** in this session.
- Elementary form of the argument: a cap K on a pool of N items is one linear moment. Every one of the C(N,K) labelings with K positives satisfies it equally well, so the count carries no information about who belongs above the cut. Only the score, the features or other labels can break the tie.

*Consequence:* one pool-level count is the degenerate one-bag case of LLP. By Lu et al., it can shift the **level** (how many items are called class c) but cannot identify the **ranking** (which items). This matches the project's settled finding that "the constraint knows how many, not who" (LEDGER #9), and the finding that the targeted step evicts 83–87% of the same items as the post-hoc cut (n=24 per cap).

**Step 6. When in-processing *can* beat post-processing, and which clause applies here.**
The results that go the other way all need one of three conditions:
- (a) The hypothesis class is restricted, so the Bayes score is not reachable and post-processing a sub-optimal score is itself sub-optimal. Woodworth et al. (2017) **[V]** show that post-hoc correction "can be highly suboptimal" in this setting, and that the optimal constrained learner is computationally intractable. Agarwal et al. (2018) **[V-bib]** give guarantees relative to the best randomized classifier within the class. Celis et al. (2019) **[V-bib]** is similar.
- (b) The constraint's attribute is unavailable at prediction time. Lipton, McAuley & Chouldechova (2018) **[V-bib]** show that disparate learning processes then reorder items *within* groups through correlated features, and not always usefully.
- (c) Representation learning with conflicting base rates. Zhao & Gordon (2019) **[V]** prove a lower bound on joint error.

None of these applies cleanly here. The count is on the same class-c score that the cut thresholds, and the network already fits its training set (train accuracy > 99%). The one open route is (a): if a training-time signal changes the *representation*, the post-hoc score of the retrained model is a different score. Step 4 shows that such a change can help only if the features carry the information.

---

## 2. What a training-time method must add to beat the post-hoc cut

The candidates below are ordered by how directly they add information about who belongs above the cut.

1. **New labels, including labels for pool items.** This breaks Step 5 directly. It is not available under the protocol.
2. **Several counts on distinguishable sub-pools with different proportions.** This is the LLP or two-unlabeled-sets route (Lu et al. 2019; Scott & Zhang 2020). It works only if the bag proportions really differ and the bags are defined by something correlated with the error. Project memory notes that group-count granularity was refuted earlier, so this needs bags with genuinely distinct priors, not random splits.
3. **A structural assumption on the pool, expressed as a loss.**
   - Consistency regularization or pseudo-labeling with a confidence threshold: FixMatch (Sohn et al. 2020) **[V]**, Pseudo-Label (Lee 2013) **[UNVERIFIED record]**, entropy minimization (Grandvalet & Bengio 2004/2005) **[V]**.
   - A distribution-alignment prior, as in ReMixMatch (Berthelot et al. 2020) **[V]**. This is the deep version of Joachims's (1999) **[V]** transductive SVM with a fixed positive fraction, and of the class-prior step in Zhu, Ghahramani & Lafferty (2003) **[V]**.

   Theory says these methods help only under an assumption: a cluster or low-density-separation assumption (Singh, Nowak & Zhu 2008 **[V-bib]**), or input-consistency and expansion (Wei et al. 2021 **[V-bib]**). Without such an assumption, unlabeled data gives no worst-case gain (Ben-David, Lu & Pál 2008 **[V-bib]**), and pseudo-labeling can lock in its own mistakes (Arazo et al. 2020 **[V-bib]**). The count or prior term in these methods only prevents collapse. The gain in ranking comes from the augmentation or neighborhood invariance.
   *Testable prediction:* a training-time method beats `capped_first` at equal K only if its gain is predicted by a measurable pool property, such as agreement of the k nearest neighbors or stability of p_3 under augmentation among near-cut items. If that property does not predict which near-cut items are errors on finished runs (dev rows, offline analysis only), no structural loss can help.
4. **A better score estimate, for example calibration or an ensemble.** This improves the post-hoc cut as well, so it does not favor training-time methods. Under Steps 1–2 it is also the only thing that improves the cut.

**Summary for the committee:** a single count on the deployment pool is one moment and cannot identify rankings (Lu et al. 2019). Its exact projection is a prior tilt that preserves the ordering (Saerens et al. 2002; Ganchev et al. 2010). Its gradient step is a known transform of each item's own probabilities, which a post-hoc rule reproduces (Step 3). With Bayes-quality scores, the cardinality-optimal rule is a threshold on η (Koyejo et al. 2014; Yan et al. 2018), and post-processing reaches the constrained optimum (Hardt et al. 2016; Menon & Williamson 2018; Cruz & Hardt 2024). A training-time method can do better only by adding information the score lacks: labels, multiple bags with distinct priors, or a structural (cluster/expansion) assumption that holds on the pool (Singh et al. 2008; Wei et al. 2021). Otherwise it is expected to tie the post-hoc cut. The project observes exactly that: target − sham is null at caps 76 and 50 (n=24 each), and the targeted step evicts 83–87% of the same items.

---

## 3. References

- Agarwal, A., Beygelzimer, A., Dudík, M., Langford, J., Wallach, H. (2018). A Reductions Approach to Fair Classification. *ICML*. [V-bib]
- Arazo, E., Ortego, D., Albert, P., O'Connor, N., McGuinness, K. (2020). Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning. *IJCNN*. [V-bib]
- Ben-David, S., Lu, T., Pál, D. (2008). Does Unlabeled Data Provably Help? Worst-case Analysis of the Sample Complexity of Semi-Supervised Learning. *COLT*. [V-bib]
- Berthelot, D., Carlini, N., Cubuk, E. D., Kurakin, A., Sohn, K., Zhang, H., Raffel, C. (2020). ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring. *ICLR* (arXiv 2019). [V]
- Boyd, S., Cortes, C., Mohri, M., Radovanovic, A. (2012). Accuracy at the Top. *NeurIPS 25*, 953–961. [V]
- Celis, L. E., Huang, L., Keswani, V., Vishnoi, N. K. (2019). Classification with Fairness Constraints: A Meta-Algorithm with Provable Guarantees. *FAT\**. [V-bib]
- Chzhen, E., Denis, C., Hebiri, M., Oneto, L., Pontil, M. (2019). Leveraging Labeled and Unlabeled Data for Consistent Fair Binary Classification. *NeurIPS*. [V-bib]
- Clémençon, S., Vayatis, N. (2007). Ranking the Best Instances. *JMLR* 8. [V-bib]
- Corbett-Davies, S., Pierson, E., Feller, A., Goel, S., Huq, A. (2017). Algorithmic Decision Making and the Cost of Fairness. *KDD*. [V-bib]
- Cruz, A. F., Hardt, M. (2024). Unprocessing Seven Years of Algorithmic Fairness. *ICLR*. arXiv:2306.07261. [V]
- Ganchev, K., Graça, J., Gillenwater, J., Taskar, B. (2010). Posterior Regularization for Structured Latent Variable Models. *JMLR* 11. [V-bib]
- Grandvalet, Y., Bengio, Y. (2005). Semi-supervised Learning by Entropy Minimization. *NeurIPS 17*. [V]
- Hardt, M., Price, E., Srebro, N. (2016). Equality of Opportunity in Supervised Learning. *NeurIPS*. [V]
- Joachims, T. (1999). Transductive Inference for Text Classification using Support Vector Machines. *ICML*. [V]
- Kar, P., Narasimhan, H., Jain, P. (2015). Surrogate Functions for Maximizing Precision at the Top. *ICML*. [V]
- Koyejo, O., Natarajan, N., Ravikumar, P., Dhillon, I. (2014). Consistent Binary Classification with Generalized Performance Metrics. *NeurIPS 27*, 2744–2752. [V]
- Kück, H., de Freitas, N. (2005). Learning about Individuals from Group Statistics. *UAI*. [V-bib]
- Lee, D.-H. (2013). Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. *ICML Workshop on Challenges in Representation Learning*. [UNVERIFIED]
- Lipton, Z. C., McAuley, J., Chouldechova, A. (2018). Does Mitigating ML's Impact Disparity Require Treatment Disparity? *NeurIPS*. [V-bib]
- Lipton, Z. C., Wang, Y.-X., Smola, A. (2018). Detecting and Correcting for Label Shift with Black Box Predictors. *ICML*. [V-bib]
- Lu, N., Niu, G., Menon, A. K., Sugiyama, M. (2019). On the Minimal Supervision for Training Any Binary Classifier from Only Unlabeled Data. *ICLR*. arXiv:1808.10585. [V]
- Mann, G. S., McCallum, A. (2007). Simple, Robust, Scalable Semi-supervised Learning via Expectation Regularization. *ICML*. [V]
- Menon, A. K., Williamson, R. C. (2018). The Cost of Fairness in Binary Classification. *FAT\**, PMLR 81:107–118. [V]
- Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., Kumar, S. (2021). Long-tail Learning via Logit Adjustment. *ICLR*. [V-bib]
- Narasimhan, H., Vaish, R., Agarwal, S. (2014). On the Statistical Consistency of Plug-in Classifiers for Non-decomposable Performance Measures. *NeurIPS*. [V-bib]
- Narasimhan, H., Kar, P., Jain, P. (2015). Optimizing Non-decomposable Performance Measures: A Tale of Two Classes. *ICML*. [V-bib]
- Neyman, J., Pearson, E. S. (1933). On the Problem of the Most Efficient Tests of Statistical Hypotheses. *Phil. Trans. R. Soc. A* 231. [UNVERIFIED this session; classical]
- Patrini, G., Nock, R., Caetano, T., Rivera, P. (2014). (Almost) No Label No Cry. *NeurIPS*. [V-bib]
- Quadrianto, N., Smola, A. J., Caetano, T. S., Le, Q. V. (2008). Estimating Labels from Label Proportions. *ICML* (extended version *JMLR* 10, 2009). [V]
- Saerens, M., Latinne, P., Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. *Neural Computation* 14(1). [V]
- Scott, C., Zhang, J. (2020). Learning from Label Proportions: A Mutual Contamination Framework. *NeurIPS 33*. arXiv:2006.07330. [V]
- Singh, A., Nowak, R., Zhu, X. (2008). Unlabeled Data: Now It Helps, Now It Doesn't. *NeurIPS*. [V-bib]
- Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., Raffel, C. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. *NeurIPS*. [V]
- Wei, C., Shen, K., Chen, Y., Ma, T. (2021). Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data. *ICLR*. [V-bib]
- Woodworth, B., Gunasekar, S., Ohannessian, M. I., Srebro, N. (2017). Learning Non-Discriminatory Predictors. *COLT*, PMLR 65:1920–1953. [V]
- Xian, R., Yin, L., Zhao, H. (2023). Fair and Optimal Classification via Post-Processing. *ICML*. [V-bib]
- Yan, B., Koyejo, O., Zhong, K., Ravikumar, P. (2018). Binary Classification with Karmic, Threshold-Quasi-Concave Metrics. *ICML*, PMLR 80. [V]
- Yu, F. X., Choromanski, K., Kumar, S., Jebara, T., Chang, S.-F. (2014). On Learning from Label Proportions. arXiv:1402.5902. [V-bib]
- Zhang, J., Wang, Y., Scott, C. (2022). Learning from Label Proportions by Learning with Label Noise. *NeurIPS*. [V-bib]
- Zhao, H., Gordon, G. J. (2019). Inherent Tradeoffs in Learning Fair Representations. *NeurIPS 32*. [V]
- Zhu, X., Ghahramani, Z., Lafferty, J. (2003). Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions. *ICML*, 912–919. [V; the name "class mass normalization" is V-bib]
