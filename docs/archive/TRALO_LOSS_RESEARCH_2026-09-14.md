# TraLO loss research

## Recommendation

The next useful experiment is a **mechanism audit of the constraint update at a fixed model state**, followed, only if that audit supports it, by a task-protected constraint step. Retain TraLO's global and local count constraints, bounded penalty, supervised training, and deployment allocator. Test whether an update that actually reduces the constraint objective while limiting damage to a fixed training anchor improves the deployed constrained-class F1. Do not start another penalty-shape or multiplier-size sweep.

This recommendation is provisional. No fresh real-dataset classification gain has been demonstrated. The new evidence below consists of CPU derivative checks, an optimizer counterexample, and two epochs through the actual TraLO trainer on synthetic inputs. They validate specific implementation statements, not a hypothesis about iwildcam, fMoW, or BCN. The research is scoped to the source hashes in the [probe receipt](receipts/20260914T073749818489Z/probe.json), with Git HEAD `d17306b30b4add3f2def94214dd1a9baaa857b9d` and concurrent uncommitted cleanup.

The companion [research ledger](RESEARCH_LEDGER.md) records considered mechanisms, historical overlap, evidence status, and promotion conditions. The current [FRAMEWORK](../FRAMEWORK.md), [MISSION](../MISSION.md), and [REJECTED](../REJECTED.md) remain authoritative. This report does not reinstate historical prohibitions or change the scientific question.

## What the engine computes

Let `j` index a constrained class and scope, with scope membership `I_j`, integer cap `K_j`, model probabilities `p_ic`, and soft count

\[
 S_j(\theta)=\sum_{i\in I_j}p_{ic}(\theta),\quad
 E_j=[S_j-K_j]_+,\quad a_j=\max(K_j,1).
\]

Ignoring numerical epsilon for readability, the default constraint objective is

\[
 C(\theta;\lambda,\rho)=\sum_j\lambda_j
 \left[\frac{E_j}{E_j+a_j}
       +\rho\frac{(E_j/a_j)^2}{1+(E_j/a_j)^2}\right].
\]

`MulticlassTransductiveLoss._penalty` implements this expression. The actual epsilon-aware derivative is implemented independently in the probe. For `S_j>K_j`, the idealized derivative is

\[
 b_j=\lambda_j\left[\frac{a_j}{(E_j+a_j)^2}
 +\frac{2\rho E_j/a_j^2}{(1+(E_j/a_j)^2)^2}\right].
\]

It is zero below the cap; PyTorch's ReLU convention also returns zero at the exact kink. The derivative can be largest at intermediate violations and small for severe violations. That is a property of the chosen bounded objective, not evidence of a coding mistake. The archived experiments already investigated this shape and its relative scope weighting.

For a particular example, define the effective class pressure

\[
 v_{ic}=\sum_{j:\,i\in I_j,\,c_j=c}b_j.
\]

The full multiclass logit gradient is

\[
 \frac{\partial C}{\partial z_{ik}}
 =p_{ik}\left(v_{ik}-\sum_c p_{ic}v_{ic}\right).
\]

The familiar `p(1-p)` is only a diagonal term, not the whole multiclass gradient. Pressure on one capped class changes every competing logit. If all class pressures on an example are equal, the logit gradient is exactly zero. More generally, competing scopes can cancel or redirect one another. A large sum of scalar penalties therefore does not establish a large or useful parameter gradient.

Through shared model parameters, `g_C = sum_i J_i^T grad_z C`. A gradient step changes every prediction through products of Jacobians, so aggregate-count training **can reorder examples**. It can also reorder them harmfully. An argument about uniform output shifts is not a proof about shared-backbone updates.

The trainer performs a supervised CE epoch, then a count forward pass, then a chunked constraint backward pass, followed by a separate update through the **same Adam optimizer used for CE**. It is not one simultaneous backward pass on `CE+C`. The current default gradient normalization acts on the constraint gradient alone. Positive overall rescaling mostly cancels before the optimizer, subject to clipping epsilon and numerical effects; relative scope changes can still change direction. Adam's history then transforms that direction and displacement.

Multipliers increase on hard-count violations while the ratchet gate is open. The penalty usually reads soft counts. Rho ramps until first hard satisfaction, after which it freezes. Thus this algorithm is not conventional signed-residual Lagrangian ascent, and `lambda` is not automatically an identified optimal shadow price. With `straight_through`, the forward penalty count is replaced by the hard count while the backward path remains a surrogate. Its logged meaning must be explicit.

## Fresh executable findings

The [diagnostic script](loss_mechanism_probe.py) imports the actual loss, step function, and trainer. It preserves timestamped logs and hashes the participating sources before and after execution. The successful receipt reports unchanged source during its execution. The initial diagnostic attempt failed because the diagnostic constructed a loss with an empty constraint list rather than `None`; that diagnostic-only mistake was corrected. Its partial synthetic logs remain preserved in the receipts directory and are not acceptance evidence.

| Check | New measurement | What it establishes |
|---|---|---|
| Penalty derivative | Analytic, autograd and finite differences agree; maximum finite-difference error below `1e-8` on five fixtures including `K=0` and an inactive cap | Formula implementation is consistent on these fixtures |
| Severe violation | At `K=10`, `rho=3.5`, slope is `0.266139` at `S=16`, versus `0.00256001` at `S=90` | Bounded penalty downweights deep violations; this is already a known mechanism |
| Default count chunking | Full versus two chunks: maximum gradient difference `0` | Default `sum` is chunk-consistent on the fixture |
| Optional uniform count | Full versus two chunks: maximum gradient difference `0.112876` | `uniform_grad_count` is chunk-dependent because its mean weight is recomputed per call |
| Raw projection and Adam | Task/constraint raw gradient dot product `0`, but actual displacement has task directional change `+0.00909864` | Raw orthogonality is not delivered task neutrality |
| Adam attribution control | Zero-current-gradient Adam has the same `+0.00909864` task drift in the constructed state | In this example the drift is optimizer history, not current constraint information |
| Trainer event reconstruction | Two logged constraint objectives reproduced to absolute error at most `1.5e-9` | Logged pre-step loss and scope inputs agree on this synthetic execution |
| Step count | Two events and two applied constraint steps | Event and trainer summary counters agree on this execution |

The optional uniform-count issue was concrete in the initial source. The function set `w_c = mean_i p_ic(1-p_ic)` while receiving one chunk at a time. Its backward direction consequently depended on chunk membership and order. The cleanup owner has now changed the trainer to compute a full-set detached mean in its existing first pass and pass it to every chunk. The [post-fix diagnostic](receipts/20260914T074403033401Z/probe.json) reports full/chunk gradient difference `0` with that weight; the intentionally incorrect per-chunk-mean control still differs by `0.112876`. Independently run focused tests of the helper and actual trainer passed (`2 passed, 9 deselected`). This is a validated local repair, not a classification improvement or full remote release certification. It does not invalidate the default `sum` path.

The optimizer fixture deliberately constructs old momentum that conflicts with the present task gradient. It proves possibility, not frequency on real models. For the actual synthetic trainer, capped soft count changed from `35.343685` before the first constraint update to `35.344635` afterward. The update was applied, yet that count increased slightly. This observation cannot be converted into a real-data failure rate or a quality conclusion.

The event records correctly label their counts `post_task_pre_constraint`. The second scope callback changes the multiplier fields using cached counts; it does not measure post-constraint predictions. Treating those rows as post-step satisfaction would be an analysis error. Future candidate diagnostics need a fresh forward pass before comparing before/after quantities.

## Historical coverage and what remains open

The archived FRAMEWORK is almost twenty thousand lines and contains mutually inconsistent conclusions. Its named receipts are leads, not automatically valid experiments. Search covered its intervention headings and mechanism terms, the old full rejection file, current source and probe names. This is substantial coverage, not proof that no forgotten remote or ignored artifact exists.

Penalty shapes, rho/lambda schedules, item scaling, finer scopes, focal CE, supervised reweighting, head-only steps, raw-gradient orthogonalization, dedicated constraint optimizers, direct SGD, uniform and margin counts, cut windows, random directions, budget permutations, checkpoint restoration, and snapshot averaging all have prior discussion or implementations. Reintroducing them under new names would not be unexplored research.

Three distinctions matter:

1. **Recorded idea versus tested implementation.** Archived lines 6661–6665 explicitly propose projecting the delivered update, while stating it was unpriced. The recommended direction therefore has a recorded precursor. No receipt for the specific two-condition, training-anchor, finite-step acceptance design below was located in the audited material.
2. **Correct surrogate versus useful classifier.** Archived sections z54–z56 describe scope-weight changes that moved constraint quantities without improving the deployment metric. A new direction must predict changes in admitted and evicted examples, not only loss or feasibility.
3. **Wrong diagnostic geometry versus a universal rejection.** Section z79 documents global-cut diagnostics where the allocator used groups. A later correct group-cut analysis can still give unfavorable setting-specific evidence. The earlier wrong analysis does not close the family, and its correction does not establish a benefit either.

A further historical attribution claim needs care: a zero-gradient Adam step is different from skipping the step. The trainer's zero-constraint arm skips its constraint optimizer update. Consequently, old statements that residual Adam drift must cancel against that twin need direct verification; they do not follow from both arms having CE history.

## What the literature contributes

These sources motivate tests, not predictions of TraLO superiority. Their assumptions and tasks differ from this project's deployment-budget experiment.

**Constraint optimization dynamics.** Sohrabi et al.'s nuPI work studies multiplier updates that damp oscillation and overshoot. It keeps primal optimizer choices fixed and explicitly distinguishes its setting from penalty methods. This suggests measuring signed residual histories and controller responsiveness, but does not establish that replacing TraLO's ratchet improves quality. Normalizing a single active constraint also removes the scalar authority a controller expects. [1]

**Gradient conflict.** PCGrad projects conflicting gradients in multitask learning. It motivates checking compatibility, but projecting the raw vector before shared Adam does not imply a task-safe parameter displacement. The repository already contains raw projection, so generic gradient surgery is not a new proposal. [2]

**Finite-step control.** Constrained Policy Optimization uses local constrained optimization and a trust region in reinforcement learning. The transferable idea is to check the step actually taken with a local model and a finite-step safeguard. Its reinforcement-learning guarantees do not apply to this classifier or its bounded loss. [3]

**Different forward and backward constraints.** Cotter et al. develop a proxy-Lagrangian for nondifferentiable constraints, with assumptions involving optimization oracles and stochastic classifiers. This is useful context for TraLO's hard/soft distinction. It does not confer a convergence guarantee on a hard ratchet paired with a soft penalty and one deterministic checkpoint. [4]

**Posterior projection.** Ganchev et al.'s posterior regularization and Pathak et al.'s constrained CNNs supply established ways to enforce expectation or output constraints through auxiliary distributions. These are relevant alternatives to a saturating scalar penalty, but adopting their machinery must be acknowledged as a substantive method extension. [5,6]

**Allocation during training.** Sinkhorn Label Allocation explicitly supports upper bounds on class proportions and uses transport-based pseudo-label assignments. This literature was already discovered in archived section z61; it is not newly uncovered here. It gives an important comparator and novelty boundary for proposals based on constrained pseudo-labels. [7]

**Recent related work.** Ma et al.'s 2026 LLP-DC combines bag-level proportion fitting with hard instance pseudo-labels obtained by min-cost max-flow. Its bag proportions are label frequencies, unlike arbitrary restrictive deployment caps. The paper supports studying the information passed from aggregate constraints to instances; its reported gains cannot be transferred to TraLO's task. The CVF record is CVPR Findings, not the main CVPR proceedings. [8]

**Information limits.** Yu et al. and Scott–Zhang analyze learning from label proportions under explicit assumptions. Such results do not imply that arbitrary upper caps identify correct individual labels. The shape and diversity of groups and the meaning of aggregate information matter. A count can restrict assignments without identifying which exchange improves classification. [9,10]

## Candidate A: task-protected constraint displacement

### Mechanism and mathematical definition

Retain the current `C(theta;lambda,rho)`. At the post-CE state, compute its gradient `g_C` and a task gradient `g_T` on a fixed, prespecified **training-only** anchor set, in a deterministic model mode. The anchor is an additional use of training labels, not development or test labels. It measures that anchor's loss, not the entire population risk.

Let `d_A` be the actual displacement proposed by the existing normalized-constraint Adam step, including its current state. Define a corrected displacement by a small convex projection:

\[
 \min_d \tfrac12(d-d_A)^T M(d-d_A)
 \quad\text{subject to}\quad
 g_T^Td\le0,\quad g_C^Td\le0,\quad \|d\|_2\le\|d_A\|_2.
\]

Use `M=I` for the first design; a learned metric would add another intervention. The zero displacement is feasible. Without the norm constraint, active linear constraints give `d=d_A-A^T mu`, with nonnegative multipliers satisfying the projection KKT conditions. The norm ball needs a corresponding active-set or scalar-dual treatment; do not assume sequential arbitrary projections solve the joint problem exactly.

This is one-sided: retain a component if it helps both objectives. Existing `project_out` removes the complete task-gradient component, including a helpful component, and uses the last CE minibatch reference. This design uses a fixed same-state anchor and operates after the Adam transformation. It therefore addresses a different, specifically testable contract.

Linear constraints only control first-order predictions. Backtrack along `d` against fresh evaluations with lambda/rho frozen during acceptance. Require anchor loss increase no larger than a declared floating-point tolerance, and sufficient decrease `C(theta+d) <= C(theta) + sigma*g_C^T d` for a fixed `0<sigma<1`, unless the update is classified zero and skipped. Finite-step measurements must use the same examples and model mode. Do not choose tolerance or `sigma` using held-out quality.

The design must define optimizer state: calculate the proposal on a saved state; commit its updated moments and step counter only when accepting a nonzero displacement, and restore them for rejection. The accepted parameter displacement may differ from the raw Adam proposal; that is an explicit algorithm choice. Test restart equivalence, rejected proposals, and repeated corrections. Never overwrite live campaign checkpoints to prototype this behavior.

### Decision-boundary prediction and falsification

For small steps, margin change is `delta(z_ic-z_ik) ~= (J_ic-J_ik)d`. Protecting a training anchor can constrain shared representation drift, but it does not determine the sign of this change for an unlabeled test example. The hypothesis is narrower: avoiding measured task-harming constraint displacements might retain useful rankings while achieving the same caps.

Necessary log signatures are: the intervention fires on real, prespecified checkpoints; accepted steps satisfy the fresh loss checks; removed task-harming displacement is nontrivial; constraint decreases survive the next CE epoch; and actual per-group deployment membership changes show better exchanges on development data. A sign-compatible raw gradient alone is not sufficient.

Reject or pause the candidate if the audit finds almost no harmful proposed steps, nearly all corrections become zero, the effect vanishes against the matched optimizer-history control, the anchor improves while development cc-F1 worsens, or the extra compute makes the comparison unmatched. Good feasibility with unchanged or worse cc-F1 is an unfavorable quality result, not success. A CI crossing zero is inconclusive, not proof of equivalence.

### Controls and cost

Compare against baseline TraLO, an identically instrumented unconstrained-proposal TraLO, the existing raw-projection idea if still under consideration, a no-constraint/no-extra-step twin, and a **counterfactual zero-current-gradient Adam step** used for attribution. That last diagnostic is not a replacement for `tralo_null`: it answers a different question. Final comparisons must still include clip, focal_clip, Fioretto, Hounie and ALM.

The first stage should only replay a bounded set of prespecified model-and-optimizer snapshots, not run a training campaign. Per assessed epoch the candidate adds roughly one anchor forward/backward, an anchor evaluation per trial, a full transductive forward per trial, and parameter/state copies. With `B` backtracking trials and 29 constraint epochs, the added full transductive passes can approach `29*B`, plus anchors and copying. The current one-constraint-step-per-epoch schedule does not make this free.

Measure target-backbone wall time and memory before choosing a budget-matched pilot. Extra reads and supervision must be present in the corresponding controls; useful extra baseline training time is a separate compute control. A fixed 30-epoch schedule and equal runtime need not coincide. Any amendment is a user decision under FRAMEWORK, not an implicit expansion of the existing GPU allowance.

## Candidate B: temporal signed-residual controller

A PI-style experimental controller could use `r_j=(S_j-K_j)/max(K_j,1)` and a projected update such as `lambda_next=[lambda+eta_I*r+eta_P*(r-r_prev)]_+`. This is an illustrative PI-like rule, not a claim to reproduce all details of nuPI. Unlike the hard-only ratchet, it can reduce a multiplier when a scope becomes inactive and respond to the direction of residual movement. The backward penalty could remain unchanged for an isolated controller test. [1]

This is lower priority. Under a single active normalized term it can be nearly inert, and with multiple scopes it primarily changes relative weighting, an already heavily explored mechanism. Promote only if real trajectories show distinct residual histories and harmful oscillation that can be changed at matched displacement. Record lag, signed residuals, before/after multipliers, controller saturation, and per-scope gradient contribution. A static replay with fixed counts cannot test a temporal-controller hypothesis. Improving oscillation without deployed quality would not meet the research objective.

## Candidate C: constrained posterior targets

A more substantial option is `q* = argmin_q sum_i KL(q_i || p_i)` subject to simplex, local and global upper bounds, followed by training against the detached feasible distribution. Under feasible constraints and suitable regularity, its dual yields `q_ic` proportional to `p_ic*exp(-lambda_global,c-lambda_local,g,c)`. For the minimized KL value, the envelope gradient in logits is `p-q*`. This produces explicit per-example targets while respecting the constraints of the projection problem. [5,6]

Projection alone does not invent true-label information. With one capped class in one group, the corresponding probability transformation can preserve that class's ranking. Shared-parameter training or multiple class prices can nevertheless change rankings. Soft feasible expectations also do not imply that argmax labels satisfy integer capacities. The existing deployment allocator remains necessary.

This is close to established posterior-regularization and allocation methods, particularly SLA. It should be framed as an acknowledged extension or comparator, not a cosmetic TraLO arm or guaranteed novelty. [7] Exact label proportions, lower bounds, extra pseudo-label confidence rules and restrictive deployment budgets are different assumptions. No conversion of the current caps to true priors is authorized. Candidate C is not selected for immediate implementation.

## Data and evaluation conditions

The cleanup audit reports two identical resized BCN images crossing train/test with conflicting labels and different lesion identifiers. BCN is blocked for launch; preserve the arrays until a documented data decision resolves this. The cleanup session reports no exact cross-split duplicates for iwildcam and fMoW, but near-duplicate, source-identity, group-lineage, and untouched-holdout questions remain open. Those reports are not an independent data audit by this research task.

Inspect learnable error at the **actual allocator output**, separately by backbone and development group. The production `targeted_correction` performs coupled class/global/local correction and an LP fallback; it should not be silently replaced by independent global top-K or an idealized per-group sort. Record actual emitted counts, selected sample IDs, alternatives, and the path used. Margin at a nominal K is only a diagnostic, not an exact model of all allocator decisions.

Use a fixed labeled development split for measuring false positives selected, available true positives outside, and feasible exchanges. Those labels may screen the study, but cannot enter transductive gradients. Error headroom is an upper bound on possible improvement, not evidence that the gradient can learn it. Train accuracy is neither a held-out cut metric nor a dataset suitability theorem.

Budget permutation is already implemented. It can test whether the group-budget mapping matters, provided the deployment allocator still gets the same true deployment caps. Arbitrary permutations across unequal group sizes can also change infeasibility and optimization difficulty. Match or stratify group size/support where possible, record induced violation/displacement, and treat an indistinguishable outcome as potentially underpowered. Never conclude that the entire family ignores information from one null permutation comparison.

For quality, prespecify cc-F1 over a fixed constrained class set, alongside macro-F1, uncapped F1, precision/recall, support, feasibility and elapsed compute. Keep raw and deployed predictions and pre-/post-restore identities. Use paired seed contrasts and intervals in native units; caps on the same seed are not independent replications. New seeds on inspected labels remain exploratory. An untouched-group claim awaits explicit confirmation.

## Required instrumentation and next experiment

| Measurement | Current evidence | Next addition |
|---|---|---|
| Source/config/data/split identity | Probe has selected source hashes; current event has config hash | Frozen release manifest plus arrays, split, anchor and optimizer snapshot IDs |
| Count and loss state | Pre-constraint events checked on synthetic run | Fresh post-constraint and post-next-CE counts with fixed-state loss decomposition |
| Optimizer contribution | Total displacement and gradient alignment recorded | Distinguish skip, zero-current-gradient Adam, unmodified proposal and accepted displacement |
| Task compatibility | Not recorded in current event | Same-state anchor loss/gradient and dot products against all displacement variants |
| Boundary effect | Final reordering exists; no new quality experiment | Actual allocator membership/exchange records, per group and class |
| Cost | CPU diagnostic only | Backbone-specific GPU time, extra passes, copying, peak memory |
| Numerical reliability | CPU formula/log checks | AMP/FP32, representative real priors, chunk parity, restart and negative controls |

The next bounded research execution is a same-state real-backbone replay after the cleanup release is frozen: select checkpoints by a predeclared schedule, capture full optimizer state, evaluate the causal controls above without updating production artifacts, and determine whether candidate A's premise exists. Then review its implementation design and compute costs before any training pilot. Both SSH hosts must be checked before dispatch; one session owns launch and monitoring. No GPU experiment was launched by this research task.

## Sources

1. Sohrabi, M., Ramirez, J., Zhang, T. H., Lacoste-Julien, S., and Gallego-Posada, J. **On PI Controllers for Updating Lagrange Multipliers in Constrained Optimization**. ICML, 2024. [Full text](https://arxiv.org/html/2406.04558v1). Used for controller dynamics and limits of applicability.
2. Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., and Finn, C. **Gradient Surgery for Multi-Task Learning**. 2020. [Paper record](https://arxiv.org/abs/2001.06782). Used for the gradient-conflict precedent, not an Adam guarantee.
3. Achiam, J., Held, D., Tamar, A., and Abbeel, P. **Constrained Policy Optimization**. ICML, 2017. [Paper](https://proceedings.mlr.press/v70/achiam17a.html). Used as a trust-region design analogy; RL guarantees are not transferred.
4. Cotter, A., et al. **Optimization with Non-Differentiable Constraints with Applications to Fairness, Recall, Churn, and Other Goals**. JMLR 20(172), 2019. [Paper](https://www.jmlr.org/papers/v20/18-616.html). Used for the proxy-Lagrangian distinction and assumptions.
5. Ganchev, K., Graca, J., Gillenwater, J., and Taskar, B. **Posterior Regularization for Structured Latent Variable Models**. JMLR 11, 2010. [Paper](https://www.jmlr.org/papers/v11/ganchev10a.html).
6. Pathak, D., Krahenbuhl, P., and Darrell, T. **Constrained Convolutional Neural Networks for Weakly Supervised Segmentation**. ICCV, 2015. [Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Pathak_Constrained_Convolutional_Neural_ICCV_2015_paper.html).
7. Tai, K. S., Bailis, P. D., and Valiant, G. **Sinkhorn Label Allocation: Semi-Supervised Classification via Annealed Self-Training**. ICML, 2021. [Paper](https://proceedings.mlr.press/v139/tai21a.html).
8. Ma, T., Li, X., Li, C., and Guan, R. **Learning from Label Proportions with Dual-proportion Constraints**. 2026. [Full text](https://arxiv.org/html/2603.21153v1), [CVPR Findings version](https://openaccess.thecvf.com/content/CVPR2026F/papers/Ma_Learning_from_Label_Proportion_with_Dual-Proportion_Constraints_CVPRF_2026_paper.pdf). Used for the recent hard-assignment comparison and its exact-proportion assumptions.
9. Yu, F. X., Choromanski, K., Kumar, S., Jebara, T., and Chang, S.-F. **On Learning from Label Proportions**. 2014. [Paper record](https://arxiv.org/abs/1402.5902).
10. Scott, C., and Zhang, J. **Learning from Label Proportions: A Mutual Contamination Framework**. NeurIPS, 2020. [Paper](https://papers.nips.cc/paper/2020/hash/fcde14913c766cf307c75059e0e89af5-Abstract.html).

Local evidence: current source symbols named above; historical `docs/archive/reset_2026-09-14/FRAMEWORK.md`, particularly sections z54–z56, z61–z62, z79, and delivered-update discussion around lines 6661–6665; `docs/archive/REJECTED_full_2026-08-18.md`; cleanup `docs/audits/2026-09-14-reset.md`; and the timestamped probe receipt. External sources were accessed September 14, 2026. Historical numerical claims have not been reused as fresh effect estimates.
