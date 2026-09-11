> 🛑 **ARCHIVED -- HISTORY, NOT INSTRUCTIONS.** (banner added 2026-09-11)
> The blind second referee round on the same manuscript, same date, same
> status.
> `docs/FRAMEWORK.md` is the ONLY operational document. Where this file
> disagrees with it, FRAMEWORK wins and this file is wrong. Do not run
> anything, and do not quote any figure, on the strength of this page.

<task-notification>
<task-id>adcdee5cac2ffb008</task-id>
<tool-use-id>toolu_01Wy96QoL5rJ2kU3m5s1Xy4W</tool-use-id>
<output-file>C:\Users\roeym\AppData\Local\Temp\claude\C--Users-roeym-Desktop-projects-OptimizationLoss\b7e735ce-291c-450b-b770-f2b9f761a8dc\tasks\adcdee5cac2ffb008.output</output-file>
<status>completed</status>
<summary>Agent "Blind peer review of manuscript" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>I've now read the full 35-page PDF, inspected all five figures, recomputed the headline numbers from the released corpus, and audited the implementation against the described algorithm. Here is my report.

---

# Referee Report — TMLR

**Two Portable Components for Meeting Hard Prediction Quotas in Deep Classifiers**

## Score: 71 / 100 — Major Revision

A recommendation of Major Revision, not Reject. I could not falsify a single headline number that the authors released data for; roughly thirty distinct claims reproduced exactly from `paper/data/corpus/`. The deductions are concentrated in (i) evidence that is claimed as released but is not, (ii) table-level bookkeeping that disagrees with the paper's own appendix, and (iii) protocol details that are visible only in the code and that touch a headline comparison. None of these require a new research idea; most require a data release, a corrected count, or a disclosed sentence.

---

## What reproduced (verification log)

I recomputed from `corpus_final.csv` (1,944 `paper_final` rows; grid confirmed complete and balanced, 0 duplicate keys), `alm_results.csv`, `review_graft_2026-07.csv`, `review_controls{2,3}`, `convergence_census.csv`, `extra_robustness_corpus.csv`, `b8_mnv2_octmnist.csv`.

| Claim (location) | Paper | My recomputation |
|---|---|---|
| Table 1, all 27×6 cells, both metrics | — | matches to 3 dp, every cell |
| Tight-cap headline (§5.1) | +0.038 cc-F1, 6/6 cells, 24/24 seeds, sign *p*=0.031, *t*-test *p*=0.013 | **+0.03766**, 6/6, 24/24, 0.031, **0.0135** |
| Table 11 per-backbone deltas | +0.016/+0.022/+0.035/+0.021/+0.081/+0.051 | identical, incl. ± stds |
| Table 11 macro column | 5 of 6 positive, 6th a +0.001 wash, mean +0.012 | identical (+0.0121) |
| Table 10 (81 cells vs ALM) | +0.005, 43/21/17 | **+0.0053, 43/21/17**; all nine per-cell rows identical |
| Table 10 loss region per cap | −0.002…−0.034 | identical, all nine |
| Table 2, ALM leads Fioretto | +0.010 on those cells | +0.0102 |
| Table 2, MobileNetV2 rows | +0.047/+0.037, +0.046/+0.035 | identical, 4/4 seeds each |
| §5.2 "best-or-tied in 23 / 22 / 19 / 18 of 27" | — | **23, 22, 19, 18** |
| §5.1 precision/recall at the cap | 0.90 vs 0.91; 0.31 vs 0.27 | 0.896/0.909; 0.307/0.273 |
| §5.1 dual undershoot at *L*30 | 48–67 of *K*=75 vs 73–74 | 47.7–66.7 vs 72.8–74.0 (derived from prec/rec) |
| §5.3 macro-F1 over clipping | +0.053 / +0.016 / +0.021; 77 of 81 cells | **+0.0534 / +0.0157 / +0.0205; 77/81**, the four exceptions at *L*10/*L*20, two within 0.001 |
| Table 1 caption "24 of 27, zero losses, mean +0.031" | — | **24, 0, +0.0309** |
| Table 6 (deploy, *L*30 MNetV3) | all 27 entries | identical incl. flips 7.0/1.0/0.0 and 99.5/136.5/109.5 |
| App. C flips at binding caps | 1.5 / 5.1 / 5.4 / 7.0 / 102.3 | **1.49 / 5.10 / 5.39 / 7.03 / 102.28** |
| App. C joint satisfaction | 0.21 vs 0.15; 0.60/0.50/0.39/0.35/0.00 | **0.213 / 0.145; 0.597/0.500/0.389/0.347/0.000** |
| Fig. 5 native satisfaction | 1.00/1.00/0.87/0.90/0.145 | identical |
| §5.4 convergence | median 17/20/49.5; Hounie slowest 18/18; TraLO first 11, within 1 ep 15/18; feasibility 100/100/87 | **all identical** (87.5% for Fioretto) |
| Table 5 graft, all 36 entries | — | identical; lifts +0.019…+0.073 and +0.035…+0.103 (one −0.027) |
| Graft summary | median 86% / 92% recovered; 8W/4T/0L; mean edge +0.0015; trails in 2 of 6 | **86.4% / 92.4%; 3W3T + 5W1T = 8/4/0; +0.0015; 2 of 6** |
| ALM graft (§6) | +0.031/+0.026/+0.026/+0.013/+0.030/+0.044, mean +0.028; TraLO +0.0002, 11/24 seeds, median 126% | **all identical** |
| Re-run controls reproduce main grid | ±0.0013 | ±0.0012 |
| Anti-windup restart identical to host | "identical in every cell" | max |Δ| = 0.0000 |
| Ablation: reset / ρ / freeze / joint | +0.079 (1.00) / +0.001 (0.42) / +0.000 (0.38) / +0.036 (1.00) | **identical**, all four |
| App. B.2 penalty shape | −0.001; 9W/9L/6 exact ties; sign *p*=1.0; +2 epochs | **−0.0011; 9/9/6; 1.000; 25→27.5** |
| App. B.3 β and λ_step sweeps | ≤0.0012; ≤0.003; 30/30 comparisons clear the baseline | **−0.0012/+0.0001; −0.0029; 30/30, min +0.0144, max +0.0828** (paper says "+0.088" — should be +0.083) |
| App. B.4 init control | 28 / 6-of-24 stalled; Hounie 40→31; TraLO 25 | 27.5 / 6-of-24; 39.5→31; 25 |
| App. A step-fairness | Fioretto moves &lt;0.001; Hounie 0.43→0.35; best trails by 0.019 | **&lt;0.001; 0.4328→0.3489; 0.0185** |
| Fig. 1 bootstrap CIs | +0.046 [+0.024,+0.074]; +0.033 [+0.021,+0.044] | **exact match** (20,000 resamples); all six per-backbone bands exclude zero |
| Table 12 Δcc-F1 | +0.012/81%, +0.009/75%, −0.000/38% | identical |
| Table 1 std range ".001–.055" | — | 0.0007–0.0551 |

I also verified `∂ℓ/∂S` in Eq. (2) symbolically against Eq. (1) (difference simplifies to 0), and App. A's bounds: `max_u u/(1+u²)² = 1/√3` giving `2ρ/K · 3√3/16 = 9ρ/(8√3 K)` as stated, and `lim_{S→K⁺} ∂ℓ/∂S = 1/K` independent of ρ. The code (`src/losses/transductive_loss.py:65-75`) implements Eq. (1) term-for-term. The checkpoint-selection rule is **character-for-character identical** across TraLO and all four trained baselines and is on the constraint-excess axis, never F1 (`tralo/train.py:490-506` vs `fioretto_ldf/train.py:359-373`, `hounie_rcl/train.py:390-404`, `fioretto_alm/train.py:358-372`); the warmup-cache hash excludes methodology and all constraint hyperparameters, so every method at a given (backbone, dataset, cap, seed) loads a byte-identical warmup; post-hoc adjustment is applied with no method-name gating. On the fairness question that usually decides papers like this, the paper is clean.

---

## Itemized deductions

### A. Evidence claimed as released but absent (−11)

**A1. Table 3 (imbalanced-learning baselines) has no released data. (−3)**
`grep` over every file in `paper/data/corpus/` returns zero rows for focal, class-balanced, or logit-adjusted training; `corpus_final.csv` contains exactly six methods (`danits_lp, fioretto_ldf, heuristic, hounie_rcl, tralo, tralo_bounded`). Table 3 is the table that *narrows the paper's own headline* ("against a focal-trained clipper it holds on two of the three datasets"), and §7 and §8 both carry that narrowed form. A reader cannot check the number that scopes the paper's main deployment claim. The Reproducibility Statement (p. 17) explicitly lists "the campaigns added in revision: … the imbalanced-learning baselines" as accompanying the submission.

**A2. Table 8 (native 224×224 HAM10000, 96 runs) has no released data. (−2)**
Same file audit. App. D carries three claims (macro-F1 margin over LP-LG reproduces in 6/6 cells; tie against the trained duals at −0.001/+0.001; native satisfaction 1.00/1.00/0.96/0.00), all of them unverifiable, and this is the paper's only evidence that anything survives outside the upsampled 28×28 setting.

**A3. Table 4's "Undershoot hinge (alone)" row is not reproducible, and the released ablation file contradicts the paper's labels. (−4)**
`extra_robustness_corpus.csv` block `B_loo_ablation` contains exactly four arms: `no_reset`, `no_rho`, `no_freeze`, `plus_kl`. **There is no `no_hinge` arm anywhere in the release.** I reproduced the reset row (+0.0790, winrate 1.000), the ρ row (+0.0007, 0.417), the freeze row (+0.0004, 0.375) and the joint row as TraLO − TraLO-bounded (+0.0364, 1.000, cell means +0.019…+0.070) exactly. The hinge-alone row (+0.032, 0.92, "cell-mean gaps +0.009 to +0.056") cannot be derived from anything shipped — I checked the obvious proxy (`no_reset` − `tralo_bounded`), which gives −0.0427 at winrate 0.00, so it is genuinely a different arm, just not released.

Worse, the shipped digest `ablation_complete.csv` labels the +0.0364 / winrate-1.0 row **`hinge`**, and `CAMPAIGN_RESULTS.md` in the same directory describes that same row as "undershoot hinge +0.036 24/24" and recommends "a two-knob ablation (reset + hinge), NOT four co-equal components". The paper's Table 4 assigns +0.036 to "Reset + hinge (TraLO-bounded)" and puts a different number (+0.032) on the hinge alone. A referee cannot tell which artifact is authoritative. This matters more than the two points of arithmetic, because "**Each of the two is load-bearing on its own, not only as a pair**" (§8, and the abstract's "each load-bearing on its own") rests entirely on the unreleased row.

**A4. The two robustness checks in §5.1 ("Robustness of the OctMNIST advantage") are not reproducible. (−1)**
(i) The leave-one-backbone-out CV reports "pooled held-out gap +0.033 at a 93% seed-winrate and 100% on the tight caps"; the pooled seed-level gap over the six tight cells is +0.0377 at winrate 100%, so +0.033/93% must come from a different cell set, and the paper says the script is "in the released corpus" — `paper/data/corpus/` contains no scripts. (ii) The "re-derivable without the gap" rule is specified as "the clipper is forced into ≥65 flips **and the class is still discriminable (mid-range cc-F1: recall headroom left at high precision)**"; the second predicate has no threshold, so the 100%-vs-0–40% contrast cannot be recomputed. A rule presented as a robustness check should be executable.

**A5. The MobileNetV2 rows of Tables 7 and 9 do not reproduce. (−1)**
Table 9 MNV2: paper +0.002 / −0.002 / +0.002 (Tissue/Derm/Oct); I get **+0.0066 / +0.0004 / +0.0025** over 7/7/5 caps. Table 7 MNV2: paper +0.049 / +0.004 / +0.040; I get **+0.0500 / +0.0267 / +0.0398** — Tissue and Oct match, Derm does not. The cause is visible in the corpus: MNV2 derm/tissue *L*50 carries 8–27 rows per method drawn from three campaigns (`blackwell_validation`, `paperv2_phase3_v3`, `l50fill_g0`), *L*40/*L*60 have no Fioretto rows at all, and TraLO has 1–5 rows there. §5.1 does warn that MNV2 was assembled from several campaigns and is paired within-campaign, but the release does not say *which* campaign each cell used, so the rows are unauditable. (Table 2's MNV2 tight-cap rows — the load-bearing ones — reproduce exactly from `b8_mnv2_octmnist.csv`.)

### B. Internal inconsistencies (−6)

**B1. Table 9's win/tie/loss counts are wrong in two cells and disagree with Tables 13/15. (−2)**
Table 9's caption states the rule: "a win needs the mean gap and at least half the seeds to clear +0.005". Applying it:
- RegNetY-400MF × OctMNIST: Table 9 says **4/5/0**. I get **5/4/0** under *either* the ≥2/4 rule or the ≥3/4 rule (wins at *L*30 +0.035 4/4, *L*40 +0.021 4/4, *L*60 +0.020 3/4, *L*70 +0.019 4/4, *L*80 +0.0056 3/4). **Table 15 in the same paper says "9 cells: 5W / 4T / 0L"** and lists all five as wins. Table 9 contradicts Table 15.
- RegNetY-400MF × TissueMNIST: Table 9 says 1/5/3 (correct under ≥2/4); **Table 13 says 0W/6T/3L** (correct under ≥3/4, which is the rule Tables 13–15 state in *their* captions).
So the two tables use different seed thresholds, one entry matches neither, and the captions disagree about the rule. The *means* — which carry the claim — are all correct; only the counts are unreliable, but they are the numbers a reader scans.

**B2. Table 12's stated n is wrong for one column. (−1)**
The caption says "n=16 paired seed-cells". For MobileNetV3 at *L*50/*G*20, seeds 1–3 have **no** Heuristic or LP-LG runs in the corpus, so the Δmacro-F1-vs-clipper column has 13 pairs, not 16. That is exactly how the reported 77% arises: 10/13 = 76.9%; 10/16 would be 62.5%. Either state n=13 for that column or fill the three missing clipper runs.

**B3. Table 4's caption is self-contradictory. (−1)**
It says "the *Undershoot hinge (alone)* row removes the hinge with the reset retained. **The second row is the joint removal** — the TraLO-bounded control strips both at once". The second row *is* "Undershoot hinge (alone)"; the joint removal is the third. The sentence then continues "It helps in 22 of the 24 paired runs…" with no recoverable antecedent. Given A3, this is the one caption a reader most needs to be unambiguous.

**B4. §3 says the verification step is "essentially never needed"; App. C says the opposite. (−1.5)**
§3, p. 6: "A post-hoc verification step is available to enforce a hard count if a run has not landed natively; for the reset+hinge package it is essentially never needed (App. C)." App. C reports joint native satisfaction of **0.21** over the grid and **0.60** at binding caps — i.e. TraLO's predictions are edited in 79% of runs overall and 40% at binding caps (my recomputation confirms 0.213 and 0.597). App. C is explicitly about not conflating global and joint satisfaction, and §3's forward reference to it asserts the conflation. Rewrite as "essentially never needed for the *global* cap".

**B5. §5.3's "lone exception" is two exceptions. (−0.5)**
"the dual baselines post nearly identical macro-F1 margins (the lone exception a tie at OctMNIST×ViT-B/16)". Per-cell margins vs the best clipper: Hounie-RCL is +0.000 at OctMNIST×RegNetY-400MF (TraLO +0.011) and −0.003 at OctMNIST×ViT-B/16 (TraLO +0.007). Two cells, not one.

### C. Implementation vs. description (−5)

**C1. Algorithm 1 misstates when the hinge is active. (−2)**
Algorithm 1, line 3: "update θ on L_CE + λ_G ℓ_G + λ_L ℓ_L (**plus the hinge once satisfied**)". The code (`src/methodologies/tralo/train.py:308-332`) gates the hinge only on `λ&gt;0` and `β&gt;0`; `satisfaction_epoch` is never consulted, so `λ_c β max(0, K_c − S_c)/K_c` is in the objective from constraint-epoch 1. §3's prose ("identically zero while S≥K … engages only once the soft count falls below the cap") is the correct description; Algorithm 1 is not. This is not cosmetic: App. F states that on TissueMNIST and DermMNIST "the warmup model already predicts the constrained class at close to its capped rate", which is precisely the regime where the hinge can push the soft count *up* while the hard count is still violated and λ is still ratcheting. §3 flags this as a theoretical possibility ("in principle the hinge can engage while a cap is still hard-violated"); the code shows it is the structural default. Fix Algorithm 1, or gate the hinge and re-run.

**C2. `force_exact=True` budget-fills the trained family but not LP-LG, and this is not stated. (−2)**
`src/pipeline/eval.py:47-49` calls `targeted_correction(..., force_exact=True)`, whose Phase 2 fills the constrained class up to **exactly K**. The LP-LG baseline solves `Σ R[s,i] ≤ Ψ(i)` and is under no obligation to reach K. The function's own docstring concedes the switch changes outcomes ("`force_exact=False`: legacy behavior … favored over-predicting methods … by not forcing under-K fill"). Within the cc-F1 headline (TraLO vs the trained duals) all arms get the same treatment, so that comparison is budget-matched and unaffected. But the *macro-F1-over-clipping* result — §5.3, Table 6, Table 7, App. D — compares budget-filled trained methods against an LP that may be under-filled. This is a result-affecting protocol choice visible only at one call site. It must be stated in §4, and ideally the trained-vs-LP-LG comparison should be re-run at matched realised counts.

**C3. macro-F1 is not pinned to `num_classes`. (−1)**
`src/training/metrics.py:184-189` calls sklearn with `average='macro'` and no `labels=` argument, so the average is over classes present in `y_true ∪ y_pred`. A run in which a rare class is never predicted averages over a smaller denominator than one where it is. This affects all methods, but it is not the "macro-F1 for overall quality" the paper describes, and on DermMNIST (DF at 1.1%, VASC at 1.4%) the denominator can plausibly differ between arms.

### D. Baseline health and tuning (−3)

**D1. `ce_nan = 1.0` for 160/160 dual-baseline runs in the released corpus, and the paper never mentions it. (−2)**
`extra_robustness_corpus.csv` carries a `ce_nan` column: **1.0 for all 80 Fioretto-LDF and all 80 Hounie-RCL rows; 0.0 for all 224 TraLO, 32 TraLO-bounded, 32 Heuristic and 32 LP-LG rows.** `CAMPAIGN_RESULTS.md`, shipped in the same directory, states "96/96 runs hit ce_nan across Fioretto step {0.001, 0.002, 0.01} AND Hounie η {0.001, 0.05, 0.1}, AND at the default paper step. No step yields finite-CE training." Nothing in the manuscript discloses this. Either (a) the dual baselines diverge numerically under the shared recipe in 100% of runs and are rescued by the checkpoint-restore rule — in which case the comparison is against numerically broken baselines and the reader must be told — or (b) the flag is a logging artifact with no meaning, in which case a column that reads as catastrophic should not ship undefined. This is the single item most likely to change a reader's confidence, and it costs one paragraph to resolve.

**D2. The step-fairness defense is one-backbone, and η=0.01 is not Hounie's best setting where it matters. (−1)**
Block C is MobileNetV3-only (`c.model.unique() == ['MobileNetV3']`), yet the largest TraLO margin is on the ViT-B/16 (+0.081 at *L*30). Within that one backbone, on the OctMNIST tight cells Hounie at η=0.001 scores 0.3861/0.4796 versus the frozen η=0.01's 0.3262/0.4562 — i.e. **the shipped setting costs Hounie +0.024 cc-F1 at *L*40**. It does not change the headline (Fioretto is the better dual there, and the paper takes the per-seed best of the two), but "no setting closes the gap" is established on one architecture at a setting that is not the baseline's best. Relatedly, the `paper_final` config generator is not in the released code, and every OctMNIST generator that *is* released omits `hounie_eta_lambda` — falling back to the code default 0.1, not the §4-stated 0.01 — and hard-codes `fioretto_step_size: 0.01` rather than 0.005. The corpus values themselves are consistent with η=0.01 and step≈0.005 for the main grid, so I do not believe the headline runs are mis-tuned; but the released configs cannot be reconciled with §4, and nothing shipped lets a reader adjudicate.

### E. Statistics (−2)

**E1. Bootstrap CIs on n=4 are presented as 95% intervals. (−1)**
Figure 1's bands are "95% bootstrap intervals (20,000 resamples of the four paired per-seed gaps)". With n=4, a percentile bootstrap interval excludes zero **exactly when all four gaps share a sign** — it carries no information beyond a 4/4 sign pattern (one-sided p=1/16), and its nominal coverage is badly off. The same applies to the cell-level intervals in §5.1 (n=4 cells at each cap). I reproduced all of them exactly, so this is not an error, but "95% CI excluding zero" reads as far stronger evidence than 4/4 concordance.

**E2. The six tight-cap cells are treated as independent units although they share seeds and backbones. (−1)**
§4 states the rationale — "cells are the independent units; the four seeds recur across cells" — but seed recurrence is precisely what breaks independence, and the six cells are 3 backbones × 2 adjacent caps sharing the same cached warmups. The *t*-test on six cell means (p=0.0135, which I reproduce) has an effective n nearer 3. The accompanying six-cell sign test is saturated: its minimum attainable two-sided p is 0.031, so p=0.031 means only "all six same sign". Both are honestly labelled, and the BH arithmetic on p. 8 is correct (at i=m=2, 0.031 ≤ 0.05), including the candid note that neither component clears q/m = 0.025 alone. But the design is low-powered throughout and the paper leans on it in several places.

### F. Scope words and presentation (−2)

**F1. The abstract's "reaches the cap natively in nearly all runs" is the global cap only. (−1)**
The abstract introduces the constraint as "a hard cap … sometimes with per-group limits", then claims constraint-time training "reaches the cap natively in nearly all runs". On the paper's own joint criterion (global met *and* zero flips), TraLO is at **0.213** over the grid and **0.597** at binding caps. §8 and App. C both state the correct scoped version ("the *global* count is met during training … the per-group caps still require some editing"); the abstract should adopt it.

**F2. Compiled-PDF defects and length. (−1)**
Seven inter-word spaces are lost in the PDF, all traceable to stripped revision braces in `main_clean.tex` — p. 2 "after training**with** an imbalance-unaware clipper"; p. 8 "on all three datasets,**dual** step 0.005"; p. 12 "none is a loss**against** Fioretto-LDF"; p. 15 "Fioretto-LDF, Hounie-RCL**and** ALM"; p. 17 "asymmetric caps),**including** against"; p. 17 "landing on the count natively.**That** last margin"; p. 17 "ablation and graft.**Each** of the two". Figure 1's top panel is, by its own caption, uninformative ("the curves nearly coincide, so the advantage is invisible at this scale") and consumes half a figure. Figure 3's two-row legend crowds the panel (a) title. §5.1 forward-references Table 11 on p. 31. 16 body pages plus 19 appendix pages is a lot for a result whose positive part is one dataset at two cap levels; §5.2 and App. E overlap substantially.

---

## What the paper does well

These are specific, and several are better than what I usually see.

1. **The negative results are not buried, and they are load-bearing.** The paper's own title component — the saturating penalty — is reported as quality-neutral, with a properly powered arm (24 runs, 9W/9L/6 exact ties, sign p=1.0) and the honest observation that 6 of 24 runs land on *identical* cc-F1. §6's conclusion that "once the recipe is grafted the three methods are statistically interchangeable" (+0.0015 over the dual hosts, +0.0002 over grafted ALM at 11/24 seeds) is stated in the abstract. Both reproduce exactly. Very few papers publish the finding that their package is dispensable.

2. **The comparator was strengthened against the authors' interest.** ALM was added specifically because the graft "upgrades only the weaker baselines", ALM does beat plain Fioretto-LDF (+0.010 on the headline cells, which I confirm), and the grafted-ALM result is the one that most nearly dissolves the contribution. §6 says so in as many words.

3. **The loss region is reported at full granularity.** Table 10 gives the whole 81-cell picture including the OctMNIST × MobileNetV3 loose-cap region where TraLO loses by −0.025, with a per-cap itemisation, and explains why reporting only the winning band would be misleading. All of it reproduces.

4. **Baseline configuration is genuinely fair where it counts.** Verified in code: identical cached warmups (the hash excludes methodology and constraint hyperparameters), identical 300-epoch budget and 5-epoch stop rule, identical gradient clipping, identical CE-saturation gate, identical post-hoc treatment with no method-name gating, and a checkpoint rule that is textually identical across all five trained methods and selects on constraint excess, never on F1. Hounie additionally receives a checkpoint-restore that its published algorithm does not have — a disclosed strengthening of the baseline.

5. **Controls for the obvious objections exist and hold.** The anti-windup arm (dual restarts) produces predictions identical to the host in every cell (max |Δ| = 0.0000), which is the right way to show the linear-penalty comparison is not a strawman. The non-zero-λ-initialisation control removes the "head start" reading of the convergence result. The β/λ_step sweep shows TraLO clears the best trained baseline in all 30 arm-by-cell comparisons.

6. **Two headline metrics, not one, and the constrained class is the primary one.** Choosing cc-F1 — the quantity the constraint most endangers — rather than an accuracy or satisfaction headline is the right call, and §5.1 volunteers that a trivial clipper posts higher raw cc-F1 in three of the six tight cells (I confirm: 0.440 vs 0.422 at OctMNIST × ViT × *L*30), then quantifies the editing that buys it.

7. **The limitations section is real.** The motivation–result mismatch on OctMNIST ("drusen is neither rare nor high-stakes … readers should not take the OctMNIST result as evidence about rare-class clinical screening"), the transductive-fitting confound, and the "margin over clipping is a margin over *vanilla* clipping" retraction are all things most authors would omit.

8. **App. A's mathematics is correct.** The bounds, the 1/K boundary slope, the rank-one direction argument, and — importantly — the explicit refusal to claim λ-invariance as a theorem ("this is a partial argument only … the near-neutrality is empirical rather than theoretical") are properly scoped. The three caveats attached to the finite-multiplier statement (κ finite is an assumption; dL_CE/dS ≤ 0 is weakest on exactly the two tie datasets; "local minimum" is along a path, not in θ) are the kind of self-policing that is normally supplied by a referee.

---

## Changes that would raise the score, in order of effect

1. **Release the missing arms** (−11 recoverable): the `no_hinge` ablation runs, the focal / class-balanced / logit-adjusted campaign behind Table 3, the 96 native-HAM10000 runs behind Table 8, the LOBO script, and a per-cell provenance key for the MobileNetV2 rows. Reconcile `ablation_complete.csv`'s `hinge` label and `CAMPAIGN_RESULTS.md` with Table 4's row assignment. If the hinge-alone arm does not exist, delete the row and retract "each of the two is load-bearing on its own" to "the pair is load-bearing".
2. **Disclose or explain `ce_nan`** (−2): one paragraph in §4 or App. A stating what the flag means, why it fires on 160/160 dual runs and 0/288 others, and what the restored checkpoints therefore represent.
3. **Fix Table 9's counts and state one seed rule** (−2): recompute all twelve cells under a single stated threshold and reconcile with Tables 13–15.
4. **State the `force_exact` asymmetry in §4** (−2), and re-run the trained-vs-LP-LG macro-F1 comparison at matched realised counts, or show the gap is unchanged.
5. **Correct Algorithm 1 line 3** (−2) to match the code, and say explicitly that the hinge is active from the first constraint epoch.
6. **Fix B2–B5, F1, F2** (−4.5): Table 12's n, Table 4's caption, §3's "essentially never needed", §5.3's "lone exception", the abstract's native-satisfaction scope, and the seven lost spaces.
7. **Reframe the n=4 uncertainty** (−2): present Figure 1's bands as "all four seed gaps positive" or use a method appropriate to n=4; note the effective sample size behind the six-cell *t*-test.
8. **Pin macro-F1 to `num_classes`** (−1) and confirm no number moves.

Items 1–3 alone would move this from Major to Minor Revision.

---

## Experiments I would require before acceptance

1. **The `no_hinge` arm, run and released** — 24 runs (2 caps × 3 backbones × 4 seeds) with the reset retained and the hinge disabled, under the frozen recipe. This is the sole evidence for a claim in the abstract and it does not currently exist in the release.
2. **Hounie-RCL at its best step on the tight cells, all three backbones** — η ∈ {0.001, 0.01} × 2 caps × 3 backbones × 4 seeds (24 runs). Block C is MobileNetV3-only and shows η=0.001 buys Hounie +0.024 cc-F1 at *L*40. The headline margin should be shown against each dual at its own best setting, especially on the ViT-B/16 where the margin is largest.
3. **A budget-matched trained-vs-LP-LG comparison** — either re-run LP-LG with the constrained-class count filled to exactly K, or re-run the trained arms with `force_exact=False`, on the *L*30 cells of all three datasets (72 runs). The +1.6–5.3 pp macro-F1 headline currently compares a budget-filled arm against one that need not fill.
4. **A `ce_nan` diagnostic** — for one cell per dataset, the per-epoch CE traces for Fioretto-LDF and Hounie-RCL, plus the epoch at which the restored checkpoint was taken. No new training required; the logs exist.
5. *(Strongly encouraged, not required)* **The inductive variant** named in §7 — constraining the train-set count only, evaluated on the test batch — on one dataset × one backbone × the nine caps. §7 identifies the constraint-training-versus-test-batch-adaptation confound as unresolved, and it is the single largest open question about what the +1.6–5.3 pp actually measures.</result>
<usage><subagent_tokens>210470</subagent_tokens><tool_uses>64</tool_uses><duration_ms>1479468</duration_ms></usage>
</task-notification>
---

## My verification of the heaviest deductions (added after the review, not by the reviewer)

**A3 (-4) — PARTLY WRONG.** The `no_hinge` runs exist: campaign `g5_hinge_oct`,
24 completed runs in `paper/data/manifest/experiments.csv`. Recomputed
TraLO - no_hinge: mean **+0.0323** (paper: +0.032), winrate **0.92** (22 of 24),
cell gaps **+0.0094 to +0.0557** (paper: +0.009 to +0.056), 6/6 cells positive.
The claim is true and reproduces exactly. This is a RELEASE GAP (manifest but
not `data/corpus/`), not a false claim. The reviewer's secondary point stands:
`ablation_complete.csv` labels the +0.0364 row `hinge` while Table 4 assigns
+0.036 to "Reset + hinge (TraLO-bounded)".

**D1 (-2) — CONFIRMED, and it is a logging artifact.** `fioretto_ldf/train.py:111`
sets `ce_losses = []`, then `for batch_X, batch_y in (train_loader if not skip_ce else [])`.
When the shared CE-saturation gate fires the loop never runs, and `np.mean([])`
at line 295 returns NaN. TraLO guards it with `num_batches = 1` when `skip_ce`
(`tralo/train.py:150-153`). The duals are not numerically broken. But the column
ships undefined and reads as catastrophic, so the disclosure request is fair.

**C2 (-2) — CONFIRMED at the code level.** `src/pipeline/eval.py:47-49` calls
`targeted_correction(..., force_exact=True)` unconditionally on the branch the
trained arms take. `danits_lp/train.py:67`, `heuristic/train.py:112` and
`imbalanced_common.py:54` all set `skip_targeted_correction=True` and supply
their own predictions, so they never enter that branch. The two families do
receive different post-hoc treatment and the paper never says so.
`docs/AUDIT_2026-07-31.md` already measured the size of this on the cc-F1
headline (+0.038 -> -0.002 when equalised).

**C1 (-2) — CONFIRMED as a text error.** `tralo/train.py:308-332` gates the hinge
on `lam <= 0 or fior_beta <= 0` only; `satisfaction_epoch` is never consulted, so
the term is in the objective from constraint-epoch 1. Algorithm 1 line 3
("plus the hinge once satisfied", `docs/main.tex:528`) is wrong. Nuance the
reviewer did not state: the term is `relu(K - S)`, so it *evaluates* to zero
while the soft count is at or above the cap. The fix is line 3's wording, not a
re-run. Sec. 3's prose is already correct.

**C3 (-1) — CONFIRMED.** `src/training/metrics.py:183-187` calls
`precision_recall_fscore_support(..., average='macro')` with no `labels=`, so
sklearn averages over the classes present in `y_true` union `y_pred`. Whether any
released number moves needs the per-run `final_predictions.csv` files, which are
on the server, not in `data/corpus/`.

**A1/A2 (-5) — CONFIRMED.** `paper/data/corpus/` has zero rows matching
focal / class_balanced / logit_adjust, and no native-224 file. Both are release
gaps.

**B5 (-0.5) — CONFIRMED, and the fix strengthens the paper.** Recomputed
macro-F1 margins over the best clipper, per (dataset, backbone), n=9 cells:
Hounie-RCL is **+0.0002** at OctMNIST x RegNetY-400MF and **-0.0030** at
OctMNIST x ViT-B/16 -- two cells, matching the reviewer to the digit.
Fioretto-LDF has one (+0.0001, OctMNIST x ViT-B/16). **TraLO has none: 0 of 9
cells at or below +0.0005.** So the honest correction ("two exceptions, both
Hounie-RCL on OctMNIST, none for TraLO") reads better for us than the current
sentence.

**B1 (-2) — REAL INCONSISTENCY, but not the one described.** Table 9's
RegNetY-400MF x OctMNIST cell reproduces **exactly** as 4W/5T/0L under its own
caption rule (mean gap > +0.005 and at least half the seeds above +0.005).
TissueMNIST x RegNetY-400MF likewise reproduces as 1W/5T/3L. The reviewer's
5W/4T/0L does not reproduce under either seed rule. What *is* wrong: the
granular table (`tab_granular_oct.tex`) reports the same cell as 5W/4T/0L, and
its per-cap `win` column is a plain **sign** winrate (gap > 0), not a
clear-the-band winrate. RegNet x Oct L70 is 4/4 by sign but 2/4 above +0.005;
L80 is 3/4 by sign but 1/4. The cell *means* agree to the digit across both
tables. So two tables label two different quantities "win". The fix is one
stated rule and a regeneration, or relabelling the granular column.
