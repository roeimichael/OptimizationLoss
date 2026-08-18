# Rejected experiments — do not re-introduce without reading this

This file documents backbones and datasets that were tried for the TraLO thesis (transductive prediction-count constraints vs Fioretto LDF / Hounie RCL baselines) and **failed or did not produce a clean win**. Their wrappers/entries have been removed from the active pipeline (`src/models/imagery/`, `src/models/model_factory.py`, `src/utils/data_loader.py`, and the now-retired `gen_model_search.py`) on 2026-05-28.

Re-adding any of these requires a concrete reason that addresses the failure mode below — otherwise you'll burn GPU time reproducing a known dead end.

The "headroom hypothesis" is the explanatory lens: TraLO's macro-F1 edge appears only when warmup train-acc on derm lands in roughly **[0.70, 0.82]**. Outside that band the warmup is either saturated (argmax locked, no slack for TraLO to redistribute) or degenerate (majority-class collapse, no signal).

**Confirmed mechanically on 2026-08-16.** At warm-up 50 train accuracy is 0.998 before the constraint phase begins, so the CE-saturation gate fires within 2 epochs and the CE loop then iterates an *empty list* (`tralo/train.py:154`) — `L_CE` is 0.0 at the last epoch in 150/150 runs. The constraint phase becomes ~30 unit-norm steps against a frozen representation, which can only prune an ordering CE already fixed. There is nothing left to learn. This is not a tuning problem and no penalty variant escapes it.

> **This file now covers methods and regimes, not just backbones and datasets.** Read the "Rejected regimes" and "Rejected loss/method directions" sections below before proposing any new arm.

---

## Rejected backbones

| Backbone | Mode of failure | Evidence |
|---|---|---|
| **DenseNet121** | Saturates: ep1 train-acc 0.877 (≥ 0.84). | Probe run in `project_model_search_2026-05-27`. |
| **MNASNet10** | Degenerate: train-acc stuck flat ~0.67 across all 8 warmup epochs (majority-class collapse on derm NV ≈ 67%). Macro-F1 0.27. | Probe run in `project_model_search_2026-05-27`. (`mnasnet1_3` not tested; allowed as a separate candidate.) |
| **RegNetY16GF** (actually `regnet_y_1_6gf`, ~11M params despite the name) | Saturates: ep1 train-acc 0.8439. | Probe run in `project_model_search_2026-05-27`. |
| **SqueezeNet11** | Failed smoke despite ideal warmup band (0.78). On aider **loses both baselines** (Δ Fior −0.0129, Δ Hounie −0.0131). On derm only Hounie-only/Fior-tie. Fire-module architecture does not host the TraLO edge — kills the simple "any mid-band warmup wins" hypothesis. | `project_model_search_2026-05-27` "Architecture-diversity follow-up". |
| **ViTTiny** (`timm.vit_tiny_patch16_224`) | Pure transformer memorizes derm in 1 epoch (ep1 train-acc 0.8279). Smoke = **Fioretto-only / loses Hounie on both** (derm Δh −0.0107, aider Δh −0.0041). Mirror image of ShuffleNet's Hounie-only — interesting datapoint but not a winner. | `project_model_search_2026-05-27`. |

### What this means for new backbone candidates
- Pretrained ImageNet weights + small fine-tuning dataset (derm has 9.6k train) = most large/modern backbones saturate ep1. Mobile-family inverted-residual + depthwise-separable convs are the only known clean winners.
- Mid-band warmup (~0.75) is **necessary but not sufficient** (SqueezeNet had it and still failed both-baselines on aider).
- Reasonable future candidates: MobileViT (under test), EfficientFormer-L1, MNASNet1_3, ConvNeXt-Pico, MobileViT variants.

### ViT-S and ConvNeXt-T — explicitly rejected (2026-06-08)

Reviewers across iterations 3, 6, 7 repeatedly suggested running ViT-S/B or ConvNeXt-T on the three medical/aerial benchmarks to corroborate the F1 win on a non-MobileNet backbone family. **Both were tried and discarded** for the same structural reason:

| Backbone | Failure mode |
|---|---|
| **ViT-S / ViT-B** | Memorizes the training distribution in 1–2 epochs (train-acc → 1.0); test-set warmup is saturated by phase-2 entry, so the constraint-phase gradient has no slack to redistribute. Results across methods (TraLO, Fioretto, Hounie, post-hoc) flatline within $\pm 0.005$ F1, masking the TraLO advantage entirely. |
| **ConvNeXt-Tiny** | Same failure mode as ViT-S: the ConvNeXt inverted-bottleneck + LayerNorm capacity overruns the 9.6k–13k train sets of derm/tissue/aider, saturating the warmup and removing the headroom the bounded-penalty story depends on. |

The dataset-side problem is that none of our three benchmarks (tissue 9.6k, derm 9.6k, aider 6.4k train) are **hard enough** to keep a 22M-parameter transformer in the un-saturated warmup band. We attempted the search for a harder imagery benchmark (PathMNIST, ISIC2019, EuroSAT, So2Sat, CIFAR-100, OctMNIST — all listed in the dataset table below) and none gave a clean TraLO story under ViT-class capacity. The corroboration of the F1 win on non-MobileNet backbones therefore lives at the MobileNetV2 / RegNetY400MF / ShuffleNetV2 tier reported in §5.2; the transformer-class corroboration is **declared out-of-scope** for this paper.

If a future revision attempts this again: budget a hard 50k+ train imagery benchmark (StanfordDogs120, NABirds, or a curated ImageNet-100 subset with an imbalanced minority class) so the transformer's capacity is matched by genuine task difficulty.

---

## Rejected datasets

| Dataset | Mode of failure | Evidence |
|---|---|---|
| **PathMNIST** (colon histology, TUM-cap) | Too easy → saturates. MobileNetV2 = Hounie-win / Fior-tie (Δf −0.0013, Δh +0.0089). Doesn't qualify as both-baselines. | `project_model_search_2026-05-27` "Dataset-search". |
| **ISIC2019** (dermoscopy, 8 classes) | MobileNetV2 = loss-Fior / tie-Hounie. | `project_model_search_2026-05-27`. |
| **EuroSAT** (satellite, 10 classes) | Dropped per `docs/PAPER_PLAN.md` v2 (2026-05-24). Failed to give a clean TraLO story. | Memory `project_paper_plan_v2_2026-05-24`. |
| **So2Sat** (real-city groups) | Previously a TraLO win story (Tables 10-11 in old paper plan), dropped 2026-05-24 along with EuroSAT when paper plan v2 narrowed to derm/tissue/aider. **Not failed, but explicitly removed from active scope.** | Memory `project_so2sat_pivot.md`, `project_paper_plan_v2_2026-05-24`. |
| **OctMNIST** | Never properly tested; flagged for deletion to free /home NFS space. | Memory `reference_shared_nfs.md`. |
| **CIFAR-100** | Failed in past sessions; explicitly noted "do NOT re-propose" in the model-search memory. | `project_model_search_2026-05-27` dataset-search decision. |

### What this means for new dataset candidates
- Need: moderate difficulty (not saturable like PathMNIST, not degenerate like ISIC2019/TissueMNIST), an imbalanced constrained class with meaningful real-world stakes, and ideally a group axis for local caps.
- MedMNIST 2D family remains a good source (same data pipeline). Untried as of 2026-05-28: **BloodMNIST** (under test), **RetinaMNIST**, **OrganAMNIST**, **BreastMNIST**.
- Non-MedMNIST options are mostly exhausted (eurosat / so2sat dropped, cifar100 failed, aider already a winner).

---

## Rejected regimes (2026-08-16)

| Regime | Verdict | Evidence |
|---|---|---|
| **warm-up 50** | **DEAD. Do not run or headline.** CE is saturated before the constraint phase starts, so the phase is ~30 unit-norm steps on a frozen representation. All three methods land within 0.1 pp of each other. | Budget-equalized macro-F1 over the clipper: TraLO +1.07 pp, Fioretto +1.00, Hounie +0.92 — spread 0.15 pp. `results/track_b`. |
| **warm-up 5** | **DEAD ZONE.** +0.02 pp (52% of cells) — worse than both neighbours. The warm-up response is **not monotone**; never interpolate between 1 and 50 or assume "lower is always better". | Same census, n=40. |
| **warm-up 1** | **The only live regime.** This is where CE is unsaturated and the two forces actually coexist. All new campaigns go here. | Regime is worth ~8 pp; method choice ~0.1 pp. |

## Rejected loss / method directions (2026-08-16)

Thirteen "new directions" arms were opened. **Eleven were fired.** The structural reason they all tied: every one of them varied the *count penalty* — its shape, scale, schedule, margin, or clip — and **none changed what the gradient is a function of**. A count penalty is a function of the aggregate soft count, so it says how MANY and never WHICH.

| Direction | Verdict | Evidence |
|---|---|---|
| **Clip magnitude tuning** (∞, 10, 1.0, 0.3, 0.1, 0.03) | **CLOSED.** An interior optimum exists but 0.3 does not beat the incumbent 1.0: 2W/5L/5T across 12 cells. | `steps`/`steps2`/`steps3`/`steps4`, 160 runs. |
| **Removing / loosening the clip** | **REJECTED — the clip is load-bearing.** Unclipped loses 6/6 cells on cc-F1, AP and own count; worst cell drives the constrained count to 0.00. | `steps5`, 72 runs, 4 seeds × 3 datasets × 2 backbones. |
| **`geom` cut-margin hinge** (γ = 1.0, 0.50, 0.25) | **REJECTED at every margin.** 1W/4L/3T on cc-F1, and it drives the model's own count *down* in 7/8 cells. | `geom_smoke` + `geom_gamma`, 64 runs. |
| **topk** (differentiable top-K) | Killed at the novelty gate — published prior art. | Never built. |
| **llp / softhard / bilevel / distmatch / selective / rank / twosided / assign / forensics** | Never shipped a runnable objective. `rank` in particular was *named and never implemented* — the ranking direction remains genuinely untested. | Probe CSVs only. |

## In flight — wave 2, the first arms that change what the gradient is a function of (2026-08-16)

Wave 1's thirteen arms all reshaped the count penalty and all tied. These five change the *input* to the gradient instead. Every one runs the contract in "How to add a new candidate" below, and every campaign carries its own control arm so nothing is compared across campaigns.

| Arm | What the gradient becomes a function of | State |
|---|---|---|
| **`joint`** | The same count penalty, but *inside each minibatch loss* — one backward pass carries both forces instead of ~110 CE steps then 1 separate constraint step. **The paper's own stated formulation, never actually run.** | **REJECTED** — see below |
| **`baseloss`** | CE swapped for focal *inside* TraLO. B1 tested focal only as a train-then-clip baseline, never as TraLO's base loss. | **FIRST POSITIVE** — see below |
| **`ortho`** | Constraint gradient projected onto the orthogonal complement of the CE gradient, so enforcing the cap cannot undo CE progress. | **READY**, queued behind baseloss on GPU2 |
| **`rank`** | Pairwise ranking across the cut. The only arm whose gradient is genuinely **per-item** rather than per-count. | **WEAK POSITIVE / TIE** — see below. Dose probe queued. |
| **`reweight`** | No penalty at all: per-item CE weights chosen so the count falls out. One force, CE-shaped. | **REJECTED** — see below |
| **`reset`** | *Not an arm* — a config correction. `reset_optimizer_at_sat` ON vs OFF at warm-up 1. See the config defect below. | **RUNNING** GPU1, 32 runs |

**Operational warnings that must be applied when these are read** (each was measured by an independent verifier, not asserted):

- **`ortho` must be scored ONLY with `paper/scripts/score_campaign.py`.** The canonical `src/evaluation/full_census.py` + `make_winning_results.py` path knows nothing about the new `prerestore_*` artifacts and would report the perfect null with no warning. Measured: post-restore ‖θ_off − θ_on‖ is still exactly 0.000000e+00 in 2 of 4 seeds, while pre-restore is non-zero in 4 of 4 — **the checkpoint selector compresses the measured effect ~13×** (AP +0.0003 restored vs +0.0041 pre-restore). Gate every run on `Ortho Fired Frac`: if it reads 0.0000 or nan the interference story is dead and nothing else in the campaign means anything. ⚠️ Native satisfaction must be read on `[final]`, **not** `[prerestore]` — the restore is *how* the incumbent reaches feasibility (sat 0.25 both arms final → 0.00 both arms pre-restore).
- **`rank`'s fatal RNG defect is closed and mutation-tested.** At `rank_weight=1e-12` the parameter delta vs control is now exactly 0 with bit-identical pool logits (nuisance/signal **1.00 → 0.000**), and all three RNG streams — torch, numpy, python-random — are byte-identical after an ON run. Reversing the two edits inside the fixed file brings the pathology straight back, so the checks bite rather than passing vacuously.
- **`reweight`'s bf16 crash is closed and demonstrated**: at HEAD the real training loop dies with `expected scalar type BFloat16 but found Float`; this branch completes. ⚠️ **Predicted structural failure worth stating before the run, not after**: the `[1/4, 4]` weight bound is a hard ceiling on the controller's authority, and this cell demands 3.39× suppression. In a campaign-shaped probe the weight pinned at its 0.25 floor for 25 consecutive epochs and the count never fell below the class's natural rate. Expect it to miss the cap by construction — that is a finding about the reweighting family, not a bug.
- ⚠️ None of the three worktrees has anything committed; the edits are working-tree-only. A checkout of these branches elsewhere gets HEAD, not the fixed code.

**Confounds that must be stated with any result from these, not discovered afterwards:**

- **`joint` changes two things at once.** It applies the constraint gradient 126× more often *and* unclipped. A win or loss cannot separate "joint coupling" from "constraint gradient applied 126× more often, unclipped" — it has to be read against `steps5`'s noclip arm. It also costs **~4×** the control (39,645 vs 10,067 grad-enabled image passes per constraint epoch), so it is not compute-matched to its own control either.
- **`ortho` was nearly nullified by the scorer, not by the science.** Under the campaign's real hyperparameters the end-of-run checkpoint restore made the two arms **bit-identical** (‖θ_off − θ_on‖ = 0.000000e+00): the checkpoint is cloned *before* the constraint step, so epoch 1 is identical in both arms by construction, and when a run never satisfies, epoch 1 stays the min-excess pick. Worse, the restore criterion is *total excess* — exactly the quantity the projection deliberately trades away — so the ON arm is systematically handed worse candidates. Any ortho result must report **pre-restore** metrics and stratify by `Restore Kind`.
- **`rank`'s treatment was smaller than its own RNG noise.** The pair draws used the global generator, so at `rank_weight=1e-12` (gradient numerically nil) the parameter delta vs control was ‖d‖ = 0.2238 — *larger* than the 0.2222 at the campaign's real weight. Also `did_backward` let the ON arm take an optimizer step in 27.7% of epochs where the control took none.
- **`baseloss`'s focal α was a hidden 4× loss scale.** The code default `focal_alpha=0.25` is a global scale on the base loss; Adam absorbs it in warm-up but not in the constraint phase. Pinned to 1.0 so the arms differ only by the (1−p_t)^γ modulation. The focal sub-hyperparameters were also missing from the warm-up cache key, so `focal(α=.25)` and `focal(α=1.0)` hashed identically — fixed, and verified not to move any existing hash.

**Scoring**: `paper/scripts/score_campaign.py --campaign results/<arm>` — seed-paired within the campaign, metric definitions imported from `score_arm.py` so the two cannot drift, and it **md5s the raw predictions of every ON/OFF pair before printing anything**. A pair whose hashes match is reported as DEAD rather than as a 0.000 delta; on `steps5` it found 2 such pairs that a plain delta would have shown as a null result.

### `joint` — REJECTED, and the reason is the interesting part (2026-08-16)

8 runs, DermMNIST × MobileNetV3 × L30_G30 × 4 seeds. Contract clean (0 violations), warm-up bit-identical across all 4 pairs, all 4 pairs live by md5, and `joint_off` reproduces the frozen `steps5` incumbent **byte-for-byte** on every completed seed.

| metric | control | joint_on | delta | cells won |
|---|---|---|---|---|
| **AP** | 0.6136 | 0.5468 | **−0.0668** | 1/4 |
| ccF1eq | 0.4034 | 0.3966 | −0.0069 | 1/4 |
| macroEq | 0.6972 | 0.6823 | −0.0149 | 0/4 |

**It loses on the primary metric while producing textbook two-force dynamics** — cap satisfied in 26 of 29 epochs versus the incumbent's 0.5 of 29, zero cap crossings, λ never leaving its initial 0.010 (the incumbent ratchets to 1.02), and CE still falling 0.467 while the cap is held.

The mechanism, measured: joint **removes the CE disruption**. It saturates train accuracy 4 epochs earlier ([11,11,14,16] vs [15,15,15,20]), reaches higher train accuracy (0.9992 vs 0.9972) and lower train CE (0.0036 vs 0.0086). It fits the training set *better* and ranks the test set *worse* — an overfitting signature.

> **So the incumbent's once-per-epoch constraint step, which looked like the defect worth fixing, is acting as a regularizer.** That is the second "obvious defect" of TraLO to turn out load-bearing, after the unit-norm clip. Treat the remaining ones as guilty-until-measured.

⚠️ Still confounded: joint applies the constraint gradient 126× more often **and** unclipped. `steps5`'s noclip arm closes the unclipped half (it oscillates across the cap 5× instead of holding, so the *hold* comes from coupling). The pressure half needs `joint_penalty_scale="matched"`, staged at `~/add_joint_matched.py`.

### `baseloss` (focal inside TraLO) — FIRST POSITIVE RESULT, not yet a win (2026-08-16)

8 runs, DermMNIST × MobileNetV3 × L30_G30 × 4 seeds. All 4 pairs live by md5.

| metric | control (CE) | focal | delta | cells won |
|---|---|---|---|---|
| **AP** | 0.6075 | 0.6379 | **+0.0305** | 3/4 |
| macroEq | 0.6969 | 0.7165 | **+0.0196** | **4/4** |
| ccF1eq | 0.4086 | 0.4052 | −0.0034 | 2/4 |

Per-seed AP: +0.097, −0.045, +0.024, +0.045. Clears the project win rule (mean ≥ 0.005, ≥ half the seeds). macro-F1 is the more consistent effect (every seed). **This is the first arm to move AP in the right direction**, and AP is the metric the whole deficit lives in.

⚠️ **The warm-up differs between the arms by design** — `base_loss` is in the warm-up cache key precisely so a focal warm-up cannot load a CE one. Score with `--warmup-is-the-treatment`. The claim being tested is therefore "focal throughout beats CE throughout, inside TraLO", **not** "focal inside the constraint phase specifically".

**REPLICATED on the bar's four cells (`baseloss4`, 64 runs, 16 seed-paired cells per arm, 16/16 pairs live):**

| base loss | AP | cells won | macroEq | ccF1eq | native sat |
|---|---|---|---|---|---|
| **focal** | **+0.0156** | **11/16** | **+0.0089** (11/16) | −0.0007 | **+0.375** |
| `logit_adjust` | +0.0021 | 7/16 | −0.0087 | +0.0039 | +0.313 |
| `class_balanced` | −0.0022 | 7/16 | +0.0095 | −0.0076 | +0.250 |

**focal is the only arm above chance.** The other two sit at exactly 7/16 — coin flips — confirming the B1 prior that they are inert. ⚠️ Note the md5 gate reports **16/16 pairs live for all three**, so those are genuine nulls, not inert flags. Do not go looking for a bug in them.

⚠️ **The one-cell number was optimistic**: +0.0305 on DermMNIST/MobileNetV3 alone, +0.0156 across four cells. Quote the four-cell figure.

⚠️ **And it does not hold at other cap tightnesses** (`focalcaps`, L20 + L40, DermMNIST/MobileNetV3, 8 pairs): AP +0.0218 but only **4/8 cells** — a coin flip, with the mean carried by two outlier seeds (+0.104, +0.090). Against 11/16 at L30, focal is either cap-sensitive or simply noisy. **Do not present focal as a robust improvement.**

**Second, independent finding**: all three imbalanced losses raise native satisfaction by **+25 to +37.5 pp** while *under*-filling the budget (count/K 0.88–0.94 vs the control's 0.997). They suppress the constrained class, making the cap easy to hold natively — a win for the method's stated selling point that is separate from the ranking story, and one that costs nothing on budget-equalized cc-F1.

🛑 **ANSWERED BY `focalbar`, AND THE ANSWER IS THAT TraLO ADDS NOTHING TO QUALITY.** 32 runs, equal compute (30 epochs both sides), the SAME base loss on both sides, 16 seed-paired cells, 16/16 live:

| metric | focal + post-hoc clip | focal + TraLO | delta | cells won |
|---|---|---|---|---|
| **AP** | **0.7323** | 0.6898 | **−0.0425** | 2/16 |
| ccF1eq | 0.4154 | 0.3998 | −0.0156 | 3/16 |
| macroEq | 0.6958 | 0.6886 | −0.0072 | 4/16 |
| native sat | 0.0000 | **0.2500** | **+0.2500** | 4/16 |

14 of 16 seeds negative on AP. Filling in the 2×2:

|  | + TraLO | + post-hoc clip |
|---|---|---|
| CE | 0.6716 (bar) | 0.7208 (bar) |
| **focal** | 0.6898 | **0.7323** |

**Focal lifts both, and lifts the clipper more.** The `baseloss` result is real but it is a base-loss result, not a TraLO result: swapping CE for focal improves the clipper from 0.7208 to 0.7323 and TraLO from 0.6716 to 0.6898, leaving the gap essentially unchanged. **TraLO's contribution is native satisfaction (+25 pp), bought at 4.25 pp of ranking.**

**Verified before launch that `heuristic` honours `base_loss`** — it defines no criterion of its own and dispatches through `run_experiment` → `run_warmup` → `make_ce_criterion`, which reads the key. Had it silently fallen back to CE, the comparison would have looked clean and meant nothing.

⚠️ **The gap and the restore cost are the same size**: −0.0425 here vs **−0.0477** for the lowest-excess checkpoint restore (measured within-run, see the restore section). The clipper never restores. If `restoreprobe` confirms on four cells, this gap is a checkpoint-selection policy rather than a property of constrained optimization — and the honest recommendation becomes *focal base loss + no restore*, which would keep native satisfaction while removing the ranking penalty.

### `reweight` (constraint as CE reweighting) — REJECTED, on a pre-registered prediction (2026-08-16)

12 runs, DermMNIST × MobileNetV3 × L30_G30 × 4 seeds, 3 arms. Warm-ups bit-identical, 8/8 treatment pairs live.

| arm | sat_frac | count/K *in training* | AP | ccF1eq | macroEq |
|---|---|---|---|---|---|
| control | 0.0172 | 1.41 | — | — | — |
| damped (β=0.3) | **0.0000** | **2.32** | +0.0136 (2/4) | −0.0052 | −0.0041 |
| undamped (β=1.0) | **0.0000** | **2.32** | −0.0048 (1/4) | −0.0103 | −0.0077 |

**It never satisfied in a single epoch of a single run**, and sat at 2.3× the cap throughout — *worse* than the control's 1.41. Damping helped relative to undamped, as the controller analysis predicted, but neither arm reaches the target.

> **This failure was predicted before launch and is the useful part.** The verifier measured that the `[1/4, 4]` weight bound is a hard ceiling on the controller's authority while this cell demands 3.39× suppression; in a campaign-shaped probe the weight pinned at its 0.25 floor for 25 consecutive epochs and the count never fell below the class's natural rate. The campaign reproduced that exactly. **A per-item CE reweighting cannot enforce a cap it lacks the authority to reach** — widening the bound would trade that for the collapse-to-zero failure the bound exists to prevent. This is a finding about the reweighting family, not a bug in the arm.

### `rank` (pairwise ranking across the cut) — WEAK POSITIVE, NOT CLOSED (2026-08-16)

8 runs, DermMNIST × MobileNetV3 × L30_G30 × 4 seeds. Warm-ups bit-identical, 4/4 pairs live, and the term is visible per epoch in `training_log.csv` (`L_rank` 0.0303 → 0.0114, `applied=1`) — the observability fix means liveness is proven from the artifact, not inferred.

| metric | control | rank (w=0.1) | delta | cells won |
|---|---|---|---|---|
| AP | 0.6136 | 0.6220 | **+0.0085** | 2/4 |
| ccF1eq | 0.4034 | 0.4069 | +0.0034 | 2/4 |
| macroEq | 0.6972 | 0.6988 | +0.0016 | 1/4 |

Per-seed AP: +0.041, +0.025, −0.026, −0.006. It clears the project win rule on paper (mean ≥ 0.005, ≥ half the seeds) but the seed spread swamps the effect — read it as a tie with a positive tilt. Against focal's +0.0305 at 3/4, it is ~3.6× smaller.

**DOSE PROBE RUN (`rankdose`, w ∈ {0, 0.3, 1.0, 3.0} × 4 seeds, same cell).** The knob does not rescue the arm:

| `rank_weight` | 0.1 | **0.3** | 1.0 | 3.0 |
|---|---|---|---|---|
| AP delta | +0.0085 (2/4) | **+0.0323 (3/4)** | **−0.0068** (2/4) | +0.0139 (3/4) |
| macroEq delta | +0.0016 | +0.0025 | +0.0016 | +0.0118 (4/4) |

🛑 **The lesson is methodological and it cost a campaign.** The dose was chosen from the verifier's measurement of *logit separation*, which peaks at w=1.0 (+0.329). **w=1.0 is the WORST dose on AP.** Separation in the loss's own objective does not transfer to ranking quality — a knob can move its own loss a long way without moving the metric that decides the arm. **Never pick a dose from a proxy measured in the loss's own space; pick it from the scoring metric.**

The AP curve is non-monotone and, against a per-seed spread of ±0.05–0.09, mostly noise. The best dose (w=0.3, +0.0323 on one cell) sits almost exactly where focal sat on one cell (+0.0305) before four-cell replication shrank it to +0.0156 — so treat it as **unestablished**, not promising. It earns four cells only if something else motivates it.

#### 🔁 IN FLIGHT: `rankrep` — the four cells, because the structural result is the motivation (2026-08-17)

`newdirections/arm_rank/results/rankrep`, 48 runs, chained ahead of `sepopt` on dsisco01 GPU0.

The something-else arrived a day after the verdict above: post-hoc's score is a function of the ranking and nothing else, so a per-item objective at the operating point is the **only** family that can win, and `rank` is the only per-item arm in the project. That promotes it from "unestablished" to "the one worth four cells".

Re-reading the two campaigns turned up three defects that make the original numbers unreadable as a claim about the method:

1. **Neither campaign contained the baseline that matters.** Both compared `rank_on` against `rank_ctrl` — TraLO with the term off. There was no post-hoc clipper in either. So `+0.0323` says the term helps TraLO; TraLO starts ~0.05 AP *behind* the clipper, so it says nothing about beating the bar. `clip` at equal compute is in this campaign.
2. **It ran at the cap with the least on the table.** derm × MobileNetV3 × L30 only. Headroom is 0.048 at L30 and 0.115 at L50. Now 2 backbones × {L30, L50} — two cap levels, because a claim from cells sharing one has been retracted here before.
3. **The restore was unconditionally ON.** `arm_rank` sits at the paper-final freeze, which predates `enable_checkpoint_restore`, so both campaigns paid the −0.0351 AP restore cost — a handicap larger than the effect being chased, and one the clipper never pays. The flag is back-ported (default `True`, so `results/rank` and `results/rankdose` keep their behaviour bit for bit) and the trained arms run with it off.

`selftest.py`'s nine checks pass on the patched file, including *control vs pristine HEAD* at ‖d‖ = 0 — so the back-port did not move the flag-OFF path.

⚠️ Read `rank_w03` vs `clip` first. `rank_w03` vs `rank_ctrl` only says the knob works, which is already known.

**Liveness gate at L50: PASSED, but the dose is not uniform across cells — recorded before any metric is read.** The worry was that `rank_split_size` disables itself when the cap cannot split the pool, and L50 is a cap this arm had never run at. It fires: opening `L_Rank` is 0.065–0.234 at L50, comparable to or larger than L30. But the *number of epochs it acts in* varies by cell:

| cell | epochs with `L_Rank` ≠ 0 |
|---|---|
| MNv3 L30 | 7/7, 7/7, 3/3 |
| **MNv3 L50** | **6/12, 7/13, 4/13** |
| RegNetY L30 | 7/7, 7/7, 7/7 |
| RegNetY L50 | 7/7, 7/7, 7/7 |

The term is gated on the incumbent's step condition — it rides along only in epochs the penalty was already nonzero, and never opens a backward pass of its own (that gate exists so an AP move cannot be confounded with step count). At MNv3/L50 the looser cap is satisfied in about half the epochs, so the treatment receives roughly half the dose it gets elsewhere.

🚨 **Consequence for the read: a null at MNv3/L50 is ambiguous between "the term does not work" and "the term barely ran."** Report the nonzero-`L_Rank` epoch count next to every cell's delta. A cell where the term acted in under half its epochs cannot carry a negative verdict on its own.

#### ⛔ VERDICT: `rank` is a NULL — 48/48, four cells, the clipper in-campaign (2026-08-17)

| comparison | AP | AUROC | cc-F1 | macro-F1 | acc |
|---|---|---|---|---|---|
| `rank_w03` vs **`rank_ctrl`** | **+0.0003** (9/16, p=0.94) | −0.0022 (p=0.86) | −0.0056 (p=0.71) | +0.0061 (p=0.19) | +0.0013 (p=0.61) |
| `rank_w03` vs **`clip`** | **−0.0265** (2/16, **p=0.0006**) | −0.0159 (**p=0.0017**) | −0.0093 (p=0.27) | −0.0062 (p=0.43) | −0.0027 (p=0.30) |

**The term does nothing.** Every metric ties against its own control, and the single-cell dose probe's +0.0323 AP does not replicate — on four cells it is +0.0003. Against the clipper the arm shows the standard TraLO signature: allocation-free metrics a significant loss, every budget-equalized metric a tie.

**The dose caveat does not rescue it, and the check is the clean one.** RegNetY400MF received full dose at both caps (7/7 epochs) and is the backbone with *negative* AP (−0.0045 pooled); MobileNetV3, the half-dose one, supplies the +0.0105. Opposite signs, cancelling, with the term acting *least* where the number looks best. A genuine dose artifact would run the other way.

> 🔑 **What this refutes, precisely.** `rank` is **self-referential**: pairs are (current top-K) vs (rest) on the transductive pool, drawn under `no_grad`, with no labels anywhere. Its only signal is the ordering it is trying to improve, so it can sharpen a cut but cannot correct one that is already wrong. **This kills the term, not the structural direction** — which asks for a per-item objective that can *reorder*, and reordering needs information the model's current ranking does not already contain.
>
> The one place labels exist is the **training set**, and no arm had put a per-item objective there at the cap's operating point. That is `results/rankpair` (`rank_pair`), carried by a plain post-hoc clipper so no constraint-phase defect can confound it.

### ⚠️ CONFIG DEFECT affecting every new-directions arm ever run (2026-08-16)

| corpus | `reset_optimizer_at_sat` |
|---|---|
| `results/headroom` (the bar) | **True**, 240/240 |
| `results/headroom_fix` | **True**, 48/48 |
| `newdirections/arm_steps/results/steps5` | **False**, 72/72 |

The code default is `False` and `gen_steps.py` never set it, so **every ND generator inherited the omission**. All thirteen wave-1 arms and all five wave-2 arms have been testing a TraLO with the component the LOO ablation called *dominant* (+0.079 oct / +0.110 derm) switched off — against a bar that has it on.

- **Within-campaign arm comparisons survive** (both arms share the setting), so the `joint` verdict above stands.
- The ablation that crowned this component ran at **warm-up 50, the dead regime**. `newdirections/arm_steps/gen_reset.py` tests it at warm-up 1: 2 arms × the bar's 4 cells × 4 seeds = 32 runs.

**RESOLVED — the flag is INERT at warm-up 1. Campaign complete: 16 of 16 seed-paired runs bit-identical** by md5, across all four cells (DermMNIST + OctMNIST × MobileNetV3 + RegNetY400MF). The liveness gate caught it; a plain delta would have printed 0.000 and read as a clean null. The mechanism is structural: the reset fires **at first satisfaction**, and at warm-up 1 the model satisfies in ~0.5 of 29 constraint epochs, so the trigger essentially never fires. It was dominant at warm-up 50 only because satisfaction there is 242/242.

> So the omission was harmless and **the ND campaigns were not running a crippled TraLO** — the earlier worry that every arm-vs-bar comparison was invalid is withdrawn. The general lesson stands and is worth more than the specific one: **a component's ablation score is only valid in the regime it was measured in.** This one is dominant in the dead regime and a no-op in the live one — which also means the paper's component-ablation table describes warm-up 50 and should not be quoted as a property of the method.

## 🎯 THE CHECKPOINT RESTORE IS ~83% OF TraLO'S RANKING DEFICIT (2026-08-17)

`newdirections/arm_ortho/results/restoreprobe`, 16 runs, the bar's 4 cells, 4 seeds. Each run scored at BOTH checkpoints — the same run, so nothing is confounded and no arm comparison is involved.

| cell | ΔAP | ΔmacroEq | ΔccF1eq |
|---|---|---|---|
| derm / MobileNetV3 | −0.0523 | −0.0094 | −0.0190 |
| derm / RegNetY400MF | −0.0297 | −0.0353 | −0.0034 |
| oct / MobileNetV3 | −0.0100 | −0.0128 | −0.0031 |
| oct / RegNetY400MF | −0.0485 | −0.0207 | −0.0138 |
| **overall** | **−0.0351** | **−0.0195** | **−0.0098** |

Negative in **all four cells**; 11/16 runs hurt, 2 helped, 3 unchanged. By restore kind: `fully_satisfied` −0.0778 (n=3), `min_excess` −0.0328 (n=10), **`none` exactly 0.0000 (n=3)** — the runs where it never fired are the internal control, and they read 0.0000 to four decimals.

**Set that against the deficit**: TraLO trails the post-hoc clipper by **−0.0425 AP** at equal compute with the same base loss (`focalbar`, 16 cells). The restore accounts for **~83%** of it.

> The mechanism: TraLO finishes training and then **discards its trained model** for an earlier checkpoint selected on *constraint satisfaction*. The clipper has no constraint phase, never restores, and fixes the count post-hoc instead. Both enforce the cap; only one pays for it by throwing away a better-ranked model. **This is a checkpoint-selection policy, not a property of constrained optimisation.**

⚠️ **It is not free.** Native satisfaction reads 0.0625 with the restore and 0.0000 without. It buys feasibility in 1 run of 16 — a poor trade for 3.5 pp of AP, but not nothing. **Read the `sat` row before recommending anything.**

⚠️ **Corrected from the 1-cell estimate**: DermMNIST/MobileNetV3 alone gave −0.0477, which I briefly called "the same number" as the deficit. Four cells give −0.0351. Same shrinkage as focal's +0.0305 → +0.0156. **One cell over-estimates; quote the four-cell figure.**

### ✅ CAPSTONE RESULT — dropping the restore turns the loss into a WIN (2026-08-17)

`arm_ortho/results/norestore`, 48 runs, ONE campaign, equal compute (30 epochs everywhere), CE base loss so only one variable moves, 16 seed-paired cells, 16/16 live. **Control = the post-hoc clipper.**

| arm | AP | ccF1eq | macroEq | native sat |
|---|---|---|---|---|
| **TraLO, restore OFF** | **+0.0085** (9/16) | **+0.0146 (13/16)** | −0.0004 (9/16) | 0.0000 |
| TraLO, restore ON | −0.0266 (5/16) | +0.0048 (9/16) | −0.0199 (5/16) | 0.0625 |

**PAIRED WILCOXON over the 16 seed-matched cells — this is what the claim rests on, not the means above:**

| metric | delta | cells | **p** | per cell (derm/MNv3, derm/RegNet, oct/MNv3, oct/RegNet) |
|---|---|---|---|---|
| **ccF1eq** | **+0.0146** | 13/16 | **0.0075** | **+0.0121, +0.0017, +0.0154, +0.0292 — positive in ALL FOUR** |
| AP | +0.0085 | 9/16 | **0.5619** | −0.0266, −0.0253, +0.0273, +0.0585 |
| macroEq | −0.0004 | 9/16 | 0.8603 | split |

🛑 **THE AP "WIN" IS NOT REAL. Do not claim it.** p=0.56, and per cell it is a **dataset split, not a method effect**: `restore_off` LOSES on AP on both DermMNIST cells and wins on both OctMNIST cells. Averaging opposite signs across datasets produced a positive number that means nothing — the exact "never pool across cells" trap this file already warns about, reproduced by me.

✅ **The cc-F1 result is real**: +0.0146, **p=0.0075**, 13/16 cells, and **positive in all four cells** — not carried by one dataset. This is the metric the paper leads with, and it is the only defensible claim from this campaign. macro-F1 is a clean tie (p=0.86).

**One line flips the headline.** With the restore on, TraLO loses to plain CE + clipping on AP and macro-F1, which is what this project has believed since 2026-08-16. With it off, TraLO **beats the clipper on cc-F1 at equal compute, significantly and in every cell**, and ties on macro-F1.

⚠️ **3 of 16 `restore_off`/`restore_on` pairs are bit-identical** — the runs where the restore never fired, so the flag correctly did nothing. Effective n for the restore contrast is **13, not 16**.

### ⚠️ THE WIN IS BASE-LOSS DEPENDENT — `focalnorestore` (2026-08-17)

The same three arms with **focal** as the base loss, 48 runs, 16 cells:

| vs `focal_clip` | ccF1eq | AP | macroEq |
|---|---|---|---|
| focal + TraLO, restore OFF | −0.0123 (3/16), p=0.13 | **−0.0318 (3/16), p=0.013** | +0.0033 (8/16), p=0.86 |
| focal + TraLO, restore ON | −0.0156 (3/16), p=0.023 | **−0.0425 (2/16), p=0.0017** | −0.0072 (4/16), p=0.23 |

**With focal, TraLO loses to the clipper even with the restore off.** Absolute 16-cell means across both capstones:

| base | arm | AP | **ccF1eq** | macroEq | native sat |
|---|---|---|---|---|---|
| CE | clip | 0.7206 | 0.4076 | 0.7068 | 0.0000 |
| CE | **TraLO restore-off** | 0.7291 | **0.4222** ← best cc-F1 | 0.7064 | 0.0000 |
| CE | TraLO restore-on | 0.6940 | 0.4124 | 0.6868 | 0.0625 |
| focal | **clip** | **0.7323** ← best AP | 0.4154 | 0.6958 | 0.0000 |
| focal | TraLO restore-on | 0.6898 | 0.3998 | 0.6886 | **0.2500** |
| focal | TraLO restore-off | 0.7005 | 0.4031 | 0.6991 | 0.1875 |

> **focal helps the clipper and hurts TraLO.** Clipper cc-F1 0.4076 → 0.4154; TraLO restore-off cc-F1 0.4222 → 0.4031. So "why not just use focal + clipping?" is a real reviewer question, and on **AP** focal+clip (0.7323) is the best number anywhere.

**On cc-F1 the best configuration anywhere is still CE + TraLO + no restore (0.4222) vs focal+clip (0.4154)** — but that is **cross-campaign**, and +0.0068 is exactly the size the measured 0.027 drift could manufacture. **`arm_baseloss/gen_headtohead.py` settles it**: `ce_tralo_norestore` vs `focal_clip` vs `ce_clip`, all three in ONE campaign, 48 runs. Until it reports, do not claim TraLO beats the best clipper.

**The two experiments agree exactly.** restore_on 0.6940 → restore_off 0.7291 is **0.0351 AP** — the same figure `restoreprobe` measured independently, within-run, on the same four cells. An arm-level comparison and a within-run comparison landing on the same number is the strongest evidence produced in this whole campaign series.

⚠️ **AP's margin is thin**: +0.0085 at 9/16 cells is barely above chance on cell count even though the mean clears the bar. **Lead with cc-F1 (13/16), not AP.**

⚠️ **It costs the last of the native satisfaction** (0.0625 → 0.0000). At warm-up 1 both are ~zero so little is lost in practice, but the restore was the method's only route to feasibility without post-hoc adjustment, and that claim cannot be made for `restore_off`.

`enable_checkpoint_restore` added to `tralo/train.py` in `arm_ortho` and `arm_baseloss`, default **True** so every existing config and completed campaign is bit-for-bit unchanged.

## 🔬 THE FULL METRIC PANEL — what the win is, and what it is not (2026-08-17)

`paper/scripts/full_panel.py`, run on `norestore` (CE, 4 cells × 4 seeds), control = clipper. Sixteen metrics in three families, each family gameable in a different way. **This is the honest ledger; quote it, not a single cherry-picked row.**

| family | metric | clip | TraLO no-restore | Δ | cells | p | verdict |
|---|---|---|---|---|---|---|---|
| allocation-free | AP | 0.7209 | 0.7291 | +0.0082 | 9/16 | 0.56 | tie |
| allocation-free | AUROC | 0.9015 | 0.8981 | −0.0034 | 8/16 | 0.71 | tie |
| allocation-free | ECE | 0.1407 | 0.1461 | +0.0054 | 5/16 | 0.16 | tie |
| allocation-free | Brier | 0.3142 | 0.3252 | +0.0110 | 6/16 | 0.21 | tie |
| allocation-free | **NLL** | 1.1003 | 1.2421 | **+0.1418** | 3/16 | **0.0092** | **LOSS** |
| allocation-free | ConfGap | 0.1032 | 0.0990 | −0.0041 | 5/16 | 0.40 | tie |
| equalized | **ccF1** | 0.4076 | 0.4222 | **+0.0146** | 13/16 | **0.0075** | **WIN** |
| equalized | macroF1 | 0.7068 | 0.7064 | −0.0004 | 9/16 | 0.86 | tie |
| equalized | acc | 0.7882 | 0.7923 | +0.0041 | 10/16 | 0.21 | tie |
| as-run | **raw count / K** | **2.3505** | **1.7087** | **−0.6418** | 13/16 | **0.0027** | **WIN** |
| as-run | **flips** | 95.06 | 51.13 | **−43.94** | 13/16 | **0.0027** | **WIN** |
| as-run | native sat | 0.0000 | 0.0000 | 0 | — | — | neither ever satisfies |

### 🚨 ccP, ccR and ccF1 are ONE metric, not three

With the budget pinned to exactly K: `ccP = TP/K`, `ccR = TP/n_pos`, `ccF1 = 2TP/(K + n_pos)`. All three are monotone in the same TP count, which is why they return p = 0.0075 / 0.0076 / 0.0075 — **the same test three times.** Reporting "precision, recall *and* F1 all improve" would be presenting one result as three corroborating ones. **Quote one.**

### The AP tie and the cc-F1 win are not in conflict — the gain is at the head of the ranking

`paper/scripts/pak.py` sweeps the budget and counts true positives in the top-k:

| budget | 0.125K | 0.25K | 0.5K | **1K** | 1.5K | 2K | **4K** | **8K** |
|---|---|---|---|---|---|---|---|---|
| Δ TP-fraction | +0.037 | +0.030 | +0.031 | **+0.027** | +0.022 | +0.021 | **−0.009** | **−0.009** |
| p | 0.11 | 0.11 | 0.050 | **0.036** | 0.044 | 0.052 | 0.55 | 0.065 |

The advantage decays monotonically and **changes sign between 2K and 4K**. AP integrates every threshold, so a gain at the head cancels against a loss in the tail and reads as a tie. **The constraint reallocates ranking precision from the tail into the head — and the cap only ever reads the head.**

⚠️ Do **not** state this as "sharpest exactly at the cap". The peak is at the smallest budgets and K sits inside the positive region; it is not a peak at K. The small-k columns are also noisy (k = 8 items at 0.125K).

### What is genuinely on our side

1. **TP inside the budget** — +0.0146 cc-F1, p = 0.0075, 13/16 pairs. **This is the only quality claim.**
2. **No collateral damage** — macro-F1, accuracy, macro-P and macro-R are all clean ties, so the constrained class does not improve at the other classes' expense.

> 🛑 **`flips` and `raw count / K` were listed here and have been struck.** Post-hoc adjustment fills to the constraint boundary **for free** at the end of every pipeline, so an arm that needs fewer flips has bought nothing — the operation it avoided was free. A small p-value on a free operation is still nothing. When quality ties, the honest report is *"this arm produced nothing"*. See the rule at the top of this section.

### What is against us — state these unprompted

1. **NLL is significantly worse** (+0.142, p = 0.0092, worse in all four cells, driven by derm/MobileNetV3 at +0.40) while ECE and Brier tie. NLL is the calibration metric most sensitive to *confident* errors, so TraLO is making a few high-confidence mistakes it did not make before. **There is no calibration claim here.**
2. **AP and AUROC are ties**, and AP is a per-cell **dataset split** (derm −0.027/−0.025, oct +0.027/+0.058). Never average it into one number.
3. **Native satisfaction is 0.0000 for both arms.** No feasibility claim survives at warm-up 1.
4. ⚠️ **Per cell, cc-F1 is +0.0121 / +0.0017 / +0.0154 / +0.0292.** Positive in all four, but derm/RegNetY400MF at +0.0017 is **below this project's own 0.005 materiality threshold** — so it is honestly **3 wins and 1 tie**, not 4 wins.

## 🛑 HEAD-TO-HEAD: the cc-F1 win DOES NOT survive against focal + clipping (2026-08-17)

`results/headtohead`, 48 runs, **all three arms in ONE campaign**, 16/16 cells live by md5, equal compute. This was run specifically because the +0.0068 cc-F1 edge over focal+clip was **cross-campaign** and the measured drift is 0.027. The drift explanation wins.

**`ce_tralo_norestore` vs `focal_clip`:**

| metric | focal_clip | TraLO no-restore | Δ | cells | p | |
|---|---|---|---|---|---|---|
| **cc-F1** | 0.4157 | 0.4222 | +0.0064 | 11/16 | **0.21** | **TIE — not a win** |
| macro-F1 | 0.6956 | 0.7064 | +0.0107 | 9/16 | 0.23 | tie |
| AP | 0.7323 | 0.7291 | −0.0032 | 9/16 | 0.94 | tie |
| AUROC | 0.9132 | 0.8981 | −0.0150 | 2/16 | **0.0021** | **LOSS** |
| **ECE** | **0.0804** | 0.1461 | +0.0657 | **0/16** | **<0.0001** | **LOSS** |
| Brier | 0.2811 | 0.3252 | +0.0441 | 4/16 | **0.0076** | **LOSS** |
| **NLL** | **0.5817** | 1.2421 | +0.6604 | **0/16** | **<0.0001** | **LOSS** |
| ConfGap | 0.1756 | 0.0990 | −0.0766 | 0/16 | **<0.0001** | **LOSS** |
| **raw count / K** | 2.6453 | **1.7087** | **−0.9366** | 13/16 | **0.0006** | **WIN** |
| **flips** | 115.81 | **51.13** | **−64.69** | 13/16 | **0.0016** | **WIN** |

**cc-F1 ranking: TraLO 0.4222 > focal_clip 0.4157 > ce_clip 0.4076 — but only the gap to `ce_clip` is significant.** Against the strongest clipper it is a tie.

**focal is a strong and well-calibrated baseline.** `focal_clip` beats `ce_clip` on cc-F1 (+0.0082) and *crushes* it on calibration (ECE 0.080 vs 0.141, NLL 0.58 vs 1.10, both p<0.0001). Any calibration table in the paper must include it.

**The head-of-ranking effect survives but weakens.** vs `focal_clip`: +0.018 at K (p=0.078), +0.023 at 2K (p=0.044), then **−0.015 at 8K at p=0.0016 — a significant tail loss.** focal already captures most of the head gain, without the tail cost and without the calibration damage.

### 🛑 What survives against BOTH clippers: NOTHING

Against `focal_clip` every quality metric is a tie or a loss. The only columns with small p-values were `flips` (51 vs 116) and `raw count / K` (1.71 vs 2.65) — **and those are not metrics.**

> **Post-hoc adjustment fills to the constraint boundary for free at the end of every pipeline.** An arm that arrives closer to the boundary has saved a free operation, which is worth exactly zero. There is no weaker phrasing that survives: not "proximity to feasibility", not "less post-hoc surgery", not "closer to the constraint natively" — all the same rejected metric renamed. **A significant p-value on a free operation is still nothing.**

**So the honest verdict on the whole no-restore line is: beats CE+clip on cc-F1, TIES the strongest clipper, loses on calibration. It has not yet produced a result.** The reviewer question *"why not just use focal + clipping?"* currently has no answer.

⚠️ This is the failure mode to watch for: when quality ties, `flips` is reliably the one column with a small p-value, and it is reliably reached for as the surviving result. It has now happened enough times that the scorer refuses to print a verdict for it (`full_panel.py` hard-codes `(not a result)`).

⚠️ `focal_clip` was therefore **added as a third arm to every replication slice** (chained into the same campaign roots, `--arms focal_clip`, so the three-way comparison never crosses a campaign boundary). Testing cap generalisation only against `ce_clip` would replicate the win against the weaker baseline.

### The replication that decides whether any of this is a paper (launched 2026-08-17)

All four cells above are **L30_G30**. focal already died at exactly this test (11/16 cells at L30, 4/8 elsewhere), so the cap axis is the most likely way for this to be an artifact. `arm_baseloss/gen_replicate.py`, two arms (`tralo_norestore`, `clip`), 176 runs, three slices of whole cells so every paired comparison stays inside one lane on one card:

| slice | cells | runs | where |
|---|---|---|---|
| `caps_derm` | dermmnist × {MNv3, RegNetY} × {L20, L40, L50} | 48 | dsisco01 GPU0 |
| `caps_oct` | octmnist × {MNv3, RegNetY} × {L20, L40, L50} | 48 | dsisco01 GPU1 |
| `wide` | tissuemnist × {MNv3, RegNetY} × {L20, L30, L40}, plus {derm, oct} × {MNv2, ShuffleNetV2} × L30 | 80 | dsisco02 GPU1 |

The cap is a **fraction of the true positive count**, not an absolute (derm L30 → K=67 against n_pos=223), so these levels genuinely move the budget. `constraint` is not in `base_model_id`, so both caps slices reuse warm-ups already cached by `norestore` — verified, 32/32 present.

## ⛔ REJECTED: the no-restore line. The L30 cc-F1 win was a CAP ARTIFACT (2026-08-17)

`results/repcaps_derm`, 72 runs, dermmnist × {MobileNetV3, RegNetY400MF} × **{L20, L40, L50}** × 4 seeds × 3 arms, one campaign.

**`tralo_norestore` vs `clip` (the same comparison that gave +0.0146 at p=0.0075 on L30):**

| metric | Δ | cells | p | |
|---|---|---|---|---|
| **cc-F1** | **−0.0098** | 10/24 | 0.14 | **TIE — the win is gone** |
| macro-F1 | −0.0117 | 9/24 | **0.016** | **LOSS** |
| accuracy | −0.0038 | 5/24 | **0.0155** | **LOSS** |
| AP | −0.0214 | 4/24 | **0.0043** | **LOSS** |
| AUROC | −0.0130 | 6/24 | **0.0004** | **LOSS** |
| ECE / Brier / NLL / ConfGap | all worse | 1–3/24 | **<0.0001** | **LOSS** |

**cc-F1 goes from +0.0146 (p=0.0075) at L30_G30 to −0.0098 (p=0.14) at L20/L40/L50, and per cell it changes sign** (derm/MNv3 +0.0121 → −0.0150; derm/RegNetY +0.0017 → −0.0046). Everything else is a significant loss.

**Against `focal_clip` it is worse still:** every quality metric ties (cc-F1 −0.0014 p=0.71, macro-F1 +0.0032 p=0.56) while **every allocation-free metric loses at 0/24 cells** — AUROC, ECE, Brier, NLL and ConfGap are all clean sweeps at p<0.0001.

**Confirmed on the second dataset.** `results/repcaps_oct`, 72 runs, octmnist × {MobileNetV3, RegNetY400MF} × {L20, L40, L50}: **every single metric is a tie.** cc-F1 +0.0072 (p=0.14), macro-F1 −0.0088 (p=0.33), AP +0.0101 (p=0.22), accuracy −0.0098 (p=0.25), all six allocation-free metrics tie. So away from L30 the arm is a tie on octmnist and a loss on dermmnist — on neither dataset is it a win.

> **This is the focal failure pattern, exactly.** focal looked like a result at 11/16 cells on L30 and collapsed to 4/8 elsewhere. The no-restore arm looked like a result at 13/16 cells on L30 and reverses sign elsewhere. **A result that exists only at one cap level is a cap artifact, not a method.** Four cells at one cap could never have distinguished the two, which is why the replication had to be run before anything was claimed.

⚠️ **Standing rule this earns:** never claim a result from cells that share a single cap level. The cap must be swept before a number is quoted, not after.

The restore *mechanism* finding survives untouched — disabling the restore really is worth +0.0351 AP against the restore-on arm, measured twice independently. It simply does not lift TraLO to the level of a clipper.

## 🔴🔴 ROOT CAUSE: the constraint gets 29 optimizer steps and Adam makes λ and the clip NO-OPS (2026-08-17)

Read from the multi-class training logs, which nobody had opened. **This is the mechanism behind every tie in this project.**

### The constraint never satisfies — not once

Across **all 15** multi-class TraLO runs: `Global_Satisfied = 0` and `Local_Satisfied = 0` at **every logged epoch of every run**. Meanwhile the penalty explodes and λ ratchets to its ceiling:

| cap | arm | final CE | L_global | L_local | λ_final | ever satisfied |
|---|---|---|---|---|---|---|
| L50 | tralo_uniform | 0.0062 | **96.7** | **284.1** | 1.31 | **0/3** |
| L50 | tralo_byk | 0.0121 | 63.2 | 188.7 | 1.40 | **0/4** |
| L70 | tralo_uniform | 0.0154 | 15.7 | 47.9 | 1.01 | **0/4** |
| L70 | tralo_byk | 0.0046 | 13.3 | 41.3 | 1.02 | **0/4** |

A representative trajectory (derm L70, caps 72/154/156) — the counts **oscillate and never converge**: C2 goes 134 → 214 → 199 → 187 → 187 → 159 → **204** against a limit of 154, while train accuracy climbs to 0.9993 and CE falls to 0.0036.

### Why: the step budget

`src/methodologies/tralo/train.py` — the constraint block calls `zero_grad()` **once**, accumulates gradients over every test chunk, then takes **one** `clip_grad_norm_(max_norm=1.0)` + `optimizer.step()`. Per epoch:

| | steps over the 29-epoch constraint phase |
|---|---|
| CE | 29 × 126 batches = **3,654** |
| constraint | **29** |

### 🚨 And Adam makes both existing knobs no-ops

The optimizer is Adam, whose update is `m/√v`. **Scaling the gradient by any constant leaves the step unchanged.** Therefore:

- **the unit-norm clip does not limit step size** — it only fixes the scale that Adam was going to normalise anyway;
- **the λ ratchet does not increase pressure** — ratcheting 0.03 → 1.40 changes *direction* (relative weighting across classes and groups) but **not step magnitude**.

Each constraint step moves ≈ `lr` in parameter space no matter what. **29 × 1e-4 is a hard ceiling on how far the constraint can move the model**, and it is nowhere near enough to fix a three-class count violation. The exploding penalty is a *symptom of nothing moving*, not evidence of pressure being applied.

> **This reframes the whole project.** The count constraint was never really being optimised — it was being *asked* 29 times, weakly, on a representation that CE had already saturated by epoch ~10. Thirteen wave-1 arms varied the penalty *shape*, which under Adam scale-invariance is one of the few things that genuinely cannot matter.

⚠️ **This also invalidates any interpretation of λ trajectories as "pressure".** λ rising means *satisfaction never came*, nothing more. The paper's own line calling λ escalation "a symptom, not the damage" is correct and now has a mechanism.

### It retro-explains `joint`

`joint_objective` puts the penalty inside every CE minibatch → **~126× more constraint steps**. That is exactly why it held the cap **98.8%** of epochs where the incumbent holds it ~0%. Its single-class rejection (overfitting, −0.067 AP) carries **no information about the multi-class case**, because on single-class the cap was trivially satisfiable and clipping was optimal anyway.

### The experiment now running: `arm_joint/results/mcjoint`, 96 runs

`tralo` vs `tralo_joint` differ **only** in step structure (both `by_k`, both `enable_checkpoint_restore=False`), against `clip` and `focal_clip`, in one campaign. Harder settings throughout: **tighter caps L30/L50** (not L70), **tissuemnist (8 classes)**, and **5 simultaneously capped classes** on tissue.

| dataset | capped | caps @L30 | oracle acc @L30 | oracle acc @L50 |
|---|---|---|---|---|
| octmnist | {0,1,2} | 75/75/75 | 0.4750 | 0.6250 |
| dermmnist | {1,2,4} | 30/66/66 | 0.8083 | 0.8632 |
| tissuemnist | **{1,2,3,4,5}** | 33/25/66/51/33 | 0.7933 | 0.8525 |

All caps ≥ 25, so none rounds to zero (a zero cap is **skipped** in the loss — a known open defect that would silently unenforce it).

## 🔴🔴🔴 WHY ~20 ARMS ALL TIED: the pipeline's score is a function of the RANKING, and nothing else (2026-08-17)

Read `src/utils/posthoc_adjustment.py` rather than assuming. The greedy path — which runs in **55 of 59** granularity runs, the LP fallback firing in 4 — is:

```python
# phase 1, class over its limit: drop the LEAST confident members
sorted_idx = indices[np.argsort(  y_proba[indices, c])]
# phase 2, class under its limit: admit the MOST confident outsiders
sorted_idx = candidates[np.argsort(- y_proba[candidates, c])]
```

Composed, that is *thresholding the ranking induced by `p[·, c]` at the budget*. And when greedy cannot finish, `_fallback_lp` minimises `c_obj[i,k] = -p[i,k]` subject to one-class-per-item plus the count inequalities — i.e. it returns **argmax over feasible labelings of Σᵢ p[i, yᵢ]**, the maximum-probability feasible assignment. **Both paths are pure functions of `p`, and both are good ones.**

### The literal top-K identity is FALSE — checked, not assumed

`paper/scripts/topk_identity.py` rebuilds top-K from the stored probabilities and compares against the labels actually shipped:

| | |
|---|---|
| shipped set == global top-K | **16 / 59 (27.1%)** |
| mean Jaccard | 0.892 (min 0.684) |

The **local per-group caps** are what break it: a high-`p` item sitting in an already-full group cannot be admitted. So the allocation is genuinely more constrained than a global threshold.

### But it is score-IDENTICAL, which is what actually matters

| | |
|---|---|
| ccF1(global top-K) − ccF1(shipped) | **+0.0017, Wilcoxon p = 0.42 — tie** |
| top-K better in | 18 / 59 |

**The local constraints reshuffle ~11% of the selected set and change the score by nothing.** So the correct claim is not "post-hoc is top-K" (false) but **"post-hoc's score is determined by the ranking"** (true, and the allocation subtleties are noise on top).

Direct confirmation — within cell, across arms *and* seeds:

**mean spearman(AP, ccF1) = +0.534, positive in 7/7 cells.**

### Why this is the whole explanation

At a fixed budget K, `ccF1 = 2·TP/(K + n_pos)` where TP is the true positives among the K selected. That is **precision@K, rescaled by a constant**. And **AP is the integral of precision@k over all k**. cc-F1 and AP are the same object read at one point versus averaged — which is exactly why the `pak.py` sweep found Δ(TP-fraction) decaying monotonically and *flipping sign between 2K and 4K*, so AP's integral cancels while cc-F1 at the head can still move.

So the chain closes:

1. The pipeline's output, and therefore every metric that decides an arm, depends on `p` **only through the ordering of `p[·, c]`**.
2. Probability *values* are discarded — which is why **no calibration claim is available even in principle**, and why NLL/ECE losing is a symptom rather than a cost.
3. A method can therefore only win by **improving the ranking**.
4. A count constraint says how many, never which: it supplies **no per-item signal** with which to reorder anything.
5. Measured, it does not merely fail to help — it **degrades** the ranking: AP −0.020 (granularity), −0.024 to −0.049 (headroom, restore probe), significant in every campaign.

**⇒ A loss whose gradient is a function of the aggregate count cannot beat post-hoc adjustment. Not incidentally — structurally.** Thirteen wave-1 arms varied the penalty *shape*; the granularity sweep varied the *number* of counts; both are moves inside the family this argument excludes.

### What it licenses, and what it forbids

- ❌ **Forbidden**: any new arm whose gradient is a function of counts (global, per-group, per-class, or any penalty shape over them). Also forbidden: reading `flips`, `sat`, or `cnt/K` as results — post-hoc re-thresholds at exactly K regardless, so those measure only how far the model's own threshold sat from K, which is corrected for free.
- ✅ **Required of any winning arm**: it must move **precision@K / AP**, which means a **per-item** objective at the operating point. Ranking or margin losses on the constrained class qualify; count penalties do not.

⚠️ The one genuine escape hatch: the argument assumes post-hoc is (near-)optimal given `p`. It is optimal over its *candidate neighbourhood*, not globally, and the multi-class + local-group case is where greedy is weakest — that is exactly where the LP fallback exists. **Multi-class coupled caps remain the one setting where "post-hoc is already optimal" is not obviously true**, which is why `mcjoint`/`beta` are still worth running.

## 🚨 SCORER BUG: every multi-class campaign was scored as if ONE class were capped (2026-08-17)

Found while auditing the code rather than the arms. **Two bugs in `paper/scripts/full_panel.py`, both mine**, both invalidating multi-class results:

1. `cls = int(cls[0] if isinstance(cls, (list, tuple)) else cls)` — **only the FIRST capped class was measured.** AP, AUROC and ccP/ccR/ccF1 described class 1 (derm), class 0 (oct) or class 1 (tissue) while being reported as the campaign's result.
2. `_sa.equalize(P, g, G, L, cls)` is **single-class by construction**: it takes top-K for one class, then assigns everything else by `argmax` over the remaining classes **with no budget check**. So in a multi-class run the "budget-equalized" allocation freely violated caps 2..n — meaning macro-F1 and accuracy were never equalized across arms, and any arm difference mixed in an allocation difference.

Fixed with `equalize_multi`: all (item, capped class) pairs in descending probability, assigned while the class has global **and** local room, leftovers taking their best class that still has room. Deterministic, feasible, identical for every arm. **With one capped class it routes to the original `equalize`, so single-class campaigns (granularity, headroom, budgetprobe, replication) are bit-identical and their verdicts stand.** Per-class metrics now average over the capped classes; count diagnostics sum over them.

### The correction it forces — mcjoint `tralo` vs `clip`

| metric | buggy (1 cap equalized) | **corrected (all caps)** |
|---|---|---|
| macroF1 | −0.0272, p=0.0010 **LOSS** | **+0.0115, p=0.15 — tie** |
| acc | −0.0169, p=0.034 **LOSS** | **+0.0018, p=0.82 — tie** |
| ccF1 | −0.0110, p=0.33 | −0.0070, p=0.17 — tie |
| NLL | +0.3109, p=0.043 **LOSS** | +0.1506, p=0.19 — n.s. |
| ECE | +0.0185, p=0.034 **LOSS** | +0.0118, p=0.17 — n.s. |
| Brier | +0.0352, p=0.034 **LOSS** | +0.0236, p=0.17 — n.s. |
| AP | −0.0274, p=0.0068 | −0.0212, p=0.022 **still LOSS** |
| AUROC | −0.0184, p=0.0034 | −0.0099, p=0.0024 **still LOSS** |

⛔ **"TraLO loses to plain clipping on every metric" is RETRACTED.** The significant macro-F1 / accuracy / calibration losses were artifacts of equalizing one cap of three. The corrected verdict is narrower and cleaner: **TraLO costs RANKING (AP, AUROC, ConfGap — all still significant) and ties on every budget-equalized quality metric.** That is the same signature seen everywhere else, now measured properly on coupled multi-class caps.

### `results/multiclass`, corrected (4 cells, 8 pairs)

| arm | AP | AUROC | ECE | NLL | ccF1 | macroF1 |
|---|---|---|---|---|---|---|
| **focal_clip** | +0.0106 | **WIN** | **−0.0440 (8/8)** | **−0.4600 (8/8)** | tie | tie |
| tralo_byk | −0.0227 | **LOSS** | **LOSS** | **LOSS** | −0.0189 lean loss | +0.0067 lean win |
| tralo_uniform | −0.0181 | **LOSS** | **LOSS** | **LOSS** | **−0.0152 LOSS** | tie |

🚨 **`focal_clip` is live here** (unlike `arm_joint`) and is a genuinely strong baseline — a calibration rout over plain clipping at 8/8 cells. TraLO loses ranking and calibration to both.

### ⚠️ Related, and NOT a problem for these numbers: the two-allocator confound

`src/pipeline/eval.py` applies `targeted_correction` (argmax + repair) to TraLO, while `heuristic`/`danits_lp` set `skip_targeted_correction=True` and ship their own `apply_allocation_heuristic` (allocate class-by-class from scratch). **The pipeline's own `evaluation_metrics.csv` therefore compares arms through two different allocators.** It does not contaminate anything scored by `full_panel.py`, because that re-allocates *every* arm through one equalizer and takes allocation-free metrics from probabilities only. **Read campaign results from `full_panel.py`, never from `evaluation_metrics.csv`.**

## 🛑 MULTI-CLASS (SUPERSEDED — see the scorer-bug correction directly above)

`arm_joint/results/mcjoint`, 3 complete cells (derm L30, derm L50, **tissuemnist L30 with FIVE simultaneously capped classes**), 12 paired seeds, one campaign. This is the setting the coupled-assignment argument nominated as the one real opening.

**`tralo` vs `clip`:**

| metric | Δ | p | verdict |
|---|---|---|---|
| AP | −0.0274 | 0.0068 | LOSS |
| AUROC | −0.0184 | 0.0034 | LOSS |
| NLL | +0.3109 | 0.0425 | LOSS |
| ccF1 | −0.0110 | 0.33 | lean loss |
| macroF1 | −0.0272 | 0.0010 | LOSS |
| acc | −0.0169 | 0.0342 | LOSS |

macroP is a lone +0.0161 "WIN" cancelled by macroR −0.0424; macro-F1 is an unambiguous loss. **`sat = 0.0000`** — TraLO never satisfies natively here either. On tissuemnist's five coupled caps specifically: ccF1 −0.0068, macroF1 −0.0245.

**`tralo_joint` vs `clip` is catastrophic**: AP **−0.2416**, AUROC −0.0832, ECE +0.1754, all p = 0.0005 at 0/12 cells. Consistent with the count-crushing/thrashing mechanism seen in the derm logs, now confirmed on tissue.

⚠️ **This is NOT yet a verdict on the multi-class idea.** In these exact runs the constraint step was measured at **~1.4% alignment with its own gradient** (see the audit below), so the constraint phase was barely acting. A broken optimiser produces a null that looks identical to a refuted hypothesis. `results/sepopt` is the clean re-test.

⚠️ **`focal_clip` is INERT in `arm_joint`** — all 16 metrics differ from `clip` by exactly 0.0000. That worktree has no `base_loss` support, so the flag never took effect and the arm is a duplicate `clip`. **Do not read focal comparisons from this campaign**; the real focal baseline lives in `arm_multiclass`/`arm_budget`.

## ✅✅ THE WIN REPLICATES ON A SECOND BACKBONE — and is LARGER there (2026-08-18)

`results/mcbar_regnet`, RegNetY400MF, same grid as `mcbar`: 3 datasets x {L30, L50} x 4 seeds = 24 pairs, 6 cells. `tralo_uniform` vs a live `focal_clip`:

| metric | RegNetY400MF | MobileNetV3 (`mcbar`) |
|---|---|---|
| **macro-F1** | **+0.0253**, 18/24, **p=0.0008** | +0.0102, 18/24, p=0.0087 |
| **accuracy** | **+0.0392**, 17/24, **p=0.0105** | +0.0130, 19/24, p=0.0023 |
| macro-P | +0.0248, 18/24, p=0.0053 | +0.0142, p=0.0340 |
| macro-R | +0.0197, 17/24, p=0.0065 | +0.0084, p=0.0315 |
| cc-F1 | +0.0071, 17/24, p=0.136 (lean win) | +0.0020, tie |
| AP | **+0.0074, tie** | −0.0109, lean loss |
| AUROC | **+0.0016, tie** | −0.0120, *** LOSS |
| ECE / NLL | +0.0712 / +0.5469, 0/24, p=0.0000 | +0.0777 / +1.2762, *** LOSS |

**Two backbones, three datasets, two cap levels, 48 matched pairs: macro-F1 and accuracy are significant wins in both.** On RegNetY the effect is ~2.5x larger, and **the ranking cost disappears** — AP and AUROC both become ties, where MobileNetV3 showed a significant AUROC loss. cc-F1 moves from tie to lean win.

⚠️ **Calibration is a significant loss on BOTH backbones** (ECE +0.07, NLL +0.55 to +1.28, 0/24 cells, p=0.0000). This is the one cost that replicates as reliably as the win. There is no calibration claim, on either backbone.

**Validation in flight** (`mcbar_mnv2`, `mcbar_shuffle`, `mcbar_caps`, 72 runs each): MobileNetV2 and ShuffleNetV2 on the same grid, plus MobileNetV3 at **L20 and L70** to bracket the cap axis. The cap axis is not optional — a one-cap-regime artifact has been retracted here three times, and the cc-F1 finding this campaign replaced was the third.

## ⛔⛔⛔ RETRACTION: "TraLO beats the strongest clipper" IS WRONG — the numbers reproduce, the framing does not (2026-08-18)

An independent re-implementation (no import of `src/` or `paper/scripts/`, allocator written from spec) reproduced **every digit** of the headline: mcbar macro-F1 +0.0102 18/24 p=0.0087, accuracy +0.0130 19/24 p=0.0023; regnet macro-F1 +0.0253 18/24 p=0.0008, accuracy +0.0392 17/24 p=0.0105. **The arithmetic is not the problem.** Three things about the framing are.

### 0a. ⚠️ WARM-UP 1 DELAYS SATURATION, IT DOES NOT PREVENT IT — and the "room to move the boundary" prediction comes out BACKWARDS

Warm-up 1 was adopted because warm-up 50 saturates CE before the constraint phase starts. But CE keeps training for all 29 constraint epochs, and it saturates anyway — just later. Measured on `results/mcbar`, 24 TraLO runs, first epoch with train-acc >= 0.995:

| dataset | runs reaching 0.995 | median epoch | TraLO vs `clip` macro-F1 | cc-F1 |
|---|---|---|---|---|
| dermmnist | **8/8** | **15** — half the constraint phase is frozen | **+0.0166, 6/8, p=0.078 lean win** | −0.0025 tie |
| octmnist | 4/8 | 25 | +0.0064, 5/8 tie | +0.0042 tie |
| tissuemnist | **3/8** | **30 — barely saturates at all** | **−0.0164, 1/8 lean loss** | **−0.0122, p=0.039 LOSS** |

🛑 **The dataset with the MOST non-saturated constraint training is where TraLO does WORST.** The hypothesis that a live CE gradient during the constraint phase is what enables a win predicts the exact opposite ordering.

⚠️ **Confounded, so this is evidence and not a refutation.** n=3 datasets, and saturation timing is not independent of difficulty: tissuemnist saturates late *because* it is hard (train acc starts 0.583, AP 0.32 vs dermmnist's 0.78). Difficulty and saturation move together here.

🎯 **The clean test is WITHIN one dataset**: hold dataset, backbone, epoch budget and cap fixed and manipulate only whether CE saturates. The lever is **data augmentation** — an open gap in this pipeline (`audit_findings.md`), applied identically to every arm so equal compute survives. On dermmnist it should hold train accuracy off 0.995 for all 29 constraint epochs. Prediction to falsify: if the mechanism is real, dermmnist's lean win GROWS with augmentation; if saturation is irrelevant, it does not move.

✅ **Contract verified across every live campaign** (`mcbar`, `mcbar_mnv2`, `mcbar_caps`): TraLO arms are `warmup_epochs=1`, `constraint_epochs=29`, `enable_ce_skip=False`, `lr_constraint == lr == 1e-4`; clippers are 30 + 0. **Nothing in this session ran at warm-up 50** — the retraction in section 0 is a warm-up-1 result.

### 0. THE SETTLING FACT — a baseline that does NO constraint training beats `focal_clip` by MORE than TraLO does

Two independent audits, different methods, same conclusion. Pooled over both campaigns, 48 seed-pairs:

| comparison | macro-F1 | p | accuracy | p |
|---|---|---|---|---|
| `tralo_uniform` vs `focal_clip` | +0.0178 | 0.0000 | +0.0261 | 0.0001 |
| **`clip` vs `focal_clip`** | **+0.0194** | **0.0002** | **+0.0288** | **0.0000** |
| **`tralo_uniform` vs `clip`** | **−0.0017** | **0.478 TIE** | −0.0027 | **0.323 TIE** |

**Plain CE + clipping beats focal + clipping by more than the entire method does.** Against that baseline TraLO is a tie on macro-F1/accuracy and a **significant loss on everything else in the same run**: cc-F1 −0.0052 (p=0.011), AP −0.0106 (p=0.007), AUROC −0.0054 (p=0.002), ECE +0.0094 (p=0.028), Brier +0.0209 (p=0.014), NLL +0.1307 (p=0.001), macro-R −0.0105 (p=0.003), ConfGap −0.0068 (p=0.011). Per cell: TraLO beats `focal_clip` in 11/12 but `clip` in only **5/12** (p=0.58), and **`clip` wins all four tissuemnist cells** — the very cells said to carry the effect.

⛔ **THE MECHANISM CLAIM DIES WITH IT.** Decomposing macro-F1 into capped vs uncapped classes:

| | vs `focal_clip` | vs `clip` |
|---|---|---|
| F1 on **uncapped** classes | +0.0356, p=0.0000 | **+0.0010, p=0.90 — dead tie** |
| F1 on **capped** classes | — | **−0.0052, p=0.011 — loss** |

"The gain is on the non-capped classes" was a statement about focal, not about the constraint.

⛔ **And with NO allocator at all** (plain argmax, no budget equalization) **`tralo_uniform` macro-F1 is −0.0245 (p=0.0000) BELOW `clip`**, and only +0.0077 (p=0.49, n.s.) above `focal_clip`. **The classifier the constraint phase produces is worse than the one plain CE produces. Budget equalization compresses that gap to a tie; it does not create a win.**

> **Bottom line: there is no win over the post-hoc clipper. The apparent result was "focal hurts these backbones", measured against the weaker of two available baselines.** Every future campaign must carry BOTH clippers and headline the stronger one.

### ✅ The scorer itself is CLEAN — `equalize_multi` survived every attack

Worth recording so this is not re-litigated. On all 144 runs: **arm-independent** (overwriting `Predicted_Label` with uniform random integers left all 7 budget-equalized and all 6 allocation-free metrics **bit-identical**; only the non-scoring `flips` family moved); **budget constant** across arms in 48/48 cells; **feasible** in 144/144; **order-independent** (5 row permutations → 0 label changes); **not calibration-convertible** (replacing the cross-class contention key with within-class rank, which destroys all magnitude information, changes 0.02–0.04% of labels and leaves the delta identical to four decimals); **not a greedy artifact** (an exact LP max-Σlog p allocation gives the same sign and significance). The two-allocator confound is genuinely neutralised — `Predicted_Label` feeds only `NON_SCORING` metrics.

### Further scorer defects found (none load-bearing, all worth fixing)

- 🚨 **`tralo_uniform` has NO treatment.** `tralo/train.py:126-127` gates on `_cw_mode != "uniform"`, and the `"uniform"` branch is documented as bit-identical to previous behaviour. **`tralo_uniform` IS plain TraLO** — do not present it as a method variant.
- 🚨 **Baseline runs are BIT-IDENTICAL across cap levels.** `clip` and `focal_clip` never train against the cap, so L30 and L50 re-use the same model — md5-verified, 18/18. There are **6 distinct baseline models behind 12 cells**, and the Wilcoxon treats them as independent, counting every baseline weakness twice. TraLO's matrices *do* differ across caps, so the duplication is one-sided.
- 🚨 **`--percell` pools the cap axis** (`full_panel.py:333` groups by `level=[0,1]`, dataset+model). **That is the identical "pooled the swept axis" failure that refuted the granularity result.** Group by `level=[0,1,2]`.
- **The greedy is suboptimal vs the LP, and the suboptimality is ARM-DEPENDENT — it flatters TraLO**, accounting for ~12% of the dermmnist margin. Mean per-item log-prob gap to optimum: `clip` 0.0326, `focal_clip` 0.0224, `tralo_uniform` 0.0211. Either call it a greedy allocation or use the LP.
- **`f1_score(..., average="macro")` has no `labels=`**, so the denominator is `union(y_true, y_pred)`. Correct on this data by luck (verified per dataset), fragile in principle. Pass `labels=list(range(n_classes))`.
- ⚠️ **"Equal compute" is true in LABELLED epochs only.** `tralo/train.py:239-268` runs 29 forward+backward passes over `X_test` the clippers never pay — ~28% more total work on dermmnist. It sees test *inputs*, never test labels, so it is the transductive setting by design, but the phrase needs that qualifier.
- Stale comment at `:226-229`: ccP/ccR/ccF1 being one metric is a **single-class** identity; with 3–5 capped classes macro-averaged, their p-values already diverge (0.273 / 0.297 / 0.282).

### 1. `focal_clip` is NOT the strongest clipper — on RegNetY it is much the WEAKER one

| RegNetY400MF | macro-F1 | accuracy | cc-F1 |
|---|---|---|---|
| **`clip` vs `focal_clip`** | **+0.0308** (20/24, p=0.0001) | **+0.0462** (20/24, p=0.0006) | +0.0140 (19/24, p=0.0006) |

On regnet/tissuemnist, `focal_clip` accuracy is 0.372–0.382 against plain `clip`'s 0.499–0.517. **So "TraLO beats focal_clip" is substantially "focal hurts RegNetY", not "TraLO helps."** The earlier note calling `focal_clip` "the hardest bar" was measured on MobileNetV3 and does not transfer.

### 2. Against PLAIN `clip`, TraLO ties on one backbone and SIGNIFICANTLY LOSES on the other

| vs `clip` | macro-F1 | accuracy | cc-F1 |
|---|---|---|---|
| mcbar (MNv3) | +0.0022, 12/24, p=0.65 — tie | +0.0015, 13/24, p=0.44 — tie | −0.0035, p=0.19 — tie |
| **mcbar_regnet** | −0.0055, 9/24, p=0.11 | **−0.0069, 6/24, p=0.043 LOSS** | **−0.0069, 9/24, p=0.020 LOSS** |

**There is no win over the plain clipper anywhere.** Every campaign must carry BOTH clippers; whichever is stronger on that backbone is the bar.

### 3. 🚨 THE STATISTICS USED THE WRONG UNIT — and it is the project's own documented rule

`full_panel.py` pairs on (dataset, model, cap, **seed**) and runs Wilcoxon over **24 pairs**, treating 4 seeds inside a cell as independent replicates. The project's rule is the opposite: **the atomic cell is (dataset, backbone, cap) averaged over seeds, and summaries COUNT CELLS.** At the correct unit there are **6 cells**, not 24 pairs:

| comparison | macro-F1 | accuracy | verdict at the right unit |
|---|---|---|---|
| mcbar vs `focal_clip` | **6/6 cells** (sign test p=0.031) | 6/6 | holds, but p=0.031 not 0.0087 |
| regnet vs `focal_clip` | 5/6 (p=0.22) | 5/6 | **NOT significant** |
| regnet vs `clip` | **1/6** | 1/6 | a loss |

**Seed-level pooling overstated significance by roughly an order of magnitude.** This affects every campaign read with this scorer today.

### 4. The effect is ONE DATASET

Per-cell accuracy deltas vs `focal_clip`, regnet: tissuemnist contributes **+0.0394 of the +0.0392 total** — the other four cells net to **−0.0002**. For mcbar accuracy tissue is 79%. Mechanistically coherent: measured against the oracle-under-constraint ceiling, derm and oct sit at **0.92–0.98 of ceiling** (nothing left to win) while tissuemnist is at **0.58–0.61**. All the headroom, and all the effect, is in the one cell furthest from its ceiling. **The honest statement is about tissuemnist, not about three datasets.**

### Two undocumented conventions the reproduction needed

- **Caps are `np.round`, not `floor`** (`constraints.py:_round_to_K`). Floor does not reproduce the claim, and floor caps are *violated* by the pipeline's own shipped predictions (derm L30 ships exactly 67 = round(66.9); floor would be 66). ⚠️ `np.round` is banker's rounding, so at L50 tissue class 1 (56.5) rounds DOWN and class 3 (111.5) rounds UP — cap direction flips on the parity of the half-integer.
- **Probabilities are row-renormalized before allocation.** A ~2e-7 perturbation, but not cosmetic: octmnist L50 seed 1 changes **54 of 1000 labels** and moves that run's macro-F1 by **+0.0040** — 40% of the whole mcbar headline. Cause: 332 exactly-duplicated float32 probability pairs, 3 of 4 classes capped, budget binding hard, so an arbitrary tie-break allocates slots. **The scorer must break ties on an explicit deterministic key, not on float noise.**

### What survives

Data health is clean: 144 runs complete, all arms md5-distinct, equal compute honoured, identical test sets and caps across arms within every cell, no NaNs, caps bind hard everywhere (raw counts exceed budget by 73–746 items).

**What is left of the claim: on tissuemnist, TraLO improves macro-F1 and accuracy over a focal clipper, at 6/6 and 5/6 cells across two backbones — and it does not beat a plain CE clipper anywhere.** That is a much smaller result and it is the one that is true.

### 🚨 NOVELTY: the regularizer framing is ALREADY PUBLISHED — only the ASYMMETRY is ours

Literature sweep, 2026-08-18. Decompose the claim and most of it is prior art:

| component | status |
|---|---|
| a count constraint on unlabelled data acts as a **regularizer improving overall accuracy** | ⛔ **KNOWN since 2007. Do not claim.** Mann & McCallum 2007 (Expectation Regularization); Xu et al. 2018 (Semantic Loss, incl. cardinality); Kervadec et al. 2019 (differentiable size penalty → near-full supervision from 0.1% labels) |
| constrained training **ties** post-hoc on the constrained class | ⚠️ **PROVED in weaker form.** Fraiman & Fraiman 2026 (arXiv 2605.03289) Prop. 4: capacity-constrained learning *coincides with post-hoc thresholding* when the score has no free parameters. **Three months old.** Novel here only as a deep-net result at equal compute |
| the gain is **asymmetric — nil on the capped class, significant on the others**, at matched budget AND matched compute | ✅ **DEFENSIBLE AS NOVEL.** No paper found makes this decomposition |
| it costs calibration | ⚠️ **PREDICTED** by van Krieken et al. ICML 2024 (constraint losses bias toward overconfidence). Frame as confirmation, not discovery |

**Closest to scooping us: Wang et al., ICML 2023, *On Regularization and Inference with Label Constraints*.** It formally establishes that constraint regularization shrinks the hypothesis class (better generalization, added bias) while constrained *inference* converts violation into risk reduction. A reviewer can call our headline "Wang et al.'s theorem measured on MedMNIST". **Cite it as the theoretical explanation of our result, not as related-but-different — distancing will read as evasion.** Its defence: no experiments, no cardinality constraints, and no split between constrained and non-constrained classes.

🚨 **Honesty flag on our own baseline.** Fioretto & Van Hentenryck's own papers state the method "dramatically decreases constraint violations and, in some applications, increases the prediction accuracy." **Claiming "constraints improve accuracy" as our finding while citing Fioretto as a baseline will be caught.** Verify the exact sentence in the ECML 2020 PDF before writing related work.

🚨 **Both dual baselines are being REPURPOSED and we must say so.** Fioretto LDF is a Lagrangian-dual framework demonstrated on optimal power flow, gas networks and fairness; Hounie RCL adapts the *constraint level itself* against a relaxation cost, demonstrated on invariance selection and federated learning. **Neither is a count-constraint method.** Describing them as such is a misrepresentation a reviewer familiar with either will catch immediately.

**The safe claim**: *on a transductive count cap, constrained training yields no gain on the capped class over an equal-compute post-hoc clipper, yet significantly improves macro-F1 and accuracy on the uncapped classes, at a significant cost in ECE/NLL. The regularization effect of expectation constraints on unlabelled data is known; what is new is that at matched budget the effect is entirely disjoint from the class the constraint names.*

**The unsafe claim**: anything of the form "we show count constraints act as a regularizer."

### 🔧 BASELINE PARITY: one blocker found and fixed, three asymmetries to disclose

An AST audit of what each runner actually reads (not what `hp_defaults` declares):

⛔ **BLOCKER, now fixed.** `enable_checkpoint_restore` was read by **`tralo` only** (`train.py:559`). `fioretto_ldf` and `hounie_rcl` restored **unconditionally**, so in any head-to-head TraLO alone kept its trained model while both duals were swapped for a checkpoint chosen on constraint satisfaction — a mechanism measured at **−0.0351 AP**. **A TraLO win over the duals would have been contaminated by an advantage only TraLO was granted.** Both dual runners now carry the same gate, default `True` so no existing run moves.

Disclose, do not "fix":

- **29 vs 28 constraint steps.** TraLO initialises `lambda_global = 0.01` so its first constraint step fires; both duals start every multiplier at 0, so their epoch-0 constraint backward is skipped by the active-set gate (verified in an archived log: epoch 0 `constraint_loss=0.0`, epoch 1 `=14.48`). A 3.4% step advantage to TraLO. **λ₀ = 0 is Fioretto Eq. 5 — changing it would misrepresent the baseline.**
- **`hounie_eta_lambda` / `hounie_eta_u` default 10× apart** between `hp_defaults.py` (0.01) and the runner's in-code default (0.1). Every shipped config used 0.01; omitting them silently runs a 10× dual step. Set explicitly.
- **`fioretto_step_size` is mandatory** — the runner raises `ValueError` without it, so an arm cloned naively from `gen_mcbar.py` crashes at dispatch.
- **Three incompatible `training_log.csv` schemas and two epoch origins.** TraLO iterates `range(warmup_epochs, total)` (ABSOLUTE) and preserves warm-up rows; both duals iterate `range(constraint_epochs)` (RELATIVE) and open the same filename with mode `"w"`, **truncating the warm-up rows**. Any dynamics or convergence figure must branch per methodology.
- **`heuristic` reads ZERO hyperparameters.** Everything a clipper does is decided in warm-up (`warmup.py:54,62`), which is why `focal_clip` is live there and inert in `arm_joint`.

✅ **Multi-class caps are genuinely supported by all three** — `data.py:40-45` and `constraints.py:10-13` normalise `constrained_class` to a list, and both duals are per-class throughout. No `[0]` truncation anywhere.

✅ **Historical dual runs are unusable**: archived configs are `constraint_epochs=300` + `lr_constraint=5e-6` — 10× the budget and the LR trap. The duals must be re-run in-campaign, which is `results/mcbar_duals` (120 runs, 5 arms).

### 🖥️ WHICH HARDWARE EACH RESULT RAN ON — and the one confound it creates

The two clusters are different GPU generations and the campaigns are split across them:

| campaign | backbone | host | GPU |
|---|---|---|---|
| `mcbar` | MobileNetV3 | dsisco02 | **RTX PRO 6000 Blackwell** |
| `mcbar_regnet` | RegNetY400MF | dsisco01 | **Quadro RTX 6000** |
| `mcjoint` | MobileNetV3 | dsisco02 | Blackwell |
| `nsteps`, `sepopt`, `rankpair`, `rankrep`, `beta` | — | dsisco01 | Quadro |
| `mcbar_mnv2`, `mcbar_shuffle`, `mcbar_caps` (in flight) | MNv2 / ShuffleNetV2 / MNv3 | dsisco01 | Quadro |

✅ **Every arm inside a campaign ran on one GPU**, so each paired comparison is internally clean — hardware cannot explain a within-campaign delta.

⛔ **RETRACTED framing — see section 0.** What reproduces on both generations is the margin over `focal_clip`, and `clip` beats `focal_clip` by more than TraLO does, so the reproduction is of *focal's* weakness, not of a method win. The hardware conclusion still stands on its own terms: the effect is not a numerics artifact of one card (Blackwell defaults to BF16 AMP; the Quadros are pre-Ampere and take the FP16+GradScaler path — a real difference in the training numerics, not just speed).

⚠️ **But backbone and hardware are CONFOUNDED between `mcbar` and `mcbar_regnet`.** The RegNetY effect is ~2.5x the MobileNetV3 one, and that comparison mixes architecture with card. **Do not attribute the larger effect to the backbone.** A clean separation needs MobileNetV3 × {L30, L50} re-run on Quadro, which is not yet done.

## ⛔ `sepopt`: the DIRECTION fix makes it significantly worse — prediction confirmed (2026-08-18)

`results/sepopt`, 4 cells x 4 seeds = 16 pairs. `tralo_sepopt` gives the constraint its own Adam, which measurement showed recovers ~10x more constraint gradient (cos 0.139 vs 0.009-0.017).

| arm | AP | AUROC | cc-F1 | macro-F1 |
|---|---|---|---|---|
| `tralo` (shared optimizer) | −0.0153 * | −0.0064 * | −0.0041 tie | **+0.0077** (12/16) lean win |
| `tralo_sepopt` (dedicated) | **−0.0938** (p=0.0006) | −0.0340 (p=0.0002) | −0.0295 (p=0.0063) | −0.0053 **tie — the lean win is GONE** |

**Delivering more constraint gradient loses AP, AUROC and cc-F1 significantly, and destroys the macro-F1 advantage.** `tralo_n4_sep` predicted exactly this from the step-count campaign; it is now confirmed independently at n=1 on four cells.

> 🔑 **Both live levers are now closed, each with its own campaign.** Step COUNT (`nsteps`, monotone across four arms) and step DIRECTION (`sepopt`, 16 pairs, significant) both point away from quality. **The constraint phase's near-total ineffectiveness is not a defect to repair — it is the operating point, and the macro-F1 win exists only while it stays that way.**

## 🔴🔴🔴 THE STEP-COUNT LEVER IS LIVE AND IT POINTS THE WRONG WAY — the 29-step defect is what was PROTECTING us (2026-08-17)

`results/nsteps`, dermmnist × MobileNetV3 × {L30, L50}, capped classes {1,2,4}, 4 seeds = 8 pairs. md5-verified live: `tralo_n1` / `tralo_n4` / `tralo_n16` / `tralo_n4_sep` all differ. Against `clip` at equal compute:

| arm | steps/epoch | AP | AUROC | cc-F1 | macro-F1 | acc |
|---|---|---|---|---|---|---|
| **`tralo_n1`** (incumbent) | 1 | −0.0235 * | −0.0100 * | −0.0073 | **+0.0157 (7/8) *WIN*** | −0.0002 tie |
| `tralo_n4` | 4 | −0.1984 * | −0.0568 * | −0.0658 * | −0.0429 | −0.0274 * |
| `tralo_n16` | 16 | −0.1819 * | −0.0529 * | −0.0609 * | −0.0295 * | −0.0236 * |
| `tralo_n4_sep` | 4 + dedicated optimizer | **−0.2501** * | −0.0846 * | **−0.0846** * | −0.0473 * | −0.0333 * |

(All five arms complete at 8/8. `*` = p ≤ 0.039; every loss at n=4 / n=16 / n4_sep is 0/8 cells at p = 0.0078, the floor for 8 pairs. `n4_sep` was −0.3051 on its first four seeds and settles at −0.2501 on eight — still the worst arm in the campaign, and now significant on every metric rather than p-floored.)

🛑 **More constraint steps make it monotonically worse, and the incumbent's single step is the best setting in the sweep — the only arm that wins anything.** Giving the constraint four steps costs 20 points of AP.

🔑 **This inverts the entire mechanism narrative.** The audit established three defects — 29 steps against CE's 3654, a direction retaining ~10% of the constraint gradient through a shared Adam, a magnitude cancelled outright by the unit-norm clip — and the implicit reading was that fixing them would let the constraint finally work. **The measurement says the opposite: the constraint phase does almost nothing, and that is precisely why TraLO is only mildly worse than a clipper. Make it do more and it destroys the model.** The 29-step starvation is not a bug holding back a good method; it is what keeps the method from being much worse.

⚠️ **And the direction fix makes it worse still.** `tralo_n4_sep` — 4 steps *plus* the dedicated constraint optimizer that measurement showed recovers ~10× more constraint gradient — is the worst arm in the campaign at AP −0.3051. Both live levers are live, and both point away from quality. `results/sepopt` tests direction alone at n=1 and now has a prediction to falsify.

### 🎯 The paper's headline claim REPRODUCES in the live regime — at the bottom of its stated range

The paper's headline is **macro-F1 vs clipping, +1.6 to 5.3 pp**, and it was measured entirely at warm-up 50 — the dead regime, where the constraint phase is ~30 unit-norm steps on a frozen representation. Nothing had checked whether it survives at warm-up 1 with equal compute. Two campaigns now say it does, at the low end:

| campaign | arm | bar | macro-F1 | cells | p |
|---|---|---|---|---|---|
| `nsteps` (complete) | `tralo_n1` = incumbent TraLO | `clip` | **+1.57 pp** | 7/8 | 0.039 |
| `mcbar` (preliminary) | `tralo_uniform` | **`focal_clip`** | **+1.40 pp** | 9/12 | 0.034 |

**The claim holds against the harder bar too**, which the paper never tested — `focal_clip` beats `clip` on calibration at 16/16 for free. Both campaigns put the effect at ~1.4–1.6 pp, i.e. the bottom of the paper's range rather than its middle.

⚠️ **They are not independent**: the derm × MNv3 × {L30, L50} cells are the same configuration in both. `mcbar`'s octmnist and tissuemnist cells are the new evidence, and macro-F1 is positive on all three datasets separately.

⚠️ **In both, cc-F1 ties.** The gain is on the **uncapped** classes — better allocation of the remaining budget, not better selection of the constrained class. Say it that way; "TraLO improves the constrained class" is not what these numbers show.

> Consequence for reading the corpus: **stop treating "the constraint phase is starved" as a defect to be repaired.** It is the operating point. The remaining question is not how to deliver more constraint gradient but whether any per-item signal can improve the ranking at all — which is what `results/rankpair` asks, with no constraint phase in it.

⚠️ One dataset, one backbone, two cap levels: a probe. But the effect is monotone across four arms and the magnitudes are 10–20× the effects usually argued over here.

## ⛔⛔ `joint_objective` IS DEAD ON MULTI-CLASS TOO — and `tralo` vs `clip` is now a tie on AP (2026-08-17)

`results/mcjoint` at 95/96, re-read with **both** scorer fixes in place (`equalize_multi` + pair-restricted `dropna`). This is the best-powered campaign in the project: 4 arms × **3 datasets** (derm, oct, tissue) × **2 cap levels** (L30, L50) × 4 seeds = 6 cells, 24 pairs, coupled multi-class caps throughout.

### `tralo_joint` vs `clip` — a total loss, on every metric, at p = 0.0000

| | AP | AUROC | ECE | NLL | cc-F1 | macro-F1 | acc |
|---|---|---|---|---|---|---|---|
| **delta** | **−0.2879** | −0.1616 | +0.2673 | +7.2847 | **−0.1522** | −0.1459 | −0.1024 |
| **cells** | 0/23 | 0/23 | 0/23 | 1/23 | 0/23 | 0/23 | 1/23 |

**All thirteen scored metrics, significant, essentially no cell won.** `joint_objective` was already rejected on a single capped class, with the caveat that the verdict said nothing about multi-class. Multi-class has now been tested across three datasets and it is far worse than the single-class rejection. ⛔ **Closed. Do not revisit the joint formulation.**

### `tralo` vs `clip` — the ranking cost is smaller than recorded, and AP no longer reaches significance

| metric | clip | tralo | delta | cells | p | verdict |
|---|---|---|---|---|---|---|
| AP | 0.6594 | 0.6471 | −0.0123 | 9/24 | 0.0787 | **tie** |
| **AUROC** | 0.9068 | 0.8995 | **−0.0072** | 7/24 | **0.0079** | *** LOSS |
| cc-F1 | 0.4357 | 0.4313 | −0.0045 | 9/24 | 0.2113 | tie |
| macro-F1 | 0.4961 | 0.5013 | +0.0052 | 13/24 | 0.2735 | tie |
| accuracy | 0.6059 | 0.6070 | +0.0011 | 11/24 | 0.9879 | tie |

⚠️ **This narrows the recorded verdict again.** The entry above says "TraLO costs RANKING (AP −0.021, AUROC −0.0099)". On the near-complete campaign with the pairing fixed, **AP is a tie (p = 0.079) and AUROC is the single significant difference — at −0.0072, which is significant and tiny.** ECE, Brier and NLL all fail to reach significance. Quote AUROC, not AP, and quote its size.

### 🚨 `focal_clip` in `mcjoint` is INERT — 24 runs of a second `clip`

**All 16 metrics differ from `clip` by exactly +0.0000, p = nan, 0/24 cells.** This is the known `arm_joint` defect (`base_loss` / `focal_alpha` / `focal_gamma` are dead keys in that worktree only) landing on a full campaign — 24 runs of a second `clip`. So `mcjoint`'s strong-baseline column is empty, and its extra reach (tissuemnist, and the L30 level) is what was lost with it.

> The liveness gate is what caught it, again. A plain delta of 0.0000 across 16 metrics reads as "no difference between two clippers", which is exactly what an inert arm looks like from the outside.

## 🛑🛑 AGAINST THE STRONGEST CLIPPER ON MULTI-CLASS, TraLO LOSES cc-F1 ITSELF (2026-08-17)

`arm_multiclass/results/multiclass` — where `focal_clip` **is** live — re-read with the pairing fix. `tralo_byk` sits at 8/16, so under the old bug every comparison in this campaign was silently cut to **8 pairs**; the numbers below are the full **16** (derm + oct × {L50, L70} × 4 seeds).

`tralo_uniform` vs `focal_clip`:

| family | metric | delta | cells | p | |
|---|---|---|---|---|---|
| allocation-free | AP | −0.0194 | 3/16 | 0.0063 | *** LOSS |
| | AUROC | −0.0089 | 2/16 | 0.0008 | *** LOSS |
| | ECE | +0.0639 | 0/16 | 0.0000 | *** LOSS |
| | NLL | +0.7648 | 0/16 | 0.0000 | *** LOSS |
| | Brier | +0.0550 | 1/16 | 0.0001 | *** LOSS |
| budget-equalized | **cc-F1** | **−0.0110** | **4/16** | **0.0309** | *** **LOSS** |
| | macro-F1 | −0.0030 | 9/16 | 0.9399 | tie |
| | accuracy | −0.0044 | 6/16 | 0.3066 | tie |

🛑 **This is worse than anything recorded, and it is the headline metric.** Everywhere else the pattern has been "TraLO costs ranking and calibration, ties on quality." Here, against the strongest clipper on coupled multi-class caps, **cc-F1 itself is a significant loss** — the number the method is supposed to be about. Only the macro metrics and accuracy tie.

⚠️ The recorded single-class head-to-head said "against the strongest clipper it is a tie" (cc-F1 +0.0064, p = 0.21). **That tie does not carry to multi-class; on 16 pairs it is a loss.** Multi-class was the setting the whole coupled-assignment argument rested on.

And `focal_clip` vs `clip`, same campaign, same fix — the calibration rout is real and **stronger** than the truncated read suggested (recorded as 8/8, actually **16/16**):

| ECE | NLL | Brier | ConfGap | every quality metric |
|---|---|---|---|---|
| −0.0581 (16/16, p=0.0004) | −0.6457 (16/16, p=0.0004) | −0.0415 (12/16, p=0.0297) | +0.0755 (16/16, p=0.0004) | tie |

**A base-loss swap on a post-hoc clipper buys a calibration rout for free and costs nothing.** It remains the bar, and it is a harder bar than `clip`.

### ✅ FINAL (`mcbar`, 72/72): the cc-F1 loss was a CAP ARTIFACT — and TraLO significantly BEATS the strongest clipper on macro-F1 and accuracy

`arm_multiclass/results/mcbar`, 3 arms × derm/oct/tissue × {L30, L50} × 4 seeds = **24 matched pairs, 6 cells, 3 datasets, 2 cap levels**. `focal_clip` **live** (md5-verified different from `clip`). `tralo_uniform` vs `focal_clip`:

| family | metric | delta | cells | p | |
|---|---|---|---|---|---|
| budget-equalized | **accuracy** | **+0.0130** | **19/24** | **0.0023** | *** **WIN** |
| | **macro-F1** | **+0.0102** | **18/24** | **0.0087** | *** **WIN** |
| | macro-P | +0.0142 | 17/24 | 0.0340 | *** WIN |
| | macro-R | +0.0084 | 17/24 | 0.0315 | *** WIN |
| | cc-F1 | +0.0020 | 10/24 | 0.8115 | tie |
| allocation-free | AP | −0.0109 | 6/24 | 0.0646 | tie |
| | AUROC | −0.0120 | 2/24 | 0.0000 | *** LOSS |
| | ECE | +0.0777 | 0/24 | 0.0000 | *** LOSS |
| | NLL | +1.2762 | 0/24 | 0.0000 | *** LOSS |

⛔ **RETRACTS "against a live `focal_clip` on multi-class, TraLO loses cc-F1 itself."** That was measured on four cells at **L50 and L70 only** — both loose caps. Swept to L30/L50 across three datasets it is a clean tie (+0.0020, p = 0.81). **The same failure mode as the no-restore retraction: a result from cells sharing one cap regime.** The rule has now bitten three times; it is not a caution, it is the default expectation.

✅ **And the macro metrics go the other way, significantly.** This is the strongest positive result the project has produced: a significant win on **the paper's headline metric** against **the hardest available baseline**, across three datasets and two cap levels, on 24 matched pairs. More data made it *more* significant than the half read (p = 0.034 → 0.0087), not less.

Three things must be stated with it, unprompted:

1. **cc-F1 ties, so the gain is on the UNCAPPED classes.** TraLO allocates the remaining budget better; it does not select the constrained class better. "TraLO improves the constrained class" is not what these numbers say.
2. **Calibration is a rout against us and it is large** — ECE +0.078 and NLL +1.28, both at 0/24 cells, p = 0.0000. There is no calibration claim here, and a reviewer will find this immediately.
3. **AUROC is a significant loss** (−0.0120, 2/24) and AP is a lean loss. The ranking cost measured everywhere else is still present; it is simply not what macro-F1 reads.

⚠️ One backbone (MobileNetV3). ⚠️ The equalized metrics run through `equalize_multi`, written 2026-08-17 — deterministic and identical across arms, so it cannot favour one, but this win lives downstream of it.

## 🚨 FIXED: a lagging third arm silently deleted pairs from every comparison in the panel (2026-08-17)

`full_panel.py` built each comparison as

```python
q = df.pivot_table(index=key, columns="arm", values=m).dropna()
```

The pivot carries **one column per arm in the campaign**, so `.dropna()` deleted any seed where *any* arm was missing — including arms with nothing to do with the pair being printed.

Caught on `results/beta` mid-flight. The header printed `{joint_b0: 4, joint_b1: 2, joint_b5: 4}`, and `joint_b5 vs joint_b0` — both complete on all four seeds — was scored on **two pairs**, because `joint_b1` had not yet finished seeds 1 and 2 and its NaNs took those rows away from everyone. It read `cells 1/2, wilcoxon 1.0000`.

🛑 **Why this is worse than it sounds: every campaign is read while it is still filling — that is how a bad arm gets killed early. Under this bug the reading silently shrank to the slowest arm's progress, and Wilcoxon at n=2 cannot return below p=0.5, so an in-flight campaign ALWAYS looked like a tie.** Arms have been abandoned on exactly that signature.

Fixed by restricting the pivot to the two arms being compared before the `dropna`. Applied to the main repo and every worktree that has the scorer. ⚠️ `arm_rank` has no `full_panel.py`; score `rankrep` from `~/OptimizationLoss` with an absolute `--campaign` path so the canonical fixed copy is the one that runs.

### ⛔ FINAL: the undershoot hinge is REJECTED at every dose — β makes `joint` monotonically worse

`results/beta` complete, now including the `clip` arm it was missing. dermmnist × MobileNetV3 × L30, capped classes {1,2,4}, 4 seeds. Against the clipper, **every arm loses every metric at 0/4 cells**:

| arm | AP | cc-F1 | macro-F1 | accuracy | NLL |
|---|---|---|---|---|---|
| `joint_b0` | −0.3088 | −0.0894 | −0.0960 | −0.0414 | +4.46 |
| `joint_b1` | −0.5052 | −0.2356 | −0.3135 | −0.2479 | +9.07 |
| `joint_b5` | −0.3982 | −0.2025 | −0.2389 | −0.2319 | +1.08 |
| `joint_b25` | −0.5399 | −0.2559 | −0.3577 | −0.3485 | +4.24 |
| `joint_b100` | −0.5867 | −0.3200 | −0.3960 | −0.3781 | +1.23 |

**The hinge does not rescue `joint`; it damages it, and the damage grows with β.** β=0 is the least bad dose in the sweep. One cell and 4 seeds means the Wilcoxon floors at p=0.125 — but at 0/4 cells and −0.3 to −0.6 AP, significance is not the question. This matches `mcjoint`'s independent verdict on `tralo_joint` (AP −0.288 across 3 datasets, p=0.0000).

🚨 **`joint_b100` seed 4 DIVERGED**: `final_predictions_raw.csv` is NaN in **all 14021 entries**, and `status` is `completed`. Nothing in the pipeline guards divergence, so it entered the corpus as a normal run and crashed `average_precision_score` — taking the scoring of all 23 healthy runs down with it. `full_panel.py` now drops non-finite runs and prints the path. A corpus-wide scan found **1 of 1298** runs affected, so this is a guard rather than a retraction — but only known to be one because the scan was run. **"The hinge diverges at β=100" is that arm's result.**

### The earlier partial read, kept for the hypothesis it killed

`results/beta`, dermmnist × MobileNetV3 × L30, capped classes {1,2,4}, 4 seeds. `joint_b5` vs `joint_b0`, all four pairs:

| family | metric | delta | cells |
|---|---|---|---|
| allocation-free | AP | **−0.0895** | 1/4 |
| | AUROC | −0.0662 | 1/4 |
| | **ECE** | **−0.1166** | **4/4** |
| | **NLL** | **−3.3730** | **4/4** |
| budget-equalized | cc-F1 | **−0.1131** | 1/4 |
| | macro-F1 | −0.1429 | 1/4 |
| | accuracy | −0.1905 | 1/4 |

**Every allocation-affecting metric moves against, and large.** The calibration gain is 4/4 on both metrics and real, but `joint_b0`'s NLL of 5.42 is pathological to begin with — β=5 reaching 2.04 is regression toward sane, not excellence.

⚠️ **One cell, 4 seeds: the Wilcoxon floor is p = 0.125, so nothing here can be significant by construction.** The cells column is the only readable part. And there is still no clipper in this campaign (`clip`, `b25`, `b100` unrun), so this is arm-vs-arm and cannot speak to the bar.

**The hypothesis it was built on does not survive.** `gen_nsteps_beta.py` argued that a model driven to 40% of its budget is being trained toward predictions it should not make, and that this is the source of the ranking loss — so lifting the count should shrink the AP deficit. AP got *worse* (−0.0895), and the worst-case undershoot got worse too (min raw/K 0.098 → 0.030; a min across runs, not a mean). **Over-suppression is not where the ranking damage comes from.** The `tralo_n4_b5` / `tralo_n16_b5` arms still stand — β=5 was always suspected of overcorrecting at 126 steps/epoch, and 4 and 16 are the point of testing it.

## 🔧 FIXED: the dispatcher discarded seed-major order, so no campaign ever ran in it (2026-08-17)

Every generator here sorts its `todo` seed-major, for a reason recorded repeatedly: cross-campaign drift is **0.027, about twice the effects being chased**, so an arm is only readable against an arm from the *same* campaign. Seed-major order makes every prefix of a campaign a set of matched slices.

`get_experiments_by_status` then threw that ordering away. `get_all_experiment_configs` walks with `Path.rglob`, which returns filesystem order — arbitrary, and in practice **grouped by arm**, which is the worst case available:

| campaign | state when checked |
|---|---|
| `nsteps` | 3 completed + 1 running, **all `tralo_n16`**; `clip` and `tralo_n1` at 0/8 |
| `rankrep` | opened on `rank_w03` **seed 4** |

A campaign interrupted there leaves one finished arm and an empty control — **not a partial result but no result**, because the finished arm has nothing it may legitimately be compared to. Fixed with `dispatch_key` in `src/utils/filesystem_manager.py` (seed, model, dataset, cap, arm, path), applied to the main repo and all nine worktrees; `rankrep` was restarted one run in and now dispatches `seed1 · L30 · {clip, rank_ctrl, rank_w03}` consecutively.

⚠️ **The lesson generalises past this bug: an ordering guarantee has to be enforced where the work list is BUILT, not where it is written.** Nine generators sorting correctly bought nothing for as long as the consumer re-walked the tree.

Safe to apply mid-campaign — only `main.py` and `scripts/dispatch_filtered.py` call it, both at startup; a live dispatcher holds its list in memory and the per-run `src.experiments.runner` subprocess never touches this path.

## ✅ SYSTEMATIC AUDIT SWEEPS — three things that are NOT broken (2026-08-17)

Run after `rho_step` turned out to be a dead key, on the principle that these failures should be found by a sweep rather than one at a time.

### Every shipped prediction is feasible — 199 runs, zero violations

`audit_findings.md` carried an open warning that the post-hoc **local** pass can re-violate a **global** cap it had already satisfied (phase 3 enforces per-group limits after phases 1–2 balanced the global counts). Never checked against real output until now. `paper/scripts/feasibility_check.py` reconstructs each run's caps and tests its shipped `final_predictions.csv`:

| campaign | runs | global violations | local violations |
|---|---|---|---|
| `multiclass` | 56 | **0** | **0** |
| `granularity` (up to **32 local caps**) | 81 | **0** | **0** |
| `mcjoint` (tissue: **5 coupled classes**) | 62 | **0** | **0** |

**199 runs, zero infeasible outputs.** The re-verify step closes the gap. ✅ **That open warning can be retired.**

### No dead hyperparameters beyond the two already known

`paper/scripts/dead_keys.py` enumerates every key appearing in a campaign's `hyperparams` and uses the **AST** to decide whether its value is actually extracted — a grep is not enough, and neither is a line-by-line check: once a warning naming `rho_step` was added, both report it as "read", because the `hp["rho_step"]` sits on a *continuation line* of the `log.warning(` call. Only the parse tree sees that the extraction is an argument to a logging call.

| worktree | never read | log-only |
|---|---|---|
| `arm_multiclass` | 0 | `rho_step` |
| `arm_budget` | 0 | `rho_step` |
| **`arm_joint`** | **`base_loss`, `focal_alpha`, `focal_gamma` (24 configs)** | `rho_step` |

The sweep **independently rediscovered the inert `focal_clip`**, which is the validation that the tool works. Otherwise clean — **the inert-flag problem is now bounded**, having previously bitten four times (CE-skip asymmetry, `focal_clip`, `by_k` on octmnist, `rho_step`).

### The scorer's caps match the runs' own

`compute_global_constraints` reconstruction vs the `Limit_Class*` actually logged: octmnist `{0:125, 1:125, 2:125}` and dermmnist `{1:52, 2:110, 4:112}` — **exact match on both**. Metrics are being scored against the caps the runs really enforced.

## 🔬 PIPELINE AUDIT — what is and is not wrong with the constraint phase (2026-08-17)

Prompted by "how are we getting worse — did you check CE saturation, and is the pipeline code actually correct?" Answers, from the logs and the source rather than from memory.

### ✅ CE saturation is NOT the binding problem (ruled out)

At warm-up 1 the constraint phase starts at epoch 2, and CE reaches 0.995 train accuracy only at epoch 15–30, so 13–28 constraint epochs run on a live representation. Measured over the 24 trained runs of `results/multiclass`:

| arm | dataset | cap | epoch acc≥0.995 | % of constraint phase frozen | ever satisfied |
|---|---|---|---|---|---|
| tralo_byk | derm | L50 | 17.5 | 47% | 0/4 |
| tralo_uniform | derm | L70 | 17.5 | 47% | 0/4 |
| **tralo_uniform** | **oct** | **L50** | **30.0** | **3%** | **0/4** |
| tralo_uniform | oct | L70 | 25.0 | 21% | 0/4 |

**The octmnist L50 row settles it**: CE stays live for 28 of the 29 constraint epochs and the constraint *still* never satisfies. Saturation is a real secondary problem on dermmnist (about half the phase runs frozen) but it cannot be the cause of the failure. `paper/scripts/saturation.py`.

### The failure, seen directly — octmnist L50, limits 125/125/125

| epoch | acc | CE | L_global | L_local | λ | c0 | c1 | c2 |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.835 | 0.411 | 0.02 | 0.07 | 0.06 | 412 | 209 | 243 |
| 10 | 0.988 | 0.036 | 17.7 | 53.2 | 0.43 | 389 | 235 | 208 |
| 20 | 0.990 | 0.030 | 86.7 | 260.8 | 0.93 | 311 | **322** | 173 |
| 30 | 0.997 | 0.011 | **190.4** | **567.4** | **1.43** | 348 | 239 | 184 |

**The penalty grows ~8,000×, λ ratchets 24×, and the counts move ~15% — non-monotonically, with class 1 ending higher than it started.** It must shed 396 predictions and sheds ~80 net. CE is healthy throughout.

### ✅ The soft/hard mismatch is NOT biting here (ruled out)

The loss optimises soft counts while satisfaction reads hard ones, so a soft count below K with a hard count above it would leave the penalty gradient-free during a live violation. Measured, they track within ~1% (E30: hard 348/239/184 vs soft 351.0/237.1/184.6). **The penalty has a large, live gradient the entire time.** The loss is signalling correctly; the optimiser is not acting on it.

### 🔴 The 8,000× penalty growth: explained, bounded, and divided straight back out

"An 8,000× growth on one component is a red alert that something is wrong in the code" — worth chasing, and the chase is instructive.

**It is not unbounded.** `sat = E/(E+K) ≤ 1` and `quad = E_norm²/(1+E_norm²) ≤ 1` with `sat_factor = quad_factor = 1.0`, so the global term is bounded by `Σ_c λ_c·w_c·(1 + ρ)`.

**But the bound is far larger than the configs suggest, because `rho_step` is a dead key.** `train.py:161` computes

```python
rho_target = hp.get("rho_target", 100.0)
rho_step   = (rho_target - initial_rho) / max(constraint_epochs, 1)
```

⚠️ **The `rho_step: 0.05` set by every `newdirections/` generator is NEVER READ.** The real step is `(100 − 0.5)/29 = 3.43` per epoch, so **ρ ramps 0.5 → 100**, not 0.5 → 1.95. With λ ratcheting 0.01 → 1.43, the ceiling is `1.43 × 101 ≈ 144` per class, and the observed `L_Global = 190` across three classes sits inside it. **The magnitude is by design, not a bug.**

**The gradient escalates too** — the penalty saturates, but λ and ρ more than compensate. For class 0 on octmnist L50, `d/d(soft count)` of `λ(sat + ρ·quad)`:

| | λ | ρ | E | dL/dcount |
|---|---|---|---|---|
| epoch 2 | 0.06 | 0.5 | 306 | **6.4e-5** |
| epoch 30 | 1.43 | 100 | 226 | **0.227** |

≈ **3,500× more gradient**. So the escalation machinery is doing exactly what it was designed to do, right up until the moment it touches a weight.

**Then two independent mechanisms delete all of it:**

1. **`clip_grad_norm_(max_norm=1.0)` renormalises to exactly 1** whenever the raw norm exceeds it. **Measured** at the real end-of-run λ=1.43, ρ=100 (`adam_contamination.py --lam 1.43 --rho 100`):

   | epoch | delivered \|g_con\| | **raw, pre-clip** | cos(upd, g_con) |
   |---|---|---|---|
   | 1 | 1.000e+00 | **5.39e+03** | 0.0511 |
   | 2 | 1.000e+00 | **4.24e+03** | 0.0413 |
   | 3 | 1.000e+00 | **2.56e+03** | 0.0297 |
   | 4 | 1.000e+00 | **1.24e+04** | 0.0157 |
   | — | fresh **dedicated** optimizer | | **0.1439** |

   **The raw gradient carries norm 2,560–12,400 and the clip delivers exactly 1.000, every step.** A ~5,000× amplification followed immediately by a ~5,000× division. (At the *initial* λ=0.01, ρ=0.5 the raw norm is ~0.42 and only crosses 1.000 around epoch 5 — so the cancellation begins early and is total thereafter.)
2. **Adam is scale-invariant** (`m/√v`), so even without the clip a uniform amplification changes nothing.

**⇒ The λ ratchet and the ρ ramp are no-ops by construction.** Every unit of escalation the log displays is divided back out before it reaches a parameter. The smoking gun is already in the contamination table: **`cos(update, g_con)` stays flat at 0.009–0.017 across epochs while the penalty escalates hard.** If escalation did anything, that number would move.

⚠️ **Consequence for the record: the 13 wave-1 arms that varied penalty shape or escalation were tuning a quantity that is provably divided out — twice.** Previously attributed to Adam alone; the clip is the *first* and more absolute killer, and it acts before Adam sees anything. ⚠️ The clip cannot simply be removed: it is load-bearing (removing it loses 4/4 cells and drives the constrained count to 0.0).

⚠️ **What this does NOT fix.** A dedicated constraint optimizer corrects the step *direction*; it does not make λ/ρ live, because the clip and Adam still normalise magnitude away. **The only remaining levers are the step DIRECTION and the step COUNT — never the penalty magnitude.**

### 🚨 Defect 1: the constraint step and the CE steps share one Adam

`optimizer = make_optimizer(...)` is built **once** (`train.py:86`) and used by both phases — 126 CE steps, then one constraint step, through the same `m`/`v` buffers. Adam's update is `m/√v` with β₁=0.9, β₂=0.999, so the constraint step computes `m_new = 0.9·m_CE + 0.1·g_con` and divides by a `√v` that is essentially the CE gradient scale. Then 126 more CE steps overwrite `m` before the next constraint step, so **constraint momentum can never accumulate across epochs**.

How much this costs is not decidable from the code alone — it depends on `|m_CE|` versus `|g_con|`, and both move during training. **So it was measured** (`paper/scripts/adam_contamination.py`, octmnist L50: real warm-up model, real CE epochs, real chunked constraint gradient, reporting the alignment between the step Adam actually takes and the direction the constraint asked for):

| epoch | \|m_CE\| | \|g_con\| | \|√v\| | **cos(upd, g_con)** | cos(upd, m_CE) |
|---|---|---|---|---|---|
| 1 | 1.244 | 0.416 | 1.677 | **0.0160** | 0.3190 |
| 3 | 1.394 | 0.864 | 3.294 | **0.0171** | 0.4355 |
| 6 | 1.256 | 0.937 | 5.065 | **0.0088** | 0.4732 |
| — | fresh **dedicated** optimizer, same `g_con` | | | **0.1389** | — |

**The constraint step is ~1.4% aligned with the constraint gradient and 32–47% aligned with stale CE momentum.** The reference is not 1.0: Adam's per-coordinate normalisation makes any update closer to `sign(g)` than to `g`, so a dedicated optimizer on the *identical* gradient scores 0.139. **The shared optimizer retains about 10% of that (0.014 / 0.139).**

Two independent mechanisms, and the second is the one that was missed: `m` is dominated by CE momentum (0.9·1.24 ≫ 0.1·1.0), **and** `√v` — norm 5.07 and growing — rescales every coordinate by *CE* gradient statistics, scrambling the constraint direction regardless of the momentum term. Note also that the clip pins `|g_con|` to ~1.0 from epoch ~4, so the constraint's share of `m` cannot grow as CE converges.

Tested end-to-end by the `separate_constraint_optimizer` flag (`results/sepopt`, 48 runs, `tralo` vs `tralo_sepopt` differing in that one flag).

This also retro-explains `reset_optimizer_at_sat` measuring as "dominant" at warm-up 50: resetting Adam clears exactly this state. It is a no-op at warm-up 1 only because nothing ever satisfies, so the reset never fires.

### ⚠️ Defect 2 (secondary): the constraint gradient is divided by `n_chunks`

`chunk_loss = chunk_loss + (lg + ll) / n_chunks`. Each chunk evaluates the penalty at the **full** count with gradient flowing only through its own samples, so summing the chunks *without* the division reconstructs the exact full gradient. With it, the accumulated constraint gradient is `∇L / n_chunks` (4 on octmnist, 8 on dermmnist). The logged `loss_global_val` is computed separately and **without** the division, so this is a gradient-scale choice, not a logging one.

Mostly washed out in practice: the unit-norm clip binds 63–84% of steps and renormalises to 1 regardless. It matters only in the steps where the clip does *not* bind, where it shrinks the constraint's share of the shared `m` by a further 4–8×.

### ✅ Not defects (checked and cleared)

- **Train/eval mode is consistent.** `model.train()` for CE (168), `model.eval()` for transductive pass 1 (205), and pass 2 runs in eval too — dropout off and BN frozen in both passes, so the `total − chunk + chunk` count construction never mixes modes.
- **The chunked `g_soft` construction is correct.** `total.detach() − chunk.detach() + chunk` has the value of the full count and routes gradient only through the current chunk; summed over chunks that is the true gradient (up to Defect 2's constant).
- **No cap rounds to 0** in these campaigns (all ≥25), so the known "K=0 constraints are silently skipped" defect is not active here.

## 🎯 HOW MUCH IS ON THE TABLE — and why tight caps were the wrong place to look (2026-08-17)

Before running another arm, the question nobody had asked: **is there anything left to win?** For the clipper arm of every cell, comparing achieved cc-F1 against the oracle `2·min(K, n_pos)/(K + n_pos)` (perfect selection of the K items):

| cell | K | achieved | oracle | **headroom** |
|---|---|---|---|---|
| derm/MNv3 **L20** | 45 | 0.3097 | 0.3358 | **0.0261** |
| derm/MNv3 L30 | 67 | 0.4138 | 0.4621 | 0.0483 |
| derm/MNv3 L40 | 89 | 0.4984 | 0.5705 | 0.0721 |
| derm/MNv3 **L50** | 112 | 0.5537 | 0.6687 | **0.1150** |
| derm/RegNetY **L20** | 45 | 0.3116 | 0.3358 | **0.0242** |
| derm/RegNetY L30 | 67 | 0.4034 | 0.4621 | 0.0587 |
| derm/RegNetY L40 | 89 | 0.4776 | 0.5705 | 0.0929 |
| derm/RegNetY **L50** | 112 | 0.5373 | 0.6687 | **0.1314** |
| oct/MNv3 L30 | 75 | 0.4246 | 0.4615 | 0.0369 |
| oct/RegNetY L30 | 75 | 0.3985 | 0.4615 | 0.0630 |

**Mean headroom 0.0669 cc-F1. The effects fought over all year are ~0.01 — about 15% of it.** So the target is real and we have been capturing essentially none of it; "there is nothing to win" is NOT the explanation for thirteen ties.

### 🚨 Headroom grows monotonically with the cap — tight caps are the WORST place to look

**L20 has 0.024 on the table; L50 has 0.12 — five times more.** The reason is mechanical: at K=45 the budget only reaches the model's most confident items, which are nearly all correct already, so the clipper is close to oracle and no reordering can help. At K=112 the budget reaches into the uncertain middle of the ranking, which is exactly where a better ordering pays.

⚠️ **This inverts the standing belief** recorded from the AIDER/regime mining work — *"TraLO wins only where the count binds HARD"*. On headroom grounds the opposite holds: **hard-binding caps are where the least is available.** That earlier finding was measured at warm-up 50, where the whole mechanism was different, and should not be carried into this regime.

**Consequence for every future arm: test at L40/L50 as well as L30, and do not read a null at L20 as evidence against a method** — at L20 there is barely enough headroom for any method to show a 0.01 effect even if it works.

⚠️ The oracle is a hard upper bound (perfect selection), not an achievable target — the realistic ceiling is the Bayes-optimal ranking, which is lower. Use these numbers for the **relative** pattern across caps, which is robust, not as a literal budget of recoverable points.

## ⛔ REJECTED: constraint GRANULARITY — more group counts make it strictly worse (2026-08-17)

`newdirections/arm_budget/results/granularity`. The hypothesis was the Learning-from-Label-Proportions one, and it was the most principled reason left to expect a win: a single global count is one scalar of supervision, but **G group counts are G scalars**, so partitioning the test set finer should hand the constraint strictly more information and let it beat post-hoc clipping. Partition sizes G ∈ {1, 2, 4, 8, 16, 32} at a fixed `L50_G50` cap, `tralo_norestore` vs `clip`, 4 seeds, one campaign. Groups assigned by a fixed hash (`PARTITION_SEED = 20260817`) so G is the *only* thing that varies.

**It is refuted, and the trend runs the wrong way** (dermmnist/MobileNetV3, complete; octmnist still filling):

| G | ccF1 Δ | macroF1 Δ | AP Δ |
|---|---|---|---|
| 1 | −0.0075 | +0.0013 | −0.0248 |
| 2 | −0.0015 | +0.0045 | −0.0219 |
| 4 | −0.0075 | +0.0028 | −0.0146 |
| 8 | −0.0045 | +0.0005 | −0.0388 |
| **16** | −0.0090 | **−0.0110** | −0.0280 |
| **32** | **−0.0179** | −0.0076 | −0.0288 |

cc-F1 degrades roughly monotonically from −0.008 to −0.018, and macro-F1 **flips sign** between G=8 and G=16. Pooled over both datasets and all G (8 cells, 28 pairs) every allocation-free metric is a significant loss — AP −0.0203, AUROC −0.0139, NLL +0.2585, all p ≤ 0.0002 — while cc-F1 (−0.0019) and macro-F1 (+0.0011) tie. macroP (+0.0153) and macroR (−0.0162) are both significant in *opposite* directions and cancel; do not quote either alone.

**Why the LLP intuition fails here, and it is the same wall as everything else.** G counts are G more statements of *how many*, and not one statement of *which*. Adding them does not supply the missing per-item signal — it multiplies the how-many while shrinking the feasible set, and each group's count is now estimated on n/G samples, so the signal per constraint gets noisier exactly as the constraints get more numerous. More supervision in the information-theoretic sense, less usable gradient.

⚠️ **Scoring trap this campaign exposed, now fixed in `full_panel.py`:** every G level writes the *same* `constraint_tag` (`L50_G50`), and the panel's cell key was `(dataset, model, cap)`. It therefore **silently collapsed the swept axis**, reported `cells: 2`, and pooled G=1 against G=32 — the first read of this campaign was a meaningless average that hid a monotone trend. The panel now folds `n_groups` into the cell key whenever it varies. **Any generator that sweeps a dimension must encode that dimension in the cell key, not only in the directory name.**

## ⛔ REJECTED (single-class): `budget_margin` — a live knob that moves the wrong metric (2026-08-17)

`results/budgetprobe`, 2 cells × 4 seeds × w ∈ {0.3, 1.0, 3.0} + `ce_clip`, all arms post-hoc clipped so only the base loss differs. n=7 pairs (one run orphaned as `running`).

| dose | cc-F1 | macro-F1 | AP | AUROC |
|---|---|---|---|---|
| w=0.3 | +0.0025 tie | −0.0028 tie | −0.0030 tie | +0.0018 tie |
| **w=1.0** | +0.0038 tie | −0.0054 tie | +0.0144 tie | **+0.0106, p=0.031 (6/7)** |
| w=3.0 | −0.0230 lean loss | **−0.0291, p=0.047 LOSS** | −0.0244 lean loss | +0.0058 |

**The knob is live** — w=3 clearly damages macro-F1, so the loss is doing something — **but at its best dose the metrics that matter are flat.** An AUROC gain with cc-F1 unmoved means the ranking improved in a region *the cap never reads*, the same head/tail decoupling measured in the budget sweep.

**Verdict: null.** AUROC alone, at n=7, on 2 cells, with the primary metric flat, is precisely the thin signal that produced two retractions this week. Not pursued further in the single-class setting.

⚠️ The idea is **not** refuted for multi-class. `budget_margin` has ONE operating point by construction; with several capped classes there is an operating point per class, and a per-class version is a natural re-test once the multi-class baseline reports.

## 🗄️ Superseded — `budget_margin` design notes (kept for the implementation, retested in multi-class)

Every arm so far varies the count **penalty**, whose gradient is a function of the aggregate soft count. A count says how MANY, never WHICH, so it can only prune an ordering CE already fixed — the reason thirteen arms tied.

But a cap of K out of N is also a **selection rate** `r = K/N`, and that rate is an **operating point known before training**. The budget sweep says that is exactly where the only real effect lives: TraLO's TP advantage is +0.027 at k=K, decays monotonically, and **changes sign between 2K and 4K**. The cap only ever reads the head; the tail is where quality is thrown away for nothing.

`budget_margin` acts only at the operating point. Per minibatch it estimates the threshold `tau` that selects the top `r` fraction, then pays a hinge for every constrained-class positive below `tau` and everything else above it:

```
L = L_ce + w * [ mean_{i in pos} relu(tau + m - s_i) + mean_{i not in pos} relu(s_i - tau + m) ]
```

**The gradient on an item is a function of its label and its position relative to the budget.** That is the distinction this file requires a new arm to make.

Implementation notes that matter:
- `tau` is **detached** — an operating point, not a parameter. Gradient through it would let the model move the threshold instead of the items.
- `tau` is an **EMA across batches**. At r ≈ 3% and batch 64 the exact top-k is 2 items, far too noisy to define a boundary.
- The hinges are averaged over positives and negatives **separately**; pooling lets the ~15× more numerous negatives collapse this into plain negative suppression.
- `rate` comes from **train** labels (`budget_frac` × train prevalence). Test labels are never touched — transduction gives us the count, not the labels.
- `budget_frac` is added to `base_model_id`. Under this loss the **warm-up itself depends on the cap**, so without it L20 would silently load L30's cached model — the same defect class as the `focal_alpha` collision already fixed once here.

**Verified live before launch** (this project's most frequent failure is an inert flag): gradient L2 difference vs CE is 0.2396, concentrated on the constrained-class column (0.2109 vs 0.03–0.07 for the others); positives receive negative gradient (pushed up), negatives above `tau` positive (pushed down); and `budget_weight=0` reproduces CE **bit-exactly**, so the control arm is clean by construction.

**Dose probe first**, because `rank` was lost by choosing its weight from a proxy measured in the loss's own space — and that value turned out to be the worst one on AP. Here the weight is swept and read off **cc-F1eq**, the metric the claim would be made on: `results/budgetprobe`, 2 cells × 4 seeds × w ∈ {0.3, 1.0, 3.0} + `ce_clip` in the same campaign = 32 runs. Confirmed 4 distinct `base_model_id`s across the arms, so the probe cannot compare a cached model against itself.

All arms are warm-up 30 + 0 constraint epochs — **post-hoc clip on both sides, isolating the base loss.** If a budget-aware objective improves allocation at the operating point it must show up before any constraint phase is added; if it only appears in combination, the loss is not what is doing the work.

Worktree `newdirections/arm_budget` (mirrored from `arm_baseloss` so the code is otherwise identical), running on dsisco01 GPU3.

## The bar a new arm has to clear (computed 2026-08-16 — this number did not exist before)

`newdirections/bench/headroom_reference.csv`, built by `python paper/scripts/score_arm.py --build-reference`. 4 cells (derm/oct × MobileNetV3/RegNetY400MF, L30_G30) × 4 seeds, **30 total epochs on both sides** — post-hoc arms warm-up 30 + 0 constraint epochs, trained arms 1 + 29.

| method | AP | ccF1eq | macroEq |
|---|---|---|---|
| **post-hoc clipper** (`heuristic` ≡ `danits_lp`) | **0.7208** | 0.4080 | **0.7069** |
| `tralo` | 0.6716 | **0.4094** | 0.6895 |
| `fioretto_ldf` | 0.6558 | 0.3994 | 0.6857 |
| `hounie_rcl` | 0.5894 | 0.3673 | 0.6697 |

**The gap is a RANKING gap, not general damage**: −4.9 pp AP, only −1.74 pp macro-F1, and cc-F1 at equal budget is already **even** (+0.0014 to TraLO). This is the numerical form of "a count says how MANY, never WHICH", and **AP is the number a new arm has to move**.

`heuristic` and `danits_lp` are identical to 4 dp on every metric and cell, necessarily: both allocate on the *same* warm-up-30 model, AP is allocation-free, and cc-F1eq/macroEq re-allocate to exactly K. At equal compute the post-hoc "methods" are one model, so the real comparison is *plain CE for 30 epochs* vs *a trained constraint phase*.

**Two bugs had kept this benchmark from ever being computed** — both are the kind that fabricate a result rather than an error:

1. **`--build-reference` could never return a row.** `collect()` carried `if keep and hp.get("warmup_epochs") != 50: continue` — a leftover from the frozen grid — while its own `DEFAULT_ROOTS` is `results/headroom`, which contains no warm-up-50 run at all. It died on the empty frame. Fixed: warm-up is a parameter defaulting to no filter, and `keep` no longer pins to seed 1 (which had been discarding 3 of every 4 runs).
2. **`results/headroom` is an `lr_constraint` SWEEP, not a campaign** — TraLO appears at 1e-4/gate-off (96 runs), 1e-4/gate-on (48), 5e-5 (48) and 5e-6 (48). Pooled, it reports **−16.7 pp macro-F1** and reads as "constrained training destroys the representation". Contract-filtered it reports **−1.74 pp**. **~15 pp of that effect was the learning-rate trap.** Fixed by `honours_contract()` (`lr_constraint == lr`, CE gate off, post-hoc methods exempt since they never train past warm-up).

⚠️ Not fixable retroactively: `fioretto_ldf` and `hounie_rcl` never *declare* `enable_ce_skip` in this corpus, so their gate sits at their code default while TraLO's is explicit. **TraLO-vs-dual numbers from this reference inherit that asymmetry; TraLO-vs-post-hoc does not**, and post-hoc is the bar that matters.

## Rejected experimental designs — these manufacture fake wins

| Design flaw | What it fabricates | Rule |
|---|---|---|
| **Unequal compute vs the clipper** | At warm-up 1 the clipper baseline is a *1-epoch* model. TraLO gets 1 + 29. This reads as **+7 to +9 pp**. Give the clipper the same 30 epochs and the sign flips to **−0.85 pp**. | **Always score against a clipper trained for the same total epochs.** `results/headroom` has warm-up-30 clipper arms (n=48) for exactly this. |
| **Flags that reach only one arm** | `--no-ce-skip` guarded on a key only TraLO declares, so the campaign ran the gate OFF for TraLO and ON for both duals. Artifact ≈ 0.22 cc-F1 against a 0.019–0.031 margin. | **md5 the raw predictions between arms before reading any metric.** If they match, the flag is inert. |
| **`lr_constraint` left at its default** | Constraint phase runs at 5e-6 vs warm-up's 1e-4, so matching *epochs* compares learning rates, not objectives. | Set `lr_constraint == lr` for short-warm-up comparisons. Not in the warm-up cache key, so caches are reused. |
| **Pooling across cells** | Averaging L30/L50 × backbones into one number hides sign flips. | Atomic cell = (dataset, backbone, cap) over seeds. **Count cells, never pool.** |
| **Raw (uncentered) correlations** | `count_cv` reads ρ = −0.847 raw and −0.165 within-cell — the raw number is dataset identity. | Centre within (dataset, model, cap) before correlating. |
| **Mixing epoch conventions** | TraLO logs *absolute* `Epoch`; both duals log *relative* `epoch` from 0. Subtracting warm-up from a dual fabricates "Fioretto runs 250 epochs vs TraLO's 34". | Per-method convention, and never use row count for epochs (TraLO logs sparsely). |

## How to add a new candidate cleanly

If a future backbone or dataset is to be tried:
1. Read this file first. If the candidate appears above, the burden of proof for retrying it rests on you.
2. Add a wrapper to `src/models/imagery/` (backbone) or a prep script + entry to `IMAGERY_DATASETS` and `DATASETS` (dataset).
3. Probe → cap → smoke following the model-search workflow (see the active `gen_*` generators in `src/config_generators/`; the original `gen_model_search.py` was retired 2026-05-28).
4. If it fails, **append the verdict to this file** with the same row format. Don't silently drop it.

If a future **loss / method arm** is to be tried:
1. State what the gradient is a **function of**. If the answer is "the aggregate count", it belongs to the family that already tied 13 times — justify it or pick something else.
2. Run at **warm-up 1**, CE alive (`enable_ce_skip=False`), `lr_constraint == lr`.
3. Score at **equal compute** against the warm-up-30 clipper arms, on budget-equalized macro-F1 and AP, counting cells.
4. Check the arm is **live by md5** before reading any metric.
5. Append the verdict here, win or lose.
