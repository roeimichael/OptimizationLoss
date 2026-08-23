# TRALO: WHERE TO GO NEXT -- HP search, and the directions that are still open

Written 2026-08-23, after the dataset scouting run. Not an operational document;
`docs/FRAMEWORK.md` still is. This prices directions before any GPU is spent.

---

## 0. THE ONE-PARAGRAPH ANSWER

**Hyperparameter search on the campaign objective is arithmetically hopeless at this
project's noise floor, and §1 proves it with a simulation you can re-run in three
seconds.** With the standard 4 seeds and only 50 configurations, a search over
configurations that are *all identical* reports a winner at **+3.04 items**, and clears
+2 items **97% of the time** -- against a total `clip`-to-perfect headroom of 1.9-9.9
items. Any HPO run at campaign scale will manufacture the entire prize out of nothing.
The fix is not a better sampler; it is **selection on few configs plus a confirmation
campaign on fresh seeds** (§2). Separately, §3 argues the highest-value direction is not a
better penalty at all: three independent measurements say the count-penalty CHANNEL is
the bottleneck, not its shape, and there is exactly one untried channel left (§4).

---

## 1. 🚨 WHY NAIVE HPO WOULD FABRICATE A RESULT HERE

The objective is noisy: the paired per-seed sd is **~2.7 items** (FRAMEWORK). HPO reports
the **max** over N configurations, and the max of N noisy draws is upward-biased even when
every configuration is identical. Simulated, 20,000 repetitions, **true effect exactly
zero everywhere**:

| configs N | seeds/config | sd of the mean | **E[reported winner]** | P(reports >2 items) |
|---|---|---|---|---|
| 10 | 4 | 1.35 | **+2.08** | 51% |
| 20 | 4 | 1.35 | **+2.52** | 76% |
| 50 | 4 | 1.35 | **+3.04** | **97%** |
| 100 | 4 | 1.35 | **+3.38** | **100%** |
| 100 | 8 | 0.95 | +2.39 | 83% |
| 100 | 16 | 0.68 | +1.69 | 15% |

And the seeds required to hold the winner's inflation under **one item**:

| configs N | seeds needed per config | total runs |
|---|---|---|
| 10 | 34 | 340 |
| 20 | 44 | 880 |
| 50 | 58 | 2,900 |
| 100 | 68 | 6,800 |

🛑 **Read the 50-config row against the project's own scale.** The whole distance from
`clip` to a PERFECT allocator is 1.9-9.9 items. A 50-config sweep at 4 seeds returns
+3.04 items **from pure noise** -- between a third and the whole of the prize. This is the
same failure that produced the retracted single-cap claims, mechanised and run at scale.

Reproduce:

```bash
python - <<'PY'
import numpy as np
rng = np.random.default_rng(0); SD = 2.7
for N in (10, 50, 100):
    d = rng.normal(0, SD/2, size=(20000, N)).max(1)     # 4 seeds -> sd/2
    print(N, round(d.mean(), 2), round((d > 2).mean(), 2))
PY
```

### What the StrategyBuilder code offers, and it is not this

`C:\Users\roeym\Desktop\projects\StrategyBuilder\src\domains\backtests\optimizer.py` holds
the only optimizer in that repo (`src/domains/optimizations/` is request validation and
dispatch, no algorithm). Grid / random / Optuna-TPE, selected by a string.

**It has no machinery for a noisy objective, because it never needed one** -- a backtest
over fixed bars is deterministic, so the objective is a single number per config. Verified
absent across the whole repo: repeated evaluation over seeds, variance-aware acquisition,
racing or sequential halving, bandit allocation, multiple-comparison correction,
significance gating of the winner. Ranking is a raw point-estimate sort. Three further
traps if it is ported as-is:

* **Random search uses the global unseeded `random` module** -- not reproducible.
* **Every axis is `suggest_categorical`**, so TPE cannot exploit ordinal structure: it
  treats `lambda_step in {0.01, 0.05, 0.25}` as unordered labels.
* **No pruner is ever passed** and `trial.report()` is never called, so Optuna's pruning
  is inert; the `TrialPruned` raises there are error handling, not early stopping.

Worth stealing regardless: `_sample_combos` (memory-bounded sampling from a discrete
product space without materialising it), `_walk_forward_sizes` (closed-form IS/OOS window
geometry), the `method_policy` block that reports requested-vs-actual method honestly, and
the `diagnostics` counter block. That is ~60 lines. Everything else is trading-hardwired.

---

## 2. ✅ THE HPO DESIGN THAT WOULD BE LEGITIMATE

If a search is run anyway, these are the conditions under which its output would mean
something. All four are required; dropping any one returns you to §1.

1. **Gate it first with `full_panel`'s RESOLUTION block.** It already prints the within-
   cell seed sd and the seeds needed at 80% power. On the live `dualbar2` one contrast
   read `observed +0.36 items, needs ~174 seeds per cell`. **If the cell you intend to
   optimise needs 174 seeds to see its own effect, no sampler helps.** Run this before
   writing any search space.
2. **Few configs, then CONFIRM on fresh seeds.** Selection bias lives entirely in the
   selection step. Search over <=10 configs on seeds {1,2,3,4}, then run the winner AND
   the incumbent on seeds {5,6,7,8} as a separate campaign and report only that. A
   confirmation run has no max-over-N bias, which is why it is the whole fix.
   `paper/data/corpus/r2_seeds10.csv` shows this project already extends to 10 seeds.
3. **Score the paired difference against each arm's OWN `_null` twin**, never the raw
   metric -- `scripts/paired_seeds.py` already does this. And carry `tralo_reseed`:
   the constraint moves the capped count RMS 75-95 items while a pure reseed moves it
   83-95, so without the reseed arm the search will optimise the RNG.
4. **Put a KNOWN-DEAD config in the search space as a liveness control.** If the search
   ranks `constraint_random_direction: true` or a `lambda_step` two orders below the
   measured floor above the incumbent, the search is reading noise and its winner is void.
   This is the same discipline as `--self-test` on `straddle_probe`.

**Where a search is actually affordable: the CPU probes, not the campaign.** Every offline
probe (`frozen_head_probe`, `scope_probe`, `graph_probe`, `straddle_probe`) runs in minutes
against artefacts that already exist. Optimising there costs no GPU. ⚠️ But
`frozen_head_probe` **does not transfer to iwildcam** -- its resolution there is 35.09
items against a 1.9-9.9 item question -- so on iwildcam every `NO DIFFERENCE` it returns is
an absence of measurement. Re-measure its resolution on any new dataset before trusting it
as a surrogate.

**The search space, if it comes to that.** From `configs/protocol.yml`, the knobs that are
not already closed by FRAMEWORK section 2: `lambda_step`, `lambda_global`, `lambda_local`,
`initial_rho`/`rho_target`, `cut_window_items` (only live under `soft_count_mode: margin`),
`straight_through`, `constraint_grad_mode`, `constraint_fp32`, `stable_count_threshold`.
🛑 **`penalty_shape` is NOT in that list** -- thirteen arms varied it and all tied
(`src/losses/transductive_loss.py` docstring), and `lr_constraint` must equal `core.lr` or
you reproduce the -16.7pp LR fabrication.

---

## 3. 🔬 THE DIAGNOSIS: IT IS THE CHANNEL, NOT THE SHAPE

Three measurements already in the repo say the same thing from three directions, and they
are why a hyperparameter is unlikely to be the answer:

1. **The penalty is a function of the AGGREGATE COUNT; the allocator scores only the
   RANKING.** An aggregate-count gradient cannot reorder two items. Stated in
   `src/losses/transductive_loss.py`'s own docstring, and it is why thirteen shape
   variants tied.
2. **Score-pushing does not fix the ranking either.** `rank`, `rankpair` and
   `budget_margin` each add a term that moves the score ordering while leaving the
   classification loss alone. All three are null (`src/methodologies/select/train.py`).
3. **Reweighting the TRAIN loss makes it worse.** `select` (SelectiveNet-style, per-class
   coverage from the budget) is the worst arm measured: **-22 items** vs `clip`, 0 of 2
   cells on every metric, and its own `select_null` ties `clip` -- so the loss is the
   selective term itself.

Add the delivery-side facts: the constraint gets **29 steps against CE's 3654**, and under
`normalize` the delivered displacement is **exactly `lr * clip`**, so magnitude is void and
only DIRECTION and STEP COUNT are live levers.

🔑 **Every arm to date pushes on the model through one of three channels -- an aggregate
count penalty, a score-ordering term, or a reweighted train loss -- and all three are
measured shut.** A hyperparameter moves the dose within a shut channel.

---

## 4. 🟡 THE DIRECTIONS THAT ARE ACTUALLY OPEN

Each is stated with its mechanism, why it is not already in FRAMEWORK section 2, an
**explicit novelty check against the literature**, and a pre-registered kill criterion.

### D1 (highest value, and it costs almost nothing): run the EXISTING TraLO on fMoW

Not a method idea -- a **confound resolution**. Everything in §3 was measured on
`iwildcam`, where the representation channel measured NEGATIVE and the constraint moves
the count below reseed noise. Today's scouting found `fmow` (see
`docs/dataset_scouting_2026-08-23.md`): NET +2969 with an ATOMIC group, and **11 of 20
per-group ceilings at K=0 on capped classes of 408 and 511 items**, against iwildcam's 7
of 14. A K=0 ceiling is the one structure that makes the LOCAL scope bind hard.

**The question it answers is the one that blocks everything else: is the method dead, or
was the dataset?** Nothing else on this list is worth running until that is known.

* Cost: 1.65 GB, `python -m scripts.prep_fmow --out data/fmow/oodslice`, then one
  campaign at the standard protocol. No new code, no new arm.
* Novelty: not applicable -- this is a replication, not a claim.
* **Kill criterion:** if `tralo` minus its own `_null` twin is <= 0 on `d capF1` in both
  cap levels with the RESOLUTION block showing the cells are powered, the count-penalty
  direction is finished across two independent datasets and the paper should say so.

### D2: the one untried CHANNEL -- per-item targets on the TEST images

**Mechanism.** Every existing arm computes an aggregate over test items. Instead: run the
allocator on the current test scores to get a **feasible hard assignment** (it already
respects every global and per-group budget by construction), treat that assignment as
pseudo-labels, and take ordinary CE steps on the TEST IMAGES against them, interleaved
with train CE. The gradient is then per-item and there are thousands of steps' worth of it
rather than 29 aggregate ones -- which attacks §3's items 1 and the 29-vs-3654 problem at
the same time.

**Why it is not in section 2.** The rejected list covers penalty shapes, step counts,
dedicated optimizers, joint objectives, hinges, granularity, score-ordering terms, and a
selection head that reweights the TRAIN loss. **No arm has ever put a per-item target on a
test image.** The closest is `select`, which reweights training-set CE by a coverage head.

🛑 **NOVELTY CHECK -- THIS IS NOT NOVEL AS A MECHANISM, AND IT MUST NOT BE CLAIMED AS
ONE.** Confirmed prior art:

* **Sinkhorn Label Allocation: Semi-Supervised Classification via Annealed Self-Training**
  (Tai, Bailis & Valiant, ICML 2021) -- self-training in which pseudo-labels are assigned
  by solving an optimal-transport problem under **class-count constraints**. That is this
  idea.
* **Beyond Invariance: Test-Time Label-Shift Adaptation** (NeurIPS 2022), plus the
  distribution-alignment family (ReMixMatch, DebiasPL) and classic transductive SVMs,
  which enforce a specified positive fraction on the test predictions.

✅ **What would be ours, and it is narrow: the budgets are PER GROUP and many of them are
K = 0.** SLA constrains one global class prior; a zero ceiling on a specific group ("none
of this species at this camera", "no BCC in this subpopulation") is a different and
strictly harder constraint, and it is exactly the structure the dataset screen selects
for. **So D2 enters the paper as a BASELINE WE MUST BEAT OR ADOPT, not as a contribution
-- and a reviewer will ask for it either way.** If it beats TraLO, the honest paper says
so; that is still a result, and a better one than a fourteenth penalty shape.

* Cost: one new methodology module reusing the existing allocator; ~150 lines.
* **Controls, non-negotiable:** a `_null` twin whose pseudo-labels come from an
  UNCONSTRAINED top-K (isolates the constraint from the self-training), and
  `tralo_reseed`. Without the first, self-training's own gain is attributed to the cap.
* **Kill criterion:** if the constrained-pseudo-label arm ties its unconstrained-
  pseudo-label twin, the count information contributed nothing and only self-training did.

### D3: spend the compute finding, rather than fighting it

`project_win_is_compute_not_method_2026-08-23` records **92/8**: every TRAINED method beats
the clipper by 1.1-1.9 pp while the POST-HOC `danits_lp` loses, and TraLO's own part is
+0.15 pp -- under its own bound. That is a real, large, reproducible effect that the
current paper frames as a method win. **Reframing it as a compute/adaptation finding costs
zero GPU and is defensible**, whereas the +0.15 pp method claim is not. This is a writing
decision, not an experiment, and it is the highest expected-value item on this page after
D1.

### D4: what NOT to spend on

⛔ Any penalty shape, any rho/lambda schedule, more constraint steps, a dedicated
constraint optimizer, the joint objective, the undershoot hinge, finer granularity, a KL
anchor, local-cap scope pinning, graph diffusion over embeddings, and any global-cap
variant. All measured, all in FRAMEWORK section 2, and 2(j) makes the global cap
structurally impossible rather than merely unmeasured. ⛔ And do not re-run `select`,
`rank`, `rankpair` or `budget_margin` at any dose.

---

## 5. THE ORDER

1. **`full_panel --campaign results/iwc2` RESOLUTION block** on the cells you care about.
   If they need ~174 seeds, stop and read §1 again before proposing any search.
2. **D1: fMoW acquisition + one standard campaign.** 1.65 GB. Answers "method or dataset".
3. **D3: reframe the compute finding** while D1 runs. Zero GPU.
4. **D2 only if D1 shows the constraint channel is alive somewhere** -- and built with its
   unconstrained-pseudo-label twin from the first commit, not added later.
5. **HPO last, and only under §2's four conditions**, over <=10 configs with a
   fresh-seed confirmation run and a known-dead config in the space as a liveness control.
