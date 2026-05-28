# Paper Coverage Plan v2

**Last updated:** 2026-05-24
**Owner:** roei
**Architecture:** all runs on dsisco02 Blackwell (RTX PRO 6000 96 GB). Turing results invalidated and excluded.

This file is the **single source of truth** for what experiments the thesis paper needs. It is structured as: (1) locked dataset/method choices, (2) the dictionary of tables we are converging to, (3) per-table TODO checklist with current completion counts, (4) review gates.

Update this file in place as cells complete. Do not let memory drift again.

---

## 1. Locked datasets and stories

| Dataset      | n_classes | Story class                   | Group column   | Notes |
|--------------|-----------|-------------------------------|----------------|-------|
| tissuemnist  | 8         | `cls=4` GE (7.1 % test)       | `synth_group`  | Only synthetic groups available; near-balanced (7.0 % vs 7.2 %). Local constraint barely binding → stress-tests optimizer when constraint is loose per-group. |
| dermmnist    | 7         | `cls=4` MEL (11.1 % test)     | `loc_group`    | Real attribute (anatomical site). 3 groups, sharply imbalanced for MEL: 9.5 % / 11.5 % / 17.0 %. Strongest fairness story. |
| aider        | 4         | `cls=0` collapsed_building (8.6 % test) | `synth_group` | Emergency-triage framing. Synthetic groups, near-balanced (7.5 % vs 9.7 %). |

**Dropped:** eurosat, so2sat. Do not re-introduce.

**Group column candidates for ablation (Table E only):** dermmnist also has `sex` (binary, balanced — MEL 12.7 % vs 9.4 %).

**Alternate constrained classes for multi-class robustness (Table D):**
- dermmnist: AKIEC (cls=0, 3.2 %), BCC (cls=1, 5.1 %), BKL (cls=2, 11.0 %), MEL (cls=4, 11.1 %) — 4 classes total. Skip DF (1.1 %) and VASC (1.4 %) — too rare, K collapses.

---

## 2. Locked method, backbone, tightness, seed sets

| Axis | Values | Count |
|------|--------|-------|
| Methods | tralo (TraLO_fix: undershoot_hinge + reset_optimizer_at_sat + fior_beta=0.5), tralo_bounded, fioretto_ldf, hounie_rcl, danits_lp, heuristic | 6 |
| Backbones | MobileNetV3, ResNet18, EfficientNetB0 | 3 |
| Symmetric tightness | L20_G20, L30_G30, L50_G50, L70_G70, L80_G80 | 5 |
| Asymmetric tightness | full 5×5 grid: L∈{20,30,50,70,80} × G∈{20,30,50,70,80} | 25 |
| Seeds | 1, 2, 3, 4 | 4 |

---

## 3. Table dictionary — what the paper actually presents

Each table is a separate paper claim. Tables share cells where they overlap (cells deduped across tables for the run plan).

### Table A — Headline method comparison
**Claim:** TraLO beats baselines on the canonical transductive setup across all three datasets.
**Axes varied:** dataset × method × seed
**Axes fixed:** backbone=MobileNetV3, story class+group per dataset, 5 symmetric tightness
**Cell count:** 3 ds × 5 tight × 6 methods × 4 seeds = **360**

### Table B — Asymmetric tightness robustness
**Claim:** TraLO win holds when local ≠ global tightness.
**Axes varied:** (L,G) pair × method × seed
**Axes fixed:** dataset=dermmnist (richest groups), backbone=MobileNetV3, cls=MEL, group=loc_group
**Cell count:** 25 (L,G) × 6 methods × 4 seeds = **600**
**Includes:** the 5 symmetric cells already counted in Table A on derm. Net new: 20 × 6 × 4 = **480**.

### Table C — Backbone robustness
**Claim:** Win is not an artifact of MobileNetV3.
**Axes varied:** backbone × method × seed
**Axes fixed:** dataset=dermmnist, cls=MEL, group=loc_group, 5 symmetric tightness
**Cell count:** 3 bb × 5 tight × 6 methods × 4 seeds = **360**
**Net new vs Table A:** 2 backbones × 5 × 6 × 4 = **240**.

### Table D — Multi-class robustness
**Claim:** Win is not specific to the chosen story class.
**Axes varied:** constrained class × method × seed
**Axes fixed:** dataset=dermmnist, backbone=MobileNetV3, group=loc_group, 5 symmetric tightness
**Cell count:** 4 cls × 5 tight × 6 methods × 4 seeds = **480**
**Net new vs Table A:** 3 alt classes × 5 × 6 × 4 = **360**.

### Table E — Group-column ablation
**Claim:** Win holds across group definitions (real vs synthetic-like grouping).
**Axes varied:** group column × method × seed
**Axes fixed:** dataset=dermmnist, backbone=MobileNetV3, cls=MEL, 5 symmetric tightness
**Cell count:** 2 grp × 5 tight × 6 methods × 4 seeds = **240**
**Net new vs Table A:** 1 alt group × 5 × 6 × 4 = **120**.

### Table F — Component ablation [COMPLETE]
**Claim:** Minimal essential recipe.
**Status:** done — 28 cells in `results/pending_runs/component_ablation`. Locked: warmup, undershoot_hinge, reset_optimizer_at_sat, ce_skip are essential; freeze_on_satisfy and rho_sched are inert.

### Table G — KL ablation [COMPLETE]
**Status:** done — 16 cells in `results/pending_runs/kl_ablation`. Locked: alpha_kl=0 default; KL only marginally helps on derm-like cases.

### Figure: Convergence dynamics [COMPLETE on tissue+eurosat]
Already plotted from existing logs. Will refresh from new dermmnist headline cells once those land.

---

## 4. Unique target cells (deduped across tables A–E)

**Total: 1,560 unique cells.**

Already complete that map onto v2 plan: **163 (10.4 %)**.
Remaining to run: **1,397 cells**.

At observed Blackwell throughput (~37 cells/hr on GPU 3, single-tenant), single GPU ≈ **38 hrs**; 2 GPUs ≈ **19 hrs**.

### Sources audited (2026-05-24)

| Location | eval_metrics count | Usable for v2 plan? |
|---|---|---|
| `results/pending_runs/` (Blackwell) | 393 | Yes — 163 in-scope, 150 out-of-scope (102 eurosat + 48 asymmetric on tissue/aider) |
| `archive_experiments/sweep40_2026-04-15/` | 64 | No — pre-breakthrough, old "our_approach" methodology name, no TraLO_fix HPs |
| `archive_experiments/frozen_lambda_ablation_2026-04-30/` | 8 | No — same reason |
| `results/baselines/` | 4 | No — same reason |
| `paper_results/` | 0 | scripts + metadata smokes only, no experiments |
| dsisco01 paths | (mirror of dsisco02 via shared NFS `/home`) | counted once |

**Single source of truth: `results/pending_runs/` on dsisco02 (NFS-shared with dsisco01).**

---

## 5. TODO checklist — what to run, in priority order

Mark each line `[x]` only after (a) all configs in the slice completed without failure, (b) the per-slice review gate (§6) passed, (c) any anomalies surfaced.

### Phase 1 — Close Table A (headline) [360 / 360 done = 100 %] ✅

**Status:** ✅ COMPLETE — sweep finished 2026-05-24 13:05 UTC in 2h23m (0 failures, avg 39s/cell)
Aggregator outputs in `docs/table_a_summary.md`, `table_a_summary.csv`, `table_a_raw.csv`.

(prior status, kept for log) Was: 🟢 RUNNING on dsisco02 GPU 3 since 2026-05-24 10:41 UTC (PID 3208961)
**Generator:** `src/config_generators/gen_paperv2_phase1.py` (221 cells queued at `results/pending_runs/paperv2_phase1/`)
**Launcher:** `/tmp/launch_paperv2_phase1.sh` (setsid+nohup, log `~/OptimizationLoss/logs/paperv2_phase1.log`)
**Aggregator:** `src/evaluation/table_a_summary.py` — run after sweep completes; emits `docs/table_a_raw.csv`, `docs/table_a_summary.csv`, `docs/table_a_summary.md`
**ETA:** ~5.5 hrs (observed ~90s/cell with warmup cache hit). Watch with `tail -f logs/paperv2_phase1.log` and `find results/pending_runs/paperv2_phase1 -name evaluation_metrics.csv | wc -l`.

- [x] **A.tissuemnist** — 42 cells run, 120/120 complete
- [x] **A.dermmnist** — 90 cells run, 120/120 complete
- [x] **A.aider** — 89 cells run, 120/120 complete

**Phase 1 total queued:** 221 cells. **Gate:** run the aggregator; verify F1m winner per (ds, tight) is TraLO or document gap; verify posthoc-flips ratio TraLO ≤ baseline on every cell; log anomalies to §8.

### Phase 2 — Close Table B (asymmetric on derm) [🟢 RUNNING]

**Status:** ✅ COMPLETE — 456/456 cells finished 2026-05-24 (600/600 incl. earlier overlap). Aggregator (Table B heatmap) still TODO.

- [x] **B.dermmnist** — full 5×5 (L,G) grid, 6 mthd, 4 seed → 600/600 complete

**Gate:** heatmap of TraLO − best-baseline F1 across the 25 (L,G) cells; confirm no asymmetric corner where TraLO collapses.

### Aider — FROZEN

Aider experiments are frozen at the Phase 1 results (120 cells). Showcase materials in `docs/aider_results/` (README + per-class CSV + head-to-head + per-seed CSV). Decision pending with thesis advisor — current framing is "easy-task regime ablation" where warmup saturates and TraLO ties F1m but dominates Flips.

### Phase 3 — Close Table C (backbones on derm) [🟢 RUNNING]

**Status:** 🟢 RUNNING on dsisco02 GPU 3 since 2026-05-24 23:44 UTC (PID 2543117)
**Generator:** `src/config_generators/gen_paperv2_phase3.py` — 240 cells queued at `results/pending_runs/paperv2_phase3/`
**Launcher:** `/tmp/launch_paperv2_phase3.sh`, log `~/OptimizationLoss/logs/paperv2_phase3.log`
**ETA:** ~6 GPU-hrs (fresh warmup caches for ResNet18/EfficientNetB0 add a little overhead)
**Note:** derm warmup is NOT saturated (acc ~0.88→0.98), unlike aider — so TraLO has genuine room to work on these backbones.

- [ ] **C.dermmnist.ResNet18** — 5 sym tight, 6 mthd, 4 seed → 120 new cells
- [ ] **C.dermmnist.EfficientNetB0** — 5 sym tight, 6 mthd, 4 seed → 120 new cells

**Gate:** rank table (TraLO position 1–6 per backbone) shows TraLO=1 on all 3 backbones, or document where it isn't.

### Phase 4 — Close Table D (multi-class on derm) [⏸ QUEUED on GPU 3 after Phase 3]

**Generator:** `src/config_generators/gen_paperv2_phase4.py` — 360 cells at `results/pending_runs/paperv2_phase4/`
**Launch:** auto via primary chain `/tmp/chain_primary.sh` (EXPERIMENT_DIR=.../paperv2_phase4), log `logs/paperv2_phase4.log`

- [ ] **D.dermmnist.cls=0 AKIEC** — 5 sym tight, 6 mthd, 4 seed → 120 cells
- [ ] **D.dermmnist.cls=1 BCC** — 5 sym tight, 6 mthd, 4 seed → 120 cells
- [ ] **D.dermmnist.cls=2 BKL** — 5 sym tight, 6 mthd, 4 seed → 120 cells

**Gate:** TraLO wins F1 macro on at least 3/4 classes, or we document the failure mode.

### Phase 5 — Close Table E (group ablation on derm) [⏸ QUEUED on GPU 3 after Phase 4]

**Generator:** `src/config_generators/gen_paperv2_phase5.py` — 120 cells at `results/pending_runs/paperv2_phase5/`
**Launch:** auto via primary chain (EXPERIMENT_DIR=.../paperv2_phase5), log `logs/paperv2_phase5.log`

- [ ] **E.dermmnist.sex** — MobileNetV3, MEL, sex, 5 sym tight, 6 mthd, 4 seed → 120 cells

**Gate:** confirm TraLO win persists under sex grouping (weaker local imbalance); document delta vs loc_group.

### Phase 6 — Tissue backbone robustness (spillover, opportunistic 2nd GPU)

**Generator:** `src/config_generators/gen_paperv2_phase6.py` — 240 cells at `results/pending_runs/paperv2_phase6/`
**Launch:** secondary watcher `/tmp/watch_secondary.sh` grabs the FIRST single-tenant-clear GPU among 0/1/2 and runs this (EXPERIMENT_DIR=.../paperv2_phase6), log `logs/paperv2_phase6.log`. Separate sweep root → no collision with the GPU-3 primary chain.

- [ ] **F.tissuemnist.ResNet18** — 5 sym tight, 6 mthd, 4 seed → 120 cells
- [ ] **F.tissuemnist.EfficientNetB0** — 5 sym tight, 6 mthd, 4 seed → 120 cells

**Gate:** mirrors Phase 3 gate — TraLO rank across backbones on a second dataset.

### Orchestration (live 2026-05-24)

- **Primary chain** PID 2577670 (`/tmp/chain_primary.sh`): GPU 3, waits for Phase 3 (PID 2542974) → Phase 4 → Phase 5. Robust wait via `kill -0 <pid>`, not pgrep.
- **Secondary watcher** PID 2579554 (`/tmp/watch_secondary.sh`): polls GPUs 0/1/2 every 60s; launches Phase 6 on first one that is single-tenant clear (double-checks 3s later to dodge races).
- Collision safety: each dispatcher pinned to its own `EXPERIMENT_DIR` sweep root, so two GPUs never grab the same config (main.py reads `EXPERIMENT_DIR`, default `results/pending_runs`).

### Phase 6 — Aggregation and writing

- [ ] **Build aggregator** that reads every `evaluation_metrics.csv` under `results/pending_runs/` and emits tables A–E as CSV + LaTeX
- [ ] **Significance tests** — paired bootstrap on F1 macro, TraLO vs each baseline, per cell
- [ ] **Convergence figure refresh** from dermmnist + aider Phase 1 logs
- [ ] **Update paper draft** §results with the v2 table set

---

## 6. Per-slice review gate — checklist before marking `[x]`

For each slice closed, verify and record:

1. **Completion**: count of `evaluation_metrics.csv` matches expected cell count, zero failures
2. **Sanity ranges**: F1 macro ∈ [0.3, 1.0]; ECE ∈ [0, 0.5]; satisfaction epoch ≤ constraint_epochs
3. **TraLO vs baselines** on this slice: which method wins on F1 macro, mean across seeds, with std; gap > 1 std = clean win, otherwise "tie"
4. **Posthoc flips**: TraLO / best-baseline ratio per cell; expect TraLO ≤ baseline
5. **Anomaly note** (if any): any cell with F1 outside sanity range or satisfaction failure — log in §8 below

---

## 7. Run-plan logistics

- Generator scripts go in `src/config_generators/gen_paperv2_<phase>_<slice>.py`.
- Output directory: `results/pending_runs/paperv2_<phase>_<slice>/<dataset>/<model>/cls_<c>/grp_<g>/<tight>/<method>/seed_<s>/`.
- `main.py` dispatches pending runs sequentially (or pinned to one GPU via `CUDA_VISIBLE_DEVICES`).
- All caches in `model_cache/` are Blackwell-warmup; do not invalidate without flagging here.
- GPU discipline: max 2 GPUs on dsisco02, never share a single GPU with another user (driver-crash risk per `feedback_gpu_sharing`).

---

## 8. Anomalies log

(append-only as Phase work proceeds)

| Date | Slice | Issue | Resolution |
|------|-------|-------|------------|
| 2026-05-24 | initial audit | discovered derm baselines were already on loc_group, not sex as previously assumed | no rerun needed; updated §3 derm story doc |
| 2026-05-24 | post-breakthrough audit | confirmed `results/pending_runs/` (393 cells) is the only valid result set; archives are pre-breakthrough April runs (methodology="our_approach", no TraLO_fix HPs); dsisco01 paths are NFS-shared mirror, not duplicates; paper_results/ has no actual experiments | aggregator confirms 163/1560 v2 cells done; no hidden results elsewhere |

---

## 9. Decisions to revisit

- Whether Table D needs all 4 derm classes or 3 is enough — depends on first 2 results
- Whether Table B's 25-cell asymmetric grid can be coarsened to 13 cells (drop diagonal-adjacent) to save ~14 hrs — decide after Phase 1 baselines on derm land
- Aider trainables performance is unknown beyond a single smoke cell — re-evaluate aider's paper weight after Phase 1.A.aider lands
