# Project cleanup plan

**Trigger**: when phase 2 results (polling task `bbswopuw3`) confirm the
tralofix pattern holds across tightness scan + ablations, spawn the
agents below in parallel.

**Why a plan**: separating regimes so each agent has a tight, isolated
scope. Avoids one agent stepping on another's diff. Five agents in
parallel finish in a fraction of the time of one.

---

## Trigger conditions (must hold to start cleanup)

- [ ] tralofix wins flips on the tight cells in n=8 results (already true)
- [ ] tralofix doesn't catastrophically lose on the new tightness cells
      (L70_G70, L80_G80, asymmetric pairs)
- [ ] component ablation shows hinge + Adam reset carry the lift
      (not some other piece)

If any of the above fails: STOP — re-investigate before cleanup. Don't
delete things we might still need.

---

## Regime A — Methodology code cleanup
**Agent type**: simplifier (or general-purpose)
**Scope**: `src/methodologies/`, `src/losses/transductive_loss.py`

Tasks:
1. **Remove dead knobs from `transductive_loss.py`**:
   - `linear_sat_tail` (May-11 breakthrough, superseded by undershoot_hinge)
   - all referring code in `src/methodologies/tralo/train.py`
   - all referring code in `paper_results/build_paper_artifacts.py`
2. **Remove diagnostic flags from `src/methodologies/tralo/train.py`**:
   - `fix_chunk_scaling` (flagged + reverted)
   - `separate_optimizers` (diagnostic only)
   - `ce_grad_clip` (diagnostic only)
3. **Remove failed ablation modes from `src/methodologies/tralo_fioretto/train.py`**:
   - `symquad` mode (proved worse on flips at n=4)
   - `dual_lambda` mode (proved worse)
   - `single_lambda` mode (proved worse — hybrid_v1 null)
   - Keep only: `bounded_only`, `undershoot_hinge`
4. **Update `MulticlassTransductiveLoss.__init__` signature**: drop
   `linear_sat_tail` param entirely.
5. **Verify tests/imports still pass** after each deletion.

**Out of scope**: don't rename `tralo_fioretto` → `tralo` here. That's
Regime D so it can be coordinated with the runner registry update.

---

## Regime B — Results + experiment dir cleanup
**Agent type**: general-purpose
**Scope**: `results/`, `paper_results/`, `archive_experiments/`

Tasks:
1. **Identify failed-sweep dirs** in `results/pending_runs/` that won't
   be re-run. Candidates:
   - `hybrid_v1/` — null result, kept for history. **Move to
     `archive_experiments/hybrid_v1_null/`**.
   - `hybrid_v2/` — drift discovery. **Move to
     `archive_experiments/hybrid_v2_drift/`** (small, ~20 configs).
   - `hybrid_v3/` — proved Adam reset fix. **KEEP** (referenced by
     `project_tralofix_breakthrough.md`).
   - `oscillation_*` — appendix material per memory. **KEEP**.
2. **Stale `paper_results/` files**: `_log_smooth*.csv`,
   `dataset_smoke*.csv`, `plot_smooth_comparison.py`,
   `smoke_eurosat.py`, `patch_html_figures.py`. Likely safe to delete;
   check git blame for last touch.
3. **Audit `archive_experiments/`** for already-deleted dermmnist
   results that can be compressed/dropped.

**Don't touch**: `model_cache/` (warmup caches still load!) ,
`data/tissuemnist/`, `data/eurosat/`, `data/so2sat/`.

Report back with proposed deletion list — DO NOT actually delete
without user confirm.

---

## Regime C — Config generators cleanup
**Agent type**: general-purpose (small task)
**Scope**: `src/config_generators/`

Tasks:
1. List all `gen_*.py` files; mark each as:
   - **Keep** (paper sources): `gen_paper400.py`, `gen_paper400_tralofix.py`,
     `gen_paper400_baselines.py`, `gen_tralofix_extra_seeds.py`,
     `gen_tralofix_tightness_scan.py`, `gen_component_ablation.py`,
     `gen_kl_ablation.py` (only if KL helps; else move to regime B archive).
   - **Archive**: `gen_hybrid_v1.py` (if exists), `gen_hybrid_v2.py`,
     `gen_hybrid_v3.py` (move to `archive_experiments/`).
2. Audit `generate_configs.py` shared module: drop any dead helper funcs
   if used only by archived gens.

---

## Regime D — Codebase rename (tralo_fioretto → tralo)
**Agent type**: general-purpose
**Scope**: methodology dirs + runner registry + config files

Tasks:
1. **Rename `src/methodologies/tralo/` → `src/methodologies/tralo_bounded/`**
   (the original bounded-only version becomes the ablation).
2. **Rename `src/methodologies/tralo_fioretto/` → `src/methodologies/tralo/`**
   (the proposed method).
3. **Update `src/experiments/runner.py` registry**:
   - `'tralo'` → new full method (was `tralo_fioretto`)
   - `'tralo_bounded'` → bounded-only (was `tralo`)
   - **Alias** `'tralo_fioretto'` → new tralo for backward-compat on
     already-completed configs.
4. **Update imports across the codebase**: scan for
   `from src.methodologies.tralo_fioretto` and rewrite.
5. **Update `gen_*.py`**: `methodology: "tralo_fioretto"` → `"tralo"`
   for new configs. Old completed configs keep their existing string
   (resolved via alias).
6. Smoke test: run one config from each renamed methodology to verify
   dispatch works.

**Coordination warning**: do not run while a sweep is dispatching
new experiments. Wait for chain to be idle.

---

## Regime E — Memory + docs cleanup
**Agent type**: general-purpose (small)
**Scope**: `~/.claude/projects/.../memory/`, `CLAUDE.md`, `docs/`

Tasks:
1. **Memory entries to remove** (superseded findings):
   - `project_hybrid_v1_finding.md` (replaced by tralofix_breakthrough)
   - `project_tralo_divergence.md` (fix supersedes — divergence was the
     problem, not the answer; keep ONLY if cited as motivation in paper)
   - `project_linear_tails_breakthrough.md` (linear_sat_tail dropped)
2. **Memory entries to keep/update**:
   - `project_tralofix_breakthrough.md` — current truth
   - `feedback_convergence_is_core.md` — still core claim
   - `feedback_lambda_toggle.md`, `feedback_gpu_sharing.md` — still valid
   - `project_so2sat_pivot.md`, `project_hounie_bugfix.md` — paper context
3. **CLAUDE.md updates**:
   - New methodology name (`tralo` is the proposed; `tralo_bounded` is
     the ablation; `tralo_fioretto` removed from registry list).
   - Drop references to `linear_sat_tail`, `unbounded_quad_coef`.
   - Add the Adam-reset-at-sat note to "Training Phases" section.
4. **docs/** — update `THESIS_CONTEXT.md` and any README to reflect
   final method.

---

## Stale branches (handled this session, audit only)

Already deleted 19 stale branches (2026-05-20). Quick audit before
cleanup: `git branch -r` should show only `origin/main` +
`origin/tralo-fioretto-hybrid` + `origin/dsisco02/cleanup-pipeline`.

---

## Execution order

1. Trigger check (results review).
2. **Regime A** + **Regime B** + **Regime C** + **Regime E** in
   parallel (4 agents, one message). They touch disjoint paths.
3. After Regime A succeeds: **Regime D** (rename — must come after A
   because old code paths get deleted in A).
4. After Regime D succeeds: tag a clean commit `v1-paper-ready` so we
   can revert if needed.
5. Update `MEMORY.md` index pointing to surviving entries only.

---

## What NOT to do during cleanup

- Don't delete `model_cache/` (warmup caches active).
- Don't touch `data/` directories.
- Don't delete `results/pending_runs/paper400_tralofix/` or
  `paper400_baselines/` or `kl_ablation/` or `component_ablation/` —
  these are the paper-ready results.
- Don't delete `hybrid_v3/` raw results (cited in memory).
- Don't run any sweeps; cleanup is purely structural.
