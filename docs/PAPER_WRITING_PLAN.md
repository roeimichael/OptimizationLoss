# Paper Writing Plan — handoff for dedicated session

**Created:** 2026-05-24
**Purpose:** This file is a self-contained brief for a separate Claude Code session that owns paper-writing work while a different session monitors running experiments.

> **Session boundary:**
> - **This (paper-writing) session** owns everything under `paper/` and writes new section content. Reads results from `docs/` but does not modify experiment infrastructure.
> - **Other (experiments) session** owns `results/`, `src/`, `main.py`, `docs/PAPER_PLAN.md`, and the running sweeps. Do not touch those.
>
> When in doubt, read but don't write outside `paper/` and `docs/PAPER_WRITING_PLAN.md`.

---

## 1. What the user asked for (verbatim intent)

> *"Break the thesis paper into blocks or sections, consider each as a task on its own, spawn an agent that will start by evaluating the full results, the md files that describes what we are doing, and also you give him context, tell him what he needs to do, plus tell him to fetch relevant skills for building the paper. The agent will be responsible on one task and one task only and will build it and that's it. After each block of the paper is written you will review it and make any needed fixes. This includes graphs and tables along with mathematical expressions and bibliography from our sources. Which means you need to review all the bib we have, all the results, and also if we have experiments that haven't been run yet just have some empty space left for later fillment once we get the results of the pipe."*

Reformulated:
- Decompose the thesis paper into discrete blocks (sections + bibliography + figures).
- Spawn one agent per block. Each agent owns ONE block end-to-end.
- Before spawning, give the agent the full context (this file, results, conventions).
- After each agent finishes, you (the orchestrator) REVIEW the output and make any necessary fixes before spawning the next block.
- Leave `\todo{...}` placeholders where experiments are still running.
- Bibliography must reflect the actual citations used and our sources.
- Final deliverables: a coherent `paper/main.tex` ready for compilation, updated `paper/references.bib`, regenerated figures.

---

## 2. State of the repo (snapshot 2026-05-24)

### Datasets locked
- `tissuemnist` — cls=4 (GE, 7.1%), group_column=`synth_group`
- `dermmnist` — cls=4 (MEL, 11.1%), group_column=`loc_group` (not `sex`)
- `aider` — cls=0 (collapsed_building, 8.6%), group_column=`synth_group` — **FROZEN** (showcase in `docs/aider_results/`)
- `eurosat`, `so2sat` — **DROPPED**, do not reintroduce

### Experiments state
- **Table A (headline)** — 360/360 complete. Aggregator output in `docs/table_a_summary.md`, `table_a_summary.csv`, `table_a_raw.csv`, `table_a_per_seed.csv`, `table_a_head_to_head.csv`.
- **Table B (asymmetric tightness, derm)** — 🟢 currently running on dsisco02 GPU 3 (PID 4128054, started 2026-05-24 15:03 UTC, ETA ~12h). 144/600 done, 456 queued under `results/pending_runs/paperv2_phase2/`.
- **Tables C/D/E** — pending. Backbones (derm × {ResNet18, EfficientNetB0}), multi-class (derm × {AKIEC, BCC, BKL}), group ablation (derm × sex).
- **Component ablation, KL ablation, arch validation** — all complete; results already locked.

When you write Results sections, treat Table A as data-backed prose and Tables B–E as `\todo{...}` placeholders.

### Existing paper materials

| Path | What | State |
|---|---|---|
| `paper/main.tex` | Main LaTeX (elsarticle) | Has Abstract (partial), Introduction (written), Related Work (mostly), Problem (written), Method (written + algorithm). **TODO**: Experimental Setup, Results, Discussion, Conclusion. Abstract has a `% TODO: Add results summary` block. |
| `paper/references.bib` | BibTeX | ~30 entries. **Missing**: Fioretto LDF (key baseline), AIDER dataset, HAM10000 original, ECE original (Naeini), Brier original, possibly Stooke PID Lagrangian, Wohlberg ADMM, Basir-Senocak ALM. |
| `paper/figures/fig_convergence.png` | Convergence plot | **Stale** — generated from old tissue+eurosat; eurosat is dropped. Regenerate. |
| `paper/figures/fig_f1_tightness.png` | F1 vs tightness | **Stale** — same reason. |
| `paper/figures/fig_satisfaction.png` | Sat rate per method | **Stale** — same reason. |
| `paper/figures/proposal_fig1_penalty.png`, `proposal_fig2_convergence.png` | Proposal-era figures | Still usable as method-illustration figures. |
| `paper/CORE_DRAFT.md` | Old narrative skeleton | **Stale** (references eurosat, Tier 1/2/3 framing). Don't take prose from here without checking. |
| `paper/results_v2.tex`, `paper/results_tables.tex` | Auto-generated tables | Generator (`paper_results/build_paper_artifacts.py`) is keyed to old sweep names. **Will need rewrite or skip — prefer hand-writing tables from `docs/table_a_summary.md`.** |

### Context files every agent should read first

| Path | Why |
|---|---|
| `CLAUDE.md` | Project conventions, data paths, current Methods/datasets |
| `docs/THESIS_CONTEXT.md` | Research problem, loss derivation, history (note: parts are dated — cross-check against `docs/PAPER_PLAN.md`) |
| `docs/PAPER_PLAN.md` | **Single source of truth** for experimental plan, Tables A–E definitions, completion status |
| `docs/table_a_summary.md` | Table A results (paper-ready) |
| `docs/aider_results/README.md` | Aider regime-saturation finding (relevant for Discussion) |
| `paper/main.tex` | Current LaTeX state |
| `paper/references.bib` | Current bib |

---

## 3. The 8 blocks (each = one agent task)

Execute serially. After each agent returns, READ the actual diff (not the summary), spot-check the LaTeX compiles or at least balances braces, verify cited keys exist in references.bib, and move on.

### Block 1 — Bibliography audit + expand
**Goal:** Make `references.bib` self-consistent and complete for every citation main.tex will need.

**Steps the agent should take:**
1. Grep `paper/main.tex` for all `\cite{...}` keys.
2. Cross-check against `paper/references.bib`. List orphan keys (cited but missing).
3. Add missing baseline citations: Fioretto LDF (Fioretto, Mak, Van Hentenryck — "Predicting AC optimal power flows..." or the more specific LDF paper "Lagrangian Duality for Constrained Deep Learning"), AIDER dataset (Kyrkou et al. 2019), HAM10000 (Tschandl et al. 2018), MedMNIST already there, ECE (Naeini et al. 2015 or Guo et al. 2017 calibration), Brier (Brier 1950).
4. Also confirm: Hounie RCL, Chamon non-convex, Shifman (danits_lp), Singer Cohen.
5. Group entries by category as the existing file does.
6. Do NOT hallucinate venues / years — search the web if unsure or leave a `% VERIFY:` comment.

**Output:** an Edit to `paper/references.bib` adding the missing entries, plus a short report of what was added/changed.

### Block 2 — Experimental Setup section
**Goal:** Replace the `% TODO` block in `paper/main.tex` §Experimental Setup with concrete v2 setup prose.

**Must include:**
- Datasets: tissuemnist (GE/synth_group), dermmnist (MEL/loc_group), aider (collapsed_building/synth_group). For each: source, n_classes, train/test count, image size, constrained class with rationale, group column with class-by-group rate table.
- Backbones: MobileNetV3, ResNet18, EfficientNetB0 (cite Howard, He, Tan). Note that ResNet18 + EfficientNetB0 results pending (`\todo{}`).
- Methods: TraLO (ours), TraLO-bounded, Fioretto LDF (cite), Hounie RCL (cite Hounie 2023), Danits LP (cite Shifman 2025), Heuristic (top-K post-hoc).
- Constraint tightness: symmetric L20/L30/L50/L70/L80 for Table A; full 5×5 (L,G) ∈ {20,30,50,70,80}² for derm asymmetric (Table B, in-progress).
- Seeds: 1,2,3,4.
- Training procedure: 50 warmup CE epochs, 300 constraint epochs, lr 1e-4 / 5e-6, Adam (fused), BF16 autocast, lambda ratchet, rho ladder, post-hoc adjustment.
- Hardware: dsisco02 (RTX PRO 6000 Blackwell 96 GB). All results from this single architecture for consistency.
- Evaluation metrics: F1 (Macro/Weighted), Accuracy, ECE (cite), Brier (cite), Post-hoc Flips Required, Constraint Satisfaction Rate (in-training, before flips), Satisfaction Epoch, Constraint Train Time.

**Output:** Edit `paper/main.tex` replacing the TODO block for `\section{Experimental Setup}` with the prose. Cite cleanly.

### Block 3 — Results §Headline (Table A)
**Goal:** Write `\subsection{Headline method comparison}` with one LaTeX table for Table A and prose discussion.

**Must include:**
- LaTeX table summarizing Table A: per (dataset, tightness) rows × method columns, key metrics F1m + Flips + Sat% + ECE.
- Source data: `docs/table_a_summary.md` (also `docs/table_a_summary.csv` machine-readable, `docs/table_a_head_to_head.csv` for gaps).
- Bold the winning method per (ds, tight, metric) — highest F1m/Acc/Sat%, lowest Flips/ECE/Brier.
- Prose discussion:
  - TraLO wins or ties on Flips 15/15 cells (5 flips mean vs heuristic 74).
  - F1m: TraLO wins tissue 5/5; derm 3/5 + 2 ties; ties on aider (regime-saturation, see §Discussion).
  - Satisfaction during training: TraLO 99%, Hounie 100%, Fioretto 93%, Bounded 83%, heur/danits 7%.
  - Calibration (ECE/Brier): post-hoc methods (heur, danits) win because they don't perturb the warmup model.

**Output:** Edit `paper/main.tex` to add `\subsection{Headline method comparison}` under `\section{Results and Analysis}`. Use `booktabs` for the table.

### Block 4 — Results §B/C/D/E placeholders
**Goal:** Stub the remaining results subsections with `\todo{...}` and the planned structure so the paper compiles and the reader can see what's coming.

**Subsections to stub:**
- `\subsection{Asymmetric tightness}` — Phase 2 in progress; placeholder for 5×5 (L,G) heatmap on derm.
- `\subsection{Backbone robustness}` — Phase 3 pending; placeholder for MobileNetV3 vs ResNet18 vs EfficientNetB0 on derm.
- `\subsection{Multi-class robustness}` — Phase 4 pending; placeholder for AKIEC / BCC / BKL / MEL on derm.
- `\subsection{Group-column ablation}` — Phase 5 pending; placeholder for sex vs loc_group on derm.

Each stub gets one paragraph describing what the experiment tests and `\todo{Fill in once Phase X completes}`. Reference `docs/PAPER_PLAN.md` for the exact spec.

**Output:** Edit `paper/main.tex`.

### Block 5 — Discussion section
**Goal:** Replace `% TODO` in `\section{Discussion}` with a coherent narrative.

**Threads to weave (in this order):**
1. **When TraLO wins.** On harder tasks (tissue: warmup ~50% test acc; derm: ~85%), TraLO finds genuinely better decision boundaries during constraint optimization. Cite Table A wins on tissue + derm.
2. **The regime-saturation finding (aider).** Aider warmup hits 99.98% train acc / ~92% test acc. With the warmup model already near-optimal, the constraint cap dictates F1 on the constrained class identically across methods (read `docs/aider_results/README.md`). TraLO's training perturbs adjacent classes (collateral damage), so F1m ties or slightly loses on aider. **Frame as a regime ablation, not a failure.** This is a real paper insight.
3. **In-training satisfaction vs post-hoc.** TraLO produces a deployable model that respects constraints without test-set access at inference time — heuristic / danits LP cannot. Reference the deployment story for streaming/online settings.
4. **The soft / hard count gap.** Differentiable soft counts are an upper bound on hard counts under high confidence; post-hoc adjustment closes the gap. Empirically TraLO's gap is small (1–5 flips) while heur/danits gap is enormous (22–84) by design.
5. **Calibration tradeoff.** Methods that don't perturb the model win ECE/Brier trivially; TraLO pays ~0.01 ECE for actually solving the constrained problem.
6. **Limitations.** Synthetic groups on tissue + aider mean local constraint barely binds. Asymmetric tightness only on derm. Multi-class results pending. ResNet/EfficientNet pending.

**Output:** Edit `paper/main.tex` to fill `\section{Discussion}`.

### Block 6 — Conclusion section
**Goal:** Replace `% TODO` in `\section{Conclusion}` with a tight 2–3 paragraph wrap.

**Must include:**
- Restate contributions (joint multi-level, bounded penalty, multi-class support).
- One sentence on each of: headline win on Flips, regime-saturation observation on easy tasks (aider), deployment claim.
- Future work: inductive extension (no test-set access at training), adaptive budgets, larger benchmark families, asymmetric tightness analysis (placeholder for Phase 2 results).

**Output:** Edit `paper/main.tex`.

### Block 7 — Abstract refresh
**Goal:** Replace the `% TODO: Add results summary` block in the abstract with concrete numbers.

**Anchor numbers (from Table A):**
- TraLO mean post-hoc flips = 5 across all (ds, tight) cells, vs heuristic 74, danits 72, Fioretto 11, Bounded 11, Hounie 20.
- In-training satisfaction: TraLO 99%, Hounie 100% (but at 20-flip cost), post-hoc baselines 7%.
- F1 macro: TraLO wins or ties on 9/15 cells across 3 datasets.

**Output:** Edit `paper/main.tex` updating the abstract (~3–4 sentences added).

### Block 8 — Figure regeneration
**Goal:** Regenerate the three convergence-related figures from current Blackwell data (drop eurosat). Place in `paper/figures/`.

**Figures:**
- `fig_convergence_v2.png` — training-time excess vs epoch, 3 methods (TraLO, Fioretto, Hounie), one curve per method per seed, panel per (tissue, derm) at L30_G30. Read training logs from `results/pending_runs/paper400_tralofix/{ds}/L30_G30/seed_1/training_log.csv` and equivalents.
- `fig_f1_tightness_v2.png` — F1 macro (left axis) and Posthoc Flips (right axis, log-scale) vs tightness, 6 methods × 2 datasets (tissue, derm). Data from `docs/table_a_summary.csv`.
- `fig_satisfaction_v2.png` — bar chart, sat% per method per tightness, faceted by dataset.

Use matplotlib, Agg backend, 300 dpi PNGs. Write the generation script under `paper/scripts/fig_regen_v2.py` (parallel to existing `paper/scripts/`). Reference the new figures from `main.tex` Results sections.

**Output:** new `paper/scripts/fig_regen_v2.py`, three new PNGs, and `\includegraphics` references updated in `main.tex`.

---

## 4. How to spawn an agent for a block

Use the Agent tool with `subagent_type: "general-purpose"`. None of the bundled subagents fit paper writing exactly — `general-purpose` has all tools and matches the open-ended nature.

**Template prompt:**

```
You are responsible for ONE block of the thesis paper. Do not touch
other blocks. After you finish, return a short report.

Block: <NAME>
Goal: <ONE-SENTENCE GOAL>

Required reading (in this order):
1. docs/PAPER_WRITING_PLAN.md — full plan; find your block's spec
2. CLAUDE.md — project conventions
3. docs/PAPER_PLAN.md — experiment plan and completion status
4. paper/main.tex — current LaTeX state (read in full)
5. paper/references.bib — current bib (read in full)
6. <BLOCK-SPECIFIC FILES, e.g. docs/table_a_summary.md, docs/aider_results/README.md>

Constraints:
- Stay in LaTeX (elsarticle). Do not switch templates.
- Use `\cite{...}` with keys that exist in references.bib (or add to bib
  in your block if Block 1 — otherwise only cite existing keys).
- For pending experiments, use `\todo{Fill in once Phase X completes}`
  with a brief description. Do not invent numbers.
- Only modify the files your block owns. Specifically:
  - Block 1: paper/references.bib only.
  - Blocks 2-7: paper/main.tex only (your assigned section).
  - Block 8: paper/scripts/fig_regen_v2.py, paper/figures/*_v2.png, and
    \includegraphics lines in main.tex.
- Do not modify anything outside paper/ except as explicitly authorized.
- The other (experiments) session is actively running sweeps in
  results/pending_runs/. Read those for data but do not write there.

Output:
- Make your edits in place via the Edit/Write tools.
- Return a 5-10 line report listing: files touched, summary of changes,
  any open questions or `\todo` markers you left.
```

Fill in `<NAME>`, `<ONE-SENTENCE GOAL>`, `<BLOCK-SPECIFIC FILES>` from §3 above.

After the agent returns, **read the actual changed files** (don't trust the summary). Spot checks:
- Run `grep -c '\\cite{' paper/main.tex` against `grep -c '@\\(article\\|inproceedings\\|book\\|misc\\)' paper/references.bib` keys.
- Verify all `\cite{key}` keys resolve in references.bib (`grep -E '^\\s*@.*\\{KEY,' references.bib`).
- Look for hallucinated numbers — if a stat appears in prose that you didn't ask the agent to put there, verify it against the source CSV.
- LaTeX brace balance: `awk -F'{' '{n+=NF-1} END {print n}' paper/main.tex` vs `\F'}' '{n+=NF-1} END {print n}'`.

If you find issues, make the fix directly with Edit (don't re-spawn the same agent). Re-spawn only if a whole section needs rewriting.

---

## 5. Execution order + dependencies

```
Block 1 (bib)
   ↓
Block 2 (Experimental Setup) — uses bib keys
   ↓
Block 3 (Results §Headline) — uses bib keys + Table A data
   ↓
Block 4 (Results §B/C/D/E placeholders)
   ↓
Block 5 (Discussion) — depends on Results being present
   ↓
Block 6 (Conclusion) — depends on Discussion
   ↓
Block 7 (Abstract) — refresh with finalized numbers
   ↓
Block 8 (Figures) — last, since main.tex must reference them
```

Serial execution. Don't parallelize — each block reads the previous block's output.

---

## 6. Hard rules for this session

1. **Do NOT touch `results/`, `src/`, `main.py`, `model_cache/`, or any sweep launcher.** Experiments are running.
2. **Do NOT modify `docs/PAPER_PLAN.md`.** That file is owned by the experiments session.
3. You MAY modify `docs/PAPER_WRITING_PLAN.md` (this file) to track block completion.
4. You MAY add subdirectories under `paper/` (e.g. `paper/scripts/`).
5. Never run experiments or invoke `python main.py`.
6. Never edit `references.bib` keys after Block 1 — only add new entries if cited in a later block and check Block 1 didn't already include them.

---

## 7. Block completion tracker

Mark `[x]` when reviewed and accepted.

- [ ] Block 1 — Bibliography audit + expand
- [ ] Block 2 — Experimental Setup section
- [ ] Block 3 — Results §Headline (Table A)
- [ ] Block 4 — Results §B/C/D/E placeholders
- [ ] Block 5 — Discussion section
- [ ] Block 6 — Conclusion section
- [ ] Block 7 — Abstract refresh
- [ ] Block 8 — Figure regeneration

After all 8 are complete, compile main.tex and circulate the PDF to the user / thesis advisor.

---

## 8. Final notes

- **Sources of truth for results:** anything in `docs/table_a_*.csv`, `docs/aider_results/`, and the per-cell `evaluation_metrics.csv` files under `results/pending_runs/`. Do not invent numbers; if a number doesn't appear in any of those, mark it `\todo{}`.
- **When citing baselines, use the exact paper:** Fioretto LDF refers to "Lagrangian Duality for Constrained Deep Learning" (Fioretto, Mak, Van Hentenryck, 2020). Hounie RCL refers to "Resilient Constrained Learning" (Hounie, Ribeiro, Chamon, NeurIPS 2023; already in bib). Danits LP refers to Shifman et al. 2025 (already in bib).
- **The aider story:** "F1m losses on aider are a structural property of the easy-task regime, not a TraLO failure. TraLO still dominates on flips." Frame consistently across Results, Discussion, and Conclusion.
- The other session monitors Phase 2 and will append results to `docs/` as Phases 2–5 complete. Re-run the appropriate aggregator and update the placeholders.
