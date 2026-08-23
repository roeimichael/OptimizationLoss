> 🛑 **ARCHIVED 2026-08-23 -- HISTORY, NOT INSTRUCTIONS.**
> `docs/FRAMEWORK.md` is the ONLY operational document. Where this file
> disagrees with it, FRAMEWORK wins and this file is wrong.
> The warm-up-50 chain is a DEAD regime (FRAMEWORK: warm-up 1 / constraint 29 for trained arms, warm-up 30 / constraint 0 for post-hoc).
> Do not run anything on the strength of this page.

# Warmup-Headroom Ablation

**One-line:** does TraLO's *quality* advantage over the baselines grow as warmup headroom shrinks?
We sweep `warmup_epochs` to build the ladder **{1, 10, 50}** (queued now) under the **frozen `paper_final`
recipe** and measure the constrained-class F1 gap vs the trained baselines. Extendable to the 4-point
**{1, 5, 10, 50}** curve by adding w5 (one env var, +240 cells).

**Status:** w1 (probe) running; **w10 (this phase, 240 cells) auto-spawns** after the probe drains; w50
already complete (it *is* `paper_final`). Created 2026-06-16. Base commit `83f4fe51` (branch `paper-final-frozen`).

**Per-warmup cost (verified on server):** one warmup on this focused grid = **240 cells** (MNV3 96 + RegNet 96
+ ViT 48). So w10 alone = 240 (~1.6 days @ ~150 cells/day); the fuller {5,10} = 480.

---

## 1. Hypothesis & why this study exists

`paper_final` (the frozen 1944-cell grid) uses `warmup_epochs=50`. At that warmup the backbone fully
saturates on the constrained class, so there is **no headroom** left for the constraint phase to exploit —
and on tissue/derm, TraLO's edge over the trained baselines (Fioretto, Hounie) collapses to a tie.

A **warmup=1 probe** (frozen recipe, only `warmup_epochs: 50→1`) showed, on matched cells (preliminary,
112/240 done, 18 fully-comparable groups):

| metric | TraLO − best-trained gap @ w50 | @ w1 | Δ |
|---|---|---|---|
| **cc-F1** (constrained-class) | −0.0006 (tie) | **+0.0053** (win) | **+0.0059** |
| cc-F1, tissue/MobileNetV3 | −0.0045 | **+0.0135** | **+0.018** |
| macro-F1 | −0.0006 | −0.0008 | ~0 (flat) |

Mechanism: at w1 the **post-hoc clippers collapse** (macro-F1: heuristic −0.26, danits_lp −0.27 vs w50)
because there is no trained model under the clip; TraLO degrades the **least** of all methods on cc-F1
(−0.035). The signal lives in **cc-F1**, not macro-F1.

This phase fills the **intermediate warmup {10}** (the partial-headroom "knot") so the three points
{1,10,50} form a clean monotonicity curve instead of two lonely endpoints. Adding **w5** (one env var)
extends it to the 4-point {1,5,10,50} curve, thickening the low-warmup end where the effect is strongest.

## 2. Claim metric

**cc-F1 = F1 of the constrained class, computed on the FINAL constraint-SATISFYING predictions.**
Every method's reported metrics are taken AFTER `targeted_correction` forces exact count satisfaction
(`src/pipeline/eval.py`), so `flips`/`saturation` are **diagnostics, not quality**. The headline rests on
cc-F1; macro-F1 is reported as a secondary "quality preserved" line; the w1 clip-collapse is a footnote.

## 3. Recipe (frozen — identical to `paper_final` except `warmup_epochs`)

`SHARED_HP`: lr 1e-4, lr_constraint 5e-6, dropout 0.3, batch_size 64, **constraint_epochs 300**,
pretrained True, class_weighted_ce False, constraint_chunk_size 256. `PER_METHOD` and `DS_CFG` are
byte-identical to `scripts/gen_paper_final.py` (e.g. **fioretto_step_size 0.005**, the TraLO_fix recipe
with `undershoot_hinge`/`fior_beta 0.50`/`reset_optimizer_at_sat`). Generator: `scripts/gen_warmup_ablation.py`.

## 4. Directory map  (`results/pending_runs/`)

| warmup | dir(s) | backbones | seeds | sweep tag | cells | source |
|---|---|---|---|---|---|---|
| **1** | `warmup1_probe`, `warmup1_probe_s34`, `warmup1_probe_vit` | MNV3, RegNet, ViT | 1-4 (ViT 1-2) | `warmup1_probe` | 240 | the probe |
| **10** | `warmup_ablation/lane_mnv3` | MobileNetV3 | 1-4 | `warmup_ablation` | 96 | this phase |
| **10** | `warmup_ablation/lane_regnet` | RegNetY400MF | 1-4 | `warmup_ablation` | 96 | this phase |
| **10** | `warmup_ablation/lane_vit` | ViTB16 | 1-2 | `warmup_ablation` | 48 | this phase |
| **50** | `paper_final/lane{0,1,2}/...` | MNV3, RegNet, ViT | 1-4 | `paper_final` | (subset of 1944) | the frozen grid |

*(Adding w5 doubles each lane → 192 / 192 / 96 = 480 total.)*

Per-cell path: `<dir>/<model>/<ds>/<tag>/w<W>/<method>/seed_<s>/{config.json,evaluation_metrics.csv,...}`
(the `paper_final` cells have no `w<W>` segment — they are the implicit w50). New-cell factorial (w10):
`{tissue, derm} × {10} × {L30_G30, L50_G50} × 6 methods × seeds` = **240 cells** (CNN 192 + ViT 48);
the full {5,10} version is **480** (CNN 384 + ViT 96).

**Lane split is by backbone** so each concurrently-running grabber owns a non-overlapping pending list
(two `main.py` on one dir would race — the documented failure mode). Mirrors `paper_final`'s lane0/1/2.

## 5. Pairing & statistics

Every new cell pairs **cell-for-cell** (same ds, tag, method, seed) against both endpoints. Headline
paired-Wilcoxon uses **CNN cells only** (MNV3+RegNet): n = 2 ds × 2 caps × 4 seeds = **16 matched pairs**
per (warmup, baseline). **ViT (2 seeds) is qualitative corroboration only — excluded from the paired stat.**

## 6. Isolation from the frozen paper numbers (the "never lose track again" firewall)

1. `load_canonical()` (`scripts/_paper_view.py`) filters `warmup_epochs == 50` → **every** w1/w5/w10 row is
   dropped from the canonical paper view automatically.
2. Belt-and-suspenders: `warmup1_probe` and `warmup_ablation` are in `SWEEP_BLOCKLIST`.
3. All new cells live under **one documented root** (`warmup_ablation/`) with this manifest.
4. `paper_final` @ `83f4fe51` is **never touched**.

## 7. EXCLUDED — old short-warmup sweeps (do NOT pool these)

`corpus_full.csv` already contains short-warmup cells from **earlier, pre-freeze recipes**. They share the
cell key but use **different hyperparameters** and are **not comparable**. Do not stitch them into this curve:

| sweep | what it is | why excluded |
|---|---|---|
| `phase_transition` | derm/MNV3, warmups {1,3,10,25} | `constraint_epochs=100`, `fioretto_step_size=0.01` (≠ frozen 300 / 0.005) |
| `tissue_lowwarm_validation` | tissue, warmups 1-5, 4-method subset | pre-freeze recipe; partial methods |
| `g5_short_warmup`, `tablef_shortwarm`, `pushpull_derm_w1`, `blackwell_validation` | misc short-warmup | pre-freeze recipes |

Only `{warmup1_probe (w1), warmup_ablation (w5,w10), paper_final (w50)}` are recipe-comparable.

## 8. Reproduce

```bash
# (the chain does this automatically after the probe drains; manual form:)
cd ~/OptimizationLoss
for lane in "lane_mnv3 MobileNetV3" "lane_regnet RegNetY400MF" "lane_vit ViTB16"; do
  set -- $lane
  ABL_ROOT=results/pending_runs/warmup_ablation/$1 ABL_BACKBONES=$2 ABL_WARMUPS=10 \
    ABL_SWEEP=warmup_ablation python scripts/gen_warmup_ablation.py   # ABL_WARMUPS=5,10 for the 4-point curve
done
# one self-healing grabber per lane (staggered so they grab distinct free GPUs; never shares a GPU):
for L in lane_mnv3 lane_regnet lane_vit; do
  setsid bash scripts/gpu_grab_dir.sh results/pending_runs/warmup_ablation/$L < /dev/null & sleep 120
done
```

**Active setup (2026-06-16), two pieces:**
1. **`lane_vit` runs PINNED on GPU 3** via `scripts/pin_gpu.sh <dir> 3` (launched detached). The ViT probe
   lane finished early and freed GPU 3; a *pinned* job (fixed GPU, never seeks) safely fills it without
   racing the still-running probe grabbers. ViT is the slow lane, so starting it earliest helps wall-clock.
2. **`scripts/chain_warmup_ablation.sh` with `ABL_LANES="lane_mnv3 lane_regnet"`** (detached, `ABL_WARMUPS=10`)
   polls until the probe drains, then launches those two grabbers staggered 120s on the freed CNN GPUs.
   Race-free: it launches only after all probe grabbers exit (GPUs free), stagger > model-load time.

**Lesson:** an earlier `scripts/dispatch_ablation_safe.sh` tried to run ALL lanes *concurrently* with the
probe via self-seeking grabbers. It raced — probe grabbers free their GPU mid-lane and re-seek, so multiple
grabbers sought the same freed GPU → GPU-sharing risk. The fix that keeps GPUs full AND race-free: a *pinned*
(non-seeking) job on the one already-free GPU, plus the chain (post-probe, staggered) for the rest. The
dispatcher is kept in the repo but is NOT used; never run self-seeking grabbers alongside the probe grabbers.

## 9. Aggregation outputs

- **Extract (server):** `python scripts/extract_warmup_ablation.py > /tmp/wabl.csv` — walks the probe +
  ablation dirs, emits tidy rows `[dataset,model,constraint_tag,method,seed,warmup,cc_f1,f1_macro,cc_rec,
  acc,flips,sat,sweep]` for warmups {1,5,10}. Pull local.
- **Merge + curve (local):** combine `/tmp/wabl.csv` with `paper_final` w50 (`load_canonical(warmup=50)`)
  → `paper/aaai_tables/warmup_ablation_tidy.csv` (all four warmups) →
  `paper/aaai_tables/warmup_ablation_summary.csv` (mean cc_f1/f1_macro per warmup×method×ds×backbone).
- **Headline:** cc-F1 gap = `cc_f1[tralo] − max(cc_f1 over {fioretto_ldf, hounie_rcl})`, paired-mean over
  the 16 CNN cells, plotted vs warmup {1,5,10,50}, one line per dataset → the gap rises as warmup→1,
  ~0 at w50. Companion table: paired-Wilcoxon TraLO vs each trained baseline per warmup.

## 10. Scope knobs

Queued default = **w10, 240 cells** (the 3-point ladder {1,10,50}). To change:

1. **Fuller curve** → `ABL_WARMUPS=5,10` → 480 cells (~3 days); the 4-point {1,5,10,50} curve.
2. **Drop ViT** → skip `lane_vit` → CNN-only (no stat-power loss; ViT is non-headline corroboration).
3. Drop seeds 3,4 → `SEEDS_BY_MODEL` MNV3/RegNet → {1,2} (weakens paired n 16→8; not recommended).

Mid-run trim: kill the lane's grabber PID (from `logs/grab_lane_*.log`); `main.py` skips already-done cells
so nothing is lost. The chain reads `ABL_WARMUPS` at launch (default `5,10`); it was launched with
`ABL_WARMUPS=10`.
