# Native-Resolution Campaign — pre-registered design

**Created:** 2026-07-25 · **Owner:** roei · **Status:** DESIGN (awaiting approval; nothing launched)
**Question:** Can TraLO win the constrained-class F1 (cc-F1) comparison at *native* image
resolution, or is native-res conclusively a non-TraLO regime? B2 showed the 28px OctMNIST
tight-cap win collapses at native OCT. This campaign tests whether that collapse is universal
across native-224 datasets and HP regimes — with a **pre-registered bar** so that *either*
outcome (a genuine win, or a conclusive + mechanistic null) is publishable and honest.

This file is the single source of truth for the campaign. It is committed **before** results
exist. Do not edit the pre-registered bar (§2) or dataset set (§3) after seeing any result.

---

## 1. Hypothesis and mechanism (the reason this is designed the way it is)

Per `docs/REJECTED.md`, TraLO's edge appears **only when the warmup leaves the constrained
class unsaturated** — train-acc roughly in [0.70, 0.82], equivalently the clip baseline's
achievable top-K precision on the constrained class is **< 1.0** (there is "headroom" for TraLO
to redistribute predictions). Native high resolution sharpens features → the constrained class
becomes easier → warmup saturates it → clip top-K precision → ~1.0 → **no headroom → TraLO ties**.

**Falsifiable prediction:** a native-res TraLO win is possible *iff* we can find a
(dataset × backbone × warmup) cell where headroom survives at native res. Therefore:
- To find a win if one exists, we must **search headroom-preserving regimes** (hard constrained
  classes; short warmups that stop before saturation).
- To conclude "bad" convincingly, a null must hold **even where headroom is present** — otherwise
  the null is just "we didn't try hard enough." The headroom probe (§4, Phase 0) guarantees we
  know whether headroom existed.

---

## 2. Pre-registered decision rule (fixed — do not change after seeing results)

**Primary metric:** cc-F1 = binary F1 of the constrained class, computed from
`final_predictions.csv` (post-hoc-feasible; all methods clipped to the cap).
**Comparison:** TraLO vs the **constrained-optimization family** {Fioretto-LDF, Hounie-RCL,
LP-clip (danits_lp)} — the family where the paper's win lives. (Focal / imbalance-aware training
is B1's separate question and is **out of scope** here.)
**Test:** paired, matched-seed (per-seed gap TraLO − baseline; never pooled std).

**WIN (a cell counts as a win) if ALL hold:**
1. mean paired cc-F1 gap ≥ τ, for τ in the sweep {0.005, 0.010, 0.020} (report the τ-regime);
2. ≥ half the seeds have gap > 0;
3. bootstrap (10k) 95% CI on the mean gap excludes 0;
4. **replicates** on ≥ 2 backbones at the same (dataset, cap).

**CONCLUSIVELY BAD (the null we can claim) if:**
- No cell clears the WIN bar at τ = 0.010 on any dataset/backbone/cap/warmup, **AND**
- The Phase-0 headroom map confirms we tested cells that *did* carry headroom (clip top-K
  precision < 0.98) — i.e. TraLO was given a fair chance and did not convert it.
- Report with BH-FDR across the whole family of tests (reuse `b7_bh_fdr_v2`).

**No post-hoc selection:** the dataset set (§3), HP grid (§5), and this bar are fixed now.
A dataset that turns out degenerate (e.g. too-small train) is *reported* as degenerate, not
silently dropped. All runs on one hardware (dsisco02 BF16) to avoid the FP16/BF16 parity
confound (max Δ≈0.012 measured in Track B).

---

## 3. Datasets (native-224, MedMNIST v2 `size=224`, same `.npy` pipeline as octnative)

Selection spans the difficulty spectrum; hard constrained classes are the best shot at a win,
easy ones anchor the "bad" boundary. Constrained class per dataset is locked in Phase -1 as
**the rarest class whose test count gives K ≥ 30 at the loosest cap (L40)** (so cc-F1 is stable);
prep reports the full distribution.

| Dataset | Classes | Train / Test | Role | Constrained class (provisional) |
|---|---|---|---|---|
| **DermaMNIST-224** | 7 | 7,007 / 2,005 | **best shot** — MEL genuinely hard at full res | MEL (cls 4, ~11% test) |
| **RetinaMNIST-224** | 5 | 1,080 / 400 | best shot — ordinal DR grades, subtle | rare severe grade (lock in prep; ⚠ small — see §8) |
| **TissueMNIST-224** | 8 | 165,466 / 47,280 | difficulty anchor — native twin of in-scope tissue | GE (cls 4, ~7.1% test) |
| **BloodMNIST-224** | 8 | 11,959 / 3,421 | moderate control | rarest cell type with K≥30 (lock in prep) |
| **OrganAMNIST-224** | 11 | 34,561 / 17,778 | moderate control (CT) | rare organ with K≥30 (lock in prep) |

**Groups:** MedMNIST has no natural group axis → synthetic binary `synth_group` (as tissue/aider)
for the local cap; the **headline is the global constrained-class cap** (that's the cc-F1 story).
DermaMNIST additionally has `loc_group` (anatomical site) available if we want a real local axis.

**Off-limits (per REJECTED.md), not in this set:** PathMNIST, ISIC2019, EuroSAT, So2Sat,
CIFAR-100; all transformer/ConvNeXt backbones.

---

## 4. Phased execution (compute goes where a win is even possible)

**Phase -1 — Prep & lock (no GPU sweep):**
- Stage each dataset at 224 as `train/test_images.npy` + labels + `test_meta.csv(synth_group)`,
  via a `prep_medmnist224.py` script (downloads `size=224`, verifies shapes, reports per-class
  test counts). Register the 5 modes in `IMAGERY_DATASETS`.
- Lock each constrained class by the K≥30 rule; write it into this file (§3) and the generator.
- Smoke: 1 warmup epoch per (dataset, MobileNetV3) to confirm the loader + shapes + normalization.

**Phase 0 — Headroom probe (cheap, warmup-only):**
- For each dataset × {MobileNetV3, RegNetY400MF} × warmup_epochs ∈ {5, 15, 30, 50}:
  run warmup only, log per-epoch **train-acc**, **constrained-class recall/precision**, and the
  **clip top-K precision** (feasible top-K on the constrained class). No constraint phase.
- Output: `headroom_map.csv` + a plot. Classify each (ds, bb, warmup) as **headroom-bearing**
  (clip top-K precision < 0.98) or **saturated** (≥ 0.98). This *predicts* where Phase 1 can win.
- Cost: 5 ds × 2 bb × 4 warmups = 40 warmup-only runs (warmups are cached & reused in Phase 1).

**Phase 1 — Full method comparison (headroom-guided, exhaustive):**
- For each dataset, take its **headroom-optimal warmup** (from Phase 0) **and** a saturated
  control warmup (=50), and run:
  TraLO vs {Fioretto-LDF, Hounie-RCL, LP-clip} × caps {L20, L30, L40} × seeds {1,2,3,4}
  × backbones {MobileNetV3, RegNetY400MF}.
- HP tuning axis (what "play with HPs" means here, ranked by expected effect):
  1. **warmup length** — the headroom lever (primary; set from Phase 0).
  2. **cap tightness** {L20/L30/L40} — how hard the count binds.
  3. **alpha_kl** {0, 0.1} on the derm-like datasets only — the one non-inert regularizer
     (memory: KL "marginally helps on derm-like"); secondary.
  4. rho/λ schedule — **held at the paper recipe**; the Adam scale-invariance result
     (rank-one gradient, memory) predicts these are absorbed, so we do not burn compute
     sweeping them (documented expectation, not an untested assumption).
- Sequenced **best-shots first**: derm + retina, so a null on the hard datasets is known early,
  then tissue/blood/organa complete the difficulty map.

**Phase 2 — Adjudicate:**
- Any WIN cell → replicate on a 3rd backbone (MobileNetV2), recompute bootstrap CI + τ-sweep.
- Compute the full stats: paired gaps, τ-sweep (`b5`), bootstrap CIs (`b6`), BH-FDR (`b7`) —
  reuse the Track B v2 evaluation scripts.
- Write the verdict against §2, with the headroom map as the mechanistic explanation.

**Cell budget (order-of-magnitude):** Phase 0 ≈ 40 warmup-only; Phase 1 ≈ 5 ds × 2 warmup
× 2 bb × 3 caps × 4 methods × 4 seeds = **960** (+ KL variant on 2 derm-like ds); Phase 2 ≈
64. At native-224 throughput on Blackwell (~5–12 min/run incl. cached warmup) ≈ several GPU-days;
phased so signal arrives after the derm+retina block, not at the end.

---

## 5. Backbones, methods, seeds (locked)
- **Backbones:** MobileNetV3 (headline) + RegNetY400MF (corroborator) for Phases 0–1;
  MobileNetV2 for Phase-2 replication. **No transformers** (REJECTED).
- **Methods:** tralo, fioretto_ldf, hounie_rcl, danits_lp. (heuristic optional as sanity.)
- **Seeds:** 1–4, matched across methods (paired test).

---

## 6. Analysis & deliverables
- `results/native_res_campaign/` experiment root (idempotent `run_lane`).
- `results/native_res_deliverables/`: `headroom_map.csv` (+plot), `native_ccf1_perseed.csv`,
  `native_paired_summary.csv`, τ-sweep / CI / FDR tables, and `VERDICT.md` (win or conclusive-bad,
  with the headroom mechanism).
- Metrics: cc-F1 (primary), macro-F1 (secondary), constrained-class recall/precision,
  clip top-K precision (headroom covariate), flips.

## 7. Infra & discipline
- **One hardware:** dsisco02 (BF16) for ALL runs → no parity confound. Free, un-shared GPUs only
  (check nvidia-smi; never share a card — driver-crash risk). Durable **tmux** launch (not
  setsid-over-SSH). `run_lane` skips completed → resumable.
- Fresh warmup caches (new `dataset_mode` → new `base_model_id`); native-224 warmups are large,
  cache & reuse across caps/methods/seeds.
- Status check-ins via cron every ~2–3h while sweeping; stop when Phase verdicts land.

## 8. Risks & how each is handled
- **RetinaMNIST too small (1,080 train / 400 test):** may saturate instantly (like rejected ViT
  on small derm) or give unstable K. Phase 0 headroom probe reveals it; if degenerate we *report*
  it as a difficulty datapoint, not drop it. If K<30 at L40 for every class, retina is global-cap
  only / excluded-with-note.
- **Native-224 is slow:** phased + warmup-cached + best-shots-first so we don't run 960 blind.
- **Manufacturing a win (the real hazard):** prevented by the pre-registered bar (§2), fixed
  dataset set, τ-sweep + CI + 2-backbone replication, and reporting nulls as nulls.
- **A win that's a resolution/K artifact:** the headroom covariate + matched-precision recall
  check (as in Track B) distinguishes a real learning win from a top-K precision artifact.

## 9. Success definition for THIS campaign
Either outcome is a success *if rigorously established*:
- **Win:** ≥1 dataset shows a native-res cc-F1 win meeting §2 → a new positive result extending
  the paper beyond 28px.
- **Conclusively bad:** null holds across all 5 datasets despite mapped headroom → a clean,
  mechanistic boundary result ("TraLO's cc-F1 edge is a low-resolution / low-separability
  phenomenon"), which *strengthens* the paper's honesty and scopes the contribution precisely.
