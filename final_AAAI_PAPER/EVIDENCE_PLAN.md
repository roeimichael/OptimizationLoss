# TraLO Evidence Plan — Regimes, Tables, Graphs, Robustness

**Status:** synthesized 2026-06-17 from Lens A/B/C + adversarial audit. All numbers below
re-verified against `paper/aaai_tables/_regime_master.csv` and `load_canonical()` this session.
**Mandate:** every regime claim is defined by a *pre-stated* rule and reports *all* cells in that
regime (n, winrate, mean gap), including ties and losses. No post-hoc cell selection.

---

## 1. THE REGIME LAW (one honest sentence)

> **TraLO universally out-qualifies the post-hoc clipping baselines on deployment quality
> (native satisfaction $\approx$ 1.0, 2–6 label flips vs 27–133, macro-F1 preserved on all 3
> datasets), and matches the constraint-trained baselines on constrained-class F1 everywhere
> *except* two narrow hard-constraint regimes where it wins — OctMNIST tight *binding* symmetric
> caps ($L{=}G\in\{30,40\}$, all backbones, +0.020) and DermMNIST asymmetric caps with the
> group cap tighter than the global cap ($G{<}L$, MobileNetV3, +0.008) — because its bounded,
> frozen-at-satisfaction multiplier plus undershoot hinge holds constrained-class recall where
> the trained baselines' unbounded dual ascent over-suppresses the constrained class.**

**Mechanism (supported by logs + the no-hinge ablation already in corpus):** the *undershoot
hinge* (`fior_beta`) is the load-bearing piece — on the win cells, removing it (`tralo_bounded`)
drops cc-F1 back to ~Fioretto level (Oct L30: 0.42→0.39; Derm asym 9/10 tags lower). The bounded
saturating penalty keeps CE finite (Fioretto's `ce_loss`→NaN at constraint-epoch 2 in 99% of
local logs) and the frozen multiplier stops dual escalation. **Honesty flag:** the *magnitude* of
baseline dual instability does **not** predict the win (Lens B: r=−0.22, multiplier highest where
TraLO loses) — frame the mechanism qualitatively ("baselines destabilize the constrained class"),
**never** as "more escalation → bigger win."

---

## 2. TABLES TO SHOW

| # | Table | Purpose | Source data | Buildable now? |
|---|-------|---------|-------------|----------------|
| **T1** | **Master regime table** (anti-cherry-pick centerpiece) | Partition the *entire* canonical grid into 5 pre-stated geometry regimes × 3 datasets; report n, cc-F1 gap vs best trained, win%, and the deployment trio for **every** cell incl. ties/losses. Bold marks the only 2 real wins. | `paper/aaai_tables/_regime_master.csv` → `paper/aaai_tables/regime_master.tex` (prototype built, 36/36 braces) | **YES — built** |
| **T2** | **Deployment table** (the universal pillar) | TraLO vs best post-hoc clipper per dataset: native sat, flips, macro-F1 gap. Show sat 1.00 vs 0.14, flips 2–6 vs 27–133, macro +0.012→+0.048 positive in all 13 regime×dataset rows. | `_regime_master.csv` (`dep_*` cols) aggregated per dataset | **YES** |
| **T3** | **Hard-regime cc-F1 table** | The two real wins, fully expanded: (a) Oct tight-sym L30/L40 × {MNV2,MNV3,RegNet,ViT}, per-backbone gap; (b) Derm asym G<L, all 10 tags MNV3 incl. the L80_G20 loss tag. Report macro side-effect (Oct +0.007, Derm −0.003). | `load_canonical()` filtered to R2-oct + R3-derm; `_regime_cells.csv` | **YES** |
| **T4** | **Component-ablation table** | Show the *undershoot hinge* is load-bearing. **Leg 1 (FREE):** tralo (hinge) vs tralo_bounded (no-hinge) on Oct L30 (4 backbones) + Derm asym G<L (10 tags) — already in corpus. **Legs 2–5 (NEED RUNS):** −freeze, −optimizer-reset, ±KL, −rho-ramp. | Leg 1: `load_canonical()` method∈{tralo,tralo_bounded}. Legs 2–5: new runs (§4) | **Leg 1 YES; legs 2–5 NEED RUNS** |
| T5 *(opt.)* | **Baseline-fairness table** | Fioretto/Hounie step-size sweep on the hard cells → prove the dual instability is structural, not mis-tuned. | New runs (§4) | NEEDS RUNS |

**Caveat baked into T1/T3:** R3-tissue (asym G<L on tissue) is **net-negative** (−0.0039, 2/10
tags) — the previously-asserted "tissue L50 loose-global pocket" does **not** survive full-regime
accounting and is **dropped** as a cherry-pick. T1 shows it as a loss row, in the open.

---

## 3. GRAPHS TO SHOW

| # | Graph | Axes | What it proves | Buildable now? |
|---|-------|------|----------------|----------------|
| **G1** | **Deployment universality** | x = dataset×regime; y = native satisfaction (TraLO bar=1.0 vs post-hoc pre-clip bar≈0.0); twin panel = flips (log scale). | The universal pillar — TraLO satisfies natively everywhere; clippers satisfy only after 27–133 destructive flips. | **YES** |
| **G2** | **OctMNIST dose-response (the regime curve)** | x = symmetric cap level L (10→90); y = cc-F1 gap vs best trained; **one line per backbone** (MNV2/MNV3/RegNet/ViT). | The inverted-U: gap peaks at L30/L40 (binding-but-discriminable), ~0 at L≤20 and L≥50. Honest: it is **not** monotonic "tighter = bigger win." All 4 backbones positive at L30. | **YES** (L40 locally MNV3-only for vs-trained — annotate) |
| **G3** | **CE / multiplier mechanism figure** | x = constraint epoch; y-left = `ce_loss` (Fioretto → NaN at epoch 2 vs TraLO finite); y-right = max dual multiplier (Fioretto escalates vs TraLO frozen ≤0.09). One hard cell (e.g. Derm L50_G20). | The mechanism: bounded-frozen multiplier keeps CE finite → preserves constrained-class recall. | **YES for derm/tissue cells** (Oct epoch logs server-only) |
| **G4** | **Convergence** (core thesis claim — never drop) | x = epoch; y = global/group hard-count vs cap line; show TraLO converging and freezing at satisfaction. | Constraint satisfaction is achieved and stable — the central contribution. | **YES** (from training logs) |

**G2 honesty annotation:** mark L40 points that are MNV3-only locally (fioretto/hounie L40 rows
for MNV2/RegNet/ViT are server-only); the *L30* cross-backbone claim is fully local and clean.

---

## 4. ABLATION / ROBUSTNESS PROTOCOL

Frozen recipe for ALL new runs: warmup=50, constraint=300 epochs, fioretto_step=0.005.
Config flags verified present in `src/methodologies/tralo/train.py` (lines noted).

### 4a. Component ablation (leave-one-out on the hard cells)

| Leg | Flag (train.py line) | Cells | Runs | Proves |
|-----|----------------------|-------|------|--------|
| Hinge OFF | `hybrid_mode='bounded_only'` = **tralo_bounded** (L40,60) | Oct L30×4bb + Derm asymG<L×10 | **0 (FREE)** | hinge is the load-bearing piece (already confirmed) |
| Freeze OFF | `disable_freeze_on_satisfy=True` (L80) | Oct L30,L40×{MNV3,ViT} + Derm L50_G20,L80_G70×MNV3, s1–4 | 24 | frozen multiplier keeps CE finite |
| Reset OFF | `reset_optimizer_at_sat=False` (L73) | same 24 cells | (shares batch → +24 configs) | optimizer reset breaks post-sat descent momentum |
| KL add-back | `alpha_kl∈{0.1,0.3,1.0}` (L82, default 0=off) | Oct L30_G30 + Derm L50_G20 ×MNV3, s1–2 | 12 | justifies alpha_kl=0 in deployed recipe |
| Rho flat | `rho_target=initial_rho` (L124–125) | Oct L30_G30×{MNV3,ViT} + Derm L50_G20×MNV3, s1–4 | 12 | rho ramp is secondary (likely non-essential) |
| **Subtotal** | | | **~72 new** | |

### 4b. Factor generality (the #1 gap: asymmetric wins are MNV3-only)

| Test | Cells | Runs | Proves |
|------|-------|------|--------|
| Asym cross-backbone (minimal) | Derm 4 winning tags (L50_G20,L50_G30,L70_G50,L80_G70) × {RegNet,ViT} × 6 methods × 4 seeds | **192** | derm asym win is not a MNV3 artifact (kills strongest reviewer objection) |
| Asym cross-backbone (full, opt.) | Derm 10 tags + Tissue ~14 tags × {RegNet,ViT,MNV2} × 6 × 4 | ~1150 | full asymmetry robustness |
| Oct asymmetric grid (opt., high value) | Oct {L40_G20,L40_G30,L50_G30} × {MNV3,RegNet,ViT} × 6 × 4 | 216 | does "group-tighter helps" transfer to the dataset with the strongest sym win |
| Seed bump 5–8 on headline cells | Oct L30/L40×4bb + Derm 4 asym tags | 88 | tightens paired CI on the n=4 sharp cells |

### 4c. Baseline fairness

| Test | Cells | Runs | Proves |
|------|-------|------|--------|
| Fioretto + Hounie step-size sweep | Oct L30,L40×MNV3 + Derm L50_G20,L50_G30×MNV3 × step∈{0.001,0.002,0.005,0.01} × s1–4, both baselines | 64 | the CE→NaN dual escalation is structural to unbounded dual ascent, not our step choice (adversarial — report if a smaller step rescues the baseline) |

### Run-count totals
- **Recommended minimal robustness block:** component LOO (~72) + minimal asym cross-backbone
  (192) + baseline fairness (64) + seed bump (88) = **~416 new runs.**
- **Full protocol** (adds Oct-asym + full tissue/derm asym cross-backbone): **~1900 new runs.**

---

## 5. BUILDABLE-NOW vs NEEDS-EXPERIMENTS

### Buildable now (ZERO new runs) — this is most of the robustness story
- [x] **T1 master regime table** — built: `paper/aaai_tables/regime_master.tex` (braces 36/36).
- [x] **T2 deployment table** — `_regime_master.csv` dep_* columns.
- [x] **T3 hard-regime cc-F1 table** — `load_canonical()` on R2-oct + R3-derm.
- [x] **T4 leg 1 (no-hinge ablation)** — tralo vs tralo_bounded, Oct L30 (4bb) + Derm asym (10 tags).
- [x] **G1 deployment universality** — sat/flips from `_regime_master.csv`.
- [x] **G2 OctMNIST dose-response** — per-backbone cc-F1 gap by L (L30 fully local; L40 MNV3-only).
- [x] **G3 mechanism figure** — derm/tissue local training logs (`_lensB_instability.csv` parsed).
- [x] **G4 convergence** — from any TraLO training_log.csv.

### Needs experiments
- [ ] **T4 legs 2–5** (−freeze/−reset/±KL/−rho) — only `tralo_bounded` exists cleanly today.
- [ ] **T5 baseline-fairness** step-size sweep.
- [ ] **Asym cross-backbone** (RegNet/ViT/MNV2) — R3/R4 are MNV3-only; biggest generality gap.
- [ ] **OctMNIST asymmetric grid** — no asym runs exist for octmnist.
- [ ] **G3 for OctMNIST** — Oct + paper_final fioretto/hounie epoch logs are **server-only**;
      pull them and re-run `scripts/_lensB_instability.py` to *locally verify* the Oct mechanism
      (currently asserted, not locally verified — the audit's top coverage gap).

---

## 6. THE HONEST "WHERE IT TIES" FRAMING

State these up front — they are what make the win claims credible:

1. **The bulk of the grid is a TIE vs trained baselines, not a win.** tissue+derm symmetric =
   122 cells, mean cc-F1 gap −0.0014, cell-winrate ~14%, per-seed sign-winrate ~30%. TraLO
   **matches** the constraint-trained baselines on the majority of cells; it does not beat them.
   Any claim of a *broad* cc-F1 advantage vs trained baselines is unsupported.
2. **Only two real cc-F1 wins vs trained:** Oct tight-binding symmetric (+0.0202, robust across
   4 backbones, the strong one) and Derm asym G<L (+0.0078, **MNV3-only**, and it **costs** macro-F1
   −0.0029 — recall-for-precision). The Oct win is an **inverted-U** at L30/L40, **not** monotonic
   in tightness (≈0 at L≤20 and L≥50).
3. **Tissue asymmetric is dropped as a cherry-pick** — net-negative regime (−0.0039, 2/10 tags
   win, clean losses down to L70_G30 −0.021). Reported as a loss row in T1, not a win.
4. **The dual-instability *magnitude* axis is a NULL/inverted predictor** (r=−0.22) and is **not**
   claimed as the regime axis. Mechanism stated qualitatively only.
5. **Deployment "flips" is misleading at near-trivial loose caps** (loose Oct L80/L90: TraLO has
   *more* flips than post-hoc). The robust deployment criterion is **native satisfaction + preserved
   macro-F1** (holds in 100% of regime×dataset rows), not raw flip count.
6. **Coverage caveats:** asym regimes are single-backbone (MNV3); Oct mechanism logs are
   server-only; the sharp cells rest on n=4 seeds × few cells (Oct L40 = local MNV3-only).

The credible paper is: **universal deployment win + one backbone-general hard win (Oct tight-binding)
+ one preliminary single-backbone hard win (Derm asym G<L), with the symmetric tissue/derm grid an
honest tie and the mechanism stated qualitatively** — with the experiments in §4 specified to close
the asym-cross-backbone and component-ablation gaps that any reviewer will demand.
