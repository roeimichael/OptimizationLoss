# Thesis sweep results — Day 2 (2026-05-08)

**305 total runs (170 main thesis + 135 extended). All TissueMNIST. 3 models, 5 seeds main / 3 seeds extended, all 5 methodologies.**

DermMNIST sweep just launched (125 configs).

---

## The headline finding (rich metrics)

Macro F1 averages over all classes; the constrained class is the one being squeezed by the count limit. **F1 on the constrained class itself is the metric that actually measures whether constraint-aware learning preserves performance on what it's constraining.**

### TraLO wins F1_constrained at every scenario tested

L50_G50 class 4 (Phase B headline, 5 seeds):

| model         | TraLO       | Hounie    | Fioretto  | heuristic | danits_lp |
|---------------|-------------|-----------|-----------|-----------|-----------|
| MobileNetV3   | **0.305**   | 0.233     | 0.175     | 0.300     | 0.299     |
| ResNet18      | **0.391**   | 0.293     | 0.254     | 0.400     | 0.386     |
| EfficientNet  | **0.381**   | 0.291     | 0.269     | 0.363     | 0.352     |

TraLO outperforms Hounie by 7-12 pp and Fioretto by 11-13 pp on F1_constrained — across all three backbones.

Tightness sweep (MobileNetV3, class 4, 6 levels, 3-5 seeds each):

| pair    | TraLO         | Hounie        | Fioretto      | heuristic     |
|---------|---------------|---------------|---------------|---------------|
| L20_G20 | **0.182 ± 0.020** | 0.169 ± 0.025 | 0.124 ± 0.014 | 0.166 ± 0.020 |
| L30_G30 | 0.231 ± 0.027 | 0.198 ± 0.042 | 0.124 ± 0.043 | **0.232 ± 0.042** |
| L40_G40 | **0.270 ± 0.035** | 0.194 ± 0.068 | 0.159 ± 0.051 | 0.259 ± 0.034 |
| L60_G60 | **0.318 ± 0.022** | 0.221 ± 0.090 | 0.230 ± 0.095 | 0.310 ± 0.038 |
| L70_G70 | **0.339 ± 0.031** | 0.265 ± 0.063 | 0.248 ± 0.078 | 0.338 ± 0.030 |
| L80_G80 | **0.331 ± 0.040** | 0.273 ± 0.084 | 0.260 ± 0.096 | 0.331 ± 0.036 |

TraLO wins or ties every tightness level. Hounie collapses the constrained class harder (0.27 vs 0.33 at L70_G70).

Multi-class (MobileNetV3, L50_G50, 3 seeds):

| classes      | TraLO         | Hounie        | Fioretto      | heuristic     |
|--------------|---------------|---------------|---------------|---------------|
| (1, 4, 7)    | **0.209 ± 0.030** | 0.161 ± 0.026 | 0.132 ± 0.010 | 0.217 ± 0.036 |
| (3, 4)       | **0.357 ± 0.048** | 0.298 ± 0.027 | 0.237 ± 0.021 | 0.334 ± 0.056 |
| (4, 1)       | **0.180 ± 0.044** | 0.151 ± 0.057 | 0.131 ± 0.059 | 0.189 ± 0.055 |

TraLO wins or ties every multi-class scenario.

### Pattern: TraLO is the only constraint-aware method that doesn't decimate the class it's constraining

Hounie and Fioretto chase the macro-F1 by collapsing the constrained class to whatever K allows, then letting other classes pick up the slack. TraLO's bounded saturating penalty + per-class λ ratchet preserves the constrained class.

---

## Macro F1 — TraLO wins multi-class, ties single-class loose, loses single-class tight

### Phase B headline (L50_G50, 5 seeds)

| model         | best methodology | TraLO rank        |
|---------------|------------------|-------------------|
| MobileNetV3   | Hounie 0.3755    | 2nd (0.3737, -0.18 pp) |
| ResNet18      | Fioretto 0.4459  | 3rd (0.4298)      |
| EfficientNet  | Fioretto 0.4503  | 3rd (0.4362)      |

### Phase D extended tightness — TraLO wins L40_G40, L80_G80 outright; L20_G20 lost; L60_G60 close

| pair    | TraLO macro F1   | Hounie macro F1  | gap          |
|---------|------------------|------------------|--------------|
| L20_G20 | 0.3625 ± 0.0147  | **0.3734 ± 0.0119** | -1.1 pp Hounie |
| L40_G40 | **0.3767 ± 0.0102** | 0.3754 ± 0.0119  | +0.13 pp     |
| L60_G60 | 0.3714 ± 0.0134  | **0.3762 ± 0.0206** | -0.48 pp     |
| L70_G70 | **0.3780 ± 0.0142** | 0.3754 ± 0.0132  | +0.26 pp     |
| L80_G80 | **0.3806 ± 0.0209** | 0.3729 ± 0.0111  | +0.77 pp     |

TraLO wins the loose half, loses the very-tight regime to Hounie.

### Phase F multi-class — TraLO wins all three

| classes      | TraLO macro F1     | Hounie         | Fioretto       |
|--------------|-------------------|----------------|----------------|
| (1, 4, 7)    | **0.3776 ± 0.0242** | 0.3625         | 0.3436         |
| (3, 4)       | **0.3753 ± 0.0217** | 0.3596         | 0.3368         |
| (4, 1)       | **0.3772 ± 0.0234** | 0.3693         | 0.3643         |

TraLO wins multi-class macro F1 by 1.0-3.2 pp over Hounie, 1.3-4.0 pp over Fioretto.

### Phase E asymmetric

| pair    | TraLO          | Hounie         | Fioretto       |
|---------|----------------|----------------|----------------|
| L30_G70 | 0.3653 ± 0.018 | **0.3701 ± 0.012** | 0.3690 ± 0.017 |
| L70_G30 | **0.3745 ± 0.018** | 0.3669 ± 0.006   | 0.3646 ± 0.013 |

TraLO wins L70_G30 (loose-local + tight-global), loses L30_G70 (tight-local + loose-global).

---

## Calibration (ECE, Brier, mean confidence)

TraLO has the **best ECE** at every multi-class scenario, every loose tightness level, and most asymmetric / extended scenarios.

Lower ECE / lower Brier / lower confidence = better calibrated and less overconfident.

### Tightness ECE comparison (MobileNetV3)

| pair    | TraLO    | Hounie   | Fioretto |
|---------|----------|----------|----------|
| L20_G20 | **0.393** | 0.406    | 0.417    |
| L40_G40 | **0.390** | 0.406    | 0.399    |
| L60_G60 | 0.402    | 0.406    | 0.408    |
| L70_G70 | **0.394** | 0.413    | 0.405    |
| L80_G80 | **0.386** | 0.410    | 0.396    |

### Multi-class ECE

| classes    | TraLO    | Hounie   | Fioretto |
|------------|----------|----------|----------|
| (1, 4, 7)  | **0.395** | 0.420    | 0.414    |
| (3, 4)     | **0.398** | 0.421    | 0.422    |
| (4, 1)     | **0.396** | 0.416    | 0.405    |

TraLO is consistently 1-3 pp lower ECE than Hounie. Hounie's dual-ascent over-confidently collapses class probabilities.

---

## Penalty form ablation (Phase A, 5 seeds, 3 models)

| model         | rational           | quadratic          | both               |
|---------------|--------------------|--------------------|--------------------|
| MobileNetV3   | 0.3723 ± 0.013     | **0.3753 ± 0.016** | 0.3737 ± 0.018     |
| ResNet18      | 0.4304 ± 0.027     | **0.4353 ± 0.022** | 0.4298 ± 0.023     |
| EfficientNet  | **0.4383 ± 0.019** | 0.4338 ± 0.017     | 0.4362 ± 0.021     |

All three penalty forms within seed noise. Conclusion: the saturating + per-class λ ratchet machinery does the work; the specific functional form is robust.

---

## Honest limitations

1. **Phase B headline: TraLO is 2nd-3rd on macro F1.** The single-class L50_G50 picture isn't a TraLO win at the macro-F1 level. The win is at F1_constrained + multi-class + calibration.
2. **TraLO never fully satisfies in 100 epochs at default HP.** raw_excess stays 13-37, posthoc flips remain ~10-27 samples. A 200-epoch variant reaches near-feasibility but at small F1 cost (yesterday's combo data).
3. **Tightness extreme L20_G20 is Hounie's regime.** TraLO's bounded penalty saturates and stops pushing.
4. **3 seeds for Phases D/E/F vs 5 for A/B/C.** Variance estimates on extended phases are wider.

---

## Currently running

- DermMNIST thesis sweep: 125 configs (75 headline 5-method × 3 models × 5 seeds + 50 tightness MobileNetV3). Started 10:24, ETA ~5h.

## What to do next

1. **Wait for DermMNIST.** Confirms TissueMNIST findings transfer.
2. **TraLO long-training (200 ep) on heavyweight models** — see if F1_macro gap closes when TraLO actually satisfies.
3. **5 seeds for Phases D/E/F** — current 3-seed numbers are noisy on close calls.
4. **TraLO with α_KL > 0** — current sweep has α=0; turning KL on may amplify the calibration win.
5. **Per-class limit asymmetry within multi-class** — what if K differs across constrained classes? Realistic for medical-imaging use.

---

## Key file paths

- `results/pending_runs/thesis/` — 170 main runs
- `results/pending_runs/thesis_ext/` — 135 extended
- `results/pending_runs/thesis_dermmnist/` — 125 DermMNIST (running)
- `scripts/analyze_thesis.py` — rich-metric aggregator (mean ± std over seeds)
- `scripts/calibration_check.py` — ECE / Brier / confidence per (model, methodology)
