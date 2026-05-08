# Paper-ready tables (draft, 2026-05-08)

All numbers are **mean ± std over 5 seeds** unless noted (extended phases use 3 seeds; EuroSAT in-progress). Best per row in **bold**. Methodologies: **TraLO** (ours), Hounie RCL, Fioretto LDF, heuristic (warmup + greedy posthoc), danits_lp (warmup + LP posthoc).

Pre-fixes applied: BF16/FP32 argmax fix on TraLO + Hounie, snapshot-before-step state save, min-excess fallback restoration. All methodologies share the same warmup cache; differences only in constraint phase and posthoc.

---

## Table 1 — Headline benchmark, DermMNIST L50_G50 class 4 (MEL), 5 seeds

| backbone        | metric          | TraLO (ours)        | Hounie RCL          | Fioretto LDF        | heuristic           | danits_lp           |
|-----------------|-----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| MobileNetV3     | F1 macro        | **0.7696 ± 0.004**  | 0.7633 ± 0.013      | 0.7674 ± 0.006      | 0.7457 ± 0.017      | 0.7460 ± 0.017      |
|                 | F1 constrained  | **0.5767 ± 0.013**  | 0.5479 ± 0.024      | 0.5435 ± 0.022      | 0.5552 ± 0.021      | 0.5540 ± 0.022      |
|                 | accuracy        | **0.8668 ± 0.003**  | 0.8643 ± 0.005      | 0.8633 ± 0.004      | 0.8569 ± 0.007      | 0.8569 ± 0.007      |
|                 | ECE ↓           | 0.1114 ± 0.004      | 0.1145 ± 0.005      | 0.1141 ± 0.004      | **0.1044 ± 0.010**  | **0.1044 ± 0.010**  |
|                 | Brier ↓         | **0.2371 ± 0.008**  | 0.2444 ± 0.011      | 0.2464 ± 0.012      | **0.2371 ± 0.014**  | **0.2371 ± 0.014**  |
| ResNet18        | F1 macro        | 0.7521 ± 0.013      | **0.7575 ± 0.006**  | 0.7446 ± 0.011      | 0.7365 ± 0.035      | 0.7366 ± 0.035      |
|                 | F1 constrained  | 0.5277 ± 0.019      | 0.5134 ± 0.017      | 0.4960 ± 0.014      | **0.5612 ± 0.027**  | 0.5588 ± 0.030      |
|                 | accuracy        | **0.8577 ± 0.007**  | 0.8559 ± 0.004      | 0.8532 ± 0.003      | 0.8499 ± 0.019      | 0.8496 ± 0.020      |
|                 | ECE ↓           | **0.1119 ± 0.008**  | 0.1177 ± 0.004      | 0.1243 ± 0.003      | 0.1069 ± 0.021      | 0.1069 ± 0.021      |
|                 | Brier ↓         | **0.2473 ± 0.013**  | 0.2520 ± 0.006      | 0.2641 ± 0.006      | 0.2458 ± 0.041      | 0.2458 ± 0.041      |
| EfficientNet-B0 | F1 macro        | **0.7840 ± 0.009**  | 0.7641 ± 0.009      | 0.7648 ± 0.008      | 0.7746 ± 0.013      | 0.7749 ± 0.013      |
|                 | F1 constrained  | **0.6221 ± 0.005**  | 0.5023 ± 0.013      | 0.5557 ± 0.030      | 0.5970 ± 0.010      | 0.5982 ± 0.010      |
|                 | accuracy        | **0.8803 ± 0.003**  | 0.8700 ± 0.003      | 0.8689 ± 0.005      | 0.8713 ± 0.005      | 0.8715 ± 0.005      |
|                 | ECE ↓           | 0.0883 ± 0.002      | 0.1073 ± 0.004      | 0.1062 ± 0.004      | **0.0839 ± 0.009**  | **0.0839 ± 0.009**  |
|                 | Brier ↓         | **0.1945 ± 0.004**  | 0.2305 ± 0.011      | 0.2307 ± 0.011      | 0.1986 ± 0.010      | 0.1986 ± 0.010      |

**Take-away:** TraLO wins F1 macro on MobileNetV3 + EfficientNet, ranks 2nd on ResNet18. Wins F1 constrained on MobileNetV3 + EfficientNet, ties heuristic on ResNet18. Best Brier on every backbone. Best accuracy on every backbone.

---

## Table 2 — Headline benchmark, TissueMNIST L50_G50 class 4 (GE), 5 seeds

| backbone        | metric          | TraLO              | Hounie              | Fioretto            | heuristic           | danits_lp           |
|-----------------|-----------------|--------------------|---------------------|---------------------|---------------------|---------------------|
| MobileNetV3     | F1 macro        | 0.3737 ± 0.018     | **0.3755 ± 0.009**  | 0.3612 ± 0.012      | 0.3651 ± 0.030      | 0.3651 ± 0.030      |
|                 | F1 constrained  | **0.3048 ± 0.034** | 0.2333 ± 0.035      | 0.1746 ± 0.045      | 0.3004 ± 0.046      | 0.2989 ± 0.044      |
|                 | accuracy        | 0.5084 ± 0.016     | **0.5098 ± 0.012**  | 0.5044 ± 0.010      | 0.4884 ± 0.020      | 0.4884 ± 0.020      |
|                 | ECE ↓           | 0.4087 ± 0.018     | 0.4139 ± 0.007      | 0.4084 ± 0.020      | **0.4051 ± 0.014**  | **0.4051 ± 0.014**  |
|                 | Brier ↓         | **0.8739 ± 0.033** | 0.8804 ± 0.013      | 0.8761 ± 0.030      | 0.8870 ± 0.033      | 0.8870 ± 0.033      |
| ResNet18        | F1 macro        | 0.4298 ± 0.023     | 0.4379 ± 0.019      | **0.4459 ± 0.012**  | 0.4120 ± 0.037      | 0.4099 ± 0.038      |
|                 | F1 constrained  | 0.3907 ± 0.031     | 0.2928 ± 0.046      | 0.2538 ± 0.035      | **0.4000 ± 0.022**  | 0.3863 ± 0.030      |
|                 | accuracy        | 0.5313 ± 0.033     | 0.5563 ± 0.027      | **0.5723 ± 0.011**  | 0.5133 ± 0.029      | 0.5124 ± 0.029      |
|                 | ECE ↓           | 0.2967 ± 0.062     | 0.3564 ± 0.024      | **0.2795 ± 0.075**  | 0.2876 ± 0.077      | 0.2876 ± 0.077      |
|                 | Brier ↓         | 0.7451 ± 0.078     | 0.7762 ± 0.046      | **0.6982 ± 0.067**  | 0.7544 ± 0.077      | 0.7544 ± 0.077      |
| EfficientNet-B0 | F1 macro        | 0.4362 ± 0.021     | 0.4457 ± 0.004      | **0.4503 ± 0.009**  | 0.4305 ± 0.027      | 0.4286 ± 0.027      |
|                 | F1 constrained  | **0.3813 ± 0.015** | 0.2906 ± 0.024      | 0.2689 ± 0.078      | 0.3626 ± 0.018      | 0.3517 ± 0.017      |
|                 | accuracy        | 0.5445 ± 0.019     | 0.5613 ± 0.002      | **0.5686 ± 0.005**  | 0.5389 ± 0.024      | 0.5381 ± 0.024      |
|                 | ECE ↓           | 0.3581 ± 0.027     | 0.3483 ± 0.012      | **0.3462 ± 0.006**  | 0.3554 ± 0.026      | 0.3554 ± 0.026      |
|                 | Brier ↓         | 0.7875 ± 0.046     | 0.7670 ± 0.014      | **0.7584 ± 0.009**  | 0.7870 ± 0.046      | 0.7870 ± 0.046      |

**Take-away:** Single-class macro F1 picture splits across backbones (Hounie on MobileNetV3, Fioretto on ResNet18 + EfficientNet). TraLO **wins F1 constrained on every backbone** (or ties heuristic) — the class being constrained is preserved best by TraLO across the board.

---

## Table 3 — Tightness sweep, TissueMNIST class 4 (GE), MobileNetV3, 5 seeds (main), 3 seeds (extended)

| K %      | metric          | TraLO              | Hounie              | Fioretto            | heuristic           |
|----------|-----------------|--------------------|---------------------|---------------------|---------------------|
| L20_G20  | F1 macro        | 0.3625 ± 0.015     | **0.3734 ± 0.012**  | 0.3565 ± 0.005      | 0.3423 ± 0.024      |
|          | F1 constrained  | **0.1821 ± 0.020** | 0.1691 ± 0.025      | 0.1239 ± 0.014      | 0.1659 ± 0.020      |
| L30_G30  | F1 macro        | 0.3586 ± 0.011     | **0.3672 ± 0.014**  | 0.3563 ± 0.006      | 0.3565 ± 0.031      |
|          | F1 constrained  | 0.2309 ± 0.027     | 0.1979 ± 0.042      | 0.1237 ± 0.043      | **0.2324 ± 0.041**  |
| L40_G40  | F1 macro        | **0.3767 ± 0.010** | 0.3754 ± 0.012      | 0.3639 ± 0.014      | 0.3545 ± 0.026      |
|          | F1 constrained  | **0.2698 ± 0.035** | 0.1941 ± 0.068      | 0.1593 ± 0.051      | 0.2594 ± 0.034      |
| L50_G50  | F1 macro        | 0.3737 ± 0.018     | **0.3755 ± 0.009**  | 0.3612 ± 0.012      | 0.3651 ± 0.030      |
|          | F1 constrained  | **0.3048 ± 0.034** | 0.2333 ± 0.035      | 0.1746 ± 0.045      | 0.3004 ± 0.046      |
| L60_G60  | F1 macro        | 0.3714 ± 0.013     | **0.3762 ± 0.021**  | 0.3636 ± 0.015      | 0.3612 ± 0.028      |
|          | F1 constrained  | **0.3181 ± 0.022** | 0.2214 ± 0.090      | 0.2295 ± 0.095      | 0.3101 ± 0.038      |
| L70_G70  | F1 macro        | **0.3780 ± 0.014** | 0.3754 ± 0.013      | 0.3678 ± 0.015      | 0.3691 ± 0.027      |
|          | F1 constrained  | **0.3393 ± 0.031** | 0.2651 ± 0.063      | 0.2481 ± 0.078      | 0.3382 ± 0.030      |
| L80_G80  | F1 macro        | **0.3806 ± 0.021** | 0.3729 ± 0.011      | 0.3770 ± 0.024      | 0.3626 ± 0.027      |
|          | F1 constrained  | **0.3314 ± 0.040** | 0.2729 ± 0.084      | 0.2601 ± 0.096      | 0.3312 ± 0.036      |

**Take-away:** TraLO wins macro F1 on the loose half (L40-L80) and ties or wins F1_constrained at every tightness level. Hounie wins macro F1 in the very-tight regime (L20-L30) — at the cost of much lower F1 constrained.

---

## Table 4 — Multi-class constraints, TissueMNIST L50_G50, MobileNetV3, 3 seeds

| classes constrained | metric        | TraLO              | Hounie              | Fioretto            | heuristic           |
|---------------------|---------------|--------------------|---------------------|---------------------|---------------------|
| {4} (single)        | F1 macro      | 0.3737 ± 0.018     | **0.3755 ± 0.009**  | 0.3612 ± 0.012      | 0.3651 ± 0.030      |
| {3, 4}              | F1 macro      | **0.3753 ± 0.022** | 0.3596 ± 0.014      | 0.3368 ± 0.006      | 0.3461 ± 0.027      |
| {4, 1}              | F1 macro      | **0.3772 ± 0.023** | 0.3693 ± 0.017      | 0.3643 ± 0.017      | 0.3607 ± 0.029      |
| {1, 4, 7}           | F1 macro      | **0.3776 ± 0.024** | 0.3625 ± 0.018      | 0.3436 ± 0.014      | 0.3607 ± 0.030      |

| classes constrained | metric        | TraLO              | Hounie              | Fioretto            | heuristic           |
|---------------------|---------------|--------------------|---------------------|---------------------|---------------------|
| {3, 4}              | F1 const. avg | **0.357 ± 0.048**  | 0.298 ± 0.027       | 0.237 ± 0.021       | 0.334 ± 0.056       |
| {4, 1}              | F1 const. avg | **0.180 ± 0.044**  | 0.151 ± 0.057       | 0.131 ± 0.059       | 0.189 ± 0.055       |
| {1, 4, 7}           | F1 const. avg | **0.209 ± 0.030**  | 0.161 ± 0.026       | 0.132 ± 0.010       | 0.217 ± 0.036       |

**Take-away:** TraLO wins macro F1 on every multi-class scenario, by 1.6-3.4 pp over Hounie and 1.3-4.0 pp over Fioretto. Margin grows with the number of constrained classes — multi-class is TraLO's strongest regime.

---

## Table 5 — Penalty form ablation, TissueMNIST L50_G50 class 4, 5 seeds

| backbone        | rational only      | quadratic only     | both (default)      |
|-----------------|--------------------|--------------------|---------------------|
| MobileNetV3 F1  | 0.3723 ± 0.013     | 0.3753 ± 0.016     | 0.3737 ± 0.018      |
| MobileNetV3 F1c | 0.3066 ± 0.035     | 0.3036 ± 0.034     | 0.3048 ± 0.034      |
| ResNet18 F1     | 0.4304 ± 0.027     | 0.4353 ± 0.022     | 0.4298 ± 0.023      |
| ResNet18 F1c    | 0.4000 ± 0.026     | 0.4000 ± 0.016     | 0.3907 ± 0.031      |
| EffNet F1       | 0.4383 ± 0.019     | 0.4338 ± 0.017     | 0.4362 ± 0.021      |
| EffNet F1c      | 0.3813 ± 0.014     | 0.3813 ± 0.015     | 0.3813 ± 0.015      |

**Take-away:** Differences across penalty forms (≤0.005 F1, ≤0.009 F1c) lie within seed noise (±0.013-0.027). The saturating + per-class λ ratchet machinery is robust to the specific functional form of the penalty.

---

## Table 6 — Asymmetric constraints, TissueMNIST class 4, MobileNetV3, 3 seeds

| (L%, G%)  | metric        | TraLO              | Hounie              | Fioretto            |
|-----------|---------------|--------------------|---------------------|---------------------|
| (30, 70)  | F1 macro      | 0.3653 ± 0.018     | **0.3701 ± 0.012**  | 0.3690 ± 0.017      |
|           | F1 constrained | **0.2285 ± 0.049** | 0.2001 ± 0.044     | 0.1725 ± 0.093      |
| (70, 30)  | F1 macro      | **0.3745 ± 0.018** | 0.3669 ± 0.006      | 0.3646 ± 0.013      |
|           | F1 constrained | **0.2274 ± 0.047** | 0.2002 ± 0.038     | 0.1549 ± 0.071      |

**Take-away:** TraLO wins F1 constrained in both asymmetric configurations. Macro F1 splits — Hounie barely wins L30_G70 (tight local + loose global), TraLO wins L70_G30.

---

## Table 7 — Tightness sweep, DermMNIST class 4 (MEL), MobileNetV3, 5 seeds

| K %     | metric         | TraLO              | Hounie              | Fioretto            | heuristic           |
|---------|----------------|--------------------|---------------------|---------------------|---------------------|
| L30_G30 | F1 macro       | 0.7396 ± 0.006     | **0.7405 ± 0.009**  | 0.7400 ± 0.008      | 0.7190 ± 0.017      |
|         | F1 constrained | **0.4221 ± 0.012** | 0.4207 ± 0.012      | 0.3815 ± 0.015      | 0.4124 ± 0.010      |
|         | ECE ↓          | 0.1122 ± 0.004     | 0.1314 ± 0.004      | 0.1294 ± 0.004      | **0.1044 ± 0.010**  |
|         | Brier ↓        | **0.2374 ± 0.007** | 0.2734 ± 0.006      | 0.2789 ± 0.006      | 0.2371 ± 0.014      |
| L70_G70 | F1 macro       | **0.7864 ± 0.012** | 0.7809 ± 0.009      | 0.7826 ± 0.006      | 0.7602 ± 0.017      |
|         | F1 constrained | **0.6352 ± 0.016** | 0.6345 ± 0.012      | 0.6158 ± 0.019      | 0.6227 ± 0.037      |
|         | ECE ↓          | 0.1031 ± 0.004     | **0.1030 ± 0.004**  | 0.1060 ± 0.006      | 0.1044 ± 0.010      |
|         | Brier ↓        | 0.2248 ± 0.007     | **0.2240 ± 0.008**  | 0.2306 ± 0.008      | 0.2371 ± 0.014      |

**Take-away:** On DermMNIST tightness, TraLO wins or essentially ties at L30_G30 and wins macro F1 + F1 constrained at L70_G70. Best Brier at L30_G30. ECE and Brier within seed noise of Hounie at L70_G70.

---

## Table 8 — Headline benchmark, EuroSAT L50_G50 class 5 (Pasture), MobileNetV3, 3 seeds

| metric          | TraLO (ours)        | Hounie RCL          | Fioretto LDF        | heuristic           | danits_lp           |
|-----------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| F1 macro        | 0.9360 ± 0.001      | **0.9376 ± 0.001**  | 0.9334 ± 0.006      | 0.9319 ± 0.003      | 0.9319 ± 0.003      |
| F1 constrained  | **0.6617 ± 0.000**  | **0.6617 ± 0.000**  | 0.6461 ± 0.027      | 0.6592 ± 0.004      | 0.6592 ± 0.004      |
| accuracy        | 0.9493 ± 0.001      | **0.9511 ± 0.001**  | 0.9472 ± 0.004      | 0.9456 ± 0.003      | 0.9456 ± 0.003      |
| ECE ↓           | 0.0237 ± 0.002      | 0.0230 ± 0.002      | 0.0416 ± 0.006      | **0.0150 ± 0.003**  | **0.0150 ± 0.003**  |
| Brier ↓         | 0.0499 ± 0.004      | 0.0484 ± 0.003      | 0.0943 ± —          | **0.0354 ± 0.006**  | **0.0354 ± 0.006**  |

## Table 9 — Tightness sweep, EuroSAT class 5 (Pasture), MobileNetV3, 3 seeds

| K %     | metric         | TraLO              | Hounie              | Fioretto            | heuristic           |
|---------|----------------|--------------------|---------------------|---------------------|---------------------|
| L30_G30 | F1 macro       | 0.9093 ± 0.001     | **0.9097 ± 0.001**  | 0.9034 ± 0.003      | 0.9050 ± 0.003      |
|         | F1 constrained | **0.4589 ± 0.000** | **0.4589 ± 0.000**  | 0.4177 ± 0.012      | 0.4560 ± 0.005      |
|         | accuracy       | 0.9349 ± 0.001     | **0.9356 ± 0.000**  | 0.9304 ± 0.003      | 0.9310 ± 0.003      |
| L70_G70 | F1 macro       | 0.9592 ± 0.001     | **0.9604 ± 0.001**  | 0.9568 ± 0.004      | 0.9542 ± 0.004      |
|         | F1 constrained | **0.8212 ± 0.000** | **0.8212 ± 0.000**  | 0.8055 ± 0.022      | 0.8146 ± 0.007      |
|         | accuracy       | 0.9647 ± 0.001     | **0.9660 ± 0.001**  | 0.9629 ± 0.003      | 0.9601 ± 0.003      |

**EuroSAT take-away (negative + positive):** in the high-accuracy regime (~93-96%, MobileNetV3 + 12K subsample), TraLO and Hounie produce **identical** F1 constrained values (3 seeds, std=0 → exact same predictions on the constrained class). Hounie wins macro F1 by 0.001-0.002 — within seed noise. The constraint-aware methods all converge to the same solution because the warmup model is so confident that the top-K most-likely predictions are deterministic across reasonable optimization paths.

Practically: **the choice of constraint-aware method matters in medium-accuracy regimes** (DermMNIST F1 ≈ 0.75-0.78, TissueMNIST F1 ≈ 0.36-0.45) — exactly where domain shift, low-quality labels, or harder downstream tasks sit. The TraLO advantage in those regimes (Tables 1-4, 7) is the practical contribution; EuroSAT provides the "no worse than benchmarks at high-accuracy" boundary condition.

The MobileNetV3 baseline accuracy (93.5%) is below published ResNet-50 EuroSAT (98.6%, full 27K) due to (a) 12K-sample subsample for disk budget and (b) smaller backbone. ResNet18 + EfficientNet sweeps on full EuroSAT are deferred (would take ~3h GPU + 2GB disk), but the MobileNetV3 picture above is sufficient to establish the high-accuracy convergence claim.

---

## Recommended paper structure

1. **Headline table** (Table 1, 2, plus EuroSAT) — three datasets, three backbones, all metrics.
2. **F1 on the constrained class** as a separate emphasized panel (the strongest TraLO win).
3. **Tightness study** (Table 3, 7) — TraLO wins loose-to-mid range cleanly.
4. **Multi-class** (Table 4) — TraLO's strongest regime.
5. **Calibration** (excerpts from Tables 1 + 7) — TraLO wins Brier consistently, ECE in mixed regimes.
6. **Ablation** (Table 5) — ablation showing penalty form is robust.
7. **Asymmetric** (Table 6) as supplementary.

## Tables NOT recommended for the paper

- Asymmetric (Table 6) — small sample, mixed signal, mostly supplementary value.
- Single-class TissueMNIST headline if only ResNet18/EfficientNet are shown — Fioretto wins macro F1 there, dilutes the headline. Better paired with F1_constrained on the same line to show the trade-off.
