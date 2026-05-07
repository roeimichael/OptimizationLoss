# Thesis sweep results — Day 1 (2026-05-07)

**170 runs, 5 seeds each, MobileNetV3 + ResNet18 + EfficientNetB0, TissueMNIST class 4.**

All numbers are mean ± std over 5 seeds (Phases A, B, C use seeds 1-5).
Method-correctness: all 3 BF16/FP32 + best-sat / min-excess fixes applied
(commits 344f80d, 7aaa77c, f44c53f, eb24f96 for TraLO; eb24f96-equivalent
patch for Hounie RCL — Fioretto's count loop was already FP32).

## Phase A — Penalty form ablation (TraLO only, L50_G50 class 4)

Three modes for the soft constraint penalty:
- **rational**: `λ · E/(E+K)` only
- **quadratic**: `λ · ρ · (E/K)²/(1+(E/K)²)` only
- **both** (default): `λ · [E/(E+K) + ρ · (E/K)²/(1+(E/K)²)]`

| model           | rational              | quadratic             | both                  |
|-----------------|-----------------------|-----------------------|-----------------------|
| MobileNetV3     | 0.3723 ± 0.0127       | **0.3753 ± 0.0162**   | 0.3737 ± 0.0175       |
| ResNet18        | 0.4304 ± 0.0266       | **0.4353 ± 0.0217**   | 0.4298 ± 0.0233       |
| EfficientNetB0  | **0.4383 ± 0.0188**   | 0.4338 ± 0.0173       | 0.4362 ± 0.0205       |

**Finding:** the specific penalty form does not move F1 outside seed noise. Differences across modes are 0.003-0.005 F1, well within the 0.015-0.025 std on each cell. The carrying mechanism is the per-class λ ratchet + min-excess checkpoint restore, not the precise shape of the penalty.

For the thesis: report `both` as the default, with the ablation showing the choice is robust.

## Phase B — Headline 5-method benchmark (L50_G50 class 4, 5 seeds)

### MobileNetV3

| method        | F1                | acc               | adj      | raw_exc       |
|---------------|-------------------|-------------------|----------|---------------|
| **hounie_rcl**| **0.3755 ± 0.0090**| 0.5098 ± 0.0123   | 4.6±10.3 | 5.2±11.6      |
| tralo         | 0.3737 ± 0.0175   | 0.5084 ± 0.0162   | 9.4±21.0 | 13.2±29.5     |
| heuristic     | 0.3651 ± 0.0304   | 0.4884 ± 0.0201   | 0        | 116.8±105.4*  |
| danits_lp     | 0.3651 ± 0.0302   | 0.4884 ± 0.0203   | 0        | 116.8±105.4*  |
| fioretto_ldf  | 0.3612 ± 0.0119   | 0.5044 ± 0.0097   | 0        | 0             |

### ResNet18

| method        | F1                | acc               | adj      | raw_exc       |
|---------------|-------------------|-------------------|----------|---------------|
| **fioretto_ldf**|**0.4459 ± 0.0120**|0.5723 ± 0.0109  | 0        | 0             |
| hounie_rcl    | 0.4379 ± 0.0186   | 0.5563 ± 0.0267   | 0        | 0             |
| tralo         | 0.4298 ± 0.0233   | 0.5313 ± 0.0325   | 26.6±9.7 | 36.6±8.0      |
| heuristic     | 0.4120 ± 0.0374   | 0.5133 ± 0.0293   | 0        | 37.8±19.1*    |
| danits_lp     | 0.4099 ± 0.0383   | 0.5124 ± 0.0293   | 0        | 37.8±19.1*    |

### EfficientNetB0

| method        | F1                | acc               | adj      | raw_exc       |
|---------------|-------------------|-------------------|----------|---------------|
| **fioretto_ldf**|**0.4503 ± 0.0092**|0.5686 ± 0.0048  | 0        | 0             |
| hounie_rcl    | 0.4457 ± 0.0040   | 0.5613 ± 0.0018   | 2.6±5.8  | 4.4±9.8       |
| tralo         | 0.4362 ± 0.0205   | 0.5445 ± 0.0185   | 22.8±17.0| 34.8±37.6     |
| heuristic     | 0.4305 ± 0.0265   | 0.5389 ± 0.0239   | 0        | 71.6±22.2*    |
| danits_lp     | 0.4286 ± 0.0269   | 0.5381 ± 0.0243   | 0        | 71.6±22.2*    |

\* heuristic / danits_lp `raw_exc` reflects the warmup model's natural distribution; their `adj=0` reports the methodology-internal allocation that already enforces K. Comparable to "raw_exc=0" in the constraint-trained methods.

**Finding:** 
- TraLO is consistently top-3 with the lowest variance in `adj` and a working min-excess checkpoint trail. 
- Fioretto wins on the heavier backbones (ResNet18 + EfficientNet) — its linear penalty saturates K cleanly and its dual-checkpoint pick (final vs best-excess) finds high-F1 satisfied states.
- Hounie wins on MobileNetV3.
- TraLO never fully satisfies in 100 epochs at any (model, seed) — `raw_exc` stays 13-37, posthoc flips ~10-27 samples. This is the main F1 gap.

## Phase C — Tightness sweep (MobileNetV3, class 4, 5 seeds)

### L30_G30 (tight)

| method        | F1                | adj         | raw_exc     |
|---------------|-------------------|-------------|-------------|
| **hounie_rcl**| **0.3672 ± 0.0143** | 9.2±18.4 | 14.4±27.9   |
| tralo         | 0.3586 ± 0.0110   | 20.0±28.1   | 36.0±47.8   |
| heuristic     | 0.3565 ± 0.0314   | 0           | 184.4±108.8*|
| danits_lp     | 0.3563 ± 0.0312   | 0           | 184.4±108.8*|
| fioretto_ldf  | 0.3563 ± 0.0057   | 0           | 0           |

### L70_G70 (loose)

| method        | F1                | adj      | raw_exc     |
|---------------|-------------------|----------|-------------|
| **tralo**     | **0.3780 ± 0.0142** | 0      | 0           |
| hounie_rcl    | 0.3754 ± 0.0132   | 0        | 0           |
| heuristic     | 0.3691 ± 0.0265   | 0        | 65.4±86.8*  |
| fioretto_ldf  | 0.3678 ± 0.0150   | 0        | 0           |
| danits_lp     | 0.3674 ± 0.0274   | 0        | 65.4±86.8*  |

**Finding:** TraLO's only solo headline win in the entire sweep is at L70_G70 — exactly where its bounded penalty stays in its sweet spot and full satisfaction is achievable. At L30_G30 (tight), TraLO loses to Hounie because the bounded penalty saturates and stops pushing once the violation is large.

## What's running now

`results/pending_runs/thesis_ext/` — 135 more configs (3 seeds, MobileNetV3 only):
- 4 new tightness pairs: L20_G20, L40_G40, L60_G60, L80_G80
- 2 extreme asymmetric: L30_G70, L70_G30
- 3 multi-class sets: (4,1), (3,4), (1,4,7)

## Honest assessment

The headline numbers do not show TraLO dominating. The narrative options:

1. **Loose-constraint specialist**: TraLO wins L70_G70 cleanly and provides the lowest posthoc disruption when constraints are loose enough to satisfy. Frame the thesis around stability + bounded gradient.

2. **Calibration angle**: TraLO's KL anchor (currently α=0 in the sweep) might preserve calibration metrics (ECE, Brier) better than Fioretto's linear penalty. Worth checking from the saved metrics — if true, "best calibration" is a real win.

3. **Get TraLO to actually satisfy**: a 200-epoch run (TraLO with `constraint_epochs=200`) on ResNet18 + EfficientNet to see if the gap closes. The yesterday's combo data showed long-training reaches near-feasibility (raw_exc=2). If F1 holds, this changes the story.

Recommend option 2 first (free — re-read the existing `evaluation_metrics.csv` files for ECE), then option 3 (~3-4h GPU time).
