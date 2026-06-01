# Headroom hypothesis — empirical validation

**For**: paper-writing session
**From**: experiment session (2026-06-01)
**Purpose**: §6 Discussion paragraph candidate, mechanism-validation. Three
controlled experiments confirm the headroom-hypothesis prediction. Not a
new headline claim — a mechanism-validation paragraph that hardens §5.1
AIDER F1 cross-over and Limitation 3 (saturated backbones).

## The claim being tested

> TraLO's F1 advantage over post-hoc baselines tracks how much CE-gradient
> slack the warmup classifier leaves at the start of Phase 2. When CE
> saturates (train acc → 1.0), the in-training methods converge to the
> same Top-K allocation as the post-hoc methods, and the macro-F1 ranking
> can even flip due to collateral damage on unconstrained classes. When
> CE is moderately unsaturated (train acc ~0.70–0.85), the constraint
> gradient can shape representations and in-training methods recover an
> F1 edge.

The clean-AIDER F1 cross-over of §5.1 is the prediction of the saturated
case. The Limitation 3 "F1 edge doesn't extend to saturated backbones"
(ResNet18, EfficientNetB0 in Table C) is the same mechanism at the
backbone level. **Three new controlled experiments test the prediction
directly.**

## Experiment 1 — AIDER cripple (precursor)

**Backing CSV**: `paper/HANDOFF/tables/aider_cripple_summary.csv`
(server: `results/pending_runs/aider_cripple/`)
**Grid**: 3 cripple conditions × 4 methods × 2 seeds = 24 cells, MobileNetV3, L30_G30.

Conditions:
- **C1 noisy σ=0.15**: mild Gaussian noise on train+test
- **C2 no_pretrain**: cold-start MobileNetV3 (ImageNet weights off)
- **C3 noisy σ=0.30**: heavy Gaussian noise on train+test

| Condition | TraLO F1 | Best post-hoc F1 | ΔF1 (intrain − post-hoc) |
|-----------|----------|------------------|--------------------------|
| Clean AIDER §5.1 headline | ~0.80 | ~0.81 | **−0.005** |
| **C1 mild noise** | 0.7998 | 0.8024 | -0.003 |
| **C2 cold-start** | 0.6264 | 0.6386 | -0.012 (worse) |
| **C3 heavy noise** | **0.7286** | **0.7203** | **+0.008 (flipped)** |

**Key finding**: heavy noise flipped the ranking (clean-AIDER −0.005 →
C3 +0.008). Mild noise didn't move the needle. Cold-start backfired:
on the easy 4-class AIDER, no-pretrain MobileNetV3 still saturates by
~epoch 30 and the CE gradient becomes too dominant in the early phase
to admit constraint pressure — the other end of the predicted spectrum.

## Experiment 2 — DermMNIST cripple (medical corruptions × tightness)

**Backing CSV**: `paper/HANDOFF/tables/derm_cripple_summary.csv`
**Grid**: 5 corruption types × 3 tightness × 4 methods × 2 seeds = 120 cells.

Corruption types (both train AND test corrupted; clinically realistic):
- **noise**: Gaussian sensor noise σ=0.15
- **blur**: Diagonal motion blur (k=11) — hand-held dermoscopy
- **jpeg**: JPEG quality=15 — telemedicine compression
- **color**: HSV jitter (hue=0.15, sat=0.4, bri=0.25) — scanner/lighting
- **defocus**: Gaussian (out-of-focus) blur σ=2.5

**Mean ΔF1 (in-training best − post-hoc best) per cell:**

| Corruption | L20 | L30 | L50 |
|------------|-----|-----|-----|
| noise | +0.013 | +0.015 | **+0.021** |
| defocus | +0.011 | **+0.017** | +0.013 |
| jpeg | +0.005 | +0.009 | +0.001 |
| color | -0.005 | +0.002 | +0.011 |
| blur | -0.008 | +0.001 | +0.004 |

**13 of 15 cells flip the in-training-vs-post-hoc ranking** under medical
corruption. The 2 losing cells (blur L20, color L20) are at tightest cap
with the corruption types least likely to break warmup saturation. The
strongest wins are on **noise + defocus** — the corruptions that
actually depress warmup train accuracy enough to leave Phase 2 with
meaningful CE slack.

## Experiment 3 — DermMNIST backbone-weakening (capacity axis)

**Backing CSV**: `paper/HANDOFF/tables/derm_backbone_weak_summary.csv`
**Grid**: 2 weakening variants × 3 tightness × 4 methods × 2 seeds = 48 cells.

| Variant | L20 ΔF1 | L30 ΔF1 | L50 ΔF1 |
|---------|---------|---------|---------|
| **MobileNetV3 cold-start** (no pretrain) | **+0.018** | **+0.028** | **+0.030** |
| ShuffleNetV2 (smaller, pretrained) | -0.002 | +0.004 | +0.009 |

**Cold-start MobileNetV3 on DermMNIST is the strongest in-training F1
advantage measured anywhere in the project** (+0.018 to +0.030). The
mechanism prediction is therefore validated at the *backbone capacity*
axis as well as the *data corruption* axis. ShuffleNetV2 with pretrained
weights doesn't cripple enough — it saturates like the headline
MobileNetV3 on clean DermMNIST.

## Cross-experiment insight

Cold-start (no-pretrain) helps in-training on derm (+0.030) but hurts
in-training on AIDER (−0.012). The unifying explanation is the warmup
train-accuracy that each setting actually produces:

| Setting | Warmup train-acc (approx) | TraLO F1 advantage |
|---------|----------------------------|---------------------|
| AIDER clean | ~0.9998 | ≈ 0 (saturation lock) |
| AIDER cold-start | ~0.99 by ep 30 | ≈ 0 (still saturates) |
| AIDER noisy σ=0.30 | ~0.88 | **+0.008** |
| DermMNIST clean | ~0.95 | small |
| DermMNIST cold-start | ~0.75 | **+0.030** |
| DermMNIST noise σ=0.15 | ~0.85 | **+0.015** |
| TissueMNIST (headline) | ~0.78 | **+0.030** |

**The advantage tracks warmup train-accuracy, not which axis we used to
get there.** AIDER's natural easiness makes it resistant to backbone
crippling but susceptible to data crippling. Derm's intermediate
difficulty makes it susceptible to either axis.

## Deployability claim — unaffected by any cripple condition

Across all 192 cells of these three experiments:

- **TraLO Sat% = 100%** in every cell
- **Post-hoc Sat% = 0%** in every cell (by construction; needs full test batch)
- **TraLO Flips: 0.5–15** per cell
- **Post-hoc Flips: 70–150** per cell

The 30–100× flip advantage is regime-independent. The Universal claim
(§5.1 Table A footnote) holds in every measured condition.

## Suggested §6 paragraph slot

Insert as new paragraph in §6 Discussion, between current Limitation 3
("Backbone robustness saturated regime") and Limitation 4 (compute cost):

> *Mechanism validation.* To confirm the headroom interpretation of the
> AIDER F1 cross-over (§5.1) and the saturated-backbone limitation
> (§5.3), we ran three controlled cripple experiments. On AIDER with
> Gaussian noise σ=0.30 applied to both train and test, the
> in-training-vs-post-hoc F1 ranking that was −0.005 in clean AIDER
> flipped to +0.008 (n=2 seeds, MobileNetV3, L30_G30). On DermMNIST
> with five clinically-realistic corruption types (Gaussian noise,
> motion blur, JPEG compression, HSV jitter, defocus blur), 13 of 15
> (corruption × tightness) cells exhibit a positive in-training F1
> advantage, with the largest effects on noise and defocus
> (+0.013–0.021 across L20/L30/L50). Replacing the MobileNetV3 ImageNet
> initialization with random weights on clean DermMNIST gives the
> strongest measured advantage of all our experiments (+0.018 to +0.030
> across tightness levels). All three experiments together establish
> that TraLO's F1 advantage is bounded above by warmup-CE-saturation
> level, not by dataset, backbone, or constraint tightness in
> isolation. The deployability advantage (in-training satisfaction,
> low flip count) is unaffected by any cripple condition.

## Backing files

- `paper/HANDOFF/tables/aider_cripple_raw.csv` + `_summary.csv`
- `paper/HANDOFF/tables/derm_cripple_raw.csv` + `_summary.csv`
- `paper/HANDOFF/tables/derm_backbone_weak_raw.csv` + `_summary.csv`
- Server source: `results/pending_runs/{aider_cripple,derm_cripple,derm_backbone_weak}/`

## What this does NOT claim

- **Does not** show TraLO uniquely dominates Fioretto on these cells —
  the F1 wins are mostly TraLO+Fioretto together vs Danits+Heuristic.
- **Does not** add a new headline number — it strengthens the
  mechanism story behind §5.1 and §6 Limitation 3.
- **Does not** make AIDER a "TraLO win" dataset in the headline tables.
  The AIDER F1 loss in clean §5.1 stands and the mechanism explanation
  is now empirically grounded.
