# Small global-only TraLO comparison

Question: does a freshly implemented global count constraint improve the same
classifier relative to its matched no-constraint control and ordinary Clipper?
The user requests basic settings without a search. Three seeds (701–703), one
fixed recipe, nine fits. This is an exploratory frozen-feature test, not an
end-to-end backbone experiment or a general verdict on TraLO.

Reuse the verified 10,000 training / 2,000 development feature cache from
`cifar100-global-44f485e1-20260922T132450Z/training`. No feature extraction or
split changes. Every head is freshly initialized; the previous trained head
and its outcomes are not used to initialize/select this experiment.

All arms: frozen ImageNet ResNet-18 features, Adam learning rate 0.001, batch256,
10 supervised cross-entropy epochs; same seed-specific initial parameters and
batch order. Adam is a simple explicit choice for this comparison, not a claim
it is optimal. The previous pilot used SGD, so its numbers are not a matched
baseline for these new runs. FP32 with TF32 off; deterministic operations.

| Arm | After five supervised epochs | Remaining five epochs |
|---|---|---|
| Clipper | Keep Adam state | Supervised updates only |
| TraLO-null | Reset Adam once | Supervised updates only |
| TraLO | Reset Adam once | Supervised epoch, then one active global-penalty update |

All ten supervised epochs are matched. TraLO additionally spends up to five
constraint updates; record their counts and runtime rather than claiming exact
compute equality. No random-generator reset, augmentation, model-selection pass,
early stopping, or evaluation-label feedback. TraLO-null differs from TraLO only
by removing constraint updates and their controller; no dummy optimizer steps.

## Explicit global TraLO definition

For the unlabeled development population, `S_c = sum_i softmax(logits_i)[c]`.
For a capped class, `E_c=max(S_c-K_c,0)`, `s_c=max(K_c,1)`, `e_c=E_c/s_c`.
The global loss is

`L_constraint = sum_c lambda_c [ e_c/(1+e_c) + rho*e_c²/(1+e_c²) ]`.

This is a fresh implementation of the bounded soft-count TraLO objective,
without the legacy epsilon because its denominators are strictly positive.
Uncapped classes have no penalty term. It is not a sample-supervised loss and
does not identify which individual development predictions are wrong.

Initial lambda0.01 per constrained class, increment0.05 when the pre-update
hard argmax count exceeds its cap. Initial rho0.5; increment
`(100-0.5)/5` after an unsatisfied phase epoch. Freeze lambda/rho adaptation
at the first snapshot satisfying all hard caps, as in the reference controller.
The last rho increment need not be consumed by a later update. These retained
reference settings are declared starting values, not proven good choices;
they will not be tuned after seeing this run's scores. A zero loss skips the
constraint Adam step, preventing movement due solely to existing momentum.

Caps remain10 for classes0–9 on the 2,000-image pool; all others uncapped.
TraLO may inspect development **features without labels** and those declared
caps. This is transductive evaluation, not untouched/inductive evaluation.
The training function accepts training labels but no development labels.
The official CIFAR test split remains unevaluated.

Every arm is evaluated raw and with both named allocation diagnostics from the
first pilot. Within each policy, compare paired seeds; never compare one arm's
raw metrics to another's allocated metrics as a method ranking.
Report accuracy, macro-F1, constrained-class F1, raw count violations, final
feasibility, allocation changes, task/constraint dose and runtime. No single
metric is selected post hoc as the verdict; call mixed results mixed.
Report all seeds and mean paired deltas (TraLO-null minus Clipper; TraLO minus
null; TraLO minus Clipper). Three seeds support a screen, not a publication claim.

## Validation and command

Before the run: analytic and finite-difference loss-gradient tests; inactive
gradient tests; controller tests; exact zero-intervention/null parity; identical
warmup weights/batch order across arms; logging neutrality; all prior regressions;
native server tests; source identity and fresh GPU ownership checks.

```text
CUDA_VISIBLE_DEVICES=<free> CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m tralo.global_comparison examples/global_comparison.json <verified-cache>/training examples/cifar100_global_caps.json <exclusive-output>
```

The cache's completion event and all artifact hashes must validate before use.
Preserve any failed release/run. No inherited runtime imports or source edits
on an active release are allowed.
