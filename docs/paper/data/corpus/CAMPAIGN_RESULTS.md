# Extra-Robustness Campaign — Results (analyzed + adversarially verified 2026-06-29)

480/480 runs complete. Side-note campaign, **isolated from `paper_final`** (separate root + `extra_robustness`
blocklist tag). Census: `scripts/census_extra_robustness.py` → `extra_experiments/extra_robustness_corpus.csv`.
Every verdict below was independently reproduced by an adversarial verifier (no over-averaging / sign error /
cherry-pick).

## Block A — Derm asymmetric (G<L) cross-backbone → NEGATIVE
Does NOT promote derm-asym to a backbone-general win.
- RegNetY400MF corroborates MNV3: cc-F1 **+0.0090** pooled (wr 0.75); MNV3 ref **+0.0117** (wr 0.81).
- **ViTB16 is a flat TIE: −0.0005 (wr 0.38).** → 2 of 3 backbones, small ~+0.01 effect.
- Does **not** improve macro-F1 (pooled ~0 / slightly negative on all 3 backbones).
- Post-hoc clipping is competitive-to-better (RegNet TraLO loses to post-hoc on all 4 cells).
- The edge is a recall-driven operating-point shift, not a uniform improvement.
- **Decision: keep derm-asym labeled "preliminary." Do NOT add as a second hard win.**

## Block B — Component leave-one-out ablation → WIN (add to paper)
Complete 5-component ablation on the OctMNIST hard-win cells (L30_G30 + L40_G40 × 3 backbones, n=24 paired).
Δ cc-F1 = full TraLO − component removed; positive = load-bearing.

| Component | Δ cc-F1 | winrate | p (paired) |
|---|--:|--:|--:|
| optimizer reset @ sat | **+0.079** | 24/24 | 3.1e-07 |
| undershoot hinge      | **+0.036** | 24/24 | 3.0e-06 |
| rho schedule          | +0.001 | 10/24 | 0.74 (neutral) |
| freeze @ sat          | +0.000 |  9/24 | 0.65 (neutral) |
| +KL (add-back)        | −0.010 |  5/24 | 0.04 (small wash) |

- **Two load-bearing knobs: optimizer-reset (dominant) + the undershoot hinge.** Clean — both arms satisfy
  fully (satisfied=1.0, no NaN), so this is an internal quality effect, not a robustness artifact.
- rho / freeze are cc-F1-neutral on these easy-to-satisfy cells; +KL is a small dataset-flipping wash
  (helps oct, hurts derm) → consistent with KL = drift-damper, not a cc-F1 driver.
- Data: `extra_experiments/ablation_complete.csv` (via `scripts/_build_ablation_complete.py`).
- **Honest framing for the paper: a two-knob ablation (reset + hinge), NOT four co-equal components.**

## Block C — Baseline step-fairness → WIN (supplementary reviewer-defense)
The Fioretto/Hounie CE→NaN is **structural, not a mis-tuning artifact**.
- **96/96 runs hit ce_nan** across Fioretto step {0.001, 0.002, 0.01} AND Hounie eta {0.001, 0.05, 0.1},
  AND at the default paper step (0.005 / 0.01). No step yields finite-CE training.
- Smaller step only slows the multiplier climb (Fioretto max_λ 2 → 23), never prevents the NaN.
- Best-recovered baseline cc-F1 still trails TraLO on every cell (−0.010 to −0.030; small, not sig-tested).
- **Use as a one-line defense: pre-empts "you mis-tuned the baseline."**

## ⚠️ Mechanism honesty refinement (affects paper prose, not figures)
`ce_nan` is corpus-wide (Fioretto 80/80, Hounie 80/80; TraLO 0/224) **BUT the baselines RECOVER** from a
best-satisfied checkpoint and stay competitive (cc-F1 0.23–0.62, satisfied=1.0, flips=0, macro within ~0.005;
frac cc-F1<0.05 = 0.0). The NaN is a **training-stability event, not end-to-end collapse**. `fig_mechanism`
(showing CE→NaN@ep2) is accurate; the **prose must not imply the NaN disqualifies the baseline**. This is
consistent with the audited "mechanism is qualitative; magnitude (r=−0.22) does not predict the win" stance.

## Net for the paper
- **Add:** the complete component-ablation table (reset + hinge load-bearing) and a one-line Block-C fairness
  defense. Apply the mechanism-prose tweak.
- **Do NOT add:** derm-asym as a second hard win (stays preliminary).
- Three headline pillars are unchanged. n=4 is sufficient for the additions (reset/hinge p≤3e-6).
