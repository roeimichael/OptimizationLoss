# AAAI-27 Reproducibility Checklist — self-audit

Fill this into the AAAI submission form. For each item: the answer to select, and where
the paper supports it. Verified against `main.tex` / `supplementary.tex` on 2026-07-09.

Legend: ✅ = answer "yes" (supported), ⚠️ = "partial", NA = not applicable.

## 1. This paper (all submissions)

| Item | Answer | Evidence |
|---|---|---|
| Conceptual outline / pseudocode of methods introduced | ✅ yes | Algorithm box, §3 (lines ~326–335); L_total in §3 |
| Delineates opinion/hypothesis vs. objective results | ✅ yes | Claims scoped ("in our grid", "preliminary"); ties reported as ties (§5.2) |
| Well-marked pedagogical references for background | ✅ yes | Related Work §2 (dual ascent, resilient learning, transductive inference) |

## 2. Theoretical contributions?  → YES (added this round)

| Item | Answer | Evidence |
|---|---|---|
| Assumptions/restrictions stated formally | ✅ yes | Penalty defined on excess E, cap K, §3 "bounded transductive penalty" |
| Novel claims stated formally | ✅ yes | Gradient bound Eq (2): ∂ℓ/∂S ≤ (1+ρ)/K |
| Proofs of novel claims included | ✅ yes | Bound derived inline in §3 from the closed-form ℓ; two-phase (frozen-objective) argument in "Multiplier ratchet and freeze" |
| Proof sketches/intuitions for complex results | ✅ yes | "strongest at the boundary, decays to 0"; contrast vs. linear/quadratic penalty |
| Citations for theoretical tools | ✅ yes | Dual ascent / augmented-Lagrangian refs in §2 |
| Theoretical claims demonstrated empirically | ✅ yes | Convergence §5.4 + one-epoch-probe (Supp §A) show the bounded multiplier stays small (λ≤0.18 vs 53) |
| Code to reproduce theory-supporting experiments | ✅ yes | Dynamics logs + probe in Code & Data Supplement |

## 3. Relies on datasets?  → YES

| Item | Answer | Evidence |
|---|---|---|
| Motivation for the datasets chosen | ✅ yes | §5 Datasets: "naturally rare class that makes a count cap meaningful" |
| Novel datasets in a data appendix | NA | no novel dataset |
| Novel datasets to be released w/ license | NA | — |
| Existing datasets cited | ✅ yes | MedMNIST v2 (yang2023medmnist), HAM10000 (tschandl2018ham10000) |
| Existing datasets publicly available | ✅ yes | MedMNIST v2 is public |
| Non-public datasets described + justified | NA | all public |
| Train/val/test splits specified | ✅ yes | "official splits" now stated in §5 Datasets (added this round) |

## 4. Computational experiments?  → YES

| Item | Answer | Evidence |
|---|---|---|
| Pre-processing code in appendix | ✅ yes | data prep + loaders in the Supplement |
| All experiment source code included | ✅ yes | "Code, all run configurations, and the full per-run results corpus accompany the submission" (§5) |
| Source code to be publicly released | ✅ yes | supplement bundled; select yes (release on publication) |
| New-method code commented w/ paper references | ⚠️ partial | code exists; confirm the loss file has comments tying steps to §3 before zipping (see supplement task) |
| Seed-setting method described | ✅ yes | "seeds 1–4 (all reported, no seed selection)" §5 Frozen recipe |
| Computational complexity reported | NA | not a complexity paper; runtime is standard supervised training |
| Explicit math/model/architecture descriptions | ✅ yes | §3 (loss, algorithm), §5 backbones + frozen recipe |
| Hyperparameter search range + selection criterion | ✅ yes | Steps "fixed a priori, not tuned per cell" (§5); 10× step-fairness sweep reported in Supp §A (Block C) |
| All final hyperparameters listed | ✅ yes | §5 Frozen recipe: batch 64, Adam, lr 1e-4/5e-6, warmup 50, constraint 300, per-method steps 0.005/0.01/0.002, β=0.5 |
| Number of runs per reported result | ✅ yes | n=4 seeds per cell, stated throughout (e.g. §5.2, captions) |
| Analysis beyond single-number summaries | ✅ yes | interquartile bands (Fig 4), seed-winrate, per-seed gaps |
| Significance via appropriate statistical test | ✅ yes | paired Wilcoxon signed-rank (§5.1, §5.2); pooled p<1e-6 |
| Computing infrastructure listed | ✅ yes | "PyTorch, torchvision backbones, MedMNIST v2 loaders; single NVIDIA GPU per run" (added this round). Add exact GPU model + PyTorch version in camera-ready (anonymized now). |

## Net result

All items answerable ✅ except:
- **§4 "code commented with paper references" → ⚠️** — verify/refresh comments in the loss +
  training files before you zip the supplement (folded into the supplement task).
- Camera-ready TODO: swap the generic "single NVIDIA GPU" for the exact GPU model and pin
  PyTorch/torchvision/medmnist versions (kept generic now for anonymity).

Fixes applied this round: official-splits sentence (§5 Datasets); infrastructure + software-stack
sentence (§5 Frozen recipe). Both are single clauses; page budget unaffected.
