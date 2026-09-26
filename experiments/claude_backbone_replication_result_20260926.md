# Targeted-step replication on MobileNetV3 and RegNetY: result -- 2026-09-26

- **Protocol:** `claude_backbone_replication_prereg_20260926.md`, with amendments 1 and 2.
- **Run:** release 91bc190c, cap 76, dsisco01. Blocks `mn3` (3001-3024) and `rgy` (3101-3124). The
  pilots are excluded.
- **Scorers:**
  - `analysis/score_sham.py` (rebuild branch), primary;
  - `analysis/ens_contrasts.py`, ensembled secondaries;
  - `analysis/score_snapshot_ensemble.py`, ensemble confirmation sets 3 and 4;
  - `analysis/eviction_precision.py` and `analysis/washout.py`, mechanism.
- **Raw output:** `analysis/repl_{mn3,rgy}.txt`, `ens_c_*.txt`, `ens_conf_*.txt`,
  `eviction_repl.txt`, `washout_repl.txt`.

**Integrity: 48/48 seeds pass.** Hashes matched, 1810 updates, and every target step lands on the
cap. The sham triggers, and the first radii are equal.

## Primary (capped_first grade-3 F1, Holm over C1, C2 within each block)

| block | C1 target - null | C2 target - sham | reading |
|---|---|---|---|
| MobileNetV3 | **+1.51 [+0.40, +2.62]**, Holm 0.016, 15/3/6 | **+1.47 [+0.43, +2.50]**, Holm 0.016, 16/2/6 | **2** |
| RegNetY | +0.18 [-1.17, +1.54], Holm 1.0 | -0.09 [-1.43, +1.25], Holm 1.0 | 1 |

- **Multiplicity across blocks:** Bonferroni over the two blocks gives family p = 0.031 for MobileNetV3.
- **Counting ResNet18 as a third backbone** (LEDGER #9), it gives 0.047.
- **Reading 2 holds for MobileNetV3 as preregistered.** At the right size, the constraint direction
  improves which patients fill the slots there, beyond a random move of the same size.

## Ensembled secondaries (amendment 2 rule: mean of epoch 06-10 snapshots)

| block | C1-ENS | C2-ENS | target - clipper, ENS |
|---|---|---|---|
| MobileNetV3 | +0.67 [+0.01, +1.32] slots, Holm 0.046 | **+1.04 [+0.22, +1.86] slots, Holm 0.030** | +0.29 [-0.45, +1.03], ns |
| RegNetY | +0.21, ns | -0.17, ns | -0.25, ns |

- **The attribution survives ensembling:** C2-ENS is Holm-significant.
- **The thesis bar is not cleared:** at equal ensembling, tralo_target ties the clipper on both
  backbones.

## Snapshot ensemble, confirmation sets 3 and 4: CONFIRMED again

ENS - BASE, correct slots:

| arm | MobileNetV3 | RegNetY |
|---|---|---|
| clipper | +2.04 [+0.54, +3.54], Holm 0.017 | +1.79 [+1.04, +2.54], Holm 0.0001 |
| tralo_null | +1.71 [+0.48, +2.94], Holm 0.017 | +2.08 [+0.89, +3.28], Holm 0.0015 |

That makes four fresh confirmation sets, three backbones and both caps.

## Mechanism (offline, development labels): why only MobileNetV3 reaches the endpoint

| backbone | correct slots per applied step | eviction overlap with the post-hoc cut |
|---|---|---|
| ResNet18 | +0.12 | 83.0% |
| MobileNetV3 | +0.51 | 76.3% |
| RegNetY | +0.48 | 76.0% |

The steps on both small backbones carry about 4x ResNet18's per-step who-information, and they deviate
further from the post-hoc cut.

**Wash-out still holds on MobileNetV3:**
- Epoch 7 after the step is +0.54, and the epoch 8 start is -0.46 (`washout_repl.txt`).
- The endpoint is +1.38 slots. Of that, +0.88 is the FINAL step's own gain, which no later CE epoch
  erases, and +0.50 is carried in.
- RegNetY's final step gained +0.38.
- The mean own-step gain over epochs 6-10 is similar on the two backbones: +0.46 vs +0.42.

**So the endpoint difference between the two small backbones is mostly the size of the last step.**
It is not a different mechanism.

## Conclusion

1. **ResNet18 is not the whole story.** On MobileNetV3 the dosed, targeted count step has a small
   attributable positive effect: about 1.3 correct slots of 76 vs a same-size random move. It survives
   ensembling and Holm, and it is the first attributable positive in this project.
2. **It is the channel LEDGER #10 identified:** who-information routed through the shared
   representation, about 0.5 slots per step on the smaller backbones. The next CE epoch erases it, so
   the endpoint keeps roughly one step's worth.
3. **It does not beat the post-hoc bar.** At equal ensembling, tralo_target - clipper is +0.29 (ns) on
   MobileNetV3 and -0.25 on RegNetY. Ensembling the clipper alone gains +1.8 to +2.0 slots.
4. **Design implication, untested:** a schedule that applies the step AFTER the last CE epoch (step-last)
   would keep one step's worth, about +0.5 slots, by construction. That is below the ensemble's gain.
