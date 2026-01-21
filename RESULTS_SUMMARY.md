# Experiment Results Summary

## Overall Performance

**Total Experiments:** 72
**Converged:** 66 (91.7%)
**Failed:** 6 (8.3%)

This is an excellent success rate! The constraint satisfaction training approach works effectively across different models and configurations.

---

## Key Findings

### 1. Best Overall Configuration
- **Fastest convergence:** FTTransformer + constraint_0.9_0.8 + combined strategy + lr=0.0001
  - Converged at epoch **77** (fastest across all experiments)

### 2. Most Reliable Constraint Level
- **constraint_0.9_0.8:** 100% convergence rate (24/24), avg 151 epochs
  - All three models achieved 100% success rate with this constraint
  - FTTransformer particularly excelled (avg 92 epochs)

### 3. Most Challenging Constraint Level
- **constraint_0.8_0.2:** 83.3% convergence rate (20/24), avg 629 epochs
  - Takes 4× longer to converge than 0.9_0.8
  - BasicNN struggled most (62.5% success rate)
  - FTTransformer and TabularResNet performed better (87.5% and 100%)

---

## Performance by Model

| Model          | Converged | Avg Epoch |
|----------------|-----------|-----------|
| FTTransformer  | 23/24     | 308.3     |
| TabularResNet  | 23/24     | 353.1     |
| BasicNN        | 20/24     | 432.9     |

**Winner:** FTTransformer - fastest convergence on average

---

## Performance by Lambda Strategy

| Strategy  | Converged | Avg Epoch |
|-----------|-----------|-----------|
| combined  | 17/18     | 322.4     |
| balanced  | 15/18     | 343.6     |
| linear    | 17/18     | 389.2     |
| transfer  | 17/18     | 389.3     |

**Winner:** Combined strategy - fastest and reliable convergence

---

## Performance by Learning Rate

| Learning Rate | Converged | Avg Epoch |
|---------------|-----------|-----------|
| 0.0001        | 34/36     | 329.7     |
| 0.00005       | 32/36     | 395.6     |

**Winner:** lr=0.0001 - 20% faster convergence than lr=0.00005

---

## Model + Constraint Combinations

| Model         | Constraint    | Success Rate | Avg Epoch |
|---------------|---------------|--------------|-----------|
| BasicNN       | 0.5_0.3       | 87.5% (7/8)  | 473.4     |
| BasicNN       | 0.8_0.2       | 62.5% (5/8)  | 740.2     |
| BasicNN       | 0.9_0.8       | 100% (8/8)   | 205.4     |
| FTTransformer | 0.5_0.3       | 100% (8/8)   | 238.2     |
| FTTransformer | 0.8_0.2       | 87.5% (7/8)  | 635.1     |
| FTTransformer | 0.9_0.8       | 100% (8/8)   | **92.2**  |
| TabularResNet | 0.5_0.3       | 87.5% (7/8)  | 350.9     |
| TabularResNet | 0.8_0.2       | 100% (8/8)   | 553.4     |
| TabularResNet | 0.9_0.8       | 100% (8/8)   | 154.8     |

**Best combo:** FTTransformer + constraint_0.9_0.8 (avg 92.2 epochs)

---

## Failed Experiments Analysis (6 total)

All 6 failures reached max epochs (1000) without satisfying both constraints:

1. **BasicNN/0.5_0.3/lr_5e-05/transfer**
   - Global: ✓ satisfied, Local: ✗ not satisfied
   - Local loss stuck at 0.888692

2. **BasicNN/0.8_0.2/lr_5e-05/balanced**
   - Global: ✗ not satisfied, Local: ✗ not satisfied
   - Global loss: 0.609820, Local loss: 0.004383

3. **BasicNN/0.8_0.2/lr_5e-05/combined**
   - Global: ✗ not satisfied, Local: ✓ satisfied
   - Global loss stuck at 0.566048

4. **BasicNN/0.8_0.2/lr_5e-05/linear**
   - Global: ✗ not satisfied, Local: ✓ satisfied
   - Global loss stuck at 0.520213

5. **FTTransformer/0.8_0.2/lr_0.0001/balanced**
   - Global: ✗ not satisfied, Local: ✓ satisfied
   - Global loss: 0.224546

6. **TabularResNet/0.5_0.3/lr_0.0001/balanced**
   - Global: ✓ satisfied, Local: ✗ not satisfied
   - Local loss extremely high: 0.990817

**Pattern:**
- 5/6 failures involve constraint_0.8_0.2 or balanced strategy
- BasicNN has most trouble with constraint_0.8_0.2 (3/4 failures with lr=5e-05)
- Balanced strategy has issues with 3 different configurations

---

## Recommendations

### For Best Performance:
1. **Use FTTransformer** - most reliable and fastest convergence
2. **Use combined lambda strategy** - best overall performance
3. **Use lr=0.0001** - faster than 0.00005
4. **constraint_0.9_0.8 is easiest** - 100% success, fastest convergence

### For constraint_0.8_0.2 (the difficult one):
- Avoid BasicNN with lr=5e-05 (high failure rate)
- Prefer TabularResNet (100% success) or FTTransformer (87.5%)
- Avoid balanced strategy - use combined or linear instead

### If using BasicNN:
- Prefer higher learning rate (0.0001)
- Avoid balanced strategy with difficult constraints
- Works best with constraint_0.9_0.8

---

## Convergence Speed Statistics

- **Fastest convergence:** 77 epochs (FTTransformer/0.9_0.8/combined/0.0001)
- **Slowest convergence:** 985 epochs (BasicNN/0.8_0.2/transfer/5e-05)
- **Average convergence:** 362 epochs
- **Median convergence:** ~328 epochs

---

## Conclusion

The constraint satisfaction training approach is **highly successful** with a 91.7% convergence rate. The results show clear patterns:

✓ FTTransformer is the most powerful model
✓ Combined lambda strategy is most effective
✓ Higher learning rate (0.0001) is better
✓ Constraint difficulty: 0.9_0.8 < 0.5_0.3 < 0.8_0.2

The 6 failed experiments all involve either very tight constraints (0.8_0.2) or suboptimal hyperparameter combinations (BasicNN + low LR + balanced strategy). These can potentially be resolved by:
- Increasing max epochs for difficult constraints
- Using better models (FTTransformer)
- Optimizing lambda strategies for specific constraint patterns
