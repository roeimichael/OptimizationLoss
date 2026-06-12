# Per-sweep manifest


## contamination_clean
- **Purpose**: Clean sigma=0 anchor for contamination grid (3 ds x 4 tight x 5 methods x 2 seeds)
- **Cells**: 120  (missing metrics: 0)
- **Datasets**: `aider`=40, `dermmnist`=40, `tissuemnist`=40
- **Models**: `MobileNetV3`=120
- **Methods**: `hounie_rcl`=24, `tralo`=24, `danits_lp`=24, `fioretto_ldf`=24, `heuristic`=24
- **Tightness**: `L20_G20`=30, `L70_G70`=30, `L30_G30`=30, `L50_G50`=30
- **Seeds**: [1, 2]

## contamination_tissuemnist
- **Purpose**: TissueMNIST contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)
- **Cells**: 120  (missing metrics: 0)
- **Datasets**: `tissuemnist`=120
- **Models**: `MobileNetV3`=120
- **Methods**: `hounie_rcl`=24, `tralo`=24, `danits_lp`=24, `fioretto_ldf`=24, `heuristic`=24
- **Tightness**: `L20_G20`=30, `L70_G70`=30, `L30_G30`=30, `L50_G50`=30
- **Seeds**: [1, 2]

## contamination_dermmnist
- **Purpose**: DermMNIST contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)
- **Cells**: 120  (missing metrics: 0)
- **Datasets**: `dermmnist`=120
- **Models**: `MobileNetV3`=120
- **Methods**: `hounie_rcl`=24, `tralo`=24, `danits_lp`=24, `fioretto_ldf`=24, `heuristic`=24
- **Tightness**: `L20_G20`=30, `L70_G70`=30, `L30_G30`=30, `L50_G50`=30
- **Seeds**: [1, 2]

## contamination_aider
- **Purpose**: AIDER contamination grid (sigma in {0.10,0.20,0.30} x 4 tight x 5 methods x 2 seeds)
- **Cells**: 120  (missing metrics: 0)
- **Datasets**: `aider`=120
- **Models**: `MobileNetV3`=120
- **Methods**: `hounie_rcl`=24, `tralo`=24, `danits_lp`=24, `fioretto_ldf`=24, `heuristic`=24
- **Tightness**: `L20_G20`=30, `L70_G70`=30, `L30_G30`=30, `L50_G50`=30
- **Seeds**: [1, 2]

## paper_backbones
- **Purpose**: Headline 3-backbone x 3-dataset x 5-tight x 6-method x 4-seed sweep (G1+G5)
- **Cells**: 753  (missing metrics: 0)
- **Datasets**: `tissuemnist`=257, `aider`=248, `dermmnist`=248
- **Models**: `ShuffleNetV2`=273, `MobileNetV2`=240, `RegNetY400MF`=240
- **Methods**: `danits_lp`=156, `heuristic`=156, `hounie_rcl`=147, `tralo`=147, `fioretto_ldf`=147
- **Tightness**: `L20_G20`=180, `L70_G70`=180, `L30_G30`=180, `L80_G80`=180, `L50_G50`=33
- **Seeds**: [1, 2, 3, 4]

## paper400_baselines
- **Purpose**: 400-config paper baseline grid: TraLO + 4 baselines x 3 ds (Turing era)
- **Cells**: 24  (missing metrics: 0)
- **Datasets**: `tissuemnist`=24
- **Models**: `MobileNetV3`=24
- **Methods**: `hounie_rcl`=12, `fioretto_ldf`=12
- **Tightness**: `L20_G20`=8, `L30_G30`=8, `L50_G50`=8
- **Seeds**: [1, 2, 3, 4]

## paper400_tralofix
- **Purpose**: TraLO-fix rerun of paper400 set (undershoot_hinge + reset_optimizer_at_sat)
- **Cells**: 12  (missing metrics: 0)
- **Datasets**: `tissuemnist`=12
- **Models**: `MobileNetV3`=12
- **Methods**: `tralo`=12
- **Tightness**: `L20_G20`=4, `L30_G30`=4, `L50_G50`=4
- **Seeds**: [1, 2, 3, 4]

## g2_asym_tissue_aider
- **Purpose**: G2: asymmetric (L != G) tightness on tissue+aider, 4 methods
- **Cells**: 672  (missing metrics: 0)
- **Datasets**: `tissuemnist`=376, `aider`=296
- **Models**: `MobileNetV3`=672
- **Methods**: `hounie_rcl`=144, `tralo`=144, `fioretto_ldf`=144, `danits_lp`=120, `heuristic`=120
- **Tightness**: `L30_G50`=40, `L30_G20`=40, `L70_G50`=40, `L70_G80`=40, `L50_G80`=40, `L80_G70`=40, `L80_G50`=40, `L50_G70`=40
- **Seeds**: [1, 2, 3, 4]

## g3_multiclass_tissue
- **Purpose**: G3: multiclass cap (multiple constrained classes) on tissuemnist
- **Cells**: 300  (missing metrics: 0)
- **Datasets**: `tissuemnist`=300
- **Models**: `MobileNetV3`=300
- **Methods**: `hounie_rcl`=60, `tralo`=60, `danits_lp`=60, `fioretto_ldf`=60, `heuristic`=60
- **Tightness**: `L20_G20`=60, `L70_G70`=60, `L30_G30`=60, `L50_G50`=60, `L80_G80`=60
- **Seeds**: [1, 2, 3, 4]

## g4_table_b_backfill
- **Purpose**: G4: backfill cells missing from Table B in paper v1
- **Cells**: 12  (missing metrics: 0)
- **Datasets**: `dermmnist`=12
- **Models**: `MobileNetV3`=12
- **Methods**: `danits_lp`=6, `heuristic`=6
- **Tightness**: `L50_G20`=6, `L20_G50`=6
- **Seeds**: [2, 3, 4]

## aider_asym
- **Purpose**: AIDER asymmetric tightness, extension seeds
- **Cells**: 80  (missing metrics: 0)
- **Datasets**: `aider`=80
- **Models**: `MobileNetV3`=80
- **Methods**: `hounie_rcl`=16, `tralo`=16, `danits_lp`=16, `fioretto_ldf`=16, `heuristic`=16
- **Tightness**: `L20_G80`=20, `L80_G20`=20, `L30_G70`=20, `L70_G30`=20
- **Seeds**: [1, 2, 3, 4]

## g5_component_ablation
- **Purpose**: G5: ablate undershoot_hinge / reset_optimizer_at_sat / lambda_toggle (the TraLO-fix components)
- **Cells**: 63  (missing metrics: 0)
- **Datasets**: `aider`=21, `dermmnist`=21, `tissuemnist`=21
- **Models**: `MobileNetV3`=63
- **Methods**: `tralo`=63
- **Tightness**: `L30_G30`=63
- **Seeds**: [1, 2, 3]

## component_ablation
- **Purpose**: Component ablation: lambda_toggle, KL, rho schedule, optimizer reset
- **Cells**: 14  (missing metrics: 0)
- **Datasets**: `tissuemnist`=14
- **Models**: `MobileNetV3`=14
- **Methods**: `tralo`=14
- **Tightness**: `L30_G30`=14
- **Seeds**: [1, 2]

## kl_ablation
- **Purpose**: KL drift damper ablation (alpha_kl in {0, 0.05, 0.1, ...})
- **Cells**: 8  (missing metrics: 0)
- **Datasets**: `tissuemnist`=8
- **Models**: `MobileNetV3`=8
- **Methods**: `tralo`=8
- **Tightness**: `L30_G30`=8
- **Seeds**: [1, 2]

## blackwell_validation
- **Purpose**: Blackwell 8-seed paired validation of Turing winners (MobileNetV2/V3 x derm/aider)
- **Cells**: 144  (missing metrics: 0)
- **Datasets**: `aider`=72, `dermmnist`=72
- **Models**: `MobileNetV3`=48, `MobileNetV2`=48, `ShuffleNetV2`=24, `RegNetY400MF`=24
- **Methods**: `hounie_rcl`=48, `tralo`=48, `fioretto_ldf`=48
- **Tightness**: `L50_G50`=144
- **Seeds**: [1, 2, 3, 4, 5, 6, 7, 8]

## blackwell_new_backbones
- **Purpose**: Blackwell extension: RegNetY16GF + DenseNet121 corroboration
- **Status**: MISSING or empty


## arch_validation
- **Purpose**: Turing vs Blackwell cell ranking comparison (architecture independence check)
- **Cells**: 12  (missing metrics: 0)
- **Datasets**: `tissuemnist`=12
- **Models**: `MobileNetV3`=12
- **Methods**: `hounie_rcl`=4, `tralo`=4, `fioretto_ldf`=4
- **Tightness**: `L30_G30`=6, `L50_G50`=6
- **Seeds**: [1, 2]

## derm_cripple
- **Purpose**: Crippled derm warmup (low train acc) -- headroom hypothesis test
- **Cells**: 120  (missing metrics: 0)
- **Datasets**: `dermmnist`=120
- **Models**: `MobileNetV3`=120
- **Methods**: `tralo`=30, `danits_lp`=30, `fioretto_ldf`=30, `heuristic`=30
- **Tightness**: `L20_G20`=40, `L30_G30`=40, `L50_G50`=40
- **Seeds**: [1, 2]

## aider_cripple
- **Purpose**: Crippled aider warmup -- saturated-warmup regime probe
- **Cells**: 24  (missing metrics: 0)
- **Datasets**: `aider`=24
- **Models**: `MobileNetV3`=24
- **Methods**: `tralo`=6, `danits_lp`=6, `fioretto_ldf`=6, `heuristic`=6
- **Tightness**: `L30_G30`=24
- **Seeds**: [1, 2]

## derm_backbone_weak
- **Purpose**: Weak-backbone search for clearer TraLO win on derm
- **Cells**: 48  (missing metrics: 0)
- **Datasets**: `dermmnist`=48
- **Models**: `ShuffleNetV2`=24, `MobileNetV3`=24
- **Methods**: `tralo`=12, `danits_lp`=12, `fioretto_ldf`=12, `heuristic`=12
- **Tightness**: `L20_G20`=16, `L30_G30`=16, `L50_G50`=16
- **Seeds**: [1, 2]

## warmup_confirm
- **Purpose**: Confirm warmup sweet-spot for TraLO F1 edge
- **Cells**: 12  (missing metrics: 0)
- **Datasets**: `dermmnist`=12
- **Models**: `MobileNetV3`=12
- **Methods**: `hounie_rcl`=4, `tralo`=4, `fioretto_ldf`=4
- **Tightness**: `L50_G50`=12
- **Seeds**: [3, 4]

## warmup_probe
- **Purpose**: Initial warmup-quality probe (train_acc range)
- **Cells**: 24  (missing metrics: 0)
- **Datasets**: `dermmnist`=24
- **Models**: `MobileNetV3`=24
- **Methods**: `hounie_rcl`=8, `tralo`=8, `fioretto_ldf`=8
- **Tightness**: `L50_G50`=24
- **Seeds**: [1, 2]

## lr_hp_smoke
- **Purpose**: LR sweep + HP variants (warmup, rho, lambda_step, alpha_kl) on derm sigma=0.20
- **Cells**: 102  (missing metrics: 0)
- **Datasets**: `dermmnist`=42, `aider`=30, `tissuemnist`=30
- **Models**: `MobileNetV3`=102
- **Methods**: `tralo`=30, `hounie_rcl`=18, `danits_lp`=18, `fioretto_ldf`=18, `heuristic`=18
- **Tightness**: `L30_G30`=102
- **Seeds**: [1, 2]

## model_search
- **Purpose**: 8-backbone x 5-dataset search for clean TraLO winners
- **Cells**: 33  (missing metrics: 0)
- **Datasets**: `dermmnist`=15, `aider`=12, `tissuemnist`=6
- **Models**: `MobileNetV2`=10, `ShuffleNetV2`=10, `RegNetY400MF`=7, `MobileNetV3`=6
- **Methods**: `tralo`=13, `hounie_rcl`=10, `fioretto_ldf`=10
- **Tightness**: `L50_G50`=33
- **Seeds**: [1]

## expansion_baselines
- **Purpose**: Expansion baselines (heuristic + danits_lp) for full grid coverage
- **Cells**: 54  (missing metrics: 0)
- **Datasets**: `tissuemnist`=54
- **Models**: `MobileNetV3`=54
- **Methods**: `danits_lp`=27, `heuristic`=27
- **Tightness**: `L50_G20`=6, `L80_G30`=6, `L20_G20`=6, `L70_G70`=6, `L30_G30`=6, `L50_G50`=6, `L30_G80`=6, `L20_G50`=6
- **Seeds**: [1, 2, 3]

## expansion_aider_baselines
- **Purpose**: AIDER baseline expansion
- **Cells**: 54  (missing metrics: 0)
- **Datasets**: `aider`=54
- **Models**: `MobileNetV3`=54
- **Methods**: `danits_lp`=27, `heuristic`=27
- **Tightness**: `L50_G20`=6, `L80_G30`=6, `L20_G20`=6, `L70_G70`=6, `L30_G30`=6, `L50_G50`=6, `L30_G80`=6, `L20_G50`=6
- **Seeds**: [1, 2, 3]

## expansion_dermmnist_baselines
- **Purpose**: DermMNIST baseline expansion
- **Cells**: 54  (missing metrics: 0)
- **Datasets**: `dermmnist`=54
- **Models**: `MobileNetV3`=54
- **Methods**: `danits_lp`=27, `heuristic`=27
- **Tightness**: `L50_G20`=6, `L80_G30`=6, `L20_G20`=6, `L70_G70`=6, `L30_G30`=6, `L50_G50`=6, `L30_G80`=6, `L20_G50`=6
- **Seeds**: [1, 2, 3]

## paperv2_phase1
- **Purpose**: Paper v2 phase 1: TraLO vs trained baselines, sym tightness
- **Cells**: 173  (missing metrics: 0)
- **Datasets**: `dermmnist`=70, `aider`=69, `tissuemnist`=34
- **Models**: `MobileNetV3`=173
- **Methods**: `hounie_rcl`=48, `fioretto_ldf`=48, `tralo`=47, `danits_lp`=15, `heuristic`=15
- **Tightness**: `L70_G70`=42, `L80_G80`=42, `L20_G20`=30, `L50_G50`=30, `L30_G30`=29
- **Seeds**: [1, 2, 3, 4]

## paperv2_phase2
- **Purpose**: Paper v2 phase 2: post-hoc baselines
- **Cells**: 376  (missing metrics: 0)
- **Datasets**: `dermmnist`=376
- **Models**: `MobileNetV3`=376
- **Methods**: `hounie_rcl`=80, `tralo`=80, `fioretto_ldf`=80, `danits_lp`=68, `heuristic`=68
- **Tightness**: `L20_G80`=20, `L30_G50`=20, `L30_G20`=20, `L80_G20`=20, `L30_G70`=20, `L70_G50`=20, `L70_G80`=20, `L50_G80`=20
- **Seeds**: [1, 2, 3, 4]

## paperv2_phase3
- **Purpose**: Paper v2 phase 3: asymmetric tightness
- **Status**: MISSING or empty


## paperv2_phase3_v3
- **Purpose**: Paper v2 phase 3 rev3: asymmetric refit
- **Cells**: 69  (missing metrics: 0)
- **Datasets**: `tissuemnist`=37, `aider`=16, `dermmnist`=16
- **Models**: `RegNetY400MF`=36, `MobileNetV2`=33
- **Methods**: `danits_lp`=24, `heuristic`=24, `hounie_rcl`=7, `tralo`=7, `fioretto_ldf`=7
- **Tightness**: `L50_G50`=69
- **Seeds**: [1, 2, 3, 4]

## paperv2_phase4
- **Purpose**: Paper v2 phase 4: multiclass constraint
- **Cells**: 300  (missing metrics: 0)
- **Datasets**: `dermmnist`=300
- **Models**: `MobileNetV3`=300
- **Methods**: `hounie_rcl`=60, `tralo`=60, `danits_lp`=60, `fioretto_ldf`=60, `heuristic`=60
- **Tightness**: `L20_G20`=60, `L70_G70`=60, `L30_G30`=60, `L50_G50`=60, `L80_G80`=60
- **Seeds**: [1, 2, 3, 4]

## paperv2_phase5
- **Purpose**: Paper v2 phase 5: corroboration backbones
- **Cells**: 100  (missing metrics: 0)
- **Datasets**: `dermmnist`=100
- **Models**: `MobileNetV3`=100
- **Methods**: `hounie_rcl`=20, `tralo`=20, `danits_lp`=20, `fioretto_ldf`=20, `heuristic`=20
- **Tightness**: `L20_G20`=20, `L70_G70`=20, `L30_G30`=20, `L50_G50`=20, `L80_G80`=20
- **Seeds**: [1, 2, 3, 4]

## paperv2_phase6
- **Purpose**: Paper v2 phase 6: component ablation
- **Status**: MISSING or empty


## aider_seed_ext
- **Purpose**: AIDER seed extension for paired stats
- **Cells**: 88  (missing metrics: 0)
- **Datasets**: `aider`=88
- **Models**: `MobileNetV3`=88
- **Methods**: `danits_lp`=20, `heuristic`=20, `hounie_rcl`=16, `tralo`=16, `fioretto_ldf`=16
- **Tightness**: `L20_G20`=20, `L70_G70`=20, `L30_G30`=20, `L80_G80`=20, `L50_G50`=8
- **Seeds**: [5, 6, 7, 8]

## class_rotation
- **Purpose**: Alternate constrained-class rotation: tissue/derm/aider (3 alt classes each); confirms universal claim across cap-class choice
- **Cells**: 48  (missing metrics: 6)
- **Datasets**: `aider`=18, `tissuemnist`=18, `dermmnist`=12
- **Models**: `MobileNetV3`=48
- **Methods**: `hounie_rcl`=16, `tralo`=16, `fioretto_ldf`=16
- **Tightness**: `L50_G50`=48
- **Seeds**: [1, 2]

## octmnist_smoke
- **Purpose**: OctMNIST smoke probe (drusen as constrained class, 12 cells)
- **Cells**: 12  (missing metrics: 12)
- **Datasets**: `octmnist`=12
- **Models**: `MobileNetV3`=12
- **Methods**: `hounie_rcl`=4, `tralo`=4, `fioretto_ldf`=4
- **Tightness**: `L30_G30`=6, `L50_G50`=6
- **Seeds**: [1, 2]

## octmnist_expansion
- **Purpose**: OctMNIST 60-cell full panel: 5 methods x 4 seeds x 3 tightness; CLEAN WIN at L30_G30 vs trained baselines
- **Cells**: 60  (missing metrics: 0)
- **Datasets**: `octmnist`=60
- **Models**: `MobileNetV3`=60
- **Methods**: `hounie_rcl`=12, `tralo`=12, `danits_lp`=12, `fioretto_ldf`=12, `heuristic`=12
- **Tightness**: `L30_G50`=20, `L30_G30`=20, `L50_G50`=20
- **Seeds**: [1, 2, 3, 4]

## cifar100_smalltrain
- **Purpose**: CIFAR-100 train-data-quantity headroom test: subset50/10/5 train samples/class
- **Cells**: 36  (missing metrics: 0)
- **Datasets**: `cifar100`=36
- **Models**: `MobileNetV3`=36
- **Methods**: `hounie_rcl`=12, `tralo`=12, `fioretto_ldf`=12
- **Tightness**: `L30_G30`=18, `L50_G50`=18
- **Seeds**: [1, 2]

## new_dataset_probes
- **Purpose**: Retina/Blood/CIFAR-100 smoke probes (paper extension)
- **Cells**: 24  (missing metrics: 12)
- **Datasets**: `retinamnist`=12, `cifar100`=12
- **Models**: `MobileNetV3`=24
- **Methods**: `hounie_rcl`=8, `tralo`=8, `fioretto_ldf`=8
- **Tightness**: `L30_G30`=12, `L50_G50`=12
- **Seeds**: [1, 2]