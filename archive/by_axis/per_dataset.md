# Per-dataset breakdown


## aider  (n=1188)

- **Models** (4): `MobileNetV3`=867, `MobileNetV2`=115, `ShuffleNetV2`=103, `RegNetY400MF`=103
- **Methods**: `tralo`=262, `fioretto_ldf`=242, `hounie_rcl`=236, `danits_lp`=224, `heuristic`=224
- **Tightness** (25 unique): `L30_G30`=214, `L50_G50`=194, `L20_G20`=140, `L70_G70`=140, `L80_G80`=100, `L50_G20`=20, `L80_G30`=20, `L30_G50`=20, `L30_G20`=20, `L70_G50`=20
- **Contamination sigma**: `0.0`=1038, `0.1`=40, `0.2`=70, `0.3`=40
- **Sweeps contributing**: `g2_asym_tissue_aider`=296, `paper_backbones`=248, `contamination_aider`=120, `aider_seed_ext`=88, `aider_asym`=80, `blackwell_validation`=72, `paperv2_phase1`=69, `expansion_aider_baselines`=54, `contamination_clean`=40, `lr_hp_smoke`=30, `aider_cripple`=24, `g5_component_ablation`=21, `class_rotation`=18, `paperv2_phase3_v3`=16, `model_search`=12

## cifar100  (n=48)

- **Models** (1): `MobileNetV3`=48
- **Methods**: `hounie_rcl`=16, `tralo`=16, `fioretto_ldf`=16
- **Tightness** (2 unique): `L30_G30`=24, `L50_G50`=24
- **Contamination sigma**: `0.0`=48
- **Sweeps contributing**: `cifar100_smalltrain`=36, `new_dataset_probes`=12

## dermmnist  (n=1702)

- **Models** (4): `MobileNetV3`=1354, `ShuffleNetV2`=128, `MobileNetV2`=116, `RegNetY400MF`=104
- **Methods**: `tralo`=388, `fioretto_ldf`=352, `danits_lp`=326, `heuristic`=326, `hounie_rcl`=310
- **Tightness** (25 unique): `L50_G50`=355, `L30_G30`=319, `L20_G20`=256, `L70_G70`=200, `L80_G80`=160, `L50_G20`=26, `L20_G50`=26, `L80_G30`=20, `L30_G80`=20, `L20_G80`=20
- **Contamination sigma**: `0.0`=1540, `0.1`=40, `0.2`=82, `0.3`=40
- **Sweeps contributing**: `paperv2_phase2`=376, `paperv2_phase4`=300, `paper_backbones`=248, `contamination_dermmnist`=120, `derm_cripple`=120, `paperv2_phase5`=100, `blackwell_validation`=72, `paperv2_phase1`=70, `expansion_dermmnist_baselines`=54, `derm_backbone_weak`=48, `lr_hp_smoke`=42, `contamination_clean`=40, `warmup_probe`=24, `g5_component_ablation`=21, `paperv2_phase3_v3`=16, `model_search`=15, `g4_table_b_backfill`=12, `warmup_confirm`=12, `class_rotation`=12

## octmnist  (n=72)

- **Models** (1): `MobileNetV3`=72
- **Methods**: `hounie_rcl`=16, `tralo`=16, `fioretto_ldf`=16, `danits_lp`=12, `heuristic`=12
- **Tightness** (3 unique): `L30_G30`=26, `L50_G50`=26, `L30_G50`=20
- **Contamination sigma**: `0.0`=72
- **Sweeps contributing**: `octmnist_expansion`=60, `octmnist_smoke`=12

## retinamnist  (n=12)

- **Models** (1): `MobileNetV3`=12
- **Methods**: `hounie_rcl`=4, `tralo`=4, `fioretto_ldf`=4
- **Tightness** (2 unique): `L30_G30`=6, `L50_G50`=6
- **Contamination sigma**: `0.0`=12
- **Sweeps contributing**: `new_dataset_probes`=12

## tissuemnist  (n=1363)

- **Models** (4): `MobileNetV3`=1063, `MobileNetV2`=100, `ShuffleNetV2`=100, `RegNetY400MF`=100
- **Methods**: `tralo`=311, `hounie_rcl`=268, `fioretto_ldf`=268, `danits_lp`=258, `heuristic`=258
- **Tightness** (25 unique): `L30_G30`=259, `L50_G50`=204, `L20_G20`=180, `L70_G70`=180, `L80_G80`=140, `L50_G20`=20, `L80_G30`=20, `L20_G80`=20, `L30_G50`=20, `L30_G20`=20
- **Contamination sigma**: `0.0`=1213, `0.1`=40, `0.2`=70, `0.3`=40
- **Sweeps contributing**: `g2_asym_tissue_aider`=376, `g3_multiclass_tissue`=300, `paper_backbones`=257, `contamination_tissuemnist`=120, `expansion_baselines`=54, `contamination_clean`=40, `paperv2_phase3_v3`=37, `paperv2_phase1`=34, `lr_hp_smoke`=30, `paper400_baselines`=24, `g5_component_ablation`=21, `class_rotation`=18, `component_ablation`=14, `paper400_tralofix`=12, `arch_validation`=12, `kl_ablation`=8, `model_search`=6