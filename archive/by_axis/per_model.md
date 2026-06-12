# Per-backbone breakdown


## MobileNetV3  (n=3416)

- **Datasets**: `dermmnist`=1354, `tissuemnist`=1063, `aider`=867, `octmnist`=72, `cifar100`=48, `retinamnist`=12
- **Methods**: `tralo`=794, `fioretto_ldf`=698, `hounie_rcl`=656, `danits_lp`=634, `heuristic`=634
- **Pretrained {0,1}**: `1`=3384, `0`=32
- **Sweeps**: `g2_asym_tissue_aider`=672, `paperv2_phase2`=376, `g3_multiclass_tissue`=300, `paperv2_phase4`=300, `paperv2_phase1`=173, `contamination_clean`=120, `contamination_tissuemnist`=120, `contamination_dermmnist`=120, `contamination_aider`=120, `derm_cripple`=120, `lr_hp_smoke`=102, `paperv2_phase5`=100, `aider_seed_ext`=88, `aider_asym`=80, `g5_component_ablation`=63, `octmnist_expansion`=60, `expansion_baselines`=54, `expansion_aider_baselines`=54, `expansion_dermmnist_baselines`=54, `blackwell_validation`=48, `class_rotation`=48, `cifar100_smalltrain`=36, `paper400_baselines`=24, `aider_cripple`=24, `derm_backbone_weak`=24, `warmup_probe`=24, `new_dataset_probes`=24, `component_ablation`=14, `paper400_tralofix`=12, `g4_table_b_backfill`=12, `arch_validation`=12, `warmup_confirm`=12, `octmnist_smoke`=12, `kl_ablation`=8, `model_search`=6

## MobileNetV2  (n=331)

- **Datasets**: `dermmnist`=116, `aider`=115, `tissuemnist`=100
- **Methods**: `tralo`=71, `hounie_rcl`=70, `fioretto_ldf`=70, `danits_lp`=60, `heuristic`=60
- **Pretrained {0,1}**: `1`=331
- **Sweeps**: `paper_backbones`=240, `blackwell_validation`=48, `paperv2_phase3_v3`=33, `model_search`=10

## ShuffleNetV2  (n=331)

- **Datasets**: `dermmnist`=128, `aider`=103, `tissuemnist`=100
- **Methods**: `tralo`=69, `fioretto_ldf`=68, `danits_lp`=66, `heuristic`=66, `hounie_rcl`=62
- **Pretrained {0,1}**: `1`=331
- **Sweeps**: `paper_backbones`=273, `blackwell_validation`=24, `derm_backbone_weak`=24, `model_search`=10

## RegNetY400MF  (n=307)

- **Datasets**: `dermmnist`=104, `aider`=103, `tissuemnist`=100
- **Methods**: `tralo`=63, `hounie_rcl`=62, `fioretto_ldf`=62, `danits_lp`=60, `heuristic`=60
- **Pretrained {0,1}**: `1`=307
- **Sweeps**: `paper_backbones`=240, `paperv2_phase3_v3`=36, `blackwell_validation`=24, `model_search`=7