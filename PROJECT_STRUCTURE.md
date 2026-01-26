# Project Structure

## Overview

This project implements and evaluates constraint satisfaction approaches for educational outcome prediction using neural networks.

**Current Phase:** Preparing for larger dataset experiments
**Previous Phase:** Initial validation with OULAD dataset (archived)

---

## 📁 Directory Structure

```
OptimizationLoss/
│
├── archive_oulad_experiments/     # 🗄️ ARCHIVED: All OULAD experiments and results
│   ├── results/                   # 443 completed experiments
│   ├── summaries/                 # CSV/Excel summaries
│   ├── paper_tables/              # LaTeX tables ready for publication
│   ├── documentation/             # Analysis and guides
│   ├── analysis_scripts/          # Data analysis and table generation
│   └── README.md                  # Complete archive documentation
│
├── src/                           # 🔧 Core implementation (active)
│   ├── experiments/               # Experiment runners
│   ├── losses/                    # Loss functions and lambda adjustment
│   ├── models/                    # Neural network architectures
│   ├── training/                  # Training loops and constraints
│   ├── utils/                     # Data loading and utilities
│   └── config_generators/         # Configuration generators
│
├── config/                        # ⚙️ Configuration files
│   ├── experiment_config.py       # Main experiment configuration
│   └── model_configs.py           # Model hyperparameters
│
├── data/                          # 📊 Dataset storage
│   ├── test.csv                   # Test split
│   └── train.csv                  # Training split
│
├── model_cache/                   # 💾 Cached warmup models
│   └── *.pth                      # Saved model states
│
├── README.md                      # 📖 Main project documentation
├── requirements.txt               # 📦 Python dependencies
├── PROJECT_STRUCTURE.md           # 📋 This file
└── main.py                        # 🚀 Main entry point

```

---

## 🗄️ Archived Experiments (OULAD Dataset)

All previous experimental results have been organized and archived in:
**`archive_oulad_experiments/`**

### What's Archived:
- ✅ 443 completed experiments (Heuristic, Our Approach, Saturated)
- ✅ 11 convergence testing experiments
- ✅ All CSV/Excel summaries and comparisons
- ✅ LaTeX tables ready for paper
- ✅ Complete documentation and analysis
- ✅ Analysis scripts for reproducibility

### When to Use Archive:
- 📝 Writing the paper (use ready-made tables)
- 📊 Comparing with OULAD baseline results
- 🔬 Understanding what was tested in Phase 1
- 📈 Pulling statistics for introduction/related work

See **`archive_oulad_experiments/README.md`** for complete documentation.

---

## 🚀 Active Development

### Current Focus
Preparing for experiments with a **larger, more robust dataset** to:
- Avoid small constraint issues with soft predictions
- Get more stable and generalizable results
- Generate final results for publication

### Core Source Code (`src/`)

**`src/experiments/`** - Experiment execution
- `run_experiment.py` - Main experiment runner
- `run_heuristic.py` - Heuristic baseline runner

**`src/losses/`** - Loss functions and optimization
- `transductive_loss.py` - Constraint satisfaction loss
- `lambda_adjusting.py` - Lambda adjustment strategies (linear, balanced, transfer, combined)

**`src/models/`** - Neural network architectures
- `basic_nn.py` - Simple feedforward network
- `tabular_resnet.py` - ResNet for tabular data
- `ft_transformer.py` - Feature Tokenizer Transformer

**`src/training/`** - Training infrastructure
- `trainer.py` - Main training loop
- `constraints.py` - Constraint computation
- `sustained_convergence.py` - Convergence checking
- `base_trainer.py` - Base trainer class

**`src/utils/`** - Utilities
- `data_loader.py` - Dataset loading and preprocessing
- `filesystem_manager.py` - File management

**`src/config_generators/`** - Configuration generators
- Various scripts for generating experiment configs

---

## 🔧 Key Configuration Files

**`config/experiment_config.py`**
- Dataset paths
- Training/test split locations
- Global experiment settings

**`config/model_configs.py`**
- Model architecture hyperparameters
- Default configurations for each model

---

## 📊 Data Directory

**`data/`** - Currently contains OULAD dataset
- Will be replaced with new larger dataset
- Format: CSV files with train/test splits

---

## 💾 Model Cache

**`model_cache/`** - Stores warmup models
- Reuses pretrained warmup phases
- Indexed by `base_model_id` hash
- Saves training time for constraint phase

---

## 🎯 Next Steps

### Phase 2: Larger Dataset Experiments

1. **Setup new dataset:**
   - Load larger, more robust dataset
   - Update paths in `config/experiment_config.py`
   - Verify constraint computation works correctly

2. **Run experiments:**
   - Same experimental setup as Phase 1
   - Test all constraint pairs
   - Compare lambda strategies

3. **Generate results:**
   - Use archived analysis scripts (from `archive_oulad_experiments/analysis_scripts/`)
   - Create new paper tables
   - Compare with Phase 1 results if needed

4. **Paper preparation:**
   - Use archived OULAD results for baseline comparisons
   - New dataset results for main findings
   - Show consistency across datasets

---

## 📝 Using Archived Results for Paper

### Quick Access to Paper-Ready Tables

**Best table for showing wins:**
```latex
\input{archive_oulad_experiments/paper_tables/paper_table_wins_only_latex.tex}
```

**Complete comparison:**
```latex
\input{archive_oulad_experiments/paper_tables/paper_table_all_constraints_latex.tex}
```

### Key Statistics to Cite

From `archive_oulad_experiments/summaries/paper_summary_stats.txt`:
- Total OULAD experiments: 442 completed
- Best accuracy: 76.24%
- Our approach improvements: +13.26% on [0.5, 0.3], +14.38% on [0.8, 0.2]

### Detailed Analysis

See `archive_oulad_experiments/documentation/` for:
- EXPERIMENT_RESULTS_README.md - Complete results documentation
- FOCUSED_COMPARISON_SUMMARY.md - Where our approach wins
- PAPER_TABLES_GUIDE.md - Guide to using tables
- convergence_issues_analysis.md - Technical findings

---

## 🔄 Regenerating Analyses (Future)

If you need to regenerate summaries for new experiments:

1. Copy relevant scripts from `archive_oulad_experiments/analysis_scripts/`
2. Update paths to point to new results directory
3. Run the scripts to generate new tables

**Example:**
```bash
# Copy script
cp archive_oulad_experiments/analysis_scripts/generate_paper_summary.py .

# Update paths in script (results/ → new_results/)
# Run
python generate_paper_summary.py
```

---

## 📚 Dependencies

**Key Libraries:**
- PyTorch - Deep learning framework
- pandas - Data manipulation
- scikit-learn - Metrics and utilities
- openpyxl - Excel file generation

See `requirements.txt` for complete list.

---

## ✅ Project Cleanliness Checklist

- ✅ Old experiments archived
- ✅ Documentation preserved
- ✅ Paper tables ready to use
- ✅ Source code clean and active
- ✅ Ready for new dataset experiments
- ✅ Analysis scripts available for reuse

---

## 🎓 Research Workflow

### For Writing the Paper:

1. **Use archived results** for OULAD baseline comparisons
2. **Generate new results** with larger dataset
3. **Compare across datasets** to show consistency
4. **Pull statistics** from archive for introduction
5. **Use ready-made tables** from archive for quick insertion

### For New Experiments:

1. **Configure** new dataset in `config/`
2. **Run experiments** using `src/experiments/`
3. **Analyze results** using scripts from archive
4. **Generate tables** for paper
5. **Archive** when moving to next phase

---

**Last Updated:** 2026-01-26
**Status:** Ready for Phase 2 (Larger Dataset Experiments)
**Archive:** Complete and documented in `archive_oulad_experiments/`
