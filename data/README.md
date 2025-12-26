# Dataset Directory

This directory should contain the dataset file for the student dropout prediction experiments.

## Required Dataset

**File**: `dataset.csv`

Place the dataset CSV file in this directory. The file should contain student records with features and a `Target` column indicating dropout status.

## Dataset Structure Expected

- Target column: `Target` (0 = Dropout, 1 = Enrolled, 2 = Graduate)
- Course column: `Course` (for local constraints per course)
- Additional features: demographic, academic, and socioeconomic indicators
- Optional: `cost_matrix` column (will be dropped during preprocessing)

## Note

The dataset file is excluded from git (see `.gitignore`) to avoid committing large data files to the repository.
