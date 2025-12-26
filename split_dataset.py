#!/usr/bin/env python3
"""
Dataset Splitter Script
Splits dataset.csv into dataset_train.csv and dataset_test.csv
Ensures consistent train/test split across all experiments
"""

import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.experiment_config import DATA_PATH, TARGET_COLUMN, TRAINING_PARAMS
from src.data import load_and_preprocess_data

def split_and_save_dataset():
    """Split dataset into train and test sets and save them."""

    print("=" * 70)
    print("Dataset Splitting Script")
    print("=" * 70)
    print()

    # Load and preprocess data
    print(f"Loading dataset from: {DATA_PATH}")
    df = load_and_preprocess_data(DATA_PATH, TARGET_COLUMN)
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.shape[1]}")
    print()

    # Get test size
    test_size = TRAINING_PARAMS['test_size']
    test_samples = int(len(df) * test_size)
    train_samples = len(df) - test_samples

    print(f"Split configuration:")
    print(f"  Test size: {test_size * 100}% ({test_samples} samples)")
    print(f"  Train size: {100 - test_size * 100}% ({train_samples} samples)")
    print()

    # Split without shuffling to maintain order (as original code does)
    train_df = df.iloc[:train_samples].copy()
    test_df = df.iloc[train_samples:].copy()

    # Verify split
    print("Class distribution:")
    print(f"  Original - {TARGET_COLUMN}:")
    print(f"    {df[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"  Train - {TARGET_COLUMN}:")
    print(f"    {train_df[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print(f"  Test - {TARGET_COLUMN}:")
    print(f"    {test_df[TARGET_COLUMN].value_counts().sort_index().to_dict()}")
    print()

    # Save split datasets
    train_path = DATA_PATH.replace('.csv', '_train.csv')
    test_path = DATA_PATH.replace('.csv', '_test.csv')

    print(f"Saving train set to: {train_path}")
    train_df.to_csv(train_path, index=False)

    print(f"Saving test set to: {test_path}")
    test_df.to_csv(test_path, index=False)

    print()
    print("=" * 70)
    print("✓ Dataset split complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  - {train_path} ({len(train_df)} samples)")
    print(f"  - {test_path} ({len(test_df)} samples)")
    print()
    print("Next steps:")
    print("  1. Verify the split files in the data/ directory")
    print("  2. Run experiments using: cd experiments && python run_experiments.py")
    print()

if __name__ == "__main__":
    split_and_save_dataset()
