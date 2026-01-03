#!/usr/bin/env python3
"""
Script to move existing results to a 'baseline' subfolder to preserve them.
This organizes results by model variant for better comparison.
"""

import os
import shutil
from pathlib import Path


def move_to_baseline():
    """
    Move existing results from results/ to results/baseline/
    Preserves the current constraint folder structure.
    """
    results_dir = Path('results')
    baseline_dir = results_dir / 'baseline'

    if not results_dir.exists():
        print("No results directory found. Nothing to move.")
        return

    if baseline_dir.exists():
        response = input(f"\n{baseline_dir} already exists. Do you want to merge/overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return

    baseline_dir.mkdir(parents=True, exist_ok=True)

    items_to_move = [
        'constraint_0.4_0.2',
        'constraint_0.5_0.3',
        'constraint_0.6_0.5',
        'constraint_0.7_0.5',
        'constraint_0.8_0.2',
        'constraint_0.8_0.7',
        'constraint_0.9_0.5',
        'constraint_0.9_0.8',
        'nn_results.json'
    ]

    moved_count = 0
    skipped_count = 0

    print("\nMoving results to baseline folder...")
    print("="*60)

    for item in items_to_move:
        source = results_dir / item
        if source.exists():
            dest = baseline_dir / item
            try:
                if source.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(source, dest)
                    shutil.rmtree(source)
                else:
                    shutil.copy2(source, dest)
                    source.unlink()

                print(f"Moved: {item}")
                moved_count += 1
            except Exception as e:
                print(f"Error moving {item}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1

    other_items = [
        'constraint_analysis',
        'comprehensive_analysis',
        'constraint_comparison'
    ]

    for item in other_items:
        source = results_dir / item
        if source.exists():
            print(f"Keeping: {item} (analysis folder)")

    print("="*60)
    print(f"\nSummary:")
    print(f"  Moved: {moved_count} items")
    print(f"  Skipped: {skipped_count} items")
    print(f"\nBaseline results saved to: {baseline_dir}")
    print("\nNext steps:")
    print("  1. Run experiments with: python experiments/run_experiments_with_variants.py --variant enhanced_gelu_residual --use-residual --activation gelu")
    print("  2. Compare results across variants using the comprehensive analysis script")


if __name__ == "__main__":
    print("="*60)
    print("Baseline Results Organization Tool")
    print("="*60)
    print("\nThis script will move your current experiment results to:")
    print("  results/baseline/")
    print("\nThis preserves your existing results and allows testing new model")
    print("variants in separate folders for easy comparison.")

    response = input("\nProceed? (y/n): ")

    if response.lower() == 'y':
        move_to_baseline()
    else:
        print("\nOperation cancelled.")
