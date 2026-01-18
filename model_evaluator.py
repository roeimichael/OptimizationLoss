#!/usr/bin/env python3
"""
Model Evaluator Script
======================

Generates visualization plots for trained models by analyzing training logs and evaluation metrics.

Usage:
    python3 model_evaluator.py

This script will:
1. Search for all result folders containing training_log.csv files
2. Generate three types of plots for each folder:
   - Loss Functions: L_pred (CE), L_target (Global), L_feat (Local) over epochs
   - Predictions by Class: Soft predictions, Hard predictions, and Constraint limits
   - Confusion Matrix: Heatmap showing true vs predicted labels

Output:
    Plots are saved in the same directory as the training logs:
    - plot_loss_functions.png
    - plot_predictions_by_class.png
    - plot_confusion_matrix.png

Requirements:
    - pandas
    - matplotlib
    - numpy
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class ModelEvaluator:
    """Generates visualization plots for trained model results."""

    def __init__(self, results_base_path: str = './results/our_approach'):
        self.results_base_path = Path(results_base_path)
        self.class_names = ['Dropout', 'Enrolled', 'Graduate']

    def find_result_folders(self):
        """Find all folders containing training_log.csv files."""
        result_folders = []

        for root, dirs, files in os.walk(self.results_base_path):
            if 'training_log.csv' in files:
                result_folders.append(Path(root))

        return result_folders

    def plot_loss_functions(self, df: pd.DataFrame, save_path: Path):
        """
        Plot loss functions through epochs.

        Args:
            df: Training log dataframe
            save_path: Path to save the plot
        """
        # Filter to constraint epochs only (where Limit_Dropout is not inf)
        df_constrained = df[df['Limit_Dropout'] != 'inf'].copy()

        if len(df_constrained) == 0:
            print(f"  No constraint epochs found, skipping loss plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each loss function
        ax.plot(df_constrained['Epoch'], df_constrained['L_pred_CE'],
                label='L_pred (Cross-Entropy)', linewidth=2, alpha=0.8)
        ax.plot(df_constrained['Epoch'], df_constrained['L_target_Global'],
                label='L_target (Global Constraints)', linewidth=2, alpha=0.8)
        ax.plot(df_constrained['Epoch'], df_constrained['L_feat_Local'],
                label='L_feat (Local Constraints)', linewidth=2, alpha=0.8)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Functions Over Training Epochs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Loss plot saved: {save_path.name}")

    def plot_predictions_by_class(self, df: pd.DataFrame, save_path: Path):
        """
        Plot soft, hard, and constraint counts for each class.

        Args:
            df: Training log dataframe
            save_path: Path to save the plot
        """
        # Filter to constraint epochs only
        df_constrained = df[df['Limit_Dropout'] != 'inf'].copy()

        if len(df_constrained) == 0:
            print(f"  No constraint epochs found, skipping predictions plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, class_name in enumerate(self.class_names):
            ax = axes[idx]

            # Get column names for this class
            limit_col = f'Limit_{class_name}'
            hard_col = f'Hard_{class_name}'
            soft_col = f'Soft_{class_name}'

            # Convert soft predictions to numeric (they might be strings)
            soft_vals = pd.to_numeric(df_constrained[soft_col], errors='coerce')
            hard_vals = df_constrained[hard_col].astype(int)

            # Plot soft and hard predictions
            ax.plot(df_constrained['Epoch'], soft_vals,
                   label='Soft Predictions', linewidth=2, alpha=0.7, color='blue')
            ax.plot(df_constrained['Epoch'], hard_vals,
                   label='Hard Predictions', linewidth=2, alpha=0.7, color='green')

            # Plot constraint limit if not unlimited
            limit_val = df_constrained[limit_col].iloc[0]
            if limit_val != 'inf' and limit_val < 1e9:
                limit_numeric = float(limit_val) if isinstance(limit_val, str) else limit_val
                ax.axhline(y=limit_numeric, color='red', linestyle='--',
                          linewidth=2, label=f'Constraint Limit ({int(limit_numeric)})', alpha=0.8)

            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f'{class_name} Predictions', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Soft vs Hard Predictions by Class', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Predictions plot saved: {save_path.name}")

    def plot_confusion_matrix(self, eval_metrics_path: Path, save_path: Path):
        """
        Plot confusion matrix heatmap.

        Args:
            eval_metrics_path: Path to evaluation_metrics.csv
            save_path: Path to save the plot
        """
        if not eval_metrics_path.exists():
            print(f"  Evaluation metrics not found, skipping confusion matrix")
            return

        # Read the file manually to handle variable column counts
        try:
            with open(eval_metrics_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  Could not read evaluation metrics: {e}")
            return

        # Find confusion matrix section
        cm_start = None
        for i, line in enumerate(lines):
            if 'Confusion Matrix' in line:
                cm_start = i + 1  # Skip to header row
                break

        if cm_start is None:
            print(f"  Confusion matrix not found in metrics file")
            return

        # Parse confusion matrix (skip header, read next 3 data rows)
        cm_data = []
        for i in range(cm_start + 1, cm_start + 4):  # Skip header row at cm_start
            if i < len(lines):
                parts = lines[i].strip().split(',')
                if len(parts) >= 4:  # Should have class name + 3 values
                    try:
                        row_data = [int(parts[j]) for j in range(1, 4)]  # Skip class name, take 3 values
                        cm_data.append(row_data)
                    except (ValueError, IndexError) as e:
                        print(f"  Error parsing confusion matrix row: {e}")
                        return

        if len(cm_data) != 3:
            print(f"  Could not parse confusion matrix (got {len(cm_data)} rows, expected 3)")
            return

        cm_array = np.array(cm_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap using imshow
        im = ax.imshow(cm_array, cmap='Blues', aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', fontsize=11)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(self.class_names)))
        ax.set_yticks(np.arange(len(self.class_names)))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)

        # Annotate cells with values
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                text = ax.text(j, i, int(cm_array[i, j]),
                             ha="center", va="center", color="black" if cm_array[i, j] < cm_array.max()/2 else "white",
                             fontsize=12, fontweight='bold')

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        # Add accuracy annotation
        accuracy = np.trace(cm_array) / np.sum(cm_array)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Confusion matrix saved: {save_path.name}")

    def evaluate_folder(self, folder_path: Path):
        """
        Generate all visualization plots for a single result folder.

        Args:
            folder_path: Path to the result folder
        """
        training_log_path = folder_path / 'training_log.csv'
        eval_metrics_path = folder_path / 'evaluation_metrics.csv'

        if not training_log_path.exists():
            return

        # Read training log
        try:
            df = pd.read_csv(training_log_path)
        except Exception as e:
            print(f"  ✗ Error reading training log: {e}")
            return

        # Generate plots
        print(f"\nProcessing: {folder_path.relative_to(self.results_base_path)}")

        # 1. Loss functions plot
        loss_plot_path = folder_path / 'plot_loss_functions.png'
        try:
            self.plot_loss_functions(df, loss_plot_path)
        except Exception as e:
            print(f"  ✗ Error generating loss plot: {e}")

        # 2. Predictions by class plot
        predictions_plot_path = folder_path / 'plot_predictions_by_class.png'
        try:
            self.plot_predictions_by_class(df, predictions_plot_path)
        except Exception as e:
            print(f"  ✗ Error generating predictions plot: {e}")

        # 3. Confusion matrix plot
        cm_plot_path = folder_path / 'plot_confusion_matrix.png'
        try:
            self.plot_confusion_matrix(eval_metrics_path, cm_plot_path)
        except Exception as e:
            print(f"  ✗ Error generating confusion matrix: {e}")

    def run(self):
        """Run evaluation on all result folders."""
        print("="*80)
        print("MODEL EVALUATOR - Generating Visualization Plots")
        print("="*80)

        result_folders = self.find_result_folders()

        if not result_folders:
            print(f"\nNo result folders found in {self.results_base_path}")
            return

        print(f"\nFound {len(result_folders)} result folder(s) to process")

        # Track statistics
        total_plots = 0
        successful_folders = 0

        for folder in result_folders:
            plots_before = len(list(folder.glob('plot_*.png')))
            self.evaluate_folder(folder)
            plots_after = len(list(folder.glob('plot_*.png')))

            plots_generated = plots_after - plots_before
            total_plots += plots_generated
            if plots_generated > 0:
                successful_folders += 1

        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"  Folders processed: {len(result_folders)}")
        print(f"  Folders with plots: {successful_folders}")
        print(f"  Total plots generated: {total_plots}")
        print(f"\nPlot types generated in each folder:")
        print(f"  1. plot_loss_functions.png - Loss functions over epochs")
        print(f"  2. plot_predictions_by_class.png - Soft/Hard predictions by class")
        print(f"  3. plot_confusion_matrix.png - Confusion matrix heatmap")


def main():
    """Main entry point."""
    evaluator = ModelEvaluator(results_base_path='./results/our_approach')
    evaluator.run()


if __name__ == '__main__':
    main()
