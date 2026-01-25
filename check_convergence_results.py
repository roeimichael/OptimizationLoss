"""
Diagnostic script to check convergence experiment results.
"""

import json
from pathlib import Path
import pandas as pd


def check_results():
    """Check for convergence experiment results and analyze them."""

    base_dir = Path('results/longer_saturation')

    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return

    print(f"Checking: {base_dir}")
    print("=" * 80)

    # Find all config files
    all_configs = list(base_dir.glob('**/config.json'))
    print(f"\n✓ Found {len(all_configs)} config files")

    # Find all result files
    eval_files = list(base_dir.glob('**/evaluation_metrics.csv'))
    log_files = list(base_dir.glob('**/training_log.csv'))
    pred_files = list(base_dir.glob('**/final_predictions.csv'))
    status_files = list(base_dir.glob('**/run_status.json'))

    print(f"✓ Found {len(eval_files)} evaluation_metrics.csv files")
    print(f"✓ Found {len(log_files)} training_log.csv files")
    print(f"✓ Found {len(pred_files)} final_predictions.csv files")
    print(f"✓ Found {len(status_files)} run_status.json files")

    if len(eval_files) == 0:
        print("\n" + "=" * 80)
        print("❌ NO RESULTS FOUND")
        print("=" * 80)
        print("\nExperiments have NOT been run yet.")
        print("Configs exist but no training has occurred.")
        print("\nTo run experiments:")
        print("  python run_all_convergence_experiments.py")
        return

    print("\n" + "=" * 80)
    print("ANALYZING RESULTS FOR CONSTRAINT [0.5, 0.3]")
    print("=" * 80)

    # Check constraint_50_30 results
    constraint_dir = base_dir / 'TabularResNet' / 'constraint_50_30' / 'convergence_test'

    if not constraint_dir.exists():
        print(f"\n❌ Constraint directory not found: {constraint_dir}")
        return

    # Analyze conv_1_1 (baseline)
    conv_1_1_dir = constraint_dir / 'conv_1_1'

    if not conv_1_1_dir.exists():
        print(f"\n❌ conv_1_1 directory not found: {conv_1_1_dir}")
        return

    print(f"\n📁 Analyzing: {conv_1_1_dir}")
    print("-" * 80)

    # Check config
    config_path = conv_1_1_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n✓ Config:")
        print(f"  Constraint: {config.get('constraint')}")
        print(f"  convergence_window: {config.get('hyperparams', {}).get('convergence_window')}")
        print(f"  convergence_required: {config.get('hyperparams', {}).get('convergence_required')}")

    # Check evaluation metrics
    eval_path = conv_1_1_dir / 'evaluation_metrics.csv'
    if eval_path.exists():
        print(f"\n✓ Evaluation metrics found")
        # Read and display key metrics
        with open(eval_path) as f:
            lines = f.readlines()
            print(f"  File contents:")
            for line in lines[:20]:  # First 20 lines
                print(f"    {line.strip()}")
    else:
        print(f"\n❌ evaluation_metrics.csv NOT FOUND")

    # Check training log
    log_path = conv_1_1_dir / 'training_log.csv'
    if log_path.exists():
        print(f"\n✓ Training log found")
        df = pd.read_csv(log_path)
        print(f"  Total epochs logged: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")

        # Check final epoch
        if len(df) > 0:
            last_row = df.iloc[-1]
            print(f"\n  Final epoch {int(last_row['Epoch'])}:")
            print(f"    Global satisfied: {last_row.get('Global_Satisfied', 'N/A')}")
            print(f"    Local satisfied: {last_row.get('Local_Satisfied', 'N/A')}")

            if 'Hard_Dropout' in df.columns:
                print(f"    Dropout predictions: {last_row.get('Hard_Dropout', 'N/A')} (limit: 43)")
            if 'Hard_Enrolled' in df.columns:
                print(f"    Enrolled predictions: {last_row.get('Hard_Enrolled', 'N/A')} (limit: 24)")
            if 'Hard_Graduate' in df.columns:
                print(f"    Graduate predictions: {last_row.get('Hard_Graduate', 'N/A')} (unlimited)")
    else:
        print(f"\n❌ training_log.csv NOT FOUND")

    # Check final predictions
    pred_path = conv_1_1_dir / 'final_predictions.csv'
    if pred_path.exists():
        print(f"\n✓ Final predictions found")
        df = pd.read_csv(pred_path)
        pred_counts = df['Predicted'].value_counts().sort_index()
        print(f"  Prediction distribution:")
        for class_id, count in pred_counts.items():
            print(f"    Class {class_id}: {count} predictions")

        # Check if exceeds limits
        if pred_counts.get(0, 0) > 43:
            print(f"\n  ⚠️  WARNING: Dropout predictions ({pred_counts.get(0, 0)}) EXCEED limit (43)!")
        if pred_counts.get(1, 0) > 24:
            print(f"  ⚠️  WARNING: Enrolled predictions ({pred_counts.get(1, 0)}) EXCEED limit (24)!")
    else:
        print(f"\n❌ final_predictions.csv NOT FOUND")

    # Check run status
    status_path = conv_1_1_dir / 'run_status.json'
    if status_path.exists():
        print(f"\n✓ Run status found")
        with open(status_path) as f:
            status = json.load(f)
        print(f"  Status: {status.get('status')}")
        print(f"  Details: {status.get('details', 'N/A')}")
    else:
        print(f"\n❌ run_status.json NOT FOUND")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    check_results()
