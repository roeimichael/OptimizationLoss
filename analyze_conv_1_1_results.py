"""
Analyze conv_1_1 results for constraint [0.5, 0.3] to diagnose issues.

This script searches for results regardless of directory naming format
(constraint_0.5_0.3 or constraint_50_30) and provides detailed analysis.
"""

import json
from pathlib import Path
import pandas as pd


def find_conv_1_1_results():
    """Find conv_1_1 results for constraint [0.5, 0.3] regardless of naming."""

    base = Path('results/longer_saturation/TabularResNet')

    if not base.exists():
        print(f"❌ Directory not found: {base}")
        return None

    # Try both naming formats
    possible_paths = [
        base / 'constraint_0.5_0.3' / 'convergence_test' / 'conv_1_1',
        base / 'constraint_50_30' / 'convergence_test' / 'conv_1_1',
    ]

    for path in possible_paths:
        if path.exists():
            print(f"✓ Found results at: {path}")
            return path

    print("❌ conv_1_1 results not found in either format:")
    print("  - constraint_0.5_0.3/convergence_test/conv_1_1")
    print("  - constraint_50_30/convergence_test/conv_1_1")
    return None


def analyze_results(results_dir: Path):
    """Analyze experiment results and diagnose issues."""

    print("\n" + "=" * 80)
    print("ANALYSIS: conv_1_1 (Baseline - Immediate Convergence)")
    print("=" * 80)

    # 1. Check config
    print("\n1. CONFIG VERIFICATION")
    print("-" * 80)
    config_path = results_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        constraint = config.get('constraint', [])
        print(f"✓ Constraint: {constraint}")
        print(f"  Expected: [0.5, 0.3] → 43 dropout limit, 24 enrolled limit")

        hyperparams = config.get('hyperparams', {})
        print(f"\n✓ Convergence parameters:")
        print(f"  convergence_window: {hyperparams.get('convergence_window', 'NOT SET')}")
        print(f"  convergence_required: {hyperparams.get('convergence_required', 'NOT SET')}")
        print(f"  → Should be (1, 1) for immediate convergence")

        if constraint != [0.5, 0.3]:
            print(f"\n⚠️  WARNING: Constraint mismatch!")
            print(f"   Expected: [0.5, 0.3]")
            print(f"   Got: {constraint}")
    else:
        print("❌ config.json NOT FOUND")
        return

    # 2. Check training log
    print("\n2. TRAINING LOG ANALYSIS")
    print("-" * 80)
    log_path = results_dir / 'training_log.csv'
    if log_path.exists():
        df = pd.read_csv(log_path)
        print(f"✓ Training log found: {len(df)} epochs logged")

        # Show column names
        print(f"\nColumns: {df.columns.tolist()}")

        # Check if constraints were ever satisfied
        if 'Global_Satisfied' in df.columns:
            global_sat_count = df['Global_Satisfied'].sum()
            local_sat_count = df['Local_Satisfied'].sum()
            both_sat = ((df['Global_Satisfied'] == 1) & (df['Local_Satisfied'] == 1)).sum()

            print(f"\nConstraint satisfaction over {len(df)} epochs:")
            print(f"  Global satisfied: {global_sat_count} epochs ({global_sat_count/len(df)*100:.1f}%)")
            print(f"  Local satisfied: {local_sat_count} epochs ({local_sat_count/len(df)*100:.1f}%)")
            print(f"  BOTH satisfied: {both_sat} epochs ({both_sat/len(df)*100:.1f}%)")

            if both_sat == 0:
                print(f"\n⚠️  CRITICAL ISSUE: Constraints were NEVER both satisfied!")
                print(f"   With conv_1_1, training should stop at first satisfaction.")
                print(f"   Reaching max epochs means constraints were never met.")

        # Show first 5 and last 5 epochs
        print(f"\nFirst 5 epochs:")
        print(df.head(5).to_string())

        print(f"\nLast 5 epochs:")
        print(df.tail(5).to_string())

        # Check final predictions
        if len(df) > 0:
            last = df.iloc[-1]
            print(f"\n📊 FINAL EPOCH {int(last['Epoch'])} PREDICTIONS:")

            if 'Hard_Dropout' in df.columns:
                dropout_pred = int(last['Hard_Dropout'])
                enrolled_pred = int(last['Hard_Enrolled'])
                graduate_pred = int(last['Hard_Graduate'])

                print(f"  Dropout:  {dropout_pred:3d} predictions (limit: 43)")
                print(f"  Enrolled: {enrolled_pred:3d} predictions (limit: 24)")
                print(f"  Graduate: {graduate_pred:3d} predictions (unlimited)")
                print(f"  Total:    {dropout_pred + enrolled_pred + graduate_pred} predictions")

                # Check for violations
                if dropout_pred > 43:
                    print(f"\n  ❌ CONSTRAINT VIOLATION: Dropout {dropout_pred} > 43!")
                    print(f"     Exceeded by: {dropout_pred - 43}")

                if enrolled_pred > 24:
                    print(f"\n  ❌ CONSTRAINT VIOLATION: Enrolled {enrolled_pred} > 24!")
                    print(f"     Exceeded by: {enrolled_pred - 24}")

                if dropout_pred <= 43 and enrolled_pred <= 24:
                    print(f"\n  ✓ Constraints satisfied in final epoch")

            # Check soft predictions too
            if 'Soft_Dropout' in df.columns:
                soft_dropout = last['Soft_Dropout']
                soft_enrolled = last['Soft_Enrolled']
                soft_graduate = last['Soft_Graduate']

                print(f"\n  Soft predictions (may differ from hard):")
                print(f"  Dropout:  {soft_dropout:.2f}")
                print(f"  Enrolled: {soft_enrolled:.2f}")
                print(f"  Graduate: {soft_graduate:.2f}")
    else:
        print("❌ training_log.csv NOT FOUND")
        return

    # 3. Check evaluation metrics
    print("\n3. EVALUATION METRICS")
    print("-" * 80)
    eval_path = results_dir / 'evaluation_metrics.csv'
    if eval_path.exists():
        print(f"✓ evaluation_metrics.csv found")
        with open(eval_path) as f:
            content = f.read()
        print(content)
    else:
        print("❌ evaluation_metrics.csv NOT FOUND")

    # 4. Check final predictions
    print("\n4. FINAL PREDICTIONS DISTRIBUTION")
    print("-" * 80)
    pred_path = results_dir / 'final_predictions.csv'
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        pred_counts = df_pred['Predicted'].value_counts().sort_index()

        print(f"✓ Final prediction distribution:")
        for class_id, count in pred_counts.items():
            class_name = ['Dropout', 'Enrolled', 'Graduate'][int(class_id)]
            limit = [43, 24, 'unlimited'][int(class_id)]
            print(f"  Class {class_id} ({class_name:8s}): {count:3d} predictions (limit: {limit})")

        # Check violations
        if pred_counts.get(0, 0) > 43:
            print(f"\n  ❌ FINAL VIOLATION: Dropout predictions {pred_counts[0]} > 43!")
        if pred_counts.get(1, 0) > 24:
            print(f"\n  ❌ FINAL VIOLATION: Enrolled predictions {pred_counts[1]} > 24!")

        if pred_counts.get(0, 0) <= 43 and pred_counts.get(1, 0) <= 24:
            print(f"\n  ✓ Final predictions satisfy constraints")
    else:
        print("❌ final_predictions.csv NOT FOUND")

    # 5. Check run status
    print("\n5. RUN STATUS")
    print("-" * 80)
    status_path = results_dir / 'run_status.json'
    if status_path.exists():
        with open(status_path) as f:
            status = json.load(f)
        print(f"✓ Status: {status.get('status')}")
        print(f"  Final epoch: {status.get('epoch')}")
        print(f"  Global satisfied: {status.get('global_satisfied')}")
        print(f"  Local satisfied: {status.get('local_satisfied')}")
        print(f"  Details: {status.get('details')}")
    else:
        print("❌ run_status.json NOT FOUND")

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


def main():
    print("Searching for conv_1_1 results for constraint [0.5, 0.3]...")
    results_dir = find_conv_1_1_results()

    if results_dir:
        analyze_results(results_dir)
    else:
        print("\n❌ Cannot analyze - results not found")
        print("\nPlease ensure experiments have been run:")
        print("  python run_all_convergence_experiments.py")


if __name__ == '__main__':
    main()
