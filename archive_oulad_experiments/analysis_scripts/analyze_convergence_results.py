"""
Analyze convergence test results for constraint 0.9_0.8
"""
import os
import json
import pandas as pd

results_dir = "results/longer_saturation/TabularResNet/constraint_0.9_0.8/convergence_test"

# Get all experiment directories
experiments = sorted([d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))])

print("=" * 100)
print("CONVERGENCE TEST ANALYSIS - Constraint [0.9, 0.8]")
print("=" * 100)
print(f"\nTotal experiments: {len(experiments)}")
print(f"Experiments: {', '.join(experiments)}\n")

results = []

for exp in experiments:
    exp_path = os.path.join(results_dir, exp)

    # Parse window and required from directory name
    parts = exp.replace('conv_', '').split('_')
    window = int(parts[0])
    required = int(parts[1])

    # Read run_status.json
    status_file = os.path.join(exp_path, 'run_status.json')
    if not os.path.exists(status_file):
        continue

    with open(status_file, 'r') as f:
        status = json.load(f)

    # Read training_log.csv to get final epoch details
    log_file = os.path.join(exp_path, 'training_log.csv')
    if not os.path.exists(log_file):
        continue

    log_df = pd.read_csv(log_file)

    # Get last row
    last_row = log_df.iloc[-1]

    # Read evaluation metrics
    eval_file = os.path.join(exp_path, 'evaluation_metrics.csv')
    # Read until we hit the confusion matrix section
    eval_lines = []
    with open(eval_file, 'r') as f:
        for line in f:
            if 'Confusion Matrix' in line or line.strip() == '':
                continue
            eval_lines.append(line.strip())

    # Parse metrics
    eval_dict = {}
    for line in eval_lines[1:]:  # Skip header
        if ',' in line:
            parts = line.split(',')
            if len(parts) >= 2:
                metric = parts[0].strip()
                value = parts[1].strip()
                if value:
                    try:
                        eval_dict[metric] = float(value)
                    except ValueError:
                        # Skip non-numeric values (headers, etc)
                        continue

    test_accuracy = eval_dict.get('Overall Accuracy', 0.0)
    test_f1 = eval_dict.get('F1 (Macro)', 0.0)

    results.append({
        'Experiment': exp,
        'Window': window,
        'Required': required,
        'Status': status.get('status', 'unknown'),
        'Final_Epoch': int(last_row['Epoch']),
        'Global_Satisfied': int(last_row['Global_Satisfied']),
        'Local_Satisfied': int(last_row['Local_Satisfied']),
        'Final_Global_Loss': float(last_row['L_target_Global']),
        'Final_Local_Loss': float(last_row['L_feat_Local']),
        'Final_Lambda_Global': float(last_row['Lambda_Global']),
        'Final_Lambda_Local': float(last_row['Lambda_Local']),
        'Hard_Dropout': int(last_row['Hard_Dropout']),
        'Soft_Dropout': float(last_row['Soft_Dropout']),
        'Limit_Dropout': int(last_row['Limit_Dropout']),
        'Hard_Enrolled': int(last_row['Hard_Enrolled']),
        'Soft_Enrolled': float(last_row['Soft_Enrolled']),
        'Limit_Enrolled': int(last_row['Limit_Enrolled']),
        'Test_Accuracy': test_accuracy,
        'Test_F1_Macro': test_f1,
    })

df = pd.DataFrame(results)

# Sort by window, then required
df = df.sort_values(['Window', 'Required'])

print("=" * 100)
print("CONVERGENCE SUMMARY")
print("=" * 100)
print(f"\nTotal converged: {len(df[df['Status'] == 'converged'])}/{len(df)}")
print(f"Total failed: {len(df[df['Status'] == 'failed'])}/{len(df)}")

converged = df[df['Status'] == 'converged']
failed = df[df['Status'] == 'failed']

if len(converged) > 0:
    print(f"\n{'-' * 100}")
    print("CONVERGED EXPERIMENTS")
    print('-' * 100)
    print(converged[['Experiment', 'Window', 'Required', 'Final_Epoch', 'Global_Satisfied', 'Local_Satisfied',
                     'Final_Global_Loss', 'Final_Local_Loss', 'Test_Accuracy', 'Test_F1_Macro']].to_string(index=False))

    print(f"\n{'-' * 100}")
    print("CONVERGENCE STATISTICS")
    print('-' * 100)
    print(f"Average convergence epoch: {converged['Final_Epoch'].mean():.1f}")
    print(f"Median convergence epoch: {converged['Final_Epoch'].median():.1f}")
    print(f"Fastest convergence: {converged['Final_Epoch'].min()} epochs ({converged.loc[converged['Final_Epoch'].idxmin(), 'Experiment']})")
    print(f"Slowest convergence: {converged['Final_Epoch'].max()} epochs ({converged.loc[converged['Final_Epoch'].idxmax(), 'Experiment']})")

    print(f"\n{'-' * 100}")
    print("ACCURACY COMPARISON")
    print('-' * 100)
    print(f"Average test accuracy: {converged['Test_Accuracy'].mean():.4f}")
    print(f"Average test F1: {converged['Test_F1_Macro'].mean():.4f}")
    print(f"Best accuracy: {converged['Test_Accuracy'].max():.4f} ({converged.loc[converged['Test_Accuracy'].idxmax(), 'Experiment']})")
    print(f"Worst accuracy: {converged['Test_Accuracy'].min():.4f} ({converged.loc[converged['Test_Accuracy'].idxmin(), 'Experiment']})")

if len(failed) > 0:
    print(f"\n{'-' * 100}")
    print("FAILED EXPERIMENTS")
    print('-' * 100)
    print(failed[['Experiment', 'Window', 'Required', 'Final_Epoch', 'Global_Satisfied', 'Local_Satisfied',
                  'Final_Global_Loss', 'Final_Local_Loss']].to_string(index=False))

print(f"\n{'=' * 100}")
print("CONSTRAINT SATISFACTION ANALYSIS")
print('=' * 100)

# Check if soft predictions satisfy constraints
for _, row in df.iterrows():
    dropout_satisfied = row['Soft_Dropout'] <= row['Limit_Dropout']
    enrolled_satisfied = row['Soft_Enrolled'] <= row['Limit_Enrolled']

    print(f"\n{row['Experiment']}:")
    print(f"  Global Constraint Check:")
    print(f"    Dropout: {row['Soft_Dropout']:.2f} {'<=' if dropout_satisfied else '>'} {row['Limit_Dropout']} ({'OK' if dropout_satisfied else 'VIOLATED'})")
    print(f"    Enrolled: {row['Soft_Enrolled']:.2f} {'<=' if enrolled_satisfied else '>'} {row['Limit_Enrolled']} ({'OK' if enrolled_satisfied else 'VIOLATED'})")
    print(f"  Reported: Global_Satisfied={row['Global_Satisfied']}, Local_Satisfied={row['Local_Satisfied']}")
    print(f"  Status: {row['Status']} at epoch {row['Final_Epoch']}")

print(f"\n{'=' * 100}")
print("WINDOW STRATEGY ANALYSIS")
print('=' * 100)

# Group by window size
for window in sorted(df['Window'].unique()):
    window_df = df[df['Window'] == window]
    converged_count = len(window_df[window_df['Status'] == 'converged'])

    print(f"\nWindow={window} ({len(window_df)} experiments, {converged_count} converged):")
    for _, row in window_df.iterrows():
        print(f"  {row['Experiment']}: {row['Status']} at epoch {row['Final_Epoch']}, "
              f"Acc={row['Test_Accuracy']:.4f}, F1={row['Test_F1_Macro']:.4f}")

print(f"\n{'=' * 100}")
print("RECOMMENDATION")
print('=' * 100)

if len(converged) > 0:
    # Find best strategy based on convergence speed and accuracy
    best_epoch = converged.loc[converged['Final_Epoch'].idxmin()]
    best_acc = converged.loc[converged['Test_Accuracy'].idxmax()]

    print(f"\nFastest convergence: {best_epoch['Experiment']} (epoch {best_epoch['Final_Epoch']})")
    print(f"  - Accuracy: {best_epoch['Test_Accuracy']:.4f}")
    print(f"  - F1 Score: {best_epoch['Test_F1_Macro']:.4f}")

    print(f"\nBest accuracy: {best_acc['Experiment']} (Acc={best_acc['Test_Accuracy']:.4f})")
    print(f"  - Converged at epoch: {best_acc['Final_Epoch']}")
    print(f"  - F1 Score: {best_acc['Test_F1_Macro']:.4f}")

    # Analyze if window/required makes a difference
    print(f"\nWindow/Required Impact:")
    print(f"  Small windows (1-5): avg epoch = {df[df['Window'] <= 5]['Final_Epoch'].mean():.1f}")
    print(f"  Medium windows (10-20): avg epoch = {df[(df['Window'] >= 10) & (df['Window'] <= 20)]['Final_Epoch'].mean():.1f}")
    print(f"  Large windows (30): avg epoch = {df[df['Window'] == 30]['Final_Epoch'].mean():.1f}")
else:
    print("\nNo experiments converged. The constraints may be too strict for soft predictions.")
