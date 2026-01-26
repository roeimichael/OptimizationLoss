"""
Comprehensive Experiment Results Summary Generator

This script extracts and summarizes ALL experiment results from the research project.
Creates a single CSV file with all experiments for easy analysis.
"""
import os
import json
import pandas as pd
from pathlib import Path

def parse_eval_metrics(eval_file):
    """Parse evaluation metrics CSV file."""
    if not os.path.exists(eval_file):
        return {}

    eval_dict = {}
    with open(eval_file, 'r') as f:
        for line in f:
            if 'Confusion Matrix' in line or line.strip() == '':
                continue
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    metric = parts[0].strip()
                    value = parts[1].strip()
                    if value:
                        try:
                            eval_dict[metric] = float(value)
                        except ValueError:
                            continue
    return eval_dict

def extract_experiment_info(config_path):
    """Extract key information from a single experiment."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    exp_dir = os.path.dirname(config_path)
    eval_file = os.path.join(exp_dir, 'evaluation_metrics.csv')

    # Parse evaluation metrics
    metrics = parse_eval_metrics(eval_file)

    # Extract path components to identify experiment type
    path_parts = Path(config_path).parts

    # Determine experiment category
    if 'longer_saturation' in path_parts:
        exp_category = 'longer_saturation'
    elif 'our_approach' in path_parts:
        exp_category = 'our_approach'
    elif 'saturated_approach' in path_parts:
        exp_category = 'saturated_approach'
    elif 'heuristic' in path_parts:
        exp_category = 'heuristic'
    else:
        exp_category = 'other'

    # Extract model name
    model_name = config.get('model_name', 'Unknown')

    # Extract constraint pair
    constraint = config.get('constraint', [None, None])
    constraint_local = constraint[0] if len(constraint) > 0 else None
    constraint_global = constraint[1] if len(constraint) > 1 else None

    # Extract hyperparameters
    hyperparams = config.get('hyperparams', {})

    # Extract results
    results = config.get('results', {})

    # Build experiment record
    record = {
        # Experiment identifiers
        'exp_category': exp_category,
        'exp_path': str(Path(config_path).parent),
        'model_name': model_name,
        'constraint_local': constraint_local,
        'constraint_global': constraint_global,

        # Hyperparameters
        'learning_rate': hyperparams.get('lr', None),
        'batch_size': hyperparams.get('batch_size', None),
        'epochs': hyperparams.get('epochs', None),
        'warmup_epochs': hyperparams.get('warmup_epochs', None),
        'lambda_global_init': hyperparams.get('lambda_global', None),
        'lambda_local_init': hyperparams.get('lambda_local', None),
        'lambda_step': hyperparams.get('lambda_step', None),
        'lambda_strategy': hyperparams.get('lambda_strategy', None),
        'constraint_threshold': hyperparams.get('constraint_threshold', None),

        # Convergence parameters (if present)
        'convergence_window': hyperparams.get('convergence_window', None),
        'convergence_required': hyperparams.get('convergence_required', None),

        # Model architecture (if present)
        'hidden_dims': str(hyperparams.get('hidden_dims', None)),
        'dropout': hyperparams.get('dropout', None),

        # Status
        'status': config.get('status', 'unknown'),

        # Results from config.json
        'used_cached_model': results.get('used_cached_model', None),
        'training_time': results.get('training_time', None),

        # Evaluation metrics from evaluation_metrics.csv
        'accuracy': metrics.get('Overall Accuracy', None),
        'precision_macro': metrics.get('Precision (Macro)', None),
        'recall_macro': metrics.get('Recall (Macro)', None),
        'f1_macro': metrics.get('F1-Score (Macro)', None),

        # Per-class metrics (if available)
        'precision_dropout': metrics.get('Precision_Dropout', None),
        'recall_dropout': metrics.get('Recall_Dropout', None),
        'f1_dropout': metrics.get('F1_Dropout', None),
        'precision_enrolled': metrics.get('Precision_Enrolled', None),
        'recall_enrolled': metrics.get('Recall_Enrolled', None),
        'f1_enrolled': metrics.get('F1_Enrolled', None),
        'precision_graduate': metrics.get('Precision_Graduate', None),
        'recall_graduate': metrics.get('Recall_Graduate', None),
        'f1_graduate': metrics.get('F1_Graduate', None),
    }

    return record

def main():
    print("="*100)
    print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
    print("="*100)

    # Find all config.json files
    print("\nSearching for experiment config files...")
    config_files = []
    for root, dirs, files in os.walk('results'):
        if 'config.json' in files:
            config_path = os.path.join(root, 'config.json')
            config_files.append(config_path)

    print(f"Found {len(config_files)} experiments")

    # Extract information from all experiments
    print("\nExtracting experiment data...")
    all_records = []
    failed_count = 0

    for i, config_path in enumerate(config_files):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(config_files)} experiments...")

        try:
            record = extract_experiment_info(config_path)
            all_records.append(record)
        except Exception as e:
            print(f"  WARNING: Failed to process {config_path}: {e}")
            failed_count += 1

    print(f"Successfully processed {len(all_records)} experiments")
    if failed_count > 0:
        print(f"Failed to process {failed_count} experiments")

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Sort by category, model, constraint, then learning rate
    df = df.sort_values([
        'exp_category',
        'model_name',
        'constraint_local',
        'constraint_global',
        'learning_rate',
        'lambda_strategy'
    ])

    # Save to CSV
    output_file = 'experiment_results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved summary to: {output_file}")

    # Print statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)

    print(f"\nTotal experiments: {len(df)}")

    print("\nBy category:")
    for cat in df['exp_category'].unique():
        count = len(df[df['exp_category'] == cat])
        print(f"  {cat}: {count}")

    print("\nBy model:")
    for model in df['model_name'].unique():
        count = len(df[df['model_name'] == model])
        print(f"  {model}: {count}")

    print("\nBy constraint pair:")
    for _, row in df[['constraint_local', 'constraint_global']].drop_duplicates().iterrows():
        cl, cg = row['constraint_local'], row['constraint_global']
        count = len(df[(df['constraint_local'] == cl) & (df['constraint_global'] == cg)])
        print(f"  [{cl}, {cg}]: {count}")

    print("\nBy lambda strategy:")
    for strategy in df['lambda_strategy'].unique():
        if pd.notna(strategy):
            count = len(df[df['lambda_strategy'] == strategy])
            print(f"  {strategy}: {count}")

    print("\nBy status:")
    for status in df['status'].unique():
        count = len(df[df['status'] == status])
        print(f"  {status}: {count}")

    # Convergence test statistics
    convergence_df = df[df['convergence_window'].notna()]
    if len(convergence_df) > 0:
        print(f"\nConvergence tests: {len(convergence_df)}")
        print(f"  Converged: {len(convergence_df[convergence_df['status'] == 'converged'])}")
        print(f"  Failed: {len(convergence_df[convergence_df['status'] == 'failed'])}")

    # Accuracy statistics
    df_with_acc = df[df['accuracy'].notna()]
    if len(df_with_acc) > 0:
        print(f"\nAccuracy statistics ({len(df_with_acc)} experiments with results):")
        print(f"  Mean: {df_with_acc['accuracy'].mean():.4f}")
        print(f"  Median: {df_with_acc['accuracy'].median():.4f}")
        print(f"  Min: {df_with_acc['accuracy'].min():.4f}")
        print(f"  Max: {df_with_acc['accuracy'].max():.4f}")

        best_exp = df_with_acc.loc[df_with_acc['accuracy'].idxmax()]
        print(f"\n  Best accuracy: {best_exp['accuracy']:.4f}")
        print(f"    Model: {best_exp['model_name']}")
        print(f"    Constraint: [{best_exp['constraint_local']}, {best_exp['constraint_global']}]")
        print(f"    LR: {best_exp['learning_rate']}, Strategy: {best_exp['lambda_strategy']}")

    print("\n" + "="*100)
    print("COMPLETE!")
    print("="*100)
    print(f"\nSummary saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

if __name__ == '__main__':
    main()
