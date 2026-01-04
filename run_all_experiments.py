"""
Experiment Runner
Iterates through all experiment configurations and executes them
"""
import pandas as pd
from pathlib import Path
import sys

from utils.filesystem_manager import get_all_experiment_configs, is_experiment_complete
from src.data import load_presplit_data
from src.training.constraints import compute_global_constraints, compute_local_constraints
from experiments.our_approach import train_with_our_approach
from experiments.benchmark import train_with_benchmark
from experiments.train_then_optimize import train_with_train_then_optimize
from experiments.hybrid import train_with_hybrid


# Map methodology names to their training functions
METHODOLOGY_FUNCTIONS = {
    'our_approach': train_with_our_approach,
    'benchmark': train_with_benchmark,
    'train_then_optimize': train_with_train_then_optimize,
    'hybrid': train_with_hybrid
}


def load_data():
    """
    Load the student dropout dataset

    Returns:
        tuple: (X_train, X_test, y_train, y_test, train_df, test_df)
    """
    from config.experiment_config import TRAIN_PATH, TEST_PATH, TARGET_COLUMN

    print("Loading dataset...")
    X_train, X_test, y_train, y_test, train_df, test_df = load_presplit_data(
        TRAIN_PATH, TEST_PATH, TARGET_COLUMN
    )

    print(f"  Train samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")

    # Combine for constraint computation
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    groups = full_df['Course'].unique()
    print(f"  Total courses: {len(groups)}")

    return X_train, X_test, y_train, y_test, train_df, test_df, full_df


def run_single_experiment(experiment_path, config, X_train, y_train, X_test, groups_test, y_test, full_df):
    """
    Run a single experiment

    Args:
        experiment_path: Path to experiment folder
        config: Experiment configuration
        X_train, y_train: Training data
        X_test, groups_test, y_test: Test data
        full_df: Combined dataframe for constraint computation

    Returns:
        dict: Results from the experiment
    """
    print("\n" + "="*80)
    print(f"Running experiment: {experiment_path}")
    print("="*80)

    # Check if already completed
    if is_experiment_complete(experiment_path):
        print("Experiment already completed. Skipping...")
        return None

    # Get methodology function
    methodology = config['methodology']
    if methodology not in METHODOLOGY_FUNCTIONS:
        print(f"Error: Unknown methodology '{methodology}'. Skipping...")
        return None

    train_func = METHODOLOGY_FUNCTIONS[methodology]

    # Compute constraints based on config
    constraint = config['constraint']
    local_percent, global_percent = constraint

    from config.experiment_config import TARGET_COLUMN
    groups = full_df['Course'].unique()

    global_constraint = compute_global_constraints(full_df, TARGET_COLUMN, global_percent)
    local_constraint = compute_local_constraints(full_df, TARGET_COLUMN, local_percent, groups)

    print(f"Global constraint: {global_constraint}")
    print(f"Local constraints: {len(local_constraint)} courses")

    # Prepare data
    X_train_clean = X_train.drop("Course", axis=1)
    X_test_clean = X_test.drop("Course", axis=1)
    groups_test_data = X_test["Course"]

    # Run training
    try:
        results = train_func(
            config,
            X_train_clean, y_train,
            X_test_clean, groups_test_data, y_test,
            global_constraint, local_constraint
        )
        return results

    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(resume=True, max_experiments=None):
    """
    Main execution: Run all pending experiments

    Args:
        resume: If True, skip already completed experiments
        max_experiments: Maximum number of experiments to run (None = all)
    """
    print("="*80)
    print("EXPERIMENT RUNNER")
    print("="*80)
    print()

    # Load data once (shared across all experiments)
    X_train, X_test, y_train, y_test, train_df, test_df, full_df = load_data()

    # Get all experiment configurations
    print("\nScanning for experiments...")
    experiments = get_all_experiment_configs('results')

    if not experiments:
        print("No experiments found!")
        print("Run 'python generate_configs.py' first to create experiment configurations.")
        return

    print(f"Found {len(experiments)} total experiments")

    # Filter completed experiments if resuming
    if resume:
        pending_experiments = [
            (path, config) for path, config in experiments
            if not is_experiment_complete(path)
        ]
        print(f"Pending experiments: {len(pending_experiments)}")
    else:
        pending_experiments = experiments
        print("Running all experiments (including completed ones)")

    # Limit if max_experiments specified
    if max_experiments is not None:
        pending_experiments = pending_experiments[:max_experiments]
        print(f"Limiting to first {max_experiments} experiments")

    print()

    # Run experiments
    completed = 0
    failed = 0

    for i, (experiment_path, config) in enumerate(pending_experiments, 1):
        print(f"\n[{i}/{len(pending_experiments)}] Processing: {experiment_path}")

        try:
            results = run_single_experiment(
                experiment_path, config,
                X_train, y_train, X_test, X_test, y_test,
                full_df
            )

            if results is not None:
                completed += 1
            else:
                print("Experiment skipped or already completed")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break

        except Exception as e:
            print(f"Failed to run experiment: {e}")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT RUN SUMMARY")
    print("="*80)
    print(f"Total experiments found: {len(experiments)}")
    print(f"Pending experiments: {len(pending_experiments)}")
    print(f"Successfully completed: {completed}")
    print(f"Failed: {failed}")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run all experiment configurations')
    parser.add_argument('--no-resume', action='store_true',
                      help='Re-run all experiments (including completed ones)')
    parser.add_argument('--max', type=int, default=None,
                      help='Maximum number of experiments to run')

    args = parser.parse_args()

    main(resume=not args.no_resume, max_experiments=args.max)
