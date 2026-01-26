"""
Create organized experiment summary with multiple sheets for easy analysis.
"""
import pandas as pd

print("Loading experiment results summary...")
df = pd.read_csv('experiment_results_summary.csv')

print(f"Total experiments: {len(df)}")

# Define key columns for different views
id_columns = [
    'exp_category',
    'model_name',
    'constraint_local',
    'constraint_global',
    'learning_rate',
    'lambda_strategy',
    'convergence_window',
    'convergence_required'
]

metric_columns = [
    'status',
    'accuracy',
    'f1_macro',
    'precision_macro',
    'recall_macro',
    'training_time'
]

hyperparam_columns = [
    'learning_rate',
    'batch_size',
    'epochs',
    'warmup_epochs',
    'lambda_global_init',
    'lambda_local_init',
    'lambda_step',
    'lambda_strategy',
    'constraint_threshold',
    'hidden_dims',
    'dropout'
]

convergence_columns = [
    'convergence_window',
    'convergence_required'
]

# Create Excel writer
output_file = 'experiment_results_organized.xlsx'
print(f"\nCreating organized Excel file: {output_file}")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

    # Sheet 1: Overview - All experiments with key columns
    print("  Creating 'Overview' sheet...")
    overview_cols = ['exp_category', 'model_name', 'constraint_local', 'constraint_global',
                     'learning_rate', 'lambda_strategy', 'status', 'accuracy', 'f1_macro',
                     'precision_macro', 'recall_macro', 'training_time']
    df_overview = df[overview_cols].copy()
    df_overview.to_excel(writer, sheet_name='Overview', index=False)

    # Sheet 2: Heuristic experiments
    print("  Creating 'Heuristic' sheet...")
    df_heuristic = df[df['exp_category'] == 'heuristic'].copy()
    if len(df_heuristic) > 0:
        cols = ['model_name', 'constraint_local', 'constraint_global', 'learning_rate',
                'lambda_strategy', 'status', 'accuracy', 'f1_macro', 'precision_macro',
                'recall_macro', 'training_time', 'exp_path']
        df_heuristic[cols].to_excel(writer, sheet_name='Heuristic', index=False)

    # Sheet 3: Our approach experiments
    print("  Creating 'Our_Approach' sheet...")
    df_our = df[df['exp_category'] == 'our_approach'].copy()
    if len(df_our) > 0:
        cols = ['model_name', 'constraint_local', 'constraint_global', 'learning_rate',
                'lambda_strategy', 'status', 'accuracy', 'f1_macro', 'precision_macro',
                'recall_macro', 'training_time', 'exp_path']
        df_our[cols].to_excel(writer, sheet_name='Our_Approach', index=False)

    # Sheet 4: Saturated approach experiments
    print("  Creating 'Saturated_Approach' sheet...")
    df_sat = df[df['exp_category'] == 'saturated_approach'].copy()
    if len(df_sat) > 0:
        cols = ['model_name', 'constraint_local', 'constraint_global', 'learning_rate',
                'lambda_strategy', 'status', 'accuracy', 'f1_macro', 'precision_macro',
                'recall_macro', 'training_time', 'exp_path']
        df_sat[cols].to_excel(writer, sheet_name='Saturated_Approach', index=False)

    # Sheet 5: Convergence tests
    print("  Creating 'Convergence_Tests' sheet...")
    df_conv = df[df['exp_category'] == 'longer_saturation'].copy()
    if len(df_conv) > 0:
        cols = ['model_name', 'constraint_local', 'constraint_global', 'convergence_window',
                'convergence_required', 'status', 'accuracy', 'f1_macro', 'precision_macro',
                'recall_macro', 'training_time', 'exp_path']
        df_conv[cols].to_excel(writer, sheet_name='Convergence_Tests', index=False)

    # Sheet 6: Best results per category
    print("  Creating 'Best_Results' sheet...")
    best_results = []

    for category in df['exp_category'].unique():
        df_cat = df[df['exp_category'] == category]
        df_cat_with_acc = df_cat[df_cat['accuracy'].notna()]

        if len(df_cat_with_acc) > 0:
            for model in df_cat_with_acc['model_name'].unique():
                df_model = df_cat_with_acc[df_cat_with_acc['model_name'] == model]

                for _, constraint_row in df_model[['constraint_local', 'constraint_global']].drop_duplicates().iterrows():
                    cl, cg = constraint_row['constraint_local'], constraint_row['constraint_global']
                    df_constraint = df_model[
                        (df_model['constraint_local'] == cl) &
                        (df_model['constraint_global'] == cg)
                    ]

                    if len(df_constraint) > 0:
                        best_idx = df_constraint['accuracy'].idxmax()
                        best_row = df_constraint.loc[best_idx]
                        best_results.append({
                            'Category': category,
                            'Model': best_row['model_name'],
                            'Constraint_Local': cl,
                            'Constraint_Global': cg,
                            'Learning_Rate': best_row['learning_rate'],
                            'Lambda_Strategy': best_row['lambda_strategy'],
                            'Accuracy': best_row['accuracy'],
                            'F1_Macro': best_row['f1_macro'],
                            'Precision_Macro': best_row['precision_macro'],
                            'Recall_Macro': best_row['recall_macro'],
                            'Training_Time': best_row['training_time']
                        })

    df_best = pd.DataFrame(best_results)
    df_best = df_best.sort_values(['Category', 'Model', 'Constraint_Local', 'Constraint_Global'], ascending=[True, True, True, True])
    df_best.to_excel(writer, sheet_name='Best_Results', index=False)

    # Sheet 7: Comparison by Lambda Strategy
    print("  Creating 'Lambda_Strategy_Comparison' sheet...")
    comparison_data = []

    for strategy in df['lambda_strategy'].dropna().unique():
        df_strategy = df[df['lambda_strategy'] == strategy]
        df_strategy_with_acc = df_strategy[df_strategy['accuracy'].notna()]

        if len(df_strategy_with_acc) > 0:
            comparison_data.append({
                'Lambda_Strategy': strategy,
                'Num_Experiments': len(df_strategy_with_acc),
                'Mean_Accuracy': df_strategy_with_acc['accuracy'].mean(),
                'Median_Accuracy': df_strategy_with_acc['accuracy'].median(),
                'Std_Accuracy': df_strategy_with_acc['accuracy'].std(),
                'Min_Accuracy': df_strategy_with_acc['accuracy'].min(),
                'Max_Accuracy': df_strategy_with_acc['accuracy'].max(),
                'Mean_F1': df_strategy_with_acc['f1_macro'].mean(),
                'Mean_Training_Time': df_strategy_with_acc['training_time'].mean()
            })

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Mean_Accuracy', ascending=False)
    df_comparison.to_excel(writer, sheet_name='Lambda_Strategy_Comp', index=False)

    # Sheet 8: Comparison by Learning Rate
    print("  Creating 'Learning_Rate_Comparison' sheet...")
    lr_comparison = []

    for lr in df['learning_rate'].dropna().unique():
        df_lr = df[df['learning_rate'] == lr]
        df_lr_with_acc = df_lr[df_lr['accuracy'].notna()]

        if len(df_lr_with_acc) > 0:
            lr_comparison.append({
                'Learning_Rate': lr,
                'Num_Experiments': len(df_lr_with_acc),
                'Mean_Accuracy': df_lr_with_acc['accuracy'].mean(),
                'Median_Accuracy': df_lr_with_acc['accuracy'].median(),
                'Std_Accuracy': df_lr_with_acc['accuracy'].std(),
                'Min_Accuracy': df_lr_with_acc['accuracy'].min(),
                'Max_Accuracy': df_lr_with_acc['accuracy'].max(),
                'Mean_F1': df_lr_with_acc['f1_macro'].mean(),
                'Mean_Training_Time': df_lr_with_acc['training_time'].mean()
            })

    df_lr_comp = pd.DataFrame(lr_comparison)
    df_lr_comp = df_lr_comp.sort_values('Mean_Accuracy', ascending=False)
    df_lr_comp.to_excel(writer, sheet_name='Learning_Rate_Comp', index=False)

    # Sheet 9: Full data (all columns)
    print("  Creating 'Full_Data' sheet...")
    df.to_excel(writer, sheet_name='Full_Data', index=False)

print("\n✓ Created organized Excel file with 9 sheets:")
print("  1. Overview - Key metrics for all experiments")
print("  2. Heuristic - Heuristic approach experiments")
print("  3. Our_Approach - Our approach experiments")
print("  4. Saturated_Approach - Saturated approach experiments")
print("  5. Convergence_Tests - Sustained convergence tests")
print("  6. Best_Results - Best result per category/model/constraint")
print("  7. Lambda_Strategy_Comp - Comparison by lambda strategy")
print("  8. Learning_Rate_Comp - Comparison by learning rate")
print("  9. Full_Data - All data with all columns")

print(f"\n✓ File saved: {output_file}")
print("\nYou can now open this Excel file to analyze your results.")
print("Each sheet is organized for specific analysis tasks.")
