"""
Generate publication-ready summary tables for research paper.
Creates compact tables showing model comparisons across approaches.
"""
import pandas as pd
import numpy as np

print("Loading experiment results...")
df = pd.read_csv('experiment_results_summary.csv')

# Filter to completed experiments only
df = df[df['status'] == 'completed'].copy()
print(f"Analyzing {len(df)} completed experiments")

# Create approach mapping for cleaner names
approach_map = {
    'heuristic': 'Heuristic',
    'our_approach': 'Our Approach',
    'saturated_approach': 'Saturated',
    'longer_saturation': 'Convergence Tests'
}
df['Approach'] = df['exp_category'].map(approach_map)

# ============================================================================
# TABLE 1: Overall Model Performance Comparison (All Constraints Combined)
# ============================================================================
print("\n" + "="*80)
print("TABLE 1: Model Performance Comparison (Aggregated)")
print("="*80)

# Group by approach and model, calculate statistics
comparison = []
for approach in ['Heuristic', 'Our Approach', 'Saturated']:
    df_approach = df[df['Approach'] == approach]

    for model in df_approach['model_name'].unique():
        df_model = df_approach[df_approach['model_name'] == model]
        df_with_metrics = df_model[df_model['accuracy'].notna()]

        if len(df_with_metrics) > 0:
            comparison.append({
                'Approach': approach,
                'Model': model,
                'N': len(df_with_metrics),
                'Accuracy (Mean)': df_with_metrics['accuracy'].mean(),
                'Accuracy (Best)': df_with_metrics['accuracy'].max(),
                'F1 (Mean)': df_with_metrics['f1_macro'].mean(),
                'F1 (Best)': df_with_metrics['f1_macro'].max(),
            })

df_table1 = pd.DataFrame(comparison)
df_table1 = df_table1.sort_values(['Approach', 'Model'])

# Format for display
df_table1_display = df_table1.copy()
df_table1_display['Accuracy (Mean)'] = df_table1_display['Accuracy (Mean)'].apply(lambda x: f'{x:.4f}')
df_table1_display['Accuracy (Best)'] = df_table1_display['Accuracy (Best)'].apply(lambda x: f'{x:.4f}')
df_table1_display['F1 (Mean)'] = df_table1_display['F1 (Mean)'].apply(lambda x: f'{x:.4f}')
df_table1_display['F1 (Best)'] = df_table1_display['F1 (Best)'].apply(lambda x: f'{x:.4f}')

print("\n" + df_table1_display.to_string(index=False))

# Save
df_table1.to_csv('paper_table1_model_comparison.csv', index=False, float_format='%.4f')
print("\n✓ Saved: paper_table1_model_comparison.csv")

# ============================================================================
# TABLE 2: Performance by Constraint Pair (Best Results per Model/Approach)
# ============================================================================
print("\n" + "="*80)
print("TABLE 2: Best Performance by Constraint Pair")
print("="*80)

constraint_comparison = []
for constraint_pair in [[0.5, 0.3], [0.8, 0.2], [0.9, 0.8]]:
    cl, cg = constraint_pair
    df_constraint = df[(df['constraint_local'] == cl) & (df['constraint_global'] == cg)]

    for approach in ['Heuristic', 'Our Approach', 'Saturated']:
        df_approach = df_constraint[df_constraint['Approach'] == approach]

        if len(df_approach) > 0:
            # Get best result for this approach/constraint combination
            best_idx = df_approach['accuracy'].idxmax()
            best = df_approach.loc[best_idx]

            constraint_comparison.append({
                'Constraint': f'[{cl}, {cg}]',
                'Approach': approach,
                'Best Model': best['model_name'],
                'Accuracy': best['accuracy'],
                'F1-Score': best['f1_macro'],
                'Precision': best['precision_macro'],
                'Recall': best['recall_macro']
            })

df_table2 = pd.DataFrame(constraint_comparison)
df_table2 = df_table2.sort_values(['Constraint', 'Accuracy'], ascending=[True, False])

# Format for display
df_table2_display = df_table2.copy()
df_table2_display['Accuracy'] = df_table2_display['Accuracy'].apply(lambda x: f'{x:.4f}')
df_table2_display['F1-Score'] = df_table2_display['F1-Score'].apply(lambda x: f'{x:.4f}')
df_table2_display['Precision'] = df_table2_display['Precision'].apply(lambda x: f'{x:.4f}')
df_table2_display['Recall'] = df_table2_display['Recall'].apply(lambda x: f'{x:.4f}')

print("\n" + df_table2_display.to_string(index=False))

# Save
df_table2.to_csv('paper_table2_constraint_comparison.csv', index=False, float_format='%.4f')
print("\n✓ Saved: paper_table2_constraint_comparison.csv")

# ============================================================================
# TABLE 3: Compact Summary for Single Constraint (e.g., [0.9, 0.8])
# ============================================================================
print("\n" + "="*80)
print("TABLE 3: Detailed Comparison for Constraint [0.9, 0.8]")
print("="*80)

# Focus on [0.9, 0.8] constraint
df_focus = df[(df['constraint_local'] == 0.9) & (df['constraint_global'] == 0.8)].copy()

focus_comparison = []
for approach in ['Heuristic', 'Our Approach', 'Saturated']:
    df_approach = df_focus[df_focus['Approach'] == approach]

    if len(df_approach) > 0:
        for model in df_approach['model_name'].unique():
            df_model = df_approach[df_approach['model_name'] == model]
            df_with_metrics = df_model[df_model['accuracy'].notna()]

            if len(df_with_metrics) > 0:
                best_idx = df_with_metrics['accuracy'].idxmax()
                best = df_with_metrics.loc[best_idx]

                focus_comparison.append({
                    'Approach': approach,
                    'Model': model,
                    'Accuracy': best['accuracy'],
                    'Precision': best['precision_macro'],
                    'Recall': best['recall_macro'],
                    'F1-Score': best['f1_macro'],
                    'Best Config': f"lr={best['learning_rate']}, λ-{best['lambda_strategy']}"
                })

df_table3 = pd.DataFrame(focus_comparison)
df_table3 = df_table3.sort_values('Accuracy', ascending=False)

# Format for display
df_table3_display = df_table3.copy()
df_table3_display['Accuracy'] = df_table3_display['Accuracy'].apply(lambda x: f'{x:.4f}')
df_table3_display['Precision'] = df_table3_display['Precision'].apply(lambda x: f'{x:.4f}')
df_table3_display['Recall'] = df_table3_display['Recall'].apply(lambda x: f'{x:.4f}')
df_table3_display['F1-Score'] = df_table3_display['F1-Score'].apply(lambda x: f'{x:.4f}')

print("\n" + df_table3_display.to_string(index=False))

# Save
df_table3.to_csv('paper_table3_constraint_0.9_0.8.csv', index=False, float_format='%.4f')
print("\n✓ Saved: paper_table3_constraint_0.9_0.8.csv")

# ============================================================================
# TABLE 4: Ultra-Compact Summary (Single Constraint, Best Results Only)
# ============================================================================
print("\n" + "="*80)
print("TABLE 4: Ultra-Compact Summary (Best per Approach)")
print("="*80)

ultra_compact = []
for constraint_pair in [[0.9, 0.8]]:  # Can change to any constraint
    cl, cg = constraint_pair
    df_constraint = df[(df['constraint_local'] == cl) & (df['constraint_global'] == cg)]

    for approach in ['Heuristic', 'Our Approach', 'Saturated']:
        df_approach = df_constraint[df_constraint['Approach'] == approach]

        if len(df_approach) > 0:
            best_idx = df_approach['accuracy'].idxmax()
            best = df_approach.loc[best_idx]

            ultra_compact.append({
                'Method': approach,
                'Accuracy': best['accuracy'],
                'Precision': best['precision_macro'],
                'Recall': best['recall_macro'],
                'F1': best['f1_macro']
            })

df_table4 = pd.DataFrame(ultra_compact)
df_table4 = df_table4.sort_values('Accuracy', ascending=False)

# Format for display
df_table4_display = df_table4.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1']:
    df_table4_display[col] = df_table4_display[col].apply(lambda x: f'{x:.4f}')

print("\n" + df_table4_display.to_string(index=False))

# Save
df_table4.to_csv('paper_table4_ultra_compact.csv', index=False, float_format='%.4f')
print("\n✓ Saved: paper_table4_ultra_compact.csv")

# ============================================================================
# Generate LaTeX versions
# ============================================================================
print("\n" + "="*80)
print("GENERATING LATEX VERSIONS")
print("="*80)

# LaTeX Table 4 (Ultra-Compact)
latex_table4 = df_table4.to_latex(
    index=False,
    float_format='%.4f',
    caption='Performance comparison across constraint satisfaction approaches on the OULAD dataset (Constraint: [0.9, 0.8]).',
    label='tab:results_comparison',
    position='htbp'
)

with open('paper_table4_latex.tex', 'w') as f:
    f.write(latex_table4)
print("\n✓ Saved: paper_table4_latex.tex")

# LaTeX Table 3 (Detailed)
latex_table3 = df_table3[['Approach', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_latex(
    index=False,
    float_format='%.4f',
    caption='Detailed performance comparison by model architecture (Constraint: [0.9, 0.8]).',
    label='tab:detailed_results',
    position='htbp'
)

with open('paper_table3_latex.tex', 'w') as f:
    f.write(latex_table3)
print("✓ Saved: paper_table3_latex.tex")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS FOR PAPER")
print("="*80)

stats = {
    'Total experiments': len(df),
    'Models tested': len(df['model_name'].unique()),
    'Constraint pairs tested': len(df[['constraint_local', 'constraint_global']].drop_duplicates()),
    'Lambda strategies tested': len(df['lambda_strategy'].dropna().unique()),
    'Learning rates tested': len(df['learning_rate'].dropna().unique()),
    'Best overall accuracy': df['accuracy'].max(),
    'Mean accuracy (all experiments)': df['accuracy'].mean(),
}

print()
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# Save stats
with open('paper_summary_stats.txt', 'w') as f:
    f.write("EXPERIMENT SUMMARY STATISTICS\n")
    f.write("="*60 + "\n\n")
    for key, value in stats.items():
        if isinstance(value, float):
            f.write(f"{key}: {value:.4f}\n")
        else:
            f.write(f"{key}: {value}\n")

print("\n✓ Saved: paper_summary_stats.txt")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. paper_table1_model_comparison.csv - Aggregated model comparison")
print("  2. paper_table2_constraint_comparison.csv - Best by constraint pair")
print("  3. paper_table3_constraint_0.9_0.8.csv - Detailed for one constraint")
print("  4. paper_table4_ultra_compact.csv - Ultra-compact for paper (RECOMMENDED)")
print("  5. paper_table3_latex.tex - LaTeX version of Table 3")
print("  6. paper_table4_latex.tex - LaTeX version of Table 4 (RECOMMENDED)")
print("  7. paper_summary_stats.txt - Key statistics")
print("\nRecommendation: Use Table 4 (Ultra-Compact) for your paper!")
