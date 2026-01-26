"""
Create ultra-compact comparison table for paper showing all constraints
with clear indicators of where Our Approach wins.
"""
import pandas as pd

# Load comparison data
df = pd.read_csv('paper_comparison_vs_baselines.csv')

print("="*80)
print("COMPACT COMPARISON TABLE FOR PAPER")
print("="*80)

# Create compact version
compact = []
for _, row in df.iterrows():
    # Mark winning method with bold indicator
    ext_baseline = row['External_Baseline']
    heuristic = row['Heuristic']
    our_approach = row['Our_Approach']

    # Determine winner
    if our_approach > heuristic and our_approach > ext_baseline:
        winner = 'Our Approach ✓✓'
    elif heuristic > our_approach and heuristic > ext_baseline:
        winner = 'Heuristic'
    else:
        winner = 'Mixed'

    compact.append({
        'Constraint': row['Constraint'],
        'External Baseline': f"{ext_baseline:.4f}",
        'Heuristic': f"{heuristic:.4f}",
        'Our Approach': f"{our_approach:.4f}",
        'Best Method': winner
    })

df_compact = pd.DataFrame(compact)

print("\n" + df_compact.to_string(index=False))

# Save
df_compact.to_csv('paper_table_all_constraints_compact.csv', index=False)
print("\n✓ Saved: paper_table_all_constraints_compact.csv")

# Create LaTeX with highlighting
print("\n" + "="*80)
print("LATEX VERSION WITH HIGHLIGHTING")
print("="*80)

# Manual LaTeX creation with bold for winners
latex_lines = []
latex_lines.append("\\begin{table}[htbp]")
latex_lines.append("\\centering")
latex_lines.append("\\caption{Performance comparison across constraint satisfaction levels. \\textbf{Bold} indicates best performance.}")
latex_lines.append("\\label{tab:all_constraints_comparison}")
latex_lines.append("\\begin{tabular}{lrrr}")
latex_lines.append("\\toprule")
latex_lines.append("Constraint & External Baseline & Heuristic & Our Approach \\\\")
latex_lines.append("\\midrule")

for _, row in df.iterrows():
    constraint = row['Constraint']
    ext = row['External_Baseline']
    heur = row['Heuristic']
    ours = row['Our_Approach']

    # Determine which to bold
    if ours > heur and ours > ext:
        # Our approach wins
        latex_lines.append(f"{constraint} & {ext:.4f} & {heur:.4f} & \\textbf{{{ours:.4f}}} \\\\")
    elif heur > ours and heur > ext:
        # Heuristic wins
        latex_lines.append(f"{constraint} & {ext:.4f} & \\textbf{{{heur:.4f}}} & {ours:.4f} \\\\")
    else:
        # No clear winner or external wins
        if ext > heur and ext > ours:
            latex_lines.append(f"{constraint} & \\textbf{{{ext:.4f}}} & {heur:.4f} & {ours:.4f} \\\\")
        else:
            latex_lines.append(f"{constraint} & {ext:.4f} & {heur:.4f} & {ours:.4f} \\\\")

latex_lines.append("\\bottomrule")
latex_lines.append("\\end{tabular}")
latex_lines.append("\\end{table}")

latex_content = '\n'.join(latex_lines)

with open('paper_table_all_constraints_latex.tex', 'w') as f:
    f.write(latex_content)

print("\n✓ Saved: paper_table_all_constraints_latex.tex")
print("\n" + latex_content)

# Create ultra-compact focused version (only wins)
print("\n" + "="*80)
print("ULTRA-COMPACT: Only Constraints Where Our Approach Wins Both")
print("="*80)

df_wins = df[df['Beats_Both'] == '✓✓'].copy()

ultra_compact = []
for _, row in df_wins.iterrows():
    ultra_compact.append({
        'Constraint': row['Constraint'],
        'Baseline': f"{row['External_Baseline']:.4f}",
        'Our Approach': f"{row['Our_Approach']:.4f}",
        'Improvement': row['Improvement_vs_External'],
        'Model': row['Our_Model']
    })

df_ultra = pd.DataFrame(ultra_compact)

if len(df_ultra) > 0:
    print("\n" + df_ultra.to_string(index=False))

    df_ultra.to_csv('paper_table_wins_only.csv', index=False)
    print("\n✓ Saved: paper_table_wins_only.csv")

    # LaTeX for wins only
    latex_ultra = df_ultra.to_latex(
        index=False,
        caption='Constraint satisfaction levels where Our Approach achieves superior performance to all baselines.',
        label='tab:wins_only',
        position='htbp',
        escape=False
    )

    with open('paper_table_wins_only_latex.tex', 'w') as f:
        f.write(latex_ultra)
    print("✓ Saved: paper_table_wins_only_latex.tex")

print("\n" + "="*80)
print("RECOMMENDATION FOR PAPER")
print("="*80)
print("\nOption 1 (Show all constraints):")
print("  Use: paper_table_all_constraints_latex.tex")
print("  Shows all 3 constraint pairs with winners bolded")
print("  Good for: Complete comparison")
print("\nOption 2 (Focus on wins only):")
print("  Use: paper_table_wins_only_latex.tex")
print("  Shows only 2 constraint pairs where you win")
print("  Good for: Highlighting your contributions")
print("\nOption 3 (Detailed wins):")
print("  Use: paper_table_our_wins_latex.tex")
print("  Shows detailed metrics for winning cases")
print("  Good for: Technical papers with space")
