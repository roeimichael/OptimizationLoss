"""Per-cell paired Wilcoxon signed-rank tests and binomial consistency tests
for TraLO vs each baseline on F1m and Flips. Outputs LaTeX-ready summary."""
import pandas as pd
from scipy.stats import wilcoxon, binomtest

df = pd.read_csv("docs/table_a_per_seed.csv")
methods = ["tralo_bounded", "fioretto_ldf", "hounie_rcl", "danits_lp", "heuristic"]
metrics = {"F1m": "higher_better", "Flips": "lower_better"}

cells = df.groupby(["ds", "tight"])

summary = []
for metric, direction in metrics.items():
    for m in methods:
        wins, ties, losses, sig_wins = 0, 0, 0, 0
        per_cell_p = []
        for (ds, tight), g in cells:
            tra = g[f"tralo__{metric}"].values
            base = g[f"{m}__{metric}"].values
            diff = tra - base
            if direction == "higher_better":
                better = diff > 0
            else:
                better = diff < 0
            mean_tra, mean_base = tra.mean(), base.mean()
            if direction == "higher_better":
                if mean_tra > mean_base + 1e-6: outcome = "win"
                elif mean_tra < mean_base - 1e-6: outcome = "loss"
                else: outcome = "tie"
            else:
                if mean_tra < mean_base - 1e-6: outcome = "win"
                elif mean_tra > mean_base + 1e-6: outcome = "loss"
                else: outcome = "tie"
            # Wilcoxon
            if (diff == 0).all():
                p = 1.0
            else:
                try:
                    p = wilcoxon(tra, base, zero_method="wilcox").pvalue
                except ValueError:
                    p = 1.0
            per_cell_p.append((ds, tight, outcome, mean_tra, mean_base, p))
            if outcome == "win":
                wins += 1
                if p < 0.10: sig_wins += 1
            elif outcome == "loss": losses += 1
            else: ties += 1
        # Binomial consistency test on wins+ties vs losses (15 cells)
        non_losses = wins + ties
        binom = binomtest(non_losses, n=15, p=0.5, alternative="greater").pvalue
        summary.append((metric, m, wins, ties, losses, sig_wins, binom))

print(f"{'metric':<8}{'baseline':<18}{'win':>4}{'tie':>4}{'loss':>5}{'sig_wins':>10}{'binom_p':>10}")
for r in summary:
    metric, m, wins, ties, losses, sw, bp = r
    print(f"{metric:<8}{m:<18}{wins:>4}{ties:>4}{losses:>5}{sw:>10}{bp:>10.4g}")

# Cell-level p<0.10 count, ALSO check pre-rounding "win" semantics
print("\nNotes: Wilcoxon with n=4 paired observations has minimum p=0.125, so")
print("'sig_wins' counts cells where p<0.10. With only 4 seeds, expect this to be")
print("dominated by cells with monotone differences across all 4 seeds.")
