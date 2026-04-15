"""Show gradient ratio evolution for our_approach diagnostics."""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'results/pending_runs/mar27_single_GE/L50_G50/MobileNetV3/our_approach/slice_1/diagnostics/epoch_diagnostics.csv'
df = pd.read_csv(path)
print(f'Epochs tracked: {len(df)}')
print()
if len(df) == 0:
    sys.exit(0)

print('=== GRADIENT RATIO EVOLUTION ===')
print('{:>6} {:>10} {:>12} {:>8} {:>10} {:>10} {:>10} {:>10}'.format(
    'Epoch', 'CE_grad', 'Constr_grad', 'Ratio', 'Satisfied', 'Lambda_G', 'Lambda_L', 'Rho'))

n = len(df)
if n > 10:
    indices = [0, n//8, n//4, 3*n//8, n//2, 5*n//8, 3*n//4, 7*n//8, n-1]
else:
    indices = list(range(n))
seen = set()
for idx in indices:
    if idx in seen or idx >= n:
        continue
    seen.add(idx)
    r = df.iloc[idx]
    sat = str(r.get('is_satisfied', '?'))
    print('{:>6} {:>10.4f} {:>12.4f} {:>7.2f}x {:>10} {:>10.4f} {:>10.4f} {:>10.4f}'.format(
        int(r['epoch']), r['ce_grad_norm'], r['constraint_grad_norm'], r['grad_ratio'],
        sat, r.get('lambda_g', 0), r.get('lambda_l', 0), r.get('rho', 0)))

# Find the moment CE skip kicked in (if any)
if 'ce_loss' in df.columns:
    ce_near_zero = df[df['ce_loss'] < 0.001]
    if len(ce_near_zero) > 0:
        first = ce_near_zero.iloc[0]
        print()
        print('First epoch with ce_loss < 0.001: epoch {}'.format(int(first['epoch'])))

# Find lambda zeroing event (only happens when toggle is active AND satisfied)
if 'lambda_g' in df.columns:
    lambda_zeroed = df[(df['lambda_g'] < 0.0001) & (df['epoch'] > 50)]
    if len(lambda_zeroed) > 0:
        first = lambda_zeroed.iloc[0]
        print('First epoch with lambda_g ~0: epoch {} (TOGGLE FIRED)'.format(int(first['epoch'])))
    else:
        print('Lambda never toggled to 0 (never satisfied, or disable_lambda_toggle=True)')

# Weight drift
if 'weight_drift_l2' in df.columns:
    drift_rows = df[df['weight_drift_l2'].notna()]
    if len(drift_rows) > 0:
        print()
        print('=== WEIGHT DRIFT ===')
        for idx in [0, len(drift_rows)//2, len(drift_rows)-1]:
            r = drift_rows.iloc[idx]
            print('  epoch {:>3}: L2 drift from warmup = {:.4f}'.format(
                int(r['epoch']), r['weight_drift_l2']))
