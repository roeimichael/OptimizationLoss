import torch
import torch.optim as optim
from tqdm import tqdm
from loss import TransductivePortfolioLoss


def run_validation():
    torch.manual_seed(42)

    N = 100
    M = 5
    K_TARGET = 20
    K_FEAT = 5

    sector_matrix = torch.zeros(M, N)
    stocks_per_sector = N // M
    for i in range(M):
        start_idx = i * stocks_per_sector
        end_idx = start_idx + stocks_per_sector if i < M - 1 else N
        sector_matrix[i, start_idx:end_idx] = 1.0

    loss_fn = TransductivePortfolioLoss(
        k_target=K_TARGET,
        k_feat=K_FEAT,
        sector_matrix=sector_matrix,
        lambda_target=1.0,
        lambda_sector=1.0,
        use_bce=False,
        average_sectors=True
    )

    print("="*60)
    print("TRANSDUCTIVE RATIONAL SATURATION LOSS - VALIDATION SUITE")
    print("="*60)

    y_hat_compliant = torch.zeros(N, 1)
    indices = torch.randperm(N)[:K_TARGET]
    sector_counts = torch.zeros(M)
    selected = []
    for idx in indices:
        sector_idx = idx.item() // stocks_per_sector
        if sector_idx >= M:
            sector_idx = M - 1
        if sector_counts[sector_idx] < K_FEAT:
            selected.append(idx.item())
            sector_counts[sector_idx] += 1
    for idx in selected:
        y_hat_compliant[idx, 0] = 1.0

    loss_1 = loss_fn(y_hat_compliant)
    assert loss_1.item() < 1e-5, f"Expected loss ~0, got {loss_1.item()}"
    print(f"Test 1: PASSED - Loss: {loss_1.item():.6f}")

    y_hat_global = torch.zeros(N, 1)
    num_select = K_TARGET + 10
    for i in range(min(num_select, N)):
        sector_idx = i // stocks_per_sector
        if sector_idx >= M:
            sector_idx = M - 1
        if i % M != 0:
            y_hat_global[i, 0] = 1.0

    loss_2 = loss_fn(y_hat_global)
    assert loss_2.item() > 0, f"Expected loss > 0, got {loss_2.item()}"
    print(f"Test 2: PASSED - Loss: {loss_2.item():.6f}")

    y_hat_sector = torch.zeros(N, 1)
    num_in_sector_0 = min(K_FEAT + 3, stocks_per_sector)
    for i in range(num_in_sector_0):
        y_hat_sector[i, 0] = 1.0

    loss_3 = loss_fn(y_hat_sector)
    assert loss_3.item() > 0, f"Expected loss > 0, got {loss_3.item()}"
    print(f"Test 3: PASSED - Loss: {loss_3.item():.6f}")

    y_hat_saturate = torch.ones(N, 1)
    loss_4 = loss_fn(y_hat_saturate)
    max_loss = loss_fn.lambda_target + loss_fn.lambda_sector
    assert loss_4.item() < max_loss, f"Loss exploded: {loss_4.item()}, expected < {max_loss}"
    print(f"Test 4: PASSED - Loss: {loss_4.item():.6f} (Max: {max_loss:.6f})")

    y_hat_opt = torch.rand(N, 1) * 2.0
    y_hat_opt.requires_grad_(True)
    optimizer = optim.SGD([y_hat_opt], lr=0.1)

    loss_initial = loss_fn(y_hat_opt).item()

    num_steps = 100
    for _ in tqdm(range(num_steps), desc="Test 5: Gradient Optimization"):
        optimizer.zero_grad()
        loss = loss_fn(y_hat_opt)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            y_hat_opt.clamp_(min=0.0)

    loss_final = loss_fn(y_hat_opt).item()
    assert loss_final < loss_initial, f"Optimization failed: initial={loss_initial:.6f}, final={loss_final:.6f}"
    print(f"Test 5: PASSED - Initial Loss: {loss_initial:.6f}, Final Loss: {loss_final:.6f}")

    print("="*60)
    print("ADDITIONAL TEST - Full Loss with BCE")
    print("="*60)

    loss_fn_full = TransductivePortfolioLoss(
        k_target=K_TARGET,
        k_feat=K_FEAT,
        sector_matrix=sector_matrix,
        lambda_target=1.0,
        lambda_sector=1.0,
        use_bce=True,
        average_sectors=True
    )

    y_hat_bce = torch.sigmoid(torch.randn(N, 1))
    y_hat_bce.requires_grad_(True)
    y_true = torch.randint(0, 2, (N, 1)).float()

    loss_full = loss_fn_full(y_hat_bce, y_true)
    print(f"Full Loss (BCE + Constraints): {loss_full.item():.6f}")

    loss_constraint_only = loss_fn(y_hat_bce)
    print(f"Constraint Loss Only: {loss_constraint_only.item():.6f}")

    assert loss_full.item() >= loss_constraint_only.item(), "Full loss should include BCE component"
    print("Test 6: PASSED - BCE integration validated")

    print("="*60)
    print("ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_validation()
