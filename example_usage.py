import torch
import torch.nn as nn
import torch.optim as optim
from transductive_saturation_loss import TransductivePortfolioLoss


class SimplePortfolioModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def create_sector_matrix(num_stocks, num_sectors):
    sector_matrix = torch.zeros(num_sectors, num_stocks)
    stocks_per_sector = num_stocks // num_sectors

    for i in range(num_sectors):
        start_idx = i * stocks_per_sector
        end_idx = start_idx + stocks_per_sector if i < num_sectors - 1 else num_stocks
        sector_matrix[i, start_idx:end_idx] = 1.0

    return sector_matrix


def main():
    print("=" * 60)
    print("TRANSDUCTIVE SATURATION LOSS - EXAMPLE USAGE")
    print("=" * 60)

    N = 100
    M = 5
    K_TARGET = 20
    K_FEAT = 5
    INPUT_DIM = 10
    BATCH_SIZE = 32
    NUM_EPOCHS = 5

    print(f"\nConfiguration:")
    print(f"  Universe Size (N): {N}")
    print(f"  Number of Sectors (M): {M}")
    print(f"  Max Portfolio Size (K_target): {K_TARGET}")
    print(f"  Max Stocks per Sector (K_feat): {K_FEAT}")
    print(f"  Input Features: {INPUT_DIM}")

    sector_matrix = create_sector_matrix(N, M)

    loss_fn = TransductivePortfolioLoss(
        k_target=K_TARGET,
        k_feat=K_FEAT,
        sector_matrix=sector_matrix,
        lambda_target=1.0,
        lambda_sector=1.0,
        use_bce=True,
        average_sectors=True
    )

    model = SimplePortfolioModel(input_dim=INPUT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\n" + "=" * 60)
    print("TRAINING SIMULATION")
    print("=" * 60)

    for epoch in range(NUM_EPOCHS):
        X_batch = torch.randn(BATCH_SIZE, N, INPUT_DIM)
        y_batch = torch.randint(0, 2, (BATCH_SIZE, N, 1)).float()

        epoch_loss = 0.0
        for i in range(BATCH_SIZE):
            optimizer.zero_grad()

            predictions = model(X_batch[i])
            loss = loss_fn(predictions, y_batch[i])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / BATCH_SIZE
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Average Loss: {avg_loss:.6f}")

    print("\n" + "=" * 60)
    print("CONSTRAINT VALIDATION ON TEST SET")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        X_test = torch.randn(N, INPUT_DIM)
        y_test_pred = model(X_test)

        total_selected = y_test_pred.sum().item()
        sector_counts = torch.matmul(sector_matrix, y_test_pred).squeeze()

        print(f"\nPredicted Portfolio:")
        print(f"  Total Stocks Selected: {total_selected:.2f} (Max: {K_TARGET})")
        print(f"  Constraint Status: {'✓ PASS' if total_selected <= K_TARGET else '✗ FAIL'}")

        print(f"\nSector Distribution:")
        for i in range(M):
            status = '✓' if sector_counts[i] <= K_FEAT else '✗'
            print(f"  Sector {i}: {sector_counts[i]:.2f} / {K_FEAT} {status}")

        constraint_loss = loss_fn(y_test_pred)
        print(f"\nConstraint Loss: {constraint_loss.item():.6f}")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
