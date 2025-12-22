import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import numpy as np

# Assuming these imports are available as per your snippet
from model import NeuralNetClassifier
from transductive_loss import MulticlassTransductiveLoss


def train_model(X_train, y_train, groups_train, global_constraint, local_constraint,
                lambda_global, lambda_local, hidden_dims, epochs, batch_size, lr, dropout, device):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if y_train.dtype == 'O' or isinstance(y_train.iloc[0], str):
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
    else:
        y_train_encoded = y_train.values

    features = torch.FloatTensor(X_train_scaled).to(device)
    labels = torch.LongTensor(y_train_encoded).to(device)
    group_ids = torch.LongTensor(groups_train.values).to(device)
    model = NeuralNetClassifier(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        n_classes=3,
        dropout=dropout
    ).to(device)

    # 4. Loss Function
    # Ensure constraints are on the correct device inside the Loss class or passed correctly
    criterion = MulticlassTransductiveLoss(
        global_constraints=global_constraint,
        local_constraints=local_constraint,
        lambda_global=lambda_global,
        lambda_local=lambda_local,
        use_ce=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5. Full Batch Training Loop
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass on the ENTIRE dataset
        logits = model(features)

        # Calculate Loss
        loss_total, loss_ce, loss_global, loss_local = criterion(logits, labels, group_ids)

        # Backward pass
        loss_total.backward()
        optimizer.step()

        # Monitoring
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_total.item():.4f} "
                  f"(CE: {loss_ce.item():.4f}, Global: {loss_global.item():.4f}, Local: {loss_local.item():.4f})")

    training_time = time.time() - start_time

    return model, scaler, training_time


def predict(model, scaler, X_test, device):
    model.eval()
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

    with torch.no_grad():
        logits = model(X_test_tensor)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return predictions


def evaluate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
