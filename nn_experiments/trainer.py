import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import time
import numpy as np

from model import NeuralNetClassifier
from dataset import StudentDataset
from transductive_loss import MulticlassTransductiveLoss


def train_model(X_train, y_train, groups_train, global_constraint, local_constraint,
                lambda_global, lambda_local, hidden_dims, epochs, batch_size, lr, dropout, device):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    train_dataset = StudentDataset(X_train_scaled, y_train, groups_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNetClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=hidden_dims,
        n_classes=3,
        dropout=dropout
    ).to(device)

    criterion = MulticlassTransductiveLoss(
        global_constraints=global_constraint,
        local_constraints=local_constraint,
        lambda_global=lambda_global,
        lambda_local=lambda_local,
        use_ce=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            group_ids_batch = batch['group_id'].to(device)

            logits = model(features)
            loss_total, loss_ce, loss_global, loss_local = criterion(logits, labels, group_ids_batch)

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                break

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
