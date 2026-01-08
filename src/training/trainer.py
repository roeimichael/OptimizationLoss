import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from src.models import get_model
from src.losses import MulticlassTransductiveLoss
from src.training.metrics import compute_train_accuracy, compute_prediction_statistics
from src.training.logging import log_progress_to_csv, print_progress


class ConstraintTrainer:
    def __init__(self, config: Dict[str, Any], experiment_path: str, device: torch.device):
        self.config = config
        self.hyperparams = config['hyperparams']
        self.experiment_path = Path(experiment_path)
        self.device = device
        self.csv_log_path = self.experiment_path / 'training_log.csv'

        # Initialize placeholders
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion_ce = nn.CrossEntropyLoss()

    def setup_model(self, input_dim: int, base_model_id: str) -> None:
        self.model = self._load_from_cache(base_model_id, input_dim)
        if self.model is None:
            print(f"[INIT] Creating new model: {self.config['model_name']}")
            self.model = get_model(
                self.config['model_name'],
                input_dim=input_dim,
                hidden_dims=self.hyperparams['hidden_dims'],
                n_classes=3,
                dropout=self.hyperparams['dropout']
            ).to(self.device)
            self.from_cache = False
        else:
            print(f"[CACHE] Loaded model: {base_model_id}")
            self.from_cache = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams['lr'])

    def _create_dataloader(self, X: torch.Tensor, y: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)

    def train_warmup(self, X_train: torch.Tensor, y_train: torch.Tensor, base_model_id: str) -> None:
        if self.from_cache:
            return

        print("\n" + "=" * 80)
        print("WARMUP TRAINING")
        print("=" * 80)

        train_loader = self._create_dataloader(X_train, y_train)
        warmup_epochs = self.hyperparams['warmup_epochs']

        for epoch in range(warmup_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion_ce(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 1 == 0:
                avg_loss = epoch_loss / len(train_loader)
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                log_progress_to_csv(str(self.csv_log_path), epoch, avg_loss, train_acc)

                if (epoch + 1) % 50 == 0:
                    print(f"Warmup Epoch {epoch + 1}/{warmup_epochs}: Loss={avg_loss:.4f}, Acc={train_acc:.4f}")

        self._save_to_cache(base_model_id)

    def train_constraints(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor,
                          groups_test: pd.Series, global_con: list, local_con: Dict[int, list]) -> nn.Module:

        print("\n" + "=" * 80)
        print("CONSTRAINT OPTIMIZATION TRAINING")
        print("=" * 80)

        train_loader = self._create_dataloader(X_train, y_train)
        X_test = X_test.to(self.device)
        group_ids = torch.LongTensor(groups_test.values).to(self.device)

        criterion_constraint = MulticlassTransductiveLoss(
            global_constraints=global_con,
            local_constraints=local_con,
            lambda_global=self.hyperparams['lambda_global'],
            lambda_local=self.hyperparams['lambda_local'],
            use_ce=False
        ).to(self.device)

        warmup_epochs = self.hyperparams['warmup_epochs']
        total_epochs = self.hyperparams['epochs']
        threshold = self.hyperparams['constraint_threshold']
        step = self.hyperparams['lambda_step']

        for epoch in range(warmup_epochs, total_epochs):
            self.model.train()
            epoch_ce_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                train_logits = self.model(batch_X)
                loss_ce = self.criterion_ce(train_logits, batch_y)
                self.model.eval()
                test_logits = self.model(X_test)
                self.model.train()
                _, _, loss_global, loss_local = criterion_constraint(test_logits, y_true=None, group_ids=group_ids)
                loss = loss_ce + (criterion_constraint.lambda_global * loss_global) + \
                       (criterion_constraint.lambda_local * loss_local)
                loss.backward()
                self.optimizer.step()
                epoch_ce_loss += loss_ce.item()

            avg_ce = epoch_ce_loss / len(train_loader)
            avg_global = loss_global.item()
            avg_local = loss_local.item()

            if avg_global > threshold:
                criterion_constraint.set_lambda(lambda_global=criterion_constraint.lambda_global + step)
            if avg_local > threshold:
                criterion_constraint.set_lambda(lambda_local=criterion_constraint.lambda_local + step)

            if (epoch + 1) % 3 == 0 or (epoch + 1) == warmup_epochs + 1:
                train_acc = compute_train_accuracy(self.model, train_loader, self.device)
                g_counts, l_counts, g_soft, l_soft = compute_prediction_statistics(self.model, X_test, group_ids)

                log_progress_to_csv(
                    str(self.csv_log_path), epoch, avg_ce, train_acc, avg_global, avg_local,
                    g_counts, l_counts, g_soft, l_soft,
                    criterion_constraint.lambda_global, criterion_constraint.lambda_local,
                    global_con, criterion_constraint.global_constraints_satisfied,
                    criterion_constraint.local_constraints_satisfied
                )
                print_progress(
                    epoch, avg_ce, avg_global, avg_local, criterion_constraint.lambda_global,
                    criterion_constraint.lambda_local, train_acc, g_counts, g_soft, global_con,
                    criterion_constraint.global_constraints_satisfied, criterion_constraint.local_constraints_satisfied
                )
            if avg_global <= threshold and avg_local <= threshold:
                print(f"\nALL CONSTRAINTS SATISFIED at epoch {epoch + 1}!")
                break

        return self.model

    def _get_cache_path(self, base_model_id: str) -> Path:
        cache_dir = Path('model_cache')
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{base_model_id}.pt"

    def _save_to_cache(self, base_model_id: str) -> None:
        path = self._get_cache_path(base_model_id)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'base_model_id': base_model_id,
            'config': self.config,
            'saved_at': time.strftime('%Y-%m-%d')
        }, path)
        print(f"[CACHE] Model saved: {base_model_id}")

    def _load_from_cache(self, base_model_id: str, input_dim: int) -> Optional[nn.Module]:
        path = self._get_cache_path(base_model_id)
        if not path.exists(): return None
        try:
            ckpt = torch.load(path, map_location=self.device)
            if ckpt['base_model_id'] != base_model_id: return None

            model = get_model(self.config['model_name'], input_dim, self.hyperparams['hidden_dims'], 3,
                              self.hyperparams['dropout']).to(self.device)
            model.load_state_dict(ckpt['model_state_dict'])
            return model
        except:
            return None
