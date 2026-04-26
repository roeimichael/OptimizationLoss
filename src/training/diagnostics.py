# Training diagnostics: toggleable instrumentation for constraint training.
#
# Levels:
#   0 = OFF   (no overhead, production mode)
#   1 = SUMMARY  (per-epoch aggregates: gradient norms, weight drift, borderline counts)
#   2 = DETAILED (per-sample tracking: prediction changes, probability shifts, full heatmaps)
#
# Usage in trainer.py:
#   diag = TrainingDiagnostics(level=hp.get('diagnostic_level', 0), ...)
#   diag.on_epoch_start(epoch)
#   diag.record_ce_grad(model)       # after CE backward, before step
#   diag.record_constraint_grad(model)  # after constraint backward, before step
#   diag.record_predictions(epoch, logits, X_test)
#   diag.on_epoch_end(epoch, ...)
#   diag.save_report()

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.constants import UNLIMITED

log = logging.getLogger(__name__)


class TrainingDiagnostics:
    """Collects diagnostic data during constraint training."""

    OFF = 0
    SUMMARY = 1
    DETAILED = 2

    def __init__(self, level: int, experiment_path: str, num_classes: int,
                 num_test_samples: int, constrained_classes: List[int],
                 global_con: list, local_con: dict):
        self.level = level
        self.path = Path(experiment_path)
        self.num_classes = num_classes
        self.n_test = num_test_samples
        self.constrained_classes = constrained_classes
        self.global_con = global_con
        self.local_con = local_con

        if self.level == self.OFF:
            return

        self.diag_dir = self.path / 'diagnostics'
        self.diag_dir.mkdir(exist_ok=True)

        # -- Per-epoch summary data --
        self.epoch_data = []

        # -- Warmup reference (set after warmup completes) --
        self.warmup_weights: Optional[Dict[str, torch.Tensor]] = None
        self.warmup_predictions: Optional[np.ndarray] = None  # (n_test,) argmax
        self.warmup_proba: Optional[np.ndarray] = None  # (n_test, n_classes)

        # -- Per-epoch scratch --
        self._ce_grad_norm = 0.0
        self._constraint_grad_norm = 0.0
        self._ce_grad_per_layer = {}
        self._constraint_grad_per_layer = {}

        # -- Level 2: per-sample tracking --
        if self.level >= self.DETAILED:
            # Store predictions every epoch: (epoch, n_test) -> predicted class
            self._prediction_history = []  # list of (epoch, np.ndarray)
            # Store probabilities for constrained classes every N epochs
            self._proba_history = []  # list of (epoch, np.ndarray of shape (n_test, len(constrained)))
            self._proba_sample_interval = 5  # save every 5 epochs to limit memory

        log.info("Diagnostics enabled: level=%d path=%s", self.level, self.diag_dir)

    def capture_warmup_state(self, model: torch.nn.Module, X_test: torch.Tensor,
                             device: torch.device, amp_dtype, use_amp: bool):
        """Snapshot model state right after warmup, before constraint training."""
        if self.level == self.OFF:
            return

        # Store weight snapshot (CPU, detached)
        self.warmup_weights = {
            name: param.detach().cpu().clone()
            for name, param in model.named_parameters()
        }

        # Store predictions
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            from src.utils.inference import chunked_forward
            logits = chunked_forward(model, X_test)
            proba = F.softmax(logits.float(), dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
        model.train()

        self.warmup_predictions = preds
        self.warmup_proba = proba

        # Save warmup prediction distribution
        warmup_dist = np.bincount(preds, minlength=self.num_classes)
        log.info("[DIAG] Warmup prediction distribution: %s", dict(enumerate(warmup_dist.tolist())))
        for c in self.constrained_classes:
            limit = int(self.global_con[c]) if self.global_con[c] < UNLIMITED else 'INF'
            log.info("[DIAG] Warmup class %d: %d predictions (limit=%s, excess=%s)",
                     c, warmup_dist[c], limit,
                     max(0, warmup_dist[c] - int(self.global_con[c])) if self.global_con[c] < UNLIMITED else 'N/A')

        # Save borderline analysis at warmup
        self._log_borderline_analysis(proba, "warmup")

    def _log_borderline_analysis(self, proba: np.ndarray, label: str):
        """Analyze how many samples are 'borderline' for constrained classes."""
        for c in self.constrained_classes:
            p_c = proba[:, c]
            # Borderline = probability of constrained class between 0.2 and 0.8
            borderline_mask = (p_c >= 0.2) & (p_c <= 0.8)
            near_decision = (p_c >= 0.4) & (p_c <= 0.6)
            high_confidence = p_c >= 0.8
            low_but_nonzero = (p_c >= 0.01) & (p_c < 0.2)

            log.info("[DIAG] %s class %d: borderline(0.2-0.8)=%d near_decision(0.4-0.6)=%d "
                     "high_conf(>0.8)=%d low_but_present(0.01-0.2)=%d",
                     label, c, borderline_mask.sum(), near_decision.sum(),
                     high_confidence.sum(), low_but_nonzero.sum())

    def record_ce_gradients(self, model: torch.nn.Module):
        """Call AFTER CE backward, BEFORE optimizer.step(). Records grad norms from CE."""
        if self.level == self.OFF:
            return

        total_norm_sq = 0.0
        per_layer = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach()
                norm = g.norm().item()
                total_norm_sq += norm ** 2
                per_layer[name] = norm

        self._ce_grad_norm = total_norm_sq ** 0.5
        self._ce_grad_per_layer = per_layer

    def record_constraint_gradients(self, model: torch.nn.Module):
        """Call AFTER constraint backward, BEFORE optimizer.step(). Records grad norms from constraint."""
        if self.level == self.OFF:
            return

        total_norm_sq = 0.0
        per_layer = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.detach()
                norm = g.norm().item()
                total_norm_sq += norm ** 2
                per_layer[name] = norm

        self._constraint_grad_norm = total_norm_sq ** 0.5
        self._constraint_grad_per_layer = per_layer

    def flush_summary(self):
        """Incrementally write epoch_diagnostics.csv so data survives crashes."""
        if self.level == self.OFF or not self.epoch_data:
            return
        try:
            self._save_epoch_summary()
        except Exception as e:
            log.warning("[DIAG] flush_summary failed: %s", e)

    def record_predictions(self, epoch: int, proba: np.ndarray, hard_preds: np.ndarray):
        """Record current predictions for analysis. Called once per epoch."""
        if self.level == self.OFF:
            return

        row = {
            'epoch': epoch,
            'ce_grad_norm': self._ce_grad_norm,
            'constraint_grad_norm': self._constraint_grad_norm,
            'grad_ratio': (self._constraint_grad_norm / max(self._ce_grad_norm, 1e-10)),
        }

        # Prediction distribution
        pred_dist = np.bincount(hard_preds, minlength=self.num_classes)
        for c in range(self.num_classes):
            row[f'pred_count_{c}'] = int(pred_dist[c])

        # Constrained class analysis
        for c in self.constrained_classes:
            limit = int(self.global_con[c]) if self.global_con[c] < UNLIMITED else float('inf')
            row[f'excess_{c}'] = max(0, pred_dist[c] - limit) if limit != float('inf') else 0
            row[f'soft_count_{c}'] = float(proba[:, c].sum())

            # Borderline counts
            p_c = proba[:, c]
            row[f'borderline_020_080_{c}'] = int(((p_c >= 0.2) & (p_c <= 0.8)).sum())
            row[f'near_decision_040_060_{c}'] = int(((p_c >= 0.4) & (p_c <= 0.6)).sum())
            row[f'high_conf_{c}'] = int((p_c >= 0.8).sum())

            # Mean probability of constrained class for: samples predicted as c, and not
            predicted_as_c = hard_preds == c
            row[f'mean_prob_if_predicted_{c}'] = float(p_c[predicted_as_c].mean()) if predicted_as_c.any() else 0.0
            row[f'mean_prob_if_not_predicted_{c}'] = float(p_c[~predicted_as_c].mean()) if (~predicted_as_c).any() else 0.0

        # Comparison to warmup
        if self.warmup_predictions is not None:
            changed = (hard_preds != self.warmup_predictions)
            row['changed_from_warmup'] = int(changed.sum())

            # Breakdown: how many changed TO constrained class vs AWAY from it
            for c in self.constrained_classes:
                moved_to_c = changed & (hard_preds == c) & (self.warmup_predictions != c)
                moved_from_c = changed & (hard_preds != c) & (self.warmup_predictions == c)
                row[f'moved_to_{c}'] = int(moved_to_c.sum())
                row[f'moved_from_{c}'] = int(moved_from_c.sum())
                row[f'net_movement_{c}'] = int(moved_to_c.sum()) - int(moved_from_c.sum())

        self.epoch_data.append(row)

        # Level 2: detailed per-sample tracking
        if self.level >= self.DETAILED:
            self._prediction_history.append((epoch, hard_preds.copy()))
            if epoch % self._proba_sample_interval == 0:
                constrained_proba = proba[:, self.constrained_classes].copy()
                self._proba_history.append((epoch, constrained_proba))

    def record_weight_drift(self, model: torch.nn.Module, epoch: int):
        """Compute L2 distance of current weights from warmup snapshot.
        Only runs every 10 epochs to avoid GPU->CPU memory pressure."""
        if self.level == self.OFF or self.warmup_weights is None:
            return
        if epoch % 10 != 0:
            return

        total_drift_sq = 0.0
        total_params = 0
        layer_drifts = {}

        for name, param in model.named_parameters():
            if name in self.warmup_weights:
                # Compute norm on CPU to avoid GPU memory pressure
                diff_norm = (param.detach().cpu() - self.warmup_weights[name]).norm().item()
                total_drift_sq += diff_norm ** 2
                total_params += param.numel()
                layer_drifts[name] = diff_norm

        total_drift = total_drift_sq ** 0.5

        # Add to latest epoch data
        if self.epoch_data:
            self.epoch_data[-1]['weight_drift_l2'] = total_drift
            self.epoch_data[-1]['weight_drift_per_param'] = total_drift / max(total_params, 1)

            # Top 5 drifting layers
            top_layers = sorted(layer_drifts.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (lname, ldrift) in enumerate(top_layers):
                self.epoch_data[-1][f'top_drift_layer_{i}'] = lname
                self.epoch_data[-1][f'top_drift_value_{i}'] = ldrift

    def record_loss_components(self, epoch: int, ce_loss: float, global_loss: float,
                                local_loss: float, kl_loss: float,
                                lambda_g: float, lambda_l: float, rho: float,
                                is_satisfied: bool):
        """Record loss component values for the epoch."""
        if self.level == self.OFF:
            return
        if self.epoch_data and self.epoch_data[-1]['epoch'] == epoch:
            row = self.epoch_data[-1]
            row['ce_loss'] = ce_loss
            row['global_loss_raw'] = global_loss
            row['local_loss_raw'] = local_loss
            row['kl_loss'] = kl_loss
            row['global_loss_weighted'] = lambda_g * global_loss
            row['local_loss_weighted'] = lambda_l * local_loss
            row['total_constraint_weighted'] = lambda_g * global_loss + lambda_l * local_loss
            row['lambda_g'] = lambda_g
            row['lambda_l'] = lambda_l
            row['rho'] = rho
            row['is_satisfied'] = is_satisfied

    def save_report(self):
        """Write all diagnostic data to files."""
        if self.level == self.OFF:
            return

        # 1. Summary CSV (always)
        self._save_epoch_summary()

        # 2. Detailed reports (level 2 only)
        if self.level >= self.DETAILED:
            self._save_prediction_stability()
            self._save_probability_evolution()
            self._save_sample_transition_matrix()

        log.info("[DIAG] Reports saved to %s", self.diag_dir)

    def _save_epoch_summary(self):
        """Save per-epoch diagnostic summary as CSV."""
        if not self.epoch_data:
            return

        csv_path = self.diag_dir / 'epoch_diagnostics.csv'
        # Collect all keys across all rows
        all_keys = []
        seen = set()
        for row in self.epoch_data:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore')
            writer.writeheader()
            for row in self.epoch_data:
                writer.writerow(row)

        log.info("[DIAG] Epoch summary: %d epochs -> %s", len(self.epoch_data), csv_path)

        # Also print a text summary to log
        if self.epoch_data:
            first = self.epoch_data[0]
            last = self.epoch_data[-1]
            log.info("[DIAG] === DIAGNOSTIC SUMMARY ===")
            log.info("[DIAG] Epochs tracked: %d to %d", first['epoch'], last['epoch'])
            log.info("[DIAG] CE grad norm: start=%.6f end=%.6f",
                     first.get('ce_grad_norm', 0), last.get('ce_grad_norm', 0))
            log.info("[DIAG] Constraint grad norm: start=%.6f end=%.6f",
                     first.get('constraint_grad_norm', 0), last.get('constraint_grad_norm', 0))
            log.info("[DIAG] Grad ratio (constraint/CE): start=%.6f end=%.6f",
                     first.get('grad_ratio', 0), last.get('grad_ratio', 0))
            if 'weight_drift_l2' in last:
                log.info("[DIAG] Weight drift from warmup: L2=%.6f per_param=%.8f",
                         last.get('weight_drift_l2', 0), last.get('weight_drift_per_param', 0))
            if 'changed_from_warmup' in last:
                log.info("[DIAG] Predictions changed from warmup: %d / %d (%.1f%%)",
                         last['changed_from_warmup'], self.n_test,
                         100 * last['changed_from_warmup'] / self.n_test)
            for c in self.constrained_classes:
                if f'net_movement_{c}' in last:
                    log.info("[DIAG] Class %d: net_movement=%d (to=%d, from=%d) excess=%d",
                             c, last.get(f'net_movement_{c}', 0),
                             last.get(f'moved_to_{c}', 0), last.get(f'moved_from_{c}', 0),
                             last.get(f'excess_{c}', 0))

    def _save_prediction_stability(self):
        """Analyze how stable per-sample predictions are across epochs."""
        if not self._prediction_history or len(self._prediction_history) < 2:
            return

        csv_path = self.diag_dir / 'prediction_stability.csv'
        epochs = [e for e, _ in self._prediction_history]
        preds = np.array([p for _, p in self._prediction_history])  # (n_epochs, n_test)

        # For each sample: count how many times its prediction changed
        changes = np.diff(preds, axis=0) != 0  # (n_epochs-1, n_test)
        change_count = changes.sum(axis=0)  # (n_test,)
        unique_preds = np.array([len(set(preds[:, i])) for i in range(self.n_test)])

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_idx', 'num_changes', 'num_unique_preds',
                             'warmup_pred', 'final_pred', 'is_oscillating'])
            for i in range(self.n_test):
                warmup_pred = self.warmup_predictions[i] if self.warmup_predictions is not None else -1
                final_pred = preds[-1, i]
                # Oscillating = changed 3+ times and final == warmup (went out and came back)
                is_osc = change_count[i] >= 3 and final_pred == warmup_pred
                writer.writerow([i, int(change_count[i]), int(unique_preds[i]),
                                 int(warmup_pred), int(final_pred), int(is_osc)])

        # Summary stats
        n_stable = (change_count == 0).sum()
        n_single_change = (change_count == 1).sum()
        n_oscillating = (change_count >= 3).sum()
        log.info("[DIAG] Prediction stability: stable=%d single_change=%d oscillating(3+)=%d",
                 n_stable, n_single_change, n_oscillating)

    def _save_probability_evolution(self):
        """Save how P(constrained_class) evolves for each sample."""
        if not self._proba_history:
            return

        for ci, c in enumerate(self.constrained_classes):
            csv_path = self.diag_dir / f'proba_evolution_class{c}.csv'
            epochs = [e for e, _ in self._proba_history]

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_idx'] + [f'epoch_{e}' for e in epochs])
                for i in range(self.n_test):
                    row = [i]
                    for _, proba_snap in self._proba_history:
                        row.append(f"{proba_snap[i, ci]:.6f}")
                    writer.writerow(row)

            log.info("[DIAG] Probability evolution class %d: %d snapshots x %d samples -> %s",
                     c, len(epochs), self.n_test, csv_path)

    def _save_sample_transition_matrix(self):
        """Save warmup->final transition matrix: how many samples moved between classes."""
        if self.warmup_predictions is None or not self._prediction_history:
            return

        csv_path = self.diag_dir / 'transition_matrix.csv'
        final_preds = self._prediction_history[-1][1]

        # Build transition matrix: [warmup_class, final_class] = count
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(self.n_test):
            matrix[self.warmup_predictions[i], final_preds[i]] += 1

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['warmup\\final'] + [f'class_{c}' for c in range(self.num_classes)]
            writer.writerow(header)
            for r in range(self.num_classes):
                row = [f'class_{r}'] + matrix[r].tolist()
                writer.writerow(row)

        # Log the constrained-class rows/columns
        for c in self.constrained_classes:
            moved_from = matrix[c, :].sum() - matrix[c, c]
            moved_to = matrix[:, c].sum() - matrix[c, c]
            stayed = matrix[c, c]
            log.info("[DIAG] Transition class %d: stayed=%d moved_away=%d moved_in=%d",
                     c, stayed, moved_from, moved_to)
