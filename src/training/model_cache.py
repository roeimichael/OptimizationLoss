"""Model caching: save/load warmup models to avoid redundant training."""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from src.models import get_model
from src.utils.error_handler import safe_execute

log = logging.getLogger(__name__)


def get_cache_path(base_model_id: str) -> Path:
    """Get file path for a cached model."""
    cache_dir = Path('model_cache')
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{base_model_id}.pt"


def save_to_cache(model: nn.Module, base_model_id: str, config: Dict[str, Any]) -> None:
    """Save model state to cache."""
    path = get_cache_path(base_model_id)
    torch.save({
        'model_state_dict': model.state_dict(),
        'base_model_id': base_model_id,
        'config': config,
        'saved_at': time.strftime('%Y-%m-%d')
    }, path)
    log.info("Model cached: %s", base_model_id)


def load_from_cache(base_model_id: str, config: Dict[str, Any],
                    input_dim: int, num_classes: int, device: torch.device) -> Optional[nn.Module]:
    """Attempt to load model from cache. Returns None if not found."""
    path = get_cache_path(base_model_id)
    if not path.exists():
        return None

    hp = config['hyperparams']
    ckpt = safe_execute(
        torch.load, path, map_location=device,
        default=None, context=f"Loading cached model {base_model_id}"
    )

    if ckpt is None or ckpt.get('base_model_id') != base_model_id:
        return None

    model = get_model(
        config['model_name'], input_dim=input_dim, n_classes=num_classes,
        hidden_dims=hp.get('hidden_dims'), dropout=hp['dropout'],
        pretrained=False  # Loading saved weights, not pretrained
    ).to(device)

    result = safe_execute(
        model.load_state_dict, ckpt['model_state_dict'],
        default=None, context=f"Loading state dict for {base_model_id}"
    )

    return model if result is not None else None
