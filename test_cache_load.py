#!/usr/bin/env python3
"""Test loading the cached model to find the actual error."""

import torch
from pathlib import Path
from src.models.model_factory import get_model

cache_file = Path('model_cache/BasicNN_6007870002bf.pt')
device = torch.device('cpu')

print(f"Loading: {cache_file}")

try:
    # Load checkpoint
    ckpt = torch.load(cache_file, map_location=device)
    print(f"✓ Checkpoint loaded")
    print(f"  base_model_id: {ckpt['base_model_id']}")

    # Get hyperparams from stored config
    config = ckpt['config']
    hyperparams = config['hyperparams']

    print(f"\nCreating model with:")
    print(f"  model_name: {config['model_name']}")
    print(f"  input_dim: 33")
    print(f"  hidden_dims: {hyperparams['hidden_dims']}")
    print(f"  dropout: {hyperparams['dropout']}")

    # Create model
    model = get_model(
        config['model_name'],
        input_dim=33,
        hidden_dims=hyperparams['hidden_dims'],
        n_classes=3,
        dropout=hyperparams['dropout']
    ).to(device)

    print(f"✓ Model created")
    print(f"  Type: {type(model).__name__}")

    # Load state dict
    print(f"\nLoading state dict...")
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"✓ State dict loaded successfully!")

    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(10, 33).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sample: {output[0]}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
