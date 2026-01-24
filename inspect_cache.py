#!/usr/bin/env python3
"""Inspect cached model file to see what's stored."""

import torch
from pathlib import Path

cache_file = Path('model_cache/BasicNN_6007870002bf.pt')

print(f"Inspecting: {cache_file}")
print(f"File exists: {cache_file.exists()}")
print(f"File size: {cache_file.stat().st_size} bytes")

try:
    ckpt = torch.load(cache_file, map_location='cpu')
    print(f"\nCheckpoint keys: {ckpt.keys()}")

    for key, value in ckpt.items():
        if key == 'model_state_dict':
            print(f"\n{key}: {type(value)}")
            print(f"  State dict keys: {list(value.keys())[:5]}...")
        elif key == 'config':
            print(f"\n{key}:")
            print(f"  model_name: {value.get('model_name')}")
            print(f"  base_model_id: {value.get('base_model_id')}")
            if 'hyperparams' in value:
                print(f"  hyperparams:")
                for k, v in value['hyperparams'].items():
                    print(f"    {k}: {v}")
        else:
            print(f"\n{key}: {value}")

    print("\n✓ Checkpoint loaded successfully")

except Exception as e:
    print(f"\n✗ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
