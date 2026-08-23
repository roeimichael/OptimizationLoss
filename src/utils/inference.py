# Shared chunked forward pass for GPU memory-safe inference.
# Used by trainer, metrics, and posthoc_adjustment modules.

import torch

INFERENCE_CHUNK_SIZE = 512


def chunked_forward(model, X, chunk_size=INFERENCE_CHUNK_SIZE):
    if len(X) <= chunk_size:
        return model(X)
    chunks = [model(X[i:i + chunk_size]) for i in range(0, len(X), chunk_size)]
    return torch.cat(chunks, dim=0)
