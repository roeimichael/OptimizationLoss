# Shared chunked forward pass for GPU memory-safe inference.
# Used by trainer, metrics, and posthoc_adjustment modules.

import torch

INFERENCE_CHUNK_SIZE = 512


def chunked_forward(model, X):
    if len(X) <= INFERENCE_CHUNK_SIZE:
        return model(X)
    chunks = [model(X[i:i + INFERENCE_CHUNK_SIZE])
              for i in range(0, len(X), INFERENCE_CHUNK_SIZE)]
    return torch.cat(chunks, dim=0)


def chunked_probs(model, X, chunk_size):
    """Softmax probabilities over X as a numpy array, chunked, in eval mode.

    One implementation. This was duplicated verbatim in heuristic/train.py and
    danits_lp/train.py, with a third spelling inside imbalanced_common -- and
    the two allocators it feeds are the arms compared against each other, so a
    drift between the copies would have been a difference in the BAR, not in
    the method.
    """
    model.eval()
    with torch.no_grad():
        chunks = [model(X[i:i + chunk_size]) for i in range(0, len(X), chunk_size)]
        return torch.softmax(torch.cat(chunks, dim=0), dim=1).cpu().numpy()
