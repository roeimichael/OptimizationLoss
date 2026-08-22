# Shared chunked forward passes for GPU memory-safe inference.
#
# TWO functions, TWO consumers, and they are not the same knob:
#   `chunked_forward` -- src/training/metrics.py only. Stride below.
#   `chunked_probs`   -- the allocator arms (heuristic, danits_lp,
#                        imbalanced_common), which pass the protocol's
#                        `inference_chunk_size`.
# The header used to say "trainer, metrics, and posthoc_adjustment"; the
# trainer and posthoc_adjustment import nothing from here, so the docstring
# named two consumers that do not exist and hid the one that does.

import torch

# NOT `inference_chunk_size`, and deliberately not unified with it. This is a
# fixed stride for the metrics forward pass; the config key is the allocators'
# knob and defaults to src.utils.constants.INFERENCE_CHUNK_SIZE (256).
# They carry different values (512 vs 256) and must keep doing so: re-chunking
# a forward pass re-associates the batch dimension, so switching this to 256
# perturbs the reported metrics in the last bits -- and this project reads
# arm-vs-arm identity at md5 resolution, where "mathematically identical" is
# not the same claim as "identical". Private, so the name cannot be imported
# and mistaken for the configurable one.
_METRICS_FORWARD_CHUNK = 512


def chunked_forward(model, X):
    if len(X) <= _METRICS_FORWARD_CHUNK:
        return model(X)
    chunks = [model(X[i:i + _METRICS_FORWARD_CHUNK])
              for i in range(0, len(X), _METRICS_FORWARD_CHUNK)]
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
