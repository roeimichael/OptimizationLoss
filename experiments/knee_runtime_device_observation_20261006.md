# Requested and observed device identity

The approved knee snapshot pilot needs to distinguish its requested physical
GPU from the device the runtime actually exposes. A requested UUID alone did
not establish historical CelebA replay identity; those old models and scores
remain untouched. The new component records both fields separately and refuses
missing, ambiguous, mismatched or malformed observations.

`tralo.knee_snapshot_device.observe_single_device` validates one full physical
UUID and its exact `CUDA_VISIBLE_DEVICES` agreement before importing Torch.
It then requires exactly one visible device and compares the runtime's 16 raw
UUID bytes with the request. It records visible index zero, name, capability
and total memory. It never substitutes the request for an absent observation,
changes the environment, allocates a tensor, claims a seed or issues permission.

The raw-byte interface is grounded in the upstream
[PyTorch 2.11 CUDA property binding](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/cuda/Module.cpp),
which exposes `uuid.bytes` as sixteen byte values. This source reference does
not authenticate the installed native library or prove a live GPU observation.
Stringifying or copying the requested UUID was rejected because it cannot
establish what the runtime observed. Missing interfaces fail closed.

Local validation exercised 22 new cases using a fictitious Torch/CUDA provider:
pre-import refusals, environment/request mismatch, missing or multiple visible
devices, malformed/unavailable raw UUIDs, a different observed UUID and invalid
hardware fields. The original missing-module failure and passing JUnit receipt
are preserved outside source. No tensor library, actual GPU, model, image,
weight, label, scientific seed or quality evaluation was used. The new focused
checks will run once on each newly deployed immutable host release; their exact
source and original receipts belong to the external checkpoint.

This operation can initialize CUDA when used with real Torch. A future real
launcher must first pass authenticated source/private boundaries, historical
seed freshness, fresh both-host exclusive ownership and certified finite budget
gates. The scientific training/scoring CLIs remain closed and do not import or
call this component. Actual runtime UUID binding, CUDA correctness, isolation,
throughput and cost remain untested. It grants no ownership or campaign
permission and does not change the approved recipe or the partial accounting
charge of 75.158211024 GPUh; certified remaining room is still unavailable.
