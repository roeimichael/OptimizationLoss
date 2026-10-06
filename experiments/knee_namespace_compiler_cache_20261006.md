# Explicit compiler-cache environment for the CPU namespace

The original b460210 two-host logging checks stopped during collection because
Torch's compiler-cache fallback requested a username absent from the namespace's
passwd database. Those failures and the separate successful operators that set
a cache path are preserved. They were CPU infrastructure outcomes, not a failed
scientific campaign or evidence about actual image quality.

The narrow production change gives the CPU command builder an explicit
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-cache`. `/tmp` is already a fresh
namespace tmpfs. This supplies the cache location without forwarding a host
username or host cache path. Read-only mounts, fresh `/dev`, closed inherited
descriptors, Python cache prefix, finite bounds and scientific settings remain
unchanged. A compiler cache path does not authenticate installed native code or
certify arbitrary imports, descendants, devices, ownership or GPU cost.

Two new stdlib examples and the directly affected mount example are declared
before execution. A separate new operator on each immutable host will import
Torch and Torchvision inside the CPU namespace, confirm the explicit cache
location and record process SELF observations and CUDA's uninitialized status.
It will use no constructed model, tensors, actual image/private label/pretrained
weight, scientific seed, manual reseeding, fitting, scoring or GPU query/job.
Library imports are real native CPU preparation, not a provenance certificate.

Each new operator has 60 CPU-seconds, 16 GiB address space, 120 alarm wall seconds,
125 seconds of collector wall and 180 seconds of outer SSH time. These are finite
infrastructure bounds, not measured GPU time. No original failed collection,
successful logging tests, stdlib child fixture, actual one-epoch model gate,
pre-import input phase or full suite will be repeated. No scientific launcher or
approval issuer is enabled; both scientific CLI refusals remain unchanged.
