# CPU namespace and owned wall-time boundary preparation

The approved knee snapshot pilot still needs an actual launch boundary. This
stage adds a small CPU-only namespace command builder and an owned child
collector. It does not issue data, seed, device, cost or campaign permission,
and it does not change either closed scientific CLI.

The namespace exposes declared runtime libraries, immutable source, public
input, one weight file and one operator as read-only mounts, with one writable
output. It creates new namespaces and a fresh device directory, drops
capabilities, clears the environment, and uses Python isolated mode. It mounts
no private or original data tree, host device tree or host home directory.
Source authentication and source symlink checks belong to the caller; building
an argument list is not an isolation certificate.

The collector accepts an explicit finite outer wall, creates an exclusive
output, closes inherited descriptors, supplies only an allowlisted startup
environment and records normal, failed, timed-out or unresolved cleanup states.
It signals only the child it created. A wrapper exit and its wall time do not
establish descendant CPU usage, descendant termination, GPU time or ownership.
The receipt makes these limits explicit. No arbitrary PID or process-group
control interface is added.

Twenty new CPU cases cover the mount/environment contract, unsafe or
overlapping paths, invalid wall bounds, exclusive outputs, normal and failed
owned children, a wall timeout and refusal to inherit fictitious private or
Python startup environment variables. The missing-module and initial startup
environment failures are retained separately. All twenty final cases passed
locally; exact target-host source/test/simulation evidence is stored outside
source under the deployed commit.

New target-host simulations are declared separately: only fictitious files,
stdlib operators and CPU namespaces. One normal operator checks direct and
symlink private denial, inherited descriptor closure, absent private startup
environment, an isolated host loopback listener, read-only public input and
writable output. One intentional outer-wall timeout exercises an owned
namespace and its heartbeat-writing child. A stopped heartbeat is a bounded
observation about that fixture, not universal descendant cleanup proof.
No prior canary, actual CPU trajectory, model, gradient gate or suite is replayed.

The scientific recipe, four prospective seeds, groups, caps and allocators
remain unchanged. The real one-epoch CPU evidence stays at source 778331 and
the full CPU baseline stays cc21. Both scientific CLIs refuse. No GPU inventory,
context, work, charge or credit is added; partial planning charge remains
75.158211024 GPUh and certified remaining balance is unavailable. Real GPU,
source/private/device/FD integration, historical seed freshness, exclusive
claims and aggregate cost coverage remain required before any campaign.
