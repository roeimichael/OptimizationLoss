# Process SELF resource observations for the approved knee driver

The completed real one-epoch CPU gate did not log Python SELF CPU totals. Its
parent `wait4` measured the Bubblewrap wrapper, whose near-zero CPU and RSS do
not represent Python descendants. Preserve that qualification and both older
failed attempts. This source stage adds prospective driver observations so a
future execution can distinguish its own CPU usage from wrapper measurements.

The bounded change reads Linux `getrusage(RUSAGE_SELF)` at driver start and normal
or exceptional completion. User/system CPU counters are cumulative seconds;
their difference covers only that interval and includes this process's threads.
Linux maximum RSS is normalized from KiB to bytes and remains a lifetime high
water mark, never a phase delta. Unsupported platforms explicitly record
unavailable values, with no invented zero. PID/platform/scope, finite counters,
counter order and matching start/end/delta records are checked before scoring.

The approved recipe, six correction arms, original PTO trajectory, gradients,
optimizer, RNG and fixed prediction averaging are unchanged. These observations
cannot measure GPU kernels, children, native device ownership, job cleanup,
throughput or a certified budget. They do not impose or relax a resource limit.
Scientific launch/scoring CLIs remain closed before Torch.

Prospective new driver fixtures use fictitious seeds 9482001, 9482002 and 9482003, the
existing 40-training/100-development synthetic RGB pack and two-epoch tiny CPU
classifier. Standard-library counter cases use no model or RNG. A separately
declared real Linux SELF observation fixture will use no actual images, labels,
weights, Torch or scientific seed. No completed CPU/model/campaign trajectory
will be replayed to repair historical resource gaps. The full CPU baseline
remains cc21; only directly affected fictitious driver cases are relevant here.

Exact committed releases, new checks, original failures and independent receipt
verification are recorded outside source. This is logging preparation, not a
new real one-epoch, GPU or scientific campaign gate.

An exceptional resource observation is recorded separately and preserves the
original fitting exception; it cannot manufacture a completion. The prospective
Linux observation fixture gives one stdlib child a 0.1 CPU-second busy loop and
32 MiB allocation, with separate child/parent SELF records. Its namespace has
60 CPU-seconds, 16 GiB address space and 120 alarm wall seconds for the observation
and directly affected tiny-driver checks, plus a 125-second collector wall.
This fixture has no RNG or science seed and does not certify universal descendant
cleanup or CPU usage. The final CLI source refusal is unchanged and will be
checked by source binding, without routinely replaying those five refusal tests.
