# Exclusive seed record preparation

The approved knee pilot still needs historical seed freshness and an actual
campaign authorization boundary. A small stdlib primitive now creates one
kernel-exclusive claim name in a caller-supplied registry. It never overwrites,
resumes, rolls back or releases a claim. A failed write leaves its exclusive name
unavailable, so an incomplete record cannot silently authorize a retry.

The identity bytes are declared finite JSON and recorded before any execution.
Each record explicitly grants no campaign permission. The primitive does not
authenticate those declarations, historical non-use, source, input, device,
cost or certified balance. Its scope is one existing registry shared by all
cooperating launchers; it does not discover other registries or old operations.
Filesystem exclusivity depends on the underlying filesystem implementation.
This is a preparation component, not a scientific launcher or approval issuer.

Fifteen new CPU cases use fictitious seeds and registries. They check identity
binding, malformed inputs, preservation of existing/incomplete claims, a
twelve-contender single-winner race, a retained write failure, registry scope
and refusal to create an undeclared registry. Missing-module failures are
preserved separately. Target-host validation and one separately declared
two-host shared-registry race use only fictitious claims. No prior suite,
namespace canary or real CPU trajectory is repeated.

Both scientific CLIs remain closed. Seeds7001-7004 remain unclaimed, historical
freshness remains open and no source/science/group/cap/recipe changes follow.
Partial planning charge stays75.158211024 GPUh, certified remaining is unavailable
and nominal headroom is not permission to spend. No GPU query, context, probe or
campaign is part of this stage.
