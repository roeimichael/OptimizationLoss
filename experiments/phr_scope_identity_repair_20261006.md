# PHR pooled/local identity repair, 2026-10-06

The neighboring PHR step used `global` for both the pooled derivative and a
permitted local group name. Unlike the previous TraLO overwrite, PHR accumulated
the two directional derivatives into one entry. The boundary controller then
received that combined value for both constraints. Its offline fixed-step
recount also overwrote the pooled count and cap with that local group's values.

A prospectively declared, fixed CPU scalar example has probabilities .8/.6,
probability derivatives +4/-6, pooled cap 1, first local cap 0, other cap 1,
zero multipliers and rho .5. Its PHR gradient is +1.2 and unit direction -1.
The pooled directional derivative +2 must remain separate from the local -4.
Their old combined -2 hides ascent of the violated pooled scope. The repaired
boundary path refuses that direction before any probe or displacement.

New records declare `scope_derivative_schema=pooled-local-v1` and use `pooled`
and `local:<exact group>` keys. Both PHR auditors accept that format and retain
unambiguous historical records. Unknown formats and ambiguous legacy records
are refused; no old evidence is changed. Canonical recount uses distinct scope
identities and positional caps; the legacy helper remains for unambiguous names.

Fixed-step residuals, PHR penalty, upstream gradient, normalized direction,
radius and dual update are unchanged. Relabeling the fixed example preserves
exact parameters and predictions, with local dual entries reordered only by
the existing lexical scope order. Independent AST reconstruction removes only
identity changes and matches the complete parent product module.

The original local regression collected 21 new cases: 19 failed and 2 passed.
After repair, the same 21 plus 44 directly affected CPU cases passed. The
original declaration, test versions, failures and independent arithmetic checks
are preserved under `phr_scope_identity_20261006T083744261992Z` outside the tree.

Completed country groups and approved H0/H1 do not use the colliding name.
This edge-case repair does not explain their F1 outcomes. Raw count descent,
valid allocation, algorithm fidelity and useful ranking remain separate claims.
No scientific seed, real image/model/label payload, scoring or CUDA job was used.
