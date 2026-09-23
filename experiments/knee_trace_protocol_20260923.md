# Overnight boundary diagnosis, 23–24 September 2026

Question: which individual constraint updates damage correct classifications,
and does subsequent supervised training recover them? This is a diagnostic
replay, not new independent efficacy evidence or a search for favorable seeds.

Preserve original Chen data, caps 82/16, seeds 901–904 and ResNet18 features.
Stage 1: original and smaller-step recipes, null and TraLO, seed 901 first.
Stage 2 only after exact instrumentation parity: remaining three seeds, then
the four adapted-feature seeds. Maximum 24 traced fits plus 24 same-host
uninstrumented references. No backbone retraining. No test scoring.

Record all validation logits after every supervised minibatch, before each
supervised epoch, and before/after each applied constraint update. Constraint
snapshots also save parameters, gradients and Adam state. Labels remain outside
the observer/training interface; join saved labels in offline analysis only.
Snapshot indices identify epoch, phase, batch and SHA256. Preserve failures.

Hypotheses and measurements:
- Overshoot: immediate correct-to-wrong transitions exceed wrong-to-correct,
  especially at the larger step. Count per-class TP/FP/FN, confidence margins,
  soft/hard excess and actual displacement before/after each update.
- Task/constraint conflict: supervised batches reverse the constraint step's
  logit movement; measure this, do not infer it from final scores alone.
- Feasible-count pressure: isolate updates starting hard-feasible but soft-
  infeasible. Controller freeze does not imply loss deactivation.
- Allocation interaction: apply both existing named rules to saved snapshots;
  distinguish correct ranking/selection changes from raw threshold crossings.

Gate: observed and unobserved same-host predictions AND parameters must match
exactly. Compare against historical artifacts separately. Blackwell historical
replay needs Blackwell. If unavailable, use a separately labeled FP32 Quadro
diagnostic with its own same-host reference; do not pool it with Blackwell or
claim historical exactness unless checked. Check both hosts and actual owners.

No gating intervention in this initial release. Only after trajectories are
audited, preregister one narrow intervention with fixed controls and signatures.
Do not select checkpoints/strengths using validation outcomes or run indefinite
sweeps. Deliver findings, limitations and unfinished work by 08:00 Israel on
24 September; pause the overnight heartbeat then.
