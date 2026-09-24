# ALM / TraLO comparison: knee and CIFAR-100

Status: implementation locally validated; server access currently times out.
User authorized both datasets, four seeds, ALM and its null on 2026-09-24.
This is a new method comparison, not a repeat of the trajectory diagnosis.

## Registered design before new results

Use verified existing frozen ImageNet ResNet18 features on BOTH datasets. This
isolates the loss comparison without a dataset-specific backbone adaptation.
Knee: Chen OAI v1, 5,778 train / 826 development; 1,656 test not scored.
CIFAR-100: existing fixed 10,000 training / 2,000 development subset of official
training data, not the official test split. This is a bounded subset experiment,
not a full CIFAR-100 benchmark. No backbone training in these new runs.

Seeds 901,902,903,904; 20 epochs, 5 CE-only warmup, batch 256, task Adam lr .001.
All methods share initialization and batch order within dataset/seed. All save
final-epoch predictions; no development checkpoint selection. FP32, TF32 off,
one host architecture for the whole comparison; log actual time and update dose.
These development populations have already been inspected: not untouched tests.

| Method | After warmup | Task optimizer at boundary |
|---|---|---|
| Clipper | CE only | Keep moments |
| TraLO Null | CE only | Reset Adam |
| TraLO | CE batches + one separate constraint Adam step per epoch | Reset task Adam; separate constraint moments |
| ALM Null | CE only, same schedule as ALM | Reset Adam |
| ALM | Joint CE + PHR inequality augmented penalty each batch; projected dual update after each epoch | Reset Adam |

ALM Null and TraLO Null should be numerically identical in this design. Verify
rather than count them as independent evidence. Equal epochs do not mean equal
compute: ALM adds a full development-feature forward pass per task batch; TraLO
adds separate optimizer steps. Report both differences explicitly.

TraLO: lambda starts .01, increases by .05 under existing controller; rho .5
fixed; separate constraint Adam lr .0001. Also run the prespecified .001 setting
as a separate sensitivity condition on BOTH datasets/all seeds, retaining both.
This two-value range tests sensitivity, not a search to select a winner.
ALM: initial multiplier 0, rho .5 fixed. One supervised epoch approximates each
primal subproblem; no claim of converged classical ALM or exact paper replication.
No auxiliary margin or false-positive term in this comparison.

## ALM definition

For each capped class c, g_c = (sum_i p_ic - K_c) / max(K_c,1).
Signed residuals retain slack; unlike TraLO, do not rectify g before dual update.

A(g,lambda,rho) = sum_c ([max(0,lambda_c + rho*g_c)]^2 - lambda_c^2)/(2*rho).
Joint batch loss = mean training CE + A. After each epoch:
lambda_c <- max(0,lambda_c + rho*g_c), evaluated at the updated parameters.

Unlabelled development features enter count constraints. Their individual labels
enter only offline metrics, never loss, allocation, quota choice or training.
Normalized residuals make fractional capacity errors comparable. Sum across
constraints is explicit; it is not normalized by number of capped classes.
Rho .5 is a starting baseline, not assumed optimal; any later tuning gets equal
budgets and a distinct report. ALM null removes A and the dual update entirely.

Primary mathematical reference: Bertsekas, *Constrained Optimization and Lagrange
Multiplier Methods*, inequality multiplier methods:
https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2019/10/Constrained-Opt-1.pdf
Projected multiplier update also documented at https://manoptjl.org/dev/solvers/augmented_Lagrangian_method/ .

## Common output budget and reporting

Knee: grade 3 has 82 slots, grade 4 has 16. CIFAR: classes 0-9 each have 10
slots; other classes are unrestricted. These are existing synthetic capacities,
not inferred from development labels. The primary allocated comparison uses
capped_first, with exact constrained-class counts asserted for every method.
Uncapped-class counts may differ. This is greedy allocation, not claimed optimal.
Also retain raw argmax and upper_bound_correction as separately named views.

Report all seeds, accuracy, macro-F1, constrained-class F1, per-class TP/FP/FN,
raw violations and allocated counts; paired TraLO-minus-null and ALM-minus-null
means, SD and Student-t 95% intervals (four seeds). Also compare to Clipper.
Intervals are exploratory, without multiple-testing correction; no universal
winner claim. Do not pool datasets, policies, architectures or step settings.

## Launch gates and staged execution

Verify cache completion receipts and every cached artifact; verify split/sample
identity, data bytes and training labels against source manifests. Test ALM with
hand values, finite differences, slack dual projection, inactive-constraint and
null parity. Native tests and committed-byte parity on both hosts before launch.
Check GPU UUID/PID owner immediately before dispatch; no occupied GPUs.
Pilot seed901 all five arms per dataset; inspect completion, finite updates,
identical warmup/batches, null exactness, exact slot counts, independent metrics.
Only then expand 902-904 and the registered TraLO .001 sensitivity. Exclusive
receipts; never retry an ambiguous launch without inspecting existing artifacts.
Preserve all failed attempts. No server result is currently claimed by this file.
