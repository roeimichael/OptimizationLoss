# First real-data global Clipper checkpoint

## What changed and why

Commit `44f485e148e8f365bdcb9d1235e4e8c6f21da11b` adds a fresh supervised image
baseline, two explicitly named global-only allocation diagnostics, saved
probabilities and an auditable report. No legacy training or allocation code is
imported. The installed **Testing on Server** skill was added in `470152c0`.
Both commits were pushed to GitHub and the dedicated DSI mirror.

Protocol and exact commands: [global_clipper_pilot.md](global_clipper_pilot.md).
Configuration: [training](../examples/cifar100_pilot.json),
[caps](../examples/cifar100_global_caps.json).

## Data, training and constraint

One seed, CIFAR-100 official training pool: 10,000 training images and 2,000
disjoint development images. The official test split was not evaluated.
Frozen ImageNet ResNet-18 features; linear 100-class head; SGD learning rate
0.1, 10 epochs, head batch 256, feature batch 64, image size 224; FP32 throughout.
These are pilot settings, not endorsed optimal hyperparameters.

The head minimizes mean supervised cross-entropy:
`L = -(1/B) * sum_i log softmax(W f(x_i) + b)[y_i]`.
Only training labels enter this loss. Development labels enter metrics afterward.

On the development pool, each class c=0,...,9 has upper bound K_c=10:
`sum_i 1[prediction_i = c] <= 10`. Other classes are uncapped.
This synthetic policy was fixed before outcomes; it is not derived from labels.

**Upper-bound correction** retains feasible raw predictions, removes the weakest
predictions from overfull classes, then assigns those released images to feasible
classes. **Capped-first** fills constrained slots by descending probability first,
then assigns remaining images. Both consume identical saved probabilities and
no evaluation labels. They are distinct greedy policies, not exact optimizers.

## Results

F1 values below are on a 0–100 scale. Macro-F1 averages all 100 class F1 scores;
constrained-class F1 averages classes 0–9. For each class,
`F1 = 2 TP / (true class count + predicted class count)`; undefined ratios are zero.
Accuracy is the percentage of all 2,000 predictions that match the true label.
No primary winner metric was selected.

| Same model scores, different allocation | Correct / 2,000 | Accuracy | Macro-F1 | Constrained-class F1 | All caps satisfied? | Predictions changed from raw |
|---|---:|---:|---:|---:|---|---:|
| Raw argmax, no quota enforcement | 1,003 | 50.15% | 52.11 | 53.16 | No | 0 |
| Upper-bound correction | 1,025 | 51.25% | 52.44 | 48.87 | Yes | 207 |
| Capped-first | 1,027 | 51.35% | 52.68 | 51.29 | Yes | 232 |

| Counts for class indices 0 through 9 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cap | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| Raw | 14 | 15 | 8 | 3 | 12 | 12 | 11 | 183 | 18 | 22 |
| Upper-bound correction | 10 | 10 | 8 | 4 | 10 | 10 | 10 | 10 | 10 | 10 |
| Capped-first | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

Upper-bound correction increases accuracy by **1.10 percentage points**, while
constrained-class F1 falls by **4.28 points**, relative to raw predictions.
Capped-first changes those metrics by **+1.20** and **-1.87 points**, respectively.
The raw model is infeasible, so it is a diagnostic reference, not a feasible
competitor. The two feasible policies differ by only two correct predictions;
one seed does not establish a reliable advantage.

The raw classifier predicts class 7 for 183 images although only 20 are truly
class 7 (18 true positives). This exposes an uneven baseline. It does not prove
the dataset is hard for a well-trained model, or identify the cause of that skew.
Before research comparisons, establish a stronger, adequately trained baseline.

## Verification and evidence

- 40 unit tests passed locally and on both hosts; deployed tracked-file hashes
  matched. A separate CUDA arithmetic/logger smoke passed on dsisco01.
- Actual host: dsisco01, Quadro RTX 6000, compute capability 7.5, GPU0,
  UUID `GPU-aa377381-804a-f7df-8903-167cea7c7414`; local CUDA index 0.
  Launch environment selected `CUDA_VISIBLE_DEVICES=0`.
  Both hosts were checked; dsisco02 Blackwell cards were occupied and untouched.
- CIFAR archive MD5 `eb9058c3a382ffc7106e4002c42a8d85` matched on client/server;
  torchvision checked dataset integrity. Exact image-byte overlap across the
  selected training/development sets was absent; this is not a semantic-duplicate audit.
- Pretrained weights SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- All 400 planned updates were attempted/applied; zero skipped updates.
  Logged mean training loss declined from 3.8002 to 1.2439. All values were finite.
- Artifact hashes, disjoint split IDs and identical allocation input bytes were
  verified. Independent scikit-learn accuracy/F1 recomputation agreed within
  1e-12; counts were independently recounted from predictions.
- A separate 500-case small-problem enumeration confirmed feasible outputs for
  both policies, but found gaps from the maximum sum of assigned probabilities.
  Feasibility does not imply objective optimality or best accuracy.

Remote run (preserved):
`/home/dsi/michaer8/tralo-rebuild/runs/cifar100-global-44f485e1-20260922T132450Z/`.
Local artifact copy:
`C:/Users/roeym/.codex/rebuild-audit-20260922/cifar100-run/`.
Contains launch receipt, log, training events, splits, data hashes, features,
head checkpoint, probabilities, allocation inputs and complete confusion matrices.
Report SHA-256: `34e62b07638f5d3606422c01e211df881fc78ae958fe694f7ad49a3d42122a3c`.

## What this establishes and what is next

The first real-data training/prediction/allocation/report path works on a DSI GPU.
It does not establish TraLO effectiveness, statistical superiority, full mixed-
precision training support, or a final research benchmark. Local constraints are
deferred. Next: choose the intended global allocation objective, compare against
an exact assignment reference, establish a stronger baseline, then add a specified
global constraint loss with a truly matched no-constraint control.
