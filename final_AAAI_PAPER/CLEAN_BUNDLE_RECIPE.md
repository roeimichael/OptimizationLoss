# Code & Data Supplement — clean-bundle recipe

How to build the anonymous "Code and Data Supplement" zip the paper promises, without
leaking identity. Audited 2026-07-09. Two in-place scrubs already applied:
`build_pdf.sh` (username → `$HOME`) and `gen_precision_majority.py` (hostname removed).
`src/` and `final_AAAI_PAPER/data/` are now clean of usernames/hostnames.

## INCLUDE (anonymous + needed to reproduce)

| Path | What it is |
|---|---|
| `src/` | method, training, eval code (hostname-clean; "Blackwell" GPU-arch mentions are fine) |
| `main.py` | experiment dispatcher |
| `requirements.txt` | pinned deps (torch≥2.0, torchvision≥0.15, medmnist≥3.0) |
| `final_AAAI_PAPER/data/corpus/corpus_final.csv` | the 1944-run results corpus (no path/user column) |
| `final_AAAI_PAPER/data/dynamics/` | per-epoch training logs (clean) |
| `README.md` (new, anonymous — template below) | how to run |

## EXCLUDE — and WHY (do not zip these)

| Path | Reason |
|---|---|
| `.git/` | ⚠️ commit author "roei michael" + email + full history — **the #1 leak** |
| `benchmarks/` | ⚠️ third-party ORIGINAL code (Fioretto, Shifman/"danit") — copyright + others' paths/names |
| `final_AAAI_PAPER/papers/` | ⚠️ copyrighted PDFs of cited works + validation artifacts |
| `docs/` | ⚠️ planning notes: "Owner: roei", server name, "Bar-Ilan"/BIU, GPU policy |
| `scripts/` | ⚠️ infra: server hostname, absolute pull/queue paths |
| `archive/`, `archive_experiments/` | legacy runs with old server paths (`michaer8`, hostname) |
| `CLAUDE.md`, `memory/` | internal project notes / agent memory |
| `.venv/`, `venv/`, `refvenv/`, `__pycache__/`, `*.pyc` | environments / caches |
| `model_cache/` | large cached warmups, not needed |
| `data/tissuemnist/ dermmnist/ aider/` (raw images) | large + public; loaders + MedMNIST pointer suffice |
| `final_AAAI_PAPER/*.tex *.pdf *.aux *.bbl *.out`, `build_pdf.sh` | paper source/build — goes via the submission PDF, not the code zip |
| `*_CHECKLIST.md`, `*_RECIPE.md`, this file | your prep docs, not for reviewers |

## One-shot staging + scrub + zip (run from repo root, Git Bash)

```bash
STAGE=/tmp/supplement && rm -rf "$STAGE" && mkdir -p "$STAGE"
# copy only the include-list
cp -r src main.py requirements.txt "$STAGE"/
mkdir -p "$STAGE/data"
cp -r final_AAAI_PAPER/data/corpus final_AAAI_PAPER/data/dynamics "$STAGE/data"/
# strip any stray caches that rode along
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
find "$STAGE" -name '*.pyc' -delete
# add your anonymous README (see template), then:
# ---- PRE-FLIGHT SCRUB: this MUST print nothing ----
grep -rniE "roeym|\broei\b|michael|danit|gmail|dsisco|bar.?ilan|\bbiu\b|/home/[a-z]|C:\\\\Users|D:\\\\" "$STAGE"
echo "scrub exit: $?  (1/empty = clean)"
# if clean, zip:
( cd "$STAGE" && zip -r ../code_data_supplement.zip . -x '*.DS_Store' )
```

If the pre-flight grep prints **anything**, fix that file before zipping. (A hit inside a
`references.bib` you deliberately included would be a cited author's name — fine; anything
under `src/` or a config is not.)

## Minimal anonymous README.md (drop into the zip)

```markdown
# Transductive count-constrained training — code & data

## Install
pip install -r requirements.txt   # torch>=2.0, torchvision>=0.15, medmnist>=3.0

## Data
MedMNIST v2 (public): the loaders fetch the official train/test splits via the
`medmnist` package. No data is redistributed here.

## Reproduce
python -m src.config_generators.generate_configs   # emit run configs
python main.py                                     # dispatch all pending runs

## Results
data/corpus/corpus_final.csv  — one row per run (1944 headline runs, sweep=paper_final)
data/dynamics/                — per-epoch training logs for the convergence figures
```

Do **not** name the GPU model, server, OS user, institution, or conda env in the README.
```
