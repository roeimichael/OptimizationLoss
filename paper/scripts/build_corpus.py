"""Derive paper/data/corpus/corpus_final.csv from the run manifest.

Why this exists: seven scripts READ corpus_final.csv (every headline table, the
OctMNIST figure, the deployment figure, the granular tables, LOBO-CV) and until
now NOTHING wrote it. It was produced once, by hand, and had no reproducible
build path -- which is how the ALM arm came to be missing from every table
without any check failing. This closes that loop, so new campaigns reach the
paper by rebuilding rather than by hand-editing.

    results/**/config.json
        -> build_experiment_manifest.py   (on the server)
        -> paper/data/manifest/experiments.csv
        -> build_corpus.py                (here)
        -> paper/data/corpus/corpus_final.csv
        -> make_main_table.py, make_octmnist_fig.py, ...

The `sweep` column the generators filter on (`sweep=='paper_final'`) maps to the
manifest's `campaign`.

VERIFY MODE is the important one. `--verify` rebuilds the corpus and diffs it
against the corpus on disk without writing anything, so you can prove the
derivation reproduces the numbers the paper was actually built from before you
let it overwrite them:

    python paper/scripts/build_corpus.py --verify
    python paper/scripts/build_corpus.py            # writes, after verify passes
"""

import argparse
import os

import pandas as pd

MANIFEST = os.path.join("paper", "data", "manifest", "experiments.csv")
CORPUS = os.path.join("paper", "data", "corpus", "corpus_final.csv")

# corpus_final.csv column order, preserved exactly -- the readers index by name,
# but a stable order keeps diffs readable.
COLS = ["sweep", "dataset", "model", "method", "constraint_tag",
        "constrained_class", "group_column", "warmup_epochs", "seed",
        "acc", "f1_macro", "cc_f1", "cc_rec", "cc_prec", "flips", "sat"]

RENAME = {"campaign": "sweep", "cap": "constraint_tag", "warmup": "warmup_epochs"}

# Sweeps the paper's figure/table generators actually read. Everything filters
# on sweep=='paper_final'; discrepancies anywhere else cannot reach the PDF.
PAPER_SWEEPS = {"paper_final"}


def derive():
    m = pd.read_csv(MANIFEST, low_memory=False)
    m = m[m.status == "completed"]
    m = m[m.cc_f1.notna()]
    df = m.rename(columns=RENAME)
    for c in COLS:
        if c not in df.columns:
            df[c] = None
    return df[COLS].reset_index(drop=True)


def key(df):
    return df.set_index(
        ["sweep", "dataset", "model", "method", "constraint_tag", "seed"])


def verify(new):
    if not os.path.exists(CORPUS):
        print("no existing corpus to verify against")
        return True
    old = pd.read_csv(CORPUS, low_memory=False)
    print("existing corpus : %d rows" % len(old))
    print("derived corpus  : %d rows" % len(new))

    o, n = key(old), key(new)
    common = o.index.intersection(n.index)
    only_old = o.index.difference(n.index)
    only_new = n.index.difference(o.index)
    print("shared rows     : %d" % len(common))
    print("only in existing: %d" % len(only_old))
    print("only in derived : %d  (new campaigns)" % len(only_new))

    ok = True
    if len(only_old):
        # A row the manifest cannot reproduce only matters if the paper reads
        # its sweep. Everything the generators consume is sweep=='paper_final';
        # make_backbone_tables additionally blocklists smoke/probe sweeps. So an
        # orphan in an unused sweep is a stale artifact, not a data loss.
        relevant = [r for r in only_old if r[0] in PAPER_SWEEPS]
        stale = [r for r in only_old if r[0] not in PAPER_SWEEPS]
        if stale:
            print("\n   %d unreproducible row(s) in sweeps the paper does not read "
                  "(stale artifacts, ignored):" % len(stale))
            for r in stale[:5]:
                print("      ", r)
        if relevant:
            ok = False
            print("\n!! %d unreproducible row(s) in a PAPER sweep -- investigate "
                  "before overwriting:" % len(relevant))
            for r in relevant[:10]:
                print("   ", r)

    # numeric agreement on the shared rows is the real test
    worst = []
    for col in ["cc_f1", "f1_macro", "acc"]:
        a = o.loc[common, col].astype(float)
        b = n.loc[common, col].astype(float)
        d = (a - b).abs()
        d = d[d.notna()]
        if len(d):
            worst.append((col, float(d.max()), int((d > 1e-6).sum())))
    print("\nnumeric agreement on shared rows:")
    for col, mx, ndiff in worst:
        flag = "OK" if ndiff == 0 else "MISMATCH"
        print("   %-9s max|diff|=%.6g  rows differing=%d   %s" % (col, mx, ndiff, flag))
        if ndiff:
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="diff against the corpus on disk, write nothing")
    ap.add_argument("-o", "--out", default=CORPUS)
    args = ap.parse_args()

    new = derive()
    ok = verify(new)
    if args.verify:
        print("\nVERIFY %s" % ("PASSED" if ok else "FAILED"))
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("verification failed; refusing to overwrite %s" % args.out)
    new.to_csv(args.out, index=False)
    print("\nwrote %s (%d rows)" % (args.out, len(new)))


if __name__ == "__main__":
    main()
