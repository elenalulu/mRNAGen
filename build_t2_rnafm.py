#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T4-v0.5: RNA-FM embeddings for decay-labeled genes in the training table.

Embedding = RNA-FM (99.5M, numpy, rna_robot utils) mean-pooled over the
FIRST 500nt of each CDS (full-length 1022nt costs 6.6s/seq = 22h; 500nt
truncation = 2.5s/seq = ~9h for 13k genes, overnight-feasible).

Scope: genes carrying at least one decay label (the M1 target is the decay
head; expr head v0 already passed its gate).

Incremental save every 100 genes (npy matrix + gene id list).
Run: D:/anaconda/python.exe -u mRNAGen/build_t2_rnafm.py
Out: mRNAGen/data/t2/t2_rnafm500.npz  (gene_ids + [n, 640] matrix)
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNAFM_UTILS = os.path.join(os.path.dirname(HERE), os.pardir, "rna_robot",
                           "utils")
TABLE = os.path.join(HERE, "data", "t2", "t2_training_table.tsv")

ENDO = os.path.join(HERE, "data", "t2", "t2_endogenous_cds_abundance.tsv")


def load_cds_map():
    m = {}
    with open(ENDO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g = r["gene"]
            if g not in m or int(r["cds_len"]) > len(m[g]):
                m[g] = r["cds"].upper()
    return m

OUT = os.path.join(HERE, "data", "t2", "t2_rnafm500.npz")

sys.path.insert(0, RNAFM_UTILS)
from rnafm_embedding import RNAFMModel  # noqa: E402

EMBED_LEN = 500


def main():
    cds_map = load_cds_map()
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8"), delimiter="\t"))
    decay_cols = ["decay_293T", "decay_hela", "decay_rpe", "decay_k562_slam"]
    targets = [r for r in rows
               if any(r[c] not in ("", None) for c in decay_cols)]
    print("[rnafm] %d decay-labeled genes of %d total"
          % (len(targets), len(rows)), flush=True)

    model = RNAFMModel()
    gene_ids = []
    embs = []
    t0 = time.time()
    done = 0

    def save():
        np.savez(OUT, gene_ids=np.array(gene_ids),
                 embeddings=np.stack(embs) if embs else np.zeros((0, 640)))

    for r in targets:
        cds = cds_map[r["gene"]]
        rna = cds.replace("T", "U")[:EMBED_LEN]
        e = model.get_sequence_embedding(rna)
        gene_ids.append(r["gene"])
        embs.append(e)
        done += 1
        if done % 100 == 0:
            save()
            el = time.time() - t0
            eta = el / done * (len(targets) - done)
            print("[rnafm] %d/%d (%.1fs, ETA %.1fh)"
                  % (done, len(targets), el, eta / 3600), flush=True)
    save()
    print("[rnafm] done: %d embeddings -> %s (%.1fs)"
          % (done, OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
