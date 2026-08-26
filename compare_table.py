#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute README comparison-table metrics for the three engines:
mRNAGen (independent deliverable) vs GEMORNA vs LinearDesign.

Metrics (all scored with our own stack on each engine's benchmark
candidates): naturalness z, CAI, GC content, manufacturing-clean %,
dsRNA risk (longest near-perfect duplex), tRNA adaptation index (tAI,
dos Reis et al. 2004, GtRNAdb Hsapi38 copy numbers), and ViennaRNA
structure features (MFE per nt, paired fraction, longest helix,
5' openness) when ViennaRNA is available.

Run: python compare_table.py
"""
import csv
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "feature_pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refine_t5 import load_env, n_sites, n_hp5
from naturalness import Scorer

TRNA_CSV = os.path.join(HERE, "data", "gtrnadb_hsapi38_trna_counts.tsv")

# wobble penalty matrix, dos Reis et al. 2004 (tAI)
# rows = tRNA anticodon base (position 34, 'I' = inosine from A34)
# cols = codon base at position 3 (RNA)
S = {"A": {"A": 1.0, "C": 0.5, "G": 1.0, "U": 0.0},
     "C": {"A": 1.0, "C": 1.0, "G": 0.0, "U": 1.0},
     "G": {"A": 1.0, "C": 0.0, "G": 1.0, "U": 0.5},
     "U": {"A": 0.0, "C": 1.0, "G": 0.5, "U": 1.0},
     "I": {"A": 0.0, "C": 0.0, "G": 1.0, "U": 0.5}}
DNA_COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_trna_counts():
    d = {}
    with open(TRNA_CSV, encoding="utf-8-sig") as f:
        lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    for r in csv.DictReader(lines, delimiter="\t"):
        d[r["anticodon"]] = float(r["copies"])
    d.pop("TCA", None)  # selenocysteine (decodes stop TGA)
    return d


def codon_weight(codon, counts):
    c1, c2, c3 = codon[0], codon[1], codon[2]
    r3 = c3.replace("T", "U")
    w = 0.0
    for ac, copies in counts.items():
        if copies <= 0:
            continue
        a1, a2, a3 = ac[0], ac[1], ac[2]
        if a3 != DNA_COMP[c1] or a2 != DNA_COMP[c2]:
            continue
        a1w = "I" if a1 == "A" else a1.replace("T", "U")
        s = S[a1w].get(r3)
        if s is None or s >= 1.0:
            continue
        w += (1.0 - s) * copies
    return w


def tai(seq, counts, cache):
    codons = [seq[i:i + 3] for i in range(0, len(seq) - 2, 3)]
    codons = [c for c in codons if c not in ("TAA", "TAG", "TGA")]
    if not codons:
        return None
    for c in set(codons):
        if c not in cache:
            cache[c] = codon_weight(c, counts)
    wmax = max(cache.values()) or 1e-9
    logsum = sum(math.log(max(cache[c] / wmax, 1e-12)) for c in codons)
    return math.exp(logsum / len(codons))


def stats(rows, seqkey, table, scorer, trna, wcache):
    zs, cais, gcs, clean, scs, mfes = [], [], [], [], [], []
    tais = []
    for r in rows:
        seq = r[seqkey].upper()
        zs.append(float(scorer.z_nat([seq])[0]))
        cais.append(table.cai(seq.replace("T", "U")))
        gcs.append((seq.count("G") + seq.count("C")) / len(seq))
        sites, hp5 = n_sites(seq), n_hp5(seq)
        clean.append(1 if (sites == 0 and hp5 == 0) else 0)
        s = r.get("selfcomp_max_near") or r.get("selfcomp")
        if s:
            scs.append(float(s))
        m = r.get("mfe_per_nt")
        if m:
            mfes.append(float(m))
        t = tai(seq, trna, wcache)
        if t is not None:
            tais.append(t)
    out = {"n": len(rows), "z_nat": st.mean(zs), "cai": st.mean(cais),
           "gc": st.mean(gcs), "clean_pct": 100.0 * st.mean(clean),
           "selfcomp": st.mean(scs) if scs else None,
           "mfe_per_nt": st.mean(mfes) if mfes else None,
           "tai": st.mean(tais) if tais else None}
    # ViennaRNA structure features (if available)
    try:
        from features.structure import (mfe_features,
                                        openness_profile_features)
        pfs, lh, o45, mfes2 = [], [], [], []
        for r in rows:
            rna = r[seqkey].upper().replace("T", "U")
            f = mfe_features(rna)
            pfs.append(f["paired_frac_mfe"])
            lh.append(f["longest_helix_mfe"])
            mfes2.append(f["mfe_per_nt"])
            o = openness_profile_features(rna)
            o45.append(o["open_start45"])
        out["paired_frac"] = st.mean(pfs)
        out["longest_helix"] = st.mean(lh)
        out["open45"] = st.mean(o45)
        out["mfe_per_nt"] = st.mean(mfes2)
    except Exception as e:
        print("[cmp] ViennaRNA unavailable, structure rows skipped: %r" % e)
    return out


def main():
    t, codon_coef, syn, models = load_env()
    scorer = Scorer()
    trna = load_trna_counts()
    wcache = {}
    ours = stats(load(os.path.join(HERE, "data", "deliverable",
                                   "independent_topk.tsv")), "seq", t,
                 scorer, trna, wcache)
    v3 = load(r"D:\WorkBuddy\alphafold-web\rna_platform\mrna_neoantigen"
              r"\mRNAGen\data\t5\topk_selection.tsv")
    gem = stats([r for r in v3 if r["source"] == "gemorna"], "seq", t,
                scorer, trna, wcache)
    ld = stats([r for r in v3 if r["source"] == "ld"], "seq", t,
               scorer, trna, wcache)
    print(f"{'metric':<15}{'mRNAGen':>10}{'GEMORNA':>10}{'LinearDesign':>12}")
    for k in ["n", "z_nat", "cai", "gc", "clean_pct", "selfcomp",
              "tai", "mfe_per_nt", "paired_frac", "longest_helix",
              "open45"]:
        def f(v):
            return "-" if v is None else (f"{v:.3f}" if isinstance(v, float)
                                          else str(v))
        print(f"{k:<15}{f(ours[k]):>10}{f(gem[k]):>10}{f(ld[k]):>12}")


if __name__ == "__main__":
    main()
