#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute README comparison-table metrics for the three engines."""
import csv
import statistics as st
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "feature_pipeline")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refine_t5 import load_env, n_sites, n_hp5
from naturalness import Scorer


def load(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def stats(rows, seqkey, table, scorer):
    zs, cais, clean, scs, mfes = [], [], [], [], []
    for r in rows:
        seq = r[seqkey].upper()
        zs.append(float(scorer.z_nat([seq])[0]))
        cais.append(table.cai(seq.replace("T", "U")))
        sites = n_sites(seq)
        hp5 = n_hp5(seq)
        clean.append(1 if (sites == 0 and hp5 == 0) else 0)
        s = r.get("selfcomp_max_near") or r.get("selfcomp")
        if s:
            scs.append(float(s))
        m = r.get("mfe_per_nt")
        if m:
            mfes.append(float(m))
    return {"n": len(rows), "z_nat": st.mean(zs), "cai": st.mean(cais),
            "clean_pct": 100.0 * st.mean(clean),
            "selfcomp": st.mean(scs) if scs else None,
            "mfe_per_nt": st.mean(mfes) if mfes else None}


def main():
    t, codon_coef, syn, models = load_env()
    scorer = Scorer()
    ours = stats(load(os.path.join(HERE, "data", "deliverable",
                                   "independent_topk.tsv")), "seq", t, scorer)
    v3 = load(r"D:\WorkBuddy\alphafold-web\rna_platform\mrna_neoantigen"
              r"\mRNAGen\data\t5\topk_selection.tsv")
    gem = stats([r for r in v3 if r["source"] == "gemorna"], "seq", t, scorer)
    ld = stats([r for r in v3 if r["source"] == "ld"], "seq", t, scorer)
    print(f"{'metric':<14}{'mRNAGen':>10}{'GEMORNA':>10}{'LinearDesign':>12}")
    for k in ["n", "z_nat", "cai", "clean_pct", "selfcomp", "mfe_per_nt"]:
        def f(v):
            return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)
        print(f"{k:<14}{f(ours[k]):>10}{f(gem[k]):>10}{f(ld[k]):>12}")


if __name__ == "__main__":
    main()
