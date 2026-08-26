#!/usr/bin/env python
"""UTR-1: assemble the Sample'19 MPRA training table.

Key = 50nt UTR (main library). The modified-nucleotide libraries are 59nt
(50nt core + 9nt 3' context incl. start-codon flank); joined via utr[:50].
Labels: mrl_unmod / mrl_pseudo / mrl_m1pseudo (mean over 2 replicates).
designed_library kept as an independent test set.

Output: data/t3_utr/utr_training_table.tsv + designed_test.tsv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "t3_utr")


def load_rl(path, cut50=False):
    """Return (mean rl series, mean abundance series) keyed by UTR."""
    df = pd.read_csv(path, usecols=["utr", "rl", "total"], low_memory=False)
    df["utr"] = df["utr"].astype(str).str.upper()
    if cut50:
        df["utr"] = df["utr"].str[:50]
    g = df.groupby("utr").agg(rl=("rl", "mean"), ab=("total", "mean"))
    return g["rl"], g["ab"]


def main():
    # unmod_1 is a 50nt library, unmod_2 is a 59nt library sharing the
    # same 50nt cores -- cut both to 50nt keys before averaging
    r1, a1 = load_rl(os.path.join(D, "GSM3130435_egfp_unmod_1.csv.gz"),
                     cut50=True)
    r2, a2 = load_rl(os.path.join(
        D, "GSM3130436_egfp_unmod_2.csv.gz"), cut50=True)
    base = r1.to_frame().join(r2.to_frame(), rsuffix="_2", how="outer")
    tab = pd.DataFrame({
        "mrl_unmod": base.mean(axis=1),
        "ab_unmod": a1.to_frame().join(a2.to_frame(), rsuffix="_2",
                                       how="outer").mean(axis=1)})

    for cond, f1, f2 in [
            ("mrl_pseudo", "GSM3130437_egfp_pseudo_1.csv.gz",
             "GSM3130438_egfp_pseudo_2.csv.gz"),
            ("mrl_m1pseudo", "GSM3130439_egfp_m1pseudo_1.csv.gz",
             "GSM3130440_egfp_m1pseudo_2.csv.gz")]:
        ra, aa = load_rl(os.path.join(D, f1), cut50=True)
        rb, ab = load_rl(os.path.join(D, f2), cut50=True)
        rr = ra.to_frame().join(rb.to_frame(), rsuffix="_2", how="outer")
        abn = aa.to_frame().join(ab.to_frame(), rsuffix="_2", how="outer")
        tab = tab.join(pd.DataFrame({cond: rr.mean(axis=1),
                                     cond.replace("mrl", "ab"):
                                         abn.mean(axis=1)}),
                       how="outer")

    tab = tab.reset_index().rename(columns={"index": "utr"})
    # filter: valid ACGT-only 50nt
    tab = tab[tab["utr"].str.fullmatch(r"[ACGT]{50}")]
    tab.to_csv(os.path.join(D, "utr_training_table.tsv"), sep="\t",
               index=False)
    print("utr_training_table.tsv: %d UTRs" % len(tab))
    print("label coverage:", {c: int(tab[c].notna().sum())
                              for c in tab.columns[1:]})

    des = pd.read_csv(os.path.join(D, "GSM3130443_designed_library.csv.gz"),
                      usecols=["utr", "rl"], low_memory=False)
    des["utr"] = des["utr"].astype(str).str.upper()
    des = des[des["utr"].str.fullmatch(r"[ACGT]{50}")]
    des.groupby("utr")["rl"].mean().reset_index().to_csv(
        os.path.join(D, "designed_test.tsv"), sep="\t", index=False)
    print("designed_test.tsv: %d UTRs (independent test)" % len(des))


if __name__ == "__main__":
    main()
