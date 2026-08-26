#!/usr/bin/env python
"""GEMORNA S2 deep-dive: what does each optimizer actually optimize?

S2 = 3,765 proteins x {Natural, GEMORNA, CAI-optimized, LinearDesign(l=1),
Random} with their reported metrics (CAI/GC/rare-codon/U%/MFE/...).

This script:
  1. verifies our CAI/naturalness implementations against their reported
     CAI sheet (implementation cross-validation)
  2. checks whether their LinearDesign(l=1) sequences match our own LD
     outputs for shared benchmark proteins (replication check)
  3. computes our naturalness z-score for every variant -- revealing what
     each optimizer actually optimizes, through our open scorer

Output: data/t3_gemorna_si/s2_analysis.tsv + s2_implementation_check.tsv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
sys.path.insert(0, HERE)

from features.codon_metrics import get_default_table  # noqa: E402
from naturalness import Scorer  # noqa: E402
from refine_t5 import load_env  # noqa: E402

SI = os.path.join(HERE, "data", "t3_gemorna_si", "science.adr8470_data-s2.xlsx")
OUT = os.path.join(HERE, "data", "t3_gemorna_si", "s2_analysis.tsv")
OUT2 = os.path.join(HERE, "data", "t3_gemorna_si",
                    "s2_implementation_check.tsv")

VARIANTS = ["Natural", "GEMORNA", "CAI-optimized",
            "LinearDesign (lambda=1)", "Random"]


def main():
    seqs = pd.read_excel(SI, sheet_name="Sequences")
    cai_sheet = pd.read_excel(SI, sheet_name="CAI")
    print("[s2] %d proteins x %d variants" % (len(seqs), len(VARIANTS)))

    table = get_default_table()
    scorer = Scorer()

    rows = []
    impl_check = {"ourCAI~theirCAI": []}
    for vi, v in enumerate(VARIANTS):
        zs_nat, cai_ours, cai_theirs, gc = [], [], [], []
        for i in range(len(seqs)):
            s = str(seqs.iloc[i][v]).upper().replace("U", "T")
            zs_nat.append(scorer.z_nat([s])[0])
            cai_ours.append(table.cai(s.replace("T", "U")))
            gc.append((s.count("G") + s.count("C")) / len(s))
            if vi == 0:
                pass
        cai_t = cai_sheet[v].to_numpy(dtype=float)
        m = ~np.isnan(cai_t)
        r = spearmanr(np.array(cai_ours)[m], cai_t[m]).correlation
        impl_check["ourCAI~theirCAI"].append((v, round(r, 4),
                                              int(m.sum())))
        rows.append({
            "variant": v,
            "z_nat_mean": round(float(np.mean(zs_nat)), 3),
            "z_nat_std": round(float(np.std(zs_nat)), 3),
            "cai_ours_mean": round(float(np.mean(cai_ours)), 4),
            "cai_theirs_mean": round(float(np.nanmean(cai_t)), 4),
            "gc_mean": round(float(np.mean(gc)), 4)})
        print("[s2] %-24s z_nat %+.3f  CAI(ours %.4f / theirs %.4f)  GC %.3f"
              % (v, np.mean(zs_nat), np.mean(cai_ours),
                 np.nanmean(cai_t), np.mean(gc)), flush=True)
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)

    # implementation check table
    chk = pd.DataFrame(
        [{"variant": v, "spearman_ourCAI_vs_theirCAI": r, "n": n}
         for v, r, n in impl_check["ourCAI~theirCAI"]])
    chk.to_csv(OUT2, sep="\t", index=False)
    print()
    print("[s2] CAI implementation check (ours vs their sheet):")
    for _, r in chk.iterrows():
        print("   %-24s rho=%.4f (n=%d)" % (r["variant"],
                                            r["spearman_ourCAI_vs_theirCAI"],
                                            r["n"]))

    # LinearDesign replication spot-check on shared benchmark proteins
    import glob
    from ld_parse import parse_lineardesign_output
    ld_dir = os.path.join(HERE, os.pardir, "github", "LinearDesign-main",
                          "ld_outputs")
    ours = {}
    for fp in glob.glob(os.path.join(ld_dir, "*.txt")):
        pname = os.path.splitext(os.path.basename(fp))[0]
        for rec in parse_lineardesign_output(fp):
            if float(rec["lam"]) == 1.0:
                ours[pname] = rec["seq"].upper().replace("U", "T")
    matched = 0
    checked = 0
    for i in range(len(seqs)):
        prot = str(seqs.iloc[i]["Protein"]).rstrip("*")
        for pname, our_seq in ours.items():
            bench = open(os.path.join(
                HERE, "data", "proteins", "benchmark",
                pname + ".txt")).read().strip().rstrip("*")
            if prot == bench:
                checked += 1
                their_ld = str(seqs.iloc[i][
                    "LinearDesign (lambda=1)"]).upper().replace("U", "T")
                if their_ld == our_seq.rstrip("*").replace("U", "T")[:-3] or \
                        their_ld[:-3] == our_seq[:-3] or their_ld == our_seq:
                    matched += 1
                else:
                    diff = sum(1 for a, b in zip(their_ld, our_seq) if a != b)
                    print("   [s2] LD mismatch %s: %d/%d nt differ"
                          % (pname, diff, min(len(their_ld), len(our_seq))))
    print("[s2] LinearDesign(l=1) replication: %d/%d shared proteins exact"
          % (matched, checked))


if __name__ == "__main__":
    main()
