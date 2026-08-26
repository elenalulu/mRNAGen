#!/usr/bin/env python
"""Structure post-hoc evaluation for GEMORNA-start refined candidates.

Same role as the LD-refined posthoc: verify greedy refinement did not
sacrifice structure quality, and produce the structure features
(selfcomp / open45 / mfe) that select_topk needs for its diversity space.

Compares each refined candidate against its GEMORNA original (features
already in all_candidates_features.tsv).

Output: data/t5/structure_posthoc_gemorna.tsv
Columns mirror structure_posthoc.tsv (protein, lam, *_ref, *_ld) so
select_topk can consume both with the same code path.
"""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))

from features.structure import structure_features

for _c in ("refined_gemorna_v2.tsv", "refined_gemorna_r25.tsv",
           "refined_gemorna.tsv"):
    REF = os.path.join(HERE, "data", "t5", _c)
    if os.path.exists(REF):
        break
ALL_FEATS = os.path.join(HERE, "data", "all_candidates_features.tsv")
_v2 = os.path.join(HERE, "data", "t5", "refined_gemorna_v2.tsv")
OUT = (os.path.join(HERE, "data", "t5", "structure_posthoc_gemorna_v2.tsv")
       if os.path.exists(_v2)
       else os.path.join(HERE, "data", "t5",
                         "structure_posthoc_gemorna_r25.tsv"))

LEN2NAME = {952: "gaa_pompe", 1273: "cov2_spike", 1368: "spy_cas9",
            2351: "factor_viii"}


def main():
    ref = pd.read_csv(REF, sep="\t")
    allf = pd.read_csv(ALL_FEATS, sep="\t")
    allf["hp_max"] = allf[["hp_max_A", "hp_max_C", "hp_max_G",
                           "hp_max_U"]].max(axis=1)
    rows = []
    for _, r in ref.iterrows():
        t0 = time.time()
        f = structure_features(r.cds_refined.replace("T", "U"))
        plen = {952: 952, 1273: 1273, 1368: 1368, 2351: 2351}
        # original = best-scoring GEMORNA rep for that protein is the
        # start; original features taken per-protein mean for context
        orig = allf[(allf.generator == "gemorna") &
                    (allf.protein == "len%d" % {
                        v: k for k, v in LEN2NAME.items()}[r.protein])]
        o = orig.iloc[0]
        rows.append({
            "protein": r.protein, "lam": r.lam_start,
            "mfe_ref": round(f["mfe"], 1), "mfe_ld": round(o.mfe, 1),
            "selfcomp_near_ref": f["selfcomp_max_near"],
            "selfcomp_near_ld": o.selfcomp_max_near,
            "selfcomp_ex_ref": f["selfcomp_max_exact"],
            "selfcomp_ex_ld": o.selfcomp_max_exact,
            "helix_ref": f["longest_helix_mfe"],
            "helix_ld": o.longest_helix_mfe,
            "open45_ref": round(f["open_start45"], 3),
            "open45_ld": round(o.open_start45, 3),
        })
        pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
        print("[ge-posthoc] %s %s: selfcomp %s->%s open45 %.2f->%.2f "
              "mfe %.0f->%.0f [%.0fs]" % (
                  r.protein, r.lam_start,
                  o.selfcomp_max_near, f["selfcomp_max_near"],
                  o.open_start45, f["open_start45"],
                  o.mfe, f["mfe"], time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    print()
    print("[ge-posthoc] aggregate (refined vs GEMORNA originals):")
    print("  selfcomp_near: %.1f -> %.1f" % (
        df.selfcomp_near_ld.mean(), df.selfcomp_near_ref.mean()))
    print("  open_start45: %.3f -> %.3f" % (
        df.open45_ld.mean(), df.open45_ref.mean()))
    print("  longest helix: %.1f -> %.1f" % (
        df.helix_ld.mean(), df.helix_ref.mean()))
    print("  -> %s (%d rows)" % (OUT, len(df)))


if __name__ == "__main__":
    main()
