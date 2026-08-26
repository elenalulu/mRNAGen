#!/usr/bin/env python
"""Refine from GEMORNA starts for the 4 long proteins (LD OOM territory).

LinearDesign dies >=952aa on our benchmark machine, so gaa_pompe /
cov2_spike / spy_cas9 / factor_viii only have GEMORNA candidate pools --
which carry heavy manufacturing violations (FVIII ~10 restriction sites
per candidate). This closes the gap: top-5 GEMORNA reps (by oracle
composite) per protein get the same T5 greedy refinement as LD starts.

Output: data/t5/refined_gemorna.tsv (schema-compatible with
refined_candidates.tsv; lam_start column holds the GEMORNA rep id).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
sys.path.insert(0, HERE)

from gemorna_parse import parse_gemorna_output
from refine_t5 import (hard_ok, load_env, n_hp4, n_hp5, n_sites, refine,
                       score_batch)

GEMORNA_OUT = os.path.join(HERE, os.pardir, "github", "GEMORNA",
                           "gemorna_output.txt")
OUT = os.environ.get("GE_OUT",
       os.path.join(HERE, "data", "t5", "refined_gemorna.tsv"))
TOP_STARTS = 5
LONG = {952: "gaa_pompe", 1273: "cov2_spike", 1368: "spy_cas9",
        2351: "factor_viii"}


def strip_stop(seq):
    s = seq.upper().replace("U", "T")
    if len(s) % 3 == 0 and s[-3:] in ("TAA", "TAG", "TGA"):
        return s[:-3]
    return s


def main():
    table, codon_coef, syn, models = load_env()
    recs = parse_gemorna_output(os.path.abspath(GEMORNA_OUT))
    rows = []
    for plen, name in sorted(LONG.items()):
        pool = [r for r in recs if len(r["protein"]) == plen]
        seqs = [strip_stop(r["seq"]) for r in pool]
        scores, _, _ = score_batch(seqs, table, codon_coef, models)
        order = np.argsort(-np.asarray(scores))[:TOP_STARTS]
        print("[gem-ref] %s: %d reps, refining top %d (scores %s)" % (
            name, len(pool), TOP_STARTS,
            [round(float(scores[i]), 2) for i in order]), flush=True)
        for i in order:
            cds = seqs[i]
            t0 = time.time()
            out, s0, s1, n_moves = refine(
                cds, table, codon_coef, models,
                max_rounds=int(os.environ.get("MAX_ROUNDS", 8)))
            rows.append({
                "protein": name, "lam_start": "ge_rep%s" % pool[i]["rep"],
                "cds_len": len(cds), "n_moves": n_moves,
                "score_start": round(float(s0), 3),
                "score_end": round(float(s1), 3),
                "delta": round(float(s1 - s0), 3),
                "sites_start": n_sites(cds), "sites_end": n_sites(out),
                "hp4_start": n_hp4(cds), "hp4_end": n_hp4(out),
                "hp5_start": n_hp5(cds), "hp5_end": n_hp5(out),
                "hard_pass_end": int(hard_ok(out)),
                "cds_refined": out,
            })
            pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
            print("[gem-ref]   %s rep%s: %.2f -> %.2f (%+0.2f, %d moves) "
                  "sites %d->%d hp5 %d->%d [%.0fs]" % (
                      name, pool[i]["rep"], s0, s1, s1 - s0, n_moves,
                      n_sites(cds), n_sites(out), n_hp5(cds), n_hp5(out),
                      time.time() - t0), flush=True)
    print("[gem-ref] done: %d refined -> %s" % (len(rows), OUT), flush=True)


if __name__ == "__main__":
    main()
