#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified independent pipeline: start pool -> refine -> selfcomp repair.

Reads data/deliverable/start_pool.tsv (DNA Chisel + Ensembl native +
our CAI-max starts; see start_pool.py) and, for every start, runs:

  1. greedy refinement on composite v2 (25 rounds, our objective)
  2. surgical self-complementarity repair (Plan C, gates preserved)

Output: data/deliverable/independent_candidates.tsv
Incremental and crash-proof (one row saved per start).

Run with the dnachisel venv (has pandas/numpy/joblib/sklearn):
  D:/WorkBuddy/home/binaries/python/envs/dnachisel/Scripts/python.exe \
      pipeline_independent.py
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
for p in (FP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from refine_t5 import load_env, refine, n_sites, n_hp4, n_hp5  # noqa: E402
from naturalness import Scorer  # noqa: E402
from repair_selfcomp import (aa_codon_map, repair as sc_repair,  # noqa: E402
                             scan_duplexes)

OUT = os.path.join(HERE, "data", "deliverable")
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "25"))

FIELDS = ["protein", "source", "start_score", "refined_score", "z_nat",
          "cai", "sites", "hp4", "hp5", "gc", "selfcomp_before",
          "selfcomp_after", "repair_swaps", "repair_score", "secs", "seq"]


def save_results(out):
    with open(os.path.join(OUT, "independent_candidates.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)


def main():
    t, codon_coef, syn, models = load_env()
    scorer = Scorer()
    cmap = aa_codon_map(t)
    pool_path = os.path.join(OUT, "start_pool.tsv")
    with open(pool_path, encoding="utf-8") as f:
        pool = list(csv.DictReader(f, delimiter="\t"))
    out = []
    for r in pool:
        name, src, seq = r["protein"], r["source"], r["seq"].upper()
        t0 = time.time()
        try:
            end, s0, s1, n_moves = refine(seq, t, codon_coef, models,
                                          max_rounds=MAX_ROUNDS)
            z_nat = float(scorer.z_nat([end])[0])
            cai = t.cai(end.replace("T", "U"))
            sites, hp4, hp5 = n_sites(end), n_hp4(end), n_hp5(end)
            gc = (end.count("G") + end.count("C")) / len(end)
            sc_b = scan_duplexes(end)[0]
            repaired, n_swaps, _a, _sa, _b, after = sc_repair(
                end, t, scorer, cmap)
            sc_a = scan_duplexes(repaired)[0]
            out.append({
                "protein": name, "source": src,
                "start_score": round(s0, 3), "refined_score": round(s1, 3),
                "z_nat": round(z_nat, 2), "cai": round(cai, 3),
                "sites": sites, "hp4": hp4, "hp5": hp5,
                "gc": round(gc, 3), "selfcomp_before": sc_b,
                "selfcomp_after": sc_a, "repair_swaps": n_swaps,
                "repair_score": round(after["comp"], 3),
                "secs": round(time.time() - t0, 1), "seq": repaired})
            save_results(out)
            print("[ind] %-12s %-8s %.2f -> %.2f (moves=%d) nat=%+.2f "
                  "cai=%.3f sites=%d hp5=%d sc %d->%d [%.0fs]"
                  % (name, src, s0, s1, n_moves, z_nat, cai, sites, hp5,
                     sc_b, sc_a, time.time() - t0), flush=True)
        except Exception as e:
            print("[ind] ERROR %s/%s: %r" % (name, src, e), flush=True)
    print("[ind] done -> %s (%d candidates)"
          % (os.path.join(OUT, "independent_candidates.tsv"), len(out)))


if __name__ == "__main__":
    main()
