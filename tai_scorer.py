#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tAI z-scorer for design score v2.5.

Empirical motivation (S3 wet-lab ablation, 2026-08-27, 67 sequences across
4 GEMORNA Science datasets): adding the tRNA adaptation index to the
objective lifts pooled Spearman from 0.583 to 0.656 at w_tai=1.0 with no
dataset regressing (grid optimum 0.708 at w=(1.0, 2.25); leave-one-dataset-
out confirms robustness). AUG accessibility showed no independent signal
and stays a display-only metric.

v2.5 objective:
    design score = z_global(nat) + z_global(CAI) + z_global(tAI)
                   - manufacturing penalties

Global calibration mirrors naturalness.Scorer: mean/std computed once over
the endogenous human CDS corpus, persisted next to composite_v2_stats.pkl.

Run once to build:  python tai_scorer.py
"""
import csv
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "feature_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compare_table import tai as tai_fn, load_trna_counts  # noqa: E402
from naturalness import ENDO  # noqa: E402

STATS_V25 = os.path.join(HERE, "models", "tai_v25_stats.pkl")


def build_tai_stats():
    """One-time: corpus-level tAI distribution over endogenous human CDS."""
    trna = load_trna_counts()
    wcache = {}
    vals = []
    with open(ENDO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["cds"].upper().replace("U", "T")
            tv = tai_fn(s, trna, wcache)
            if tv is not None:
                vals.append(tv)
    stats = {"tai_mean": float(np.mean(vals)),
             "tai_std": float(np.std(vals)),
             "n_corpus": len(vals)}
    os.makedirs(os.path.dirname(STATS_V25), exist_ok=True)
    with open(STATS_V25, "wb") as f:
        pickle.dump(stats, f)
    print("[tai] corpus n=%d  mean=%.4f +/- %.4f" %
          (len(vals), stats["tai_mean"], stats["tai_std"]))
    print("[tai] -> %s" % STATS_V25)


class TaiScorer(object):
    """Lazily-loaded global tAI z-transform (same contract as Scorer)."""

    def __init__(self):
        with open(STATS_V25, "rb") as f:
            st = pickle.load(f)
        self.mean, self.std = st["tai_mean"], st["tai_std"]
        self._trna = load_trna_counts()
        self._wcache = {}

    def tai(self, dna_seq):
        return tai_fn(dna_seq.upper().replace("U", "T"),
                      self._trna, self._wcache)

    def z_tai(self, seqs):
        vals = [self.tai(s) or 0.0 for s in seqs]
        return (np.asarray(vals, dtype=float) - self.mean) / self.std


if __name__ == "__main__":
    build_tai_stats()
