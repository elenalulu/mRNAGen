#!/usr/bin/env python
"""Naturalness scorer: codon-pair log-likelihood vs endogenous CDS corpus.

Rebuilt (not copied) from GEMORNA's strongest external signal: a sequence's
similarity to natural human coding context predicts expression/stability
better than any other feature we tested (S3 external validation, 2026-08-25).

design score = z_global(naturalness) + z_global(CAI) - manufacturing
penalties. Global z-stats are computed against the 18,963 endogenous CDS
corpus so scores are comparable across sequences, pools and runs.

Run once to build stats:  python naturalness.py
"""
import csv
import os
import pickle
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENDO = os.path.join(HERE, "data", "t2", "t2_endogenous_cds_abundance.tsv")
STATS = os.path.join(HERE, "models", "composite_v2_stats.pkl")
_V = 4096  # codon-pair vocabulary


def _iter_pairs(seq):
    for i in range(0, len(seq) - 6, 3):
        yield seq[i:i + 6]


def naturalness_raw(seq, pair_count, total):
    lp = 0.0
    n = 0
    for p in _iter_pairs(seq):
        lp += np.log((pair_count.get(p, 0) + 1) / (total + _V))
        n += 1
    return lp / max(1, n)


def build_stats():
    """One-time: count codon pairs + global z-stats over endogenous corpus."""
    import sys
    sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
    from features.codon_metrics import get_default_table

    pair_count = Counter()
    cai_vals, nat_vals = [], []
    table = get_default_table()
    with open(ENDO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["cds"].upper().replace("U", "T")
            for p in _iter_pairs(s):
                pair_count[p] += 1
    total = sum(pair_count.values())
    with open(ENDO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["cds"].upper().replace("U", "T")
            nat_vals.append(naturalness_raw(s, pair_count, total))
            cai_vals.append(table.cai(s.replace("T", "U")))
    stats = {
        "pair_count": dict(pair_count),
        "total": total,
        "nat_mean": float(np.mean(nat_vals)),
        "nat_std": float(np.std(nat_vals)),
        "cai_mean": float(np.mean(cai_vals)),
        "cai_std": float(np.std(cai_vals)),
    }
    os.makedirs(os.path.dirname(STATS), exist_ok=True)
    with open(STATS, "wb") as f:
        pickle.dump(stats, f)
    print("[nat] corpus stats: %d pairs, nat %.2f+/-%.2f, cai %.3f+/-%.3f"
          % (total, stats["nat_mean"], stats["nat_std"],
             stats["cai_mean"], stats["cai_std"]))
    print("[nat] -> %s" % STATS)


class Scorer:
    """Lazily-loaded naturalness scorer with global z-transforms."""

    def __init__(self):
        with open(STATS, "rb") as f:
            st = pickle.load(f)
        self.pc = st["pair_count"]
        self.total = st["total"]
        self.nat_mean, self.nat_std = st["nat_mean"], st["nat_std"]
        self.cai_mean, self.cai_std = st["cai_mean"], st["cai_std"]

    def nat(self, seq):
        return naturalness_raw(seq, self.pc, self.total)

    def z_nat(self, seqs):
        v = np.array([self.nat(s) for s in seqs])
        return (v - self.nat_mean) / self.nat_std

    def z_cai(self, cai_vals):
        v = np.asarray(cai_vals, dtype=float)
        return (v - self.cai_mean) / self.cai_std


if __name__ == "__main__":
    build_stats()
