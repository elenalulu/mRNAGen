#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diversity Top-K over the INDEPENDENT candidate pool only.

Reads data/deliverable/independent_candidates.tsv and, per protein,
selects K=5 candidates with the same greedy max-min philosophy as
select_topk.py but restricted to our clean pool and a reduced diversity
space that needs no ViennaRNA:

  blend: alpha * z(score) + (1 - alpha) * min_l2(z_feats, selected)
  feats: cai, gc, enc, upa_odds, selfcomp_after, sites, hp_max
  first pick: highest score, selfcomp tiebreak within 0.5 raw points

Output: data/deliverable/independent_topk.tsv (deliverable, per protein).
"""
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
for p in (FP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from features.codon_metrics import get_default_table  # noqa: E402

OUT = os.path.join(HERE, "data", "deliverable")
K = 5
ALPHA = 0.5
FEATS = ["cai", "gc", "enc", "upa_odds", "selfcomp_after",
         "sites", "hp_max"]


def upa_odds(seq):
    n = len(seq)
    t, a = seq.count("T"), seq.count("A")
    if t == 0 or a == 0:
        return 0.0
    return (seq.count("TA") * n) / (t * a)


def enc(seq, table):
    return table.enc(seq.replace("T", "U"))


def hp_max_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def select_topk(df_rows, table, k=K, alpha=ALPHA):
    """Greedy max-min diversity selection (independent pool only)."""
    n = len(df_rows)
    if n == 0:
        return []
    score = [float(r["repair_score"]) for r in df_rows]
    feats = []
    for r in df_rows:
        seq = r["seq"]
        feats.append([float(r["cai"]), float(r["gc"]),
                      enc(seq, table), upa_odds(seq),
                      float(r["selfcomp_after"]), float(r["sites"]),
                      float(hp_max_run(seq))])
    mean = sum(score) / n
    sd = (sum((x - mean) ** 2 for x in score) / n) ** 0.5
    zs = [(s - mean) / max(1e-9, sd) for s in score]
    zf = []
    for j in range(len(FEATS)):
        col = [f[j] for f in feats]
        m = sum(col) / n
        s = (sum((x - m) ** 2 for x in col) / n) ** 0.5
        zf.append([(x - m) / max(1e-9, s) for x in col])
    smax = max(score)
    elig = [i for i in range(n) if score[i] >= smax - 0.5]
    selected = [min(elig, key=lambda i: feats[i][4])]
    while len(selected) < min(k, n):
        best, best_v = None, -math.inf
        for i in range(n):
            if i in selected:
                continue
            dmin = min(
                math.sqrt(sum((zf[j][i] - zf[j][s]) ** 2
                              for j in range(len(FEATS))))
                for s in selected)
            v = alpha * zs[i] + (1 - alpha) * dmin
            if v > best_v:
                best, best_v = i, v
        selected.append(best)
    return [df_rows[i] for i in selected]


def main():
    table = get_default_table()
    with open(os.path.join(OUT, "independent_candidates.tsv"),
              encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f, delimiter="\t"))
    by_protein = {}
    for r in all_rows:
        by_protein.setdefault(r["protein"], []).append(r)
    out = []
    for protein in sorted(by_protein):
        picked = select_topk(by_protein[protein], table)
        for i, r in enumerate(picked, 1):
            out.append({"protein": protein, "rank": i,
                        "source": r["source"], "score": r["repair_score"],
                        "z_nat": r["z_nat"], "cai": r["cai"],
                        "gc": r["gc"], "sites": r["sites"],
                        "selfcomp": r["selfcomp_after"],
                        "seq": r["seq"]})
    with open(os.path.join(OUT, "independent_topk.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        for r in out:
            w.writerow(r)
    print("[sel] -> %s (%d candidates, %d proteins)"
          % (os.path.join(OUT, "independent_topk.tsv"), len(out),
             len(by_protein)))


if __name__ == "__main__":
    main()
