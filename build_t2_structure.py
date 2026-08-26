#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T4-v0.5: fast ViennaRNA structure features for the training table.

Per gene (from t2_training_table.tsv CDS):
  open_start45    : mean unpaired prob of first 45nt (local 150nt pf)
  selfcomp_max_exact / selfcomp_max_near : intramolecular duplex scan
  mfe_local5      : MFE of first 300nt (per-nt)
  mfe_local3      : MFE of last 300nt (per-nt)

Global MFE/helix stats intentionally skipped (too slow for 19k genes);
they stay in the candidate-evaluation path (T3 pipeline).

Incremental save every 500 genes (crash-safe).
Run: python -u build_t2_structure.py
Out: data/t2/t2_structure_features.tsv
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
if FP not in sys.path:
    sys.path.insert(0, FP)

import RNA as vrna  # noqa: E402

from features.structure import unpaired_probs, _mfe_unpaired, selfcomp_scan  # noqa: E402

TABLE = os.path.join(HERE, "data", "t2", "t2_training_table.tsv")

ENDO = os.path.join(HERE, "data", "t2", "t2_endogenous_cds_abundance.tsv")


def load_cds_map():
    m = {}
    with open(ENDO, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            g = r["gene"]
            if g not in m or int(r["cds_len"]) > len(m[g]):
                m[g] = r["cds"].upper()
    return m

OUT = os.path.join(HERE, "data", "t2", "t2_structure_features.tsv")


def local_openness(seq):
    local = seq[:150]
    pun = unpaired_probs(local)
    if pun is None:
        pun = _mfe_unpaired(local)
    first = pun[:45]
    return sum(first) / len(first) if first else 0.0


def local_mfe_per_nt(seq):
    w = seq[:300]
    fc = vrna.fold_compound(w)
    _, mfe = fc.mfe()
    return mfe / len(w)


def main():
    cds_map = load_cds_map()
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8"), delimiter="\t"))
    print("[struct] %d genes" % len(rows), flush=True)
    out_rows = []
    t0 = time.time()
    for k, r in enumerate(rows):
        cds = cds_map[r["gene"]].replace("T", "U")
        if k % 500 == 0 and k > 0:
            _save(out_rows)
            el = time.time() - t0
            print("[struct] %d/%d (%.1fs elapsed, ~%.1fs/gene)"
                  % (k, len(rows), el, el / k), flush=True)
        ex, nr = selfcomp_scan(cds)
        out_rows.append({
            "gene": r["gene"],
            "open_start45": round(local_openness(cds), 4),
            "selfcomp_max_exact": ex,
            "selfcomp_max_near": nr,
            "mfe_local5_pernt": round(local_mfe_per_nt(cds), 4),
            "mfe_local3_pernt": round(local_mfe_per_nt(cds[-300:]), 4),
        })
    _save(out_rows)
    print("[struct] done: %d rows -> %s (%.1fs)"
          % (len(out_rows), OUT, time.time() - t0), flush=True)


def _save(rows):
    cols = ["gene", "open_start45", "selfcomp_max_exact", "selfcomp_max_near",
            "mfe_local5_pernt", "mfe_local3_pernt"]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
