#!/usr/bin/env python
"""First non-circular external validation of the CDS oracle.

Data: GEMORNA Science SI Data S3 -- same-protein codon variants with
wet-lab labels:
  Bicknell GFP  (n=30, half-life HEK293)     -> stability axis
  Bicknell Luc  (n=5,  AUC expr mouse)
  Leppek NanoLuc(n=24, expr 6h)
  GEMORNA Fluc  (n=8,  expr 24h HEK293T)     -> expression axis

Predictors compared:
  our_expr (oracle v0 expr head), our_stab (mean decay heads),
  our_composite (no manufacturing penalties, expr 0.6 + stab 0.4),
  CAI (LinearDesign's objective), MFE (theirs), Naturalness (theirs).

If our oracle Spearman > CAI's on these REAL labels, the core thesis
("learned oracle beats proxy objectives") gets its first external win.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
sys.path.insert(0, HERE)

from refine_t5 import DECAY_HEADS, FEATS, feature_vec, load_env  # noqa: E402

SI = os.path.join(HERE, "data", "t3_gemorna_si",
                  "science.adr8470_data-s3.xlsx")
OUT = os.path.join(HERE, "data", "t3_gemorna_si", "s3_validation.tsv")

AXIS = {"Bickness et al. GFP": "stability",
        "Bickness et al. Luciferase": "expression",
        "Leppek et al. NanoLuc": "expression",
        "This study Fluc": "expression"}


def main():
    table, codon_coef, syn, models = load_env()
    xl = pd.ExcelFile(SI)
    rows = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        seq_col = [c for c in df.columns if "oding" in c][0]
        label_col = [c for c in df.columns
                     if "half" in c.lower() or "xpress" in c.lower()
                     or "AUC" in c][0]
        seqs = [s.upper().replace("U", "T") for s in df[seq_col]]
        X = np.array([feature_vec(s, table, codon_coef) for s in seqs])
        expr = models["expr"].predict(X)
        stab = np.mean([models[h].predict(X) for h in DECAY_HEADS], axis=0)
        comp = 0.6 * expr + 0.4 * stab
        cai = X[:, FEATS.index("cai")]
        mfe = df["MFE"].to_numpy() if "MFE" in df else None
        nat = (df["Naturalness"].to_numpy()
               if "Naturalness" in df.columns else None)
        y = df[label_col].to_numpy()
        entry = {"dataset": sheet, "n": len(df), "axis": AXIS[sheet],
                 "label": label_col,
                 "our_expr": spearmanr(expr, y).correlation,
                 "our_stab": spearmanr(stab, y).correlation,
                 "our_composite": spearmanr(comp, y).correlation,
                 "CAI": spearmanr(cai, y).correlation}
        if mfe is not None:
            entry["MFE_theirs"] = spearmanr(mfe, y).correlation
        if nat is not None:
            entry["Naturalness_theirs"] = spearmanr(nat, y).correlation
        rows.append(entry)
        print("[%s] n=%d axis=%s" % (sheet, len(df), AXIS[sheet]))
        for k, v in entry.items():
            if isinstance(v, float):
                print("   %-20s %+.3f" % (k, v))
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
