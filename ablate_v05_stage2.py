#!/usr/bin/env python
"""Oracle v0.5 ablation stage 2: v0 features vs v0+structure.

Structure features (ViennaRNA, 18,923 genes): open_start45,
selfcomp_max_exact, selfcomp_max_near, mfe_local5_pernt, mfe_local3_pernt.

Same protocol as train_oracle_v0 (5-fold CV, Spearman on held-out,
harmonized stability-direction labels) so numbers are directly comparable.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "data", "t2", "t2_training_table.tsv")
STRUCT = os.path.join(HERE, "data", "t2", "t2_structure_features.tsv")
OUT = os.path.join(HERE, "data", "t2", "ablation_v05_stage2.tsv")

V0 = ["opt_293T", "opt_hela", "opt_rpe", "opt_293T_orfome",
      "opt_k562_orfome", "opt_k562_slam",
      "cai", "gc_global", "gc3", "enc", "cds_len"]
STRUCT_COLS = ["open_start45", "selfcomp_max_exact", "selfcomp_max_near",
               "mfe_local5_pernt", "mfe_local3_pernt"]
V05 = V0 + STRUCT_COLS

HARMONIZE = {"decay_hela": -1.0, "decay_rpe": -1.0}


def cv_spearman(df, feats, label, n_splits=5, seed=13):
    y_all = df[label].to_numpy() * HARMONIZE.get(label, 1.0)
    X = df[feats].to_numpy()
    mask = ~np.isnan(y_all)
    X, y = X[mask], y_all[mask]
    preds = np.zeros(len(y))
    for tr, te in KFold(n_splits, shuffle=True,
                        random_state=seed).split(X):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, max_depth=None,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return spearmanr(preds, y).correlation, len(y)


def main():
    df = pd.read_csv(TABLE, sep="\t").merge(
        pd.read_csv(STRUCT, sep="\t"), on="gene", how="left")
    miss = df[STRUCT_COLS].isna().any(axis=1).sum()
    print("[abl2] merged: %d rows, structure-missing: %d" % (len(df), miss))
    df = df.dropna(subset=STRUCT_COLS)

    labels = {"expr": None, "decay_293T": None, "decay_hela": None,
              "decay_rpe": None, "decay_k562_slam": None}
    # expr label from abundance
    df["expr"] = np.log10(df["abundance_ppm"].clip(lower=1e-6))

    rows = []
    for label in labels:
        for name, feats in [("v0(11)", V0), ("v0+struct(16)", V05)]:
            rho, n = cv_spearman(df, feats, label)
            rows.append({"head": label, "features": name, "n": n,
                         "spearman_cv": round(rho, 4),
                         "pass_0.5": "PASS" if rho >= 0.5 else "fail"})
            print("[abl2] %-16s %-14s rho=%.4f (n=%d)" % (
                label, name, rho, n), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print("[abl2] -> %s" % OUT)

    # feature importance snapshot for structure columns (decay_293T head)
    y = df["decay_293T"].to_numpy()
    X = df[V05].to_numpy()
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                      min_samples_leaf=40,
                                      random_state=13).fit(X, y)
    # HistGBM has no direct feature_importances_; use permutation-lite via
    # correlation of partial residuals is overkill -- report null-model
    # single-feature baselines for the 5 structure cols instead
    print()
    print("[abl2] structure single-feature Spearman vs decay_293T:")
    for c in STRUCT_COLS:
        d = df[[c, "decay_293T"]].dropna()
        print("   %-20s %.3f" % (c, spearmanr(d[c], d["decay_293T"]).correlation))


if __name__ == "__main__":
    main()
