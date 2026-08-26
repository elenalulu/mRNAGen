#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T4: Oracle v0 -- GBDT ranking models for expression & stability heads.

Design (mRNA_design_engine_v1_plan.md sec 3):
  v0 = engineered features + histogram gradient boosting (sklearn's
  LightGBM port -- zero new dependencies, runs read-only on the conda env with ViennaRNA).

Heads (one model each, 5-fold CV, Spearman on held-out raw labels):
  expr        : log10(PaxDb protein abundance + eps)
  decay_293T / decay_hela / decay_rpe / decay_k562_slam : Wu 2019 decay rates

Extra honesty checks:
  - cross-cell-line transfer: train decay on 293T -> evaluate on HeLa/K562
    (common genes) -- direction consistency per plan sec 2.2
  - feature importances (remember the Wu sign-convention caveat:
    positive opt_* = optimal-codon direction)
  - self-play isolation: training table contains ONLY endogenous genes
    (no LinearDesign/GEMORNA candidates), per plan sec 2.2

Run: python train_oracle_v0.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "data", "t2", "t2_training_table.tsv")
MODEL_DIR = os.path.join(HERE, "models")
REPORT = os.path.join(HERE, "data", "t2", "oracle_v0_report.tsv")

FEATURES = ["opt_293T", "opt_hela", "opt_rpe", "opt_293T_orfome",
            "opt_k562_orfome", "opt_k562_slam",
            "cai", "gc_global", "gc3", "enc", "cds_len"]
HEADS = ["expr", "decay_293T", "decay_hela", "decay_rpe", "decay_k562_slam"]

# Empirical label harmonization (2026-08-24): the Wu 2019 xlsx sheets use two
# opposite sign conventions. Evidence: raw corr(decay_293T, decay_hela)=-0.724,
# corr(decay_293T, decay_rpe)=-0.723, corr(decay_293T, decay_k562_slam)=+0.587;
# biologically cross-cell-line decay must correlate POSITIVELY. Also
# corr(opt_293T, V_293T)=+0.13 (opt positive = stabilizing per paper) implies
# 293T/K562 sheets are stability-scaled, HeLa/RPE are rate-scaled.
# Flip HeLa/RPE so every label is "stability-direction" (higher = more stable).
# TODO: verify against Wu 2019 Fig 1 panels before publishing anything.
HARMONIZE = {"decay_hela": -1.0, "decay_rpe": -1.0}


def load():
    df = pd.read_csv(TABLE, sep="\t")
    df["expr"] = np.log10(df["abundance_ppm"].clip(lower=1e-6))
    for col, sign in HARMONIZE.items():
        df[col] = df[col] * sign
    return df


def cv_head(df, head, n_splits=5, seed=42):
    d = df[df[head].notna() & np.isfinite(df[head])].copy()
    X = d[FEATURES].values
    y = d[head].values
    yq = pd.Series(y).rank(pct=True).values  # quantile target for training
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(len(d), np.nan)
    importances = np.zeros(len(FEATURES))
    for tr, te in kf.split(X):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_depth=None,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], yq[tr])
        preds[te] = m.predict(X[te])
        # permutation-free importance via variance of predictions on shuffled
        # features is expensive; use binned split counts via a surrogate:
        importances += tree_importance(m, X[tr])
    rho = spearmanr(preds, y).correlation
    importances /= n_splits
    return rho, len(d), importances, d, preds


def tree_importance(model, X):
    """HGB has no feature_importances_; approximate by single-feature
    ablation R2 drop on a subsample (cheap, 11 features)."""
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
    base = model.predict(X[idx])
    base_var = base.var()
    imp = np.zeros(X.shape[1])
    if base_var <= 0:
        return imp
    for j in range(X.shape[1]):
        Xp = X[idx].copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp[j] = 1.0 - model.predict(Xp).var() / base_var
    return imp


def main():
    df = load()
    print("[t4] table: %d rows, features=%d" % (len(df), len(FEATURES)))
    print()
    results = []
    trained = {}
    for head in HEADS:
        rho, n, imp, d, preds = cv_head(df, head)
        print("[t4] %-16s n=%6d  5-fold held-out Spearman rho = %.3f"
              % (head, n, rho))
        top = sorted(zip(FEATURES, imp), key=lambda x: -x[1])[:5]
        print("      top features: %s"
              % ", ".join("%s %.2f" % (f, v) for f, v in top))
        results.append({"head": head, "n": n, "spearman_cv": round(rho, 4),
                        **{"imp_" + f: round(v, 3) for f, v in
                           zip(FEATURES, imp)}})
        d = d.assign(pred=pd.Series(preds).values)
        trained[head] = d
    # final models on all data (for downstream T5 use)
    os.makedirs(MODEL_DIR, exist_ok=True)
    import joblib
    for head in HEADS:
        d = df[df[head].notna() & np.isfinite(df[head])]
        yq = pd.Series(d[head].values).rank(pct=True).values
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, min_samples_leaf=40,
            l2_regularization=1.0, random_state=42)
        m.fit(d[FEATURES].values, yq)
        joblib.dump(m, os.path.join(MODEL_DIR, "oracle_v0_%s.joblib" % head))
    print()
    print("[t4] full-data models -> %s/oracle_v0_<head>.joblib" % MODEL_DIR)

    # ---------- cross-cell-line transfer (direction consistency) ----------
    print()
    print("[t4] cross-cell-line transfer: train 293T -> test others")
    d293 = df[df["decay_293T"].notna()]
    m = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, min_samples_leaf=40,
        l2_regularization=1.0, random_state=42)
    yq = pd.Series(d293["decay_293T"].values).rank(pct=True).values
    m.fit(d293[FEATURES].values, yq)
    for other in ["decay_hela", "decay_rpe", "decay_k562_slam"]:
        do = df[df[other].notna() & df["decay_293T"].notna()]
        p = m.predict(do[FEATURES].values)
        rho = spearmanr(p, do[other]).correlation
        print("      293T -> %-16s n=%5d  rho = %.3f"
              % (other, len(do), rho))
        results.append({"head": "transfer_293T_to_%s" % other,
                        "n": len(do), "spearman_cv": round(rho, 4)})

    out = pd.DataFrame(results)
    out.to_csv(REPORT, sep="\t", index=False)
    print()
    print("[t4] report -> %s" % REPORT)
    print()
    print("[t4] M1 gate check (plan: held-out Spearman >= 0.5):")
    for r in results[:len(HEADS)]:
        status = "PASS" if r["spearman_cv"] >= 0.5 else "FAIL"
        print("      %-16s %.3f  %s" % (r["head"], r["spearman_cv"], status))


if __name__ == "__main__":
    main()
