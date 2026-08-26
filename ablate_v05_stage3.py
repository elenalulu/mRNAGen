#!/usr/bin/env python
"""Oracle v0.5 ablation stage 3: RNA-FM embeddings (the last card for decay).

Configs (same 5-fold CV Spearman protocol as stage 2):
  v0(11)          baseline, recomputed for seed-consistency
  v0+emb(651)     v0 + 640-dim RNA-FM mean-pooled embedding (500nt 5' trunc)
  v0+st+emb(656)  everything

Run: python -u ablate_v05_stage3.py
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
NPZ = os.path.join(HERE, "data", "t2", "t2_rnafm500.npz")
OUT = os.path.join(HERE, "data", "t2", "ablation_v05_stage3.tsv")

V0 = ["opt_293T", "opt_hela", "opt_rpe", "opt_293T_orfome",
      "opt_k562_orfome", "opt_k562_slam",
      "cai", "gc_global", "gc3", "enc", "cds_len"]
STRUCT_COLS = ["open_start45", "selfcomp_max_exact", "selfcomp_max_near",
               "mfe_local5_pernt", "mfe_local3_pernt"]
HARMONIZE = {"decay_hela": -1.0, "decay_rpe": -1.0}
HEADS = ["expr", "decay_293T", "decay_hela", "decay_rpe", "decay_k562_slam"]


def cv_spearman(X, y, n_splits=5, seed=13, max_iter=300):
    preds = np.zeros(len(y))
    for tr, te in KFold(n_splits, shuffle=True,
                        random_state=seed).split(X):
        m = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=0.08, min_samples_leaf=40,
            l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return spearmanr(preds, y).correlation


def main():
    df = pd.read_csv(TABLE, sep="\t").merge(
        pd.read_csv(STRUCT, sep="\t"), on="gene", how="left")
    df["expr"] = np.log10(df["abundance_ppm"].clip(lower=1e-6))

    z = np.load(NPZ, allow_pickle=True)
    genes = list(z["gene_ids"])
    emb = z["embeddings"]
    print("[abl3] embeddings: %s x %s" % (emb.shape, emb.dtype))
    gidx = {g: i for i, g in enumerate(genes)}
    df = df[df.gene.isin(gidx)].reset_index(drop=True)
    E = np.stack([emb[gidx[g]] for g in df.gene])

    X0 = df[V0].to_numpy()
    XS = df[STRUCT_COLS].to_numpy()
    configs = [
        ("v0(11)", X0),
        ("v0+emb(651)", np.hstack([X0, E])),
        ("v0+st+emb(656)", np.hstack([X0, XS, E])),
    ]
    rows = []
    for head in HEADS:
        y = df[head].to_numpy() * HARMONIZE.get(head, 1.0)
        mask = ~np.isnan(y)
        for name, X in configs:
            Xc = X[mask]
            yc = y[mask]
            # drop rows with NaN structure cols for the st config
            if "st" in name:
                ok2 = ~np.isnan(Xc[:, -641:-636].any(axis=1)) if False else \
                    ~np.isnan(XS[mask]).any(axis=1)
                Xc, yc = Xc[ok2], yc[ok2]
            rho = cv_spearman(Xc, yc)
            rows.append({"head": head, "features": name, "n": len(yc),
                         "spearman_cv": round(rho, 4),
                         "pass_0.5": "PASS" if rho >= 0.5 else "fail"})
            print("[abl3] %-16s %-14s rho=%.4f (n=%d)" % (
                head, name, rho, len(yc)), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, sep="\t", index=False)
    print("[abl3] -> %s" % OUT)


if __name__ == "__main__":
    main()
