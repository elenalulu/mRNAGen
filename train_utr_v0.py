#!/usr/bin/env python
"""UTR-2/3: v0 UTR expression model (k-mer GBDT) + cross-modality analysis.

Experiments:
  A. mrl_unmod   80/20 held-out Spearman   (literature CNN reference 0.93)
  B. mrl_m1psi   80/20 held-out Spearman   (our N1-psi setting)
  C. transfer: train unmod -> eval m1psi on common held-out UTRs
  D. label correlation unmod vs m1psi (does psi change UTR ranking?)
  E. designed_library as independent test (train full unmod -> eval)

Features: k-mer counts k=1..4 (340) + GC + length.
Output: data/t3_utr/utr_v0_report.tsv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "t3_utr")
TAB = os.path.join(D, "utr_training_table.tsv")
DES = os.path.join(D, "designed_test.tsv")
OUT = os.path.join(D, "utr_v0_report.tsv")


def featurize(seqs):
    cv = CountVectorizer(analyzer="char", ngram_range=(1, 4),
                         lowercase=False)
    X = cv.fit_transform(seqs).toarray().astype(np.float32)
    X = X / 50.0  # normalize by length (all 50nt here)
    return X


def gbm():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.08,
                                         min_samples_leaf=50,
                                         l2_regularization=1.0,
                                         random_state=13)


def cv_spearman(X, y, seed=13):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed)
    m = gbm().fit(Xtr, ytr)
    return spearmanr(m.predict(Xte), yte).correlation


def main():
    tab = pd.read_csv(TAB, sep="\t")
    print("[utr-v0] %d UTRs" % len(tab))
    X = featurize(tab["utr"].tolist())
    print("[utr-v0] features: %s" % (X.shape,), flush=True)
    rows = []

    # A/B: per-condition held-out
    for cond in ["mrl_unmod", "mrl_m1pseudo"]:
        m = tab[cond].notna().to_numpy()
        rho = cv_spearman(X[m], tab.loc[m, cond].to_numpy())
        rows.append({"exp": "heldout_80/20", "cond": cond, "n": int(m.sum()),
                     "spearman": round(rho, 4)})
        print("[utr-v0] %-14s held-out rho=%.4f (n=%d)" % (
            cond, rho, m.sum()), flush=True)

    # C+D: transfer + label correlation on common UTRs
    both = tab.dropna(subset=["mrl_unmod", "mrl_m1pseudo"])
    common = both.index.to_numpy()
    Xb = X[[tab.index.get_loc(i) for i in common]]
    yu = both["mrl_unmod"].to_numpy()
    yp = both["mrl_m1pseudo"].to_numpy()
    rho_lab = spearmanr(yu, yp).correlation
    rows.append({"exp": "label_correlation", "cond": "unmod~m1psi",
                 "n": len(both), "spearman": round(rho_lab, 4)})
    print("[utr-v0] label corr(unmod, m1psi) = %.4f (n=%d)" % (
        rho_lab, len(both)), flush=True)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yu, test_size=0.2,
                                          random_state=13)
    m = gbm().fit(Xtr, ytr)
    # map test indices back to positions in `both`
    te_pos = np.arange(len(Xb))[-len(Xte):] if len(Xte) == len(Xb) else None
    # simpler: recompute via boolean masks
    idx_all = np.arange(len(Xb))
    te_mask = np.zeros(len(Xb), bool)
    # replicate the split: train_test_split shuffles; redo with indices
    tr_i, te_i = train_test_split(idx_all, test_size=0.2, random_state=13)
    m = gbm().fit(Xb[tr_i], yu[tr_i])
    pred = m.predict(Xb[te_i])
    rho_tr_unmod = spearmanr(pred, yu[te_i]).correlation
    rho_tr_m1psi = spearmanr(pred, yp[te_i]).correlation
    rows.append({"exp": "transfer_unmod->m1psi", "cond": "eval on m1psi",
                 "n": len(te_i), "spearman": round(rho_tr_m1psi, 4)})
    print("[utr-v0] transfer: held-out unmod rho=%.4f | same UTRs m1psi "
          "rho=%.4f" % (rho_tr_unmod, rho_tr_m1psi), flush=True)

    # E: designed library independent test
    des = pd.read_csv(DES, sep="\t")
    mu = tab.dropna(subset=["mrl_unmod"])
    Xd = featurize(list(mu["utr"]) + list(des["utr"]))
    m_full = gbm().fit(Xd[:len(mu)], mu["mrl_unmod"].to_numpy())
    rho_des = spearmanr(m_full.predict(Xd[len(mu):]),
                        des["rl"].to_numpy()).correlation
    rows.append({"exp": "designed_library", "cond": "unmod->designed",
                 "n": len(des), "spearman": round(rho_des, 4)})
    print("[utr-v0] designed library (independent) rho=%.4f (n=%d)" % (
        rho_des, len(des)), flush=True)

    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print("[utr-v0] -> %s" % OUT)


if __name__ == "__main__":
    main()
