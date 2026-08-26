#!/usr/bin/env python
"""UTR v0.5: position-aware features (windowed k-mers) on the corrected table.

v0 (bag-of-kmers, position-blind) got 0.671 unmod / 0.499 m1psi.
Position matters for translation initiation -> 5 x 10nt windows with
k=1..3 counts each + global k=4. Ceiling context:
  unmod rep concordance 0.827 -> mean-of-2 reliability ~0.91
  m1psi rep concordance 0.627 -> ~0.77

Output: data/t3_utr/utr_v05_report.tsv
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
OUT = os.path.join(D, "utr_v05_report.tsv")


def featurize(seqs):
    """Global k=1..4 + 5 windows x k=1..3."""
    cv4 = CountVectorizer(analyzer="char", ngram_range=(1, 4),
                          lowercase=False)
    Xg = cv4.fit_transform(seqs).toarray().astype(np.float32)
    wins = []
    cv3 = CountVectorizer(analyzer="char", ngram_range=(1, 3),
                          lowercase=False)
    for w in range(5):
        ws = [s[w * 10:(w + 1) * 10] for s in seqs]
        Xw = cv3.fit_transform(ws).toarray().astype(np.float32)
        wins.append(Xw / 10.0)
    return np.hstack([Xg / 50.0] + wins)


def gbm():
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07,
                                         min_samples_leaf=50,
                                         l2_regularization=1.0,
                                         random_state=13)


def run(X, y, seed=13):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=seed)
    m = gbm().fit(X[tr], y[tr])
    return spearmanr(m.predict(X[te]), y[te]).correlation, m


def main():
    tab = pd.read_csv(os.path.join(D, "utr_training_table.tsv"), sep="\t")
    print("[utr-v05] %d UTRs" % len(tab), flush=True)
    X = featurize(tab["utr"].tolist())
    print("[utr-v05] features: %s" % (X.shape,), flush=True)
    rows = []
    for cond in ["mrl_unmod", "mrl_m1pseudo"]:
        m_ = tab[cond].notna().to_numpy()
        rho, model = run(X[m_], tab.loc[m_, cond].to_numpy())
        rows.append({"exp": "heldout_80/20", "cond": cond, "n": int(m_.sum()),
                     "spearman": round(rho, 4)})
        print("[utr-v05] %-14s rho=%.4f (n=%d)" % (cond, rho, m_.sum()),
              flush=True)
        if cond == "mrl_unmod":
            unmod_model, unmod_mask = model, m_
    # transfer unmod -> m1psi on common held-out
    both = tab.dropna(subset=["mrl_unmod", "mrl_m1pseudo"])
    pos = tab.index.get_indexer(both.index)
    tr, te = train_test_split(np.arange(len(both)), test_size=0.2,
                              random_state=13)
    yu = both["mrl_unmod"].to_numpy()
    yp = both["mrl_m1pseudo"].to_numpy()
    Xb = X[pos]
    m = gbm().fit(Xb[tr], yu[tr])
    rho_t = spearmanr(m.predict(Xb[te]), yp[te]).correlation
    rows.append({"exp": "transfer_unmod->m1psi", "cond": "m1psi held-out",
                 "n": len(te), "spearman": round(rho_t, 4)})
    print("[utr-v05] transfer -> m1psi rho=%.4f" % rho_t, flush=True)
    # designed library independent test (unmod model)
    des = pd.read_csv(os.path.join(D, "designed_test.tsv"), sep="\t")
    mu = tab[unmod_mask]
    Xd = featurize(list(mu["utr"]) + list(des["utr"]))
    m_full = gbm().fit(Xd[:len(mu)], mu["mrl_unmod"].to_numpy())
    rho_d = spearmanr(m_full.predict(Xd[len(mu):]), des["rl"].to_numpy())
    rows.append({"exp": "designed_library", "cond": "unmod->designed",
                 "n": len(des), "spearman": round(rho_d.correlation, 4)})
    print("[utr-v05] designed library rho=%.4f" % rho_d.correlation,
          flush=True)
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print("[utr-v05] -> %s" % OUT)


if __name__ == "__main__":
    main()
