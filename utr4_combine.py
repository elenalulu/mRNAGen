#!/usr/bin/env python
"""UTR-4: full-mRNA combination (UTR head x CDS Top-K v2 deliverable).

Inputs:
  - utr_training_table.tsv (m1psi labels + abundance)
  - data/t5/topk_selection.tsv (CDS Top-K v2 with oracle composite scores)

Steps:
  1. Train the final m1psi GBDT (windowed k-mers, abundance top-50%,
     held-out rho 0.706) -- saved as models/utr_m1psi_gbdt.joblib
  2. UTR shortlist: top-500 by MEASURED m1psi MRL (abundance-filtered),
     greedy max-min diversity -> 10 UTRs (sequences + labels, deliverable)
  3. Combined ranking: for each protein's Top-5 CDS candidates x 10 UTRs,
     combined = z(UTR m1psi MRL) + z(CDS oracle score)  (additive,
     no UTRxCDS interaction modeled -- honest caveat v1)

Outputs: data/t3_utr/utr_shortlist.tsv, utr4_full_mrna_ranking.tsv
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "t3_utr")
MODEL_DIR = os.path.join(HERE, "models")
TOPK = os.path.join(HERE, "data", "t5", "topk_selection.tsv")


def featurize(seqs):
    cv4 = CountVectorizer(analyzer="char", ngram_range=(1, 4),
                          lowercase=False)
    Xg = cv4.fit_transform(seqs).toarray().astype(np.float32)
    cv3 = CountVectorizer(analyzer="char", ngram_range=(1, 3),
                          lowercase=False)
    wins = []
    for w in range(5):
        ws = [s[w * 10:(w + 1) * 10] for s in seqs]
        wins.append(cv3.fit_transform(ws).toarray().astype(np.float32) / 10.0)
    return np.hstack([Xg / 50.0] + wins)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    tab = pd.read_csv(os.path.join(D, "utr_training_table.tsv"), sep="\t")
    tab = tab[tab["utr"].str.fullmatch(r"[ACGT]{50}")]
    sub = tab.dropna(subset=["mrl_m1pseudo", "ab_m1pseudo"])
    keep = sub[sub["ab_m1pseudo"] >= sub["ab_m1pseudo"].quantile(0.5)]
    keep = keep.reset_index(drop=True)
    print("[utr4] m1psi pool: %d UTRs (abundance top-50%%)" % len(keep))

    # 1. final model (train on 80%, report held-out, save)
    X = featurize(keep["utr"].tolist())
    y = keep["mrl_m1pseudo"].to_numpy()
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=13)
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07,
                                          min_samples_leaf=50,
                                          l2_regularization=1.0,
                                          random_state=13).fit(X[tr], y[tr])
    rho = spearmanr(model.predict(X[te]), y[te]).correlation
    print("[utr4] final m1psi model held-out rho=%.4f" % rho)
    joblib.dump(model, os.path.join(MODEL_DIR, "utr_m1psi_gbdt.joblib"))

    # 2. UTR shortlist: top-500 by MEASURED mrl, greedy max-min diversity
    top500 = keep.sort_values("mrl_m1pseudo",
                              ascending=False).head(500).reset_index(
                                  drop=True)
    sel = [0]
    while len(sel) < 10:
        best_i, best_d = None, -1
        for i in range(len(top500)):
            if i in sel:
                continue
            dmin = min(hamming(top500.utr[i], top500.utr[j])
                       for j in sel)
            if dmin > best_d:
                best_i, best_d = i, dmin
        sel.append(best_i)
    short = top500.iloc[sel].copy()
    short["rank"] = range(1, 11)
    short[["rank", "utr", "mrl_m1pseudo", "ab_m1pseudo"]].to_csv(
        os.path.join(D, "utr_shortlist.tsv"), sep="\t", index=False)
    print("[utr4] shortlist: 10 UTRs, measured m1psi MRL %.1f-%.1f"
          % (short.mrl_m1pseudo.min(), short.mrl_m1pseudo.max()))
    print("   min pairwise hamming: %d/50" % min(
        hamming(short.utr.iloc[i], short.utr.iloc[j])
        for i in range(10) for j in range(i + 1, 10)))

    # 3. combined ranking
    cds = pd.read_csv(TOPK, sep="\t")
    zu = (short.mrl_m1pseudo - short.mrl_m1pseudo.mean()) / \
        short.mrl_m1pseudo.std()
    zc = (cds.score - cds.score.mean()) / cds.score.std()
    cds = cds.assign(z_cds=zc.values)
    short = short.assign(z_utr=zu.values, utr_id=[
        "utr_top%d" % r for r in short["rank"]])
    rows = []
    for _, c in cds.iterrows():
        for _, u in short.iterrows():
            rows.append({
                "protein": c.protein, "cds_id": "%s_%s" % (c.source,
                                                           c.seq_id),
                "cds_score": round(c.score, 2), "cds_z": round(c.z_cds, 2),
                "utr_id": u.utr_id, "utr_mrl": round(u.mrl_m1pseudo, 2),
                "utr_z": round(u.z_utr, 2),
                "combined": round(c.z_cds + u.z_utr, 3)})
    comb = pd.DataFrame(rows).sort_values(
        ["protein", "combined"], ascending=[True, False])
    comb.to_csv(os.path.join(D, "utr4_full_mrna_ranking.tsv"), sep="\t",
                index=False)
    print("[utr4] -> %d combos (%d proteins x %d CDS x %d UTRs)"
          % (len(comb), comb.protein.nunique(), 5, 10))
    best = comb.groupby("protein").head(1)
    print()
    print("[utr4] best full-mRNA design per protein:")
    print(best[["protein", "cds_id", "utr_id", "combined"]].to_string(
        index=False))


if __name__ == "__main__":
    main()
