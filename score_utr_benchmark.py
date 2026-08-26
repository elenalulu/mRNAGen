#!/usr/bin/env python
"""Score famous UTRs (vaccine / natural / GEMORNA / literature) with our
m1psi model + shortlist context.

Library: GEMORNA Science S1 5'UTRs sheet (35 sequences) -- includes
BNT162b2, mRNA-1273, hHBB, CYBA, AG (alpha-globin), Sample'19 best designs,
GEMORNA's own GMR-5U1..12.

Model: m1psi GBDT (abundance top-50%, windowed k-mers) retrained here so
the feature space is consistent (the saved joblib lacks the vectorizer).

Caveat (honest): the model was trained on exactly-50nt random UTRs; here we
score the first 50nt of each UTR (initiation-proximal). Predictions for
non-50nt UTRs are directional, not calibrated.

Output: data/t3_utr/utr_benchmark_scores.tsv
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
SI = os.path.join(HERE, "data", "t3_gemorna_si",
                  "science.adr8470_data-s1.xlsx")
OUT = os.path.join(D, "utr_benchmark_scores.tsv")


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


def main():
    tab = pd.read_csv(os.path.join(D, "utr_training_table.tsv"), sep="\t")
    tab = tab[tab["utr"].str.fullmatch(r"[ACGT]{50}")]
    sub = tab.dropna(subset=["mrl_m1pseudo", "ab_m1pseudo"])
    keep = sub[sub["ab_m1pseudo"] >= sub["ab_m1pseudo"].quantile(
        0.5)].reset_index(drop=True)
    # NOTE: vectorizer must be fitted on keep's corpus first, then extended
    # rows appended so vocabulary/columns stay consistent
    bench = pd.read_excel(SI, sheet_name="5'UTRs")
    bench["seq50"] = (bench["Sequence"].astype(str).str.upper()
                      .str.replace("U", "T", regex=False).str[:50]
                      .str.replace("[^ACGT]", "A", regex=True))
    bench = bench[bench["seq50"].str.len() == 50]

    seqs_all = list(keep["utr"]) + list(bench["seq50"])
    Xall = featurize(seqs_all)
    y = keep["mrl_m1pseudo"].to_numpy()
    Xtr, Xte = Xall[:len(keep)], Xall[len(keep):]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=13)
    model = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07,
                                          min_samples_leaf=50,
                                          l2_regularization=1.0,
                                          random_state=13).fit(Xtr[tr], y[tr])
    rho = spearmanr(model.predict(Xtr[te]), y[te]).correlation
    print("[bench] model rebuilt, held-out rho=%.4f" % rho)

    # library reference distribution (predicted on full kept pool)
    lib_pred = model.predict(Xtr)
    print("[bench] library predicted MRL: median %.2f  top10%% %.2f"
          % (np.median(lib_pred), np.percentile(lib_pred, 90)))

    # shortlist (measured) for calibration context
    short = pd.read_csv(os.path.join(D, "utr_shortlist.tsv"), sep="\t")

    bench["pred_m1psi_mrl"] = model.predict(Xte)
    bench["pct_library"] = [
        round(100.0 * (lib_pred < p).mean(), 1) for p in bench[
            "pred_m1psi_mrl"]]
    out = bench[["Name", "seq50", "pred_m1psi_mrl", "pct_library"]].copy()
    out["orig_len"] = bench["Sequence"].astype(str).str.len().values
    out = out.sort_values("pred_m1psi_mrl", ascending=False)
    out.to_csv(OUT, sep="\t", index=False)
    print()
    print("%-16s %5s %8s %7s" % ("UTR", "len", "predMRL", "pct"))
    for _, r in out.iterrows():
        print("%-16s %5d %8.2f %6.1f%%" % (str(r["Name"])[:16],
                                            r["orig_len"],
                                            r["pred_m1psi_mrl"],
                                            r["pct_library"]))
    print()
    print("shortlist measured m1psi MRL: %.1f-%.1f (reference points)"
          % (short.mrl_m1pseudo.min(), short.mrl_m1pseudo.max()))
    print("-> %s" % OUT)


if __name__ == "__main__":
    main()
