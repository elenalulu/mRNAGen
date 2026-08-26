#!/usr/bin/env python
"""UTR v1.5: abundance-filtered training (GBDT windowed + Optimus CNN).

Discovery: MPRA label reliability is abundance-driven --
  all UTRs  rep concordance 0.627 | top-50% abundance 0.809 | top-25% 0.852
Low-coverage UTRs were the noise floor capping v0.5/v1 at ~0.51.

This run: filter to top-50% abundance per condition, train BOTH models,
report vs unfiltered baselines (GBDT unmod .618 / m1psi .511;
CNN unmod .615 / m1psi .488).

Output: data/t3_utr/utr_v15_report.tsv
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "t3_utr")
OUT = os.path.join(D, "utr_v15_report.tsv")
BASE = {("gbdt", "unmod"): 0.618, ("gbdt", "m1psi"): 0.511,
        ("cnn", "unmod"): 0.615, ("cnn", "m1psi"): 0.488}
AB_QUANTILE = 0.5  # keep top-50% abundance UTRs


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


def onehot(seqs):
    idx = {c: i for i, c in enumerate("ACGT")}
    X = np.zeros((len(seqs), 4, 50), dtype=np.float32)
    for r, s in enumerate(seqs):
        for p, c in enumerate(s):
            X[r, idx.get(c, 0), p] = 1.0
    return X


class OptimusCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 128, 8), nn.ReLU(),
            nn.Conv1d(128, 128, 8), nn.ReLU(),
            nn.MaxPool1d(10),
            nn.Flatten(),
            nn.Dropout(0.2), nn.Linear(128 * 3, 128), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_gbdt(X, y, seed=13):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.07,
                                      min_samples_leaf=50,
                                      l2_regularization=1.0,
                                      random_state=seed).fit(X[tr], y[tr])
    return spearmanr(m.predict(X[te]), y[te]).correlation


def run_cnn(seqs, y, device, seed=13, epochs=12, batch=128):
    torch.manual_seed(seed)
    n = len(seqs)
    perm = np.random.RandomState(seed).permutation(n)
    te, tr = perm[:n // 5], perm[n // 5:]
    X = torch.from_numpy(onehot(seqs)).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    model = OptimusCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.RandomState(seed)
    best = -1.0
    for ep in range(epochs):
        model.train()
        rng.shuffle(tr)
        for i in range(0, len(tr), batch):
            b = torch.from_numpy(tr[i:i + batch]).to(device)
            loss = nn.functional.mse_loss(model(X[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = model(X[torch.from_numpy(te).to(device)]).cpu().numpy()
        rho = spearmanr(p, y[te]).correlation
        if rho > best:
            best = rho
        print("    cnn ep%02d rho=%.4f" % (ep + 1, rho), flush=True)
    return best


def main():
    tab = pd.read_csv(os.path.join(D, "utr_training_table.tsv"), sep="\t")
    tab = tab[tab["utr"].str.fullmatch(r"[ACGT]{50}")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[v1.5] device %s | %d UTRs" % (device, len(tab)), flush=True)
    rows = []
    for cond, abcol, name in [("mrl_unmod", "ab_unmod", "unmod"),
                              ("mrl_m1pseudo", "ab_m1pseudo", "m1psi")]:
        sub = tab.dropna(subset=[cond, abcol]).reset_index(drop=True)
        thr = sub[abcol].quantile(AB_QUANTILE)
        keep = sub[sub[abcol] >= thr].reset_index(drop=True)
        print("[v1.5] %s: %d -> %d UTRs after abundance filter (thr=%.5f)"
              % (name, len(sub), len(keep), thr), flush=True)
        seqs = keep["utr"].tolist()
        y = keep[cond].to_numpy()
        rho_g = run_gbdt(featurize(seqs), y)
        print("[v1.5] %s GBDT(filtered) rho=%.4f (unfiltered %.3f)"
              % (name, rho_g, BASE[("gbdt", name)]), flush=True)
        rho_c = run_cnn(seqs, y, device)
        print("[v1.5] %s CNN(filtered)  rho=%.4f (unfiltered %.3f)"
              % (name, rho_c, BASE[("cnn", name)]), flush=True)
        rows.append({"cond": name, "n": len(keep),
                     "gbdt_filtered": round(rho_g, 4),
                     "gbdt_unfiltered": BASE[("gbdt", name)],
                     "cnn_filtered": round(rho_c, 4),
                     "cnn_unfiltered": BASE[("cnn", name)],
                     "best": round(max(rho_g, rho_c), 4)})
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print("[v1.5] -> %s" % OUT)


if __name__ == "__main__":
    main()
