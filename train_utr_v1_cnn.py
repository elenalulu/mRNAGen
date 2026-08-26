#!/usr/bin/env python
"""UTR v1: Optimus-style CNN (Sample'19 architecture) per condition.

v0.5 GBDT (position-windowed k-mers): unmod 0.618 / m1psi 0.511.
Ceilings (mean-of-2 reliability): unmod ~0.91 / m1psi ~0.77.
Target: sequence-order model closes the gap (literature CNN 0.93 unmod).

Architecture (Optimus, Nat Biotech 2019):
  one-hot(4 x 50) -> Conv(128, k=8) ReLU -> Conv(128, k=8) ReLU ->
  MaxPool(10) -> Dropout(0.2) -> Dense(128) ReLU -> Dropout(0.2) -> Dense(1)

Trains separate models for mrl_unmod and mrl_m1psi (our setting).
CPU torch is sufficient (~0.5M params).

Output: data/t3_utr/utr_v1_report.tsv + models utr_v1_<cond>.pt
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data", "t3_utr")
TAB = os.path.join(D, "utr_training_table.tsv")
OUT = os.path.join(D, "utr_v1_report.tsv")
BASE = {"mrl_unmod": 0.618, "mrl_m1pseudo": 0.511}  # v0.5 GBDT baselines
NAME = {"mrl_unmod": "unmod", "mrl_m1pseudo": "m1psi"}


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


def train_cond(tab, cond, device, epochs=12, batch=128, seed=13):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sub = tab.dropna(subset=[cond]).reset_index(drop=True)
    X = torch.from_numpy(onehot(sub["utr"].tolist()))
    y = sub[cond].to_numpy().astype(np.float32)
    n = len(sub)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    te = perm[:n // 5]
    tr = perm[n // 5:]
    model = OptimusCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = X.to(device)
    yt = torch.from_numpy(y).to(device)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        rng.shuffle(tr)
        for i in range(0, len(tr), batch):
            b = torch.from_numpy(tr[i:i + batch]).to(device)
            pred = model(Xt[b])
            loss = nn.functional.mse_loss(pred, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = model(Xt[torch.from_numpy(te).to(device)]).cpu().numpy()
        rho = spearmanr(p, y[te]).correlation
        print("[utr-v1] %s epoch %02d  held-out rho=%.4f  [%.0fs]"
              % (NAME[cond], ep, rho, time.time() - t0), flush=True)
        if rho > best:
            best = rho
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    torch.save(best_state, os.path.join(D, "utr_v1_%s.pt" % NAME[cond]))
    return best, len(sub)


def main():
    tab = pd.read_csv(TAB, sep="\t")
    tab = tab[tab["utr"].str.fullmatch(r"[ACGT]{50}")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[utr-v1] device: %s | %d UTRs" % (device, len(tab)), flush=True)
    rows = []
    for cond in ["mrl_unmod", "mrl_m1pseudo"]:
        best, n = train_cond(tab, cond, device)
        rows.append({"cond": NAME[cond], "n": n,
                     "cnn_rho": round(best, 4),
                     "gbdt_v05": BASE[cond],
                     "delta": round(best - BASE[cond], 4)})
        print("[utr-v1] %s final: CNN %.4f vs GBDT %.3f (%+.4f)"
              % (NAME[cond], best, BASE[cond], best - BASE[cond]),
              flush=True)
    pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
    print("[utr-v1] -> %s" % OUT)


if __name__ == "__main__":
    main()
