#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2-3: assemble the T4 oracle training table.

Inputs (all local, see data/t2/):
  - t2_endogenous_cds_abundance.tsv   (18,963 rows: ENSG + CDS + PaxDb abundance)
  - wu2019_elifescience/elife-45396-fig1-data1.xlsx
      per-GENE decay_rate, 6 sheets (293T/HeLa/RPE endogenous, 293T/K562 ORFome,
      K562 SLAM-seq), Name = ENSG for endogenous entries
  - wu2019_elifescience/elife-45396-fig1-data2.csv
      per-codon stability coefficients, 61 sense codons x 6 cell contexts

Output: data/t2/t2_training_table.tsv
  meta      : gene (ENSG), ensp, gene_symbol, cds_len
  labels    : abundance_ppm, decay_{ctx} x 5 contexts (K562-ORFome dropped:
              ORFome assays are reporter-construct based, keep endogenous +
              SLAM which measure endogenous transcripts; SLAM kept, labelled)
  features  : codon optimality scores opt_{ctx} x 6 (mean Wu coefficient over
              the CDS's sense codons), cai, gc_global, gc3, enc

Run: python build_t2_training_table.py
"""
import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.join(HERE, "data", "t2")
FP = os.path.join(HERE, "feature_pipeline")
if FP not in sys.path:
    sys.path.insert(0, FP)

from features.codon_metrics import get_default_table  # noqa: E402
from features.seq_stats import normalize_rna, gc_content  # noqa: E402

ENDO_TSV = os.path.join(T2, "t2_endogenous_cds_abundance.tsv")
DECAY_XLSX = os.path.join(T2, "wu2019_elifescience",
                          "elife-45396-fig1-data1.xlsx")
COEF_CSV = os.path.join(T2, "wu2019_elifescience",
                        "elife-45396-fig1-data2.csv")
OUT_TSV = os.path.join(T2, "t2_training_table.tsv")

# decay sheets to keep -> output column
DECAY_CTX = {
    "293T-endogenous": "decay_293T",
    "HeLa-endogenous": "decay_hela",
    "RPE-endogenous": "decay_rpe",
    "k562-SLAM-seq": "decay_k562_slam",
}
# codon coefficient columns -> output column
OPT_CTX = {
    "293T_endo": "opt_293T",
    "HeLa_endo": "opt_hela",
    "RPE_endo": "opt_rpe",
    "293T_ORFome": "opt_293T_orfome",
    "K562_ORFome": "opt_k562_orfome",
    "K562_SLAM": "opt_k562_slam",
}
STOPS = {"TAA", "TAG", "TGA"}


def main():
    # ---------- load endogenous table, one CDS per ENSG (longest) ----------
    rows = []
    with open(ENDO_TSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            rows.append(r)
    print("[t2-3] endogenous rows:", len(rows))
    best = {}
    for r in rows:
        g = r["gene"]
        if g not in best or int(r["cds_len"]) > int(best[g]["cds_len"]):
            best[g] = r
    print("[t2-3] genes after longest-CDS dedup:", len(best))

    # ---------- Wu 2019 decay rates (ENSG-keyed) ----------
    xl = pd.ExcelFile(DECAY_XLSX)
    decay = {}
    for sheet, col in DECAY_CTX.items():
        df = xl.parse(sheet)
        sub = df[df["Name"].astype(str).str.startswith("ENSG")]
        decay[col] = dict(zip(sub["Name"].astype(str),
                              sub["decay_rate"].astype(float)))
        print("[t2-3] decay %s: %d ENSG rows" % (col, len(sub)))

    # ---------- Wu 2019 per-codon coefficients ----------
    coef = pd.read_csv(COEF_CSV)
    codon_coef = {}
    for src_col, out_col in OPT_CTX.items():
        codon_coef[out_col] = dict(zip(coef["codon"].astype(str).str.upper(),
                                       coef[src_col].astype(float)))
    print("[t2-3] codon coefficients: %d codons x %d contexts"
          % (len(coef), len(OPT_CTX)))

    # ---------- assemble ----------
    table = get_default_table()
    out_rows = []
    n_stop_stripped = 0
    for g, r in best.items():
        cds = r["cds"].upper()
        if cds[-3:] in STOPS:
            cds = cds[:-3]
            n_stop_stripped += 1
        if len(cds) < 30 or len(cds) % 3 != 0:
            continue
        codons = [cds[i:i + 3] for i in range(0, len(cds), 3)]
        # skip internal stops (shouldn't happen; guard)
        if any(c in STOPS for c in codons):
            continue
        row = {
            "gene": g,
            "ensp": r["ensp"],
            "gene_symbol": r["gene_symbol"],
            "cds_len": len(cds),
            "abundance_ppm": float(r["abundance_ppm"]),
        }
        # decay labels (joined on ENSG)
        for col, d in decay.items():
            row[col] = d.get(g, "")
        # codon optimality scores (mean coefficient over codons)
        for out_col, cmap in codon_coef.items():
            vals = [cmap.get(c) for c in codons if c in cmap]
            row[out_col] = sum(vals) / len(vals) if vals else ""
        # cheap sequence features
        rna = cds.replace("T", "U")
        row["cai"] = table.cai(rna)
        row["gc_global"] = gc_content(rna)
        row["gc3"] = table.gc3(rna)
        row["enc"] = table.enc(rna)
        out_rows.append(row)
    print("[t2-3] assembled rows:", len(out_rows),
          "| trailing stops stripped:", n_stop_stripped)

    cols = list(out_rows[0].keys())
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print("[t2-3] -> %s (%d rows x %d cols)"
          % (OUT_TSV, len(out_rows), len(cols)))

    # ---------- sanity correlations ----------
    df = pd.DataFrame(out_rows)
    df["log_ab"] = df["abundance_ppm"].apply(
        lambda x: __import__("math").log10(x + 1e-6))
    have_decay = df["decay_k562_slam"] != ""
    print()
    print("[sanity] n with decay labels: 293T=%d hela=%d rpe=%d k562slam=%d"
          % ((df.decay_293T != "").sum(), (df.decay_hela != "").sum(),
             (df.decay_rpe != "").sum(), have_decay.sum()))
    for feat in ["cai", "opt_293T", "opt_k562_slam", "gc_global", "gc3"]:
        c = df["log_ab"].corr(df[feat])
        print("[sanity] corr(%s, log10 abundance) = %.3f" % (feat, c))
    d = df[have_decay]
    print("[sanity] corr(opt_k562_slam, decay_k562_slam) = %.3f"
          % d["opt_k562_slam"].corr(d["decay_k562_slam"]))
    print("[sanity] corr(opt_293T, decay_293T) = %.3f"
          % (df[df.decay_293T != ""]["opt_293T"]
             .corr(df[df.decay_293T != ""]["decay_293T"])))


if __name__ == "__main__":
    main()
