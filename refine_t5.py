#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T5: oracle-guided greedy refinement over synonymous codons.

Loop closure (plan sec T5): LinearDesign lambda-scan candidates as starting
points -> greedy synonymous-codon moves scored by oracle v0 + manufacturing
constraints.

Score = 0.6 * expr_pct + 0.4 * mean(stability_pct)   [oracle heads, pct-rank
outputs, higher = better]
        - 0.5 * restriction_site_count                  [soft]
        - 0.1 * n_homopolymer_runs_ge4                  [soft]
Hard masks (move rejected): GC global outside [0.30, 0.70]; 60nt sliding GC
outside [0.20, 0.80]; any homopolymer run >= 5.

Inputs:
  github/LinearDesign-main/ld_outputs/*.txt  (start candidates, per lambda)
  models/oracle_v0_*.joblib          (oracle v0 heads)
  wu2019 per-codon coefficients              (opt features)
Run: python refine_t5.py [--demo]
Output: data/t5/refined_candidates.tsv + m2_summary.tsv
"""
import csv
import glob
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
ROOT = os.path.dirname(HERE)
for p in (FP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from ld_parse import parse_lineardesign_output  # noqa: E402
from features.codon_metrics import get_default_table  # noqa: E402
from features.rules import RESTRICTION_SITES, CRYPTIC_DONOR_MOTIFS  # noqa: E402

LD_DIR = os.path.join(ROOT, "github", "LinearDesign-main", "ld_outputs")
MODEL_DIR = os.path.join(HERE, "models")
COEF_CSV = os.path.join(HERE, "data", "t2", "wu2019_elifescience",
                        "elife-45396-fig1-data2.csv")
OUT_DIR = os.path.join(HERE, "data", "t5")

FEATS = ["opt_293T", "opt_hela", "opt_rpe", "opt_293T_orfome",
         "opt_k562_orfome", "opt_k562_slam", "cai", "gc_global", "gc3",
         "enc", "cds_len"]
DECAY_HEADS = ["decay_293T", "decay_hela", "decay_rpe", "decay_k562_slam"]
OPT_SRC = {"opt_293T": "293T_endo", "opt_hela": "HeLa_endo",
           "opt_rpe": "RPE_endo", "opt_293T_orfome": "293T_ORFome",
           "opt_k562_orfome": "K562_ORFome", "opt_k562_slam": "K562_SLAM"}
STOPS = {"TAA", "TAG", "TGA"}
W = {"expr": 0.6, "stab": 0.4, "site": 0.5, "hp4": 0.1, "hp5": 0.8}
HP_HARD = 5  # runs >= 5 rejected outright
GC_LO, GC_HI = 0.30, 0.70
GCS_LO, GCS_HI = 0.20, 0.80


def load_env():
    table = get_default_table()
    coef = pd.read_csv(COEF_CSV)
    codon_coef = {}
    for out_col, src in OPT_SRC.items():
        codon_coef[out_col] = dict(zip(coef["codon"].str.upper(),
                                       coef[src].astype(float)))
    # synonymous map (DNA)
    syn = {}
    for rna_codon, aa in table.codon_to_aa.items():
        if aa in ("*", "STOP"):
            continue
        dna = rna_codon.replace("U", "T")
        syn.setdefault(aa, []).append(dna)
    models = {"expr": joblib.load(
        os.path.join(MODEL_DIR, "oracle_v0_expr.joblib"))}
    for h in DECAY_HEADS:
        models[h] = joblib.load(
            os.path.join(MODEL_DIR, "oracle_v0_%s.joblib" % h))
    return table, codon_coef, syn, models


def max_run(seq, base):
    best = cur = 0
    for ch in seq:
        if ch == base:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def hard_ok(cds):
    n = len(cds)
    g = (cds.count("G") + cds.count("C")) / n
    if not (GC_LO <= g <= GC_HI):
        return False
    w = 60
    if n > w:
        for s in range(0, n - w + 1, 10):
            wg = (cds[s:s + w].count("G") + cds[s:s + w].count("C")) / w
            if not (GCS_LO <= wg <= GCS_HI):
                return False
    for b in "ACGT":
        if max_run(cds, b) >= HP_HARD:
            return False
    return True


def n_sites(cds):
    return sum(cds.count(m) for ms in RESTRICTION_SITES.values()
               for m in ms)


def n_hp4(cds):
    total = 0
    for b in "ACGT":
        # count non-overlapping runs >= 4
        run = 0
        for ch in cds:
            if ch == b:
                run += 1
            else:
                if run >= 4:
                    total += 1
                run = 0
        if run >= 4:
            total += 1
    return total


def n_hp5(cds):
    total = 0
    for b in "ACGT":
        run = 0
        for ch in cds:
            if ch == b:
                run += 1
            else:
                if run >= 5:
                    total += 1
                run = 0
        if run >= 5:
            total += 1
    return total


def feature_vec(cds, table, codon_coef):
    codons = [cds[i:i + 3] for i in range(0, len(cds), 3)]
    row = {}
    for out_col, cmap in codon_coef.items():
        vals = [cmap.get(c) for c in codons if c in cmap]
        row[out_col] = sum(vals) / len(vals) if vals else 0.0
    rna = cds.replace("T", "U")
    row["cai"] = table.cai(rna)
    row["gc_global"] = (cds.count("G") + cds.count("C")) / len(cds)
    row["gc3"] = table.gc3(rna)
    row["enc"] = table.enc(rna)
    row["cds_len"] = len(cds)
    return [row[f] for f in FEATS]


def score_batch(seqs, table, codon_coef, models):
    """Batch scoring. Design score (default, externally validated on
    GEMORNA Science S3, 2026-08-25):
        z_global(naturalness) + z_global(CAI) - manufacturing penalties
    The oracle expr/decay heads proved non-transferable within-protein
    (negative transfer on S3) and are returned for diagnostics only.
    Set COMPOSITE=v1 to restore the original oracle composite."""
    X = np.array([feature_vec(s, table, codon_coef) for s in seqs])
    expr = models["expr"].predict(X)
    stab = np.mean([models[h].predict(X) for h in DECAY_HEADS], axis=0)
    pens = np.array([
        W["site"] * n_sites(s) + W["hp4"] * n_hp4(s) + W["hp5"] * n_hp5(s)
        + 5.0 * (max(0.0, (s.count("G") + s.count("C")) / len(s) - GC_HI)
                 + max(0.0, GC_LO - (s.count("G") + s.count("C")) / len(s)))
        for s in seqs])
    if os.environ.get("COMPOSITE", "v2") == "v2":
        from naturalness import Scorer
        if not hasattr(score_batch, "_scorer"):
            score_batch._scorer = Scorer()
        sc = score_batch._scorer
        z_nat = sc.z_nat(seqs)
        z_cai = sc.z_cai([table.cai(s.replace("T", "U")) for s in seqs])
        return z_nat + z_cai - pens, expr, stab
    return W["expr"] * expr + W["stab"] * stab - pens, expr, stab


def composite(cds, table, codon_coef, models):
    sc, e, st = score_batch([cds], table, codon_coef, models)
    return float(sc[0]), float(e[0]), float(st[0])


def refine(cds, table, codon_coef, models, max_rounds=8):
    codons = [cds[i:i + 3] for i in range(0, len(cds), 3)]
    aa_seq = []
    for c in codons:
        a = table.codon_to_aa[c.replace("T", "U")]
        if a in ("*", "STOP"):
            break
        aa_seq.append(a)
    cur = list(codons)
    cur_score, _, _ = composite("".join(cur), table, codon_coef, models)
    start_score = cur_score
    n_moves = 0
    for _ in range(max_rounds):
        # build ALL single-codon variants of the current sequence
        cands = []
        moves = []
        for i, aa in enumerate(aa_seq):
            alts = [a for a in table_aa_codons(aa) if a != cur[i]]
            if not alts:
                continue
            prefix = "".join(cur[:i])
            suffix = "".join(cur[i + 1:])
            for alt in alts:
                cands.append(prefix + alt + suffix)
                moves.append((i, alt))
        if not cands:
            break
        scores, _, _ = score_batch(cands, table, codon_coef, models)
        j = int(np.argmax(scores))
        if scores[j] <= cur_score + 1e-6:
            break
        i, alt = moves[j]
        cur[i] = alt
        cur_score = float(scores[j])
        n_moves += 1
    return "".join(cur), start_score, cur_score, n_moves


_SYN_CACHE = {}


def table_aa_codons(aa):
    if aa not in _SYN_CACHE:
        _SYN_CACHE[aa] = sorted(
            c.replace("U", "T")
            for c, a in get_default_table().codon_to_aa.items()
            if a == aa and a not in ("*", "STOP"))
    return _SYN_CACHE[aa]


def main():
    demo = "--demo" in sys.argv
    table, codon_coef, syn, models = load_env()
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(LD_DIR, "*.txt")))
    if demo:
        files = files[:1]
    rows = []
    t0 = time.time()
    for fp in files:
        pname = os.path.splitext(os.path.basename(fp))[0]
        recs = parse_lineardesign_output(fp)
        seen = set()
        for rec in recs:
            cds = rec["seq"].replace("U", "T").upper()
            if cds[-3:] in STOPS:
                cds = cds[:-3]
            if cds in seen:
                continue
            seen.add(cds)
            s0, _, _ = composite(cds, table, codon_coef, models)
            t1 = time.time()
            out, s_start, s_end, n_moves = refine(
                cds, table, codon_coef, models,
                max_rounds=int(os.environ.get("MAX_ROUNDS", 8)))
            dt = time.time() - t1
            rows.append({
                "protein": pname,
                "lam_start": rec["lam"],
                "cds_len": len(cds),
                "n_moves": n_moves,
                "score_start": round(s_start, 4),
                "score_end": round(s_end, 4),
                "delta": round(s_end - s_start, 4),
                "sites_start": n_sites(cds),
                "sites_end": n_sites(out),
                "hp4_start": n_hp4(cds),
                "hp4_end": n_hp4(out),
                "hp5_start": n_hp5(cds),
                "hp5_end": n_hp5(out),
                "hard_pass_end": int(hard_ok(out)),
                "cds_refined": out,
            })
            print("[t5] %-14s lam=%-4s moves=%-4d score %.3f->%.3f "
                  "(+%.3f) sites %d->%d hp4 %d->%d [%.1fs]"
                  % (pname, rec["lam"], n_moves, s_start, s_end,
                     s_end - s_start, rows[-1]["sites_start"],
                     rows[-1]["sites_end"], rows[-1]["hp4_start"],
                     rows[-1]["hp4_end"], dt), flush=True)
            # incremental save every 5 rows (lost-run lesson 2026-08-24)
            if len(rows) % 5 == 0:
                pd.DataFrame(rows).to_csv(
                    os.environ.get(
                        "T5_OUT",
                        os.path.join(OUT_DIR, "refined_candidates.tsv")),
                    sep="\t", index=False)
    out_tsv = os.environ.get(
        "T5_OUT", os.path.join(OUT_DIR, "refined_candidates.tsv"))
    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print()
    print("[t5] %d refined candidates -> %s (%.1fs total)"
          % (len(rows), out_tsv, time.time() - t0))

    # ---------------- M2 summary ----------------
    if not demo:
        df = pd.DataFrame(rows)
        summ = df.groupby("protein").agg(
            n_starts=("lam_start", "count"),
            delta_mean=("delta", "mean"),
            delta_max=("delta", "max"),
            sites_start_sum=("sites_start", "sum"),
            sites_end_sum=("sites_end", "sum"),
            hp4_start_sum=("hp4_start", "sum"),
            hp4_end_sum=("hp4_end", "sum"),
            hard_pass=("hard_pass_end", "sum")).round(3)
        print("[t5] M2 summary (per protein, over its lambda starts):")
        print(summ.to_string())
        try:
            summ.to_csv(os.path.join(OUT_DIR, "m2_summary.tsv"), sep="\t")
        except PermissionError:
            print("[t5] WARN: m2_summary.tsv locked, skipped", flush=True)


if __name__ == "__main__":
    main()
