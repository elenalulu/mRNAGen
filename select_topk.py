#!/usr/bin/env python
"""Top-K diversity selection over candidate pools (LD / GEMORNA / refined).

Score-then-diversify: every candidate is scored by the current oracle
composite (score_batch -- swappable; rerun this script after oracle upgrades),
then per protein we run greedy max-min selection in standardized feature
space:

    pick = argmax  alpha * z(score) + (1 - alpha) * min_l2(z_feat, selected)

Rationale: pure top-K-by-score collapses to near-identical sequences (mode
collapse); the diversity term guarantees the delivered set covers different
regions of the design space (CAI/GC/structure/manufacturing trade-offs),
which is what a wet-lab cherry-pick actually wants.

Diversity space (9 dims): cai, gc_global, enc, upa_odds,
selfcomp_max_near, open_start45, mfe_per_nt, restriction_site_count,
hp_max (longest homopolymer run).

Output: data/t5/topk_selection.tsv + per-protein summary.
"""
import csv
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
sys.path.insert(0, HERE)

from ld_parse import parse_lineardesign_output
from gemorna_parse import parse_gemorna_output
from refine_t5 import (FEATS, feature_vec, load_env, n_hp4, n_hp5,
                       n_sites, score_batch)

DATA = os.path.join(HERE, "data")
T5 = os.path.join(DATA, "t5")
LD_DIR = os.path.join(HERE, os.pardir, "github", "LinearDesign-main",
                      "ld_outputs")
LD_CONCAT = os.path.join(HERE, os.pardir, "github", "LinearDesign-main",
                         "lineardesign_output.txt")
GEMORNA_OUT = os.path.join(HERE, os.pardir, "github", "GEMORNA",
                           "gemorna_output.txt")
ALL_FEATS = os.path.join(DATA, "all_candidates_features.tsv")
LD_BASE = os.path.join(HERE, "feature_pipeline", "data",
                       "ld_baseline_features.tsv")

LEN2NAME = {64: "concatemer_64aa", 193: "epo", 239: "egfp", 354: "otc",
            429: "gla_fabry", 452: "pah_pku", 461: "factor_ix", 550: "fluc",
            565: "flu_ha_pr8", 574: "rsv_f", 609: "albumin",
            952: "gaa_pompe", 1273: "cov2_spike", 1368: "spy_cas9",
            2351: "factor_viii"}

DIV_FEATS = ["cai", "gc_global", "enc", "upa_odds", "selfcomp_max_near",
             "open_start45", "mfe_per_nt", "restriction_site_count",
             "hp_max"]

K = 5           # deliverable count per protein
ALPHA = 0.5     # score vs diversity blend


def strip_stop(seq):
    s = seq.upper().replace("U", "T")
    if len(s) % 3 == 0 and s[-3:] in ("TAA", "TAG", "TGA"):
        return s[:-3]
    return s


def upa_odds(seq):
    n = len(seq)
    t, a = seq.count("T"), seq.count("A")
    if t == 0 or a == 0:
        return 0.0
    return (seq.count("TA") * n) / (t * a)


def hp_max_run(seq):
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best


def load_feature_frame():
    """LD + GEMORNA features from featurize outputs; concatemer from T1."""
    df = pd.read_csv(ALL_FEATS, sep="\t")
    base = pd.read_csv(LD_BASE, sep="\t")
    base["generator"] = "lineardesign"
    keep = list(df.columns)
    base = base[[c for c in keep if c in base.columns]].reindex(
        columns=keep)
    df = pd.concat([df, base], ignore_index=True)
    df["hp_max"] = df[["hp_max_A", "hp_max_C", "hp_max_G", "hp_max_U"]].max(
        axis=1)
    # unified protein name
    def pname(row):
        if row["generator"] == "gemorna":
            return LEN2NAME.get(int(str(row["protein"]).replace("len", "")),
                                str(row["protein"]))
        return row["protein"]
    df["protein"] = df.apply(pname, axis=1)
    return df


def collect_candidates(table, codon_coef, models):
    """Return list of dicts: protein/source/seq_id/seq + features+score."""
    fdf = load_feature_frame()
    # index features: LD by (protein, lam), GEMORNA by (protein, rep)
    fidx = {}
    for _, r in fdf.iterrows():
        if r["generator"] == "gemorna":
            fidx[("gemorna", r["protein"], int(r["rep"]))] = r
        else:
            fidx[("lineardesign", r["protein"], float(r["lam"]))] = r

    cands = []

    def add(protein, source, seq_id, seq, feat_row=None, extra=None):
        seq = strip_stop(seq)
        if len(seq) < 30 or len(seq) % 3 != 0:
            return
        sc, _, _ = score_batch([seq], table, codon_coef, models)
        row = {"protein": protein, "source": source, "seq_id": seq_id,
               "seq": seq, "score": float(sc[0])}
        if feat_row is not None:
            for f in DIV_FEATS:
                row[f] = float(feat_row[f])
        if extra:
            row.update(extra)
        cands.append(row)

    # --- LinearDesign (benchmark + concatemer) ---
    for path in sorted(glob.glob(os.path.join(LD_DIR, "*.txt"))):
        protein = os.path.splitext(os.path.basename(path))[0]
        for rec in parse_lineardesign_output(path):
            fr = fidx.get(("lineardesign", protein, float(rec["lam"])))
            if fr is None:
                continue
            add(protein, "ld", "lam%s" % rec["lam"], rec["seq"], fr)
    for rec in parse_lineardesign_output(LD_CONCAT):
        fr = fidx.get(("lineardesign", "concatemer_64aa",
                       float(rec["lam"])))
        if fr is not None:
            add("concatemer_64aa", "ld", "lam%s" % rec["lam"], rec["seq"],
                fr)

    # --- GEMORNA ---
    for rec in parse_gemorna_output(GEMORNA_OUT):
        protein = LEN2NAME.get(len(rec["protein"]))
        if protein is None:
            continue
        fr = fidx.get(("gemorna", protein, int(rec["rep"])))
        if fr is None:
            continue
        add(protein, "gemorna", "rep%s" % rec["rep"], rec["seq"], fr,
            extra={"naturalness": rec.get("naturalness")})

    # --- Refined (T5) + structure post-hoc ---
    # prefer _sc (selfcomp-repaired, 2026-08-25) then r25 (deeper) pools
    for cand in ("refined_candidates_v2_sc.tsv", "refined_candidates_v2.tsv",
                 "refined_candidates_r25.tsv", "refined_candidates.tsv"):
        ref_p = os.path.join(T5, cand)
        if os.path.exists(ref_p):
            break
    for cand in ("structure_posthoc_sc.tsv", "structure_posthoc.tsv"):
        ld_post_p = os.path.join(T5, cand)
        if os.path.exists(ld_post_p):
            break
    ref = pd.read_csv(ref_p, sep="\t")
    post = pd.read_csv(ld_post_p, sep="\t")
    # GEMORNA-start refined (long proteins) if present
    for cand in ("refined_gemorna_v2_sc.tsv", "refined_gemorna_v2.tsv",
                 "refined_gemorna_r25.tsv", "refined_gemorna.tsv"):
        ge_ref_p = os.path.join(T5, cand)
        if os.path.exists(ge_ref_p):
            break
    for cand in ("structure_posthoc_gemorna_sc.tsv",
                 "structure_posthoc_gemorna_v2.tsv",
                 "structure_posthoc_gemorna_r25.tsv",
                 "structure_posthoc_gemorna.tsv"):
        ge_post_p = os.path.join(T5, cand)
        if os.path.exists(ge_post_p):
            break
    if os.path.exists(ge_ref_p):
        ge_ref = pd.read_csv(ge_ref_p, sep="\t")
        ref = pd.concat([ref, ge_ref], ignore_index=True)
        if os.path.exists(ge_post_p):
            post = pd.concat(
                [post, pd.read_csv(ge_post_p, sep="\t")], ignore_index=True)
    post = post.drop_duplicates(subset=["protein", "lam"])
    ref = ref.merge(post, left_on=["protein", "lam_start"],
                    right_on=["protein", "lam"], how="left",
                    suffixes=("", "_ph"))
    for _, r in ref.iterrows():
        seq = r["cds_refined"].upper()
        fv = feature_vec(seq, table, codon_coef)
        cai = fv[FEATS.index("cai")]
        gc = fv[FEATS.index("gc_global")]
        enc = fv[FEATS.index("enc")]
        mfe_nt = (r.get("mfe_ref") or 0.0) / float(len(seq))
        add(r["protein"], "refined", "lam%s" % r["lam_start"], seq, None,
            extra={"cai": cai, "gc_global": gc, "enc": enc,
                   "upa_odds": upa_odds(seq),
                   "selfcomp_max_near": r.get("selfcomp_near_ref"),
                   "open_start45": r.get("open45_ref"),
                   "mfe_per_nt": mfe_nt,
                   "restriction_site_count": r.get("sites_end"),
                   "hp_max": float(hp_max_run(seq))})
    return cands


def select_topk(pool, k=K, alpha=ALPHA):
    """Greedy max-min selection over z-scored features/score."""
    df = pool if isinstance(pool, pd.DataFrame) else pd.DataFrame(pool)
    feats = df[DIV_FEATS].astype(float)
    z = (feats - feats.mean()) / (feats.std(ddof=0) + 1e-9)
    zs = (df["score"] - df["score"].mean()) / (df["score"].std(ddof=0) +
                                               1e-9)
    z = z.to_numpy()
    k = min(k, len(df))
    # first pick: highest score, with a selfcomp tiebreak -- among
    # candidates within 0.5 raw score points of the max, prefer the
    # lowest selfcomp_max_near (dsRNA-risk aware, 2026-08-25)
    smax = float(df["score"].max())
    elig = [i for i in range(len(df))
            if float(df["score"].iloc[i]) >= smax - 0.5]
    selected = [int(min(elig,
                        key=lambda i: float(df["selfcomp_max_near"]
                                            .iloc[i])))]
    while len(selected) < k:
        best, best_v = None, -np.inf
        for i in range(len(df)):
            if i in selected:
                continue
            dmin = min(np.linalg.norm(z[i] - z[j]) for j in selected)
            v = alpha * zs.iloc[i] + (1 - alpha) * dmin
            if v > best_v:
                best, best_v = i, v
        selected.append(best)
    out = df.iloc[selected].copy()
    out["rank"] = range(1, len(selected) + 1)
    # min pairwise feature distance among the selected set
    zz = z[selected]
    n = len(selected)
    mind = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            mind = min(mind, np.linalg.norm(zz[i] - zz[j]))
    out["pool_size"] = len(df)
    out["sel_min_pair_dist"] = round(float(mind), 3)
    return out


def main():
    table, codon_coef, syn, models = load_env()
    print("[topk] loading candidate pools ...", flush=True)
    cands = collect_candidates(table, codon_coef, models)
    df = pd.DataFrame(cands)
    print("[topk] %d candidates over %d proteins" % (
        len(df), df.protein.nunique()), flush=True)

    sel_all, summary = [], []
    for protein, pool in df.groupby("protein"):
        pool = pool.dropna(
            subset=[f for f in DIV_FEATS if f != "naturalness"])
        if len(pool) < 2:
            continue
        sel = select_topk(pool)
        sel["protein"] = protein
        sel_all.append(sel)
        # compare: pure top-K by score vs diverse top-K
        pure = pool.sort_values("score", ascending=False).head(K)
        spread = lambda d, f: float(d[f].astype(float).std())
        summary.append({
            "protein": protein, "pool": len(pool),
            "pure_mean_score": round(pure.score.mean(), 2),
            "div_mean_score": round(sel.score.mean(), 2),
            "score_cost": round(pure.score.mean() - sel.score.mean(), 2),
            "pure_cai_std": round(spread(pure, "cai"), 3),
            "div_cai_std": round(spread(sel, "cai"), 3),
            "pure_gc_std": round(spread(pure, "gc_global"), 3),
            "div_gc_std": round(spread(sel, "gc_global"), 3),
            "pure_sites_sum": int(pure.restriction_site_count.sum()),
            "div_sites_sum": int(sel.restriction_site_count.sum()),
            "n_sources": sel.source.nunique(),
            "sel_min_pair_dist": sel.sel_min_pair_dist.iloc[0],
        })
    sel_df = pd.concat(sel_all, ignore_index=True)
    sel_df.to_csv(os.path.join(T5, "topk_selection.tsv"), sep="\t",
                  index=False)
    smry = pd.DataFrame(summary)
    smry.to_csv(os.path.join(T5, "topk_summary.tsv"), sep="\t", index=False)
    print()
    print(smry.to_string(index=False))
    print()
    print("[topk] -> %s (%d rows)" % (os.path.join(T5, "topk_selection.tsv"),
                                      len(sel_df)))


if __name__ == "__main__":
    main()
