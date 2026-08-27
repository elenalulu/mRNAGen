#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-candidate 12-metric scorecard appended to the deliverable Top-K TSV.

Covers the 12 requested metrics; honesty tiers are explicit:

  exact        -- computed, component exists / small extension:
    expr_z         z(naturalness) + z(CAI)   (S3 externally validated signal)
    tai            tAI (dos Reis 2004, GtRNAdb Hsapi38)     [from compare_table]
    cpb            codon-pair bias, log(f_pair / f_i*f_j) over corpus pairs
    gc_w60_max/min sliding-window GC extremes (60 nt window)
    dsrna_bp       longest near-perfect intramolecular duplex (MDA5 proxy)
    dg_start45     ViennaRNA MFE of the first 45 nt (start region)
    acc_aug30      mean unpaired probability over the first 30 nt
                   (partition-function bpp on a local 150 nt fold)
    pairing_prob   fraction of bases paired (windowed bpp average)

  heuristic composite -- assembled from existing components, clearly labelled:
    te_comp        0.6*CAI + 0.4*tAI (translation-efficiency proxy)
    init_comp      start-region openness + start MFE + +4G Kozak bonus
                   (no UTR context in CDS-only design: -3/-1 unavailable)
    halflife_pct   oracle decay heads mean stability percentile
                   (wu2019-trained; NOT transferable across genes -- weak)

  rule flag -- no validated model, needs wet-lab labels:
    immune_flag    0-4 additive flags: long duplex (MDA5), U-rich density
                   (TLR7), CpG odds (TLR9-ish); UNVALIDATED, not a prediction

Usage (python of a venv with numpy/sklearn/joblib/ViennaRNA):
  python scorecard.py                 # updates independent_topk.tsv in place
  python scorecard.py --in X --out Y  # separate output file
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
for p in (FP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from naturalness import Scorer  # noqa: E402
from compare_table import tai as tai_fn, load_trna_counts  # noqa: E402
from refine_t5 import load_env, DECAY_HEADS  # noqa: E402

# ---------------------------------------------------------------- structure
LOCAL_WINDOW = 150      # local fold for AUG accessibility (CDS-only design)
START_DG_LEN = 45       # start-region window for Delta-G
ACC_WINDOW = 30         # accessibility window around/at AUG (first 30 nt)
PAIR_WIN = 200          # sampling window for global pairing stats
PAIR_STRIDE = 500
DUPLEX_MED = 24         # duplex bp thresholds for immune flag (guideline,
DUPLEX_HIGH = 30        # ~30 bp long-dsRNA sensing scale; see docstring)
URICH_PER_KB = 25.0
CPG_ODDS_HI = 1.6


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def window_unpaired(seq):
    """Mean unpaired prob via windowed partition function (long-seq safe)."""
    n = len(seq)
    if n < 5:
        return None
    vals = []
    if n <= LOCAL_WINDOW:
        wins = [seq]
    else:
        starts = list(range(0, max(1, n - PAIR_WIN + 1), PAIR_STRIDE))
        wins = [seq[s:s + PAIR_WIN] for s in starts]
    for w in wins:
        pun = _pf_unpaired(w)
        if pun is None:
            return None
        vals.append(_mean(pun))
    return _mean(vals)


def _pf_unpaired(w):
    import RNA as vrna
    try:
        fc = vrna.fold_compound(w)
        fc.pf()
        bp = fc.bpp()
        m = len(w)
        out = []
        for i in range(1, m + 1):
            s = sum(bp[i][j] for j in range(1, m + 1) if j != i)
            out.append(1.0 - min(s, 1.0))
        return out
    except Exception:
        return None


def start_region_features(rna):
    """dg_start45 and acc_aug30 from a single local fold of the 5' end."""
    import RNA as vrna
    head = rna[:max(LOCAL_WINDOW, START_DG_LEN)]
    fc = vrna.fold_compound(head)
    ss, mfe = fc.mfe()          # dg_start45 uses the 45 nt sub-window MFE
    sub = rna[:START_DG_LEN]
    fc_sub = vrna.fold_compound(sub)
    _, dg45 = fc_sub.mfe()
    pun = _pf_unpaired(head)
    if pun is None:
        acc = 1.0 - sum(1 for c in ss[:ACC_WINDOW] if c != ".") / ACC_WINDOW
    else:
        acc = _mean(pun[:ACC_WINDOW])
    return dg45, acc


def selfcomp_max_near(rna):
    """Longest near-perfect duplex via features.structure.selfcomp_scan."""
    try:
        from features.structure import selfcomp_scan
        _, nr = selfcomp_scan(rna)
        return nr
    except Exception:
        return 0


# ---------------------------------------------------------------- CPB block

def build_cpb_fn(stats_pkl_path):
    """Standard-style CPB: mean cps = log(f_pair / f_ci*f_cj) over corpus.

    Codon monomer frequencies are derived from the corpus pair counts stored
    by naturalness.build_stats(), so no extra data file is needed.
    """
    import pickle
    with open(stats_pkl_path, "rb") as f:
        pc = pickle.load(f)["pair_count"]
    mono = {}
    total_codons = 0
    for pair, cnt in pc.items():
        total_codons += 2 * cnt
        mono[pair[:3]] = mono.get(pair[:3], 0) + cnt
        mono[pair[3:]] = mono.get(pair[3:], 0) + cnt
    f_mono = {c: v / total_codons for c, v in mono.items()}
    total_pairs = sum(pc.values())
    f_pair = {p: v / total_pairs for p, v in pc.items()}

    def cpb(dna_seq):
        dna = dna_seq.upper().replace("U", "T")
        vals = []
        for i in range(0, len(dna) - 6, 3):
            p = dna[i:i + 6]
            fi = f_mono.get(p[:3], 0.0)
            fj = f_mono.get(p[3:], 0.0)
            fp = f_pair.get(p, 0.0)
            if fi > 0 and fj > 0 and fp > 0:
                vals.append(math.log(fp / (fi * fj)))
        return _mean(vals) if vals else 0.0

    return cpb


# ------------------------------------------------------------- tier helpers

def immune_flag(dsrna_bp, upa_odds_val, urich_cnt, seqlen, cpg_odds_val):
    """Additive rule flag 0-4. HEURISTIC/UNVALIDATED, not an activation
    prediction -- real immunogenicity requires wet-lab dose-response data."""
    flag = 0
    if dsrna_bp >= DUPLEX_HIGH:
        flag += 2
    elif dsrna_bp >= DUPLEX_MED:
        flag += 1
    if (urich_cnt * 1000.0 / max(seqlen, 1)) >= URICH_PER_KB:
        flag += 1
    if cpg_odds_val >= CPG_ODDS_HI:
        flag += 1
    return flag


def dinuc_odds(seq, a, b):
    n = len(seq)
    if n < 2:
        return 0.0
    exp = seq.count(a) * seq.count(b) / n
    return (seq.count(a + b) * n / (seq.count(a) * seq.count(b))) if exp else 0.0


def score_row(seq_dna, ctx):
    """Compute the scorecard dict for one candidate (DNA sequence)."""
    seq = seq_dna.upper().replace("U", "T")
    rna = seq.replace("T", "U")
    n = len(seq)

    # --- expression (exact, S3-validated signal family)
    z_nat = float(ctx.scorer.z_nat([seq])[0])
    z_cai = float(ctx.scorer.z_cai([ctx.table.cai(rna)])[0])

    # --- codon-level metrics (exact)
    tv = tai_fn(seq, ctx.trna, ctx.wcache)
    cai = ctx.table.cai(rna)
    cpb_v = ctx.cpb(seq)

    # --- GC window profile (exact)
    gvals = []
    w, step = 60, 10
    if n <= w:
        g = (rna.count("G") + rna.count("C")) / n
        gw_max = gw_min = g
    else:
        for s in range(0, n - w + 1, step):
            gvals.append((rna[s:s + w].count("G") + rna[s:s + w].count("C")) / w)
        gw_max, gw_min = max(gvals), min(gvals)

    # --- structure metrics (exact; ViennaRNA)
    dsrna = selfcomp_max_near(rna)
    dg45, acc30 = start_region_features(rna)
    unpaired = window_unpaired(rna)
    pairing = round(1.0 - unpaired, 3) if unpaired is not None else ""

    # --- heuristic composites (labelled)
    te = round(0.6 * cai + 0.4 * (tv or 0.0), 3)
    kozak_g4 = 1 if len(seq) > 6 and seq[6] == "G" else 0
    init = round(
        0.5 * min(max(acc30, 0.0), 1.0)
        + 0.4 * (1.0 - min(max(-dg45, 0.0) / 20.0, 1.0))
        + 0.1 * kozak_g4, 3)

    # --- half-life proxy: oracle decay heads (weak, cross-gene caveat)
    X = np.array([ctx.feature_vec(seq)])
    hlf = float(np.mean([ctx.models[h].predict(X)[0] for h in DECAY_HEADS]))

    # --- immune rule flag (unvalidated)
    upa = dinuc_odds(rna, "U", "A")
    cpg = dinuc_odds(rna, "C", "G")
    urich = sum(1 for i in range(len(rna) - 5)
                if rna[i:i + 6].count("U") >= 4)
    imp = immune_flag(dsrna, upa, urich, n, cpg)

    return {
        "expr_z": round(z_nat + z_cai, 2),
        "te_comp": te,
        "init_comp": init,
        "dg_start45": round(dg45, 2),
        "acc_aug30": round(acc30, 3),
        "pairing_prob": pairing,
        "tai": round(tv, 3) if tv is not None else "",
        "cpb": round(cpb_v, 3),
        "gc_w60_max": round(gw_max, 3),
        "gc_w60_min": round(gw_min, 3),
        "dsrna_bp": dsrna,
        "halflife_pct": round(hlf, 1),
        "immune_flag": imp,
    }


NEW_COLS = ["expr_z", "te_comp", "init_comp", "dg_start45", "acc_aug30",
            "pairing_prob", "tai", "cpb", "gc_w60_max", "gc_w60_min",
            "dsrna_bp", "halflife_pct", "immune_flag"]


class _Ctx(object):
    pass


def build_ctx():
    ctx = _Ctx()
    ctx.table, ctx.codon_coef, ctx.syn, ctx.models = load_env()
    ctx.scorer = Scorer()

    def feature_vec(cds):
        from refine_t5 import feature_vec as fv
        return fv(cds, ctx.table, ctx.codon_coef)

    ctx.feature_vec = feature_vec
    ctx.trna = load_trna_counts()
    ctx.wcache = {}
    ctx.cpb = build_cpb_fn(os.path.join(HERE, "models",
                                        "composite_v2_stats.pkl"))
    return ctx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(HERE, "data", "deliverable",
                                         "independent_topk.tsv"))
    ap.add_argument("--out", dest="out", default=None)
    args = ap.parse_args()
    out_path = args.out or args.inp

    ctx = build_ctx()
    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    fields = list(rows[0].keys()) + \
        [c for c in NEW_COLS if c not in rows[0].keys()]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields, delimiter="\t",
                              extrasaction="ignore")
        wcsv.writeheader()
        for i, r in enumerate(rows, 1):
            r.update(score_row(r["seq"], ctx))
            wcsv.writerow(r)
            print("[card] %d/%d %-12s te=%.3f init=%.3f dg=%s acc=%s "
                  "tai=%s cpb=%s dsrna=%s hl=%.1f flag=%d"
                  % (i, len(rows), r["protein"], r["te_comp"], r["init_comp"],
                     r["dg_start45"], r["acc_aug30"], r["tai"], r["cpb"],
                     r["dsrna_bp"], r["halflife_pct"], r["immune_flag"]),
                  flush=True)
    print("[card] -> %s (%d rows)" % (out_path, len(rows)))


if __name__ == "__main__":
    main()
