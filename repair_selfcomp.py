#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Surgical self-complementarity repair (Plan C, 2026-08-25).

For each refined candidate: repeatedly locate the longest intramolecular
near-perfect duplex (selfcomp_max_near = MDA5-class dsRNA risk proxy) and
try single synonymous-codon swaps restricted to codons overlapping the
duplex arms. Greedy accept if ALL gates hold vs the PRE-REPAIR baseline:

  gates (vs pre-repair sequence):
    - global selfcomp_max_near strictly decreases (vs current)
    - composite v2 >= baseline - 0.10
    - z_nat        >= baseline - 0.05
    - CAI          >= baseline - 0.005
    - restriction sites / hp5 runs do not increase
    - global GC stays in [0.30, 0.70]

Protein sequence provably unchanged (synonymous swaps only, asserted by
back-translation before/after). S3 external-validation evidence carries
over: only the deterministic manufacturing/self-comp axes move, the
naturalness+CAI ranking core is bounded by the gates above.

Why arm-restricted: a swap that breaks no aligned pair of the current
near-max duplexes cannot shrink the global max (that duplex survives
unchanged), so the pre-filter is exact, not heuristic.

Outputs (consumed by select_topk.py via the *_sc preference chain):
  data/t5/refined_candidates_v2_sc.tsv        repaired LD-start pool
  data/t5/refined_gemorna_v2_sc.tsv           repaired GEMORNA-start pool
  data/t5/structure_posthoc_sc.tsv            recomputed structure feats
  data/t5/structure_posthoc_gemorna_sc.tsv   recomputed (GEMORNA starts)
  data/t5/repair_selfcomp_report.tsv          per-candidate deltas

Run: python repair_selfcomp.py
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
ROOT = os.path.dirname(HERE)
for p in (FP, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from features.codon_metrics import get_default_table  # noqa: E402
from refine_t5 import (GC_LO, GC_HI, hard_ok, n_hp4, n_hp5,  # noqa: E402
                       n_sites)

T5 = os.path.join(HERE, "data", "t5")

DNA_COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}
SEED = 10            # feature_pipeline SELFCOMP_SEED
TOL = 2              # feature_pipeline SELFCOMP_MAX_MISMATCH
CAP = 20000          # feature_pipeline SELFCOMP_PAIR_CAP
TARGET = 12          # stop repairing once at/below this
REPAIR_MIN = 15      # skip candidates already below this
MAX_ROUNDS = 15
NEAR = 3             # also attack duplexes within 3 bp of the max
G_COMP = 0.10        # max composite-v2 drop allowed
G_NAT = 0.05         # max z_nat drop allowed
G_CAI = 0.005        # max CAI drop allowed


def revcomp(s):
    return "".join(DNA_COMP.get(c, "N") for c in reversed(s))


def _extend_span(seq, i, j):
    """feature_pipeline _extend_duplex, position-returning variant.

    Arms X=[xx1..x2], Y=[y1..yy2]; aligned pairs satisfy p+q == S with
    S = xx1 + yy2 (== x2 + y1). Returns (L, xx1, x2, y1, yy2).
    """
    n = len(seq)
    x2, y1 = i + SEED - 1, j
    xx1, yy2 = i, j + SEED - 1
    mis = 0
    while mis <= TOL:
        moved = False
        if y1 - 1 > x2 + 1 and x2 + 1 < n and y1 - 1 >= 0:
            if DNA_COMP.get(seq[x2 + 1]) == seq[y1 - 1]:
                x2 += 1
                y1 -= 1
                moved = True
            elif mis < TOL:
                x2 += 1
                y1 -= 1
                mis += 1
                moved = True
        if xx1 - 1 >= 0 and yy2 + 1 < n:
            if DNA_COMP.get(seq[xx1 - 1]) == seq[yy2 + 1]:
                xx1 -= 1
                yy2 += 1
                moved = True
            elif mis < TOL:
                xx1 -= 1
                yy2 += 1
                mis += 1
                moved = True
        if not moved:
            break
    return x2 - xx1 + 1, xx1, x2, y1, yy2


def scan_duplexes(seq):
    """(best_len, spans) where spans are near-max duplex arm tuples.

    Mirrors feature_pipeline selfcomp_scan on the DNA alphabet (duplex
    geometry is identical to the RNA version)."""
    n = len(seq)
    if n < 2 * SEED + 3:
        return 0, []
    from collections import defaultdict
    pos = defaultdict(list)
    for i in range(n - SEED + 1):
        pos[seq[i:i + SEED]].append(i)
    spans = []
    expanded = 0
    done = False
    for kmer, plist in pos.items():
        if done:
            break
        rc = revcomp(kmer)
        partners = pos.get(rc)
        if not partners:
            continue
        same = (rc == kmer)
        for i in plist:
            if done:
                break
            for j in partners:
                if same and j <= i:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                if hi - lo < SEED + 3:
                    continue
                expanded += 1
                if expanded > CAP:
                    done = True
                    break
                L, a, b, c, d = _extend_span(seq, lo, hi)
                spans.append((L, a, b, c, d))
    best = max((s[0] for s in spans), default=0)
    keep = [s for s in spans if s[0] >= max(SEED, best - NEAR)]
    return best, keep


def aa_codon_map(table):
    m = {}
    for codon, aa in table.codon_to_aa.items():
        if aa in ("*", "STOP"):
            continue
        m.setdefault(aa, []).append(codon.replace("U", "T"))
    return {k: sorted(v) for k, v in m.items()}


def translate(seq, table):
    aas = []
    for i in range(0, len(seq) - 2, 3):
        aa = table.codon_to_aa.get(seq[i:i + 3].replace("T", "U"))
        if aa is None:
            return None
        if aa in ("*", "STOP"):
            break
        aas.append(aa)
    return "".join(aas)


def metrics(seq, table, scorer):
    """Composite v2 (same formula as refine_t5.score_batch, v2 branch)."""
    z_nat = float(scorer.z_nat([seq])[0])
    cai = table.cai(seq.replace("T", "U"))
    z_cai = float(scorer.z_cai([cai])[0])
    sites, hp4, hp5 = n_sites(seq), n_hp4(seq), n_hp5(seq)
    gc = (seq.count("G") + seq.count("C")) / len(seq)
    gc_pen = 5.0 * (max(0.0, gc - GC_HI) + max(0.0, GC_LO - gc))
    comp = z_nat + z_cai - (0.5 * sites + 0.1 * hp4 + 0.8 * hp5 + gc_pen)
    return {"comp": comp, "nat": z_nat, "cai": cai, "sites": sites,
            "hp4": hp4, "hp5": hp5, "gc": gc}


def gates_ok(m, base):
    return (m["comp"] >= base["comp"] - G_COMP
            and m["nat"] >= base["nat"] - G_NAT
            and m["cai"] >= base["cai"] - G_CAI
            and m["sites"] <= base["sites"]
            and m["hp5"] <= base["hp5"]
            and GC_LO <= m["gc"] <= GC_HI)


def breaks_any_pair(cur, spans, ci, alt):
    """True if swapping codon `ci` to `alt` breaks >=1 aligned pair in one
    of the near-max duplexes (necessary to shrink the global max)."""
    p0 = ci * 3
    for (_L, x1, x2, y1, y2) in spans:
        S = x1 + y2
        for k in range(3):
            p = p0 + k
            q = S - p
            if (x1 <= p <= x2 and y1 <= q <= y2) or \
               (y1 <= p <= y2 and x1 <= q <= x2):
                if DNA_COMP.get(alt[k]) != cur[q]:
                    return True
    return False


def variants_breaking(cur, spans, cmap, table):
    ncod = len(cur) // 3
    touched = set()
    for (_L, x1, x2, y1, y2) in spans:
        for (a, b) in ((x1, x2), (y1, y2)):
            for ci in range(max(0, a // 3), min(ncod - 1, (b - 1) // 3) + 1):
                touched.add(ci)
    for ci in sorted(touched):
        cod = cur[ci * 3:ci * 3 + 3]
        aa = table.codon_to_aa.get(cod.replace("T", "U"))
        if aa in (None, "*", "STOP"):
            continue
        for alt in cmap[aa]:
            if alt == cod:
                continue
            if breaks_any_pair(cur, spans, ci, alt):
                yield ci, alt, cur[:ci * 3] + alt + cur[ci * 3 + 3:]


def repair(seq, table, scorer, cmap):
    base = metrics(seq, table, scorer)
    sc0, _spans = scan_duplexes(seq)
    if sc0 < REPAIR_MIN:
        return seq, 0, sc0, sc0, base, base
    cur, cur_sc, n_swaps = seq, sc0, 0
    for _ in range(MAX_ROUNDS):
        L, spans = scan_duplexes(cur)
        if L < SEED or not spans:
            break
        best = None
        for _ci, _alt, var in variants_breaking(cur, spans, cmap, table):
            v_sc, _vs = scan_duplexes(var)
            if v_sc >= cur_sc:
                continue
            m = metrics(var, table, scorer)
            if not gates_ok(m, base):
                continue
            key = (v_sc, -m["comp"])
            if best is None or key < best[0]:
                best = (key, var, v_sc, m)
        if best is None:
            break
        cur, cur_sc = best[1], best[2]
        n_swaps += 1
        if cur_sc <= TARGET:
            break
    return cur, n_swaps, sc0, cur_sc, base, metrics(cur, table, scorer)


def structure_row(seq):
    from features.structure import structure_features
    f = structure_features(seq.replace("T", "U"))
    return f


def write_tsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


POOLS = [
    ("refined_candidates_v2.tsv", "refined_candidates_v2_sc.tsv",
     "structure_posthoc_sc.tsv"),
    ("refined_gemorna_v2.tsv", "refined_gemorna_v2_sc.tsv",
     "structure_posthoc_gemorna_sc.tsv"),
]


def main():
    table = get_default_table()
    from naturalness import Scorer
    scorer = Scorer()
    cmap = aa_codon_map(table)
    report = []
    for in_name, out_name, post_name in POOLS:
        in_path = os.path.join(T5, in_name)
        if not os.path.exists(in_path):
            print("[sc] skip missing %s" % in_name)
            continue
        with open(in_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        out_rows, post_rows = [], []
        t0 = time.time()
        n_rep = 0
        for r in rows:
            seq = r["cds_refined"].upper().replace("U", "T")
            t1 = time.time()
            aa_before = translate(seq, table)
            out, n_swaps, sc0, sc1, base, after = repair(
                seq, table, scorer, cmap)
            assert translate(out, table) == aa_before, "protein changed!"
            rr = dict(r)
            rr["cds_refined"] = out
            if n_swaps:
                rr["sites_end"] = n_sites(out)
                rr["hp4_end"] = n_hp4(out)
                rr["hp5_end"] = n_hp5(out)
                rr["score_end"] = round(after["comp"], 4)
                rr["delta"] = round(after["comp"]
                                    - float(r["score_start"]), 4)
                rr["hard_pass_end"] = int(hard_ok(out))
                n_rep += 1
            out_rows.append(rr)
            sf = structure_row(out)
            post_rows.append({
                "protein": r["protein"], "lam": r["lam_start"],
                "mfe_ref": round(sf["mfe"], 1),
                "selfcomp_near_ref": sf["selfcomp_max_near"],
                "selfcomp_ex_ref": sf["selfcomp_max_exact"],
                "open45_ref": round(sf["open_start45"], 3),
            })
            report.append({
                "pool": in_name, "protein": r["protein"],
                "lam": r["lam_start"], "n_swaps": n_swaps,
                "selfcomp_before": sc0, "selfcomp_after": sc1,
                "repaired": int(bool(n_swaps)),
                "score_before": round(base["comp"], 3),
                "score_after": round(after["comp"], 3),
            })
            if n_swaps:
                print("[sc] %-14s lam=%-8s %2d->%2d bp (%d swaps) "
                      "score %.2f->%.2f [%.1fs]"
                      % (r["protein"], r["lam_start"], sc0, sc1, n_swaps,
                         base["comp"], after["comp"], time.time() - t1),
                      flush=True)
        write_tsv(os.path.join(T5, out_name), out_rows)
        write_tsv(os.path.join(T5, post_name), post_rows)
        print("[sc] %s: %d rows, %d repaired (%.1fs)"
              % (out_name, len(out_rows), n_rep, time.time() - t0),
              flush=True)
    write_tsv(os.path.join(T5, "repair_selfcomp_report.tsv"), report)
    rep = [r for r in report if r["repaired"]]
    if rep:
        mean_b = sum(r["selfcomp_before"] for r in rep) / len(rep)
        mean_a = sum(r["selfcomp_after"] for r in rep) / len(rep)
        print("[sc] repaired %d/%d candidates: selfcomp %.1f -> %.1f bp"
              % (len(rep), len(report), mean_b, mean_a))
        print("[sc] score cost: mean %+.3f (max %+.3f)"
              % (sum(r["score_after"] - r["score_before"] for r in rep)
                 / len(rep),
                 min(r["score_after"] - r["score_before"] for r in rep)))
    print("[sc] -> %s" % os.path.join(T5, "repair_selfcomp_report.tsv"))


if __name__ == "__main__":
    main()
