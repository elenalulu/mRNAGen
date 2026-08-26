#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DNA Chisel (MIT) independent-start pipeline.

Start lineage: 100% independent of GEMORNA / LinearDesign tools.

  DNA Chisel (manufacturing-clean start: restriction sites + GC windows
  + translation preserved, human CAI objective)
      -> our greedy refinement (composite v2, 25 rounds)
      -> our surgical selfcomp repair (Plan C)

Run with the isolated venv that has dnachisel:
  D:/WorkBuddy/home/binaries/python/envs/dnachisel/Scripts/python.exe \
      pipeline_dnachisel.py

Outputs (incremental, crash-proof):
  data/deliverable/dnachisel_candidates.tsv
"""
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FP = os.path.join(HERE, "feature_pipeline")
for p in (FP, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from refine_t5 import load_env, refine, n_sites, n_hp4, n_hp5  # noqa: E402
from naturalness import Scorer  # noqa: E402
from repair_selfcomp import (aa_codon_map, repair as sc_repair,  # noqa: E402
                             scan_duplexes)

OUT = os.path.join(HERE, "data", "deliverable")
os.makedirs(OUT, exist_ok=True)

TOPK_V3 = (r"D:\WorkBuddy\alphafold-web\rna_platform\mrna_neoantigen"
           r"\mRNAGen\data\t5\topk_selection.tsv")

SITES = ["GGTCTC", "GAGACC", "CGTCTC", "GAGACG", "GAATTC",
         "CTCGAG", "GGATCC", "GCGGCCGC"]
MAX_ROUNDS = 25

FIELDS = ["protein", "seed_score", "dc_score", "refined_score",
          "z_nat", "cai", "sites", "hp4", "hp5", "gc",
          "selfcomp_before", "selfcomp_after", "repair_swaps",
          "repair_score", "v3_best", "gap_v3", "secs"]


def load_aa(name):
    with open(os.path.join(HERE, "data", "proteins", "benchmark",
                           name + ".txt")) as f:
        return "".join(f.read().split())


def cai_max_start(aa):
    """Independent seed: highest-frequency codon per AA (Kazusa table)."""
    freq = {}
    with open(os.path.join(HERE, "feature_pipeline", "data",
                           "codon_usage_freq_table_human.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) >= 3 and len(r[0]) == 3 and r[1] != "*":
                freq.setdefault(r[1], []).append((r[0].replace("U", "T"),
                                                  float(r[2])))
    return "".join(max(freq[a], key=lambda x: x[1])[0] for a in aa)


def translate(seq, table):
    aas = []
    for i in range(0, len(seq) - 2, 3):
        a = table.codon_to_aa.get(seq[i:i + 3].replace("T", "U"))
        if a in (None, "*", "STOP"):
            break
        aas.append(a)
    return "".join(aas)


def dnachisel_start(name, seed, aa, table):
    """Manufacturing-clean start via DNA Chisel (MIT)."""
    from dnachisel import (DnaOptimizationProblem, AvoidPattern,
                           EnforceGCContent, EnforceTranslation,
                           CodonOptimize)
    constraints = [AvoidPattern(s) for s in SITES]
    constraints.append(EnforceGCContent(mini=0.30, maxi=0.70, window=60))
    constraints.append(EnforceTranslation(translation=aa,
                                          location=(0, len(seed))))
    objectives = [CodonOptimize("h_sapiens", location=(0, len(seed)))]
    prob = DnaOptimizationProblem(sequence=seed, constraints=constraints,
                                  objectives=objectives)
    prob.resolve_constraints()
    prob.optimize()
    s = prob.sequence.upper()
    if translate(s, table) != aa:
        print("[dc] WARN %s: translation changed after DNA Chisel, "
              "keeping CAI-max seed" % name)
        return seed
    return s


def save_results(out):
    if not out:
        return
    with open(os.path.join(OUT, "dnachisel_candidates.tsv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)


def v3_best():
    best = {}
    if not os.path.exists(TOPK_V3):
        return best
    with open(TOPK_V3, encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("source") != "refined":
                continue
            s = float(r["score"])
            if r["protein"] not in best or s > best[r["protein"]]:
                best[r["protein"]] = s
    return best


def run_one(name, table, codon_coef, models, scorer, cmap, v3, out):
    t0 = time.time()
    aa = load_aa(name)
    seed = cai_max_start(aa)
    dc = dnachisel_start(name, seed, aa, table)
    end, _s0, s1, n_moves = refine(dc, table, codon_coef, models,
                                   max_rounds=MAX_ROUNDS)
    z_nat = float(scorer.z_nat([end])[0])
    cai = table.cai(end.replace("T", "U"))
    sites, hp4, hp5 = n_sites(end), n_hp4(end), n_hp5(end)
    gc = (end.count("G") + end.count("C")) / len(end)
    sc_b = scan_duplexes(end)[0]
    repaired, n_swaps, _a, sc_after, _b, after = sc_repair(
        end, table, scorer, cmap)
    sc_a = scan_duplexes(repaired)[0]
    v3b = v3.get(name)
    row = {
        "protein": name,
        "seed_score": round(score_of(seed, table, codon_coef, models,
                                     scorer), 3),
        "dc_score": round(score_of(dc, table, codon_coef, models, scorer), 3),
        "refined_score": round(s1, 3),
        "z_nat": round(z_nat, 2), "cai": round(cai, 3),
        "sites": sites, "hp4": hp4, "hp5": hp5, "gc": round(gc, 3),
        "selfcomp_before": sc_b, "selfcomp_after": sc_a,
        "repair_swaps": n_swaps,
        "repair_score": round(after["comp"], 3),
        "v3_best": v3b, "gap_v3": round(v3b - after["comp"], 2) if v3b else None,
        "secs": round(time.time() - t0, 1),
    }
    out.append(row)
    save_results(out)
    print("[dc] %-12s seed=%+.2f dc=%+.2f refine=%+.2f (moves=%d) "
          "nat=%+.2f cai=%.3f sites=%d hp5=%d selfcomp %d->%d "
          "repair=%+.2f | v3_best=%s gap=%s [%.0fs]"
          % (name, row["seed_score"], row["dc_score"], s1, n_moves,
             z_nat, cai, sites, hp5, sc_b, sc_a, row["repair_score"],
             v3b, row["gap_v3"], time.time() - t0), flush=True)


def score_of(seq, table, codon_coef, models, scorer):
    from refine_t5 import composite
    return composite(seq, table, codon_coef, models)[0]


def main():
    t, codon_coef, syn, models = load_env()
    scorer = Scorer()
    cmap = aa_codon_map(t)
    v3 = v3_best()
    out = []
    proteins = [f[:-4] for f in sorted(os.listdir(
        os.path.join(HERE, "data", "proteins", "benchmark")))]
    for name in proteins:
        try:
            run_one(name, t, codon_coef, models, scorer, cmap, v3, out)
        except Exception as e:
            import traceback
            print("[dc] ERROR %s: %r" % (name, e), flush=True)
            traceback.print_exc()
    print("[dc] done -> %s" % os.path.join(OUT,
                                           "dnachisel_candidates.tsv"))


if __name__ == "__main__":
    main()
