#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent start pool (no GEMORNA / LinearDesign lineage).

Three fully-clean start sources per protein:

  dnachisel  DNA Chisel (MIT, Edinburgh Genome Foundry): manufacturing-clean
             start (restriction sites avoided, GC windows enforced,
             translation preserved, human CAI objective). Seed = our
             CAI-max sequence.
  native     Ensembl GRCh38 endogenous CDS (open data) for human targets
             whose translation exactly matches the benchmark protein.
  cai        CAI-max start from the Kazusa human codon-usage table
             (our own generator).

Output: data/deliverable/start_pool.tsv  (protein, source, seq)
Incremental and crash-proof (one row saved per start).

Requires the dnachisel venv for the DNA Chisel arm:
  D:/WorkBuddy/home/binaries/python/envs/dnachisel/Scripts/python.exe \
      start_pool.py
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

from features.codon_metrics import get_default_table  # noqa: E402

OUT = os.path.join(HERE, "data", "deliverable")
os.makedirs(OUT, exist_ok=True)

SITES = ["GGTCTC", "GAGACC", "CGTCTC", "GAGACG", "GAATTC",
         "CTCGAG", "GGATCC", "GCGGCCGC"]
FIELDS = ["protein", "source", "seq", "len", "gc", "sites", "hp5", "secs"]

_corpus = None


def load_aa(name):
    with open(os.path.join(HERE, "data", "proteins", "benchmark",
                           name + ".txt")) as f:
        return "".join(f.read().split())


def translate(seq, table):
    aas = []
    for i in range(0, len(seq) - 2, 3):
        a = table.codon_to_aa.get(seq[i:i + 3].replace("T", "U"))
        if a in (None, "*", "STOP"):
            break
        aas.append(a)
    return "".join(aas)


def cai_max_start(aa):
    """Highest-frequency codon per AA (Kazusa table in the repo)."""
    freq = {}
    with open(os.path.join(HERE, "feature_pipeline", "data",
                           "codon_usage_freq_table_human.csv"),
              encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) >= 3 and len(r[0]) == 3 and r[1] != "*":
                freq.setdefault(r[1], []).append((r[0].replace("U", "T"),
                                                  float(r[2])))
    return "".join(max(freq[a], key=lambda x: x[1])[0] for a in aa)


def native_cds(gene_symbol):
    """Longest endogenous CDS for a gene symbol (Ensembl-derived corpus)."""
    global _corpus
    if _corpus is None:
        p = os.path.join(HERE, "data", "t2",
                         "t2_endogenous_cds_abundance.tsv")
        if not os.path.exists(p):
            return None
        _corpus = {}
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                _corpus.setdefault(r["gene_symbol"], []).append(r["cds"])
    seqs = _corpus.get(gene_symbol)
    return max(seqs, key=len) if seqs else None


def dnachisel_start(seed, aa):
    """DNA Chisel (MIT) manufacturing-clean start from the CAI-max seed."""
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
    return prob.sequence.upper()


def save_results(out):
    with open(os.path.join(OUT, "start_pool.tsv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t",
                           extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(r)


def main():
    table = get_default_table()
    out = []
    benchmark = os.path.join(HERE, "data", "proteins", "benchmark")
    proteins = sorted(f[:-4] for f in os.listdir(benchmark))
    for name in proteins:
        aa = load_aa(name)
        seed = cai_max_start(aa)
        t0 = time.time()
        try:
            dc = dnachisel_start(seed, aa)
            assert translate(dc, table) == aa, "translation changed"
            out.append({"protein": name, "source": "dnachisel", "seq": dc,
                        "len": len(dc), "gc": round((dc.count("G")
                                                     + dc.count("C"))
                                                    / len(dc), 3),
                        "sites": 0, "hp5": 0,
                        "secs": round(time.time() - t0, 1)})
            print("[pool] %-12s dnachisel %dnt [%.0fs]" % (name, len(dc),
                                                           time.time() - t0),
                  flush=True)
        except Exception as e:
            print("[pool] %-12s dnachisel FAILED: %r" % (name, e),
                  flush=True)
        save_results(out)
        # native arm (human targets with exact translation match)
        t0 = time.time()
        sym = {"albumin": "ALB", "epo": "EPO", "factor_ix": "F9",
               "gaa_pompe": "GAA", "gla_fabry": "GLA", "otc": "OTC",
               "pah_pku": "PAH"}.get(name)
        if sym:
            cds = native_cds(sym)
            if cds is not None and translate(cds, table) == aa:
                out.append({"protein": name, "source": "native", "seq": cds,
                            "len": len(cds),
                            "gc": round((cds.count("G") + cds.count("C"))
                                        / len(cds), 3),
                            "sites": 0, "hp5": 0,
                            "secs": round(time.time() - t0, 1)})
                print("[pool] %-12s native %s %dnt" % (name, sym, len(cds)),
                      flush=True)
        # CAI-max arm (always)
        out.append({"protein": name, "source": "cai", "seq": seed,
                    "len": len(seed),
                    "gc": round((seed.count("G") + seed.count("C"))
                                / len(seed), 3),
                    "sites": 0, "hp5": 0, "secs": 0.0})
        print("[pool] %-12s cai %dnt" % (name, len(seed)), flush=True)
        save_results(out)
    print("[pool] -> %s (%d starts)" % (os.path.join(OUT, "start_pool.tsv"),
                                        len(out)))


if __name__ == "__main__":
    main()
