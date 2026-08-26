#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T2-1: assemble endogenous human CDS x protein abundance weak supervision.

Sources (all downloaded to data/t2/):
  - Homo_sapiens.GRCh38.cds.all.fa.gz   (Ensembl release 115; ID = ENST)
  - Homo_sapiens.GRCh38.pep.all.fa.gz   (Ensembl release 115; ENSP -> ENST bridge)
  - paxdb_human_whole_organism.tsv      (PaxDb v6 whole-organism integrated,
                                         dataset_id 2882570067, ENSP-keyed)

Join chain: PaxDb ENSP -> (pep header) ENST + gene -> (CDS fasta) CDS seq.

Output: data/t2/t2_endogenous_cds_abundance.tsv
  ensp / enst / gene / gene_symbol / biotype / cds_len / abundance_ppm /
  rank / cds_sequence (DNA, T alphabet)

Run: python build_t2_endogenous.py
"""
import csv
import gzip
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.join(HERE, "data", "t2")

CDS_FA = os.path.join(T2, "Homo_sapiens.GRCh38.cds.all.fa.gz")
PEP_FA = os.path.join(T2, "Homo_sapiens.GRCh38.pep.all.fa.gz")
PAX_TSV = os.path.join(T2, "paxdb_human_whole_organism.tsv")
OUT_TSV = os.path.join(T2, "t2_endogenous_cds_abundance.tsv")

_GENE_RE = re.compile(r"gene:([A-Z0-9.]+)")
_ENST_RE = re.compile(r"transcript:([A-Z0-9.]+)")
_SYM_RE = re.compile(r"gene_symbol:([^ ]+)")
_BIO_RE = re.compile(r"gene_biotype:([^ ]+)")


def _parse_fasta_headers(path):
    """Yield (id_no_version, header) for each entry."""
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                hid = line[1:].split()[0].split(".")[0]
                yield hid, line


def main():
    # 1. ENSP -> (ENST, gene, symbol, biotype) from pep fasta
    ens_map = {}
    for ensp, header in _parse_fasta_headers(PEP_FA):
        m_e = _ENST_RE.search(header)
        m_g = _GENE_RE.search(header)
        m_s = _SYM_RE.search(header)
        m_b = _BIO_RE.search(header)
        ens_map[ensp] = (
            m_e.group(1).split(".")[0] if m_e else "",
            m_g.group(1).split(".")[0] if m_g else "",
            m_s.group(1) if m_s else "",
            m_b.group(1) if m_b else "",
        )
    print("[t2] pep entries (ENSP):", len(ens_map))

    # 2. ENST -> CDS sequence from CDS fasta
    cds = {}
    cur_id = None
    buf = []
    with gzip.open(CDS_FA, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if cur_id is not None:
                    cds[cur_id] = "".join(buf)
                cur_id = line[1:].split()[0].split(".")[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur_id is not None:
        cds[cur_id] = "".join(buf)
    print("[t2] CDS entries (ENST):", len(cds))

    # 3. PaxDb abundances
    pax = []
    with open(PAX_TSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            pax.append(row)
    print("[t2] PaxDb proteins:", len(pax))

    # 4. join + filter
    rows = []
    n_bridge = n_cds = n_badlen = n_nonstd = n_noncoding = 0
    seen = set()
    for row in pax:
        ensp = row["ensp"]
        if ensp in seen:
            continue
        seen.add(ensp)
        if ensp not in ens_map:
            n_bridge += 1
            continue
        enst, gene, sym, bio = ens_map[ensp]
        if enst not in cds:
            n_cds += 1
            continue
        seq = cds[enst].upper()
        if len(seq) < 90 or len(seq) % 3 != 0:
            n_badlen += 1
            continue
        if not re.match(r"^[ACGT]+$", seq):
            n_nonstd += 1
            continue
        if bio and bio != "protein_coding":
            n_noncoding += 1
            continue
        rows.append({
            "ensp": ensp,
            "enst": enst,
            "gene": gene,
            "gene_symbol": sym,
            "biotype": bio,
            "cds_len": len(seq),
            "abundance_ppm": row["abundance_ppm"],
            "rank": row["rank"],
            "cds": seq,
        })

    print("[t2] joined %d | no-ENSP-bridge %d | no-CDS %d | bad-len %d | "
          "nonstd %d | non-coding %d"
          % (len(rows), n_bridge, n_cds, n_badlen, n_nonstd, n_noncoding))

    # 5. write
    cols = ["ensp", "enst", "gene", "gene_symbol", "biotype", "cds_len",
            "abundance_ppm", "rank", "cds"]
    with open(OUT_TSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print("[t2] -> %s (%d rows)" % (OUT_TSV, len(rows)))

    # 6. quick stats
    ab = sorted(float(r["abundance_ppm"]) for r in rows)
    n = len(ab)
    print("[t2] abundance ppm: median=%.4g p90=%.4g p99=%.4g max=%.4g"
          % (ab[n // 2], ab[int(n * 0.9)], ab[int(n * 0.99)], ab[-1]))


if __name__ == "__main__":
    sys.exit(main())
