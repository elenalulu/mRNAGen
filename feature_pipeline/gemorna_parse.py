# -*- coding: utf-8 -*-
"""GEMORNA offline output parser (generate.py stdout dump).

Input format (run_gemorna_offline.sh output):

    @@PROTEIN MGADGVGKS...
    @@REP 1
    (blank)
    Generated CDS & Naturalness
    ATGGGGGCTGAT...GAGAAG 0.59
    (blank)
    @@REP 2
    ...

Sequences are UPPERCASE DNA (T alphabet) with a trailing "naturalness"
score printed by GEMORNA. No stop codon is appended (protein has no '*').
"""
import re

_SEQ_LINE = re.compile(
    r"^\s*([ATUGCatugc]{30,})\s+([01]?\.\d+)\s*$"
)


def parse_gemorna_output(path):
    records = []
    protein = None
    rep = None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("@@PROTEIN"):
                protein = line[len("@@PROTEIN"):].strip()
            elif line.startswith("@@REP"):
                try:
                    rep = int(line[len("@@REP"):].strip())
                except ValueError:
                    rep = None
            else:
                m = _SEQ_LINE.match(line)
                if m:
                    records.append({
                        "protein": protein or "",
                        "rep": rep,
                        "seq": m.group(1).upper(),
                        "naturalness": float(m.group(2)),
                        "seq_id": "GE_len%d_rep%s" % (
                            len(protein or ""), rep),
                    })
    return records
