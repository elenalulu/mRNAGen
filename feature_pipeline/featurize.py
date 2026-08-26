#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""T3 feature pipeline CLI: FASTA or LinearDesign output -> feature TSV.

Usage (MUST run under D:/anaconda/python.exe -- it has ViennaRNA):

  D:/anaconda/python.exe featurize.py --in lineardesign_output.txt \
      --out features.tsv --ld-check

  D:/anaconda/python.exe featurize.py --in candidates.fasta --out features.tsv

Input auto-detection: file containing '@@PROTEIN' -> LinearDesign lambda-grid
dump (T1 output, scp'd from the GPU box); file with '>' headers -> FASTA.

--ld-check cross-validates our MFE/CAI computation against the values
LinearDesign itself reported (built-in correctness gate; CAI must match to
3 decimals since we ported their exact formula, MFE within ViennaRNA
parameter-version tolerance).
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from features.seq_stats import seq_features, normalize_rna  # noqa: E402
from features.codon_metrics import (  # noqa: E402
    get_default_table, codon_features,
)
from features.structure import structure_features  # noqa: E402
from features.rules import rule_features  # noqa: E402
from ld_parse import parse_lineardesign_output  # noqa: E402


def read_fasta(path):
    recs = []
    name = None
    buf = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(buf)))
                name = line[1:].split()[0] if len(line) > 1 else "seq"
                buf = []
            elif line:
                buf.append(line)
    if name is not None:
        recs.append((name, "".join(buf)))
    return recs


def detect_format(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "@@PROTEIN" in line or line.startswith("mRNA sequence:"):
                return "lineardesign"
            if line.startswith(">"):
                return "fasta"
    return "fasta"


META_COLS = ["seq_id", "source", "lam", "protein_len", "stop_codon",
             "translate_ok"]


def featurize(seq, table, fast=False, protein=None):
    rna = normalize_rna(seq)
    if len(rna) % 3 != 0:
        raise ValueError("sequence length not a multiple of 3: %d" % len(rna))
    out = {}
    out.update(seq_features(rna))
    out.update(codon_features(rna, table))
    out.update(structure_features(rna, fast=fast))
    out.update(rule_features(rna))
    if protein:
        out["translate_ok"] = int(table.check_translate(rna, protein))
    else:
        out["translate_ok"] = ""
    return out


def order_columns(records):
    """Stable column order: meta, seq, codon, structure, rules, LD check."""
    seq_cols = ["seq_len", "gc_global", "gc_slide60_max", "gc_slide60_min",
                "upa_odds", "cpg_odds", "hp_max_A", "hp_max_C", "hp_max_G",
                "hp_max_U", "urich6_count"]
    codon_cols = ["cai", "cai_excl_stop", "gc3", "enc"]
    struct_cols = ["mfe", "mfe_per_nt", "paired_frac_mfe", "longest_helix_mfe",
                   "n_helix_ge8", "ensemble_ok", "mean_unpaired",
                   "mean_unpaired_q25", "open_start45", "selfcomp_max_exact",
                   "selfcomp_max_near"]
    rule_cols = ["rule_gc_global_pass", "rule_gc_slide_pass",
                 "rule_homopolymer_pass", "restriction_site_count",
                 "cryptic_donor_count", "urich6_count_rules_dup",
                 "rules_all_pass"]
    ld_cols = ["mfe_reported", "cai_reported", "delta_mfe", "delta_cai"]
    extra = []
    seen = set()
    for group in (META_COLS, seq_cols, codon_cols, struct_cols, rule_cols,
                  ld_cols):
        seen.update(group)
    for r in records:
        for k in r:
            if k not in seen:
                seen.add(k)
                extra.append(k)
    cols = [c for c in META_COLS if any(c in r for r in records)]
    cols += [c for c in seq_cols + codon_cols + struct_cols + rule_cols
             if any(c in r for r in records)]
    cols += [c for c in ld_cols if any(c in r for r in records)]
    cols += extra
    return cols


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="skip ensemble pf + self-complementarity scan")
    ap.add_argument("--ld-check", action="store_true",
                    help="print MFE/CAI cross-validation vs LinearDesign")
    args = ap.parse_args()

    table = get_default_table()
    fmt_in = detect_format(args.inp)
    records = []

    if fmt_in == "lineardesign":
        for rec in parse_lineardesign_output(args.inp):
            f = featurize(rec["seq"], table, fast=args.fast,
                          protein=rec["protein"])
            f["seq_id"] = rec["seq_id"]
            f["source"] = "lineardesign"
            f["lam"] = rec["lam"]
            f["protein_len"] = len(rec["protein"])
            if rec["mfe_reported"] is not None:
                f["mfe_reported"] = rec["mfe_reported"]
                f["delta_mfe"] = f["mfe"] - rec["mfe_reported"]
            if rec["cai_reported"] is not None:
                f["cai_reported"] = rec["cai_reported"]
                f["delta_cai"] = f["cai"] - rec["cai_reported"]
            records.append(f)
    else:
        for name, seq in read_fasta(args.inp):
            f = featurize(seq, table, fast=args.fast)
            f["seq_id"] = name
            f["source"] = "fasta"
            f["lam"] = ""
            f["protein_len"] = len(seq) // 3
            records.append(f)

    if not records:
        print("no sequences parsed from %s" % args.inp, file=sys.stderr)
        sys.exit(1)

    cols = order_columns(records)
    out_path = args.out or (
        os.path.splitext(args.inp)[0] + "_features.tsv"
    )
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in records:
            fh.write("\t".join(fmt(r.get(c)) for c in cols) + "\n")
    print("[featurize] %d sequences -> %s (%d cols)" % (
        len(records), out_path, len(cols)))

    if args.ld_check:
        print("\n[ld-check] cross-validation vs LinearDesign reported values")
        print("%-18s %10s %10s %9s | %8s %8s %9s" % (
            "seq_id", "MFE_rep", "MFE_ours", "dMFE",
            "CAI_rep", "CAI_ours", "dCAI"))
        worst_mfe = worst_cai = 0.0
        for r in records:
            if "mfe_reported" not in r:
                continue
            dm = abs(r["delta_mfe"])
            dc = abs(r["delta_cai"])
            worst_mfe = max(worst_mfe, dm)
            worst_cai = max(worst_cai, dc)
            print("%-18s %10.2f %10.2f %9.2f | %8.3f %8.3f %9.4f" % (
                r["seq_id"], r["mfe_reported"], r["mfe"], r["delta_mfe"],
                r["cai_reported"], r["cai"], r["delta_cai"]))
        print("worst |dMFE| = %.2f kcal/mol, worst |dCAI| = %.4f" % (
            worst_mfe, worst_cai))
        if worst_cai > 0.0015:
            print("WARNING: CAI deviation exceeds LinearDesign print "
                  "precision -- formula mismatch!")
        else:
            print("CAI port OK (matches LinearDesign to print precision).")


if __name__ == "__main__":
    main()
