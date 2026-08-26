#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch featurizer: LD ld_outputs/ + GEMORNA output -> merged feature TSVs.

Usage (python has ViennaRNA):

  python featurize_all.py \
      --ld-dir github/LinearDesign-main/ld_outputs \
      --gemorna github/GEMORNA/gemorna_output.txt \
      --out-dir data

Outputs:
  <out-dir>/ld_benchmark_features.tsv    (LD lambda-scan, one row per
                                          protein x lambda, protein name kept)
  <out-dir>/gemorna_features.tsv         (GEMORNA reps, naturalness kept)
  <out-dir>/all_candidates_features.tsv  (both, with a generator column)
"""
import argparse
import glob
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_FP = os.path.join(_HERE, "feature_pipeline")
for p in (_FP, os.path.dirname(_FP)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ld_parse import parse_lineardesign_output  # noqa: E402
from gemorna_parse import parse_gemorna_output  # noqa: E402
from featurize import featurize, order_columns, fmt  # noqa: E402
from features.codon_metrics import get_default_table  # noqa: E402


def write_tsv(records, path):
    cols = order_columns(records)
    extra = [c for c in ("protein", "generator", "rep", "naturalness")
             if c not in cols and any(c in r for r in records)]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(extra + cols) + "\n")
        for r in records:
            fh.write("\t".join(fmt(r.get(c)) for c in extra + cols) + "\n")
    print("[featurize-all] %d rows x %d cols -> %s" % (
        len(records), len(extra) + len(cols), path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ld-dir", default=None,
                    help="dir of LinearDesign ld_outputs/<protein>.txt")
    ap.add_argument("--gemorna", default=None,
                    help="GEMORNA gemorna_output.txt")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "data"))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    table = get_default_table()
    all_recs = []

    if args.ld_dir:
        files = sorted(glob.glob(os.path.join(args.ld_dir, "*.txt")))
        if not files:
            print("[featurize-all] WARN: no .txt in %s" % args.ld_dir,
                  file=sys.stderr)
        for fp in files:
            name = os.path.splitext(os.path.basename(fp))[0]
            recs = parse_lineardesign_output(fp)
            if not recs:
                print("[featurize-all] WARN: %s parsed 0 records" % fp,
                      file=sys.stderr)
                continue
            t0 = time.time()
            out = []
            for rec in recs:
                f = featurize(rec["seq"], table, fast=args.fast,
                              protein=rec["protein"])
                f["seq_id"] = "%s_lam%s" % (name, rec["lam"])
                f["source"] = "lineardesign"
                f["generator"] = "lineardesign"
                f["protein"] = name
                f["lam"] = rec["lam"]
                f["protein_len"] = len(rec["protein"])
                if rec["mfe_reported"] is not None:
                    f["mfe_reported"] = rec["mfe_reported"]
                    f["delta_mfe"] = f["mfe"] - rec["mfe_reported"]
                if rec["cai_reported"] is not None:
                    f["cai_reported"] = rec["cai_reported"]
                    f["delta_cai"] = f["cai"] - rec["cai_reported"]
                out.append(f)
            write_tsv(out, os.path.join(
                args.out_dir, "ld_%s_features.tsv" % name))
            all_recs.extend(out)
            print("[featurize-all] %s: %d seqs in %.1fs" % (
                name, len(out), time.time() - t0))
        if all_recs:
            write_tsv(all_recs, os.path.join(
                args.out_dir, "ld_benchmark_features.tsv"))

    if args.gemorna:
        recs = parse_gemorna_output(args.gemorna)
        if not recs:
            print("[featurize-all] WARN: GEMORNA parse 0 records",
                  file=sys.stderr)
        else:
            # group by protein length (unique per benchmark protein),
            # featurize + WRITE INCREMENTALLY so a kill mid-run keeps
            # completed proteins on disk
            order = []
            grouped = {}
            for rec in recs:
                key = len(rec["protein"])
                if key not in grouped:
                    grouped[key] = []
                    order.append(key)
                grouped[key].append(rec)
            all_ge = []
            for plen in order:
                t0 = time.time()
                rows = []
                for rec in grouped[plen]:
                    f = featurize(rec["seq"], table, fast=args.fast,
                                  protein=rec["protein"])
                    f["seq_id"] = "gemorna_rep%s" % rec["rep"]
                    f["source"] = "gemorna"
                    f["generator"] = "gemorna"
                    f["rep"] = rec["rep"]
                    f["naturalness"] = rec["naturalness"]
                    f["protein"] = "len%d" % plen
                    f["lam"] = ""
                    f["protein_len"] = plen
                    rows.append(f)
                write_tsv(rows, os.path.join(
                    args.out_dir, "gemorna_len%d_features.tsv" % plen))
                print("[featurize-all] gemorna len%d (%daa): %d seqs in "
                      "%.1fs" % (plen, plen, len(rows), time.time() - t0),
                      flush=True)
                all_ge.extend(rows)
            write_tsv(all_ge, os.path.join(
                args.out_dir, "gemorna_features.tsv"))
            all_recs.extend(all_ge)

    if all_recs:
        write_tsv(all_recs, os.path.join(
            args.out_dir, "all_candidates_features.tsv"))


if __name__ == "__main__":
    main()
