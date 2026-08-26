#!/usr/bin/env bash
# Download upstream data used by mRNAGen for training / validation /
# benchmarks. These are NOT redistributed in the repo (see NOTICE).
#
# Usage: bash scripts/fetch_upstream_data.sh [--dry-run]
# Run from the repo root. Each step is independent; skip what you do not need.

set -euo pipefail
DRY="${1:-}"
say() { echo -e "\n[fetch] $*"; }

say "1) Ensembl GRCh38 CDS (naturalness corpus, ~30MB)"
say "   http://ftp.ensembl.org/pub/release-111/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz"
say "   -> data/t2/ ; then run: python build_t2_endogenous.py"

say "2) Wu et al. 2019 eLife (CC-BY 4.0) per-codon coefficients"
say "   https://doi.org/10.7554/eLife.45396  -> data/t2/wu2019_elifescience/elife-45396-fig1-data2.csv"

say "3) Sample et al. 2019 Nat Biotech 5'UTR MPRA (GEO GSE114002)"
say "   https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002"
say "   -> data/t3_utr/ ; then run: python build_utr_table.py"

say "4) GEMORNA Science 2025 SI (external validation S1-S4, benchmark)"
say "   https://www.science.org/doi/10.1126/science.adr8470"
say "   -> data/t3_gemorna_si/ (research use only; see NOTICE)"

say "5) GEMORNA tool (benchmark baseline; NON-COMMERCIAL license)"
say "   https://github.com/rainabio/GEMORNA  -> ../github/GEMORNA"
say "   NOTE: outputs of this tool are NOT redistributable; use only for research benchmarking."

say "6) LinearDesign tool (benchmark baseline; redistribution requires permission)"
say "   https://github.com/LinearDesignSoftware/LinearDesign -> ../github/LinearDesign-main"

say "Done. Large data files are ignored by git (see .gitignore)."
