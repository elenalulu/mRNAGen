# mRNAGen — open-source mRNA CDS + UTR design engine

A fully open, commercially-usable mRNA sequence design engine. Core claim:
the ranking components were **externally validated on wet-lab data**
(GEMORNA Science 2025 SI), and the manufacturing-compliance layer is a
deterministic improvement — this is a *design objective*, not an absolute
expression predictor.

## What it does

```
CDS axis        naturalness (codon-pair, rebuilt from 18,963 human CDS)
              + CAI (human codon-usage)
              - manufacturing penalties (restriction sites / homopolymers / GC)
              => design score, externally validated (S3, mean Spearman ~0.63)

UTR axis        N1-methylpseudouridine-conditioned MPRA model
              (366k 50-nt 5'UTRs, held-out Spearman 0.706)

Design loop     independent start pool (DNA Chisel MIT + Ensembl native
              CDS + our CAI-max) -> greedy synonymous refinement
              (design score) -> surgical self-complementarity repair
              -> diversity Top-K deliverable
```

The start pool is built entirely from fully-open sources — an
MIT-licensed sequence optimizer, open genome data, and our own
generator — so all deliverables are commercially redistributable under
Apache-2.0.

## Comparison with published baselines

Benchmark candidates from GEMORNA (Science 2025) and LinearDesign
(Nature 2023) were scored with our design score metrics on the same
14 protein targets; the mRNAGen column is our independent deliverable
(35 candidates). Higher z_nat / CAI is better; lower dsRNA risk is better
(longest near-perfect self-duplex, bp); "clean" = 0 restriction sites and
no homopolymer run >= 5 nt.

| Aspect | mRNAGen (this work) | GEMORNA | LinearDesign |
|---|---|---|---|
| Design objective | naturalness + CAI - manufacturing penalties (design score) | codon-pair naturalness + secondary structure | CAI + MFE (lambda-tunable) |
| 5'UTR model | N1-psi-conditioned MPRA, 366k 5'UTRs, held-out rho = 0.706 | — | — |
| Start lineage | fully open (DNA Chisel MIT + Ensembl + own CAI-max) | tool-generated | lambda-scan |
| naturalness z (achieved) | **2.76** | 1.35 | 0.31 |
| CAI / codon-usage preference (achieved) | **0.95** | 0.84 | 0.82 |
| GC content (achieved) | **0.61** | 0.55 | 0.57 |
| Manufacturing clean (0 sites, no homopolymers) | **91%** | 0% | 0% |
| dsRNA risk (longest duplex, bp) | **8.9** | 9.6 | 23.4 |
| MFE stability (kcal/nt) | n/a\* | -0.35 | -0.64 |
| Ranking validation | external wet-lab (S3): naturalness rho 0.47-0.91 | own wet-lab (Science 2025) | in-vitro / in-vivo expression |
| License | Apache-2.0 (commercial use OK) | non-commercial research only | redistribution requires permission |

\* MFE is LinearDesign's own optimization objective; we deliberately do
not optimize it (see honest boundaries below). All scores above are
computed with our metric stack on each engine's benchmark candidates.
tRNA adaptation (tAI) is not evaluated — CAI serves as the codon-usage
preference proxy; computing tAI would require tRNA copy-number data.

## Repo layout

```
*.py                     pipeline scripts (see docs/architecture.md)
feature_pipeline/        features: codon metrics, structure (ViennaRNA), rules
models/                  trained weights (scoring z-stats, UTR GBDT, legacy learned models)
data/proteins/           example target proteins (public sequences)
data/                    large/upstream data is NOT bundled — see fetch script
scripts/fetch_upstream_data.sh   download data + benchmark baselines
docs/                    reports
```

## Quick start

```bash
# 1. environment (ViennaRNA is the only non-pip system dep)
conda create -n mrna python=3.10
conda activate mrna
conda install -c conda-forge viennarna
pip install numpy pandas scikit-learn joblib
pip install dnachisel          # MIT start generator

# 2. score a CDS (design score)
python - <<'PY'
import sys; sys.path.insert(0, ".")
from refine_t5 import load_env, composite
table, codon_coef, syn, models = load_env()
seq = "ATGGAA..."   # your CDS (DNA, multiple of 3)
print("design score = %.2f" % composite(seq, table, codon_coef, models)[0])
PY

# 3. independent deliverable pipeline (fully independent start sources)
python start_pool.py              # 3-source start pool (dnachisel/native/cai)
python pipeline_independent.py    # refine (design score) + self-complementarity repair
python select_deliverable.py      # diversity Top-K deliverable
```

## External validation (honest boundaries)

- S3 (GEMORNA Science SI): naturalness is the strongest same-protein
  predictor (Spearman 0.47–0.91; CAI 0.38–0.80; MFE mostly negative).
  **S3 was used as a development set — the blind test is wet-lab.**
- RNA-FM fine-tuning and a learned expression-prediction head were tried
  and **closed as negative results** (see docs/).
- design score = design objective (externally validated components), not
  an absolute expression prediction.

## License

Apache-2.0 (see LICENSE). Third-party data/upstream terms: see NOTICE.

## Citation

See CITATION.cff. If you use this in research, please cite the project and
the upstream works listed in NOTICE (GEMORNA, LinearDesign, Wu 2019,
Sample 2019).
