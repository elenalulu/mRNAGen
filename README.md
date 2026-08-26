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
              - manufacturing penalties (restriction sites / hp4 / hp5 / GC)
              => composite v2, externally validated (S3, mean Spearman ~0.63)

UTR axis        N1-methylpseudouridine-conditioned MPRA model
              (366k 50-nt 5'UTRs, held-out Spearman 0.706)

Design loop     independent start pool (DNA Chisel MIT + Ensembl native
              CDS + our CAI-max) -> greedy synonymous refinement
              (composite v2) -> surgical self-complementarity repair
              -> diversity Top-K deliverable
```

The start pool is built entirely from fully-open sources — an
MIT-licensed sequence optimizer, open genome data, and our own
generator — so all deliverables are commercially redistributable under
Apache-2.0.

## Repo layout

```
*.py                     pipeline scripts (see docs/architecture.md)
feature_pipeline/        features: codon metrics, structure (ViennaRNA), rules
models/                  trained weights (composite v2 stats, oracle v0, UTR GBDT)
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

# 2. score a CDS (composite v2)
python - <<'PY'
import sys; sys.path.insert(0, ".")
from refine_t5 import load_env, composite
table, codon_coef, syn, models = load_env()
seq = "ATGGAA..."   # your CDS (DNA, multiple of 3)
print("composite v2 = %.2f" % composite(seq, table, codon_coef, models)[0])
PY

# 3. independent deliverable pipeline (fully independent start sources)
python start_pool.py              # 3-source start pool (dnachisel/native/cai)
python pipeline_independent.py    # refine (composite v2) + selfcomp repair
python select_deliverable.py      # diversity Top-K deliverable
```

## External validation (honest boundaries)

- S3 (GEMORNA Science SI): naturalness is the strongest same-protein
  predictor (Spearman 0.47–0.91; CAI 0.38–0.80; MFE mostly negative).
  **S3 was used as a development set — the blind test is wet-lab.**
- RNA-FM fine-tuning and the learned oracle v0 head were tried and
  **closed as negative results** (see docs/).
- composite v2 = design objective (externally validated components), not
  an absolute expression prediction.

## License

Apache-2.0 (see LICENSE). Third-party data/upstream terms: see NOTICE.

## Citation

See CITATION.cff. If you use this in research, please cite the project and
the upstream works listed in NOTICE (GEMORNA, LinearDesign, Wu 2019,
Sample 2019).
