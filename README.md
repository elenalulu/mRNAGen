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
              (36.6万 50-nt 5'UTRs, held-out Spearman 0.706)

Design loop     greedy synonymous refinement -> diversity Top-K selection
              -> surgical self-complementarity repair (dsRNA-risk aware)
```

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

# 2. score a CDS (composite v2)
python - <<'PY'
import sys; sys.path.insert(0, ".")
from refine_t5 import load_env, composite
table, codon_coef, syn, models = load_env()
seq = "ATGGAA..."   # your CDS (DNA, multiple of 3)
print("composite v2 = %.2f" % composite(seq, table, codon_coef, models)[0])
PY

# 3. full pipeline from any start sequence
python refine_t5.py            # greedy refinement (reads github/LinearDesign-main/ld_outputs if present)
python select_topk.py          # diversity Top-K over pools
python repair_selfcomp.py      # surgical dsRNA-risk repair (Plan C)
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
