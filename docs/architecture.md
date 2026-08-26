# Architecture

## Objective

`design score = z_global(naturalness) + z_global(CAI) - manufacturing
penalties`

- naturalness: codon-pair log-likelihood vs 18,963 endogenous human CDS
  (rebuilt open-source; strongest same-protein expression predictor on
  GEMORNA Science 2025 SI, Spearman 0.47-0.91)
- CAI: human codon-usage (Kazusa-derived table in this repo)
- penalties: 0.5 x restriction sites + 0.1 x hp4 + 0.8 x hp5
  + 5 x GC-out-of-window

Ranking components were externally validated; composite is a **design
objective**, not an absolute expression predictor. S3 was used as a
development set — the blind test is wet-lab.

## Independent start pool (no GEMORNA / LinearDesign lineage)

```
start_pool.py
  dnachisel  DNA Chisel (MIT): manufacturing-clean start
             (restriction sites avoided, GC windows, translation preserved,
             human CAI objective); seed = our CAI-max sequence
  native     Ensembl GRCh38 endogenous CDS (open data), human targets only
             (exact translation match required)
  cai        CAI-max start from the Kazusa table (our own generator)

pipeline_independent.py
  every start -> greedy refinement (design score, 25 rounds)
              -> surgical self-comp repair (Plan C gates)

select_deliverable.py
  per protein -> greedy max-min diversity Top-K
  (7-dim feature space, no ViennaRNA required)
```

## Key scripts

| Script | Role |
|---|---|
| `refine_t5.py` | load_env / composite / greedy refinement (core scoring) |
| `naturalness.py` | codon-pair naturalness scorer + corpus z-stats |
| `repair_selfcomp.py` | surgical dsRNA-risk repair (gates: score/z_nat/CAI/sites/hp5/GC) |
| `select_deliverable.py` | independent-pool diversity Top-K |
| `train_oracle_v0.py` / `train_utr_*.py` | model training (public data) |
| `validate_s3.py` | external validation on GEMORNA SI S3 (dev set) |

## Honest boundaries

- S3 used as dev set; no wet-lab data of our own yet
- UTR x CDS interaction not modeled (additive scoring)
- non-50nt UTRs scored directionally
- manufacturing compliance is a deterministic improvement, not a prediction
