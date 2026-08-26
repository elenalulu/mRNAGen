<a id="en"></a>

# mRNAGen — open-source mRNA CDS + UTR design engine

<div align="center">

**English** · [**中文版**](#zh)

</div>

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

---

<a id="zh"></a>

# mRNAGen — 开源 mRNA CDS + UTR 设计引擎

<div align="center">

[**English**](#en) · **中文版**

</div>

一个完全开源、可商用的 mRNA 序列设计引擎。核心主张：排序组件在
湿实验数据上经过**外部验证**（GEMORNA Science 2025 SI），制造合规层是
确定性改进——这是*设计目标*，不是绝对表达量预测。

## 功能概览

```
CDS 轴         naturalness（密码子对，基于 18,963 条人源 CDS 重建）
              + CAI（人源密码子使用）
              - 制造合规惩罚（酶切位点 / hp4 / hp5 / GC）
              => composite v2，外部验证（S3，均值 Spearman ≈ 0.63）

UTR 轴         N1-甲基假尿苷（N1-ψ）修饰条件 MPRA 模型
              （36.6 万条 50nt 5'UTR，held-out Spearman 0.706）

设计闭环       独立起点池（DNA Chisel MIT + Ensembl 天然 CDS + 自有 CAI-max）
              -> 贪心同义密码子精修（composite v2）
              -> 外科式自互补修复
              -> 多样性 Top-K 交付集
```

起点池完全由全开源来源构建——MIT 许可的序列优化器、开放基因组数据与
我们自己的生成器——因此全部交付物可在 Apache-2.0 下商用再分发。

## 仓库结构

```
*.py                     管线脚本（见 docs/architecture.md）
feature_pipeline/        特征模块：密码子指标、结构（ViennaRNA）、规则
models/                  训练权重（composite v2 统计、oracle v0、UTR GBDT）
data/proteins/           示例靶蛋白（公共序列）
data/                    大体积/上游数据不入库——见 fetch 脚本
scripts/fetch_upstream_data.sh   下载数据与基准基线
docs/                    报告
```

## 快速开始

```bash
# 1. 环境（ViennaRNA 是唯一的非 pip 系统依赖）
conda create -n mrna python=3.10
conda activate mrna
conda install -c conda-forge viennarna
pip install numpy pandas scikit-learn joblib
pip install dnachisel          # MIT 许可的起点生成器

# 2. 打分一个 CDS（composite v2）
python - <<'PY'
import sys; sys.path.insert(0, ".")
from refine_t5 import load_env, composite
table, codon_coef, syn, models = load_env()
seq = "ATGGAA..."   # 你的 CDS（DNA，长度需为 3 的倍数）
print("composite v2 = %.2f" % composite(seq, table, codon_coef, models)[0])
PY

# 3. 独立交付管线（完全独立的起点来源）
python start_pool.py              # 3 源起点池（dnachisel / native / cai）
python pipeline_independent.py    # 精修（composite v2）+ 自互补修复
python select_deliverable.py      # 多样性 Top-K 交付集
```

## 外部验证（诚实边界）

- S3（GEMORNA Science SI）：naturalness 是同蛋白密码子变异表达的最强
  预测子（Spearman 0.47–0.91；CAI 0.38–0.80；MFE 大多为负）。
  **S3 已用作开发集——盲测依赖湿实验。**
- RNA-FM 微调与 learned oracle v0 head 均已尝试，并**以负结果关闭**
  （见 docs/）。
- composite v2 = 设计目标（组件经外部验证），非绝对表达量预测。

## 许可证

Apache-2.0（见 LICENSE）。第三方数据/上游条款：见 NOTICE。

## 引用

见 CITATION.cff。研究中使用本项目时，请同时引用本项目及 NOTICE 中
列出的上游工作（GEMORNA、LinearDesign、Wu 2019、Sample 2019）。
