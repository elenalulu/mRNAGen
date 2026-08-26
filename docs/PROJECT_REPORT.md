# mRNAGen 项目报告 v1.0
**AI-native mRNA Design Engine —— 三天冲刺总结（2026-08-23 至 08-25）**

## 一句话结论

以外部实证的 naturalness+CAI 排序、修饰条件（N1-ψ）UTR 打分器、制造合规硬约束、多样性交付为核心，建成了全开源、可商用、每一层有验证依据的 mRNA CDS+UTR 设计引擎；下一步唯一的关键杠杆是湿实验验证数据。

---

## 1. 交付物清单

| 资产 | 位置 | 验证状态 |
|---|---|---|
| CDS 设计引擎（composite v2） | `mRNAGen/refine_t5.py` + `naturalness.py` | S3 外部验证（见 §3） |
| Top-K v3 交付集（14 蛋白 × 5） | `data/t5/topk_selection.tsv` | 70/70 翻译正确 + 唯一 |
| m1ψ UTR 打分器（held-out ρ=0.706） | `models/utr_m1psi_gbdt.joblib` | MPRA 36.6 万条训练 + designed 独立库 0.73 |
| UTR 短名单（10 条实测高 MRL） | `data/t3_utr/utr_shortlist.tsv` | 实测 m1ψ MRL 7.8-9.2，汉明距离 ≥36/50 |
| 全 mRNA 组合排名（700 组合） | `data/t3_utr/utr4_full_mrna_ranking.tsv` | 加性假设（v1 边界已标注） |
| 知名 UTR 对照打分表 | `data/t3_utr/utr_benchmark_scores.tsv` | 三重外部一致性（见 §4） |
| CRO 送测包 v1.1（19 构造 + 预注册） | `data/t4_cro/` | 同蛋白设计，条件镜像 GEMORNA Science |
| GEMORNA 对标分析 | `data/t3_gemorna_si/s2_analysis.tsv` | CAI 复验 ρ=1.000 |

## 2. 方法栈（自下而上）

```
数据层    18,963 天然人 CDS（Ensembl+PaxDb）│ Sample'19 MPRA 326k×6 条件
          （含 m1ψ）│ GEMORNA Science S1-S4（67 条同蛋白湿实验标签）
特征层    naturalness（密码子对频率，开源重建）│ CAI（逐行复刻 LinearDesign
          口径，ρ=1.000 复验）│ ViennaRNA 结构 │ 制造规则（酶切位点/同聚物/GC）
模型层    composite v2 = z(nat)+z(CAI)−合规惩罚  ← 外部验证的设计目标
          m1ψ UTR GBDT（丰度过滤 top-50%，ρ=0.706，天花板 ~0.88）
          oracle v0（GBDT，仅作诊断——蛋白内不迁移，见 §5）
搜索层    贪心同义替换（r25，批量预测 8× 加速）│ Top-K max-min 多样性选择
          （α·z(score)+(1−α)·min_dist）
```

## 3. 关键科学结论（按证据强度）

### 3.1 外部实证（最强）
- **naturalness 是同蛋白密码子变异表达的最强预测子**（GEMORNA S3 四个
  数据集 ρ=0.47-0.91）；我们从公开数据重建的版本与其打平/局部反超
  （NanoLuc 0.544 vs 0.468）。
- **GEMORNA 的机制透视**（S2，3,765 蛋白）：其生成模型 z_nat +2.82，
  本质是 naturalness 最大化器——比 CAI 优化器优化 CAI（+2.05）更极端。
- **CAI 在蛋白内排序有真实信号**（0.38-0.80），但基因级对蛋白丰度零预测
  力（r=0.019）——LinearDesign 代理目标的适用边界被精确刻画。

### 3.2 负结果（同样重要，防止后续浪费）
- **基因级 oracle 头不可迁移**：S3 上 composite v1 仅 0.28/-0.01/-0.83，
  远低于 CAI（0.80/0.38/0.67）——跨基因监督 ≠ 蛋白内设计能力。
- **RNA-FM（99.5M）端到端微调全面劣于 11 特征 GBDT**（decay 0.23-0.25
  vs 0.30-0.37；frozen embedding 零增益）——CDS 序列对 decay/expr 的
  可提取信息枯竭，剩余方差在 UTR/RBP/细胞状态。
- **修饰显著改变 UTR 排序**（unmod~m1ψ 仅 0.61）但两种 ψ 几乎等价
  （0.945）——UTR 模块必须用修饰条件训练（我们直接持有 m1ψ 标签）。

### 3.3 工程结论
- 丰度过滤是 MPRA 标签质量的钥匙（top-50% 丰度 replicate 一致性
  0.627→0.809；m1ψ 模型 0.51→0.71）。
- 硬约束贪心会卡死（起点违规→全部 move 被拒）→ 全软惩罚设计。
- sklearn 单样本 predict ×5 头是热点 → 批量预测 8× 加速。
- MRL 库丰度列 = 相对丰度非绝对读数；59nt 库需 `[:50]` 对齐。

### 3.4 自互补修复（surgical repair，2026-08-25）
- **诊断**：selfcomp 与 MFE 强耦合（池内 ρ=−0.74，同一分子内配对现象），
  与 CAI/设计分/5′开放弱耦合（|ρ|≤0.22）→ 可在不伤排序核心的前提下降
  dsRNA 风险。vs GEMORNA 9.3 bp 的差距判定为"防御纵深"，MFE 轴对
  LinearDesign 维持 NO-GO（S3 上 MFE 是最差预测子）。
- **方法**（`repair_selfcomp.py`）：贪心同义密码子替换，仅限最长
  near-duplex 两臂；门控 = selfcomp 严格下降 + composite ≥−0.10 +
  Δz_nat ≥−0.05 + ΔCAI ≥−0.005 + sites/hp5 不增 + GC 窗内。氨基酸
  序列零改变（回译断言）。S3 证据按构造保留。
- **结果**（Top-K v3，`select_topk.py` 含 selfcomp tiebreak + mfe_per_nt
  归一化修正）：精修候选 selfcomp **18.6→13.8 bp**（max 29→21），
  设计分 +0.15（门控下不降反升），CAI/开放度/制造合规全保持
  （clean 71%）。距 GEMORNA 9.3 仍差 1.5×——进一步收敛需 Plan A
  （目标函数惩罚项，已评估暂缓）。
- 顺带修复：`select_topk.py` mfe_per_nt 除以密码子数（3× 虚高）的
  归一化 bug；修正后精修候选真实 mfe ≈ −0.47/nt（LD −0.65，GEMORNA
  −0.36），三引擎稳定性排序与 selfcomp 排序完全一致（同一旋钮）。

## 4. 知名 UTR 锚点（产品化叙事用）

m1ψ 模型排名：NCA-7d (99.9%) > GMR-5U10 > **Sample'19 最优设计** >
PABPv3 > hHBB (89.8%) > **mRNA-1273 (81.2%) > BNT162b2 (72.4%)**。
我们的实测短名单（7.8-9.2）高于全部知名 UTR 预测分。

## 5. 诚实边界（对外表述纪律）

1. composite v2 的排序依据（nat+CAI）在 S3 验证均值 0.63，非完美；
   S3 已用作 dev set，新设计的盲验证依赖湿实验。
2. UTR×CDS 交互（scanning 偶联）未建模；组合分为加性假设。
3. UTR 模型 50nt 定长，非 50nt UTR 打分为方向性。
4. Top-K 的制造合规（sites 153→0）是确定性改进（非概率预测），
   这是对外最可防守的宣称。
5. 无任何自有湿实验数据；本报告所有"外部验证"指公共数据交叉。

## 6. 状态与下一步

| 项 | 状态 |
|---|---|
| RL（Phase 3） | NO-GO，条件触发（真实验证数据 + decay>0.5） |
| RNA-FM 微调 | 关闭（负结果定论） |
| UTR 生成 | 缓（触发 = 第一轮湿实验数据） |
| CRO 送测包 | 就绪待发（`data/t4_cro/`，一块 96 孔板，几万元级） |
| 可选打磨 | UTR v2 ensemble（0.71→~0.75，天花板 0.88）边际价值低 |

**结论：干实验侧到达干净里程碑。项目价值解锁的钥匙 = 验证数据
（CRO 荧光素酶筛选 / 合作方通道），送测包已备好，随时可发。**
