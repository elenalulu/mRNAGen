# 对比报告 v1 — GEMORNA ×280 vs LinearDesign ×50（全 benchmark）

日期：2026-08-24 ｜ 数据：330 行 × 45 列特征矩阵（`all_candidates_features.tsv`）
覆盖：GEMORNA 14 蛋白 × 20 reps；LD 10 蛋白 × 5 λ（4 个最长蛋白 LD OOM）

## v0 结论修正（先还科学债）

concatemer 上的 "GEMORNA selfcomp = 0 vs LD 26bp" **是 64aa 短序列特例**（短序列里
碰巧没有 ≥10bp 的自互补种子）。真实蛋白上的正确结论：

- **GEMORNA 的自互补双链长度稳定约为 LD 的一半**（10 蛋白均值 9.7 vs 21.9 bp），
  不是归零——优势真实存在且全长度成立，但幅度要诚实
- LD 起始区开放度并非都是 concatemer 那种 0.15 极端值——10 蛋白上 0.37–0.70，
  GEMORNA 对比增益缩到 ~1.07×；concatemer 是离群点

## v1 核心结论

**1. "开放可译" 是 GEMORNA 的普适设计规律，全长度保持**
MFE/nt：LD −0.61~−0.75 vs GEMORNA −0.26~−0.48，193aa 到 2351aa 无一例外约 2× 更开放。
open_start45 在全部 14 个蛋白（含 7053nt factor_viii 的 0.486）保持 0.43–0.63。

**2. MDA5 结构风险轴：GEMORNA 系统性 ~50% 优势（见配对图）**
10/10 蛋白上 GEMORNA 自互补双链更短；均值 9.7 vs 21.9 bp。
egfp 最悬殊（1.3 vs 22.6），rsv_f 最接近（14.4 vs 21.4）。

**3. CAI：GEMORNA 8/10 蛋白 ≥ LD**（例外 flu_ha_pr8、rsv_f）。
短蛋白上 GEMORNA CAI 均值 0.84–0.90，覆盖 LD 全 λ 谱。

**4. 制造约束两家都不管——且随长度恶化**
- 同聚物（严格阈值）：两家 100% 违反 → 该规则应改为连续特征（hp_max）参与排序，
  而非硬门槛
- 酶切位点：GEMORNA 均值随长度线性涨（epo 0.2 个/条 → factor_viii **10.6 个/条**）；
  LD 同样不管。长序列下这是硬伤，oracle 筛选的主战场
- GC：GEMORNA 因蛋白而异（0.41–0.70），非 v0 以为的恒定偏高

**5. 工程可扩展性差异本身是 benchmark 发现**
同机同批：LD（精确 DP）在 ≥952aa 全部 OOM（单跑也死，exit 137）；
GEMORNA（17MB transformer）7053nt 轻松完整生成。精确搜索 vs 生成近似的
代价结构完全不同。

## 对 oracle 设计的直接输入

1. selfcomp / openness / MFE 三轴上两个生成器给出**系统性不同的候选分布** →
   多生成器候选池（LD + GEMORNA + 未来自训）喂同一个 oracle 是对的
2. 制造规则从"硬门槛"降级为"连续特征 + 软惩罚"，阈值按长度归一
3. naturalness（GEMORNA 自报）与 CAI r=+0.84、与 MFE r=−0.80 → 可作为
   oracle expression head 的免费弱监督信号源之一（self-play 隔离规则不适用，
   它是生成器自评分，非外部标签）

## 下一步（关键路径不变）

- **T2 监督数据收集**（高表达基因 CDS + decay 数据集 + 文献案例）→ T4 GBDT oracle
- 330 行矩阵已是 T4 特征表雏形；加标签列即可训练排序
- LD 缺的 4 蛋白：加 swap 重跑或接受现状（非阻塞）
