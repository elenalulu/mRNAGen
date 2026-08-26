# GEMORNA vs LinearDesign 同蛋白对比报告 v0

日期：2026-08-23 ｜ 蛋白：concatemer 64aa（两家同款）｜ 特征：T3 管线 41 维

数据：GEMORNA 20 reps（18 unique，REPS=20）vs LinearDesign 5 λ（0.0/0.3/1.0/1.5/2.5）
环境注意：GEMORNA 经 vocab_compat 修复后在 torch 2.10 下正常出序列。

## 头条发现

**1. 设计哲学分裂——两家优化的是相反的目标**
LinearDesign 走"深度折叠"（MFE/nt = -0.83，起始区 85% 配对关闭），
GEMORNA 走"开放可译"（MFE/nt -0.30~-0.54，起始区开放度 0.20-0.54）。
在 LinearDesign 自己的 MFE 轴上 GEMORNA 惨败；在翻译可及性轴上 LD 惨败。
两家都没有显式优化对方的轴——正是 oracle-first 的论据。

**2. MDA5 类结构风险：GEMORNA 碾压**
最长近完美自互补双链：GEMORNA 20/20 全部 = 0 bp；LD 5/5 = 26 bp。
MFE 结构最长茎：GEMORNA 6-12 bp vs LD 18 bp。
（ψ 设定下 MDA5 轴是免疫 head 主特征，此轴差异直接进 oracle。）

**3. CAI：GEMORNA 覆盖并超越 LD 全 λ 谱**
LD：0.784-0.826（λ 单调）；GEMORNA：0.741-0.968，均值 0.889。
GEMORNA 的 naturalness 与 CAI 相关 r=+0.84，与 MFE r=-0.80。

**4. 制造规则两家都不管（oracle 增值空间）**
同聚物规则（从严阈值 run≥4 fail）：LD 50/50 违反（11 蛋白全 benchmark）、
GEMORNA 19/20 违反。酶切位点：LD 43/50 含 ≥1 个位点、GEMORNA 3/20。
GC：GEMORNA 偏高（均值 0.725，逼近 0.70 上限）——高 CAI 的代价。

**5. 组合策略验证：GEMORNA 池内选择有大幅空间**
最优 rep7：CAI 0.968 + open_start45 0.485 + selfcomp 0 + 零酶切位点，
全轴优于 LD（除 MFE 轴）；最差 rep 仅 CAI 0.741。
→ "GEMORNA 生成 → 本项目 oracle 筛选" 的分布基础成立。

## 附：benchmark 现状

- LinearDesign：11/14 蛋白完成（50 候选）。factor_viii / gaa_pompe / spy_cas9
  三个大蛋白 GPU 侧二进制**空产出**（文件只有 @@LAMBDA 头、零 mRNA sequence 行，
  错误信息在 GPU 控制台 stderr，未进文件），待诊断重跑。
- GEMORNA：仅 concatemer。benchmark_oneline.txt（14 蛋白单行格式）已备好，
  GPU 机 REPS=20 一次跑完（280 生成，CPU 每个秒级）。

## 产出文件

- `mRNAGen/data/all_candidates_features.tsv`（70 行 × 45 列，含 generator 列）
- `mRNAGen/data/ld_benchmark_features.tsv`（LD 11 蛋白 × 5λ）
- `mRNAGen/data/gemorna_features.tsv`（20 reps，含 naturalness）
- 逐蛋白 TSV：`mRNAGen/data/ld_<protein>_features.tsv`
