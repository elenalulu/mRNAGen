# T3 特征管线（mRNA Design Engine / Oracle 特征层）

CDS 序列 → 41 维特征 TSV。输入支持 FASTA 与 LinearDesign λ-grid 输出（T1 产物）双格式，
输出直接喂 Oracle v0（GBDT 排序）。

## 运行（必须用 python——ViennaRNA 在里面）

```bash
# LinearDesign T1 输出（scp 回来的 lineardesign_output.txt）
python feature_pipeline/featurize.py \
    --in github/LinearDesign-main/lineardesign_output.txt \
    --out feature_pipeline/data/ld_baseline_features.tsv --ld-check

# 普通 FASTA（候选池）
python feature_pipeline/featurize.py --in candidates.fasta --out features.tsv

# 单元 + 集成测试（17 项）
python feature_pipeline/tests/test_featurize.py
```

`--fast` 跳过 partition function 与自互补扫描（调试用）；`--ld-check` 打印与
LinearDesign 自报值的交叉验证表。

## 特征分组

| 组 | 列 | 说明 |
|---|---|---|
| meta | seq_id / source / lam / protein_len / stop_codon / translate_ok | translate_ok = 密码子回译与蛋白一致 |
| 序列 | seq_len, gc_global, gc_slide60_max/min, upa_odds, cpg_odds, hp_max_[ACGU], urich6_count | UpA↔稳定性、CpG↔免疫（争议，交给数据） |
| 密码子 | cai, cai_excl_stop, gc3, enc | **CAI 逐行复刻 LinearDesign C++ 口径**（全密码子含起始/终止） |
| 结构 | mfe, mfe_per_nt, paired_frac_mfe, longest_helix_mfe, n_helix_ge8, ensemble_ok, mean_unpaired(_q25), open_start45, selfcomp_max_exact, selfcomp_max_near | MFE 仅作背景变量；主角 = 自互补双链（MDA5）与起始区 openness |
| 规则 | rule_gc_global/slide/homopolymer_pass, restriction_site_count, cryptic_donor_count, rules_all_pass | 酶切位点：BsaI/BsmBI/EcoRI/XhoI/BamHI/NotI |
| LD 对照 | mfe_reported, cai_reported, delta_mfe, delta_cai | 仅 LinearDesign 输入时填充 |

结构特征口径（plan §2.3 修正版）：
- `selfcomp_max_near`：种子扩展（10bp 精确种子 + 共享 mismatch 预算≤2）找到的最长
  分子内近完美双链——MDA5 类风险主特征，**LinearDesign 的 CAI+MFE 完全无视它**
- `open_start45`：起始密码子下游 45nt 平均 unpaired 概率（150nt 局部 partition
  function；CDS-only 设定无 UTR 上下文）
- 长序列（>3000nt）：全局 pf 降级为窗口采样（500nt 步长 / 200nt 窗）

## 验证状态（2026-08-23）

- 17/17 测试通过
- CAI 黄金校验：README 例子 MNDTEAI → 0.695（实测 0.6947）；对 T1 全部 5 条 λ 记录
  worst |ΔCAI| = 0.00005（打印精度内完全一致）
- MFE：**结构逐碱基一致**（5/5 λ），能量差恒定 -3.10 kcal/mol——LinearDesign 自带
  旧版 ViennaRNA .so 与本机新版参数的版本偏移，非计算错误；Oracle 内部统一用本机
  口径，只保证候选间可比
- 实战信号：λ=1.0/1.5 候选违反同聚物规则（rules_all_pass=0），λ=0 起始区 85%
  配对关闭（open_start45=0.15）——两类 LinearDesign 不看的缺陷都被抓到

## 环境铁律

- 只读使用本机已装 ViennaRNA 的环境（conda env），**绝不 pip install 进基环境**
- 新依赖一律装隔离 venv，经子进程调用本管线产物（TSV），与 mhcflurry 桥接同模式

## 下游（T4 衔接）

`ld_baseline_features.tsv` 即 GBDT 特征矩阵雏形；T2（监督标签）到位后，
`features/` 里的结构/序列/密码子模块直接复用，训练脚本读 TSV 即可。
