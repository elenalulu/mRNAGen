#!/usr/bin/env python
"""Assemble the CRO submission package v1 (Fluc expression validation).

Assay design (mirrors GEMORNA Science 'This study Fluc' for
comparability): Fluc mRNA with N1-psi modification, transfect HEK293T,
24h luciferase readout.

Backbone: fixed 5'UTR = BNT162b2, 3'UTR = BNT162b2 (industry anchor, both
from GEMORNA S1 -- sequences verified from their SI).

Arms:
  CDS arm (fixed UTRs): our Top-K v3 fluc refined (5) + S1 controls
    (pGL4.11 natural, LinearDesign, CAI-G, CAI-I, BiLSTM-CRF,
     GMR-FL1..4)  -> 14 constructs
  UTR arm (fixed our best CDS): shortlist top1/top2, BNT162b2,
    mRNA-1273, hHBB 5'UTRs  -> 5 constructs

Pre-registration discipline: predicted metrics + ranking are locked in
cro_package_v1_preregistration.md BEFORE any wet-lab measurement.

Outputs: data/t4_cro/cro_package_v1.tsv + preregistration MD
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "feature_pipeline"))
sys.path.insert(0, HERE)

from features.codon_metrics import get_default_table  # noqa: E402
from naturalness import Scorer  # noqa: E402
from refine_t5 import load_env, n_hp4, n_hp5, n_sites  # noqa: E402

SI = os.path.join(HERE, "data", "t3_gemorna_si",
                  "science.adr8470_data-s1.xlsx")
TOPK = os.path.join(HERE, "data", "t5", "topk_selection.tsv")
SHORT = os.path.join(HERE, "data", "t3_utr", "utr_shortlist.tsv")
OUTD = os.path.join(HERE, "data", "t4_cro")
OUT = os.path.join(OUTD, "cro_package_v1.tsv")
PREREG = os.path.join(OUTD, "cro_package_v1_preregistration.md")


def norm(s):
    return str(s).upper().replace("U", "T")


def main():
    os.makedirs(OUTD, exist_ok=True)
    table, codon_coef, syn, models = load_env()
    scorer = Scorer()
    gb = get_default_table()

    xl = pd.ExcelFile(SI)
    s1_cds = xl.parse("CDSs")
    s1_u5 = xl.parse("5'UTRs")
    s1_u3 = xl.parse("3'UTRs")

    def u5(name):
        return norm(s1_u5[s1_u5.Name == name].iloc[0].Sequence)

    def u3(name):
        return norm(s1_u3[s1_u3.Name == name].iloc[0].Sequence)

    anchor_u5 = u5("BNT162b2")
    anchor_u3 = u3("BNT162b2")

    def cds_stats(seq):
        s = norm(seq)
        if len(s) % 3 == 0 and s[-3:] in ("TAA", "TAG", "TGA"):
            s_core = s[:-3]
        else:
            s_core = s
        return {
            "z_nat": round(float(scorer.z_nat([s_core])[0]), 2),
            "cai": round(gb.cai(s_core.replace("T", "U")), 3),
            "gc": round((s_core.count("G") + s_core.count("C"))
                        / len(s_core), 3),
            "sites": n_sites(s_core), "hp4": n_hp4(s_core),
            "hp5": n_hp5(s_core)}

    # ---- CDS arm ----
    rows = []
    topk = pd.read_csv(TOPK, sep="\t")
    fl = topk[topk.protein == "fluc"].sort_values("rank")
    for _, r in fl.iterrows():
        st = cds_stats(r.seq)
        rows.append({"construct_id": "CDS-OURS-%d" % r["rank"],
                     "group": "ours", "cds_name": "topk_v3_fluc_r%d"
                     % r["rank"], "utr5_name": "BNT162b2",
                     "utr3_name": "BNT162b2", "cds": r.seq, **st})
    fl_lib = s1_cds[s1_cds.Protein.astype(str).str.contains("Firefly")]
    keep_controls = ["pGL4.11", "LinearDesign", "CAI-G", "CAI-I",
                     "BiLSTM-CRF", "GMR-FL1", "GMR-FL2", "GMR-FL3",
                     "GMR-FL4"]
    for _, r in fl_lib.iterrows():
        if r.Name not in keep_controls:
            continue
        st = cds_stats(r.Sequence)
        rows.append({"construct_id": "CDS-CTRL-%s" % r.Name,
                     "group": "control", "cds_name": str(r.Name),
                     "utr5_name": "BNT162b2", "utr3_name": "BNT162b2",
                     "cds": norm(r.Sequence), **st})

    # ---- UTR arm (fixed best our CDS) ----
    best_cds = fl.iloc[0].seq
    best_st = cds_stats(best_cds)
    short = pd.read_csv(SHORT, sep="\t").sort_values("rank")
    utr_variants = [("utr_top1", short.iloc[0].utr),
                    ("utr_top2", short.iloc[1].utr),
                    ("BNT162b2", anchor_u5),
                    ("mRNA-1273", u5("mRNA-1273")),
                    ("hHBB", u5("hHBB"))]
    for name, seq in utr_variants:
        rows.append({"construct_id": "UTR-%s" % name, "group":
                     "utr_variant", "cds_name": "topk_v3_fluc_r1",
                     "utr5_name": name, "utr3_name": "BNT162b2",
                     "cds": best_cds, "utr5_seq": seq, **best_st})

    df = pd.DataFrame(rows)
    # fill UTR5 sequences (anchor default), then build full construct
    if "utr5_seq" not in df.columns:
        df["utr5_seq"] = anchor_u5
    df["utr5_seq"] = df["utr5_seq"].fillna(anchor_u5)
    df["full_seq"] = df["utr5_seq"] + df["cds"] + anchor_u3
    df["full_len"] = df.full_seq.str.len()
    df.to_csv(OUT, sep="\t", index=False)
    print("[cro] %d constructs -> %s" % (len(df), OUT))
    print(df[["construct_id", "group", "cds_name", "utr5_name", "z_nat",
              "cai", "sites", "hp5", "full_len"]].to_string(index=False))

    # ---- pre-registration doc ----
    pred = df[df.group.isin(["ours", "control"])].sort_values(
        "z_nat", ascending=False)
    with open(PREREG, "w", encoding="utf-8") as f:
        f.write("""# CRO 送测包 v1 预注册
# 实验：Fluc mRNA（N1-ψ 修饰）转染 HEK293T，24h 荧光素酶读数
# 条件镜像 GEMORNA Science 'This study Fluc'（可比性）
# 骨架：5'UTR = BNT162b2，3'UTR = BNT162b2（S1 序列）
# 复孔：每构造 3 技术重复（96 孔板 1 块内）

## 预注册预测（湿实验前锁定，防事后挑选）

按我们的 naturalness（S3 外部验证的最强单预测子）排序：

""")
        for _, r in pred.iterrows():
            f.write("%d. %s (%s) — z_nat %+.2f, CAI %.3f, sites %d\n"
                    % (list(pred.construct_id).index(r.construct_id) + 1,
                       r.construct_id, r.group, r.z_nat, r.cai, r.sites))
        f.write("""
## 判定标准（预注册）

1. 我们的 5 条 vs 全部对照（naturalness 排序与表达排序的 Spearman）
2. 主要对照比较：ours 最优 vs GMR-FL 系列最优（倍数差）
3. 若 ours 中位 < CAI-G → composite v2 的 nat+CAI 权重需复盘
""")
    print("[cro] preregistration -> %s" % PREREG)


if __name__ == "__main__":
    main()
