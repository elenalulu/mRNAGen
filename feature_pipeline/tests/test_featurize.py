# -*- coding: utf-8 -*-
"""T3 pipeline unit + integration tests. Run:

  python feature_pipeline/tests/test_featurize.py
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from features.seq_stats import seq_features, normalize_rna  # noqa: E402
from features.codon_metrics import get_default_table  # noqa: E402
from features.structure import (  # noqa: E402
    parse_db, helix_lengths, selfcomp_scan, structure_features,
)
from features.rules import rule_features  # noqa: E402
from ld_parse import parse_lineardesign_output  # noqa: E402

PASS = 0


def ok(name, cond, detail=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    print("[%s] %s %s" % (status, name, detail))
    if cond:
        PASS += 1
    else:
        raise AssertionError("%s failed: %s" % (name, detail))


def main():
    table = get_default_table()

    # 1. CAI golden check -- LinearDesign README example MNDTEAI
    #    -> AUGAACGAUACGGAGGCGAUC, CAI 0.695
    cai = table.cai("AUGAACGAUACGGAGGCGAUC")
    ok("cai_golden_readme", abs(cai - 0.695) < 0.002, "cai=%.4f" % cai)

    # 2. helix extraction
    pairs = parse_db("(((...)))")
    h = helix_lengths(pairs)
    ok("helix_simple", h == [3], "h=%s" % h)
    pairs2 = parse_db("..(((..)))..((((((((....))))))))..")
    h2 = helix_lengths(pairs2)
    ok("helix_two", sorted(h2) == [3, 8], "h2=%s" % h2)

    # 3. self-complementarity: engineered 12bp stem with neutral flanks
    x = "AUGGCCAAGGCU"
    seq = "AAAAAA" + x + "UUUU" + x.translate(
        str.maketrans("AUGC", "UACG"))[::-1] + "AAAAAA"
    ex, nr = selfcomp_scan(seq)
    ok("selfcomp_12bp", ex >= 12 and ex <= 13,
       "exact=%d near=%d" % (ex, nr))

    # 4. rules: restriction site on DNA level (RNA GGUCUC -> DNA GGTCTC)
    r = rule_features("AUGGGUCUCAAG")
    ok("rules_bsaI_site", r["restriction_site_count"] >= 1,
       "sites=%d" % r["restriction_site_count"])
    r2 = rule_features("AAAAUGCCAGCUAAC")
    ok("rules_homopolymer_fail", r2["rule_homopolymer_pass"] == 0,
       "hp A run triggers fail")
    r3 = rule_features(normalize_rna(
        "ATGGCTAGCAAAGGAGAAGAACTCTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATG"
        "TTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACC"
    ))
    ok("rules_egfp_like", isinstance(r3["rules_all_pass"], int),
       "runs without error, gc=%.3f" % r3.get("rule_gc_global_pass", -1))

    # 5. ENC sanity
    enc = table.enc("AUGAACGAUACGGAGGCGAUC")
    ok("enc_range", 2.0 <= enc <= 61.0, "enc=%.2f" % enc)

    # 6. seq stats basics
    s = seq_features("AUGGCCAAGGCU")
    ok("seq_len", s["seq_len"] == 12)
    ok("gc_global", abs(s["gc_global"] - 7.0 / 12) < 1e-9,
       "gc=%.4f" % s["gc_global"])

    # 7. structure features smoke (uses ViennaRNA)
    demo = normalize_rna(
        "ATGGCTAGCAAAGGAGAAGAACTCTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATG"
        "TTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCA"
        "GAAACACTCCCCCTTTGAAGGTTTCAACTGGGACCGCCCGCCGAGGTGAAGTTCGAGGGCGACACC"
        "CTGGTGAACCGCATCGAGCTGAAGGGCATCGACTTCAAGGAGGACGGCAACATCCTGGGGCACAAG"
        "CTGGAGTACAACTACAACAGCCACAACGTCTATATCATGGCCGACAAGCAGAAGAACGGCATCAAG"
        "GTGAACTTCAAGATCCGCCACAACATCGAGGACGGCAGCGTGCAGCTCGCCGACCACTACCAGCAG"
        "AACACCCCCATCGGCGACGGCCCCGTGCTGCTGCCCGACAACCACTACCTGAGCACCCAGTCCGCC"
        "CTGAGCAAAGACCCCAACGAGAAGCGCGATCACATGGTCCTGCTGGAGTTCGTGACCGCCGCCGGG"
        "ATCACTCTCGGCATGGACGAGCTGTACAAGTAA"
    )
    f = structure_features(demo)
    ok("mfe_negative", f["mfe"] < 0, "mfe=%.2f" % f["mfe"])
    ok("open_start45_range", 0.0 <= f["open_start45"] <= 1.0,
       "open=%.3f" % f["open_start45"])
    ok("selfcomp_present", f["selfcomp_max_near"] >= 0,
       "near=%d" % f["selfcomp_max_near"])

    # 8. LD output integration (the real T1 output file)
    ld_path = os.path.join(
        BASE, os.pardir, os.pardir, "github", "LinearDesign-main",
        "lineardesign_output.txt")
    if os.path.exists(ld_path):
        recs = parse_lineardesign_output(ld_path)
        ok("ld_parse_count", len(recs) >= 4, "records=%d" % len(recs))
        worst_cai = max(abs(table.cai(r["seq"]) - r["cai_reported"])
                        for r in recs if r["cai_reported"] is not None)
        ok("ld_cai_match", worst_cai <= 0.0015,
           "worst |dCAI|=%.5f" % worst_cai)
        sf = structure_features(recs[0]["seq"])
        # MFE delta vs LinearDesign is a constant parameter-version offset
        # (their bundled old ViennaRNA .so vs our newer params): structures
        # are IDENTICAL; verified consistent across all lambda records.
        import RNA  # noqa: E402
        deltas = []
        for rr in recs:
            fcc = RNA.fold_compound(rr["seq"])
            ssr, mfer = fcc.mfe()
            assert ssr == rr["structure_reported"], \
                "structure mismatch at lam=%s" % rr["lam"]
            deltas.append(mfer - rr["mfe_reported"])
        spread = max(deltas) - min(deltas)
        ok("ld_mfe_offset_consistent",
           abs(deltas[0]) < 5.0 and spread < 0.5,
           "delta=%.2f spread=%.2f (const param offset, structs identical)"
           % (deltas[0], spread))
        ok("ld_translate", table.check_translate(
            recs[0]["seq"], recs[0]["protein"]), "protein round-trip")
    else:
        print("[SKIP] lineardesign_output.txt not found at", ld_path)

    print("\n%d checks passed" % PASS)


if __name__ == "__main__":
    main()
