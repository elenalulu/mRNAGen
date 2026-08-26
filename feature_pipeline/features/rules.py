"""Manufacturing / safety rule checker (hard constraints + soft counts).

All checks run on the DNA equivalent (U->T) since restriction sites and
splice motifs are DNA-level design conventions.

Rule defaults (all configurable):
- global GC in [30, 70]%
- 60nt sliding window GC in [20, 80]%
- homopolymer runs: max allowed = 3 (a 4-run fails) for every base
- restriction sites: BsaI, BsmBI, EcoRI, XhoI, BamHI, NotI (both strands)
- cryptic splice donor motifs: GTAAGT, GTGAGT (coarse proxy for MAG|GTRAGT
  consensus; a serious version should run SpliceAI later)
- U-rich: number of 6-mers with >= 4 U

Outputs are numeric (0/1 pass flags + counts) so they can enter the oracle
directly; rules_all_pass is the hard gate for candidate filtering.
"""
from .seq_stats import normalize_rna, to_dna, gc_content, gc_slide

RESTRICTION_SITES = {
    "BsaI": ["GGTCTC", "GAGACC"],
    "BsmBI": ["CGTCTC", "GAGACG"],
    "EcoRI": ["GAATTC"],
    "XhoI": ["CTCGAG"],
    "BamHI": ["GGATCC"],
    "NotI": ["GCGGCCGC"],
}

CRYPTIC_DONOR_MOTIFS = ["GTAAGT", "GTGAGT"]

DEFAULTS = {
    "gc_global_lo": 0.30,
    "gc_global_hi": 0.70,
    "gc_window": 60,
    "gc_slide_lo": 0.20,
    "gc_slide_hi": 0.80,
    "homopolymer_max": 3,
}


def _max_run(seq, base):
    best = cur = 0
    for ch in seq:
        if ch == base:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def rule_features(rna_seq, config=None):
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)
    rna = normalize_rna(rna_seq)
    dna = to_dna(rna)
    n = len(rna)

    gc_g = gc_content(rna)
    gmax, gmin = gc_slide(rna, cfg["gc_window"])

    hp_runs = {b: _max_run(rna, b) for b in "ACGU"}
    hp_pass = all(v <= cfg["homopolymer_max"] for v in hp_runs.values())

    site_count = 0
    for name, motifs in RESTRICTION_SITES.items():
        site_count += sum(dna.count(m) for m in motifs)

    donor_count = sum(dna.count(m) for m in CRYPTIC_DONOR_MOTIFS)

    urich = 0
    if n >= 6:
        for i in range(n - 5):
            if rna[i:i + 6].count("U") >= 4:
                urich += 1

    gc_global_pass = cfg["gc_global_lo"] <= gc_g <= cfg["gc_global_hi"]
    gc_slide_pass = gmin >= cfg["gc_slide_lo"] and gmax <= cfg["gc_slide_hi"]

    return {
        "rule_gc_global_pass": int(gc_global_pass),
        "rule_gc_slide_pass": int(gc_slide_pass),
        "rule_homopolymer_pass": int(hp_pass),
        "restriction_site_count": site_count,
        "cryptic_donor_count": donor_count,
        "urich6_count": urich,
        "rules_all_pass": int(
            gc_global_pass and gc_slide_pass and hp_pass and site_count == 0
        ),
    }
