"""Sequence composition features: GC, dinucleotide odds, homopolymers.

All features are computed on RNA (AUGC). DNA input (T) is normalized to U
before this module sees it.
"""
from collections import Counter


def normalize_rna(seq):
    """Uppercase, T->U, keep only AUGC."""
    return seq.upper().replace("T", "U")


def to_dna(seq):
    return seq.upper().replace("U", "T")


def gc_content(seq):
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_slide(seq, window=60, step=10):
    """Sliding-window GC extremes. Returns (max, min) over window centers."""
    n = len(seq)
    if n == 0:
        return 0.0, 0.0
    if n <= window:
        g = gc_content(seq)
        return g, g
    vals = []
    for start in range(0, n - window + 1, step):
        vals.append(gc_content(seq[start:start + window]))
    return max(vals), min(vals)


def dinuc_odds(seq, a, b):
    """Observed/expected frequency of dinucleotide ab.

    expected = count(a)*count(b)/n. Odds > 1 = over-represented.
    UpA under-representation correlates with mRNA stability;
    CpG content relates to innate immune sensing (TLR9 -- contested under psi).
    """
    n = len(seq)
    if n < 2:
        return 0.0
    obs = seq.count(a + b)
    exp = seq.count(a) * seq.count(b) / n
    if exp == 0:
        return 0.0
    return obs / exp


def max_homopolymer(seq, base):
    best = cur = 0
    for ch in seq:
        if ch == base:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def homopolymer_max_runs(seq):
    return {
        "hp_max_A": max_homopolymer(seq, "A"),
        "hp_max_C": max_homopolymer(seq, "C"),
        "hp_max_G": max_homopolymer(seq, "G"),
        "hp_max_U": max_homopolymer(seq, "U"),
    }


def urich_count(seq, k=6, min_u=4):
    """Number of k-mers containing >= min_u uridines (TLR7 legacy /
    manufacturing flag; under N1-psi mostly a manufacturing/rule concern)."""
    if len(seq) < k:
        return 0
    n = 0
    for i in range(len(seq) - k + 1):
        if seq[i:i + k].count("U") >= min_u:
            n += 1
    return n


def seq_features(seq, gc_window=60, gc_step=10):
    seq = normalize_rna(seq)
    n = len(seq)
    gmax, gmin = gc_slide(seq, gc_window, gc_step)
    feats = {
        "seq_len": n,
        "gc_global": gc_content(seq),
        "gc_slide{w}_max".format(w=gc_window): gmax,
        "gc_slide{w}_min".format(w=gc_window): gmin,
        "upa_odds": dinuc_odds(seq, "U", "A"),
        "cpg_odds": dinuc_odds(seq, "C", "G"),
    }
    feats.update(homopolymer_max_runs(seq))
    feats["urich6_count"] = urich_count(seq)
    return feats
