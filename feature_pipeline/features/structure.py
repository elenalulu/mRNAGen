"""Structure features via ViennaRNA (the immunogenicity head's main course).

Design principles (from mRNA_design_engine_v1_plan.md sec 2.3):
- MFE is LinearDesign's OWN proxy -> kept as a background variable, NEVER the
  headline feature.
- MDA5 senses long near-perfect duplexes -> headline features are
  self-complementarity scan (longest intramolecular duplex) and helix stats.
- Ensemble (partition function) features where affordable; windowed fallback
  for long sequences; MFE-structure fallback if pf unavailable.

Environment: runs under D:/anaconda/python.exe (ViennaRNA python bindings,
no new dependencies -- anaconda base is used read-only, never pip-installed).
"""
import RNA as vrna

PF_MAX_LEN = 3000        # global partition function above this is too heavy
LOCAL_WINDOW = 150       # window for start-region openness
START_REGION = 45        # Kozak-downstream ~30-50 nt of CDS
SAMPLE_WINDOW = 200      # window size for long-seq pairing stats
SAMPLE_STRIDE = 500
SELFCOMP_SEED = 10       # exact revcomp seed length
SELFCOMP_MAX_MISMATCH = 2
SELFCOMP_PAIR_CAP = 20000  # safety cap on seed pair expansions

_COMP = str.maketrans("AUGC", "UACG")
_COMP_BASES = {"A": "U", "U": "A", "G": "C", "C": "G"}


def revcomp(s):
    return s.translate(_COMP)[::-1]


# ---------------------------------------------------------------- helix utils

def parse_db(ss):
    """dot-bracket -> dict {i: j} (0-based, i < j). Handles () [] {}."""
    pairs = {}
    stacks = {"(": [], "[": [], "{": []}
    closes = {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(ss):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in closes:
            j = stacks[closes[ch]].pop()
            pairs[j] = i
    return pairs


def helix_lengths(pairs):
    """Lengths of contiguous stems in a base-pair map.

    A helix of length L: i, i+1, ..., i+L-1 paired with j, j-1, ..., j-L+1.
    """
    lengths = []
    seen = set()
    for i in sorted(pairs):
        if i in seen:
            continue
        j = pairs[i]
        L = 1
        seen.add(i)
        while (i + L) in pairs and pairs[i + L] == j - L:
            seen.add(i + L)
            L += 1
        lengths.append(L)
    return lengths


# ------------------------------------------------------------------ MFE block

def mfe_features(seq):
    fc = vrna.fold_compound(seq)
    ss, mfe = fc.mfe()
    n = len(seq)
    pairs = parse_db(ss)
    h = helix_lengths(pairs) or [0]
    return {
        "mfe": mfe,
        "mfe_per_nt": mfe / n if n else 0.0,
        "paired_frac_mfe": (2.0 * len(pairs) / n) if n else 0.0,
        "longest_helix_mfe": max(h),
        "n_helix_ge8": sum(1 for x in h if x >= 8),
    }


# -------------------------------------------------------------- ensemble part

def unpaired_probs(seq):
    """Per-base unpaired probability via partition function.

    Returns None on failure or if sequence too long for global pf.
    """
    n = len(seq)
    if n < 5 or n > PF_MAX_LEN:
        return None
    try:
        fc = vrna.fold_compound(seq)
        fc.pf()
        bp = fc.bpp()
        pun = [1.0] * n
        for i in range(1, n + 1):
            row = bp[i]
            s = 0.0
            for j in range(1, n + 1):
                if j != i:
                    s += row[j]
            if s > 1.0:
                s = 1.0
            pun[i - 1] = 1.0 - s
        return pun
    except Exception:
        return None


def _mfe_unpaired(seq):
    """Fallback: unpaired indicator vector from MFE structure."""
    fc = vrna.fold_compound(seq)
    ss, _ = fc.mfe()
    return [1.0 if ch == "." else 0.0 for ch in ss]


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def openness_profile_features(seq, fast=False):
    """Start-region openness + global/mean pairing statistics.

    start openness: mean unpaired probability over the first START_REGION nt,
    computed on a LOCAL_WINDOW local fold (CDS-only design: no UTR context).
    """
    n = len(seq)
    local = seq[:LOCAL_WINDOW]
    if not fast:
        pun_local = unpaired_probs(local)
    else:
        pun_local = None
    if pun_local is None:
        pun_local = _mfe_unpaired(local)
    open_start = _mean(pun_local[:START_REGION])

    out = {"open_start45": open_start}
    if fast:
        return out

    if n <= PF_MAX_LEN:
        pun = unpaired_probs(seq)
        if pun is None:
            pun = _mfe_unpaired(seq)
            out["ensemble_ok"] = 0
        else:
            out["ensemble_ok"] = 1
        out["mean_unpaired"] = _mean(pun)
        sorted_pun = sorted(pun)
        out["mean_unpaired_q25"] = _mean(sorted_pun[: max(1, len(pun) // 4)])
    else:
        # long sequence: sample windows
        vals = []
        for start in range(0, max(1, n - SAMPLE_WINDOW + 1), SAMPLE_STRIDE):
            w = seq[start:start + SAMPLE_WINDOW]
            p = unpaired_probs(w)
            if p is None:
                p = _mfe_unpaired(w)
            vals.append(_mean(p))
        out["ensemble_ok"] = 0
        out["mean_unpaired"] = _mean(vals)
        out["mean_unpaired_q25"] = None
    return out


# ------------------------------------------------------ self-complementarity

def _extend_duplex(seq, i, j, seed_k, tol):
    """Extend a seed alignment X=[i..i+seed_k-1] ~ Y=[j..j+seed_k-1] (i < j,
    Y is the reverse complement of X).

    All aligned pairs satisfy p + q == i + j + seed_k - 1. Extends inward
    (toward the loop) and outward with a SHARED mismatch budget of `tol`;
    mismatch positions remain part of the alignment (near-perfect duplex).
    Returns the total alignment length (bp of duplex span).
    """
    n = len(seq)
    x2, y1 = i + seed_k - 1, j            # inner frontier
    xx1, yy2 = i, j + seed_k - 1          # outer frontier
    mis = 0
    while mis <= tol:
        moved = False
        # inward
        if y1 - 1 > x2 + 1 and x2 + 1 < n and y1 - 1 >= 0:
            if _COMP_BASES.get(seq[x2 + 1]) == seq[y1 - 1]:
                x2 += 1
                y1 -= 1
                moved = True
            elif mis < tol:
                x2 += 1
                y1 -= 1
                mis += 1
                moved = True
        # outward
        if xx1 - 1 >= 0 and yy2 + 1 < n:
            if _COMP_BASES.get(seq[xx1 - 1]) == seq[yy2 + 1]:
                xx1 -= 1
                yy2 += 1
                moved = True
            elif mis < tol:
                xx1 -= 1
                yy2 += 1
                mis += 1
                moved = True
        if not moved:
            break
    return x2 - xx1 + 1


def selfcomp_scan(seq, seed_k=SELFCOMP_SEED,
                  max_mismatch=SELFCOMP_MAX_MISMATCH):
    """Longest intramolecular near-perfect duplex (MDA5-class risk).

    Seeds with exact reverse-complement k-mer matches between non-overlapping
    positions, then extends with the shared mismatch budget.

    Returns (best_exact, best_near):
      best_exact -- longest duplex with 0 mismatches
      best_near  -- longest duplex with <= max_mismatch mismatches
    """
    n = len(seq)
    if n < 2 * seed_k + 3:
        return 0, 0

    from collections import defaultdict

    pos = defaultdict(list)
    for i in range(n - seed_k + 1):
        pos[seq[i:i + seed_k]].append(i)

    best_exact = 0
    best_near = 0
    expanded = 0
    for kmer, plist in pos.items():
        rc = revcomp(kmer)
        partners = pos.get(rc)
        if not partners:
            continue
        same = rc == kmer
        for i in plist:
            for j in partners:
                if same and j <= i:
                    continue
                # non-overlapping with >= 3 nt between the two segments
                lo, hi = (i, j) if i < j else (j, i)
                if hi - lo < seed_k + 3:
                    continue
                expanded += 1
                if expanded > SELFCOMP_PAIR_CAP:
                    return best_exact, best_near
                best_exact = max(
                    best_exact, _extend_duplex(seq, lo, hi, seed_k, 0)
                )
                best_near = max(
                    best_near,
                    _extend_duplex(seq, lo, hi, seed_k, max_mismatch),
                )
    return best_exact, best_near


def structure_features(seq, fast=False):
    feats = mfe_features(seq)
    feats.update(openness_profile_features(seq, fast=fast))
    if not fast:
        ex, nr = selfcomp_scan(seq)
        feats["selfcomp_max_exact"] = ex
        feats["selfcomp_max_near"] = nr
    return feats
