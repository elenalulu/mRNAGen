"""Codon-level features: CAI (exact LinearDesign port), ENC, GC3.

CAI replicates src/Utils/codon.h::calc_cai from LinearDesign bit-for-bit in
spirit: iterate over ALL codons of the full RNA sequence (start AUG and stop
codon INCLUDED), w_i = f(codon) / max f(synonymous family), geometric mean.

Golden check: README example MNDTEAI -> AUGAACGAUACGGAGGCGAUC, CAI 0.695.
"""
import math
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_TABLE = os.path.join(
    os.path.dirname(_THIS_DIR), "data", "codon_usage_freq_table_human.csv"
)

# Wright (1990) ENC family sizes by amino acid
_FAMILY = {
    "A": 4, "R": 6, "N": 2, "D": 2, "C": 2, "Q": 2, "E": 2, "G": 4,
    "H": 2, "I": 3, "L": 6, "K": 2, "M": 1, "F": 2, "P": 4, "S": 6,
    "T": 4, "W": 1, "Y": 2, "V": 4, "*": 3,
}


class CodonTable(object):
    def __init__(self, path=_DEFAULT_TABLE):
        self.codon_to_aa = {}
        self.codon_freq = {}
        self.aa_max = {}
        with open(path, "r", encoding="utf-8-sig") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line or idx == 0:
                    continue  # header row: '#,,'
                parts = line.split(",")
                if len(parts) != 3:
                    continue
                codon, aa, frac = parts[0].strip().upper(), parts[1].strip(), parts[2].strip()
                if len(codon) != 3:
                    continue
                try:
                    f = float(frac)
                except ValueError:
                    continue
                self.codon_to_aa[codon] = aa
                self.codon_freq[codon] = f
                if aa not in self.aa_max or f > self.aa_max[aa]:
                    self.aa_max[aa] = f
        if len(self.codon_to_aa) != 64:
            raise ValueError(
                "codon table has {} codons (need 64): {}".format(
                    len(self.codon_to_aa), path
                )
            )

    def codons(self, rna):
        return [rna[i:i + 3] for i in range(0, len(rna) - len(rna) % 3, 3)]

    def cai(self, rna):
        """LinearDesign-caliber CAI: all codons, start & stop included."""
        codons = self.codons(rna)
        if not codons:
            return 0.0
        total = 0.0
        for c in codons:
            aa = self.codon_to_aa[c]
            w = self.codon_freq[c] / self.aa_max[aa]
            total += math.log2(w)
        return 2.0 ** (total / len(codons))

    def cai_excl_stop(self, rna):
        """CAI excluding terminal stop codon (reporting variant)."""
        codons = self.codons(rna)
        if codons and self.codon_to_aa[codons[-1]] in ("*", "STOP"):
            codons = codons[:-1]
        if not codons:
            return 0.0
        total = 0.0
        for c in codons:
            aa = self.codon_to_aa[c]
            total += math.log2(self.codon_freq[c] / self.aa_max[aa])
        return 2.0 ** (total / len(codons))

    def gc3(self, rna):
        codons = self.codons(rna)
        if not codons:
            return 0.0
        third = "".join(c[2] for c in codons)
        return (third.count("G") + third.count("C")) / len(third)

    def stop_codon(self, rna):
        codons = self.codons(rna)
        if not codons:
            return ""
        return codons[-1]

    def enc(self, rna):
        """Wright's effective number of codons (F statistic version)."""
        codons = self.codons(rna)
        counts = {}
        for c in codons:
            aa = self.codon_to_aa[c]
            counts.setdefault(aa, {})
            counts[aa][c] = counts[aa].get(c, 0) + 1
        # F per family size, averaged over amino acids of that family size
        fam_F = {2: [], 3: [], 4: [], 6: []}
        for aa, ccounts in counts.items():
            size = _FAMILY.get(aa, 1)
            if size == 1 or aa == "*":
                continue
            n = sum(ccounts.values())
            if n <= 1:
                fam_F[size].append(1.0)
                continue
            ssum = 0.0
            for c in self._family_codons(aa):
                p = ccounts.get(c, 0) / n
                ssum += p * p
            F = (n * ssum - 1.0) / (n - 1.0)
            fam_F[size].append(F)
        mean_F = {}
        for size, vals in fam_F.items():
            if vals:
                mean_F[size] = sum(vals) / len(vals)
        enc = 2.0  # Met, Trp
        # F can hit 0 in edge cases (e.g. Ile family, n=2 split evenly
        # between two codons -> F=0); clamp denominators.
        enc += 9.0 / max(mean_F[2], 1e-6) if 2 in mean_F else 9.0
        enc += 1.0 / max(mean_F[3], 1e-6) if 3 in mean_F else 1.0
        enc += 5.0 / max(mean_F[4], 1e-6) if 4 in mean_F else 5.0
        enc += 3.0 / max(mean_F[6], 1e-6) if 6 in mean_F else 3.0
        return min(enc, 61.0)  # theoretical maximum

    def _family_codons(self, aa):
        return [c for c, a in self.codon_to_aa.items() if a == aa]

    def check_translate(self, rna, protein):
        """True iff rna codons translate to protein (stop-tolerant tail)."""
        codons = self.codons(rna)
        aa = []
        for c in codons:
            a = self.codon_to_aa[c]
            if a in ("*", "STOP"):
                break
            aa.append(a)
        return "".join(aa) == protein.upper()


def codon_features(rna, table):
    return {
        "cai": table.cai(rna),
        "cai_excl_stop": table.cai_excl_stop(rna),
        "gc3": table.gc3(rna),
        "enc": table.enc(rna),
    }


_DEFAULT_TABLE_OBJ = None


def get_default_table():
    global _DEFAULT_TABLE_OBJ
    if _DEFAULT_TABLE_OBJ is None:
        _DEFAULT_TABLE_OBJ = CodonTable()
    return _DEFAULT_TABLE_OBJ
