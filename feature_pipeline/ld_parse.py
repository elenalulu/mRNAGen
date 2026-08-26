"""LinearDesign lambda-grid output parser (T1 -> T3 glue).

Input format (produced by run_lineardesign_offline.sh on the GPU box,
scp'd back to Windows):

    @@PROTEIN MGADGVGKSALG...
    @@LAMBDA 0.0
    j=0j=1... (noisy GPU progress line -- ignored)
    mRNA sequence:  AUGGGCGCUGAU...
    mRNA structure: ....(((((...
    mRNA folding free energy: -157.10 kcal/mol; mRNA CAI: 0.784
    ...
    @@LAMBDA 0.3
    ...

Usage:
    from ld_parse import parse_lineardesign_output
    records = parse_lineardesign_output("lineardesign_output.txt")
    # list of dicts: protein, lam, seq, mfe_reported, cai_reported
"""
import re

_SEQ_RE = re.compile(r"mRNA sequence:\s*([AUGCaucg]+)")
_STR_RE = re.compile(r"mRNA structure:\s*([().]+)")
_EN_RE = re.compile(
    r"mRNA folding free energy:\s*(-?[\d.]+)\s*kcal/mol;\s*mRNA CAI:\s*(-?[\d.]+)"
)


def parse_lineardesign_output(path):
    """Parse a LinearDesign lambda-grid dump into record dicts."""
    records = []
    protein = None
    lam = None
    seq = None
    structure = None
    mfe = None
    cai = None

    def flush():
        nonlocal seq, structure, mfe, cai, lam
        if lam is not None and seq:
            records.append(
                {
                    "protein": protein or "",
                    "lam": lam,
                    "seq": seq.upper(),
                    "structure_reported": structure or "",
                    "mfe_reported": mfe,
                    "cai_reported": cai,
                    "seq_id": "LD_len{}_lam{}".format(
                        len(protein or ""), lam
                    ),
                }
            )
        seq, structure, mfe, cai, lam = None, None, None, None, None

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("@@PROTEIN"):
                flush()
                protein = line[len("@@PROTEIN"):].strip()
            elif line.startswith("@@LAMBDA"):
                flush()
                try:
                    lam = float(line[len("@@LAMBDA"):].strip())
                except ValueError:
                    lam = None
            elif line.startswith("mRNA sequence:"):
                m = _SEQ_RE.search(line)
                seq = m.group(1).upper() if m else None
            elif line.startswith("mRNA structure:"):
                m = _STR_RE.search(line)
                structure = m.group(1) if m else None
            elif line.startswith("mRNA folding free energy:"):
                m = _EN_RE.search(line)
                if m:
                    mfe = float(m.group(1))
                    cai = float(m.group(2))
            # noisy 'j=0j=1...' progress lines and anything else: ignored
    flush()
    return records
