"""Generate the untouched replica dataset (same generator, different seed) for validation.

    python scripts/make_replica.py [--seed 2026] [--out data_replica_seed2026]

The original dataset is never modified: the generator source is executed with a
different SEED and output directory.
"""
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "src" / "leak_hunter" / "Dataset" / "generate_synthetic_data.py"

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = Path(a.out or ROOT / f"data_replica_seed{a.seed}").resolve()
src = GEN.read_text()
src, n1 = re.subn(r"^SEED = \d+$", f"SEED = {a.seed}", src, flags=re.M)
src, n2 = re.subn(r"^OUT_DIR = .*$", f"OUT_DIR = {str(out)!r}", src, flags=re.M)
assert n1 == 1 and n2 == 1, "generator layout changed: SEED / OUT_DIR not found"
exec(compile(src, str(GEN), "exec"), {"__name__": "__main__", "__file__": str(GEN)})
