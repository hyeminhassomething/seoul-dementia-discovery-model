"""Scan every file in data/raw/cctv/ and report (encoding, shape, columns, sample)."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/cctv")

def try_read(p: Path):
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(p, encoding=enc, low_memory=False)
            return df, enc
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return None, None


for f in sorted(RAW.glob("*.csv")):
    print(f"\n===== {f.name} ({f.stat().st_size:,} bytes) =====")
    df, enc = try_read(f)
    if df is None:
        print("  ❌ failed to read")
        continue
    print(f"  enc={enc} shape={df.shape}")
    print(f"  cols: {df.columns.tolist()}")
    if len(df) > 0:
        print("  sample 2 rows:")
        print(df.head(2).to_string(max_colwidth=30))
