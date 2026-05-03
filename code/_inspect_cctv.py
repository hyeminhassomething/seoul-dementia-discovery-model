"""Quick inspection of CCTV raw files."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/cctv")
for f in sorted(RAW.glob("*.csv")):
    print(f"\n===== {f.name} =====")
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(f, encoding=enc)
            print(f"enc={enc} shape={df.shape}")
            print("cols:", df.columns.tolist())
            print(df.head(10).to_string())
            break
        except UnicodeDecodeError:
            continue
