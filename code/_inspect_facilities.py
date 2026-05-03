"""Inspect raw elderly-facility files: encoding, columns, head, dtypes."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/elderly_facilities")


def read_csv_auto(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to decode {p}")


def report_df(label: str, df: pd.DataFrame) -> None:
    print(f"\n{'=' * 70}\n[{label}] rows={len(df)}, cols={len(df.columns)}")
    print("columns:", df.columns.tolist())
    print("dtypes:")
    print(df.dtypes.to_string())
    print("head(3):")
    print(df.head(3).to_string(max_colwidth=40))


# CSVs
for f in sorted(RAW.glob("*.csv")):
    df = read_csv_auto(f)
    # detect which encoding worked
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            pd.read_csv(f, encoding=enc, nrows=1)
            used_enc = enc
            break
        except UnicodeDecodeError:
            continue
    report_df(f"{f.name}  (enc={used_enc})", df)

# XLSX (multiple sheets)
for f in sorted(RAW.glob("*.xlsx")):
    xl = pd.ExcelFile(f)
    print(f"\n{'#' * 70}\n[{f.name}] sheets={xl.sheet_names}")
    for sheet in xl.sheet_names:
        # try header row 0,1,2 to find best header (Korean gov files often have title rows)
        for hdr in (0, 1, 2, 3):
            try:
                df = pd.read_excel(f, sheet_name=sheet, header=hdr)
                if df.shape[1] > 1 and not df.columns.astype(str).str.startswith("Unnamed").all():
                    break
            except Exception:
                continue
        report_df(f"{f.name} :: {sheet}  (header_row={hdr})", df)
