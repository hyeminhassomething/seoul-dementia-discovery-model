"""Inspect newly uploaded data files."""
import json
import sys
from pathlib import Path

import pandas as pd

# 1. zoning geojson
zoning_path = Path("data/raw/zoning/seoul_zoning.geojson")
print(f"\n===== {zoning_path.name} ({zoning_path.stat().st_size:,} bytes) =====")
try:
    with zoning_path.open(encoding="utf-8") as f:
        first_chars = f.read(2000)
    print("first 1000 chars:")
    print(first_chars[:1000])
    # full parse
    with zoning_path.open(encoding="utf-8") as f:
        gj = json.load(f)
    print(f"\ntype: {gj.get('type')}")
    print(f"features count: {len(gj.get('features', []))}")
    if gj.get("features"):
        feat0 = gj["features"][0]
        print(f"first feature properties keys: {list(feat0.get('properties', {}).keys())}")
        print(f"first feature properties: {feat0.get('properties')}")
        print(f"first feature geometry type: {feat0.get('geometry', {}).get('type')}")
except Exception as e:
    print(f"  error reading as utf-8 json: {e}")
    # try cp949
    try:
        with zoning_path.open(encoding="cp949") as f:
            first = f.read(500)
        print(f"  cp949 first 200: {first[:200]}")
    except Exception as e2:
        print(f"  cp949 also failed: {e2}")


# 2-5. CSVs
csv_paths = [
    "data/raw/rest_areas/01_seoul_parks.csv",
    "data/raw/rest_areas/02_seoul_senior_centers.csv",
    "data/raw/rest_areas/03_seoul_cool_shelter.csv",
    "data/raw/rest_areas/04_seoul_warm_shelters.csv",
    "data/raw/population/주민등록인구(내국인+각+세별_구별)(2014년+이후)_20260503175227.csv",
]
for p in csv_paths:
    p = Path(p)
    print(f"\n===== {p.name} ({p.stat().st_size:,} bytes) =====")
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(p, encoding=enc, low_memory=False)
            print(f"enc={enc}, shape={df.shape}")
            print(f"cols: {df.columns.tolist()}")
            print(df.head(5).to_string(max_colwidth=40))
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  error: {e}")
            break
