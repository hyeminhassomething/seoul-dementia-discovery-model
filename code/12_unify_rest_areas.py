"""
12_unify_rest_areas.py
공원 + 무더위쉼터 + 한파쉼터 + (옵션)경로당 → 단일 휴식지 점 테이블.
경로당은 좌표가 없으므로 카카오 키 있으면 지오코딩, 없으면 스킵.

[입력]
  data/raw/rest_areas/01_seoul_parks.csv          (133개, WGS84 좌표)
  data/raw/rest_areas/02_seoul_senior_centers.csv (3,649개, 좌표 없음)
  data/raw/rest_areas/03_seoul_cool_shelter.csv   (4,087개, 좌표)
  data/raw/rest_areas/04_seoul_warm_shelters.csv  (1,644개, 좌표)

[통합 스키마]
  rest_id, kind (park/senior_center/cool_shelter/warm_shelter),
  name, address, area_m2, lon, lat, source_file

[모델 의도]
  쉼터 = 치매 노인이 배회 중 멈춰서 발견될 가능성이 높은 지점.
  공원 = 큰 공원은 발견 잘 되지만 산책로 미궁은 어려움.
  → 일단 통합 후 점 단위로 처리. 가중치는 다음 단계에서.

[출력]
  data/interim/rest_areas_unified.csv
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data/raw/rest_areas"
INTERIM = PROJECT_ROOT / "data/interim"
INTERIM.mkdir(parents=True, exist_ok=True)
OUT = INTERIM / "rest_areas_unified.csv"

SEOUL_LAT = (37.41, 37.72)
SEOUL_LON = (126.74, 127.20)

UNIFIED_COLS = [
    "rest_id", "kind", "name", "address", "area_m2",
    "lon", "lat", "source_file",
]


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def parse_area_m2(s) -> float:
    """'2896887㎡', '79258.7㎡', '5,666.00' 같은 다양한 형식을 m² 숫자로."""
    if pd.isna(s):
        return np.nan
    s = str(s)
    # 첫 숫자 패턴만 추출 (콤마 포함)
    m = re.search(r"[\d,]+\.?\d*", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return np.nan


def filter_seoul(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> pd.DataFrame:
    df = df.dropna(subset=[lon_col, lat_col]).copy()
    return df[
        df[lat_col].between(*SEOUL_LAT)
        & df[lon_col].between(*SEOUL_LON)
    ].reset_index(drop=True)


# ─────── 1. 공원 (133) ───────
def load_parks() -> pd.DataFrame:
    src = "01_seoul_parks.csv"
    df = pd.read_csv(RAW / src, encoding="utf-8-sig")
    out = pd.DataFrame()
    out["name"] = df["공원명"].astype(str).str.strip()
    out["address"] = df["공원주소"].astype(str).str.strip()
    out["area_m2"] = df["면적"].map(parse_area_m2)
    out["lon"] = pd.to_numeric(df["X좌표(WGS84)"], errors="coerce")
    out["lat"] = pd.to_numeric(df["Y좌표(WGS84)"], errors="coerce")
    out["kind"] = "park"
    out["source_file"] = src
    return filter_seoul(out)


# ─────── 2. 경로당 (좌표 없음) ───────
def load_senior_centers_addresses_only() -> pd.DataFrame:
    """좌표 없는 raw 데이터를 주소만 추출하여 별도 저장. 지오코딩은 다음 단계."""
    src = "02_seoul_senior_centers.csv"
    raw = pd.read_csv(RAW / src, encoding="utf-8-sig", skiprows=2, header=0)
    # row 2 (skiprows=2 후 첫 행)이 진짜 헤더 = 연번/시도명/시군구명/시설종류/시설명(경로당명)/주소(도로명)/관할 지자체
    # raw.columns = ['연번','시도명','시군구명','시설종류','시설명(경로당명)','주소(도로명)','관할 지자체', ...]
    # 진짜 데이터는 row 3+ (skiprows 후 row 1+)
    raw = raw.dropna(subset=[raw.columns[4]])  # 시설명 비어있는 행 제거
    name_col = raw.columns[4]
    addr_col = raw.columns[5]
    gu_col = raw.columns[2]
    df = pd.DataFrame({
        "name": raw[name_col].astype(str).str.strip(),
        "address": raw[addr_col].astype(str).str.strip(),
        "gu_raw": raw[gu_col].astype(str).str.strip(),
    })
    df = df[df["name"].str.len() > 1]  # 헤더 잔여물 제거
    df = df[~df["name"].isin(["시설명(경로당명)", "(운영, 운영예정, 미운영, 휴지, 폐지 등 경로당포함 )"])]
    return df.reset_index(drop=True)


# ─────── 3. 무더위쉼터 (4,087) ───────
def load_cool_shelters() -> pd.DataFrame:
    src = "03_seoul_cool_shelter.csv"
    df = pd.read_csv(RAW / src, encoding="cp949")
    out = pd.DataFrame()
    out["name"] = df["쉼터명칭"].astype(str).str.strip()
    out["address"] = df["도로명주소"].astype(str).str.strip()
    out["area_m2"] = df["시설면적"].map(parse_area_m2)
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["kind"] = "cool_shelter"
    out["source_file"] = src
    return filter_seoul(out)


# ─────── 4. 한파쉼터 (1,644) ───────
def load_warm_shelters() -> pd.DataFrame:
    src = "04_seoul_warm_shelters.csv"
    df = pd.read_csv(RAW / src, encoding="cp949")
    # 사용여부=Y만
    df = df[df["사용여부"].astype(str).str.upper() == "Y"]
    out = pd.DataFrame()
    out["name"] = df["쉼터명칭"].astype(str).str.strip()
    out["address"] = df["도로명주소"].astype(str).str.strip()
    out["area_m2"] = df["시설면적"].map(parse_area_m2)
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["kind"] = "warm_shelter"
    out["source_file"] = src
    return filter_seoul(out)


def main() -> None:
    parts = []

    df_p = load_parks()
    print(f"[parks] kept={len(df_p)}")
    parts.append(df_p)

    df_c = load_cool_shelters()
    print(f"[cool_shelter] kept={len(df_c)}")
    parts.append(df_c)

    df_w = load_warm_shelters()
    print(f"[warm_shelter] kept={len(df_w)}")
    parts.append(df_w)

    combined = pd.concat(parts, ignore_index=True)

    # ─── dedup: 같은 이름·근접 좌표 (50m 이내) ───
    combined["_lat_r"] = combined["lat"].round(4)
    combined["_lon_r"] = combined["lon"].round(4)
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["_lat_r", "_lon_r", "name"], keep="first"
    ).reset_index(drop=True)
    print(f"[dedup] {before} → {len(combined)} (removed {before - len(combined)})")
    combined = combined.drop(columns=["_lat_r", "_lon_r"])

    combined = combined.sort_values(["kind", "lat", "lon"]).reset_index(drop=True)
    combined["rest_id"] = "R-" + (combined.index + 1).astype(str).str.zfill(5)
    combined = combined[UNIFIED_COLS]

    combined.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"   total: {len(combined):,}")
    print("\n[kind 분포]")
    print(combined["kind"].value_counts().to_string())

    # 경로당은 별도로 주소만 추출하여 저장 (지오코딩 대기)
    df_sc = load_senior_centers_addresses_only()
    sc_path = INTERIM / "senior_centers_addresses_pending_geocode.csv"
    df_sc.to_csv(sc_path, index=False, encoding="utf-8-sig")
    print(f"\n[senior_center] {len(df_sc):,} addresses awaiting geocoding")
    print(f"  saved: {sc_path.relative_to(PROJECT_ROOT)}")
    print(f"  (KAKAO_REST_API_KEY 환경변수 설정 후 별도 단계에서 처리 예정)")


if __name__ == "__main__":
    main()
