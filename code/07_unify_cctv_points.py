"""
07_unify_cctv_points.py
data/raw/cctv/ 의 다양한 형식 파일을 읽어 단일 점(point) 테이블로 통합한다.

[입력 처리 정책]
  - 마스터: '서울시 불법주정차_전용차로 위반 단속 CCTV 위치정보.csv'
  - 추가:  강북/금천/은평/영등포/관악 일반 CCTV 파일들
  - 구별 불법주정차 파일들은 마스터와 중복이므로 무시 (마스터가 모든 구 커버)
  - 폐기: 강서구(2020 대전 좌표), 송파구 조도 DB, 성북구 단속내역, 도봉구 디비디비맵, 자치구 통계 4개

[통합 스키마]
  cctv_id, gu_name, address, purpose_main, purpose_detail,
  install_year, camera_count, lon, lat, source_file

[좌표 sanity]
  서울 bbox 안만 유지 (lat 37.41~37.72, lon 126.74~127.20)
  → 자동으로 잘못된 좌표(대전 등) 제거
  → 영등포구처럼 X/Y 컬럼 명이 lat/lon과 뒤바뀐 경우는 어댑터에서 처리
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def nfc(s: str) -> str:
    """macOS의 NFD 한글 파일명을 NFC로 정규화."""
    return unicodedata.normalize("NFC", s)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw" / "cctv"
INTERIM = PROJECT_ROOT / "data" / "interim"
OUT = INTERIM / "cctv_points_unified.csv"

SEOUL_LAT = (37.41, 37.72)
SEOUL_LON = (126.74, 127.20)

UNIFIED_COLS = [
    "cctv_id", "gu_name", "address", "purpose_main", "purpose_detail",
    "install_year", "camera_count", "lon", "lat", "source_file",
]

# ─────────── 공통 유틸 ───────────
GU_RE = re.compile(r"([가-힣]+구)")


def read_csv_auto(p: Path) -> pd.DataFrame | None:
    for enc in ("cp949", "utf-8-sig", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(p, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        except pd.errors.EmptyDataError:
            return None
    return None


def normalize_purpose(value) -> tuple[str, str]:
    """원본 목적 문자열 → (purpose_main, purpose_detail)."""
    if not isinstance(value, str):
        return ("unknown", "")
    v = value.strip().replace(" ", "").replace("·", "")
    detail = value.strip()
    if "방범" in v or "범죄예방" in v or "수사" in v:
        return ("crime_prevention", detail)
    if "어린이" in v or "스쿨" in v:
        return ("school_zone", detail)
    if "주정차" in v or "주.정차" in v or "전용차로" in v:
        return ("parking_enforcement", detail)
    if "교통" in v:
        return ("traffic", detail)
    if "공원" in v or "놀이터" in v:
        return ("park", detail)
    if "쓰레기" in v or "무단투기" in v:
        return ("illegal_dumping", detail)
    if "화재" in v or "시설안전" in v:
        return ("facility_safety", detail)
    return ("other", detail)


def extract_year(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value)
    m = re.search(r"(19|20)\d{2}", s)
    return float(m.group(0)) if m else np.nan


def extract_gu_from_address(addr) -> str:
    if not isinstance(addr, str):
        return ""
    m = GU_RE.search(addr)
    return m.group(1) if m else ""


# ─────────── 어댑터: 파일 종류별 매퍼 ───────────
# 각 어댑터는 raw DataFrame 받아서 UNIFIED_COLS 형식의 DataFrame 반환

def adapter_parking_master(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """서울시 불법주정차_전용차로 (마스터) - 6컬럼."""
    out = pd.DataFrame()
    out["address"] = df["고정형CCTV지번주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["gu_name"] = df["자치구"].astype(str).str.strip()
    purposes = df["현장구분"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = np.nan
    out["camera_count"] = 1
    # 단속지점명을 address 보강
    spot = df.get("단속지점명", "").astype(str).str.strip()
    out["address"] = out["address"].where(out["address"].str.len() > 0, spot)
    out["source_file"] = source_file
    return out


def adapter_gangbuk(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """강북구: 행정동, 설치 수, 도로명주소, 설치목적, 카메라화소수, 보관일수, 위도, 경도."""
    out = pd.DataFrame()
    out["address"] = df["도로명주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["gu_name"] = "강북구"
    purposes = df["설치목적"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = df.get("데이터기준일자", "").apply(extract_year)
    out["camera_count"] = pd.to_numeric(df.get("설치 수", 1), errors="coerce").fillna(1).astype(int)
    out["source_file"] = source_file
    return out


def adapter_geumcheon(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """금천구: 명칭, 좌표주소, 위도, 경도, 상세주소, 상세내용."""
    out = pd.DataFrame()
    out["address"] = df["상세주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["gu_name"] = "금천구"
    purposes = df["명칭"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = np.nan
    out["camera_count"] = 1
    out["source_file"] = source_file
    return out


def adapter_eunpyeong(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """은평구: 관리기관명, 소재지도로명주소, 소재지지번주소, 설치목적구분, 카메라대수, 설치연월, 위도, 경도."""
    out = pd.DataFrame()
    out["address"] = df["소재지도로명주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["gu_name"] = "은평구"
    purposes = df["설치목적구분"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = df["설치연월"].apply(extract_year)
    out["camera_count"] = pd.to_numeric(df["카메라대수"], errors="coerce").fillna(1).astype(int)
    out["source_file"] = source_file
    return out


def adapter_yeongdeungpo(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """영등포구: 시도명, 시군구명, 안심주소, 용도, X좌표(WSG), Y좌표(WSG), 수량.
    ※ X/Y 라벨이지만 실제로 X=위도, Y=경도 (sample 값으로 확인)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]  # leading spaces in column names
    out = pd.DataFrame()
    out["address"] = df["안심주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["X좌표(WSG)"], errors="coerce")  # actual lat
    out["lon"] = pd.to_numeric(df["Y좌표(WSG)"], errors="coerce")  # actual lon
    out["gu_name"] = "영등포구"
    purposes = df["용도"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = df.get("데이터기준일자", "").apply(extract_year)
    out["camera_count"] = pd.to_numeric(df.get("수량", 1), errors="coerce").fillna(1).astype(int)
    out["source_file"] = source_file
    return out


def adapter_gwanak(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """관악구: 연번, 행정구역, 고정형CCTV지번주소, 위도, 경도, 단속지점명, 현장구분."""
    out = pd.DataFrame()
    out["address"] = df["고정형CCTV지번주소"].astype(str).str.strip()
    out["lat"] = pd.to_numeric(df["위도"], errors="coerce")
    out["lon"] = pd.to_numeric(df["경도"], errors="coerce")
    out["gu_name"] = "관악구"
    purposes = df["현장구분"].apply(normalize_purpose)
    out["purpose_main"] = [p[0] for p in purposes]
    out["purpose_detail"] = [p[1] for p in purposes]
    out["install_year"] = np.nan
    out["camera_count"] = 1
    spot = df.get("단속지점명", "").astype(str).str.strip()
    out["address"] = out["address"].where(out["address"].str.len() > 0, spot)
    out["source_file"] = source_file
    return out


# ─────────── 파일별 어댑터 매핑 ───────────
ADAPTERS: dict[str, Callable[[pd.DataFrame, str], pd.DataFrame]] = {
    "서울시 불법주정차_전용차로 위반 단속 CCTV 위치정보.csv": adapter_parking_master,
    "서울시 강북구 CCTV 설치 현황.csv": adapter_gangbuk,
    "서울시 금천구 CCTV 설치 위치정보.csv": adapter_geumcheon,
    "서울특별시 은평구_CCTV 현황_20250101.csv": adapter_eunpyeong,
    "서울특별시 영등포구_서울특별시_영등포구_CCTV설치현황_20260403.csv": adapter_yeongdeungpo,
    "서울특별시 관악구_불법주정차 위반 단속 CCTV 위치 정보_20251105.csv": adapter_gwanak,
}

# ─────────── main ───────────
def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    print(f"[scan] {RAW}")
    # NFC 정규화한 키로 어댑터 매칭
    adapters_nfc = {nfc(k): v for k, v in ADAPTERS.items()}
    for f in sorted(RAW.glob("*.csv")):
        name_nfc = nfc(f.name)
        adapter = adapters_nfc.get(name_nfc)
        if adapter is None:
            skipped.append((name_nfc, "no adapter (구별 불법주정차/통계/폐기)"))
            continue
        df = read_csv_auto(f)
        if df is None or df.empty:
            skipped.append((f.name, "read failed / empty"))
            continue
        try:
            sub = adapter(df, name_nfc)
        except Exception as e:
            skipped.append((name_nfc, f"adapter error: {e}"))
            continue
        before = len(sub)
        # 좌표 정제
        sub = sub.dropna(subset=["lat", "lon"])
        in_seoul = (
            sub["lat"].between(*SEOUL_LAT)
            & sub["lon"].between(*SEOUL_LON)
        )
        sub = sub[in_seoul]
        # gu_name fallback (raw에 없거나 비정상이면 주소에서 추출)
        gu_from_addr = sub["address"].apply(extract_gu_from_address)
        sub.loc[~sub["gu_name"].astype(str).str.endswith("구"), "gu_name"] = gu_from_addr
        print(f"  ✔ {name_nfc}: raw={len(df)} → kept={len(sub)} (filtered {before - len(sub)})")
        frames.append(sub)

    if skipped:
        print("\n[skipped files]")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")

    if not frames:
        raise SystemExit("no data after filtering")

    combined = pd.concat(frames, ignore_index=True)

    # ─── gu_name 보정: 좌표 기반 nearest-neighbor (이전 단계 facility 데이터 활용) ───
    facility_path = PROJECT_ROOT / "data" / "processed" / "elderly_facilities_geocoded_grid_seoul.csv"
    if facility_path.exists():
        fac = pd.read_csv(facility_path)
        fac = fac.dropna(subset=["lat", "lon", "gu_name"])
        fac = fac[fac["gu_name"].str.endswith("구")]
        # 빈 gu_name 또는 "서울시" 같은 잘못된 값들을 식별
        bad_mask = (
            combined["gu_name"].fillna("").str.strip().eq("")
            | combined["gu_name"].eq("서울시")
            | ~combined["gu_name"].astype(str).str.endswith("구")
        )
        n_bad = int(bad_mask.sum())
        if n_bad > 0 and len(fac) > 0:
            fac_lat = fac["lat"].to_numpy()
            fac_lon = fac["lon"].to_numpy()
            fac_gu = fac["gu_name"].to_numpy()
            for idx in combined.index[bad_mask]:
                lat = combined.at[idx, "lat"]
                lon = combined.at[idx, "lon"]
                # 단순 유클리디안 (서울 스케일에선 충분)
                d2 = (fac_lat - lat) ** 2 + (fac_lon - lon) ** 2
                combined.at[idx, "gu_name"] = fac_gu[d2.argmin()]
            print(f"[gu_fix] {n_bad} rows reassigned via nearest facility")

    # ─── dedup ───
    # lat/lon 1e-5 정밀도(약 1m)로 라운드한 키로 중복 제거
    # 같은 위치에 다른 purpose가 등록된 경우는 union 의미로 살림: dedup key에 purpose_main 포함
    combined["_lat_r"] = combined["lat"].round(5)
    combined["_lon_r"] = combined["lon"].round(5)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(
        subset=["_lat_r", "_lon_r", "purpose_main"], keep="first"
    ).reset_index(drop=True)
    combined = combined.drop(columns=["_lat_r", "_lon_r"])
    print(f"\n[dedup] {before_dedup} → {len(combined)} (removed {before_dedup - len(combined)})")

    # cctv_id 부여 (정렬 후 안정적인 순번)
    combined = combined.sort_values(["gu_name", "lat", "lon"]).reset_index(drop=True)
    combined["cctv_id"] = "C-" + (combined.index + 1).astype(str).str.zfill(6)
    combined = combined[UNIFIED_COLS]

    combined.to_csv(OUT, index=False, encoding="utf-8-sig")

    # ─── 검증 출력 ───
    print("\n" + "=" * 70)
    print(f"✅ saved: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"   total points: {len(combined):,}")
    print(f"   camera_count sum: {combined['camera_count'].sum():,}")

    print("\n[gu별 분포]")
    gu_dist = combined.groupby("gu_name").size().sort_values(ascending=False)
    print(gu_dist.to_string())
    print(f"\n  unique gus: {gu_dist.index.nunique()} / 25 expected")
    missing = set(["강남구","강동구","강북구","강서구","관악구","광진구","구로구","금천구","노원구","도봉구","동대문구","동작구","마포구","서대문구","서초구","성동구","성북구","송파구","양천구","영등포구","용산구","은평구","종로구","중구","중랑구"]) - set(gu_dist.index)
    if missing:
        print(f"  MISSING: {sorted(missing)}")

    print("\n[purpose_main 분포]")
    print(combined["purpose_main"].value_counts().to_string())

    print("\n[source_file 분포]")
    print(combined["source_file"].value_counts().to_string())


if __name__ == "__main__":
    main()
