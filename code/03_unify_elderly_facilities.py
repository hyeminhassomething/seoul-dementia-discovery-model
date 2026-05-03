"""
03_unify_elderly_facilities.py
4개의 서울시 노인복지시설 원본 파일을 통합 스키마로 합쳐
data/interim/elderly_facilities_unified.csv 로 저장한다.

[입력]
  data/raw/elderly_facilities/
    01_노인주거복지시설.csv   (cp949)   → category_main='주거'   prefix='H-'
    02_노인의료복지시설.csv   (cp949)   → category_main='의료'   prefix='M-'
    03_노인의료복지시설현황.xlsx (2 sheets) → category_main='의료현황' prefix='MS-'
    04_노인여가복지시설.csv   (cp949)   → category_main='여가'   prefix='L-'

[통합 스키마]
  facility_id, facility_name, category_main, category_detail,
  address, gu_name, dong_name, capacity, source_file

[매핑 규칙]
  CSV (01/02/04) — 동일 스키마
    facility_id     : <prefix> + 시설코드
    facility_name   : 시설명
    category_main   : 파일별 hardcoded
    category_detail : 시설종류명(시설유형)   e.g. "(노인복지시설) 양로시설"
    address         : 시설주소
    gu_name         : address 정규식 추출 (서울특별시 다음 'XX구')
    dong_name       : address 정규식 추출 ('(XX동)' 형태 우선, 없으면 'XX동' fallback)
    capacity        : NaN
    source_file     : 원본 파일명

  XLSX 시트1/시트2 — header=[1,2] multi-index, skiprows 후 사용
    facility_id     : 'MS-' + 장기요양기관기호 (장기요양기관기호가 unique)
    facility_name   : 기관명칭
    category_main   : '의료현황'
    category_detail : sheet1→'노인요양시설', sheet2→'노인요양공동생활가정'
    address         : 기관소재지(새주소)
    gu_name         : 관할자치구 (이미 'XX구' 형태)
    dong_name       : address 정규식 추출
    capacity        : 이용자→현원 (= 이용현원)
    source_file     : '03_노인의료복지시설현황.xlsx::<sheet>'
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "elderly_facilities"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUT_PATH = INTERIM_DIR / "elderly_facilities_unified.csv"

UNIFIED_COLS = [
    "facility_id",
    "facility_name",
    "category_main",
    "category_detail",
    "address",
    "gu_name",
    "dong_name",
    "capacity",
    "source_file",
]

# ────────────────────────────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────────────────────────────

GU_RE = re.compile(r"서울특별시\s+([가-힣]+구)")
GU_FALLBACK_RE = re.compile(r"([가-힣]+구)")
DONG_PAREN_RE = re.compile(r"\(([가-힣0-9]+동)\)")
DONG_PLAIN_RE = re.compile(r"([가-힣]+[0-9]?동)")  # 화곡동, 신내동, 면목3동 등


def read_csv_auto(path: Path) -> pd.DataFrame:
    """cp949 → utf-8-sig → utf-8 → euc-kr 순으로 시도."""
    last_err: Exception | None = None
    for enc in ("cp949", "utf-8-sig", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise RuntimeError(f"failed to decode {path}: {last_err}")


def extract_gu(address: str | float) -> str | float:
    if not isinstance(address, str):
        return np.nan
    m = GU_RE.search(address)
    if m:
        return m.group(1)
    m = GU_FALLBACK_RE.search(address)
    return m.group(1) if m else np.nan


def extract_dong(address: str | float) -> str | float:
    if not isinstance(address, str):
        return np.nan
    m = DONG_PAREN_RE.search(address)
    if m:
        return m.group(1)
    m = DONG_PLAIN_RE.search(address)
    return m.group(1) if m else np.nan


# ────────────────────────────────────────────────────────────────────
# CSV (01, 02, 04) 처리
# ────────────────────────────────────────────────────────────────────

CSV_SOURCES = [
    {
        "filename": "01_노인주거복지시설.csv",
        "category_main": "주거",
        "prefix": "H-",
    },
    {
        "filename": "02_노인의료복지시설.csv",
        "category_main": "의료",
        "prefix": "M-",
    },
    {
        "filename": "04_노인여가복지시설.csv",
        "category_main": "여가",
        "prefix": "L-",
    },
]


def transform_csv(df: pd.DataFrame, *, category_main: str, prefix: str, filename: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["facility_id"] = prefix + df["시설코드"].astype(str).str.strip()
    out["facility_name"] = df["시설명"].astype(str).str.strip()
    out["category_main"] = category_main
    out["category_detail"] = df["시설종류명(시설유형)"].astype(str).str.strip()
    out["address"] = df["시설주소"].astype(str).str.strip()
    out["gu_name"] = out["address"].map(extract_gu)
    out["dong_name"] = out["address"].map(extract_dong)
    out["capacity"] = np.nan
    out["source_file"] = filename
    return out[UNIFIED_COLS]


# ────────────────────────────────────────────────────────────────────
# XLSX (03) 처리 — multi-row header
# ────────────────────────────────────────────────────────────────────

XLSX_FILENAME = "03_노인의료복지시설현황.xlsx"
SHEET_TO_CATEGORY_DETAIL = {
    "시트1 노인요양시설(241)": "노인요양시설",
    "시트2 노인요양공동생활가정시설(246)": "노인요양공동생활가정",
}


def read_xlsx_status(path: Path, sheet: str) -> pd.DataFrame:
    """
    행 0=제목, 행 1-2=2단 헤더, 행 3+=데이터 인 시트를 읽는다.
    skiprows=3 으로 데이터만 읽고, 컬럼명을 위치 기반으로 직접 부여.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, skiprows=3)
    # 위치 기반 컬럼 부여 (행 1-2 멀티헤더 분석 결과)
    cols = [
        "연번", "관할자치구", "장기요양기관기호", "기관명칭", "설립구분", "법인명",
        "병설시설_母기관", "지정일", "이용자_정원", "이용자_현원",
        "이용현원_구성_계", "이용현원_구성_남", "이용현원_구성_여",
        "이용현원_구성2_계", "이용현원_구성2_치매", "이용현원_구성2_비치매",
        "대기인원", "종사자_현원_계",
        "종사자_시설장", "종사자_사무국장", "종사자_사회복지사", "종사자_의사",
        "종사자_간호사", "종사자_물리치료사", "종사자_요양보호사", "종사자_사무원",
        "종사자_영양사", "종사자_조리원", "종사자_위생원", "종사자_관리인", "종사자_기타",
        "전화", "기관소재지_새주소", "휴업시설",
    ]
    if raw.shape[1] != len(cols):
        # 컬럼 수 불일치 시 가능한 만큼만 매핑
        cols = cols[: raw.shape[1]] + [f"_extra_{i}" for i in range(raw.shape[1] - len(cols))]
    raw.columns = cols
    # 빈 행 제거: 기관명칭이 없는 행은 skip
    raw = raw[raw["기관명칭"].notna() & (raw["기관명칭"].astype(str).str.strip() != "")]
    return raw.reset_index(drop=True)


def transform_xlsx_sheet(df: pd.DataFrame, *, sheet: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["facility_id"] = "MS-" + df["장기요양기관기호"].astype(str).str.strip()
    out["facility_name"] = df["기관명칭"].astype(str).str.strip()
    out["category_main"] = "의료현황"
    out["category_detail"] = SHEET_TO_CATEGORY_DETAIL.get(sheet, sheet)
    out["address"] = df["기관소재지_새주소"].astype(str).str.strip().replace({"nan": np.nan})
    # gu_name: 관할자치구 그대로 (이미 'XX구' 형태). 비어있으면 address fallback.
    gu_from_col = df["관할자치구"].astype(str).str.strip().replace({"nan": np.nan})
    gu_from_addr = out["address"].map(extract_gu)
    out["gu_name"] = gu_from_col.where(gu_from_col.notna(), gu_from_addr)
    out["dong_name"] = out["address"].map(extract_dong)
    out["capacity"] = pd.to_numeric(df["이용자_현원"], errors="coerce")
    out["source_file"] = f"{XLSX_FILENAME}::{sheet}"
    return out[UNIFIED_COLS]


# ────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────

def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    # CSV 3개
    for spec in CSV_SOURCES:
        path = RAW_DIR / spec["filename"]
        df_raw = read_csv_auto(path)
        frames.append(
            transform_csv(
                df_raw,
                category_main=spec["category_main"],
                prefix=spec["prefix"],
                filename=spec["filename"],
            )
        )
        print(f"  loaded {spec['filename']}: rows={len(df_raw)}")

    # XLSX 2 sheets
    xlsx_path = RAW_DIR / XLSX_FILENAME
    for sheet in SHEET_TO_CATEGORY_DETAIL:
        df_raw = read_xlsx_status(xlsx_path, sheet)
        frames.append(transform_xlsx_sheet(df_raw, sheet=sheet))
        print(f"  loaded {XLSX_FILENAME}::{sheet}: rows={len(df_raw)}")

    unified = pd.concat(frames, ignore_index=True)

    # 결과 저장
    unified.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # ────── 검증 출력 ──────
    print("\n" + "=" * 60)
    print(f"✅ saved: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"   total rows: {len(unified)}")
    print("\n[category_main 분포]")
    print(unified["category_main"].value_counts().to_string())
    print("\n[gu_name NaN 비율]")
    print(f"   {unified['gu_name'].isna().sum()} / {len(unified)} "
          f"({unified['gu_name'].isna().mean():.1%})")
    print("[dong_name NaN 비율]")
    print(f"   {unified['dong_name'].isna().sum()} / {len(unified)} "
          f"({unified['dong_name'].isna().mean():.1%})")
    print("[capacity 보유 비율]")
    print(f"   {unified['capacity'].notna().sum()} / {len(unified)} "
          f"({unified['capacity'].notna().mean():.1%})")
    print("\n[샘플 5건]")
    print(unified.sample(min(5, len(unified)), random_state=0).to_string(index=False))


if __name__ == "__main__":
    main()
