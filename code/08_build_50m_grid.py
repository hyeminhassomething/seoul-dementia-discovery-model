"""
08_build_50m_grid.py
서울 bbox 위에 50m × 50m 격자를 생성하고, 점 좌표 ↔ grid_id 변환 함수를 제공한다.

[출력]
  data/interim/seoul_grid_50m.csv  (전체 격자 ID와 좌표 메타. 파일 크기 약 25MB)
  하지만 매번 디스크에 쓰기엔 너무 크므로, 필요한 격자만 lazy 생성하는 함수형으로 운영.
  → 이 스크립트는 grid_id 변환 함수를 정의하고, '실제로 점이 떨어진' 격자만 추출한 작은 CSV를 만든다.

[격자 ID 형식]
  G50-{row:05d}-{col:05d}
    - row: 북쪽 끝(max_lat)에서 남쪽으로 0,1,2... (행 인덱스)
    - col: 서쪽 끝(min_lon)에서 동쪽으로 0,1,2... (열 인덱스)
  예: G50-00123-00456

[격자 스펙]
  cell size: 50m × 50m
  서울 위도 기준 (37.5°)에서:
    1도 위도 = 111,000m → 50m = 0.000450°
    1도 경도 = 88,800m  → 50m = 0.000563°
  bbox: lat [37.41, 37.72], lon [126.74, 127.20]
  → 약 689행 × 817열 = 약 56만 셀 (전체 메모리 보관 가능)

[제공 함수]
  point_to_grid(lon, lat) -> str | None
  grid_to_bounds(grid_id) -> dict (min/max lat/lon)
  grid_centroid(grid_id) -> (lon, lat)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM = PROJECT_ROOT / "data" / "interim"

SEOUL_BBOX = {
    "min_lat": 37.41,
    "max_lat": 37.72,
    "min_lon": 126.74,
    "max_lon": 127.20,
}
CELL_SIZE_M = 50.0

# 서울 위도 기준 (37.5°): 1도 lat ≈ 111km, 1도 lon ≈ 88.8km (cos(37.5°) × 111.32km)
LAT_DEG_PER_METER = 1.0 / 111_000.0
LON_DEG_PER_METER = 1.0 / (111_320.0 * math.cos(math.radians(37.5)))

LAT_STEP = CELL_SIZE_M * LAT_DEG_PER_METER  # 약 0.0004505
LON_STEP = CELL_SIZE_M * LON_DEG_PER_METER  # 약 0.0005625

N_ROWS = math.ceil((SEOUL_BBOX["max_lat"] - SEOUL_BBOX["min_lat"]) / LAT_STEP)
N_COLS = math.ceil((SEOUL_BBOX["max_lon"] - SEOUL_BBOX["min_lon"]) / LON_STEP)


def point_to_grid_idx(lon: float, lat: float) -> tuple[int, int] | None:
    """좌표 → (row, col). 범위 밖이면 None."""
    if not (SEOUL_BBOX["min_lat"] <= lat <= SEOUL_BBOX["max_lat"]):
        return None
    if not (SEOUL_BBOX["min_lon"] <= lon <= SEOUL_BBOX["max_lon"]):
        return None
    # row: 북쪽이 0
    row = int((SEOUL_BBOX["max_lat"] - lat) / LAT_STEP)
    col = int((lon - SEOUL_BBOX["min_lon"]) / LON_STEP)
    # 경계 보정
    row = min(row, N_ROWS - 1)
    col = min(col, N_COLS - 1)
    return row, col


def make_grid_id(row: int, col: int) -> str:
    return f"G50-{row:05d}-{col:05d}"


def point_to_grid(lon: float, lat: float) -> str | None:
    idx = point_to_grid_idx(lon, lat)
    if idx is None:
        return None
    return make_grid_id(*idx)


def grid_idx_to_bounds(row: int, col: int) -> dict[str, float]:
    max_lat = SEOUL_BBOX["max_lat"] - row * LAT_STEP
    min_lat = max_lat - LAT_STEP
    min_lon = SEOUL_BBOX["min_lon"] + col * LON_STEP
    max_lon = min_lon + LON_STEP
    return {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}


def grid_idx_to_centroid(row: int, col: int) -> tuple[float, float]:
    b = grid_idx_to_bounds(row, col)
    return (b["min_lon"] + b["max_lon"]) / 2, (b["min_lat"] + b["max_lat"]) / 2


def parse_grid_id(grid_id: str) -> tuple[int, int]:
    # G50-00123-00456
    parts = grid_id.split("-")
    return int(parts[1]), int(parts[2])


# ─── 벡터화 버전 (DataFrame 처리용) ───
def points_to_grid_id(lon: pd.Series, lat: pd.Series) -> pd.Series:
    """벡터화: NaN/범위밖은 빈 문자열 반환."""
    mask = (
        lat.between(SEOUL_BBOX["min_lat"], SEOUL_BBOX["max_lat"])
        & lon.between(SEOUL_BBOX["min_lon"], SEOUL_BBOX["max_lon"])
    )
    rows = ((SEOUL_BBOX["max_lat"] - lat) / LAT_STEP).clip(0, N_ROWS - 1).astype(int)
    cols = ((lon - SEOUL_BBOX["min_lon"]) / LON_STEP).clip(0, N_COLS - 1).astype(int)
    ids = "G50-" + rows.astype(str).str.zfill(5) + "-" + cols.astype(str).str.zfill(5)
    return ids.where(mask, "")


def main() -> None:
    INTERIM.mkdir(parents=True, exist_ok=True)
    print("[grid spec]")
    print(f"  bbox: lat [{SEOUL_BBOX['min_lat']}, {SEOUL_BBOX['max_lat']}], "
          f"lon [{SEOUL_BBOX['min_lon']}, {SEOUL_BBOX['max_lon']}]")
    print(f"  cell: {CELL_SIZE_M}m × {CELL_SIZE_M}m")
    print(f"  step: lat={LAT_STEP:.6f}°, lon={LON_STEP:.6f}°")
    print(f"  size: {N_ROWS} rows × {N_COLS} cols = {N_ROWS * N_COLS:,} total cells")

    # 샘플 변환 검증
    samples = [
        ("강남역", 127.0276, 37.4979),
        ("서울시청", 126.9780, 37.5665),
        ("광화문", 126.9770, 37.5759),
        ("범위 밖(부산)", 129.0, 35.0),
    ]
    print("\n[sample conversions]")
    for name, lon, lat in samples:
        gid = point_to_grid(lon, lat)
        print(f"  {name} ({lon}, {lat}) → {gid}")

    # 노인시설/CCTV 점 데이터에 grid_id_50m 부여하여 별도 저장
    print("\n[apply to data]")

    # facilities
    fac_in = PROJECT_ROOT / "data" / "processed" / "elderly_facilities_geocoded_grid_seoul.csv"
    fac_out = PROJECT_ROOT / "data" / "processed" / "elderly_facilities_grid50.csv"
    if fac_in.exists():
        df = pd.read_csv(fac_in)
        df["grid_id_50m"] = points_to_grid_id(df["lon"], df["lat"])
        df.to_csv(fac_out, index=False, encoding="utf-8-sig")
        non_empty = (df["grid_id_50m"] != "").sum()
        unique_cells = df.loc[df["grid_id_50m"] != "", "grid_id_50m"].nunique()
        print(f"  facilities: {len(df)} rows → mapped={non_empty}, unique grids={unique_cells}")
        print(f"    saved: {fac_out.relative_to(PROJECT_ROOT)}")

    # CCTV
    cctv_in = INTERIM / "cctv_points_unified.csv"
    cctv_out = PROJECT_ROOT / "data" / "processed" / "cctv_points_grid50.csv"
    if cctv_in.exists():
        df = pd.read_csv(cctv_in)
        df["grid_id_50m"] = points_to_grid_id(df["lon"], df["lat"])
        df.to_csv(cctv_out, index=False, encoding="utf-8-sig")
        non_empty = (df["grid_id_50m"] != "").sum()
        unique_cells = df.loc[df["grid_id_50m"] != "", "grid_id_50m"].nunique()
        print(f"  cctv: {len(df)} rows → mapped={non_empty}, unique grids={unique_cells}")
        print(f"    saved: {cctv_out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
