"""
14_rest_areas_grid_features.py
휴식지 점 데이터 → 50m 격자 단위 feature.

[입력]
  data/interim/rest_areas_unified.csv  (7,652 점)

[모델 의도]
  feature 가중치: 0.10
  의미: 격자 주변에 휴식지(공원·경로당·쉼터)가 있으면
       치매 노인이 멈춰 발견될 가능성 ↑

[격자 feature]
  rest_count_in_cell        (50m 격자 내 점 수)
  rest_count_within_200m    (200m 반경 점 수, kernel)
  rest_count_within_500m    (500m 반경 점 수)
  has_park_within_200m      (큰 휴식 공간)
  rest_kind_diversity       (몇 종류 휴식지 있는지: 0~4)

[출력]
  data/processed/grid50_rest_features.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM = PROJECT_ROOT / "data/interim"
PROCESSED = PROJECT_ROOT / "data/processed"

# 50m grid spec
SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
LAT_STEP = 50.0 / 111_000.0
LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))
N_ROWS = math.ceil((SEOUL_BBOX["max_lat"] - SEOUL_BBOX["min_lat"]) / LAT_STEP)
N_COLS = math.ceil((SEOUL_BBOX["max_lon"] - SEOUL_BBOX["min_lon"]) / LON_STEP)

LAT_M = 111_000.0
LON_M = 88_800.0


def points_to_grid_id(lon: pd.Series, lat: pd.Series) -> pd.Series:
    mask = (
        lat.between(SEOUL_BBOX["min_lat"], SEOUL_BBOX["max_lat"])
        & lon.between(SEOUL_BBOX["min_lon"], SEOUL_BBOX["max_lon"])
    )
    rows = ((SEOUL_BBOX["max_lat"] - lat) / LAT_STEP).clip(0, N_ROWS - 1).astype(int)
    cols = ((lon - SEOUL_BBOX["min_lon"]) / LON_STEP).clip(0, N_COLS - 1).astype(int)
    ids = "G50-" + rows.astype(str).str.zfill(5) + "-" + cols.astype(str).str.zfill(5)
    return ids.where(mask, "")


def main() -> None:
    rest = pd.read_csv(INTERIM / "rest_areas_unified.csv")
    rest = rest.dropna(subset=["lon", "lat"]).copy()
    rest["grid_id_50m"] = points_to_grid_id(rest["lon"], rest["lat"])
    rest = rest[rest["grid_id_50m"] != ""].reset_index(drop=True)
    print(f"[input] rest_areas: {len(rest):,} points in Seoul bbox")

    # 격자 단위 in-cell count
    in_cell = (
        rest.groupby("grid_id_50m")
        .agg(
            rest_count_in_cell=("rest_id", "size"),
            kinds_in_cell=("kind", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    in_cell["rest_kind_diversity"] = in_cell["kinds_in_cell"].apply(
        lambda s: len(s.split(","))
    )

    # 도로 격자 ID 기준으로 base 만들기 (도로 없는 격자엔 어차피 시설·CCTV 없음)
    base = pd.read_csv(PROCESSED / "grid50_road_features.csv")[["grid_id_50m"]]
    base = base.merge(in_cell, on="grid_id_50m", how="left").fillna(
        {"rest_count_in_cell": 0, "kinds_in_cell": "", "rest_kind_diversity": 0}
    )
    base["rest_count_in_cell"] = base["rest_count_in_cell"].astype(int)
    base["rest_kind_diversity"] = base["rest_kind_diversity"].astype(int)

    # 격자 → 좌표
    base[["row", "col"]] = base["grid_id_50m"].str.extract(r"G50-(\d+)-(\d+)").astype(int)
    base["cell_lat"] = SEOUL_BBOX["max_lat"] - (base["row"] + 0.5) * LAT_STEP
    base["cell_lon"] = SEOUL_BBOX["min_lon"] + (base["col"] + 0.5) * LON_STEP

    # 반경 기반 카운트 (200m, 500m) — 청크 처리
    print("[radius] computing 200m/500m point counts...")
    rest_lat = rest["lat"].to_numpy()
    rest_lon = rest["lon"].to_numpy()
    rest_kind = rest["kind"].to_numpy()
    is_park = (rest_kind == "park")

    cell_lat = base["cell_lat"].to_numpy()
    cell_lon = base["cell_lon"].to_numpy()
    n_cells = len(base)
    rest_200 = np.zeros(n_cells, dtype=int)
    rest_500 = np.zeros(n_cells, dtype=int)
    has_park_200 = np.zeros(n_cells, dtype=int)

    CHUNK = 1000
    for start in range(0, n_cells, CHUNK):
        end = min(start + CHUNK, n_cells)
        dlat = (cell_lat[start:end, None] - rest_lat[None, :]) * LAT_M
        dlon = (cell_lon[start:end, None] - rest_lon[None, :]) * LON_M
        d2 = dlat * dlat + dlon * dlon  # m^2
        within_200 = d2 <= 200 * 200
        within_500 = d2 <= 500 * 500
        rest_200[start:end] = within_200.sum(axis=1)
        rest_500[start:end] = within_500.sum(axis=1)
        has_park_200[start:end] = (within_200 & is_park[None, :]).any(axis=1).astype(int)

    base["rest_count_within_200m"] = rest_200
    base["rest_count_within_500m"] = rest_500
    base["has_park_within_200m"] = has_park_200

    out = base[
        [
            "grid_id_50m", "rest_count_in_cell", "rest_kind_diversity",
            "rest_count_within_200m", "rest_count_within_500m", "has_park_within_200m",
        ]
    ]
    out_path = PROCESSED / "grid50_rest_features.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ saved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"   rows: {len(out):,}")
    print("\n[reset feature 분포]")
    desc = out[
        ["rest_count_in_cell", "rest_count_within_200m",
         "rest_count_within_500m", "has_park_within_200m", "rest_kind_diversity"]
    ].describe(percentiles=[0.5, 0.9, 0.99]).round(2)
    print(desc.to_string())

    print(f"\n  has_park_within_200m TRUE 격자: {out['has_park_within_200m'].sum():,} "
          f"({out['has_park_within_200m'].mean():.1%})")
    print(f"  rest_count_within_500m=0 격자 (휴식지 사각): "
          f"{(out['rest_count_within_500m'] == 0).sum():,} "
          f"({(out['rest_count_within_500m'] == 0).mean():.1%})")


if __name__ == "__main__":
    main()
