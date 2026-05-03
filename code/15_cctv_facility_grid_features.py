"""
15_cctv_facility_grid_features.py
CCTV (가중치 0.08) 와 노인복지시설 (0.04) 의 50m 격자 밀도 + 200m kernel.

[입력]
  data/processed/cctv_points_grid50.csv          (9,535 CCTV 점)
  data/processed/elderly_facilities_grid50.csv   (1,292 시설 점)

[격자 feature]
  cctv_count_in_cell, cctv_count_within_100m, cctv_count_within_200m,
  cctv_count_within_500m, cctv_crime_prev_within_200m,
  facility_count_in_cell, facility_count_within_200m, facility_count_within_500m,
  facility_capacity_within_500m

[출력]
  data/processed/grid50_cctv_features.csv
  data/processed/grid50_facility_features.csv
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"

SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
LAT_STEP = 50.0 / 111_000.0
LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))
N_ROWS = math.ceil((SEOUL_BBOX["max_lat"] - SEOUL_BBOX["min_lat"]) / LAT_STEP)
N_COLS = math.ceil((SEOUL_BBOX["max_lon"] - SEOUL_BBOX["min_lon"]) / LON_STEP)
LAT_M = 111_000.0
LON_M = 88_800.0


def cell_center(grid_id: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rc = grid_id.str.extract(r"G50-(\d+)-(\d+)").astype(int)
    lat = SEOUL_BBOX["max_lat"] - (rc[0].to_numpy() + 0.5) * LAT_STEP
    lon = SEOUL_BBOX["min_lon"] + (rc[1].to_numpy() + 0.5) * LON_STEP
    return lon, lat


def radius_counts(
    cell_lon: np.ndarray, cell_lat: np.ndarray,
    pt_lon: np.ndarray, pt_lat: np.ndarray,
    radii_m: list[int], chunk: int = 1000,
    weights: np.ndarray | None = None,
    masks: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """각 격자에서 각 반경 내 점 수를 효율적으로 계산. masks={col_name: bool_array}."""
    n_cells = len(cell_lon)
    out: dict[str, np.ndarray] = {f"count_{r}m": np.zeros(n_cells, dtype=int) for r in radii_m}
    if weights is not None:
        for r in radii_m:
            out[f"weight_sum_{r}m"] = np.zeros(n_cells)
    if masks:
        for name, _ in masks.items():
            for r in radii_m:
                out[f"{name}_{r}m"] = np.zeros(n_cells, dtype=int)

    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        dlat = (cell_lat[start:end, None] - pt_lat[None, :]) * LAT_M
        dlon = (cell_lon[start:end, None] - pt_lon[None, :]) * LON_M
        d2 = dlat * dlat + dlon * dlon
        for r in radii_m:
            within = d2 <= r * r
            out[f"count_{r}m"][start:end] = within.sum(axis=1)
            if weights is not None:
                out[f"weight_sum_{r}m"][start:end] = (within * weights[None, :]).sum(axis=1)
            if masks:
                for name, mask in masks.items():
                    out[f"{name}_{r}m"][start:end] = (within & mask[None, :]).sum(axis=1)
    return out


def main() -> None:
    base = pd.read_csv(PROCESSED / "grid50_road_features.csv")[["grid_id_50m"]]
    cell_lon, cell_lat = cell_center(base["grid_id_50m"])

    # ─── CCTV ───
    print("[CCTV] loading and computing radius counts...")
    cctv = pd.read_csv(PROCESSED / "cctv_points_grid50.csv")
    cctv = cctv.dropna(subset=["lon", "lat"])
    cctv_lon = cctv["lon"].to_numpy()
    cctv_lat = cctv["lat"].to_numpy()

    masks = {
        "crime_prev": (cctv["purpose_main"] == "crime_prevention").to_numpy(),
        "school_zone": (cctv["purpose_main"] == "school_zone").to_numpy(),
        "parking": (cctv["purpose_main"] == "parking_enforcement").to_numpy(),
    }
    cctv_radii = radius_counts(
        cell_lon, cell_lat, cctv_lon, cctv_lat,
        radii_m=[50, 100, 200, 500],
        masks=masks,
    )
    # 격자 in-cell count
    in_cell_cctv = cctv.groupby("grid_id_50m").size().rename("cctv_count_in_cell")
    cctv_feat = base.merge(in_cell_cctv, on="grid_id_50m", how="left").fillna({"cctv_count_in_cell": 0})
    cctv_feat["cctv_count_in_cell"] = cctv_feat["cctv_count_in_cell"].astype(int)
    for r in [50, 100, 200, 500]:
        cctv_feat[f"cctv_count_within_{r}m"] = cctv_radii[f"count_{r}m"]
    cctv_feat["cctv_crime_prev_within_200m"] = cctv_radii["crime_prev_200m"]
    cctv_feat["cctv_school_zone_within_200m"] = cctv_radii["school_zone_200m"]
    cctv_feat["cctv_parking_within_200m"] = cctv_radii["parking_200m"]

    cctv_out = PROCESSED / "grid50_cctv_features.csv"
    cctv_feat.to_csv(cctv_out, index=False, encoding="utf-8-sig")
    print(f"   saved: {cctv_out.relative_to(PROJECT_ROOT)}")

    # ─── 노인복지시설 ───
    print("\n[facility] loading and computing radius counts...")
    fac = pd.read_csv(PROCESSED / "elderly_facilities_grid50.csv")
    fac = fac.dropna(subset=["lon", "lat"])
    cap = pd.to_numeric(fac["capacity"], errors="coerce").fillna(0).to_numpy()

    fac_radii = radius_counts(
        cell_lon, cell_lat,
        fac["lon"].to_numpy(), fac["lat"].to_numpy(),
        radii_m=[200, 500, 1000],
        weights=cap,
    )
    in_cell_fac = fac.groupby("grid_id_50m").size().rename("facility_count_in_cell")
    fac_feat = base.merge(in_cell_fac, on="grid_id_50m", how="left").fillna({"facility_count_in_cell": 0})
    fac_feat["facility_count_in_cell"] = fac_feat["facility_count_in_cell"].astype(int)
    for r in [200, 500, 1000]:
        fac_feat[f"facility_count_within_{r}m"] = fac_radii[f"count_{r}m"]
        fac_feat[f"facility_capacity_within_{r}m"] = fac_radii[f"weight_sum_{r}m"]

    fac_out = PROCESSED / "grid50_facility_features.csv"
    fac_feat.to_csv(fac_out, index=False, encoding="utf-8-sig")
    print(f"   saved: {fac_out.relative_to(PROJECT_ROOT)}")

    # ─── 검증 출력 ───
    print("\n" + "=" * 70)
    print("[CCTV feature 분포]")
    cols = ["cctv_count_in_cell", "cctv_count_within_100m", "cctv_count_within_200m",
            "cctv_count_within_500m", "cctv_crime_prev_within_200m"]
    print(cctv_feat[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())

    print("\n[facility feature 분포]")
    cols = ["facility_count_in_cell", "facility_count_within_200m",
            "facility_count_within_500m", "facility_capacity_within_500m"]
    print(fac_feat[cols].describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_string())

    # 사각지대 비율
    cctv_blind = (cctv_feat["cctv_count_within_200m"] == 0).sum()
    print(f"\n[CCTV 사각지대 격자 (200m=0)]: "
          f"{cctv_blind:,} / {len(cctv_feat):,} ({cctv_blind / len(cctv_feat):.1%})")
    fac_zero = (fac_feat["facility_count_within_500m"] == 0).sum()
    print(f"[시설 sparse 격자 (500m=0)]: "
          f"{fac_zero:,} ({fac_zero / len(fac_feat):.1%})")


if __name__ == "__main__":
    main()
