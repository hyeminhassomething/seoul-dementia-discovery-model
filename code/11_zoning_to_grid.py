"""
11_zoning_to_grid.py
용도지역 SHP (EPSG:5174) → WGS84 변환 → 50m 격자 분류 feature 생성.

[입력]
  data/raw/zoning/shp파일/UPIS_C_UQ111.shp  (8,312 폴리곤)
    - DGM_NM: 용도지역명 (한글)
    - geometry: EPSG:5174 폴리곤

[카테고리 매핑 — Bayat 기반 발견확률 모델]
  residential_low   : 제1종전용, 제2종전용, 제1종일반 (배회 위험 ↑↑, 인적 적음)
  residential_mid   : 제2종일반, 제3종일반 (보통 주거지)
  residential_high  : 준주거 (상업 혼재)
  commercial        : 중심·일반·근린·유통 상업 (사람 많음, 발견 ↑)
  industrial        : 전용·일반·준공업
  green             : 보전·생산·자연 녹지
  other             : 그 외

[출력]
  data/interim/seoul_zoning_wgs84.geojson  (변환된 폴리곤, QGIS 등에서 확인용)
  data/processed/grid50_zoning.csv         (격자 단위 feature) ⭐
    grid_id_50m, zone_main_category, zone_main_dgm_nm,
    is_residential (0/1), is_residential_low (0/1),
    is_commercial (0/1), is_green (0/1), is_industrial (0/1),
    residential_low_score (0~1, 200m kernel)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_SHP = PROJECT_ROOT / "data/raw/zoning/shp파일/UPIS_C_UQ111.shp"
INTERIM = PROJECT_ROOT / "data/interim"
PROCESSED = PROJECT_ROOT / "data/processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# 50m 격자 spec
SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
LAT_STEP = 50.0 / 111_000.0
LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))
N_ROWS = math.ceil((SEOUL_BBOX["max_lat"] - SEOUL_BBOX["min_lat"]) / LAT_STEP)
N_COLS = math.ceil((SEOUL_BBOX["max_lon"] - SEOUL_BBOX["min_lon"]) / LON_STEP)


def make_grid_id(row: int, col: int) -> str:
    return f"G50-{row:05d}-{col:05d}"


def classify_zone(name: str) -> str:
    if not isinstance(name, str):
        return "other"
    s = name.replace(" ", "")
    # 주거 (세분)
    if "전용주거" in s and "1종" in s:
        return "residential_low"
    if "전용주거" in s and "2종" in s:
        return "residential_low"
    if "1종일반주거" in s or "제1종일반" in s:
        return "residential_low"
    if "2종일반주거" in s or "제2종일반" in s:
        return "residential_mid"
    if "3종일반주거" in s or "제3종일반" in s:
        return "residential_mid"
    if "준주거" in s:
        return "residential_high"
    if "일반주거" in s:  # 분류 없는 일반
        return "residential_mid"
    if "주거" in s:  # 잔여
        return "residential_mid"
    # 상업
    if "상업" in s:
        return "commercial"
    # 공업
    if "공업" in s:
        return "industrial"
    # 녹지
    if "녹지" in s:
        return "green"
    return "other"


def main() -> None:
    print(f"[load] {RAW_SHP.relative_to(PROJECT_ROOT)}")
    gdf = gpd.read_file(RAW_SHP, encoding="cp949")
    print(f"  rows={len(gdf):,}, crs={gdf.crs}")

    # WGS84 변환
    gdf_wgs = gdf.to_crs("EPSG:4326")
    print(f"  reprojected to WGS84, bounds: {gdf_wgs.total_bounds}")

    # 카테고리 분류
    gdf_wgs["zone_category"] = gdf_wgs["DGM_NM"].map(classify_zone)
    print("\n[zone_category 분포]")
    print(gdf_wgs["zone_category"].value_counts().to_string())

    # 변환본 저장 (디버그·시각화용, 가벼운 컬럼만)
    out_geo = INTERIM / "seoul_zoning_wgs84.geojson"
    gdf_wgs[["DGM_NM", "zone_category", "DGM_AR", "geometry"]].to_file(
        out_geo, driver="GeoJSON"
    )
    print(f"  saved {out_geo.relative_to(PROJECT_ROOT)}")

    # ─────── 격자 매핑 ───────
    # 각 격자의 중심점이 어느 폴리곤에 속하는지 spatial join
    # 561,000개 격자 × 8,312 폴리곤 → 너무 많으므로 도로 데이터 있는 121k 격자만 사용
    print("\n[grid join] computing grid centroids and spatial join...")
    road_feats = pd.read_csv(PROCESSED / "grid50_road_features.csv")
    grid_ids = road_feats["grid_id_50m"].tolist()

    # grid_id → centroid (lon, lat)
    rc = pd.DataFrame(grid_ids, columns=["grid_id_50m"])
    rc[["row", "col"]] = rc["grid_id_50m"].str.extract(r"G50-(\d+)-(\d+)").astype(int)
    rc["lat"] = SEOUL_BBOX["max_lat"] - (rc["row"] + 0.5) * LAT_STEP
    rc["lon"] = SEOUL_BBOX["min_lon"] + (rc["col"] + 0.5) * LON_STEP

    points_gdf = gpd.GeoDataFrame(
        rc[["grid_id_50m"]],
        geometry=[Point(lon, lat) for lon, lat in zip(rc["lon"], rc["lat"])],
        crs="EPSG:4326",
    )
    print(f"  grid points: {len(points_gdf):,}")

    # spatial join (point in polygon)
    print("  performing sjoin (this takes ~30~60s)...")
    joined = gpd.sjoin(
        points_gdf,
        gdf_wgs[["DGM_NM", "zone_category", "geometry"]],
        how="left",
        predicate="within",
    )
    # 한 점이 여러 폴리곤에 들어갈 수 있음 (overlap 등) → 가장 처음 매칭된 것만 유지
    joined = joined.drop_duplicates("grid_id_50m", keep="first")
    print(f"  matched grids: {(joined['DGM_NM'].notna()).sum():,} / {len(joined):,}")

    # feature 컬럼 부착
    feat = pd.DataFrame({
        "grid_id_50m": joined["grid_id_50m"].values,
        "zone_main_dgm_nm": joined["DGM_NM"].values,
        "zone_main_category": joined["zone_category"].fillna("unknown").values,
    })
    feat["is_residential_low"] = (feat["zone_main_category"] == "residential_low").astype(int)
    feat["is_residential_mid"] = (feat["zone_main_category"] == "residential_mid").astype(int)
    feat["is_residential_high"] = (feat["zone_main_category"] == "residential_high").astype(int)
    feat["is_residential"] = feat["zone_main_category"].isin(
        ["residential_low", "residential_mid", "residential_high"]).astype(int)
    feat["is_commercial"] = (feat["zone_main_category"] == "commercial").astype(int)
    feat["is_green"] = (feat["zone_main_category"] == "green").astype(int)
    feat["is_industrial"] = (feat["zone_main_category"] == "industrial").astype(int)

    # 200m kernel: 주변 4셀 평균 (배회 영역 단위)
    print("\n[kernel] 200m smoothing of residential_low score...")
    feat[["row", "col"]] = feat["grid_id_50m"].str.extract(r"G50-(\d+)-(\d+)").astype(int)
    grid_lookup = feat.set_index(["row", "col"])
    radius = 4
    res_low_kernel = []
    res_kernel = []
    for _, row in feat.iterrows():
        r, c = row["row"], row["col"]
        keys = [(rr, cc) for rr in range(max(0, r - radius), r + radius + 1)
                for cc in range(max(0, c - radius), c + radius + 1)]
        present = [k for k in keys if k in grid_lookup.index]
        if not present:
            res_low_kernel.append(row["is_residential_low"])
            res_kernel.append(row["is_residential"])
            continue
        sub = grid_lookup.loc[present]
        if isinstance(sub, pd.Series):
            res_low_kernel.append(float(sub["is_residential_low"]))
            res_kernel.append(float(sub["is_residential"]))
        else:
            res_low_kernel.append(float(sub["is_residential_low"].mean()))
            res_kernel.append(float(sub["is_residential"].mean()))

    feat["residential_low_score_200m"] = res_low_kernel
    feat["residential_score_200m"] = res_kernel
    feat = feat.drop(columns=["row", "col"])

    out_path = PROCESSED / "grid50_zoning.csv"
    feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"   rows: {len(feat):,}")

    print("\n[격자 카테고리 분포]")
    print(feat["zone_main_category"].value_counts().to_string())
    print(f"\nis_residential 비율: {feat['is_residential'].mean():.1%}")
    print(f"is_residential_low 비율: {feat['is_residential_low'].mean():.1%}")
    print(f"\n[residential_score_200m 분포]")
    print(feat["residential_score_200m"].describe(percentiles=[.5, .9]).round(3).to_string())


if __name__ == "__main__":
    main()
