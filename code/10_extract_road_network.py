"""
10_extract_road_network.py
OpenStreetMap에서 서울 보행 가능 도로망을 다운로드하고
50m 격자별 (1) 교차로 밀도, (2) 도로 복잡도 feature를 계산한다.

[Phase 1 Step 1+2]

[Feature 정의]
  교차로 밀도 (가중치 0.35):
    - 격자 내 node degree ≥ 3 인 노드 수 (한국 도로망 특성: T자/+자 교차)
    - 격자 200m 반경 내 평균 교차로 밀도 (kernel)

  도로 복잡도 (가중치 0.25):
    - 격자 내 edge 수 (도로 segment 개수)
    - 격자 내 막다른 길 (degree=1) 비율
    - 격자 내 평균 거리 (street_length_avg) — 짧을수록 미로 같음

[저장]
  data/interim/road_network_intersections.csv  (교차로 점)
  data/interim/road_network_edges.csv          (도로 segment 메타)
  data/processed/grid50_road_features.csv      (격자별 feature) ⭐ 모델 입력
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM = PROJECT_ROOT / "data" / "interim"
PROCESSED = PROJECT_ROOT / "data" / "processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# 50m grid spec (08_build_50m_grid.py 와 동일)
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


def download_seoul_road_network():
    """서울시 전체 보행 도로망을 OSM에서 다운로드. 5~10분 소요."""
    print("[osmnx] downloading Seoul walking network...")
    print("        (this can take 5-10 minutes the first time)")
    # 'walk' 네트워크: 보도, 일반 도로, 골목, 계단 포함. 고속도로 제외.
    # 치매 노인 배회 모델에는 walk가 가장 적합.
    G = ox.graph_from_place(
        "Seoul, South Korea",
        network_type="walk",
        simplify=True,
        retain_all=False,
    )
    print(f"        nodes={G.number_of_nodes():,}, edges={G.number_of_edges():,}")
    return G


def graph_to_dfs(G):
    """networkx 그래프 → 노드/엣지 DataFrame."""
    print("[convert] graph → GeoDataFrames")
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # nodes: x=lon, y=lat, street_count(=degree)
    nodes_df = pd.DataFrame(
        {
            "node_id": nodes_gdf.index.values,
            "lon": nodes_gdf["x"].values,
            "lat": nodes_gdf["y"].values,
            "degree": nodes_gdf["street_count"].astype(int).values,
        }
    )

    # edges: length(meters), highway type, geometry midpoint
    edges_df = pd.DataFrame(
        {
            "u": [e[0] for e in edges_gdf.index],
            "v": [e[1] for e in edges_gdf.index],
            "length_m": edges_gdf["length"].astype(float).values,
            "highway": edges_gdf["highway"].astype(str).values,
        }
    )
    # 엣지 중간점 좌표 (geometry centroid)
    cents = edges_gdf.geometry.centroid
    edges_df["mid_lon"] = cents.x.values
    edges_df["mid_lat"] = cents.y.values

    return nodes_df, edges_df


def compute_grid_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
    """노드/엣지 → 격자별 feature."""
    print("[features] computing per-grid features...")

    # 격자 ID 부여
    nodes_df = nodes_df.copy()
    nodes_df["grid_id_50m"] = points_to_grid_id(nodes_df["lon"], nodes_df["lat"])
    nodes_df = nodes_df[nodes_df["grid_id_50m"] != ""]

    edges_df = edges_df.copy()
    edges_df["grid_id_50m"] = points_to_grid_id(edges_df["mid_lon"], edges_df["mid_lat"])
    edges_df = edges_df[edges_df["grid_id_50m"] != ""]

    # 격자별 노드 통계
    node_g = nodes_df.groupby("grid_id_50m")
    node_feats = pd.DataFrame(
        {
            "node_count": node_g.size(),
            "intersection_count": node_g.apply(lambda d: int((d["degree"] >= 3).sum())),
            "deadend_count": node_g.apply(lambda d: int((d["degree"] == 1).sum())),
            "max_degree": node_g["degree"].max(),
        }
    ).reset_index()

    # 격자별 엣지 통계
    edge_g = edges_df.groupby("grid_id_50m")
    edge_feats = pd.DataFrame(
        {
            "edge_count": edge_g.size(),
            "edge_length_total_m": edge_g["length_m"].sum(),
            "edge_length_avg_m": edge_g["length_m"].mean(),
            "edge_length_min_m": edge_g["length_m"].min(),
        }
    ).reset_index()

    # merge
    feats = node_feats.merge(edge_feats, on="grid_id_50m", how="outer").fillna(0)

    # 도로 복잡도 점수 (raw, normalize는 최종 모델 단계에서)
    # 직관: 교차로↑ + 막다른길↑ + 짧은 segment↑ + 다양한 degree↑
    feats["road_complexity_raw"] = (
        feats["intersection_count"] * 1.0
        + feats["deadend_count"] * 0.7
        + feats["edge_count"] * 0.2
        + (1.0 / (feats["edge_length_avg_m"].replace(0, 1e9) / 50)).clip(0, 5)  # 평균 segment가 짧을수록 ↑
    )
    return feats


def kernel_density_200m(feats: pd.DataFrame) -> pd.DataFrame:
    """200m 반경 평균 (배회 시 도달 가능 영역 단위 평탄화)."""
    print("[kernel] 200m radius smoothing...")
    # grid_id → (row, col) → 좌표
    rc = feats["grid_id_50m"].str.extract(r"G50-(\d+)-(\d+)").astype(int)
    feats = feats.copy()
    feats["row"] = rc[0]
    feats["col"] = rc[1]

    # 200m = 약 4 cells (50m × 4)
    radius_cells = 4
    grid_lookup = feats.set_index(["row", "col"])

    intersection_kernel = []
    complexity_kernel = []
    for _, row in feats.iterrows():
        r, c = row["row"], row["col"]
        rs = range(max(0, r - radius_cells), r + radius_cells + 1)
        cs = range(max(0, c - radius_cells), c + radius_cells + 1)
        # 인접 격자 인덱스 추출
        keys = [(rr, cc) for rr in rs for cc in cs]
        present = [k for k in keys if k in grid_lookup.index]
        if not present:
            intersection_kernel.append(row["intersection_count"])
            complexity_kernel.append(row["road_complexity_raw"])
            continue
        sub = grid_lookup.loc[present]
        # sub는 단일 행이면 Series, 다중이면 DataFrame
        if isinstance(sub, pd.Series):
            intersection_kernel.append(float(sub["intersection_count"]))
            complexity_kernel.append(float(sub["road_complexity_raw"]))
        else:
            intersection_kernel.append(float(sub["intersection_count"].mean()))
            complexity_kernel.append(float(sub["road_complexity_raw"].mean()))

    feats["intersection_density_200m"] = intersection_kernel
    feats["road_complexity_200m"] = complexity_kernel
    feats = feats.drop(columns=["row", "col"])
    return feats


def main() -> None:
    G = download_seoul_road_network()
    nodes_df, edges_df = graph_to_dfs(G)

    # 원본 저장
    nodes_path = INTERIM / "road_network_nodes.csv"
    edges_path = INTERIM / "road_network_edges.csv"
    nodes_df.to_csv(nodes_path, index=False, encoding="utf-8-sig")
    edges_df.to_csv(edges_path, index=False, encoding="utf-8-sig")
    print(f"  saved {nodes_path.relative_to(PROJECT_ROOT)} ({len(nodes_df):,} rows)")
    print(f"  saved {edges_path.relative_to(PROJECT_ROOT)} ({len(edges_df):,} rows)")

    # 교차로만 별도 저장 (시각화·디버그용)
    intersections = nodes_df[nodes_df["degree"] >= 3].copy()
    intersections["grid_id_50m"] = points_to_grid_id(intersections["lon"], intersections["lat"])
    inter_path = INTERIM / "road_network_intersections.csv"
    intersections.to_csv(inter_path, index=False, encoding="utf-8-sig")
    print(f"  saved {inter_path.relative_to(PROJECT_ROOT)} "
          f"(intersections only, {len(intersections):,})")

    # 격자별 feature
    feats = compute_grid_features(nodes_df, edges_df)
    feats = kernel_density_200m(feats)

    out_path = PROCESSED / "grid50_road_features.csv"
    feats.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 검증 출력
    print("\n" + "=" * 70)
    print(f"✅ saved: {out_path.relative_to(PROJECT_ROOT)}")
    print(f"   grids with road data: {len(feats):,} / {N_ROWS * N_COLS:,} total cells")
    print("\n[feature 분포 — 격자 단위]")
    summary_cols = [
        "intersection_count", "deadend_count", "edge_count",
        "edge_length_avg_m", "road_complexity_raw",
        "intersection_density_200m", "road_complexity_200m",
    ]
    print(feats[summary_cols].describe(percentiles=[0.5, 0.9, 0.99]).round(2).to_string())

    print("\n[교차로 밀도 상위 격자 10개]")
    top = feats.nlargest(10, "intersection_density_200m")[
        ["grid_id_50m", "intersection_count", "intersection_density_200m",
         "road_complexity_200m", "edge_count"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
