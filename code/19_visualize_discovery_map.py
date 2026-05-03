"""
19_visualize_discovery_map.py
발견확률 격자 데이터 → folium HTML 지도 2개 생성.

(1) seoul_discovery_heatmap.html
    서울 전체 발견확률 분포 (상위 5,000개 격자만 그려서 가벼움)
    + 자치구 평균 마커

(2) incident_demo_map.html
    광화문 실종 6시간 시나리오:
    실종 지점(빨강) + 검색 반경(원) + 추천 격자 상위 50개 (사이즈/색)
    + 노인복지시설·CCTV·휴식지 레이어
"""
from __future__ import annotations

import math
from pathlib import Path

import folium
from folium.plugins import HeatMap, MarkerCluster
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"

SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
LAT_STEP = 50.0 / 111_000.0
LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))


def grid_to_center(grid_id: str) -> tuple[float, float]:
    parts = grid_id.split("-")
    r, c = int(parts[1]), int(parts[2])
    lat = SEOUL_BBOX["max_lat"] - (r + 0.5) * LAT_STEP
    lon = SEOUL_BBOX["min_lon"] + (c + 0.5) * LON_STEP
    return lon, lat


def color_for_prob(p: float) -> str:
    # 발견확률 0~1 → red(낮음) yellow orange green(높음)
    if p < 0.2:
        return "#5e4fa2"  # purple — 사각지대
    if p < 0.35:
        return "#3288bd"  # blue
    if p < 0.5:
        return "#66c2a5"  # green
    if p < 0.65:
        return "#fee08b"  # yellow
    if p < 0.8:
        return "#f46d43"  # orange
    return "#d53e4f"      # red — 발견 높음


# ───────── 1. 서울 전체 히트맵 ─────────
def make_seoul_heatmap() -> Path:
    df = pd.read_csv(PROCESSED / "grid50_discovery_probability.csv")
    print(f"[heatmap] grids: {len(df):,}")

    # 가벼운 시각화를 위해 상위 5,000 + 무작위 1,000 (분포 대표)
    top = df.nlargest(5000, "discovery_probability")
    sample = df.sample(min(1000, len(df) - 5000), random_state=0)
    show = pd.concat([top, sample], ignore_index=True).drop_duplicates("grid_id_50m")

    coords = [grid_to_center(g) for g in show["grid_id_50m"]]
    show["lon"] = [c[0] for c in coords]
    show["lat"] = [c[1] for c in coords]

    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron")

    # HeatMap layer
    heat_data = [
        [r["lat"], r["lon"], r["discovery_probability"]]
        for _, r in show.iterrows()
    ]
    HeatMap(heat_data, radius=8, blur=10, max_zoom=15,
            min_opacity=0.3, name="발견확률 히트맵").add_to(m)

    # 자치구별 평균 마커
    gu_avg = df.groupby("gu_name")["discovery_probability"].mean().sort_values(ascending=False)
    gu_centers = df.merge(
        df.groupby("gu_name").apply(lambda d: (
            (SEOUL_BBOX["max_lat"] - (d["grid_id_50m"].str.split("-").str[1].astype(int).mean() + 0.5) * LAT_STEP),
            (SEOUL_BBOX["min_lon"] + (d["grid_id_50m"].str.split("-").str[2].astype(int).mean() + 0.5) * LON_STEP),
        ), include_groups=False).reset_index().rename(columns={0: "ctr"}),
        on="gu_name", how="left"
    ).drop_duplicates("gu_name")[["gu_name", "ctr"]]
    gu_layer = folium.FeatureGroup(name="자치구 평균 발견확률", show=True)
    for _, r in gu_centers.iterrows():
        if pd.isna(r["ctr"]):
            continue
        lat, lon = r["ctr"]
        prob = gu_avg.get(r["gu_name"], 0)
        folium.CircleMarker(
            location=[lat, lon], radius=8 + prob * 30,
            fill=True, fill_color=color_for_prob(prob),
            color="white", weight=1, fill_opacity=0.7,
            tooltip=f"{r['gu_name']}: 평균 발견확률 {prob:.3f}",
        ).add_to(gu_layer)
    gu_layer.add_to(m)

    folium.LayerControl().add_to(m)
    out = PROCESSED / "seoul_discovery_heatmap.html"
    m.save(out)
    print(f"  saved: {out}")
    return out


# ───────── 2. 사건 데모 지도 ─────────
def make_incident_demo() -> Path:
    incident_lon, incident_lat, hours = 126.9770, 37.5759, 6.0
    radius_m = 2500

    cands = pd.read_csv(PROCESSED / "incident_demo_gwanghwamun_6h.csv")
    if len(cands) == 0:
        print("  no candidates loaded; rerun 18 first"); return PROCESSED / "incident_demo_map.html"
    coords = [grid_to_center(g) for g in cands["grid_id_50m"]]
    cands["lon"] = [c[0] for c in coords]
    cands["lat"] = [c[1] for c in coords]

    m = folium.Map(location=[incident_lat, incident_lon], zoom_start=14, tiles="cartodbpositron")

    # 실종 지점
    folium.Marker(
        location=[incident_lat, incident_lon],
        popup=f"<b>실종 신고 지점</b><br>광화문<br>경과 6시간",
        tooltip="실종 신고 지점 (광화문)",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # 검색 반경
    folium.Circle(
        location=[incident_lat, incident_lon], radius=radius_m,
        color="red", fill=False, weight=2, dash_array="5",
        popup=f"검색 반경 {radius_m}m (보수적 임계치, 6시간)",
    ).add_to(m)

    # 추천 격자 (상위 N)
    rec_layer = folium.FeatureGroup(name="추천 수색 격자 (상위 10)", show=True)
    for _, r in cands.iterrows():
        rank = int(r["search_rank"])
        prob = r["discovery_probability"]
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=14 - rank * 0.8,  # 1위가 가장 큼
            fill=True, fill_color=color_for_prob(prob),
            color="black", weight=1, fill_opacity=0.8,
            popup=(
                f"<b>#{rank} {r['grid_id_50m']}</b><br>"
                f"발견확률: {prob:.3f}<br>"
                f"거리: {r['distance_m']:.0f}m<br>"
                f"교차로: {r['norm_intersection']:.2f}<br>"
                f"도로복잡: {r['norm_road_complexity']:.2f}"
            ),
            tooltip=f"#{rank} 발견확률 {prob:.3f}",
        ).add_to(rec_layer)
    rec_layer.add_to(m)

    # CCTV / 시설 레이어 (반경 안 것만)
    LAT_M, LON_M = 111_000.0, 88_800.0

    cctv = pd.read_csv(PROCESSED / "cctv_points_grid50.csv").dropna(subset=["lon", "lat"])
    cctv_d = np.sqrt(((cctv["lat"] - incident_lat) * LAT_M) ** 2
                     + ((cctv["lon"] - incident_lon) * LON_M) ** 2)
    cctv = cctv[cctv_d <= radius_m]
    cctv_layer = folium.FeatureGroup(name=f"CCTV ({len(cctv)}개)", show=False)
    for _, r in cctv.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=2,
            color="blue", fill=True, fill_opacity=0.6,
            popup=f"CCTV / {r.get('purpose_main', '')}",
        ).add_to(cctv_layer)
    cctv_layer.add_to(m)

    fac = pd.read_csv(PROCESSED / "elderly_facilities_grid50.csv").dropna(subset=["lon", "lat"])
    fac_d = np.sqrt(((fac["lat"] - incident_lat) * LAT_M) ** 2
                    + ((fac["lon"] - incident_lon) * LON_M) ** 2)
    fac = fac[fac_d <= radius_m]
    fac_layer = folium.FeatureGroup(name=f"노인복지시설 ({len(fac)}개)", show=True)
    for _, r in fac.iterrows():
        folium.Marker(
            location=[r["lat"], r["lon"]],
            icon=folium.Icon(color="purple", icon="home", prefix="glyphicon"),
            popup=f"<b>{r['facility_name']}</b><br>{r.get('category_main', '')}",
        ).add_to(fac_layer)
    fac_layer.add_to(m)

    folium.LayerControl().add_to(m)
    out = PROCESSED / "incident_demo_map.html"
    m.save(out)
    print(f"  saved: {out}")
    return out


def fix_html_height(path: Path) -> None:
    """folium 저장본의 body를 viewport 100vh로 강제 — 흰화면 버그 방지."""
    html = path.read_text(encoding="utf-8")
    fix_css = """<style>
html, body { width: 100%; height: 100vh !important; min-height: 100vh !important; margin: 0; padding: 0; }
.folium-map { width: 100% !important; height: 100vh !important; min-height: 600px !important; position: absolute !important; top: 0; left: 0; }
</style>
</head>"""
    if "100vh" not in html:
        html = html.replace("</head>", fix_css, 1)
        path.write_text(html, encoding="utf-8")


def main() -> None:
    p1 = make_seoul_heatmap()
    fix_html_height(p1)
    p2 = make_incident_demo()
    fix_html_height(p2)
    print(f"\n✅ Open in browser:")
    print(f"   open {p1}")
    print(f"   open {p2}")


if __name__ == "__main__":
    main()
