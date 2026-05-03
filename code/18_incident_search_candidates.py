"""
18_incident_search_candidates.py
Phase 2 — 실종 사건이 들어오면 보수적 임계치 반경 안에서
발견확률 상위 격자를 우선 수색지점으로 추천.

[보수적 임계치 (Bayat + 한국 경찰 발견지점 분포 절충)]
  1시간   →  800m
  6시간   →  2,500m
  12시간  →  4,000m
  24시간  →  6,000m
  (선형 보간)

[사용 예시]
  python code/18_incident_search_candidates.py \
    --lon 126.9770 --lat 37.5759 --hours 6 --top 50 \
    --out data/processed/incident_demo_광화문_6h.csv

[기능]
  - 실종 신고 좌표 + 경과 시간 → 후보 격자 추출
  - 각 격자의 발견확률 sorted
  - 상위 N개 = 우선 수색지점

[데모용 시뮬레이션]
  --simulate-demo: 노인복지시설 30개를 가상 실종 지점으로 시뮬레이션
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"
INTERIM = PROJECT_ROOT / "data/interim"

# 50m grid spec
SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
LAT_STEP = 50.0 / 111_000.0
LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))
LAT_M = 111_000.0
LON_M = 88_800.0

# 보수적 임계치 (m)
THRESHOLDS = {1: 800, 6: 2500, 12: 4000, 24: 6000}


def threshold_radius_m(hours: float) -> float:
    """경과시간(h) → 배회 반경(m), 선형 보간."""
    keys = sorted(THRESHOLDS.keys())
    if hours <= keys[0]:
        return THRESHOLDS[keys[0]]
    if hours >= keys[-1]:
        return THRESHOLDS[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= hours < keys[i + 1]:
            k1, k2 = keys[i], keys[i + 1]
            v1, v2 = THRESHOLDS[k1], THRESHOLDS[k2]
            return v1 + (v2 - v1) * (hours - k1) / (k2 - k1)
    return THRESHOLDS[keys[-1]]


def cell_center(grid_id_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    rc = grid_id_series.str.extract(r"G50-(\d+)-(\d+)").astype(int)
    lat = SEOUL_BBOX["max_lat"] - (rc[0].to_numpy() + 0.5) * LAT_STEP
    lon = SEOUL_BBOX["min_lon"] + (rc[1].to_numpy() + 0.5) * LON_STEP
    return lon, lat


def find_candidates(
    incident_lon: float, incident_lat: float, hours_elapsed: float,
    discovery: pd.DataFrame, top_n: int = 50,
) -> pd.DataFrame:
    radius_m = threshold_radius_m(hours_elapsed)
    cell_lon, cell_lat = cell_center(discovery["grid_id_50m"])
    dlat = (cell_lat - incident_lat) * LAT_M
    dlon = (cell_lon - incident_lon) * LON_M
    dist_m = np.sqrt(dlat * dlat + dlon * dlon)

    df = discovery.copy()
    df["distance_m"] = dist_m
    df = df[df["distance_m"] <= radius_m].copy()

    # 발견확률 sort, top N
    df = df.sort_values("discovery_probability", ascending=False).head(top_n)
    df["search_rank"] = range(1, len(df) + 1)
    df["incident_lon"] = incident_lon
    df["incident_lat"] = incident_lat
    df["hours_elapsed"] = hours_elapsed
    df["search_radius_m"] = radius_m
    return df.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lon", type=float, help="실종 신고 경도")
    p.add_argument("--lat", type=float, help="실종 신고 위도")
    p.add_argument("--hours", type=float, default=6.0, help="경과 시간(h), 기본=6")
    p.add_argument("--top", type=int, default=50, help="상위 격자 수")
    p.add_argument("--out", type=Path, help="출력 CSV 경로")
    p.add_argument("--simulate-demo", action="store_true",
                   help="노인복지시설 30개로 시뮬레이션")
    args = p.parse_args()

    discovery = pd.read_csv(PROCESSED / "grid50_discovery_probability.csv")
    print(f"[load] discovery: {len(discovery):,} grids")

    if args.simulate_demo:
        # 시뮬레이션: 노인복지시설 30개 무작위 → 6시간 경과 시나리오
        fac = pd.read_csv(PROCESSED / "elderly_facilities_grid50.csv")
        fac = fac.dropna(subset=["lon", "lat"]).sample(30, random_state=42)
        all_results = []
        for _, row in fac.iterrows():
            cands = find_candidates(
                row["lon"], row["lat"], hours_elapsed=6.0,
                discovery=discovery, top_n=20,
            )
            cands["incident_facility_id"] = row["facility_id"]
            cands["incident_facility_name"] = row["facility_name"]
            cands["incident_gu"] = row["gu_name"]
            all_results.append(cands)
        out = pd.concat(all_results, ignore_index=True)
        out_path = args.out or (PROCESSED / "incident_simulation_demo.csv")
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n✅ saved: {out_path}")
        print(f"   simulated incidents: {fac.shape[0]}, total candidates: {len(out)}")

        # 시뮬레이션 통계
        print(f"\n[시뮬레이션 통계 — 6시간 경과]")
        print(f"  사건당 후보 격자: {out.groupby('incident_facility_id').size().describe().round(1).to_string()}")
        print(f"  평균 search_radius: {out['search_radius_m'].mean():.0f}m")
        print(f"  상위 1위 격자의 평균 발견확률: "
              f"{out[out['search_rank']==1]['discovery_probability'].mean():.3f}")
        return

    if args.lon is None or args.lat is None:
        print("--lon, --lat 필수 (또는 --simulate-demo)")
        return

    cands = find_candidates(
        args.lon, args.lat, args.hours,
        discovery=discovery, top_n=args.top,
    )
    out_path = args.out or (PROCESSED / "incident_search_result.csv")
    cands.to_csv(out_path, index=False, encoding="utf-8-sig")
    radius = threshold_radius_m(args.hours)
    print(f"\n✅ saved: {out_path}")
    print(f"   incident: ({args.lon}, {args.lat}), hours={args.hours}")
    print(f"   search radius: {radius:.0f}m")
    print(f"   candidates returned: {len(cands)}")
    print(f"\n[상위 10 격자]")
    show = cands.head(10)[
        ["search_rank", "grid_id_50m", "gu_name", "discovery_probability",
         "distance_m", "norm_intersection", "norm_road_complexity"]
    ]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
