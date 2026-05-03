"""
17_discovery_probability_model.py
Phase 1 마스터: 모든 feature 합쳐서 격자별 정적 발견확률 산출.

[가중치 (이전 대화 합의)]
  교차로 밀도        × 0.35  ← Bayat 실증
  도로 복잡도        × 0.25  ← 길을 잃기 쉬움
  용도지역(주택가)    × 0.15  ← 배회 목적지
  공원·쉼터 밀도      × 0.10  ← 휴식지
  CCTV 밀도          × 0.08  ← 발견 수단
  노인복지시설 밀도   × 0.04  ← 약한 신호
  KT 유동인구        × 0.03  ← 보조적 (없으면 65+ 인구로 대체)

[정규화]
  각 feature → 0~1 사이 (clipped percentile rank: P5~P99 → 0~1)
  → outlier에 모델이 휘둘리지 않도록

[입력]
  data/processed/grid50_road_features.csv      (intersection, complexity)
  data/processed/grid50_zoning.csv              (residential)
  data/processed/grid50_rest_features.csv       (rest)
  data/processed/grid50_cctv_features.csv       (cctv)
  data/processed/grid50_facility_features.csv   (facility)
  data/processed/grid50_population.csv          (population, KT 대체)

[출력]
  data/processed/grid50_master_features.csv     (모든 feature 통합)
  data/processed/grid50_discovery_probability.csv  ⭐ 최종 모델 출력
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"

WEIGHTS = {
    "intersection": 0.35,
    "road_complexity": 0.25,
    "residential": 0.15,
    "rest_area": 0.10,
    "cctv": 0.08,
    "facility": 0.04,
    "population_proxy": 0.03,  # KT 유동인구 자리 — 65+ 인구로 임시 대체
}


def normalize_pct(s: pd.Series, low: float = 0.05, high: float = 0.99) -> pd.Series:
    """percentile clipping + min-max → 0~1."""
    s = s.astype(float)
    lo = s.quantile(low)
    hi = s.quantile(high)
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def main() -> None:
    print("[load] all feature tables")
    road = pd.read_csv(PROCESSED / "grid50_road_features.csv")
    zoning = pd.read_csv(PROCESSED / "grid50_zoning.csv")
    rest = pd.read_csv(PROCESSED / "grid50_rest_features.csv")
    cctv = pd.read_csv(PROCESSED / "grid50_cctv_features.csv")
    fac = pd.read_csv(PROCESSED / "grid50_facility_features.csv")
    pop = pd.read_csv(PROCESSED / "grid50_population.csv")

    # ─── merge ───
    m = road[[
        "grid_id_50m", "intersection_count", "deadend_count", "edge_count",
        "edge_length_avg_m", "road_complexity_raw",
        "intersection_density_200m", "road_complexity_200m",
    ]]
    m = m.merge(
        zoning[[
            "grid_id_50m", "zone_main_category", "is_residential",
            "is_residential_low", "residential_low_score_200m", "residential_score_200m",
        ]],
        on="grid_id_50m", how="left",
    )
    m = m.merge(
        rest[["grid_id_50m", "rest_count_in_cell", "rest_count_within_200m",
              "rest_count_within_500m", "has_park_within_200m", "rest_kind_diversity"]],
        on="grid_id_50m", how="left",
    )
    m = m.merge(
        cctv[["grid_id_50m", "cctv_count_in_cell", "cctv_count_within_100m",
              "cctv_count_within_200m", "cctv_count_within_500m",
              "cctv_crime_prev_within_200m"]],
        on="grid_id_50m", how="left",
    )
    m = m.merge(
        fac[["grid_id_50m", "facility_count_in_cell", "facility_count_within_200m",
             "facility_count_within_500m", "facility_capacity_within_500m"]],
        on="grid_id_50m", how="left",
    )
    m = m.merge(
        pop[["grid_id_50m", "gu_name", "pop_65plus_per_grid", "pop_75plus_per_grid",
             "pop_65plus_gu", "pct_elderly_75_gu"]],
        on="grid_id_50m", how="left",
    )
    m = m.fillna(0)
    print(f"  merged: {len(m):,} grids × {len(m.columns)} cols")

    master_path = PROCESSED / "grid50_master_features.csv"
    m.to_csv(master_path, index=False, encoding="utf-8-sig")
    print(f"  saved master: {master_path.relative_to(PROJECT_ROOT)}")

    # ─── 7개 feature 정규화 ───
    print("\n[normalize] percentile clipping → 0~1")
    f = pd.DataFrame({"grid_id_50m": m["grid_id_50m"]})
    f["norm_intersection"] = normalize_pct(m["intersection_density_200m"])
    f["norm_road_complexity"] = normalize_pct(m["road_complexity_200m"])
    # residential: 저밀주거 우선, 그다음 일반 주거
    f["norm_residential"] = (
        m["residential_low_score_200m"] * 0.7 + m["residential_score_200m"] * 0.3
    ).clip(0, 1)
    f["norm_rest_area"] = normalize_pct(m["rest_count_within_200m"])
    f["norm_cctv"] = normalize_pct(m["cctv_count_within_200m"])
    f["norm_facility"] = normalize_pct(m["facility_count_within_500m"])
    f["norm_population_proxy"] = normalize_pct(m["pop_75plus_per_grid"])

    # ─── 가중합 ───
    f["discovery_probability"] = (
        f["norm_intersection"] * WEIGHTS["intersection"]
        + f["norm_road_complexity"] * WEIGHTS["road_complexity"]
        + f["norm_residential"] * WEIGHTS["residential"]
        + f["norm_rest_area"] * WEIGHTS["rest_area"]
        + f["norm_cctv"] * WEIGHTS["cctv"]
        + f["norm_facility"] * WEIGHTS["facility"]
        + f["norm_population_proxy"] * WEIGHTS["population_proxy"]
    )

    # 0~1 범위에 percentile rank 형태로 변환 (시각화용)
    f["discovery_pct_rank"] = f["discovery_probability"].rank(pct=True).round(4)

    f["gu_name"] = m["gu_name"]

    # weighted contribution도 컬럼으로 보관 (해석용)
    for k, w in WEIGHTS.items():
        col = f"norm_{k.replace('road_complexity', 'road_complexity').replace('rest_area', 'rest_area')}"
        contrib_col = f"contrib_{k}"
        if col in f.columns:
            f[contrib_col] = (f[col] * w).round(4)

    out_path = PROCESSED / "grid50_discovery_probability.csv"
    f.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {out_path.relative_to(PROJECT_ROOT)}")

    # ─── 검증 ───
    print(f"\n[discovery_probability 분포]")
    print(f.discovery_probability.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3).to_string())

    print(f"\n[가중치별 평균 기여도]")
    for k in WEIGHTS:
        col = f"contrib_{k}"
        if col in f.columns:
            print(f"  {k:20s}: mean contrib = {f[col].mean():.4f}  (max weight = {WEIGHTS[k]})")

    print(f"\n[발견확률 상위 10개 격자]")
    top = f.nlargest(10, "discovery_probability")[
        ["grid_id_50m", "gu_name", "discovery_probability",
         "norm_intersection", "norm_road_complexity", "norm_residential",
         "norm_cctv", "norm_facility"]
    ]
    print(top.round(3).to_string(index=False))

    print(f"\n[자치구별 평균 발견확률]")
    by_gu = f.groupby("gu_name")["discovery_probability"].agg(["mean", "max", "count"])
    by_gu = by_gu.sort_values("mean", ascending=False).round(3)
    print(by_gu.to_string())


if __name__ == "__main__":
    main()
