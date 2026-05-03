"""
09_cctv_coverage_features.py
각 노인복지시설 위치에서 반경 N미터 내 CCTV 수를 집계한다.
50m 격자 모델의 핵심 feature 생성.

[입력]
  data/processed/elderly_facilities_grid50.csv  (1,292개)
  data/processed/cctv_points_grid50.csv         (9,535개)

[출력]
  data/processed/facility_cctv_coverage.csv
    facility_id, ..., cctv_50m, cctv_100m, cctv_200m, cctv_300m,
    cctv_50m_crime_prev, cctv_100m_crime_prev, ...,
    nearest_cctv_dist_m, nearest_cctv_purpose

[알고리즘]
  haversine 정확도가 필요할 정도로 큰 거리는 아니므로 (반경 ≤ 300m),
  서울 위도 기준 단순 미터 변환 사용:
    Δm = sqrt((Δlat × 111000)² + (Δlon × 88800)²)
  9,535 CCTV × 1,292 facility = 1,232만 거리 계산. numpy로 1초 내.

[모델 설계 의도]
  - cctv_50m: 시설 코앞 사각지대 여부 (실종 즉시 추적 가능)
  - cctv_100m: 일상 동선 커버리지
  - cctv_200~300m: 1차 수색 반경
  - 목적별 분리: 방범(crime_prevention)이 노인 실종에 가장 직접적
                주정차(parking_enforcement)는 도로 위주라 보조 지표
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

FAC_PATH = PROCESSED / "elderly_facilities_grid50.csv"
CCTV_PATH = PROCESSED / "cctv_points_grid50.csv"
OUT_PATH = PROCESSED / "facility_cctv_coverage.csv"

RADII_M = [50, 100, 200, 300, 500]
LAT_M_PER_DEG = 111_000.0
LON_M_PER_DEG = 88_800.0  # cos(37.5°) × 111,320


def main() -> None:
    fac = pd.read_csv(FAC_PATH)
    cctv = pd.read_csv(CCTV_PATH)

    # 좌표 NaN 제거
    fac = fac.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    cctv = cctv.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    print(f"[input] facilities={len(fac)}, cctv={len(cctv)}")

    fac_lat = fac["lat"].to_numpy()
    fac_lon = fac["lon"].to_numpy()
    cctv_lat = cctv["lat"].to_numpy()
    cctv_lon = cctv["lon"].to_numpy()
    cctv_purpose = cctv["purpose_main"].to_numpy()

    # 결과 컬럼 초기화
    result = fac.copy()
    for r in RADII_M:
        result[f"cctv_{r}m"] = 0
        result[f"cctv_{r}m_crime_prev"] = 0
        result[f"cctv_{r}m_parking"] = 0
    result["nearest_cctv_dist_m"] = np.nan
    result["nearest_cctv_purpose"] = ""

    # 청크 단위 처리 (메모리 절약). 시설 100개씩.
    CHUNK = 200
    print(f"[compute] chunked distance calc, chunk={CHUNK}")
    for start in range(0, len(fac), CHUNK):
        end = min(start + CHUNK, len(fac))
        # (chunk, n_cctv) shape
        dlat = (fac_lat[start:end, None] - cctv_lat[None, :]) * LAT_M_PER_DEG
        dlon = (fac_lon[start:end, None] - cctv_lon[None, :]) * LON_M_PER_DEG
        dist = np.sqrt(dlat * dlat + dlon * dlon)  # (chunk, n_cctv) meters

        # 반경별 카운트 (전체 / crime_prev / parking)
        is_crime = cctv_purpose == "crime_prevention"
        is_parking = cctv_purpose == "parking_enforcement"
        for r in RADII_M:
            within = dist <= r
            result.loc[start:end - 1, f"cctv_{r}m"] = within.sum(axis=1)
            result.loc[start:end - 1, f"cctv_{r}m_crime_prev"] = (within & is_crime).sum(axis=1)
            result.loc[start:end - 1, f"cctv_{r}m_parking"] = (within & is_parking).sum(axis=1)

        # 가장 가까운 CCTV
        nearest_idx = dist.argmin(axis=1)
        nearest_dist = dist[np.arange(end - start), nearest_idx]
        result.loc[start:end - 1, "nearest_cctv_dist_m"] = nearest_dist
        result.loc[start:end - 1, "nearest_cctv_purpose"] = cctv_purpose[nearest_idx]

    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    # ─── 요약 통계 ───
    print("\n" + "=" * 70)
    print(f"✅ saved: {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"   rows: {len(result)}")

    print("\n[반경별 CCTV 카운트 분포 (시설 기준)]")
    summary = result[[f"cctv_{r}m" for r in RADII_M]].describe().T
    print(summary[["mean", "50%", "min", "max"]].round(1).to_string())

    print("\n[사각지대 시설 — 100m 내 CCTV=0]")
    blind = result[result["cctv_100m"] == 0]
    print(f"  count: {len(blind)} / {len(result)} ({len(blind)/len(result):.1%})")
    if len(blind) > 0:
        print("  by category_main:")
        print(blind["category_main"].value_counts().to_string())
        print("  by gu (top 10):")
        print(blind["gu_name"].value_counts().head(10).to_string())

    print("\n[방범 CCTV 사각지대 — 200m 내 crime_prev=0]")
    crime_blind = result[result["cctv_200m_crime_prev"] == 0]
    print(f"  count: {len(crime_blind)} ({len(crime_blind)/len(result):.1%})")

    print("\n[가장 가까운 CCTV 거리]")
    print(result["nearest_cctv_dist_m"].describe().round(1).to_string())

    print("\n[모델 인사이트 후보 — 위험도 상위 10개 시설]")
    # 단순 점수: nearest 거리 + (200m 내 방범 부족)
    scored = result.copy()
    scored["risk_score"] = (
        scored["nearest_cctv_dist_m"].fillna(1000) / 100
        + (10 - scored["cctv_200m_crime_prev"].clip(0, 10))
    )
    top = scored.nlargest(10, "risk_score")[
        ["facility_id", "facility_name", "category_main", "gu_name",
         "nearest_cctv_dist_m", "cctv_100m", "cctv_200m_crime_prev", "risk_score"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
