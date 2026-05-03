"""
16_population_to_grid.py
주민등록인구 (자치구 × 성별 × 0~100세) → 자치구별 65+/75+ 인구 → 격자 broadcast.

[입력]
  data/raw/population/주민등록인구...csv
    - 행 0: 헤더 (자치구별, 성별, 합계, 0세, 1세, ..., 100세 이상)
    - 행 1+: 자치구="합계" + 성별="합계/남자/여자" 의 첫 3행
    - 이후: 25개 구 × 3 성별 (75행)

[출력]
  data/interim/population_by_gu.csv     (자치구별 인구 종합)
  data/processed/grid50_population.csv  (격자에 broadcast)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data/raw/population"
INTERIM = PROJECT_ROOT / "data/interim"
PROCESSED = PROJECT_ROOT / "data/processed"

_csvs = list(RAW.glob("*.csv"))
if not _csvs:
    raise SystemExit(f"No CSV found in {RAW}")
INPUT = _csvs[0]
OUT_GU = INTERIM / "population_by_gu.csv"
OUT_GRID = PROCESSED / "grid50_population.csv"


def main() -> None:
    print(f"[load] {INPUT.name}")
    raw = pd.read_csv(INPUT, encoding="utf-8-sig", header=None)
    print(f"  raw shape: {raw.shape}")

    # 행 0: 첫 데이터 헤더 (자치구별(1), 성별(1), 2026 1/4, ...)
    # 행 1: 진짜 컬럼명 (자치구별, 성별, 합계, 0세, 1세, ..., 99세, 100세 이상)
    # 행 2~: 데이터
    header_row = raw.iloc[1].tolist()
    df = raw.iloc[2:].copy()
    df.columns = header_row
    df = df.reset_index(drop=True)

    # "자치구별(1)" 컬럼이 두 번째 헤더로 들어왔을 수 있어 정리
    df = df.rename(columns={df.columns[0]: "gu", df.columns[1]: "gender"})

    # 합계행 / 비-구 행 제외
    seoul_gus = {"종로구","중구","용산구","성동구","광진구","동대문구","중랑구","성북구",
                 "강북구","도봉구","노원구","은평구","서대문구","마포구","양천구","강서구",
                 "구로구","금천구","영등포구","동작구","관악구","서초구","강남구","송파구","강동구"}
    df = df[df["gu"].isin(seoul_gus)].reset_index(drop=True)

    # 합계 성별만 (전체 인구)
    df_total = df[df["gender"] == "합계"].copy()
    print(f"  rows after filtering: {len(df_total)} (expect 25)")

    # 연령 컬럼: '0세','1세',...,'99세','100세 이상'
    age_cols = [c for c in df_total.columns if isinstance(c, str) and ("세" in c)]
    # 숫자 추출 (100세 이상 → 100)
    age_map = {}
    for c in age_cols:
        nums = "".join(ch for ch in c if ch.isdigit())
        if nums:
            age_map[c] = int(nums)

    # 콤마 제거 후 정수화
    for c in age_cols + ["합계"]:
        if c in df_total.columns:
            df_total[c] = (
                df_total[c]
                .astype(str)
                .str.replace(",", "")
                .replace({"-": "0", "nan": "0"})
                .astype(float)
                .astype(int)
            )

    # 자치구별 65+, 75+, 85+ 합계
    age_65plus_cols = [c for c, age in age_map.items() if age >= 65]
    age_75plus_cols = [c for c, age in age_map.items() if age >= 75]
    age_85plus_cols = [c for c, age in age_map.items() if age >= 85]

    out = pd.DataFrame()
    out["gu_name"] = df_total["gu"]
    out["pop_total"] = df_total["합계"]
    out["pop_65plus"] = df_total[age_65plus_cols].sum(axis=1)
    out["pop_75plus"] = df_total[age_75plus_cols].sum(axis=1)
    out["pop_85plus"] = df_total[age_85plus_cols].sum(axis=1)
    out["pct_elderly_65"] = (out["pop_65plus"] / out["pop_total"] * 100).round(2)
    out["pct_elderly_75"] = (out["pop_75plus"] / out["pop_total"] * 100).round(2)

    out = out.sort_values("pct_elderly_75", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_GU, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {OUT_GU.relative_to(PROJECT_ROOT)}")
    print("\n[자치구별 65+/75+ 인구 상위 10]")
    print(out.head(10).to_string(index=False))
    print(f"\n[서울 전체]")
    print(f"  총인구: {out['pop_total'].sum():,}")
    print(f"  65+:    {out['pop_65plus'].sum():,} ({out['pop_65plus'].sum() / out['pop_total'].sum() * 100:.1f}%)")
    print(f"  75+:    {out['pop_75plus'].sum():,} ({out['pop_75plus'].sum() / out['pop_total'].sum() * 100:.1f}%)")

    # ─── 격자 broadcast ───
    # 시설/CCTV 점 데이터로 각 격자의 gu_name 추정 (point-in-polygon이 좋지만 비용 큼;
    # 격자에 떨어진 시설/CCTV 의 dominant gu_name을 활용)
    print("\n[broadcast] assigning gu_name to each grid via nearest facility/cctv...")

    # 1) facility/cctv가 떨어진 격자는 그 점의 gu_name 사용
    fac = pd.read_csv(PROCESSED / "elderly_facilities_grid50.csv")[["grid_id_50m", "gu_name"]]
    cctv = pd.read_csv(PROCESSED / "cctv_points_grid50.csv")[["grid_id_50m", "gu_name"]]
    # rest_areas는 gu_name 없으니 skip

    pts = pd.concat([fac, cctv], ignore_index=True)
    pts = pts[pts["gu_name"].astype(str).str.endswith("구")]
    grid_gu = pts.groupby("grid_id_50m")["gu_name"].agg(lambda s: s.mode().iloc[0]).reset_index()
    print(f"  grids with direct gu match: {len(grid_gu):,}")

    # 2) base 격자에 gu_name 부여
    base = pd.read_csv(PROCESSED / "grid50_road_features.csv")[["grid_id_50m"]]
    base = base.merge(grid_gu, on="grid_id_50m", how="left")
    n_unmatched = base["gu_name"].isna().sum()
    print(f"  unmatched grids: {n_unmatched:,} ({n_unmatched / len(base):.1%})")

    # 3) 미매칭 격자는 가장 가까운 매칭 격자의 gu_name 빌리기
    if n_unmatched > 0:
        # 격자 ID → 좌표
        rc = base["grid_id_50m"].str.extract(r"G50-(\d+)-(\d+)").astype(int)
        SEOUL_BBOX = {"min_lat": 37.41, "max_lat": 37.72, "min_lon": 126.74, "max_lon": 127.20}
        LAT_STEP = 50.0 / 111_000.0
        LON_STEP = 50.0 / (111_320.0 * math.cos(math.radians(37.5)))
        base["_lat"] = SEOUL_BBOX["max_lat"] - (rc[0] + 0.5) * LAT_STEP
        base["_lon"] = SEOUL_BBOX["min_lon"] + (rc[1] + 0.5) * LON_STEP

        matched = base[base["gu_name"].notna()]
        unmatched = base[base["gu_name"].isna()]
        m_lat = matched["_lat"].to_numpy()
        m_lon = matched["_lon"].to_numpy()
        m_gu = matched["gu_name"].to_numpy()

        for idx in unmatched.index:
            lat = base.at[idx, "_lat"]
            lon = base.at[idx, "_lon"]
            d2 = (m_lat - lat) ** 2 + (m_lon - lon) ** 2
            base.at[idx, "gu_name"] = m_gu[d2.argmin()]
        base = base.drop(columns=["_lat", "_lon"])
        print(f"  fallback nearest-match applied to {n_unmatched:,} grids")

    # 4) 자치구별 인구 부착
    pop_dict = out.set_index("gu_name").to_dict("index")
    for col in ["pop_total", "pop_65plus", "pop_75plus", "pop_85plus",
                "pct_elderly_65", "pct_elderly_75"]:
        base[col + "_gu"] = base["gu_name"].map(lambda g: pop_dict.get(g, {}).get(col, 0))

    # 격자별 65+ 인구 broadcast (gu 안 격자 수로 균등 분배)
    gu_grid_count = base.groupby("gu_name").size()
    base["pop_65plus_per_grid"] = base.apply(
        lambda r: pop_dict.get(r["gu_name"], {}).get("pop_65plus", 0) / gu_grid_count.get(r["gu_name"], 1),
        axis=1,
    ).round(2)
    base["pop_75plus_per_grid"] = base.apply(
        lambda r: pop_dict.get(r["gu_name"], {}).get("pop_75plus", 0) / gu_grid_count.get(r["gu_name"], 1),
        axis=1,
    ).round(2)

    base.to_csv(OUT_GRID, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {OUT_GRID.relative_to(PROJECT_ROOT)}")
    print(f"   rows: {len(base):,}")
    print("\n[격자별 broadcast 분포]")
    print(base[["pop_65plus_gu", "pop_75plus_gu", "pop_65plus_per_grid",
                "pct_elderly_75_gu"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
