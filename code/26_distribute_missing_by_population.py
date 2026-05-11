"""
26_distribute_missing_by_population.py
구별 실종자 수를 격자별 노령인구에 비례하여 분배 (Dasymetric Mapping).

[입력]
  data/raw/missing/missing_count_by_gu.csv 또는 사용자 지정
    - 컬럼: gu_name, missing_count
  data/processed/grid50_population.csv
    - grid_id_50m, gu_name, pop_65plus_per_grid, pop_75plus_per_grid

[로직]
  1. 자치구 안 모든 격자의 노령인구 합 = gu_pop_total
  2. 각 격자의 share = pop_grid / gu_pop_total
  3. 격자별 추정 실종자 수 = gu_missing × share
  → 자치구 안에서 합산하면 원본 missing_count 와 정확히 일치

[합 보존 검증 포함]

[출력]
  data/processed/grid50_missing_distributed.csv
    - grid_id_50m, gu_name, pop_65plus_per_grid, pop_75plus_per_grid,
      gu_missing_count, missing_share, missing_count_estimated_65,
      missing_count_estimated_75
  data/interim/missing_validation_report.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data/raw/missing"
INTERIM = PROJECT_ROOT / "data/interim"
PROCESSED = PROJECT_ROOT / "data/processed"
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

DEFAULT_OUT = PROCESSED / "grid50_missing_distributed.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input", type=Path,
        help="구별 실종자 수 CSV (gu_name, missing_count). "
             "미지정 시 명령줄 또는 stdin 입력 모드",
    )
    p.add_argument("--weight-by", choices=["pop_65plus", "pop_75plus"], default="pop_65plus",
                   help="가중치 기준 인구 (기본: 65+)")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_missing_counts(args) -> pd.DataFrame:
    """구별 실종건수 로드. 파일 없으면 빈 DF 반환 (수동 입력 안내)."""
    if args.input and args.input.exists():
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                df = pd.read_csv(args.input, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        return df

    # 기본 경로 검색
    candidates = sorted(RAW.glob("*.csv"))
    if candidates:
        for enc in ("utf-8-sig", "utf-8", "cp949"):
            try:
                df = pd.read_csv(candidates[0], encoding=enc)
                print(f"[input] auto-detected: {candidates[0]}")
                return df
            except UnicodeDecodeError:
                continue

    print("ERROR: 실종건수 CSV 미지정.\n"
          "방법 1: --input data/raw/missing/missing_count_by_gu.csv\n"
          "방법 2: data/raw/missing/ 에 CSV 한 개 두면 자동 인식\n"
          "필요 컬럼: gu_name, missing_count\n",
          file=sys.stderr)
    sys.exit(1)


def normalize_gu_input(df: pd.DataFrame) -> pd.DataFrame:
    """입력 DataFrame에서 gu_name, missing_count 컬럼 표준화."""
    cols = {c.strip(): c for c in df.columns}
    # 가능한 컬럼명 매핑
    gu_aliases = ["gu_name", "gu", "자치구", "자치구명", "구", "구분", "구별",
                  "행정구", "city", "지역"]
    cnt_aliases = ["missing_count", "count", "n", "실종건수", "실종자수",
                   "건수", "발생수", "신고건수", "발생건수"]

    gu_col = next((cols[a] for a in gu_aliases if a in cols), None)
    cnt_col = next((cols[a] for a in cnt_aliases if a in cols), None)

    if gu_col is None or cnt_col is None:
        print(f"입력 CSV 컬럼: {list(df.columns)}", file=sys.stderr)
        print("ERROR: 'gu_name'(또는 한글 자치구) + 'missing_count'(또는 한글 건수) 필요",
              file=sys.stderr)
        sys.exit(1)

    out = df[[gu_col, cnt_col]].rename(columns={gu_col: "gu_name", cnt_col: "missing_count"})
    out["gu_name"] = out["gu_name"].astype(str).str.strip()
    # "강남구" 형태로 통일
    out["gu_name"] = out["gu_name"].apply(
        lambda s: s if s.endswith("구") else (s + "구" if s and s[-1] != "구" else s)
    )
    out["missing_count"] = pd.to_numeric(out["missing_count"], errors="coerce").fillna(0)
    out = out[out["gu_name"].str.endswith("구")]
    out = out.groupby("gu_name", as_index=False)["missing_count"].sum()
    return out


def main() -> None:
    args = parse_args()

    # 1) 구별 실종건수
    raw_missing = load_missing_counts(args)
    gu_missing = normalize_gu_input(raw_missing)
    print(f"\n[input] gu missing counts: {len(gu_missing)} 자치구")
    print(f"  total missing events: {int(gu_missing['missing_count'].sum())}")
    print(f"  range: {int(gu_missing['missing_count'].min())} ~ "
          f"{int(gu_missing['missing_count'].max())}")

    # 2) 격자 인구
    pop = pd.read_csv(PROCESSED / "grid50_population.csv")
    weight_col = "pop_65plus_per_grid" if args.weight_by == "pop_65plus" else "pop_75plus_per_grid"
    print(f"\n[load] grid50_population.csv: {len(pop):,} grids")
    print(f"[weight] using {weight_col} as proportion weight")

    # 3) merge
    df = pop.merge(gu_missing, on="gu_name", how="left")
    df["missing_count"] = df["missing_count"].fillna(0)

    matched_gus = df.loc[df["missing_count"] > 0, "gu_name"].nunique()
    print(f"  matched gus: {matched_gus} / {len(gu_missing)}")
    unmatched = set(gu_missing["gu_name"]) - set(df["gu_name"])
    if unmatched:
        print(f"  ⚠️  unmatched gus (격자에 없음): {sorted(unmatched)}")

    # 4) Dasymetric 분배
    # share = pop_i / sum(pop in gu)
    # estimated = gu_missing × share
    gu_pop_sum = df.groupby("gu_name")[weight_col].transform("sum")
    df["missing_share"] = np.where(
        gu_pop_sum > 0,
        df[weight_col] / gu_pop_sum,
        0.0,
    )

    # 65+ 기반 추정
    gu_pop_65sum = df.groupby("gu_name")["pop_65plus_per_grid"].transform("sum")
    df["share_65"] = np.where(gu_pop_65sum > 0, df["pop_65plus_per_grid"] / gu_pop_65sum, 0.0)
    df["missing_count_estimated_65"] = df["missing_count"] * df["share_65"]

    # 75+ 기반 추정 (보너스)
    gu_pop_75sum = df.groupby("gu_name")["pop_75plus_per_grid"].transform("sum")
    df["share_75"] = np.where(gu_pop_75sum > 0, df["pop_75plus_per_grid"] / gu_pop_75sum, 0.0)
    df["missing_count_estimated_75"] = df["missing_count"] * df["share_75"]

    # 5) 합 보존 검증
    print("\n" + "=" * 70)
    print("[합 보존 검증]")
    val = df.groupby("gu_name").agg(
        original=("missing_count", "first"),
        sum_estimated_65=("missing_count_estimated_65", "sum"),
        sum_estimated_75=("missing_count_estimated_75", "sum"),
        n_grids=("grid_id_50m", "count"),
    ).round(4)
    val["delta_65"] = (val["sum_estimated_65"] - val["original"]).round(6)
    val["delta_75"] = (val["sum_estimated_75"] - val["original"]).round(6)

    max_delta = val["delta_65"].abs().max()
    if max_delta < 1e-6:
        print(f"  ✅ 모든 자치구에서 합 보존 (max delta = {max_delta:.2e})")
    else:
        print(f"  ⚠️  최대 오차: {max_delta:.6f}")

    val_path = INTERIM / "missing_validation_report.csv"
    val.to_csv(val_path, encoding="utf-8-sig")
    print(f"  saved validation: {val_path.relative_to(PROJECT_ROOT)}")

    # 6) 출력 컬럼 정리
    out = df[[
        "grid_id_50m", "gu_name",
        "pop_65plus_per_grid", "pop_75plus_per_grid",
        "missing_count",                # 구 총 건수 (broadcast)
        "share_65", "share_75",          # 격자별 비중
        "missing_count_estimated_65",
        "missing_count_estimated_75",
    ]].copy()
    out = out.rename(columns={"missing_count": "gu_missing_count"})

    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {args.output.relative_to(PROJECT_ROOT)}")
    print(f"   rows: {len(out):,}")

    # 7) 요약 통계
    print(f"\n[격자별 추정 실종건수 분포]")
    print(out["missing_count_estimated_65"].describe(percentiles=[0.5, 0.9, 0.99]).round(4).to_string())
    print(f"\n  격자 합: {out['missing_count_estimated_65'].sum():.4f} "
          f"(원본 합: {gu_missing['missing_count'].sum()})")

    print(f"\n[자치구별 검증 (상위 5)]")
    print(val.head(5).to_string())


if __name__ == "__main__":
    main()
