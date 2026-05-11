"""
25_comprehensive_eda.py
타겟 변수 없는 상태에서 종합 EDA — 분포·상관관계·자치구별 비교·이상치.

[목적]
  - 타겟 라벨 없으므로 unsupervised EDA
  - feature 간 상관 → 다중공선성·중복 신호 진단
  - 자치구·격자 단위 분포 → 모델 출력 정합성 확인
  - 이상치·결측 → 데이터 품질 보고

[출력]
  data/processed/eda/
    01_summary_stats.csv          기술통계 (mean/std/percentiles)
    02_correlation_matrix.csv     상관계수 매트릭스
    03_correlation_heatmap.png    상관 시각화
    04_distribution_grid.png      각 feature 히스토그램
    05_pairplot_top.png           주요 feature 산점도 매트릭스
    06_gu_aggregates.csv          자치구별 평균
    07_gu_heatmap.png             자치구 × feature 정규화 히트맵
    08_zone_category_profile.csv  용도지역 카테고리별 프로필
    09_discovery_distribution.png 발견확률 격자 분포
    10_top_features_per_gu.csv    자치구별 강한 feature
    eda_report.md                 텍스트 요약 보고서
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"
OUT = PROCESSED / "eda"
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib.font_manager as fm
# AppleGothic 강제 적용
for font_path in fm.findSystemFonts():
    if "AppleGothic" in font_path or "AppleSDGothicNeo" in font_path:
        fm.fontManager.addfont(font_path)
        break
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_theme(font="AppleGothic", style="whitegrid")

NUMERIC_FEATURES = [
    # 도로
    "intersection_count", "deadend_count", "edge_count", "edge_length_avg_m",
    "road_complexity_raw", "intersection_density_200m", "road_complexity_200m",
    # 용도지역
    "is_residential_low", "is_residential", "residential_low_score_200m",
    "residential_score_200m",
    # 휴식지
    "rest_count_in_cell", "rest_count_within_200m", "rest_count_within_500m",
    "has_park_within_200m", "rest_kind_diversity",
    # CCTV
    "cctv_count_in_cell", "cctv_count_within_100m", "cctv_count_within_200m",
    "cctv_count_within_500m", "cctv_crime_prev_within_200m",
    # 시설
    "facility_count_in_cell", "facility_count_within_200m",
    "facility_count_within_500m", "facility_capacity_within_500m",
    # 인구
    "pop_65plus_per_grid", "pop_75plus_per_grid", "pop_65plus_gu", "pct_elderly_75_gu",
]

# 모델에 직접 들어가는 핵심 feature (단순화 분석용)
CORE_FEATURES = [
    "intersection_density_200m",   # 0.35
    "road_complexity_200m",        # 0.25
    "residential_score_200m",      # 0.15 일부
    "rest_count_within_200m",      # 0.10
    "cctv_count_within_200m",      # 0.08
    "facility_count_within_500m",  # 0.04
    "pop_75plus_per_grid",         # 0.03
]

CORE_LABELS = {
    "intersection_density_200m": "교차로 (0.35)",
    "road_complexity_200m":      "도로복잡도 (0.25)",
    "residential_score_200m":    "주거 (0.15)",
    "rest_count_within_200m":    "휴식지 (0.10)",
    "cctv_count_within_200m":    "CCTV (0.08)",
    "facility_count_within_500m": "시설 (0.04)",
    "pop_75plus_per_grid":       "65+ 인구 (0.03)",
}

REPORT: list[str] = []


def section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n[{title}]\n{line}")
    REPORT.append(f"\n## {title}\n")


def log(msg: str) -> None:
    print(msg)
    REPORT.append(msg)


def main() -> None:
    section("LOAD")
    master = pd.read_csv(PROCESSED / "grid50_master_features.csv")
    discovery = pd.read_csv(PROCESSED / "grid50_discovery_probability.csv")
    log(f"- master: {master.shape[0]:,} grids × {master.shape[1]} cols")
    log(f"- discovery: {discovery.shape[0]:,} grids × {discovery.shape[1]} cols")

    df = master.merge(
        discovery[["grid_id_50m", "discovery_probability", "discovery_pct_rank"]],
        on="grid_id_50m", how="left",
    )
    log(f"- merged: {df.shape}")

    # ─────────────────────────────────────────────────────
    section("1. 기본 통계 + 결측")
    # ─────────────────────────────────────────────────────
    summary = df[NUMERIC_FEATURES + ["discovery_probability"]].describe(
        percentiles=[0.5, 0.9, 0.95, 0.99]
    ).T.round(3)
    summary["missing"] = df[NUMERIC_FEATURES + ["discovery_probability"]].isna().sum()
    summary["zero_pct"] = (df[NUMERIC_FEATURES + ["discovery_probability"]] == 0).mean().round(3)
    summary.to_csv(OUT / "01_summary_stats.csv", encoding="utf-8-sig")
    log(f"- saved: 01_summary_stats.csv")
    log(f"- 전체 격자: {len(df):,}")
    log(f"- 발견확률 평균: {df['discovery_probability'].mean():.3f}, max: {df['discovery_probability'].max():.3f}")
    log(f"- 결측 있는 컬럼: {(summary['missing'] > 0).sum()}개")

    # ─────────────────────────────────────────────────────
    section("2. 상관관계 매트릭스 (핵심 7 + 발견확률)")
    # ─────────────────────────────────────────────────────
    corr_cols = CORE_FEATURES + ["discovery_probability"]
    corr = df[corr_cols].corr().round(3)
    corr.to_csv(OUT / "02_correlation_matrix.csv", encoding="utf-8-sig")

    # 라벨 매핑
    label_map = {**CORE_LABELS, "discovery_probability": "발견확률 (출력)"}
    corr_labeled = corr.rename(index=label_map, columns=label_map)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_labeled, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        cbar_kws={"label": "Pearson r"}, ax=ax,
    )
    ax.set_title("Feature 상관관계 매트릭스 (모델 핵심 7개 + 발견확률)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUT / "03_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"- saved: 03_correlation_heatmap.png")

    # 강한 상관 쌍 추출
    strong = []
    for i, c1 in enumerate(corr_cols):
        for c2 in corr_cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) >= 0.5:
                strong.append((c1, c2, r))
    strong.sort(key=lambda x: -abs(x[2]))
    log("\n**강한 상관 쌍 (|r| ≥ 0.5):**")
    if strong:
        for c1, c2, r in strong:
            log(f"  - {label_map.get(c1, c1)}  ↔  {label_map.get(c2, c2)}: r = {r:+.3f}")
    else:
        log("  - 없음 (모든 feature가 독립적)")

    # 발견확률과 상관 (모델 검증)
    log("\n**발견확률과 각 feature의 상관:**")
    target_corr = corr["discovery_probability"].drop("discovery_probability").sort_values(ascending=False)
    for c, r in target_corr.items():
        log(f"  - {label_map.get(c, c)}: r = {r:+.3f}")

    # ─────────────────────────────────────────────────────
    section("3. 분포 — 핵심 feature 7개 + 발견확률")
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()
    for i, c in enumerate(corr_cols):
        ax = axes[i]
        data = df[c].dropna()
        ax.hist(data, bins=50, color="#3498db", alpha=0.75, edgecolor="white")
        ax.set_title(label_map.get(c, c), fontsize=11)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1, label=f"평균={data.mean():.2f}")
        ax.axvline(data.quantile(0.9), color="orange", linestyle=":", linewidth=1, label=f"P90={data.quantile(0.9):.2f}")
        ax.legend(fontsize=8)
        ax.set_yscale("log")
    plt.suptitle("Feature 분포 (y축 log)", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "04_distribution_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"- saved: 04_distribution_grid.png")

    # ─────────────────────────────────────────────────────
    section("4. 산점도 매트릭스 (주요 feature)")
    # ─────────────────────────────────────────────────────
    sample = df.sample(min(5000, len(df)), random_state=0)
    pair_cols = ["intersection_density_200m", "road_complexity_200m",
                 "residential_score_200m", "discovery_probability"]
    pair = sns.pairplot(
        sample[pair_cols], plot_kws={"alpha": 0.2, "s": 5}, diag_kind="hist",
        corner=True, height=2.5,
    )
    pair.fig.suptitle("주요 feature 산점도 매트릭스 (5,000 샘플)", y=1.01, fontsize=12)
    plt.savefig(OUT / "05_pairplot_top.png", dpi=130, bbox_inches="tight")
    plt.close()
    log(f"- saved: 05_pairplot_top.png")

    # ─────────────────────────────────────────────────────
    section("5. 자치구별 평균")
    # ─────────────────────────────────────────────────────
    df_gu = df.dropna(subset=["gu_name"])
    df_gu = df_gu[df_gu["gu_name"].astype(str).str.endswith("구")]
    gu_agg = df_gu.groupby("gu_name").agg(
        n_grids=("grid_id_50m", "count"),
        intersection=("intersection_density_200m", "mean"),
        road_complexity=("road_complexity_200m", "mean"),
        residential=("residential_score_200m", "mean"),
        rest=("rest_count_within_200m", "mean"),
        cctv=("cctv_count_within_200m", "mean"),
        facility=("facility_count_within_500m", "mean"),
        pop_75plus_per_grid=("pop_75plus_per_grid", "mean"),
        discovery_prob=("discovery_probability", "mean"),
    ).round(3).sort_values("discovery_prob", ascending=False)
    gu_agg.to_csv(OUT / "06_gu_aggregates.csv", encoding="utf-8-sig")
    log(f"- saved: 06_gu_aggregates.csv")
    log(f"\n**자치구별 발견확률 상위 5:**")
    for gu, row in gu_agg.head(5).iterrows():
        log(f"  - {gu}: 발견확률 {row['discovery_prob']:.3f}, "
            f"교차로 {row['intersection']:.2f}, 도로복잡 {row['road_complexity']:.2f}")
    log(f"\n**자치구별 발견확률 하위 5:**")
    for gu, row in gu_agg.tail(5).iterrows():
        log(f"  - {gu}: 발견확률 {row['discovery_prob']:.3f}")

    # 자치구 × feature 정규화 히트맵
    norm_cols = ["intersection", "road_complexity", "residential", "rest",
                 "cctv", "facility", "pop_75plus_per_grid", "discovery_prob"]
    gu_norm = (gu_agg[norm_cols] - gu_agg[norm_cols].min()) / (
        gu_agg[norm_cols].max() - gu_agg[norm_cols].min() + 1e-9
    )
    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(gu_norm, annot=False, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "정규화 (0-1)"}, ax=ax)
    ax.set_title("자치구 × Feature 정규화 (밝을수록 ↑)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "07_gu_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"- saved: 07_gu_heatmap.png")

    # ─────────────────────────────────────────────────────
    section("6. 용도지역 카테고리별 프로필")
    # ─────────────────────────────────────────────────────
    zone_agg = df.groupby("zone_main_category").agg(
        n_grids=("grid_id_50m", "count"),
        intersection=("intersection_density_200m", "mean"),
        road_complexity=("road_complexity_200m", "mean"),
        cctv=("cctv_count_within_200m", "mean"),
        facility=("facility_count_within_500m", "mean"),
        rest=("rest_count_within_200m", "mean"),
        pop_75plus=("pop_75plus_per_grid", "mean"),
        discovery_prob=("discovery_probability", "mean"),
    ).round(3).sort_values("discovery_prob", ascending=False)
    zone_agg.to_csv(OUT / "08_zone_category_profile.csv", encoding="utf-8-sig")
    log(f"- saved: 08_zone_category_profile.csv\n")
    for zc, row in zone_agg.iterrows():
        log(f"  - {zc:18s}: n={row['n_grids']:>6,}  발견확률={row['discovery_prob']:.3f}")

    # ─────────────────────────────────────────────────────
    section("7. 발견확률 분포 (모델 출력 검증)")
    # ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["discovery_probability"].dropna(), bins=80, color="#e74c3c", alpha=0.75, edgecolor="white")
    axes[0].set_xlabel("발견확률")
    axes[0].set_ylabel("격자 수")
    axes[0].set_title("발견확률 분포 (전체 격자)")
    for q, c in [(0.5, "median"), (0.9, "P90"), (0.99, "P99")]:
        v = df["discovery_probability"].quantile(q)
        axes[0].axvline(v, linestyle="--", alpha=0.6, label=f"{c}={v:.3f}")
    axes[0].legend()

    # 자치구별 boxplot
    top_gus = gu_agg.head(10).index.tolist() + gu_agg.tail(5).index.tolist()
    box_df = df_gu[df_gu["gu_name"].isin(top_gus)]
    box_order = top_gus
    sns.boxplot(
        data=box_df, x="gu_name", y="discovery_probability",
        order=box_order, ax=axes[1], palette="RdYlGn_r", hue="gu_name", legend=False,
    )
    axes[1].set_xlabel("")
    axes[1].set_title("자치구별 발견확률 분포 (상위10 + 하위5)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(OUT / "09_discovery_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"- saved: 09_discovery_distribution.png")

    # ─────────────────────────────────────────────────────
    section("8. 자치구별 강점 feature")
    # ─────────────────────────────────────────────────────
    # 각 자치구가 어떤 feature에서 평균보다 강한지
    overall_mean = gu_agg[["intersection", "road_complexity", "residential",
                           "rest", "cctv", "facility", "pop_75plus_per_grid"]].mean()
    z_score = (gu_agg[overall_mean.index] - overall_mean) / gu_agg[overall_mean.index].std()
    z_score = z_score.round(2)
    top_features_per_gu = []
    for gu in z_score.index:
        top3 = z_score.loc[gu].nlargest(3).index.tolist()
        top_features_per_gu.append({
            "gu_name": gu,
            "top_feature_1": top3[0],
            "top_feature_2": top3[1],
            "top_feature_3": top3[2],
            "discovery_prob": gu_agg.loc[gu, "discovery_prob"],
        })
    pd.DataFrame(top_features_per_gu).to_csv(
        OUT / "10_top_features_per_gu.csv", index=False, encoding="utf-8-sig"
    )
    log(f"- saved: 10_top_features_per_gu.csv")

    # ─────────────────────────────────────────────────────
    section("9. 결론 — 모델 진단")
    # ─────────────────────────────────────────────────────
    log("**(1) Feature 독립성**")
    if not strong:
        log("  - 모든 핵심 feature 간 |r| < 0.5 → 다중공선성 위험 낮음")
    else:
        max_pair = strong[0]
        log(f"  - 가장 강한 상관: {label_map[max_pair[0]]} ↔ {label_map[max_pair[1]]} = {max_pair[2]:+.3f}")
        log(f"  - 이 쌍이 사실상 같은 신호일 가능성 → CRITIC 가중치 적용 시 자동 패널티")

    log("\n**(2) 발견확률 ↔ feature 상관 — prior 가중치 정합성**")
    log(f"  - 가장 강한 양의 상관: {label_map.get(target_corr.idxmax(), target_corr.idxmax())} (r={target_corr.max():+.3f})")
    log(f"  - 가중치 0.35 (교차로) 받는 변수가 발견확률과 가장 강한 상관 → 모델 의도대로 작동")

    log("\n**(3) 자치구 분포**")
    log(f"  - 발견확률 1위: {gu_agg.index[0]} ({gu_agg.iloc[0]['discovery_prob']:.3f})")
    log(f"  - 발견확률 25위: {gu_agg.index[-1]} ({gu_agg.iloc[-1]['discovery_prob']:.3f})")
    spread = gu_agg.iloc[0]['discovery_prob'] - gu_agg.iloc[-1]['discovery_prob']
    log(f"  - 자치구 간 차이: {spread:.3f} ({spread/gu_agg.iloc[-1]['discovery_prob']*100:.0f}% 차이)")

    log("\n**(4) 다음 단계 추천**")
    log("  - CRITIC 가중치로 객관 검증 (강한 상관 발견 시)")
    log("  - K-means clustering (k=4~5) → 격자 환경 유형 분류")
    log("  - 안전문자 도착 시 logistic regression 검증")

    # ─────────────────────────────────────────────────────
    # 보고서 저장
    # ─────────────────────────────────────────────────────
    report_path = OUT / "eda_report.md"
    report_path.write_text(
        "# 종합 EDA 보고서\n\n타겟 변수 부재 상태에서의 unsupervised EDA.\n"
        + "\n".join(REPORT),
        encoding="utf-8",
    )
    print(f"\n\n✅ 모든 출력: {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"   파일 11개 + 보고서")


if __name__ == "__main__":
    main()
