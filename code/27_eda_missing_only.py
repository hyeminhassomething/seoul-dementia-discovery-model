"""
27_eda_missing_only.py
실종 데이터 단독 EDA — 시계열 + 자치구별 + 인구 보정 발생률.

[입력]
  data/raw/missing/seoul_missing_by_month.csv    (375행, 25구 × 15개월)
  data/raw/missing/missing_count_by_gu.csv       (자치구 합계)
  data/interim/population_by_gu.csv              (인구 보정용)

[출력]
  data/processed/eda_missing/
    01_timeseries_total.png       전체 월별 추세
    02_timeseries_by_gu.png       자치구별 시계열 (heatmap)
    03_seasonality.png            계절성 (월별 평균)
    04_gu_count_bar.png           자치구별 절대 건수
    05_gu_rate_bar.png            자치구별 발생률 (per 10,000 65+)
    06_count_vs_rate.png          건수 vs 발생률 산점도
    07_top_bottom_gu.png          상하위 비교
    08_summary.md                 텍스트 요약
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data/raw/missing"
INTERIM = PROJECT_ROOT / "data/interim"
PROCESSED = PROJECT_ROOT / "data/processed"
OUT = PROCESSED / "eda_missing"
OUT.mkdir(parents=True, exist_ok=True)

# 한글 폰트
for fp in fm.findSystemFonts():
    if "AppleGothic" in fp or "AppleSDGothicNeo" in fp:
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(font="AppleGothic", style="whitegrid")

REPORT: list[str] = []


def section(t):
    print(f"\n{'=' * 60}\n[{t}]\n{'=' * 60}")
    REPORT.append(f"\n## {t}\n")


def log(s):
    print(s)
    REPORT.append(s)


def main():
    section("LOAD")
    monthly = pd.read_csv(RAW / "seoul_missing_by_month.csv")
    gu_total = pd.read_csv(RAW / "missing_count_by_gu.csv")
    pop = pd.read_csv(INTERIM / "population_by_gu.csv")

    log(f"- monthly: {len(monthly):,} 행 ({monthly['year_month'].min()} ~ {monthly['year_month'].max()})")
    log(f"- 자치구: {gu_total['gu_name'].nunique()}, 총 실종건수: {int(gu_total['missing_count'].sum())}")

    monthly["year_month_dt"] = pd.to_datetime(monthly["year_month"])
    monthly["year"] = monthly["year_month_dt"].dt.year
    monthly["month"] = monthly["year_month_dt"].dt.month

    # ───────── 1. 시계열 (전체) ─────────
    section("1. 전체 월별 추세")
    total_by_month = monthly.groupby("year_month_dt")["missing_count"].sum().sort_index()
    log(f"- 월평균: {total_by_month.mean():.1f}건")
    log(f"- 월최대: {total_by_month.max()}건 ({total_by_month.idxmax().strftime('%Y-%m')})")
    log(f"- 월최소: {total_by_month.min()}건 ({total_by_month.idxmin().strftime('%Y-%m')})")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(total_by_month.index, total_by_month.values, "o-", color="#e74c3c", linewidth=2, markersize=8)
    ax.fill_between(total_by_month.index, total_by_month.values, alpha=0.2, color="#e74c3c")
    ax.set_title("서울시 65+ 노인 실종 신고 — 월별 추세 (2023.09 ~ 2024.11)", fontsize=13)
    ax.set_xlabel("연-월")
    ax.set_ylabel("월별 실종 신고 건수")
    ax.grid(alpha=0.3)
    # 평균선
    ax.axhline(total_by_month.mean(), color="gray", linestyle="--", alpha=0.7,
               label=f"평균 {total_by_month.mean():.1f}건")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "01_timeseries_total.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("- saved: 01_timeseries_total.png")

    # ───────── 2. 자치구별 시계열 heatmap ─────────
    section("2. 자치구별 시계열 (heatmap)")
    pivot = monthly.pivot_table(
        index="gu_name", columns="year_month", values="missing_count", aggfunc="sum"
    ).fillna(0)
    # 발견확률 모델과 일관성 맞추기 위해 자치구별 합계 순으로 정렬
    pivot = pivot.loc[gu_total.sort_values("missing_count", ascending=False)["gu_name"]]

    fig, ax = plt.subplots(figsize=(15, 9))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3,
                cbar_kws={"label": "실종 건수"}, annot=True, fmt=".0f", ax=ax)
    ax.set_title("자치구 × 월별 실종 신고 건수 (정렬: 총합 내림차순)", fontsize=13)
    ax.set_xlabel("연-월")
    ax.set_ylabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "02_timeseries_by_gu.png", dpi=130, bbox_inches="tight")
    plt.close()
    log("- saved: 02_timeseries_by_gu.png")

    # ───────── 3. 계절성 (월 평균) ─────────
    section("3. 계절성 — 월별 평균")
    seasonal = monthly.groupby("month")["missing_count"].agg(["sum", "mean", "std"]).round(2)
    log("월별 평균 패턴 (2023.09 ~ 2024.11 기간 내):")
    for m, row in seasonal.iterrows():
        log(f"  - {m}월: 합 {int(row['sum'])}, 자치구당 평균 {row['mean']:.1f}건")

    fig, ax = plt.subplots(figsize=(11, 5))
    months = seasonal.index
    bars = ax.bar(months, seasonal["sum"], color="#3498db", alpha=0.8, edgecolor="white")
    overall_mean = seasonal["sum"].mean()
    ax.axhline(overall_mean, color="red", linestyle="--", alpha=0.6, label=f"평균 {overall_mean:.0f}")
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}월" for m in months])
    ax.set_xlabel("월")
    ax.set_ylabel("총 실종 신고 건수")
    ax.set_title("계절성 — 월별 실종 신고 (서울 25개 자치구 합계)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    # 막대 위에 값
    for bar, v in zip(bars, seasonal["sum"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{int(v)}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "03_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("- saved: 03_seasonality.png")

    # ───────── 4. 자치구 절대 건수 ─────────
    section("4. 자치구별 절대 건수")
    gu_sorted = gu_total.sort_values("missing_count", ascending=True)
    log("상위 5: " + ", ".join(f"{r['gu_name']}({int(r['missing_count'])})"
                              for _, r in gu_sorted.tail(5).iterrows()))
    log("하위 5: " + ", ".join(f"{r['gu_name']}({int(r['missing_count'])})"
                              for _, r in gu_sorted.head(5).iterrows()))

    fig, ax = plt.subplots(figsize=(10, 9))
    colors = ["#e74c3c" if v >= gu_sorted["missing_count"].quantile(0.8)
              else "#3498db" if v <= gu_sorted["missing_count"].quantile(0.2)
              else "#95a5a6" for v in gu_sorted["missing_count"]]
    ax.barh(gu_sorted["gu_name"], gu_sorted["missing_count"], color=colors)
    ax.set_xlabel("총 실종 신고 건수 (15개월)")
    ax.set_title("자치구별 65+ 노인 실종 신고 — 절대 건수", fontsize=13)
    for i, v in enumerate(gu_sorted["missing_count"]):
        ax.text(v + 0.5, i, f"{int(v)}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "04_gu_count_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("- saved: 04_gu_count_bar.png")

    # ───────── 5. 인구 보정 발생률 ─────────
    section("5. 인구 보정 발생률 (per 10,000 of 65+)")
    rate_df = gu_total.merge(pop[["gu_name", "pop_65plus"]], on="gu_name")
    rate_df["rate_per_10k"] = (rate_df["missing_count"] / rate_df["pop_65plus"] * 10000).round(3)
    rate_df = rate_df.sort_values("rate_per_10k", ascending=False)

    log("발생률 상위 5 (실제 위험 자치구):")
    for _, r in rate_df.head(5).iterrows():
        log(f"  - {r['gu_name']}: {r['rate_per_10k']:.2f}/10k "
            f"(건수 {int(r['missing_count'])}, 65+인구 {int(r['pop_65plus']):,})")
    log("\n발생률 하위 5:")
    for _, r in rate_df.tail(5).iterrows():
        log(f"  - {r['gu_name']}: {r['rate_per_10k']:.2f}/10k "
            f"(건수 {int(r['missing_count'])}, 65+인구 {int(r['pop_65plus']):,})")
    rate_df.to_csv(OUT / "gu_rate.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 9))
    rate_sorted = rate_df.sort_values("rate_per_10k", ascending=True)
    colors = ["#e74c3c" if v >= rate_sorted["rate_per_10k"].quantile(0.8)
              else "#3498db" if v <= rate_sorted["rate_per_10k"].quantile(0.2)
              else "#95a5a6" for v in rate_sorted["rate_per_10k"]]
    ax.barh(rate_sorted["gu_name"], rate_sorted["rate_per_10k"], color=colors)
    ax.set_xlabel("발생률 (10,000명당 실종 신고)")
    ax.set_title("자치구별 65+ 인구 보정 실종 발생률 — 진짜 위험도", fontsize=13)
    for i, v in enumerate(rate_sorted["rate_per_10k"]):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "05_gu_rate_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("- saved: 05_gu_rate_bar.png")

    # ───────── 6. 건수 vs 발생률 산점도 ─────────
    section("6. 건수 vs 발생률 — 인구 영향 확인")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(rate_df["missing_count"], rate_df["rate_per_10k"],
               s=rate_df["pop_65plus"] / 500, alpha=0.6, color="#9b59b6", edgecolors="black")
    for _, r in rate_df.iterrows():
        ax.annotate(r["gu_name"], (r["missing_count"], r["rate_per_10k"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax.set_xlabel("절대 실종 건수")
    ax.set_ylabel("발생률 (per 10k of 65+)")
    ax.set_title("건수 ↔ 발생률 비교 (점 크기 = 65+ 인구)", fontsize=13)
    # 상관계수
    corr = rate_df["missing_count"].corr(rate_df["rate_per_10k"])
    ax.text(0.02, 0.98, f"건수-발생률 상관 r = {corr:.3f}",
            transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "06_count_vs_rate.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"- 상관 r = {corr:.3f}")
    log("- saved: 06_count_vs_rate.png")

    # ───────── 7. 상위/하위 비교 ─────────
    section("7. 상하위 자치구 비교")
    top5_count = gu_total.nlargest(5, "missing_count")["gu_name"].tolist()
    top5_rate = rate_df.nlargest(5, "rate_per_10k")["gu_name"].tolist()
    log(f"건수 Top 5: {top5_count}")
    log(f"발생률 Top 5: {top5_rate}")
    overlap = set(top5_count) & set(top5_rate)
    log(f"공통: {sorted(overlap)} ({len(overlap)}개)")
    only_count = set(top5_count) - set(top5_rate)
    only_rate = set(top5_rate) - set(top5_count)
    if only_count:
        log(f"  - 건수만 ↑ (인구 많아서): {sorted(only_count)}")
    if only_rate:
        log(f"  - 발생률만 ↑ (진짜 위험): {sorted(only_rate)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sub1 = gu_total.nlargest(5, "missing_count").sort_values("missing_count")
    axes[0].barh(sub1["gu_name"], sub1["missing_count"], color="#e74c3c")
    axes[0].set_title("절대 건수 Top 5", fontsize=12)
    axes[0].set_xlabel("실종 신고 건수")
    for i, v in enumerate(sub1["missing_count"]):
        axes[0].text(v + 0.5, i, f"{int(v)}", va="center", fontsize=9)

    sub2 = rate_df.nlargest(5, "rate_per_10k").sort_values("rate_per_10k")
    axes[1].barh(sub2["gu_name"], sub2["rate_per_10k"], color="#9b59b6")
    axes[1].set_title("발생률 Top 5 (인구 보정)", fontsize=12)
    axes[1].set_xlabel("10k 노인당 실종건수")
    for i, v in enumerate(sub2["rate_per_10k"]):
        axes[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT / "07_top_bottom_gu.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("- saved: 07_top_bottom_gu.png")

    # ───────── 결론 ─────────
    section("결론 — 핵심 발견")
    cv = gu_total["missing_count"].std() / gu_total["missing_count"].mean()
    log(f"\n**1. 자치구 간 불평등도**")
    log(f"   - 변동계수 CV = {cv:.2f} (1보다 작으면 비교적 균등)")
    log(f"   - 최대(노원 56) ÷ 최소(서초 11) = 5.1배")

    log(f"\n**2. 인구 보정 효과**")
    log(f"   - 건수↔발생률 상관 r = {corr:.3f}")
    if corr > 0.7:
        log("   → 강한 양의 상관: 건수 많은 구가 발생률도 높음 (인구 외 다른 요인 큼)")
    elif corr > 0.4:
        log("   → 중간 양의 상관: 인구 일부 영향, 다른 요인도 존재")
    else:
        log("   → 약한 상관: 건수는 인구 규모 효과, 발생률이 진짜 위험 신호")

    log(f"\n**3. 계절성**")
    seasonal_max = seasonal["sum"].idxmax()
    seasonal_min = seasonal["sum"].idxmin()
    log(f"   - 최대월: {seasonal_max}월 ({int(seasonal['sum'].loc[seasonal_max])}건)")
    log(f"   - 최소월: {seasonal_min}월 ({int(seasonal['sum'].loc[seasonal_min])}건)")
    log(f"   - 변동: {seasonal['sum'].max() / seasonal['sum'].min():.1f}배")

    log(f"\n**4. 다음 단계 권장**")
    log(f"   - 모델 검증: 우리 발견확률 ↔ 자치구 실종 건수/발생률 상관 측정")
    log(f"   - 발생률 Top 5({sorted(set(top5_rate))})를 우선 분석 대상으로")

    # 보고서 저장
    (OUT / "08_summary.md").write_text(
        "# 실종 데이터 단독 EDA 보고서\n\n" + "\n".join(REPORT),
        encoding="utf-8",
    )
    print(f"\n\n✅ 산출물: {OUT.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
