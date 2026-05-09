"""
29_final_eda_full.py
final_data.csv 기반 종합 EDA + 보고서용 시각자료 일괄 생성.

[입력] data/processed/final_data.csv (250,450 × 67)
[출력] data/processed/report_viz/   (9~12개 PNG + HTML dashboard)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data/processed/final_data.csv"
OUT = PROJECT_ROOT / "data/processed/report_viz"
OUT.mkdir(parents=True, exist_ok=True)

# 한글 폰트
for fp in fm.findSystemFonts():
    if "AppleGothic" in fp or "AppleSDGothicNeo" in fp:
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
sns.set_theme(font="AppleGothic", style="whitegrid")

PALETTE_RISK = "RdYlGn_r"

# ─────── 핵심 feature ───────
KEY_FEATURES = [
    "intersection_density_200m", "road_complexity_200m",
    "deadend_count", "is_residential",
    "rest_count_within_200m", "has_park_within_200m",
    "cctv_count_within_200m", "facility_count_within_500m",
    "pop_65plus_per_grid", "추정치매환자수_총합",
    "INFRASTRUCTURE", "STORE", "POPULATION", "SALES",
    "slope", "치매센터수",
]
LABEL_KO = {
    "intersection_density_200m": "교차로 밀도 (200m)",
    "road_complexity_200m":      "도로 복잡도 (200m)",
    "deadend_count":             "막다른 길",
    "is_residential":            "주거지역",
    "rest_count_within_200m":    "휴식지 (200m)",
    "has_park_within_200m":      "공원 (200m)",
    "cctv_count_within_200m":    "CCTV (200m)",
    "facility_count_within_500m": "노인시설 (500m)",
    "pop_65plus_per_grid":       "65+ 인구",
    "추정치매환자수_총합":        "치매 환자 (추정)",
    "INFRASTRUCTURE":            "인프라",
    "STORE":                     "상점",
    "POPULATION":                "유동인구",
    "SALES":                     "상권 매출",
    "slope":                     "경사도",
    "치매센터수":                "치매센터",
    "missing_estimated":         "실종 추정 (격자)",
    "실종건수":                  "실종 (구별)",
    "risk_score":                "위험도 점수",
}


def lab(c: str) -> str:
    return LABEL_KO.get(c, c)


def main() -> None:
    print("[load]", DATA)
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["gu_name"].astype(str).str.endswith("구")].copy()  # 서울만
    print(f"  rows: {len(df):,}, cols: {df.shape[1]}")

    # risk_score 재계산 (notebook과 동일 방식: PCA loadings × ev_ratio)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    pca_features = [
        "intersection_count", "deadend_count", "road_complexity_raw",
        "is_residential", "rest_count_in_cell", "has_park_within_200m",
        "facility_count_in_cell", "rest_count_within_200m",
        "facility_count_within_200m", "slope", "INFRASTRUCTURE", "STORE",
        "POPULATION", "추정치매환자수_총합",
    ]
    pca_features = [c for c in pca_features if c in df.columns]
    sub = df[pca_features].fillna(0)
    Xs = StandardScaler().fit_transform(sub)
    pca = PCA().fit(Xs)
    ev = pca.explained_variance_ratio_
    k = int(np.argmax(np.cumsum(ev) >= 0.9)) + 1
    weights = (ev[:k] @ np.abs(pca.components_[:k]))
    weights /= weights.sum()
    df["risk_score"] = Xs @ weights
    df["risk_pct_rank"] = df["risk_score"].rank(pct=True)
    print(f"  PCA: {k} components → 90% variance, risk_score 생성")

    # ─────────────────────────────────────────────────
    # 01. 자치구별 실종건수 + 치매환자 + 발생률 (인포그래픽)
    # ─────────────────────────────────────────────────
    print("\n[01] 자치구별 실종 + 치매 종합")
    gu = df.groupby("gu_name", as_index=False).agg(
        missing_count=("실종건수", "first"),
        pop_65plus=("pop_65plus_gu", "first"),
        dementia_total=("추정치매환자수_총합", "sum"),
        risk_score_mean=("risk_score", "mean"),
        risk_score_max=("risk_score", "max"),
        n_grids=("grid_id_50m", "count"),
    )
    gu["rate_per_10k"] = gu["missing_count"] / gu["pop_65plus"] * 10000
    gu["dementia_rate"] = gu["dementia_total"] / gu["pop_65plus"] * 100  # 65+ 중 비율
    gu = gu.sort_values("rate_per_10k", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 9), sharey=True)
    order = gu["gu_name"].tolist()
    # (1) 절대 건수
    sns.barplot(data=gu, y="gu_name", x="missing_count", order=order,
                ax=axes[0], color="#3498db")
    axes[0].set_title("절대 실종건수", fontsize=13)
    axes[0].set_xlabel("건수 (15개월)")
    axes[0].set_ylabel("")
    for i, v in enumerate(gu["missing_count"]):
        axes[0].text(v + 0.5, i, str(int(v)), va="center", fontsize=9)

    # (2) 인구 보정 발생률
    colors = ["#c0392b" if i < 5 else "#7f8c8d" for i in range(len(gu))]
    sns.barplot(data=gu, y="gu_name", x="rate_per_10k", order=order,
                ax=axes[1], palette=colors, hue="gu_name", legend=False, dodge=False)
    axes[1].set_title("인구 보정 발생률 (per 10k of 65+)", fontsize=13)
    axes[1].set_xlabel("10,000명당 실종건수")
    for i, v in enumerate(gu["rate_per_10k"]):
        axes[1].text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)

    # (3) PCA 위험도 평균
    sns.barplot(data=gu, y="gu_name", x="risk_score_mean", order=order,
                ax=axes[2], color="#e67e22")
    axes[2].set_title("PCA 위험도 점수 평균", fontsize=13)
    axes[2].set_xlabel("위험도 점수")

    plt.suptitle("서울 25개 자치구 — 실종 위기 다각도 비교", fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "01_gu_3panel_comparison.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 02. risk_score ↔ missing_estimated 산점도 + 회귀
    # ─────────────────────────────────────────────────
    print("[02] risk_score ↔ 실종 산점도")
    fig, ax = plt.subplots(figsize=(10, 8))
    sample = df.sample(min(15000, len(df)), random_state=0)
    sc = ax.scatter(sample["risk_score"], sample["missing_estimated"],
                    c=sample["risk_pct_rank"], cmap=PALETTE_RISK,
                    s=4, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="risk pct rank")

    from scipy.stats import spearmanr, pearsonr
    rs, ps = spearmanr(df["risk_score"], df["missing_estimated"])
    rp, pp = pearsonr(df["risk_score"].fillna(0), df["missing_estimated"].fillna(0))
    ax.set_xlabel("PCA Risk Score")
    ax.set_ylabel("실종 추정 (격자 단위)")
    ax.set_title(f"위험도 ↔ 실종 추정\nSpearman r = {rs:.3f} (p<0.001), "
                 f"Pearson r = {rp:.3f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "02_risk_vs_missing_scatter.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 03. PCA loadings — 어떤 변수가 위험도에 기여하나
    # ─────────────────────────────────────────────────
    print("[03] PCA loadings")
    contrib = pd.Series(weights, index=pca_features) * 100
    contrib = contrib.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = sns.color_palette("RdYlGn_r", n_colors=len(contrib))
    bars = ax.barh([lab(c) for c in contrib.index], contrib.values, color=colors[::-1])
    ax.set_xlabel("기여도 (%)")
    ax.set_title("PCA 위험도 점수 — 변수별 기여도", fontsize=13)
    for bar, v in zip(bars, contrib.values):
        ax.text(v + 0.1, bar.get_y() + bar.get_height()/2, f"{v:.2f}",
                va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "03_pca_feature_contribution.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 04. 핵심 feature ↔ missing_estimated 상관 매트릭스
    # ─────────────────────────────────────────────────
    print("[04] 상관 매트릭스")
    corr_cols = [c for c in KEY_FEATURES + ["missing_estimated", "risk_score"]
                 if c in df.columns]
    corr = df[corr_cols].corr()
    corr_labeled = corr.rename(index=lab, columns=lab)
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr_labeled, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson r"}, ax=ax)
    ax.set_title("Feature 상관 매트릭스 (실종 추정 + 위험도 포함)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(OUT / "04_correlation_matrix.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 05. 종로구 vs 도심 4구 비교 (case study)
    # ─────────────────────────────────────────────────
    print("[05] 종로구 case study")
    central_4 = ["종로구", "중구", "용산구", "서대문구"]
    sub_c = df[df["gu_name"].isin(central_4)]
    feats_compare = [
        "intersection_density_200m", "road_complexity_200m",
        "is_residential", "cctv_count_within_200m",
        "rest_count_within_200m", "facility_count_within_500m",
        "pop_65plus_per_grid", "추정치매환자수_총합",
        "INFRASTRUCTURE", "STORE", "slope", "risk_score",
    ]
    feats_compare = [c for c in feats_compare if c in df.columns]
    profile = sub_c.groupby("gu_name")[feats_compare].mean()
    # min-max 정규화 (각 feature)
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(profile_norm.rename(columns=lab), annot=True, fmt=".2f",
                cmap="RdYlGn_r", linewidths=0.4, ax=axes[0],
                cbar_kws={"label": "정규화 (0~1)"})
    axes[0].set_title("도심 4구 Feature 프로필 (정규화)", fontsize=13)
    axes[0].set_xlabel(""); axes[0].set_ylabel("")

    # 발생률 비교
    rate_c = gu[gu["gu_name"].isin(central_4)].sort_values("rate_per_10k", ascending=False)
    colors_c = ["#c0392b" if g == "종로구" else "#3498db" for g in rate_c["gu_name"]]
    axes[1].bar(rate_c["gu_name"], rate_c["rate_per_10k"], color=colors_c)
    for i, v in enumerate(rate_c["rate_per_10k"]):
        axes[1].text(i, v + 0.2, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    axes[1].set_title("도심 4구 발생률 (per 10k of 65+)", fontsize=13)
    axes[1].set_ylabel("실종 발생률")
    axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("종로구 vs 도심 4구 비교 — Case Study", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / "05_jongno_central4_comparison.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 06. 자치구별 risk_score boxplot (분포)
    # ─────────────────────────────────────────────────
    print("[06] 자치구별 risk 분포")
    fig, ax = plt.subplots(figsize=(15, 8))
    order_byrisk = gu.sort_values("risk_score_mean", ascending=False)["gu_name"].tolist()
    sns.boxplot(data=df, x="gu_name", y="risk_score", order=order_byrisk,
                ax=ax, palette="RdYlGn_r", hue="gu_name", legend=False,
                showfliers=False)
    ax.set_title("자치구별 PCA 위험도 점수 분포 (격자 단위)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Risk Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT / "06_gu_risk_boxplot.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 07. 위험 격자 지도 (산점도, lon/lat 직접) — Top 5%
    # ─────────────────────────────────────────────────
    print("[07] 위험 격자 지도")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    # (1) 전체 risk
    sample = df.sample(min(40000, len(df)), random_state=0)
    sc1 = axes[0].scatter(sample["center_lon"], sample["center_lat"],
                          c=sample["risk_score"], cmap=PALETTE_RISK,
                          s=2, alpha=0.6)
    plt.colorbar(sc1, ax=axes[0], label="Risk Score")
    axes[0].set_title("서울 전체 격자 위험도 (40k 샘플)", fontsize=13)
    axes[0].set_xlabel("경도"); axes[0].set_ylabel("위도")
    axes[0].set_aspect("equal")

    # (2) Top 5% 격자만
    threshold = df["risk_score"].quantile(0.95)
    top = df[df["risk_score"] >= threshold]
    axes[1].scatter(df.sample(20000, random_state=0)["center_lon"],
                    df.sample(20000, random_state=0)["center_lat"],
                    color="lightgray", s=1, alpha=0.3)
    sc2 = axes[1].scatter(top["center_lon"], top["center_lat"],
                          c=top["risk_score"], cmap="Reds", s=3, alpha=0.7)
    plt.colorbar(sc2, ax=axes[1], label="Risk Score")
    axes[1].set_title(f"위험 상위 5% 격자만 ({len(top):,}개, threshold={threshold:.2f})",
                      fontsize=13)
    axes[1].set_xlabel("경도"); axes[1].set_ylabel("위도")
    axes[1].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT / "07_seoul_risk_map.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 08. 종로구 zoom — 격자별 위험도
    # ─────────────────────────────────────────────────
    print("[08] 종로구 zoom")
    jongno = df[df["gu_name"] == "종로구"]
    if len(jongno) > 0:
        fig, ax = plt.subplots(figsize=(12, 10))
        sc = ax.scatter(jongno["center_lon"], jongno["center_lat"],
                        c=jongno["risk_score"], cmap=PALETTE_RISK,
                        s=8, alpha=0.85, edgecolors="white", linewidths=0.05)
        plt.colorbar(sc, ax=ax, label="Risk Score")
        ax.set_title(f"종로구 격자별 위험도 ({len(jongno):,}개 격자)", fontsize=13)
        ax.set_xlabel("경도"); ax.set_ylabel("위도")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.savefig(OUT / "08_jongno_zoom.png", bbox_inches="tight")
        plt.close()

    # ─────────────────────────────────────────────────
    # 09. 핵심 feature 자치구별 정규화 히트맵 (25 × 12)
    # ─────────────────────────────────────────────────
    print("[09] 자치구 feature 히트맵")
    fcols = [c for c in feats_compare if c in df.columns]
    gu_feat = df.groupby("gu_name")[fcols].mean()
    gu_feat = gu_feat.loc[order_byrisk]
    gu_feat_norm = (gu_feat - gu_feat.min()) / (gu_feat.max() - gu_feat.min() + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 11))
    sns.heatmap(gu_feat_norm.rename(columns=lab), annot=False, cmap="YlOrRd",
                linewidths=0.3, cbar_kws={"label": "정규화"}, ax=ax)
    ax.set_title("자치구 × Feature 정규화 (위험도 내림차순)", fontsize=13)
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(OUT / "09_gu_feature_heatmap.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 10. 발견율 vs 위험도 자치구 산점도 + 회귀선
    # ─────────────────────────────────────────────────
    print("[10] 위험도 vs 발생률 자치구 산점도")
    fig, ax = plt.subplots(figsize=(11, 9))
    for _, r in gu.iterrows():
        col = "#c0392b" if r["rate_per_10k"] > gu["rate_per_10k"].quantile(0.8) else "#3498db"
        ax.scatter(r["risk_score_mean"], r["rate_per_10k"], s=r["pop_65plus"]/300,
                   alpha=0.65, color=col, edgecolors="black", linewidths=0.6)
        ax.annotate(r["gu_name"], (r["risk_score_mean"], r["rate_per_10k"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=10)
    # 회귀선
    xs = gu["risk_score_mean"].to_numpy()
    ys = gu["rate_per_10k"].to_numpy()
    a, b = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xx, a * xx + b, "--", color="gray", alpha=0.6,
            label=f"y = {a:.2f}x + {b:.2f}")

    rg, pg = pearsonr(xs, ys)
    ax.set_xlabel("PCA 위험도 점수 (자치구 평균)")
    ax.set_ylabel("실종 발생률 (per 10k of 65+)")
    ax.set_title(f"자치구 위험도 ↔ 발생률\nPearson r = {rg:.3f}, p = {pg:.4f}, n = 25",
                 fontsize=13)
    ax.text(0.02, 0.98, "점 크기 = 65+ 인구\n빨강 = 발생률 상위 20%",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "10_gu_risk_vs_rate.png", bbox_inches="tight")
    plt.close()

    # ─────────────────────────────────────────────────
    # 결과 요약 CSV
    # ─────────────────────────────────────────────────
    print("[summary CSV]")
    gu.to_csv(OUT / "gu_summary.csv", index=False, encoding="utf-8-sig")
    contrib_df = pd.DataFrame({
        "feature": [lab(c) for c in contrib.index],
        "raw": contrib.index,
        "weight_pct": contrib.values,
    })
    contrib_df.to_csv(OUT / "pca_feature_contribution.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUT / "correlation_matrix.csv", encoding="utf-8-sig")

    print(f"\n✅ {OUT.relative_to(PROJECT_ROOT)}/  에 PNG 10개 + CSV 3개 생성")
    print(f"\n[자치구 위험도 ↔ 발생률 상관]: r = {rg:.3f} (p = {pg:.4f})")
    print(f"[risk_score ↔ missing_estimated Spearman]: r = {rs:.3f}")


if __name__ == "__main__":
    main()
