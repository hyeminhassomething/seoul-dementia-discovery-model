"""
31_cluster_analysis_full.py
LISA + HDBSCAN 클러스터 분석 → 보고서/PPT용 시각자료 일괄 생성.

[입력] data/processed/cluster_labeled_data.xlsx (53,390 격자)
[출력] data/processed/report_viz/cluster/
    01_pipeline_funnel.png             파이프라인 3단 깔때기
    02_lisa_hdbscan_crosstab.png       LISA × HDBSCAN 교차표 히트맵
    03_hdbscan_profile_heatmap.png     군집별 feature 정규화 히트맵
    04_top_clusters_radar.png          상위 6개 군집 radar 차트
    05_top15_dong_table.png            정책 우선 행정동 TOP 15
    06_final_target_map.png            최종 타겟 격자 지도
    07_cluster_naming.csv              자동 명명 결과
    08_top15_dong_data.csv             TOP 15 raw 데이터
    HH_HL_cluster_dashboard.html       인터랙티브 대시보드
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data/processed/cluster_labeled_data.xlsx"
OUT = PROJECT_ROOT / "data/processed/report_viz/cluster"
OUT.mkdir(parents=True, exist_ok=True)

for fp in fm.findSystemFonts():
    if "AppleGothic" in fp or "AppleSDGothicNeo" in fp:
        fm.fontManager.addfont(fp)
        break
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
sns.set_theme(font="AppleGothic", style="whitegrid")

FEATURES = [
    "intersection_count", "deadend_count", "road_complexity_raw",
    "slope", "pop_65plus_per_grid", "INFRASTRUCTURE", "STORE", "risk_score",
]
LABEL = {
    "intersection_count": "교차로",
    "deadend_count":      "막다른길",
    "road_complexity_raw": "도로복잡도",
    "slope":              "경사도",
    "pop_65plus_per_grid": "65+ 인구",
    "INFRASTRUCTURE":     "인프라",
    "STORE":              "상점",
    "risk_score":         "위험도",
}


def auto_name_cluster(profile_row: pd.Series) -> str:
    """군집의 정규화된 feature 프로필 → 별명 자동 생성."""
    top2 = profile_row.nlargest(2)
    has = lambda k: k in top2.index
    if (has("deadend_count") or has("road_complexity_raw")) and has("slope"):
        return "물리적 고립형"
    if has("pop_65plus_per_grid") and not has("INFRASTRUCTURE"):
        return "인구 밀집·인프라 소외형"
    if has("INFRASTRUCTURE") or has("STORE"):
        return "도심 상권형"
    if has("intersection_count") and has("road_complexity_raw"):
        return "골목 미로형"
    if has("slope"):
        return "경사·언덕형"
    if has("pop_65plus_per_grid"):
        return "고령 밀집형"
    return "복합형"


def main():
    print("[load]", DATA.name)
    df = pd.read_excel(DATA, sheet_name="Sheet1")
    df_valid = df[df["HDBSCAN_Cluster"] != -1].copy()
    print(f"  total: {len(df):,}, valid (cluster ≠ -1): {len(df_valid):,}")
    print(f"  LISA: {df_valid['LISA_Cluster'].value_counts().to_dict()}")

    # ─────────── 01. 파이프라인 깔때기 ───────────
    print("\n[01] pipeline funnel")
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    n0 = 250450
    n1 = len(df)
    n2 = len(df_valid)
    n3 = (df_valid["LISA_Cluster"] == "HH").sum() + (df_valid["LISA_Cluster"] == "HL").sum()

    stages = [
        (8.5, "#bdc3c7", f"전체 격자\n{n0:,}개", "Step 0\n50m × 50m 격자"),
        (7.0, "#95a5a6", f"위험도 평가\n{n1:,}개", "Step 1\nXGBoost · risk_score"),
        (5.5, "#e67e22", f"공간 핫스팟\nLISA: HH/HL\n{n3:,}개", "Step 2\nLISA Moran's I"),
        (4.0, "#c0392b", f"맞춤 타겟\nHDBSCAN 군집\n{n2:,}개", "Step 3\n특성 분류"),
        (2.5, "#7d2c1e", "정책 개입\nTOP 15 행정동", "Step 4\n실행 계획"),
    ]
    for i, (y, color, label, desc) in enumerate(stages):
        width = 8 - i * 1.3
        x_left = 5 - width / 2
        rect = mpatches.FancyBboxPatch(
            (x_left, y - 0.55), width, 1.1,
            boxstyle="round,pad=0.02", linewidth=1.5,
            facecolor=color, edgecolor="white",
        )
        ax.add_patch(rect)
        ax.text(5, y, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color="white")
        ax.text(x_left + width + 0.15, y, desc, ha="left", va="center",
                fontsize=10, color="#2c3e50")
        if i < len(stages) - 1:
            ax.annotate("", xy=(5, stages[i+1][0] + 0.55), xytext=(5, y - 0.55),
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    ax.set_title("치매 위험구역 타겟팅 파이프라인", fontsize=16, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(OUT / "01_pipeline_funnel.png", bbox_inches="tight")
    plt.close()

    # ─────────── 02. LISA × HDBSCAN 교차표 ───────────
    print("[02] LISA × HDBSCAN crosstab")
    cross = pd.crosstab(df_valid["LISA_Cluster"], df_valid["HDBSCAN_Cluster"])
    # 격자 많은 군집 상위 12개만
    top_clusters = df_valid["HDBSCAN_Cluster"].value_counts().head(12).index.tolist()
    cross_top = cross[top_clusters]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(cross_top, annot=True, fmt="d", cmap="Reds", linewidths=0.5,
                cbar_kws={"label": "격자 수"}, ax=ax)
    ax.set_title("LISA 공간 클러스터 × HDBSCAN 군집 교차표 (상위 12 군집)",
                 fontsize=13, pad=12)
    ax.set_xlabel("HDBSCAN 군집 ID")
    ax.set_ylabel("LISA 클러스터")
    plt.tight_layout()
    plt.savefig(OUT / "02_lisa_hdbscan_crosstab.png", bbox_inches="tight")
    plt.close()

    # ─────────── 03. 군집별 프로필 히트맵 ───────────
    print("[03] cluster profile heatmap")
    profiles = df_valid.groupby("HDBSCAN_Cluster")[FEATURES].mean()
    profiles_top = profiles.loc[top_clusters]
    scaler = MinMaxScaler()
    profiles_scaled = pd.DataFrame(
        scaler.fit_transform(profiles_top), index=profiles_top.index,
        columns=[LABEL[c] for c in FEATURES],
    )
    cluster_sizes = df_valid["HDBSCAN_Cluster"].value_counts().loc[top_clusters]
    profiles_scaled.index = [f"군집 {c}\n(n={cluster_sizes[c]:,})" for c in top_clusters]

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(profiles_scaled, annot=True, fmt=".2f", cmap="Reds",
                linewidths=0.5, cbar_kws={"label": "정규화 (0~1)"},
                vmin=0, vmax=1, ax=ax)
    ax.set_title("HDBSCAN 군집별 핵심 특성 프로필 (상위 12 군집)", fontsize=13, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(OUT / "03_hdbscan_profile_heatmap.png", bbox_inches="tight")
    plt.close()

    # ─────────── 군집 자동 명명 ───────────
    print("[name] auto-naming top clusters")
    cluster_names = {}
    profiles_scaled_for_name = pd.DataFrame(
        MinMaxScaler().fit_transform(profiles_top),
        index=profiles_top.index, columns=FEATURES,
    )
    for cid in top_clusters:
        nm = auto_name_cluster(profiles_scaled_for_name.loc[cid])
        cluster_names[cid] = nm

    naming_df = pd.DataFrame({
        "cluster_id": top_clusters,
        "n_grids": [cluster_sizes[c] for c in top_clusters],
        "auto_name": [cluster_names[c] for c in top_clusters],
        "top_feature_1": [profiles_scaled_for_name.loc[c].idxmax() for c in top_clusters],
        "top_feature_2": [profiles_scaled_for_name.loc[c].nlargest(2).index[1] for c in top_clusters],
    })
    naming_df["top_feature_1_ko"] = naming_df["top_feature_1"].map(LABEL)
    naming_df["top_feature_2_ko"] = naming_df["top_feature_2"].map(LABEL)
    naming_df.to_csv(OUT / "07_cluster_naming.csv", index=False, encoding="utf-8-sig")
    print(naming_df.to_string(index=False))

    # ─────────── 04. 상위 6개 군집 radar ───────────
    print("[04] top 6 cluster radar")
    top6 = top_clusters[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11), subplot_kw={"projection": "polar"})
    axes = axes.flatten()
    angles = np.linspace(0, 2 * np.pi, len(FEATURES), endpoint=False).tolist()
    angles += angles[:1]
    palette = sns.color_palette("Set1", n_colors=6)

    for i, cid in enumerate(top6):
        ax = axes[i]
        vals = profiles_scaled_for_name.loc[cid].tolist()
        vals += vals[:1]
        ax.fill(angles, vals, color=palette[i], alpha=0.4)
        ax.plot(angles, vals, color=palette[i], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([LABEL[c] for c in FEATURES], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["0.25", "0.5", "0.75"], fontsize=7, color="gray")
        ax.set_title(f"군집 {cid} — {cluster_names[cid]}\n(n={cluster_sizes[cid]:,})",
                     fontsize=11, pad=18, fontweight="bold")

    plt.suptitle("상위 6개 HDBSCAN 군집의 환경 프로필 (Radar)",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "04_top_clusters_radar.png", bbox_inches="tight")
    plt.close()

    # ─────────── 05. TOP 15 행정동 ───────────
    print("[05] top 15 dong")
    region = df_valid.groupby(
        ["gu_name", "ADM_NM", "LISA_Cluster", "HDBSCAN_Cluster"]
    ).size().reset_index(name="격자수")
    top15 = region.sort_values("격자수", ascending=False).head(15).copy()
    top15["군집_별명"] = top15["HDBSCAN_Cluster"].map(cluster_names).fillna("기타")
    top15.to_csv(OUT / "08_top15_dong_data.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    table_data = top15[["gu_name", "ADM_NM", "LISA_Cluster",
                        "HDBSCAN_Cluster", "군집_별명", "격자수"]]
    cell_colors = []
    for _, row in table_data.iterrows():
        c = "#fadbd8" if row["LISA_Cluster"] == "HH" else "#fcf3cf"
        cell_colors.append([c] * len(table_data.columns))
    tbl = ax.table(
        cellText=table_data.values,
        colLabels=["자치구", "행정동", "LISA", "군집ID", "군집 별명", "격자수"],
        cellLoc="center", loc="center", cellColours=cell_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.6)
    for i in range(len(table_data.columns)):
        cell = tbl[0, i]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("정책 개입 최우선 행정동 TOP 15 — 격자 수 기준",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(OUT / "05_top15_dong_table.png", bbox_inches="tight")
    plt.close()

    # ─────────── 06. 최종 타겟 지도 ───────────
    print("[06] final target map")
    fig, ax = plt.subplots(figsize=(14, 11))

    # 배경: 전체 격자 회색 (샘플)
    bg = df.sample(min(20000, len(df)), random_state=0)
    ax.scatter(bg["center_lon"], bg["center_lat"], c="#d5dbdb", s=0.4, alpha=0.4)

    # HH (빨강), HL (노랑) 분리
    hh = df_valid[df_valid["LISA_Cluster"] == "HH"]
    hl = df_valid[df_valid["LISA_Cluster"] == "HL"]
    ax.scatter(hh["center_lon"], hh["center_lat"], c="#e74c3c", s=3,
               alpha=0.6, label=f"HH ({len(hh):,}개)")
    ax.scatter(hl["center_lon"], hl["center_lat"], c="#f1c40f", s=8,
               alpha=0.85, marker="*", label=f"HL ({len(hl):,}개)",
               edgecolors="black", linewidths=0.4)

    # TOP 5 행정동 마킹
    top5_dong = top15.head(5)
    for _, r in top5_dong.iterrows():
        sub = df_valid[(df_valid["gu_name"] == r["gu_name"]) &
                       (df_valid["ADM_NM"] == r["ADM_NM"])]
        cx, cy = sub["center_lon"].mean(), sub["center_lat"].mean()
        ax.annotate(
            f"{r['gu_name']}\n{r['ADM_NM']}\n({r['군집_별명']})",
            xy=(cx, cy), xytext=(15, 15), textcoords="offset points",
            fontsize=10, fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#c0392b", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5),
        )

    ax.set_xlabel("경도")
    ax.set_ylabel("위도")
    ax.set_title("서울시 치매 노인 실종 위험 핫스팟 — 최종 타겟",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT / "06_final_target_map.png", bbox_inches="tight")
    plt.close()

    # ─────────── HTML 인터랙티브 대시보드 ───────────
    print("[07] interactive dashboard")
    # plotly 지도
    sample_bg = df.sample(min(15000, len(df)), random_state=0)
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermap(
        lat=sample_bg["center_lat"], lon=sample_bg["center_lon"],
        mode="markers", marker=dict(size=3, color="#bdc3c7", opacity=0.4),
        name="일반 격자", hoverinfo="skip",
    ))
    fig_map.add_trace(go.Scattermap(
        lat=hh["center_lat"], lon=hh["center_lon"],
        mode="markers",
        marker=dict(size=4, color="#e74c3c", opacity=0.55),
        name=f"HH ({len(hh):,})",
        text=[f"{g} {a}<br>군집 {c}: {cluster_names.get(c, '기타')}<br>위험도 {r:.2f}"
              for g, a, c, r in zip(hh["gu_name"], hh["ADM_NM"],
                                    hh["HDBSCAN_Cluster"], hh["risk_score"])],
        hoverinfo="text",
    ))
    fig_map.add_trace(go.Scattermap(
        lat=hl["center_lat"], lon=hl["center_lon"],
        mode="markers",
        marker=dict(size=10, color="#f1c40f", symbol="star",
                    opacity=0.9),
        name=f"HL ({len(hl):,})",
        text=[f"{g} {a}<br>군집 {c}<br>위험도 {r:.2f}"
              for g, a, c, r in zip(hl["gu_name"], hl["ADM_NM"],
                                    hl["HDBSCAN_Cluster"], hl["risk_score"])],
        hoverinfo="text",
    ))
    fig_map.update_layout(
        map=dict(style="carto-positron",
                 center=dict(lat=37.553, lon=126.99), zoom=10.5),
        margin=dict(r=0, t=40, l=0, b=0), height=700,
        title="HH/HL 격자 인터랙티브 지도 (호버: 군집 별명·위험도)",
    )
    fig_map.write_html(OUT / "06_final_target_map.html", include_plotlyjs="cdn")

    # 통합 dashboard
    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>HH/HL 클러스터 분석 대시보드</title>
<style>
  body {{ font-family: -apple-system, "Apple SD Gothic Neo", sans-serif;
          margin: 0; background: #f4f5f7; color: #2c3e50; }}
  header {{ background: linear-gradient(135deg, #c0392b 0%, #7d2c1e 100%);
            color: white; padding: 32px 24px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 0; opacity: 0.92; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px; margin-bottom: 20px; }}
  .stat {{ background: white; padding: 18px; border-radius: 10px;
           box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center; }}
  .stat-num {{ font-size: 28px; font-weight: bold; color: #c0392b; }}
  .stat-lbl {{ font-size: 11px; color: #7f8c8d; margin-top: 6px; }}
  .card {{ background: white; padding: 20px; margin-bottom: 16px;
           border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
  .card h2 {{ margin: 0 0 10px; color: #34495e; font-size: 17px; }}
  .card img {{ width: 100%; border-radius: 6px; border: 1px solid #eee; }}
  iframe {{ width: 100%; border: none; border-radius: 6px; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 1000px) {{ .row {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<header>
  <h1>🎯 HH/HL 클러스터 분석 대시보드</h1>
  <p>LISA 공간 자기상관 + HDBSCAN 환경 분류 → 정책 우선순위 도출</p>
</header>
<div class="container">
  <div class="stats">
    <div class="stat"><div class="stat-num">53,390</div><div class="stat-lbl">분석 격자</div></div>
    <div class="stat"><div class="stat-num">52,298</div><div class="stat-lbl">HH (광역 위험)</div></div>
    <div class="stat"><div class="stat-num">1,092</div><div class="stat-lbl">HL (사각지대)</div></div>
    <div class="stat"><div class="stat-num">38</div><div class="stat-lbl">HDBSCAN 군집</div></div>
    <div class="stat"><div class="stat-num">12</div><div class="stat-lbl">유효 상위 군집</div></div>
  </div>
  <div class="card"><h2>📐 1. 분석 파이프라인</h2>
    <img src="01_pipeline_funnel.png"></div>
  <div class="card"><h2>🗺️ 2. 최종 타겟 인터랙티브 지도</h2>
    <iframe src="06_final_target_map.html" height="720"></iframe></div>
  <div class="row">
    <div class="card"><h2>📊 3. LISA × HDBSCAN 교차표</h2>
      <img src="02_lisa_hdbscan_crosstab.png"></div>
    <div class="card"><h2>🎨 4. 군집 프로필 히트맵</h2>
      <img src="03_hdbscan_profile_heatmap.png"></div>
  </div>
  <div class="card"><h2>🎯 5. 상위 6 군집 환경 Radar</h2>
    <img src="04_top_clusters_radar.png"></div>
  <div class="card"><h2>🚨 6. 정책 개입 우선 행정동 TOP 15</h2>
    <img src="05_top15_dong_table.png"></div>
</div>
</body></html>"""
    (OUT / "HH_HL_cluster_dashboard.html").write_text(html, encoding="utf-8")

    print(f"\n✅ 모든 출력: {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"   대시보드: open {OUT / 'HH_HL_cluster_dashboard.html'}")
    print(f"\n📊 분석 요약:")
    print(f"   HH: {(df_valid['LISA_Cluster']=='HH').sum():,}, HL: {(df_valid['LISA_Cluster']=='HL').sum():,}")
    print(f"   상위 군집 별명:")
    for c, n in list(cluster_names.items())[:6]:
        print(f"     군집 {c} ({cluster_sizes[c]:,}개) → {n}")
    print(f"\n   TOP 5 정책 우선 동:")
    for _, r in top15.head(5).iterrows():
        print(f"     {r['gu_name']} {r['ADM_NM']} ({r['LISA_Cluster']}) "
              f"군집 {r['HDBSCAN_Cluster']}: {r['군집_별명']} — {r['격자수']}개")


if __name__ == "__main__":
    main()
