"""
30_interactive_dashboard.py
인터랙티브 plotly 시각화 — 발표용 통합 dashboard.

[출력]
  data/processed/report_viz/dashboard/
    01_seoul_risk_map.html             서울 전체 위험도 격자 지도 (인터랙티브)
    02_jongno_zoom.html                종로구 zoom (격자 클릭→상세정보)
    03_gu_3panel.html                  자치구 3패널 비교
    04_central4_radar.html             도심4구 radar chart
    05_correlation_heatmap.html        상관 히트맵 (인터랙티브)
    06_pca_contribution.html           PCA 기여도
    07_risk_distribution.html          분포 + 박스플롯
    08_top_grids_table.html            위험 Top 100 격자 테이블
    99_dashboard.html                  ⭐ 모든 차트 종합 페이지
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data/processed/final_data.csv"
OUT = PROJECT_ROOT / "data/processed/report_viz/dashboard"
OUT.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["gu_name"].astype(str).str.endswith("구")].copy()
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
    return df, pca_features, weights


# ───────── 01. Seoul risk map ─────────
def viz_seoul_map(df: pd.DataFrame) -> Path:
    sample = df.sample(min(40000, len(df)), random_state=0)
    fig = px.scatter_mapbox(
        sample, lat="center_lat", lon="center_lon",
        color="risk_score", color_continuous_scale="RdYlGn_r",
        size_max=4, zoom=10.5, height=750,
        center={"lat": 37.553, "lon": 126.99},
        hover_data={"gu_name": True, "risk_score": ":.2f",
                    "missing_estimated": ":.4f",
                    "추정치매환자수_총합": ":.1f",
                    "center_lat": False, "center_lon": False},
        title="서울 전체 50m 격자 위험도 — Risk Score 분포 (40,000 샘플)",
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_traces(marker={"size": 4, "opacity": 0.6})
    out = OUT / "01_seoul_risk_map.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 02. Jongno zoom ─────────
def viz_jongno(df: pd.DataFrame) -> Path:
    j = df[df["gu_name"] == "종로구"].copy()
    fig = px.scatter_mapbox(
        j, lat="center_lat", lon="center_lon",
        color="risk_score", color_continuous_scale="RdYlGn_r",
        zoom=12.5, height=750,
        hover_data={
            "ADM_NM": True, "risk_score": ":.2f",
            "intersection_density_200m": ":.2f",
            "road_complexity_200m": ":.2f",
            "추정치매환자수_총합": ":.1f",
            "cctv_count_within_200m": True,
            "center_lat": False, "center_lon": False,
        },
        title=f"종로구 격자별 위험도 ({len(j):,}개) — 호버로 상세정보 확인",
    )
    fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_traces(marker={"size": 7, "opacity": 0.85})
    out = OUT / "02_jongno_zoom.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 03. Gu 3 panel ─────────
def viz_gu_3panel(df: pd.DataFrame) -> Path:
    gu = df.groupby("gu_name", as_index=False).agg(
        missing_count=("실종건수", "first"),
        pop_65plus=("pop_65plus_gu", "first"),
        risk_mean=("risk_score", "mean"),
        dementia_total=("추정치매환자수_총합", "sum"),
    )
    gu["rate_per_10k"] = gu["missing_count"] / gu["pop_65plus"] * 10000
    gu = gu.sort_values("rate_per_10k", ascending=True)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("절대 실종건수", "발생률 (per 10k of 65+)", "PCA 위험도 평균"),
        shared_yaxes=True,
    )

    fig.add_trace(go.Bar(
        y=gu["gu_name"], x=gu["missing_count"], orientation="h",
        marker_color="#3498db",
        text=gu["missing_count"].astype(int), textposition="outside",
        name="건수",
        hovertemplate="<b>%{y}</b><br>건수: %{x}건<extra></extra>",
    ), row=1, col=1)

    rate_colors = ["#c0392b" if v > gu["rate_per_10k"].quantile(0.8) else "#7f8c8d"
                   for v in gu["rate_per_10k"]]
    fig.add_trace(go.Bar(
        y=gu["gu_name"], x=gu["rate_per_10k"], orientation="h",
        marker_color=rate_colors,
        text=gu["rate_per_10k"].round(2), textposition="outside",
        name="발생률",
        hovertemplate="<b>%{y}</b><br>%{x:.2f}/10k<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=gu["gu_name"], x=gu["risk_mean"], orientation="h",
        marker_color="#e67e22",
        text=gu["risk_mean"].round(3), textposition="outside",
        name="위험도",
        hovertemplate="<b>%{y}</b><br>위험도 %{x:.3f}<extra></extra>",
    ), row=1, col=3)

    fig.update_layout(height=750, showlegend=False, title_text="자치구별 실종 위기 다각도 비교")
    out = OUT / "03_gu_3panel.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 04. Central 4-gu radar ─────────
def viz_radar(df: pd.DataFrame) -> Path:
    central_4 = ["종로구", "중구", "용산구", "서대문구"]
    feats = [
        "intersection_density_200m", "road_complexity_200m",
        "is_residential", "cctv_count_within_200m",
        "rest_count_within_200m", "facility_count_within_500m",
        "추정치매환자수_총합", "INFRASTRUCTURE", "STORE", "slope",
    ]
    feats = [c for c in feats if c in df.columns]
    label_map = {
        "intersection_density_200m": "교차로",
        "road_complexity_200m":      "도로복잡도",
        "is_residential":            "주거",
        "cctv_count_within_200m":    "CCTV",
        "rest_count_within_200m":    "휴식지",
        "facility_count_within_500m": "노인시설",
        "추정치매환자수_총합":        "치매환자",
        "INFRASTRUCTURE":            "인프라",
        "STORE":                     "상점",
        "slope":                     "경사",
    }

    sub = df[df["gu_name"].isin(central_4)]
    profile = sub.groupby("gu_name")[feats].mean()
    norm = (profile - df[feats].min()) / (df[feats].max() - df[feats].min() + 1e-9)
    norm = norm.loc[central_4]

    colors = {"종로구": "#c0392b", "중구": "#e67e22", "용산구": "#3498db", "서대문구": "#27ae60"}
    fig = go.Figure()
    for gu in central_4:
        vals = norm.loc[gu].tolist()
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[label_map[c] for c in feats] + [label_map[feats[0]]],
            name=gu, line_color=colors[gu],
            fill="toself", opacity=0.35,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(0.4, norm.max().max())])),
        title="도심 4구 환경 프로필 비교 — Radar Chart",
        height=650, showlegend=True,
    )
    out = OUT / "04_central4_radar.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 05. Correlation heatmap ─────────
def viz_corr(df: pd.DataFrame) -> Path:
    cols = [
        "intersection_density_200m", "road_complexity_200m", "deadend_count",
        "is_residential", "rest_count_within_200m", "has_park_within_200m",
        "cctv_count_within_200m", "facility_count_within_500m",
        "pop_65plus_per_grid", "추정치매환자수_총합",
        "INFRASTRUCTURE", "STORE", "POPULATION", "slope",
        "missing_estimated", "risk_score",
    ]
    cols = [c for c in cols if c in df.columns]
    label_map = {
        "intersection_density_200m": "교차로", "road_complexity_200m": "도로복잡",
        "deadend_count": "막다른길", "is_residential": "주거",
        "rest_count_within_200m": "휴식200m", "has_park_within_200m": "공원200m",
        "cctv_count_within_200m": "CCTV200m", "facility_count_within_500m": "시설500m",
        "pop_65plus_per_grid": "65+인구", "추정치매환자수_총합": "치매환자",
        "INFRASTRUCTURE": "인프라", "STORE": "상점", "POPULATION": "유동인구",
        "slope": "경사", "missing_estimated": "실종추정", "risk_score": "위험도",
    }
    corr = df[cols].corr().round(3)
    labels = [label_map.get(c, c) for c in cols]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        textfont={"size": 9},
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Feature 상관 매트릭스 (실종추정·위험도 포함)",
        height=720, width=900,
    )
    out = OUT / "05_correlation_heatmap.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 06. PCA contribution ─────────
def viz_pca_contrib(weights: np.ndarray, features: list[str]) -> Path:
    label_map = {
        "intersection_count": "교차로", "deadend_count": "막다른길",
        "road_complexity_raw": "도로복잡도", "is_residential": "주거",
        "rest_count_in_cell": "휴식(셀)", "has_park_within_200m": "공원200m",
        "facility_count_in_cell": "시설(셀)", "rest_count_within_200m": "휴식200m",
        "facility_count_within_200m": "시설200m", "slope": "경사",
        "INFRASTRUCTURE": "인프라", "STORE": "상점",
        "POPULATION": "유동인구", "추정치매환자수_총합": "치매환자",
    }
    df = pd.DataFrame({
        "feature": [label_map.get(c, c) for c in features],
        "weight_pct": weights * 100,
    }).sort_values("weight_pct", ascending=True)
    fig = go.Figure(go.Bar(
        y=df["feature"], x=df["weight_pct"], orientation="h",
        marker_color=df["weight_pct"], marker_colorscale="RdYlGn_r",
        text=df["weight_pct"].round(2), textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="PCA 위험도 점수 — 변수별 기여도 (%)",
        xaxis_title="기여도 %", yaxis_title="",
        height=600,
    )
    out = OUT / "06_pca_contribution.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 07. Risk distribution ─────────
def viz_risk_dist(df: pd.DataFrame) -> Path:
    gu_order = (
        df.groupby("gu_name")["risk_score"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("전체 격자 risk_score 분포",
                                        "자치구별 box plot"),
                        column_widths=[0.4, 0.6])
    fig.add_trace(go.Histogram(x=df["risk_score"], nbinsx=80,
                               marker_color="#e74c3c",
                               name="격자"), row=1, col=1)
    for gu in gu_order:
        sub = df[df["gu_name"] == gu]
        fig.add_trace(go.Box(y=sub["risk_score"], name=gu,
                             boxpoints=False, marker_color="#3498db"),
                      row=1, col=2)
    fig.update_layout(height=600, showlegend=False,
                      title_text="위험도 점수 분포 — 전체 + 자치구별")
    out = OUT / "07_risk_distribution.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 08. Top 100 grid table ─────────
def viz_top_table(df: pd.DataFrame) -> Path:
    top = df.nlargest(100, "risk_score")[
        ["grid_id_50m", "gu_name", "ADM_NM", "risk_score",
         "intersection_density_200m", "road_complexity_200m",
         "추정치매환자수_총합", "cctv_count_within_200m",
         "missing_estimated"]
    ].round(3)
    fig = go.Figure(data=[go.Table(
        header=dict(values=["grid_id", "자치구", "행정동", "위험도",
                            "교차로밀도", "도로복잡도", "치매환자",
                            "CCTV", "실종추정"],
                    fill_color="#34495e", font_color="white", align="center"),
        cells=dict(values=[top[c] for c in top.columns],
                   fill_color=[["#fafafa" if i%2==0 else "#fff"
                                for i in range(len(top))]],
                   align="center"),
    )])
    fig.update_layout(title="위험도 Top 100 격자 — 우선 정책 대상",
                      height=900)
    out = OUT / "08_top_grids_table.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


# ───────── 99. Combined dashboard ─────────
def build_dashboard():
    html = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<title>치매 노인 실종 분석 대시보드 — 서울 50m 격자</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Apple SD Gothic Neo", sans-serif; margin: 0;
         background: #f4f5f7; color: #2c3e50; }
  header { background: linear-gradient(135deg, #c0392b 0%, #e67e22 100%);
           color: white; padding: 36px 28px; }
  header h1 { margin: 0 0 8px; font-size: 28px; }
  header p { margin: 0; opacity: 0.92; }
  .container { max-width: 1300px; margin: 0 auto; padding: 24px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 14px; margin: -32px 0 20px; }
  .stat { background: white; padding: 18px; border-radius: 10px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.08); text-align: center; }
  .stat-num { font-size: 30px; font-weight: bold; color: #c0392b; }
  .stat-lbl { font-size: 12px; color: #7f8c8d; margin-top: 6px; }
  .card { background: white; padding: 24px; margin-bottom: 18px;
          border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
  .card h2 { margin: 0 0 12px; color: #34495e; font-size: 18px; }
  .card p { color: #5a6c7d; font-size: 13px; margin-bottom: 12px; }
  .frame { width: 100%; border: 1px solid #e8ecef; border-radius: 6px; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 1000px) { .row { grid-template-columns: 1fr; } }
  footer { text-align: center; padding: 24px; color: #95a5a6; font-size: 12px; }
  a { color: #3498db; text-decoration: none; }
  .pill { display: inline-block; padding: 3px 10px; border-radius: 12px;
          background: #ecf0f1; font-size: 11px; margin-right: 4px; }
</style></head>
<body>
<header>
  <h1>🧠 치매 노인 실종 분석 대시보드</h1>
  <p>서울 50m 격자 단위 분석 · PCA 위험도 + 공간 자기상관 + 자치구 비교</p>
</header>
<div class="container">
  <div class="stats">
    <div class="stat"><div class="stat-num">120,821</div><div class="stat-lbl">분석 격자 (50m × 50m)</div></div>
    <div class="stat"><div class="stat-num">741</div><div class="stat-lbl">실종 신고 (15개월)</div></div>
    <div class="stat"><div class="stat-num">14.17</div><div class="stat-lbl">종로구 발생률 / 10k (1위)</div></div>
    <div class="stat"><div class="stat-num">192만</div><div class="stat-lbl">서울 65+ 인구</div></div>
    <div class="stat"><div class="stat-num">14</div><div class="stat-lbl">PCA feature</div></div>
  </div>

  <div class="card">
    <h2>📍 1. 서울 전체 위험도 격자 지도</h2>
    <p>40,000 격자 샘플의 PCA 위험도 점수. <span class="pill">호버: 격자별 상세</span> 빨강: 고위험.</p>
    <iframe src="01_seoul_risk_map.html" class="frame" height="780"></iframe>
  </div>

  <div class="card">
    <h2>🔍 2. 종로구 zoom — 격자별 환경 정보</h2>
    <p>발생률 1위 종로구 격자별 위험도. 호버 시 행정동·교차로·CCTV·치매환자 수 확인 가능.</p>
    <iframe src="02_jongno_zoom.html" class="frame" height="780"></iframe>
  </div>

  <div class="row">
    <div class="card">
      <h2>📊 3. 자치구 3패널 비교</h2>
      <p>절대 건수 vs 발생률 vs PCA 위험도. 인구 보정 시 종로·중구·용산이 상위.</p>
      <iframe src="03_gu_3panel.html" class="frame" height="780"></iframe>
    </div>
    <div class="card">
      <h2>🎯 4. 도심 4구 Radar</h2>
      <p>종로·중구·용산·서대문 환경 프로필. 종로가 인프라·도로복잡도에서 두드러짐.</p>
      <iframe src="04_central4_radar.html" class="frame" height="680"></iframe>
    </div>
  </div>

  <div class="row">
    <div class="card">
      <h2>🔗 5. 상관 매트릭스</h2>
      <p>16개 변수 상관계수. 도로 인프라 변수 간 강한 양의 상관 확인.</p>
      <iframe src="05_correlation_heatmap.html" class="frame" height="750"></iframe>
    </div>
    <div class="card">
      <h2>📈 6. PCA 변수 기여도</h2>
      <p>위험도 점수에 각 변수가 기여하는 비율 (%).</p>
      <iframe src="06_pca_contribution.html" class="frame" height="630"></iframe>
    </div>
  </div>

  <div class="card">
    <h2>📉 7. 위험도 분포</h2>
    <p>전체 격자 분포 + 자치구별 box plot.</p>
    <iframe src="07_risk_distribution.html" class="frame" height="630"></iframe>
  </div>

  <div class="card">
    <h2>🎯 8. 위험도 Top 100 격자 — 정책 우선 대상</h2>
    <p>모델이 식별한 가장 위험한 100개 격자. 발견·정비 우선순위.</p>
    <iframe src="08_top_grids_table.html" class="frame" height="930"></iframe>
  </div>
</div>
<footer>
  서울시 데이터 분석 경진대회 · 2026 · 50m grid spatial analysis
  · <a href="https://github.com/hyeminhassomething/seoul-dementia-discovery-model">GitHub</a>
</footer>
</body></html>"""
    out = OUT / "99_dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


def main():
    print("[load]")
    df, features, weights = load_data()
    print(f"  rows: {len(df):,}")

    print("[viz] generating interactive HTMLs...")
    paths = []
    paths.append(viz_seoul_map(df));    print("  ✔ 01 seoul map")
    paths.append(viz_jongno(df));        print("  ✔ 02 jongno zoom")
    paths.append(viz_gu_3panel(df));     print("  ✔ 03 gu 3panel")
    paths.append(viz_radar(df));         print("  ✔ 04 central4 radar")
    paths.append(viz_corr(df));          print("  ✔ 05 correlation")
    paths.append(viz_pca_contrib(weights, features)); print("  ✔ 06 pca")
    paths.append(viz_risk_dist(df));     print("  ✔ 07 distribution")
    paths.append(viz_top_table(df));     print("  ✔ 08 top table")

    dash = build_dashboard()
    print(f"  ✔ 99 dashboard → {dash.relative_to(PROJECT_ROOT)}")

    print(f"\n✅ 모든 출력: {OUT.relative_to(PROJECT_ROOT)}/")
    print(f"   대시보드 열기: open {dash}")


if __name__ == "__main__":
    main()
