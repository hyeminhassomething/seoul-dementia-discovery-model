"""
22_sensitivity_analysis.py
가중치 민감도 분석 — 7개 feature 가중치 각각 ±20% 변동 시
상위 격자 순위가 얼마나 안정적인지 측정.

[목적]
  현재 가중치(prior)가 임의 설정에 가깝지만,
  ±20% 흔들어도 상위 N개 격자 추천이 크게 바뀌지 않으면
  → 모델이 "robust" → 가중치 부정확해도 의사결정 신뢰성 ↑

[입력]
  data/processed/grid50_master_features.csv     (정규화 전 raw feature)
  data/processed/grid50_discovery_probability.csv (현재 모델 출력)

[출력]
  data/processed/sensitivity_overlap.csv   각 시나리오의 top-N overlap
  data/processed/sensitivity_summary.csv   요약 통계
  data/processed/sensitivity_chart.png     발표용 그래프

[시나리오]
  baseline                          : 원래 가중치
  intersection ±20%                 : 교차로만 흔들기 (×2 시나리오)
  road_complexity ±20%              : ...
  ... 7개 feature × 상하 = 14개 시나리오
  random_perturbation               : 모든 가중치 동시에 ±10% 무작위
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"

BASELINE_WEIGHTS = {
    "intersection":      0.35,
    "road_complexity":   0.25,
    "residential":       0.15,
    "rest_area":         0.10,
    "cctv":              0.08,
    "facility":          0.04,
    "population_proxy":  0.03,
}

NORM_COLS = {
    "intersection":     "norm_intersection",
    "road_complexity":  "norm_road_complexity",
    "residential":      "norm_residential",
    "rest_area":        "norm_rest_area",
    "cctv":             "norm_cctv",
    "facility":         "norm_facility",
    "population_proxy": "norm_population_proxy",
}

TOP_NS = [50, 100, 500, 1000]
PERTURBATION = 0.20  # ±20%


def compute_score(norm_df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """정규화된 feature + 가중치 → 발견확률 점수."""
    score = pd.Series(0.0, index=norm_df.index)
    for k, w in weights.items():
        score += norm_df[NORM_COLS[k]] * w
    return score


def renormalize(weights: dict[str, float]) -> dict[str, float]:
    """가중치 합 = 1로 재정규화."""
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def overlap(a: pd.Series, b: pd.Series, n: int) -> int:
    """두 점수 시리즈의 상위 N 격자 ID overlap."""
    top_a = set(a.nlargest(n).index)
    top_b = set(b.nlargest(n).index)
    return len(top_a & top_b)


def main() -> None:
    print("[load] grid50_discovery_probability.csv")
    df = pd.read_csv(PROCESSED / "grid50_discovery_probability.csv")
    df = df.set_index("grid_id_50m")

    # baseline 점수
    baseline = compute_score(df, BASELINE_WEIGHTS)
    print(f"  baseline mean={baseline.mean():.3f}, max={baseline.max():.3f}")

    results = []

    # ─── 시나리오 1: 각 feature ±20% 흔들기 ───
    for feat in BASELINE_WEIGHTS:
        for sign, label in [(+1, "+20%"), (-1, "-20%")]:
            w = BASELINE_WEIGHTS.copy()
            w[feat] = w[feat] * (1 + sign * PERTURBATION)
            w = renormalize(w)
            score = compute_score(df, w)

            row = {"scenario": f"{feat}_{label}", "feature": feat, "delta": label}
            for n in TOP_NS:
                row[f"top{n}_overlap"] = overlap(baseline, score, n)
                row[f"top{n}_overlap_pct"] = overlap(baseline, score, n) / n
            results.append(row)

    # ─── 시나리오 2: 모든 가중치 동시 ±10% 무작위 (50회 평균) ───
    print("\n[random perturbation] 50 trials of ±10% on all weights")
    rng = np.random.default_rng(42)
    rand_overlaps = {n: [] for n in TOP_NS}
    for trial in range(50):
        w = {k: v * (1 + rng.uniform(-0.10, 0.10)) for k, v in BASELINE_WEIGHTS.items()}
        w = renormalize(w)
        score = compute_score(df, w)
        for n in TOP_NS:
            rand_overlaps[n].append(overlap(baseline, score, n))
    rand_row = {"scenario": "random_±10%_50trials", "feature": "all", "delta": "±10%"}
    for n in TOP_NS:
        rand_row[f"top{n}_overlap"] = float(np.mean(rand_overlaps[n]))
        rand_row[f"top{n}_overlap_pct"] = float(np.mean(rand_overlaps[n])) / n
    results.append(rand_row)

    # ─── 시나리오 3: 극단 케이스 — 한 feature 가중치 0 (제거) ───
    print("\n[ablation] each feature removed (weight=0)")
    for feat in BASELINE_WEIGHTS:
        w = BASELINE_WEIGHTS.copy()
        w[feat] = 0.0
        w = renormalize(w)
        score = compute_score(df, w)
        row = {"scenario": f"{feat}_REMOVED", "feature": feat, "delta": "ablation"}
        for n in TOP_NS:
            row[f"top{n}_overlap"] = overlap(baseline, score, n)
            row[f"top{n}_overlap_pct"] = overlap(baseline, score, n) / n
        results.append(row)

    res_df = pd.DataFrame(results)
    out_path = PROCESSED / "sensitivity_overlap.csv"
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ saved: {out_path.relative_to(PROJECT_ROOT)}")

    # ─── 요약 통계 ───
    print("\n" + "=" * 70)
    print("[±20% 변동 시나리오 — top100 overlap 평균/최소]")
    sub = res_df[res_df["delta"].isin(["+20%", "-20%"])]
    print(f"  평균 overlap (top100): {sub['top100_overlap'].mean():.1f} / 100")
    print(f"  최소 overlap (top100): {sub['top100_overlap'].min():.0f} / 100")
    print(f"  최소 발생 시나리오: {sub.loc[sub['top100_overlap'].idxmin(), 'scenario']}")

    print("\n[모든 가중치 동시 ±10% (random)]")
    rand = res_df[res_df["scenario"].str.startswith("random")]
    for n in TOP_NS:
        print(f"  top{n} overlap 평균: {rand[f'top{n}_overlap'].iloc[0]:.1f} / {n} "
              f"({rand[f'top{n}_overlap_pct'].iloc[0]:.1%})")

    print("\n[Ablation — feature 제거 시 top100 overlap]")
    abl = res_df[res_df["delta"] == "ablation"].sort_values("top100_overlap")
    print(abl[["feature", "top100_overlap"]].to_string(index=False))

    print("\n[해석]")
    avg_top100 = sub["top100_overlap"].mean()
    if avg_top100 >= 90:
        verdict = "매우 robust — 가중치 정확하지 않아도 결과 신뢰 가능"
    elif avg_top100 >= 75:
        verdict = "robust — 발표·심사 변호 가능"
    elif avg_top100 >= 60:
        verdict = "보통 — 가중치 캘리브레이션 권장"
    else:
        verdict = "fragile — 가중치 정확히 결정 필요 (Logistic regression 등)"
    print(f"  → ±20% 변동 시 평균 top100 overlap = {avg_top100:.1f}%")
    print(f"  → 결론: {verdict}")

    # ─── 시각화 ───
    print("\n[chart] generating sensitivity_chart.png ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (좌) ±20% 영향
    sub_pos = res_df[res_df["delta"] == "+20%"].set_index("feature")
    sub_neg = res_df[res_df["delta"] == "-20%"].set_index("feature")
    feats = list(BASELINE_WEIGHTS.keys())
    x = np.arange(len(feats))
    w = 0.35
    axes[0].bar(x - w/2, [sub_pos.loc[f, "top100_overlap"] for f in feats],
                w, label="+20%", color="#3498db")
    axes[0].bar(x + w/2, [sub_neg.loc[f, "top100_overlap"] for f in feats],
                w, label="-20%", color="#e74c3c")
    axes[0].axhline(100, color="gray", linestyle="--", alpha=0.3, label="perfect")
    axes[0].axhline(80, color="green", linestyle=":", alpha=0.5, label="robust threshold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feats, rotation=30, ha="right")
    axes[0].set_ylabel("Top-100 overlap with baseline")
    axes[0].set_title("Sensitivity: ±20% on each weight (top-100)")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # (우) Top N별 overlap (평균)
    sub_2 = res_df[res_df["delta"].isin(["+20%", "-20%"])]
    avg_by_n = [sub_2[f"top{n}_overlap_pct"].mean() * 100 for n in TOP_NS]
    rand_by_n = [rand[f"top{n}_overlap_pct"].iloc[0] * 100 for n in TOP_NS]
    axes[1].plot(TOP_NS, avg_by_n, "o-", label="±20% on single weight (avg)", linewidth=2)
    axes[1].plot(TOP_NS, rand_by_n, "s-", label="±10% all weights random", linewidth=2)
    axes[1].axhline(80, color="green", linestyle=":", alpha=0.5)
    axes[1].set_xscale("log")
    axes[1].set_xticks(TOP_NS)
    axes[1].set_xticklabels(TOP_NS)
    axes[1].set_xlabel("Top-N grids")
    axes[1].set_ylabel("Overlap %")
    axes[1].set_title("Robustness across N")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    chart_path = PROCESSED / "sensitivity_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  saved: {chart_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
