"""
Slide 5 (기존 분석의 한계) visualizations — both options.

A안: L5A_composite.png — single 3-column composite (limit ①②③ + 해결책)
B안: 4 separate transparent PNGs for user manual layout
  L5B_scale.png       — ① scale mismatch (자치구 vs 격자)
  L5B_choropleth.png  — ② hidden outliers (자치구 LL → 격자 HL)
  L5B_mapping.png     — ③ 1:N vs N:N policy mapping
  L5B_solution.png    — WHERE × WHY = WHAT 3-layer

Palette consistent with slide 3:
  DARK #1F3A5F  ACCENT #2E5C9E  RED #D9534F  GREY #9AA5B1  LIGHTGREY #C9D4E5
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUT = Path('data/processed/report_viz/v2')
OUT.mkdir(parents=True, exist_ok=True)

C_DARK = '#1F3A5F'
C_ACCENT = '#2E5C9E'
C_RED = '#D9534F'
C_AMBER = '#F0AD4E'
C_GREEN = '#5CB85C'
C_GREY = '#9AA5B1'
C_LIGHTGREY = '#C9D4E5'
C_BOXBG = '#E8EEF7'


# ============================================================
# Helper drawing primitives
# ============================================================
def scale_mismatch(ax):
    """Draw true-to-scale mismatch: big 25km² rectangle + tiny 250m square inside."""
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.axis('off')
    # Big box (자치구 25km² → use 7×5 abstract units)
    big = Rectangle((1, 1.2), 7.5, 5, facecolor=C_LIGHTGREY,
                    edgecolor=C_ACCENT, linewidth=2.5)
    ax.add_patch(big)
    ax.text(1.2, 5.8, '자치구  25 km²', fontsize=11, fontweight='bold', color=C_ACCENT)
    # Tiny dot near center — area should be 1/10000 of big box visually impossible,
    # use a tiny red square + arrow to magnify
    small = Rectangle((4.7, 3.4), 0.08, 0.08, facecolor=C_RED, edgecolor='white')
    ax.add_patch(small)
    # Magnified callout box
    mag = Rectangle((6.5, 1.6), 1.6, 1.6, facecolor='white', edgecolor=C_RED,
                    linewidth=2, linestyle='--')
    ax.add_patch(mag)
    mag_inner = Rectangle((7.0, 2.05), 0.6, 0.7, facecolor=C_RED, edgecolor='white')
    ax.add_patch(mag_inner)
    ax.annotate('', xy=(6.5, 2.4), xytext=(4.85, 3.45),
                arrowprops=dict(arrowstyle='-', color=C_RED, lw=1, linestyle=':'))
    ax.text(7.3, 1.3, '50 m 격자', fontsize=9, color=C_RED, ha='center')
    # Big number
    ax.text(5, 7.3, '× 10,000', fontsize=24, fontweight='bold',
            color=C_RED, ha='center')
    ax.text(5, 0.5, '예산은 위험·안전 격자에 일률 분산',
            fontsize=9, color=C_GREY, ha='center', style='italic')


def hidden_outliers(ax):
    """Choropleth-style: left = aggregated (all light), right = grid (red dots scattered)."""
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis('off')

    # Left mini-map: aggregated district view (uniform light)
    left = Rectangle((0.5, 1.5), 4.5, 4.5, facecolor=C_LIGHTGREY,
                     edgecolor=C_ACCENT, linewidth=1.5)
    ax.add_patch(left)
    # Subdivisions (grid lines) — districts uniform
    for x in [2.0, 3.5]:
        ax.plot([x, x], [1.5, 6], color=C_ACCENT, lw=0.8, alpha=0.6)
    for y in [3.0, 4.5]:
        ax.plot([0.5, 5.0], [y, y], color=C_ACCENT, lw=0.8, alpha=0.6)
    ax.text(2.75, 6.4, '자치구 단위', fontsize=11, fontweight='bold',
            color=C_ACCENT, ha='center')
    ax.text(2.75, 0.9, '"안전" (LL 평균)', fontsize=10, color=C_GREY, ha='center')

    # Arrow
    ax.annotate('', xy=(6.5, 3.75), xytext=(5.3, 3.75),
                arrowprops=dict(arrowstyle='-|>', color=C_DARK, lw=2.5))
    ax.text(5.9, 4.3, '줌인', fontsize=9, color=C_DARK, ha='center')

    # Right mini-map: grid with scattered HL dots
    right = Rectangle((7.0, 1.5), 4.5, 4.5, facecolor=C_LIGHTGREY,
                      edgecolor=C_ACCENT, linewidth=1.5)
    ax.add_patch(right)
    # Fine grid
    for x in np.arange(7.2, 11.5, 0.35):
        ax.plot([x, x], [1.5, 6], color='white', lw=0.3, alpha=0.5)
    for y in np.arange(1.7, 6, 0.35):
        ax.plot([7.0, 11.5], [y, y], color='white', lw=0.3, alpha=0.5)
    # HL outlier dots
    np.random.seed(7)
    for _ in range(38):
        x = 7.2 + np.random.uniform(0, 4.1)
        y = 1.7 + np.random.uniform(0, 4.1)
        ax.add_patch(Circle((x, y), 0.10, facecolor=C_RED, edgecolor='white',
                            linewidth=0.5))
    ax.text(9.25, 6.4, '50 m 격자 단위', fontsize=11, fontweight='bold',
            color=C_RED, ha='center')
    ax.text(9.25, 0.9, 'HL 1,092개 노출', fontsize=10, color=C_RED, ha='center',
            fontweight='bold')

    ax.text(6, 7.4, '평균이 가린 사각지대', fontsize=14, fontweight='bold',
            color=C_DARK, ha='center')


def mapping_compare(ax):
    """1:N vs N:N: left = 5 terrains → 1 policy, right = 5 → 5."""
    ax.set_xlim(0, 12); ax.set_ylim(0, 8)
    ax.axis('off')

    icons = ['상권', '골목', '경사', '인구', '복합']
    labels = ['도심', '미로', '언덕', '고령', '중첩']

    # Left: 5 terrains → 1 policy
    for i, (icon, lab) in enumerate(zip(icons, labels)):
        y = 6.5 - i * 1.1
        ax.add_patch(Circle((0.8, y), 0.42, facecolor=C_LIGHTGREY,
                            edgecolor=C_ACCENT, linewidth=1.5))
        ax.text(0.8, y, icon, fontsize=9, ha='center', va='center',
                color=C_DARK, fontweight='bold')
        ax.text(1.55, y, lab, fontsize=10, color=C_DARK, va='center')
        # converging arrow to single policy box
        ax.plot([2.4, 3.8], [y, 4], color=C_GREY, lw=1.2, alpha=0.7)
    # Single policy box
    pol_left = FancyBboxPatch((3.8, 3.5), 1.6, 1.0,
                              boxstyle="round,pad=0.05",
                              facecolor=C_GREY, edgecolor='white', linewidth=2)
    ax.add_patch(pol_left)
    ax.text(4.6, 4, '일률\n정책', fontsize=11, fontweight='bold',
            color='white', ha='center', va='center')
    ax.text(2.8, 7.4, '기존 — 1:N', fontsize=13, fontweight='bold',
            color=C_GREY, ha='center')

    # Divider
    ax.plot([6, 6], [0.8, 7.2], color=C_LIGHTGREY, lw=1, linestyle='--')

    # Right: 5 terrains → 5 policies (1:1 mapping)
    policy_colors = [C_RED, C_AMBER, C_ACCENT, C_GREEN, C_DARK]
    pol_names = ['CCTV', '안내표지', '손잡이', '안심네트', '풀패키지']
    for i, (icon, lab, col, pn) in enumerate(zip(icons, labels, policy_colors, pol_names)):
        y = 6.5 - i * 1.1
        ax.add_patch(Circle((6.8, y), 0.42, facecolor=C_LIGHTGREY,
                            edgecolor=C_ACCENT, linewidth=1.5))
        ax.text(6.8, y, icon, fontsize=9, ha='center', va='center',
                color=C_DARK, fontweight='bold')
        ax.text(7.55, y, lab, fontsize=10, color=C_DARK, va='center')
        # 1:1 arrow
        ax.plot([8.4, 9.5], [y, y], color=col, lw=1.8, alpha=0.85)
        # Policy badge
        pol_box = FancyBboxPatch((9.5, y-0.32), 2.0, 0.65,
                                 boxstyle="round,pad=0.04",
                                 facecolor=col, edgecolor='white', linewidth=1.5)
        ax.add_patch(pol_box)
        ax.text(10.5, y, pn, fontsize=9, fontweight='bold', color='white',
                ha='center', va='center')
    ax.text(9, 7.4, '본 연구 — N:N 매핑', fontsize=13, fontweight='bold',
            color=C_RED, ha='center')


def solution_layers(ax):
    """3-Layer: WHERE × WHY = WHAT"""
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis('off')

    # Three stacked layer boxes
    layers = [
        (7.5, 'WHERE',  'LISA · 공간 자기상관',   'HH 52,298 · HL 1,092', C_ACCENT),
        (5.5, 'WHY',    'HDBSCAN · 환경 군집',    '96 valid clusters',     C_AMBER),
        (3.5, 'WHAT',   '정책 매핑',              '환경 유형 × 정책 5종',   C_RED),
    ]
    for y, key, sub, det, col in layers:
        box = FancyBboxPatch((1, y), 8, 1.5, boxstyle="round,pad=0.05",
                             facecolor=col, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(1.5, y+1.1, key, fontsize=18, fontweight='bold', color='white')
        ax.text(1.5, y+0.55, sub, fontsize=11, color='white')
        ax.text(8.5, y+0.75, det, fontsize=11, fontweight='bold', color='white',
                ha='right')

    # Down arrows between layers
    for y in [5.05, 3.05]:
        ax.annotate('', xy=(5, y), xytext=(5, y+0.45),
                    arrowprops=dict(arrowstyle='-|>', color=C_DARK, lw=2.5))

    # Bottom result
    ax.text(5, 2.2, '격자 단위 비지도 진단 도구', fontsize=13, fontweight='bold',
            color=C_DARK, ha='center')
    ax.text(5, 9.3, '본 연구 — 3축 결합 진단', fontsize=15, fontweight='bold',
            color=C_DARK, ha='center')


# ============================================================
# Option A — single composite (4-column horizontal)
# ============================================================
fig = plt.figure(figsize=(20, 7.5), dpi=150, facecolor='white')

def make_box(rect, label, label_color):
    l, b, w, h = rect
    ax = fig.add_axes([l, b, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    bg = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                        boxstyle="round,pad=0.0,rounding_size=0.02",
                        facecolor=C_BOXBG, edgecolor='none')
    ax.add_patch(bg)
    # Numbered badge top-left
    ax.add_patch(Circle((0.07, 0.92), 0.045, facecolor=label_color))
    ax.text(0.07, 0.92, label[0], ha='center', va='center',
            fontsize=15, fontweight='bold', color='white')
    ax.text(0.14, 0.92, label[2:], fontsize=14, fontweight='bold',
            color=C_DARK, va='center')
    return ax

# 4 columns layout
COL_W = 0.235
GAP = 0.012
LEFT0 = 0.012
BOT = 0.08
HEIGHT = 0.89

rects = [
    (LEFT0 + i*(COL_W + GAP), BOT, COL_W, HEIGHT) for i in range(4)
]

# Box 1
b1 = make_box(rects[0], '① 공간 해상도 미스매치', C_RED)
b1.text(0.5, 0.83, '자치구 25 km² vs 50 m 격자', ha='center',
        fontsize=11, color=C_DARK, fontweight='bold')
chart_ax = fig.add_axes([rects[0][0]+0.01, BOT+0.10, COL_W-0.02, 0.55])
scale_mismatch(chart_ax)
b1.text(0.5, 0.09, '예산 1만배 희석 → 위험·안전 동일 처방',
        ha='center', fontsize=10, color=C_DARK, style='italic')

# Box 2
b2 = make_box(rects[1], '② 사각지대 식별 불가', C_RED)
b2.text(0.5, 0.83, '자치구 평균에 묻힌 HL 1,092개', ha='center',
        fontsize=11, color=C_DARK, fontweight='bold')
chart_ax = fig.add_axes([rects[1][0]+0.01, BOT+0.13, COL_W-0.02, 0.50])
hidden_outliers(chart_ax)
b2.text(0.5, 0.09, '본 연구: LISA로 outlier 직접 식별',
        ha='center', fontsize=10, color=C_DARK, style='italic')

# Box 3
b3 = make_box(rects[2], '③ 원인 분석 부재', C_RED)
b3.text(0.5, 0.83, '같은 위험 ≠ 같은 처방', ha='center',
        fontsize=11, color=C_DARK, fontweight='bold')
chart_ax = fig.add_axes([rects[2][0]+0.005, BOT+0.10, COL_W-0.01, 0.58])
mapping_compare(chart_ax)
b3.text(0.5, 0.05, '본 연구: 환경 유형 5종 × 정책 5종 매핑',
        ha='center', fontsize=10, color=C_DARK, style='italic')

# Box 4 — solution
b4 = make_box(rects[3], '★ 본 연구의 해결책', C_GREEN)
chart_ax = fig.add_axes([rects[3][0]+0.005, BOT+0.05, COL_W-0.01, 0.78])
solution_layers(chart_ax)

# Footer
fig.text(0.012, 0.02,
         '출처: 서울특별시(자치구 면적 2023) · 본 연구 LISA·HDBSCAN 분석 결과 · 차성준 cuML 노트북',
         fontsize=8, color='#6B7682')

plt.savefig(OUT/'L5A_composite.png', dpi=150, bbox_inches='tight',
            facecolor='white', pad_inches=0.12)
plt.close()
print(f'OK -> L5A_composite.png ({(OUT/"L5A_composite.png").stat().st_size//1024} KB)')


# ============================================================
# Option B — 4 transparent charts
# ============================================================
# B-1 Scale
fig = plt.figure(figsize=(7, 5), dpi=200, facecolor='none')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.92])
scale_mismatch(ax)
plt.savefig(OUT/'L5B_scale.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f'OK -> L5B_scale.png')

# B-2 Choropleth
fig = plt.figure(figsize=(8, 5.5), dpi=200, facecolor='none')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.92])
hidden_outliers(ax)
plt.savefig(OUT/'L5B_choropleth.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f'OK -> L5B_choropleth.png')

# B-3 Mapping
fig = plt.figure(figsize=(9, 5.5), dpi=200, facecolor='none')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.92])
mapping_compare(ax)
plt.savefig(OUT/'L5B_mapping.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f'OK -> L5B_mapping.png')

# B-4 Solution
fig = plt.figure(figsize=(7, 6), dpi=200, facecolor='none')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.92])
solution_layers(ax)
plt.savefig(OUT/'L5B_solution.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print(f'OK -> L5B_solution.png')
