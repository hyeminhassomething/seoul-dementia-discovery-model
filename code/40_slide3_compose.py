"""
Generate slide 3 composite visualization (치매 실종 문제) — 3-box layout
matching the reference PDF style.

Layout (1920×900 canvas, will be placed below subtitle on slide 3):
- Top-left  [현황]      : 65+ 192만 중 치매 20만 도넛 + 사망·미발견 49건 강조
- Top-right [심각성]    : 도시 80% vs 시골 20% 길 잃음 비교
- Bottom    [구조적 공백]: 자치구 49km² 큰 박스 + 250m 반경 작은 원 + 5배 텍스트
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUT = Path('data/processed/report_viz/v2/S3_situation.png')
OUT.parent.mkdir(parents=True, exist_ok=True)

# Canvas
fig = plt.figure(figsize=(19.2, 9.0), dpi=160, facecolor='white')

# Color palette (cool blue tone like reference)
C_BG_BOX = '#E8EEF7'
C_DARK = '#1F3A5F'
C_ACCENT = '#2E5C9E'
C_RED = '#D9534F'
C_AMBER = '#F0AD4E'
C_GREY = '#9AA5B1'

# === Box positions (figure coords 0..1) ===
# top-left: [현황]
# top-right: [심각성]
# bottom: [구조적 공백]
BOX_HYUN = (0.025, 0.42, 0.475, 0.55)      # left, bottom, width, height
BOX_SIM  = (0.515, 0.42, 0.46, 0.55)
BOX_GAP  = (0.025, 0.025, 0.95, 0.36)

def add_box(rect, label, label_color=C_DARK):
    l, b, w, h = rect
    ax = fig.add_axes([l, b, w, h])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    # rounded background
    bg = FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                        boxstyle="round,pad=0.0,rounding_size=0.02",
                        linewidth=0, facecolor=C_BG_BOX, transform=ax.transAxes)
    ax.add_patch(bg)
    # label bar (left vertical)
    ax.add_patch(Rectangle((0.025, 0.86), 0.008, 0.10,
                           facecolor=label_color, transform=ax.transAxes))
    ax.text(0.05, 0.91, label, fontsize=22, fontweight='bold',
            color=label_color, va='center', transform=ax.transAxes)
    return ax

# =========================================================
# Box 1 — [현황]
# =========================================================
ax1 = add_box(BOX_HYUN, '[현황]')

# Sub-section A: donut chart (left half)
donut_ax = fig.add_axes([BOX_HYUN[0]+0.018, BOX_HYUN[1]+0.10,
                         BOX_HYUN[2]*0.45, BOX_HYUN[3]*0.62])
sizes = [20, 172]  # 만명: 치매환자 vs 65+ 비치매
colors = [C_RED, '#C9D4E5']
donut_ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
             wedgeprops=dict(width=0.32, edgecolor='white', linewidth=2))
donut_ax.text(0, 0.08, '10.4%', ha='center', va='center',
              fontsize=26, fontweight='bold', color=C_RED)
donut_ax.text(0, -0.18, '20만 명', ha='center', va='center',
              fontsize=11, color=C_DARK)
donut_ax.set_aspect('equal')

ax1.text(0.25, 0.18, '서울 65+ 192만 중\n추정 치매환자 약 20만 명',
         ha='center', va='top', fontsize=12, color=C_DARK,
         transform=ax1.transAxes, linespacing=1.4)

# Sub-section B: 741건 / 49건 사망·미발견 (right half)
ax1.text(0.74, 0.78, '실종 신고', ha='center', fontsize=12,
         color=C_DARK, transform=ax1.transAxes)
ax1.text(0.74, 0.66, '741건', ha='center', fontsize=34, fontweight='bold',
         color=C_ACCENT, transform=ax1.transAxes)
ax1.text(0.74, 0.55, '(15개월 누적, 2023.09~2024.11)',
         ha='center', fontsize=9, color=C_GREY, transform=ax1.transAxes)

# Down arrow
ax1.annotate('', xy=(0.74, 0.40), xytext=(0.74, 0.50),
             xycoords='axes fraction',
             arrowprops=dict(arrowstyle='-|>', color=C_RED, lw=2))

ax1.text(0.74, 0.34, '사망·미발견', ha='center', fontsize=12,
         color=C_DARK, transform=ax1.transAxes)
ax1.text(0.74, 0.20, '49건', ha='center', fontsize=34, fontweight='bold',
         color=C_RED, transform=ax1.transAxes)
ax1.text(0.74, 0.10, '(6.6% · 4시간 초과 시 생존율 급락)',
         ha='center', fontsize=9, color=C_GREY, transform=ax1.transAxes)

# =========================================================
# Box 2 — [심각성]
# =========================================================
ax2 = add_box(BOX_SIM, '[심각성]')

ax2.text(0.50, 0.80, '도시 거주 치매 노인의 80%가 외출 시 길을 잃음',
         ha='center', fontsize=13, color=C_DARK, fontweight='bold',
         transform=ax2.transAxes)
ax2.text(0.50, 0.73, '시골 거주자 20% 대비 4배 — 도시 구조 자체가 위험 변수',
         ha='center', fontsize=10, color=C_GREY, transform=ax2.transAxes)

# Horizontal bar comparison
bar_ax = fig.add_axes([BOX_SIM[0]+0.06, BOX_SIM[1]+0.08,
                       BOX_SIM[2]*0.85, BOX_SIM[3]*0.32])
bar_ax.barh([1, 0], [80, 20], color=[C_RED, C_GREY], height=0.55,
            edgecolor='white', linewidth=2)
bar_ax.set_yticks([0, 1])
bar_ax.set_yticklabels(['시골 거주', '도시 거주'], fontsize=12, color=C_DARK)
bar_ax.set_xlim(0, 100)
bar_ax.set_xticks([])
bar_ax.spines['top'].set_visible(False)
bar_ax.spines['right'].set_visible(False)
bar_ax.spines['bottom'].set_visible(False)
bar_ax.spines['left'].set_visible(False)
bar_ax.text(82, 1, '80%', va='center', fontsize=22, fontweight='bold', color=C_RED)
bar_ax.text(22, 0, '20%', va='center', fontsize=18, fontweight='bold', color=C_GREY)
bar_ax.text(50, -0.85, '길 잃음 비율 (한국일보 2023.09)',
            ha='center', fontsize=9, color=C_GREY)
bar_ax.invert_yaxis()

# =========================================================
# Box 3 — [구조적 공백] (scale mismatch visualization)
# =========================================================
ax3 = add_box(BOX_GAP, '[구조적 공백]')

ax3.text(0.50, 0.78, '정책 집행 단위 ≠ 실제 사건 단위',
         ha='center', fontsize=15, color=C_DARK, fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0.50, 0.69, '현재 자치구 단위 정책 vs 실종자 실제 발견 250m 반경',
         ha='center', fontsize=11, color=C_GREY, transform=ax3.transAxes)

# Scale visualization: big square (autonomous district) with tiny circle inside
# Left side: autonomous district area
vis_ax = fig.add_axes([BOX_GAP[0]+0.08, BOX_GAP[1]+0.05,
                       BOX_GAP[2]*0.35, BOX_GAP[3]*0.55])
vis_ax.set_xlim(0, 10); vis_ax.set_ylim(0, 7)
vis_ax.axis('off')
# Big square: 자치구 49km²
big = Rectangle((1, 0.5), 7, 6, facecolor='none', edgecolor=C_ACCENT,
                linewidth=2.5, linestyle='-')
vis_ax.add_patch(big)
vis_ax.text(1.2, 6.0, '자치구', fontsize=12, fontweight='bold', color=C_ACCENT)
vis_ax.text(1.2, 5.55, '평균 49 km²', fontsize=11, color=C_ACCENT)
# Tiny dot near center
small = Circle((4.5, 3.0), 0.18, facecolor=C_RED, edgecolor='white', linewidth=1.5)
vis_ax.add_patch(small)
# annotation arrow
vis_ax.annotate('실종자 발견 반경\n250 m (0.2 km²)',
                xy=(4.7, 3.0), xytext=(7.0, 1.4),
                fontsize=10, color=C_RED, fontweight='bold',
                ha='left',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

# Right side: ratio + text
ax3.text(0.70, 0.60, '약  245×', ha='center', fontsize=54, fontweight='bold',
         color=C_RED, transform=ax3.transAxes)
ax3.text(0.70, 0.46, '공간 미스매치', ha='center', fontsize=15,
         color=C_DARK, fontweight='bold', transform=ax3.transAxes)

ax3.text(0.70, 0.28,
         '동일 자치구 내에서도 격자별 위험은 5배 이상 차이\n자치구 평균에 숨겨진 사각지대 — 격자 단위 진단 필요',
         ha='center', fontsize=11, color=C_DARK, transform=ax3.transAxes,
         linespacing=1.6)

# Footer — sources
fig.text(0.025, 0.005,
         '출처: 보건복지부(치매환자 추정) · 경찰청 안전드림(실종·발견 통계 2023.09~2024.11) · 한국일보(2023.09 도시-시골 비교) · 서울특별시(2023 자치구 면적)',
         fontsize=8, color='#6B7682')

plt.savefig(OUT, dpi=160, bbox_inches='tight', facecolor='white', pad_inches=0.15)
plt.close()
print(f'OK -> {OUT}  ({OUT.stat().st_size//1024} KB)')
