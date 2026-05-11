"""
Option A: 3 chart-only transparent PNGs for slide 3.
User adds text/box backgrounds natively in Canva.

Charts:
  S3A_donut.png  — 추정 치매환자 도넛 + 741→49 콜아웃
  S3A_bar.png    — 도시 80% vs 시골 20% 가로 막대
  S3A_scale.png  — 자치구 49km² vs 250m 반경 다이어그램
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTDIR = Path('data/processed/report_viz/v2')
OUTDIR.mkdir(parents=True, exist_ok=True)

C_DARK = '#1F3A5F'
C_ACCENT = '#2E5C9E'
C_RED = '#D9534F'
C_GREY = '#9AA5B1'
C_LIGHTGREY = '#C9D4E5'

# ==================== Chart 1: Donut + 741→49 ====================
fig = plt.figure(figsize=(8, 5), dpi=200, facecolor='none')

# Left: donut
ax1 = fig.add_axes([0.02, 0.10, 0.42, 0.85])
ax1.set_aspect('equal')
sizes = [20, 172]  # 만명: 치매환자 vs 65+ 비치매
ax1.pie(sizes, colors=[C_RED, C_LIGHTGREY], startangle=90, counterclock=False,
        wedgeprops=dict(width=0.32, edgecolor='white', linewidth=2.5))
ax1.text(0, 0.08, '10.4%', ha='center', va='center',
         fontsize=30, fontweight='bold', color=C_RED)
ax1.text(0, -0.18, '20만 명', ha='center', va='center',
         fontsize=12, color=C_DARK)
ax1.text(0, -1.30, '서울 65+ 192만 중 추정 치매환자',
         ha='center', va='center', fontsize=11, color=C_DARK)
ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.5, 1.3)
ax1.axis('off')

# Right: 741 → 49 callout
ax2 = fig.add_axes([0.48, 0.0, 0.50, 1.0])
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.text(0.5, 0.92, '실종 신고', ha='center', fontsize=13, color=C_DARK)
ax2.text(0.5, 0.78, '741건', ha='center', fontsize=40, fontweight='bold', color=C_ACCENT)
ax2.text(0.5, 0.65, '(2023.09~2024.11, 15개월)', ha='center', fontsize=9, color=C_GREY)
ax2.annotate('', xy=(0.5, 0.40), xytext=(0.5, 0.55),
             arrowprops=dict(arrowstyle='-|>', color=C_RED, lw=2.5))
ax2.text(0.5, 0.30, '사망·미발견', ha='center', fontsize=13, color=C_DARK)
ax2.text(0.5, 0.14, '49건', ha='center', fontsize=40, fontweight='bold', color=C_RED)
ax2.text(0.5, 0.03, '(6.6% · 4시간 초과 시 생존율 급락)',
         ha='center', fontsize=9, color=C_GREY)

plt.savefig(OUTDIR/'S3A_donut.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f'OK -> S3A_donut.png ({(OUTDIR/"S3A_donut.png").stat().st_size//1024} KB)')


# ==================== Chart 2: 80% vs 20% bar ====================
fig = plt.figure(figsize=(8, 4), dpi=200, facecolor='none')
ax = fig.add_axes([0.15, 0.18, 0.80, 0.70])
ax.barh([1, 0], [80, 20], color=[C_RED, C_GREY], height=0.5,
        edgecolor='white', linewidth=2.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['시골 거주', '도시 거주'], fontsize=14, color=C_DARK)
ax.set_xlim(0, 100)
ax.set_xticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.text(82, 1, '80%', va='center', fontsize=26, fontweight='bold', color=C_RED)
ax.text(22, 0, '20%', va='center', fontsize=22, fontweight='bold', color=C_GREY)
ax.invert_yaxis()
fig.text(0.5, 0.04, '도시 거주 치매 노인의 80%가 외출 시 길을 잃음 · 시골 4배',
         ha='center', fontsize=10, color=C_DARK)

plt.savefig(OUTDIR/'S3A_bar.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f'OK -> S3A_bar.png ({(OUTDIR/"S3A_bar.png").stat().st_size//1024} KB)')


# ==================== Chart 3: Scale mismatch diagram ====================
fig = plt.figure(figsize=(10, 4.5), dpi=200, facecolor='none')
ax = fig.add_axes([0.02, 0.05, 0.96, 0.92])
ax.set_xlim(0, 20); ax.set_ylim(0, 9)
ax.axis('off')

# Left: scale diagram
big = Rectangle((1, 1.0), 7, 6.5, facecolor='none',
                edgecolor=C_ACCENT, linewidth=3, linestyle='-')
ax.add_patch(big)
ax.text(1.2, 6.9, '자치구', fontsize=14, fontweight='bold', color=C_ACCENT)
ax.text(1.2, 6.3, '평균 49 km²', fontsize=13, color=C_ACCENT)

small = Circle((4.5, 4.0), 0.20, facecolor=C_RED,
               edgecolor='white', linewidth=2)
ax.add_patch(small)
ax.annotate('실종자 발견 반경\n250 m (0.2 km²)',
            xy=(4.7, 4.0), xytext=(7.5, 2.0),
            fontsize=11, color=C_RED, fontweight='bold',
            ha='left',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=2))

# Right: ratio + text
ax.text(13, 6.2, '약', ha='center', fontsize=15, color=C_DARK)
ax.text(15.5, 6.0, '245×', ha='center', fontsize=58,
        fontweight='bold', color=C_RED)
ax.text(18, 6.2, '공간 미스매치', ha='center', fontsize=14,
        fontweight='bold', color=C_DARK)
ax.text(14.5, 3.3, '동일 자치구 내에서도 격자별 위험은',
        ha='center', fontsize=12, color=C_DARK)
ax.text(14.5, 2.4, '5배 이상 차이',
        ha='center', fontsize=14, fontweight='bold', color=C_RED)
ax.text(14.5, 1.5, '자치구 평균에 숨겨진 사각지대 — 격자 단위 진단 필요',
        ha='center', fontsize=10, color=C_GREY)

plt.savefig(OUTDIR/'S3A_scale.png', dpi=200, transparent=True,
            bbox_inches='tight', pad_inches=0.1)
plt.close()
print(f'OK -> S3A_scale.png ({(OUTDIR/"S3A_scale.png").stat().st_size//1024} KB)')
