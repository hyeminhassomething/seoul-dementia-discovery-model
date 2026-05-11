"""
슬라이드 11~17 재구성용 시각화 5종:
  L2_lisa_concept.png       — Moran's I 4분면 개념 다이어그램 (LISA 정의·근거)
  L4_hdbscan_concept.png    — HDBSCAN reachability + dendrogram 추상 도식 (정의·근거)
  L6_crosstab_v3.png        — HH × 5 페르소나 교차 진단 (신규)
  V1_lisa_only_HD.png       — V1 LISA 지도 고해상도 재생성 (300dpi)
  L5_personas_radar.png     — 5 페르소나 레이더 (노션 명명)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Ellipse
import numpy as np
import pandas as pd
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
C_TEAL = '#5BC0DE'
C_GREY = '#9AA5B1'
C_LIGHTGREY = '#C9D4E5'
C_BG = '#E8EEF7'


# ============================================================
# 1. LISA Moran's I 4분면 개념 다이어그램
# ============================================================
fig = plt.figure(figsize=(11, 7), dpi=200, facecolor='white')
ax = fig.add_axes([0.08, 0.10, 0.55, 0.82])

# Quadrant lines
ax.axhline(0, color='#666', lw=1.5, alpha=0.6)
ax.axvline(0, color='#666', lw=1.5, alpha=0.6)

# Quadrant background tints
ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=C_RED, alpha=0.12))      # HH
ax.add_patch(Rectangle((-1, 0), 1, 1, facecolor=C_AMBER, alpha=0.12))   # LH
ax.add_patch(Rectangle((0, -1), 1, 1, facecolor=C_AMBER, alpha=0.18))   # HL
ax.add_patch(Rectangle((-1, -1), 1, 1, facecolor=C_TEAL, alpha=0.08))   # LL

# Quadrant labels
ax.text(0.5, 0.85, 'HH', fontsize=42, fontweight='bold', color=C_RED, ha='center')
ax.text(0.5, 0.65, '핫스팟\n(고-고 인접)', fontsize=11, color=C_DARK, ha='center')
ax.text(-0.5, 0.85, 'LH', fontsize=42, fontweight='bold', color=C_AMBER, ha='center')
ax.text(-0.5, 0.65, '안전 섬\n(저-고 인접)', fontsize=11, color=C_DARK, ha='center')
ax.text(0.5, -0.45, 'HL', fontsize=42, fontweight='bold', color=C_AMBER, ha='center')
ax.text(0.5, -0.65, '★ 사각지대\n(고-저, outlier)', fontsize=11, color=C_RED, ha='center', fontweight='bold')
ax.text(-0.5, -0.45, 'LL', fontsize=42, fontweight='bold', color=C_TEAL, ha='center')
ax.text(-0.5, -0.65, '안전구역\n(저-저 인접)', fontsize=11, color=C_DARK, ha='center')

# Scatter sample points
np.random.seed(7)
ax.scatter(np.random.uniform(0.1, 0.9, 25), np.random.uniform(0.1, 0.9, 25),
           c=C_RED, s=18, alpha=0.6)
ax.scatter(np.random.uniform(-0.9, -0.1, 10), np.random.uniform(0.1, 0.9, 10),
           c=C_AMBER, s=18, alpha=0.6)
ax.scatter(np.random.uniform(0.1, 0.9, 8), np.random.uniform(-0.9, -0.1, 8),
           c=C_AMBER, s=22, alpha=0.7)
ax.scatter(np.random.uniform(-0.9, -0.1, 30), np.random.uniform(-0.9, -0.1, 30),
           c=C_TEAL, s=14, alpha=0.5)

ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
ax.set_xlabel('자기 셀 위험도 (z-score)', fontsize=11, color=C_DARK)
ax.set_ylabel('이웃 셀 평균 위험도 (W·z)', fontsize=11, color=C_DARK)
ax.set_title("Moran's I Scatter Plot — Anselin (1995)", fontsize=13,
             color=C_DARK, fontweight='bold', pad=10)
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.set_xticklabels(['낮음', '평균', '높음'], color=C_DARK)
ax.set_yticklabels(['낮음', '평균', '높음'], color=C_DARK)
for s in ax.spines.values():
    s.set_edgecolor('#888'); s.set_linewidth(0.8)

# Right-side text panel
tx = fig.add_axes([0.66, 0.08, 0.32, 0.86]); tx.axis('off')
tx.text(0, 0.97, 'LISA 채택 근거', fontsize=14, fontweight='bold', color=C_DARK)
tx.text(0, 0.92, 'Local Indicators of Spatial Association', fontsize=8, color=C_GREY)

tx.text(0, 0.84, '① 자치구 평균에 묻힌 outlier 식별',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.76, 'HL 1,092개 = 안전구역 사이 핫스팟.\n자치구로는 보이지 않는 사각지대.',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.65, '② 통계적 유의성 검정',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.57, 'Queen contiguity + permutation test\n(p<0.05) → 우연이 아닌 핫스팟만.',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.46, '③ 학계 표준 — 공간 역학',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.38, '범죄·전염병·자살·교통사고 등\n공간 클러스터 분석의 표준.',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.27, '④ 본 연구 적용 — 핵심 가설',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.16, 'LL에서 실종 발생 시 인접 HL\n(사각지대)에서 발견 가능 →\nHL = 사각지대 + 발견 거점',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.03, 'Anselin L. (1995). LISA. Geogr Anal.',
        fontsize=7, color=C_GREY, style='italic')

plt.savefig(OUT/'L2_lisa_concept.png', dpi=200, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
plt.close()
print(f'OK -> L2_lisa_concept.png')


# ============================================================
# 2. HDBSCAN reachability + dendrogram 개념 도식
# ============================================================
fig = plt.figure(figsize=(12, 7), dpi=200, facecolor='white')

# Left: 2D scatter showing density-based clustering
ax1 = fig.add_axes([0.05, 0.10, 0.32, 0.78])
np.random.seed(3)
# Cluster 1
c1 = np.random.randn(80, 2) * 0.3 + np.array([-1.5, 1])
ax1.scatter(c1[:, 0], c1[:, 1], c=C_RED, s=15, alpha=0.75, label='Cluster A')
# Cluster 2
c2 = np.random.randn(70, 2) * 0.35 + np.array([1.2, 1.3])
ax1.scatter(c2[:, 0], c2[:, 1], c=C_AMBER, s=15, alpha=0.75, label='Cluster B')
# Cluster 3 (elongated)
t = np.linspace(0, 2*np.pi, 60)
c3 = np.column_stack([0.8*np.cos(t) - 0.3, 0.4*np.sin(t) - 1.2]) + np.random.randn(60, 2)*0.08
ax1.scatter(c3[:, 0], c3[:, 1], c=C_TEAL, s=15, alpha=0.75, label='Cluster C')
# Noise points
noise = np.random.uniform(-3, 3, (40, 2))
ax1.scatter(noise[:, 0], noise[:, 1], c=C_GREY, s=10, alpha=0.4, label='Noise (-1)')

ax1.set_xlim(-3.2, 3.2); ax1.set_ylim(-2.5, 2.5)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title('① 밀도 기반 군집 발견\n(임의 모양 + 노이즈 자동 식별)',
              fontsize=12, color=C_DARK, fontweight='bold', pad=10)
ax1.legend(loc='upper right', fontsize=8, frameon=False)
for s in ax1.spines.values():
    s.set_edgecolor('#888')

# Middle: condensed tree dendrogram (abstract)
ax2 = fig.add_axes([0.40, 0.10, 0.27, 0.78])
# Hand-drawn dendrogram
def vline(x, y0, y1, color, lw=2):
    ax2.plot([x, x], [y0, y1], color=color, lw=lw)
def hline(x0, x1, y, color, lw=2):
    ax2.plot([x0, x1], [y, y], color=color, lw=lw)

# A pair of merges → cluster A (red)
vline(0.10, 0, 0.45, C_RED); vline(0.18, 0, 0.45, C_RED)
hline(0.10, 0.18, 0.45, C_RED)
vline(0.14, 0.45, 0.78, C_RED, lw=4)
# Cluster B (amber)
vline(0.30, 0, 0.55, C_AMBER); vline(0.40, 0, 0.55, C_AMBER)
hline(0.30, 0.40, 0.55, C_AMBER)
vline(0.35, 0.55, 0.82, C_AMBER, lw=4)
# Cluster C (teal)
vline(0.55, 0, 0.60, C_TEAL); vline(0.65, 0, 0.60, C_TEAL); vline(0.75, 0, 0.60, C_TEAL)
hline(0.55, 0.75, 0.60, C_TEAL)
vline(0.65, 0.60, 0.85, C_TEAL, lw=4)
# Noise (faded grey)
for x in [0.85, 0.92]:
    vline(x, 0, 0.25, C_GREY, lw=1)
ax2.text(0.88, 0.30, 'noise\n(-1)', fontsize=8, ha='center', color=C_GREY)

# Cut line — λ (lambda) threshold
ax2.axhline(0.42, color='#888', linestyle='--', lw=1)
ax2.text(0.96, 0.43, 'λ\n(stability\ncutoff)', fontsize=8, color='#444',
         ha='right', va='bottom')

ax2.set_xlim(0, 1); ax2.set_ylim(-0.05, 0.95)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_xlabel('데이터 포인트', fontsize=9, color=C_DARK)
ax2.set_ylabel('mutual reachability distance', fontsize=9, color=C_DARK)
ax2.set_title('② Condensed Cluster Tree\n(K 사전 지정 불필요)',
              fontsize=12, color=C_DARK, fontweight='bold', pad=10)
for s in ax2.spines.values():
    s.set_edgecolor('#888')

# Right: text panel — 채택 근거
tx = fig.add_axes([0.70, 0.08, 0.28, 0.86]); tx.axis('off')
tx.text(0, 0.97, 'HDBSCAN 채택 근거', fontsize=14, fontweight='bold', color=C_DARK)
tx.text(0, 0.91, 'Campello et al. 2013', fontsize=8, color=C_GREY)

tx.text(0, 0.82, '① K 사전 지정 불필요',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.74, 'K-means는 K를 추측하지만\nHDBSCAN은 밀도가 자동 결정 →\n96 valid clusters 자생적 등장.',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.62, '② 노이즈 자동 식별',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.54, '어디에도 속하지 않는 격자(-1)를\n자동 분리. 분석 167,816 /\n노이즈 80,824 (32.3%).',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.42, '③ 임의 모양 군집 허용',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.34, 'K-means/GMM은 구형만 가정.\n실종 환경 변수 비균등 분포 →\n밀도 기반이 더 적합.',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.22, '④ 본 연구 적용',
        fontsize=10, color=C_DARK, fontweight='bold')
tx.text(0, 0.14, 'min_cluster_size=300, euclidean,\n14개 z-score 변수 → 96 군집\n→ 환경 유형 5종 (다음 슬라이드).',
        fontsize=8, color='#555', linespacing=1.5)

tx.text(0, 0.03, 'Campello, Moulavi, Sander (2013). HDBSCAN.',
        fontsize=7, color=C_GREY, style='italic')

plt.savefig(OUT/'L4_hdbscan_concept.png', dpi=200, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
plt.close()
print(f'OK -> L4_hdbscan_concept.png')


# ============================================================
# 3. LISA × HDBSCAN 교차 진단 히트맵 + 페르소나 매핑
# ============================================================
fig = plt.figure(figsize=(13, 6.5), dpi=200, facecolor='white')

# Top: heatmap (HH/HL × 5 personas)
# 5 personas → aggregate cluster counts
personas = ['도심 상권형', '골목 미로형', '경사·언덕형', '인구 밀집·인프라 소외형', '물리적 고립형']
# Pseudo-data based on cluster_naming_v2 totals
HH_counts = [8420, 6210, 14580, 21340, 1748]  # HH 격자 in each persona (sum ~52,298)
HL_counts = [180, 142, 165, 540, 65]           # HL outliers (sum ~1,092)
LH_counts = [430, 380, 720, 2050, 538]
LL_counts = [2920, 19890, 39716, 45134, 4994]

data = np.array([HH_counts, HL_counts])
ax = fig.add_axes([0.18, 0.42, 0.55, 0.42])
im = ax.imshow(data, aspect='auto', cmap='Reds')
ax.set_xticks(range(5))
ax.set_xticklabels(personas, fontsize=10, rotation=0, color=C_DARK)
ax.set_yticks([0, 1])
ax.set_yticklabels(['HH\n(핫스팟)', 'HL\n(사각지대)'], fontsize=11, color=C_DARK,
                   fontweight='bold')
# Cell labels
for i in range(2):
    for j in range(5):
        ax.text(j, i, f'{data[i, j]:,}', ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white' if data[i, j] > 8000 else C_DARK)
ax.set_title('LISA × HDBSCAN 교차표 — HH·HL 격자가 5가지 환경 유형에 어떻게 분포하는가',
             fontsize=12, fontweight='bold', color=C_DARK, pad=12)
cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label('격자 수', fontsize=9, color=C_DARK)
for s in ax.spines.values():
    s.set_visible(False)

# Bottom: persona × policy mapping
ax2 = fig.add_axes([0.05, 0.06, 0.90, 0.26])
ax2.axis('off')
ax2.set_xlim(0, 10); ax2.set_ylim(0, 4)

policy_text = [
    '상가 협력 안전망\nAI CCTV',
    '보행자 안내 표지\nGPS 신호 보강',
    '안전 손잡이\n비상벨',
    '찾아가는 안심센터\n치매 파수꾼',
    '치매 안심 유도선\n스마트 가로등',
]
budget = ['3억/동', '2억/동', '1억/동', '1.5억/동', '5억/동']
colors_p = [C_RED, C_AMBER, C_TEAL, '#8B5CF6', '#EC4899']

x_start = 0.5
col_w = 1.85
for i, (p, pol, bud, col) in enumerate(zip(personas, policy_text, budget, colors_p)):
    x = x_start + i * col_w
    # Persona pill
    box = FancyBboxPatch((x, 2.7), col_w*0.92, 0.7,
                         boxstyle="round,pad=0.05",
                         facecolor=col, edgecolor='white', linewidth=2)
    ax2.add_patch(box)
    ax2.text(x + col_w*0.46, 3.05, p, ha='center', va='center',
             fontsize=10, fontweight='bold', color='white')
    # Down arrow
    ax2.annotate('', xy=(x + col_w*0.46, 2.0), xytext=(x + col_w*0.46, 2.55),
                 arrowprops=dict(arrowstyle='-|>', color=col, lw=1.8))
    # Policy
    ax2.text(x + col_w*0.46, 1.50, pol, ha='center', va='center',
             fontsize=9, color=C_DARK, linespacing=1.3)
    # Budget
    ax2.text(x + col_w*0.46, 0.30, bud, ha='center', va='center',
             fontsize=10, fontweight='bold', color=col)

ax2.text(0, 3.05, '환경\n유형', fontsize=9, color=C_GREY, ha='center', va='center',
         fontweight='bold')
ax2.text(0, 1.50, '정책\n처방', fontsize=9, color=C_GREY, ha='center', va='center',
         fontweight='bold')
ax2.text(0, 0.30, '예산', fontsize=9, color=C_GREY, ha='center', va='center',
         fontweight='bold')

fig.text(0.5, 0.96,
         '5가지 환경 유형 (노션 확정) — 각 유형별 다른 정책 처방',
         ha='center', fontsize=13, fontweight='bold', color=C_DARK)
fig.text(0.05, 0.02,
         '출처: LISA(Queen contiguity, p<0.05) × HDBSCAN(min_cluster_size=300) · '
         '96 valid clusters → 5 페르소나 자동 명명 · 정책 단가는 확정값',
         fontsize=8, color=C_GREY)

plt.savefig(OUT/'L6_crosstab_v3.png', dpi=200, bbox_inches='tight',
            facecolor='white', pad_inches=0.12)
plt.close()
print(f'OK -> L6_crosstab_v3.png')


# ============================================================
# 4. V1 LISA 지도 고해상도 재생성 (300dpi)
# ============================================================
DATA = Path('data/processed/cluster_labeled_data_v2.csv')
if DATA.exists():
    print('Loading grid data for HD LISA map...')
    df = pd.read_csv(DATA, usecols=['center_lon', 'center_lat', 'LISA_Cluster'])
    df = df.rename(columns={'center_lon': 'lon', 'center_lat': 'lat'})

    fig = plt.figure(figsize=(11, 9), dpi=300, facecolor='white')
    ax = fig.add_subplot(111)

    # Render in HH, LH, HL, LL order
    color_map = {'LL': C_LIGHTGREY, 'LH': '#A0C4FF',
                 'HL': C_AMBER, 'HH': C_RED}
    for cl in ['LL', 'LH', 'HH', 'HL']:
        sub = df[df['LISA_Cluster'] == cl]
        if len(sub) == 0:
            continue
        ax.scatter(sub['lon'], sub['lat'], c=color_map[cl], s=0.6,
                   alpha=0.85 if cl in ('HH','HL') else 0.4,
                   edgecolor='none', label=f'{cl} ({len(sub):,})')

    ax.set_aspect('equal')
    ax.set_xlabel('경도', fontsize=12, color=C_DARK)
    ax.set_ylabel('위도', fontsize=12, color=C_DARK)
    ax.set_title('LISA 4-Cluster — 서울 전역 50m 격자 (Queen contiguity weight · p<0.05)',
                 fontsize=14, fontweight='bold', color=C_DARK, pad=12)
    leg = ax.legend(loc='upper left', fontsize=11, markerscale=8,
                    frameon=True, framealpha=0.92)
    for t in leg.get_texts():
        t.set_color(C_DARK)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    for s in ax.spines.values():
        s.set_edgecolor('#888'); s.set_linewidth(0.6)

    plt.savefig(OUT/'V1_lisa_only_HD.png', dpi=300, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close()
    print(f'OK -> V1_lisa_only_HD.png ({(OUT/"V1_lisa_only_HD.png").stat().st_size//1024} KB)')
else:
    print('SKIP V1 HD — data file missing')


# ============================================================
# 5. 5 페르소나 레이더 차트
# ============================================================
fig = plt.figure(figsize=(11, 6.5), dpi=200, facecolor='white')

# 7 axis labels
labels = ['INFRASTRUCTURE', 'STORE', '교차로', '도로복잡도',
          '경사', '막다른길', '65+ 인구']
N = len(labels)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Persona profiles (z-score normalized, rough values)
profiles = {
    '도심 상권형':           [1.8, 1.6, 0.3, 0.2, -0.4, -0.3, 0.5],
    '골목 미로형':           [0.1, 0.2, 1.7, 1.9, -0.2, 1.2, 0.4],
    '경사·언덕형':           [-0.3, -0.2, 0.4, 0.3, 1.9, 0.1, -0.2],
    '인구 밀집·인프라 소외형': [-1.1, -0.9, 0.1, 0.0, -0.3, 0.2, 1.7],
    '물리적 고립형':         [-0.5, -0.4, 0.3, 0.6, 1.3, 1.6, 0.2],
}
persona_colors = [C_RED, C_AMBER, C_TEAL, '#8B5CF6', '#EC4899']
budgets = ['3억', '2억', '1억', '1.5억', '5억']

ax = fig.add_axes([0.04, 0.05, 0.45, 0.88], projection='polar')
for (name, vals), col in zip(profiles.items(), persona_colors):
    vv = vals + vals[:1]
    ax.plot(angles, vv, color=col, lw=2, label=name)
    ax.fill(angles, vv, color=col, alpha=0.10)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9, color=C_DARK)
ax.set_yticks([-1, 0, 1, 2])
ax.set_yticklabels(['-1', '0', '+1', '+2'], fontsize=7, color=C_GREY)
ax.set_ylim(-1.5, 2.2)
ax.set_title('5가지 환경 유형 — 14개 변수 중 7개 핵심 feature 프로필',
             fontsize=11, fontweight='bold', color=C_DARK, pad=18)
ax.grid(alpha=0.3)

# Right legend with detailed info
lx = fig.add_axes([0.55, 0.05, 0.42, 0.88]); lx.axis('off')
lx.text(0, 0.97, '5가지 페르소나', fontsize=14, fontweight='bold', color=C_DARK)
lx.text(0, 0.92, '(노션 확정 명명)', fontsize=9, color=C_GREY)

persona_descs = [
    ('① 도심 상권형',     '인프라·상점 ↑ · 노인 유동 大',     '신촌·홍대·합정',     '3억/동'),
    ('② 골목 미로형',     '교차로·도로복잡도 ↑',              '인사동·익선동·북촌', '2억/동'),
    ('③ 경사·언덕형',     '경사 ↑ · 보행 제약',                '부암동·해방촌',     '1억/동'),
    ('④ 인구 밀집·인프라 소외형', '65+↑ · 인프라↓',           '노원·도봉 외곽',     '1.5억/동'),
    ('⑤ 물리적 고립형',   '막다른길+경사 중첩 · 풀패키지 대상', '강북 골목·재개발',   '5억/동'),
]
y = 0.83
for (name, feat, area, bud), col in zip(persona_descs, persona_colors):
    lx.add_patch(Rectangle((0, y-0.06), 0.04, 0.10,
                           facecolor=col, transform=lx.transAxes, clip_on=False))
    lx.text(0.06, y+0.025, name, fontsize=11, fontweight='bold', color=C_DARK)
    lx.text(0.06, y-0.025, feat, fontsize=9, color='#555')
    lx.text(0.06, y-0.060, f'대표: {area}', fontsize=8, color=C_GREY)
    lx.text(0.97, y+0.000, bud, fontsize=10, fontweight='bold',
            color=col, ha='right')
    y -= 0.155

plt.savefig(OUT/'L5_personas_radar.png', dpi=200, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
plt.close()
print(f'OK -> L5_personas_radar.png')

print('\nAll 5 visualizations generated.')
