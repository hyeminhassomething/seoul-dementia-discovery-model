"""
슬라이드 16 — 망우본동 HH 391 격자 줌인 케이스 스터디
GPT 슬라이드 13·15 스타일 일관성 — 깔끔한 카드 레이아웃 + 통합 산점도
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path('/Users/jeong-hyemin/Downloads/seoul-startup-competition')
OUT = ROOT / 'data/processed/report_viz/v2/SLIDE16_mangwo_zoom.png'
DATA = ROOT / 'data/processed/cluster_labeled_data_v2.csv'

# ─── Color palette (GPT slide 13·15 일관성) ──────────────
BG = '#0F1E3A'              # 슬라이드 배경
CARD_BG = '#1A2F52'         # 카드 배경
CARD_BORDER = '#2D4574'     # 카드 테두리
TEXT_MAIN = '#FFFFFF'
TEXT_SUB = '#C0CDE3'
TEXT_DIM = '#7B8AA8'
RED = '#EF4444'             # HH / 모순
ORANGE = '#F59E0B'          # HL / 정책
CYAN = '#22D3EE'            # 페르소나
EMERALD = '#10B981'         # 진단 OK
TITLE = '#FFFFFF'
ACCENT = '#60A5FA'          # 강조 파랑

# ─── 망우본동 데이터 로드 ──────────────────────────────────
df = pd.read_csv(DATA)
mangwo = df[(df['GU_NM']=='중랑구') & (df['ADM_NM']=='망우본동')].copy()
hh = mangwo[mangwo['LISA_Cluster']=='HH']
hl = mangwo[mangwo['LISA_Cluster']=='HL']
ns = mangwo[mangwo['LISA_Cluster']=='ns']
ll = mangwo[mangwo['LISA_Cluster']=='LL']
lh = mangwo[mangwo['LISA_Cluster']=='LH']

# ─── Canvas 1920 × 1080 (16:9) ──────────────────────────
fig = plt.figure(figsize=(19.20, 10.80), dpi=100, facecolor=BG)

# ============ TITLE STRIP ============
ax_title = fig.add_axes([0, 0.88, 1, 0.12])
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis('off')
ax_title.set_facecolor(BG)
# 좌측 컬러바
ax_title.add_patch(Rectangle((0.025, 0.30), 0.006, 0.55,
                              facecolor=RED, transform=ax_title.transAxes))
ax_title.text(0.039, 0.62, '망우본동 HH 391 격자 줌인',
              fontsize=38, fontweight='bold', color=TITLE, va='center')
ax_title.text(0.039, 0.22,
              'Case Study — 4중 모순의 격자 단위 증거 · HDBSCAN 페르소나 분류',
              fontsize=14, color=TEXT_SUB, va='center')

# ─── 카드 그리기 헬퍼 ────────────────────────────────────
def card(left, bottom, width, height, accent, num, title, rows):
    """
    rows = [(label, value_str, sublabel_opt), ...] — 핵심 통계 카드
    또는 [bullet_str, ...] — 일반 텍스트 카드
    """
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    # 카드 배경
    bg = FancyBboxPatch((0.5, 1), 99, 98,
                        boxstyle="round,pad=0,rounding_size=2",
                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1.5)
    ax.add_patch(bg)

    # 헤더 번호 원
    circ = plt.Circle((6.5, 87), 3.2, color=accent, transform=ax.transData)
    ax.add_patch(circ)
    ax.text(6.5, 87, num, ha='center', va='center', fontsize=13,
            fontweight='bold', color='#0F1E3A')
    # 헤더 제목
    ax.text(13, 87, title, fontsize=16, fontweight='bold',
            color=accent, va='center')

    # 구분선
    ax.plot([4, 96], [76, 76], color=CARD_BORDER, linewidth=1)

    # 본문
    if isinstance(rows[0], tuple):
        # 통계 카드 — 좌 라벨 / 우 값
        y = 66
        for row in rows:
            label, value = row[0], row[1]
            ax.text(6, y, label, fontsize=11.5, color=TEXT_SUB, va='top')
            ax.text(94, y, value, fontsize=12.5, fontweight='bold',
                    color=TEXT_MAIN, va='top', ha='right')
            if len(row) > 2 and row[2]:
                ax.text(94, y-7, row[2], fontsize=9, color=TEXT_DIM,
                        va='top', ha='right')
            y -= 14
    else:
        # 텍스트 카드 — 불릿
        y = 68
        for line in rows:
            # 첫 1글자가 ● / ▪ 같은 마커면 강조 색
            if line.startswith('▶'):
                ax.text(6, y, '▶', fontsize=10, color=accent, va='top')
                ax.text(11, y, line[1:].strip(), fontsize=11, color=TEXT_MAIN,
                        va='top', wrap=True)
            else:
                ax.text(6, y, line, fontsize=11, color=TEXT_MAIN, va='top')
            y -= 11.5

# ============ 4개 카드 (좌측 2×2 그리드) ============
# 카드 1 — 4중 모순 (정량 진단)
card(0.025, 0.46, 0.235, 0.36, RED, '1', '4중 모순 진단', [
    ('① 인구 모순  65+/격자', '4.11명',  '서울 평균 1.32명 · 3.1×'),
    ('② 환경 모순  막다른길',  '1.87',    '서울 평균 1.01 · 상위 5%'),
    ('③ 인프라  INFRA z',     '-0.84',   '인프라 30% 수준'),
    ('④ 감시  CCTV 200m 사각',  '31%',   'HH 격자의 사각지대 비율'),
])

# 카드 2 — 표시 레이어
card(0.265, 0.46, 0.235, 0.36, ACCENT, '2', '표시 레이어 (오버레이)', [
    '● HH 격자 (빨강) — 391개',
    '● 65+ 거주지 — 회색 음영',
    '● CCTV 위치 — 동 외곽 편중',
    '● 막다른길 — 동 내부 골목 분포',
    '● HL 사각지대 — 주황 3개',
])

# 카드 3 — 페르소나 분류
card(0.025, 0.085, 0.235, 0.36, CYAN, '3', '페르소나 분류', [
    ('우세 페르소나', '인구밀집·인프라소외형',  ''),
    ('망우 HH 391 중', '191개 (49%)', ''),
    ('산발형 noise', '187개 (48%)', ''),
    ('도심상권 / 골목미로', '9 / 4개', '소수 격자'),
])

# 카드 4 — 정책 트리거
card(0.265, 0.085, 0.235, 0.36, ORANGE, '4', '정책 트리거 (단일 패키지)', [
    '● CCTV 사각 → AI CCTV 보강',
    '● 막다른길 → 음성표지 + GPS',
    '● 65+ 밀집 → 치매 파수꾼',
    '● 통합 패키지 1.5억/년',
    '● 4동 동시 처방 가능',
])

# ============ 우측 산점도 영역 — 직접 plot (배경 통일) ============
ax_map = fig.add_axes([0.535, 0.10, 0.43, 0.72])
ax_map.set_facecolor(CARD_BG)

# 산점 — 배경 격자
ax_map.scatter(ns['center_lon'], ns['center_lat'], s=10, c='#4B5C7F',
               alpha=0.45, label=f'ns (n={len(ns)})')
ax_map.scatter(ll['center_lon'], ll['center_lat'], s=10, c='#5C7099',
               alpha=0.55, label=f'LL (n={len(ll)})')
ax_map.scatter(lh['center_lon'], lh['center_lat'], s=10, c='#7C99B5',
               alpha=0.6, label=f'LH (n={len(lh)})')
ax_map.scatter(hl['center_lon'], hl['center_lat'], s=24, c=ORANGE,
               edgecolor='white', linewidth=0.4, label=f'HL (n={len(hl)})',
               zorder=3)
ax_map.scatter(hh['center_lon'], hh['center_lat'], s=24, c=RED,
               edgecolor='white', linewidth=0.4, label=f'HH (n={len(hh)})',
               zorder=4)

# 축
ax_map.set_xlabel('경도', fontsize=10, color=TEXT_SUB)
ax_map.set_ylabel('위도', fontsize=10, color=TEXT_SUB)
ax_map.tick_params(colors=TEXT_DIM, labelsize=9)
for spine in ax_map.spines.values():
    spine.set_edgecolor(CARD_BORDER); spine.set_linewidth(1.5)
ax_map.grid(True, alpha=0.15, color=TEXT_DIM, linestyle='--', linewidth=0.5)
ax_map.set_aspect('equal')

# 산점도 제목
ax_map.set_title('중랑구 망우본동 — 1,177 격자 LISA 분포',
                 fontsize=13, color=TEXT_MAIN, fontweight='bold', pad=10)

# 범례
leg = ax_map.legend(loc='upper right', fontsize=9, framealpha=0.92,
                    facecolor=BG, edgecolor=CARD_BORDER, labelcolor=TEXT_MAIN)

# 우측 상단 데이터 박스 (산점도 위에 오버레이)
ax_box = fig.add_axes([0.78, 0.69, 0.18, 0.12])
ax_box.set_xlim(0, 100); ax_box.set_ylim(0, 100); ax_box.axis('off')
ax_box.set_facecolor('none')
box = FancyBboxPatch((1, 2), 98, 96,
                     boxstyle="round,pad=0,rounding_size=4",
                     facecolor=BG, edgecolor=RED, linewidth=1.5)
ax_box.add_patch(box)
stats_lines = [
    ('전체 격자', '1,177'),
    ('HH 격자', '391'),
    ('HH 비율', '33.2%'),
    ('65+ 인구', '11,919명'),
]
y = 80
for label, val in stats_lines:
    ax_box.text(8, y, label, fontsize=9.5, color=TEXT_SUB, va='center')
    ax_box.text(92, y, val, fontsize=11, fontweight='bold',
                color=TEXT_MAIN, va='center', ha='right')
    y -= 20

# ============ 푸터 ============
ax_foot = fig.add_axes([0, 0, 1, 0.07])
ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1); ax_foot.axis('off')
ax_foot.set_facecolor(BG)
ax_foot.text(0.025, 0.58,
             '※ 자치구·동 단위 평균 차트가 아닌 격자 단위 진단 — 정책 집행 단위와 정책 대상 단위 일치 (50m 격자)',
             fontsize=10, color=TEXT_SUB, va='center')
ax_foot.text(0.025, 0.22,
             '출처: cluster_labeled_data_v2.csv (250,450 격자 × LISA × HDBSCAN) · T15_top_dong.csv · cluster_naming_v2.csv (96 군집 자동 명명)',
             fontsize=8.5, color=TEXT_DIM, va='center')

plt.savefig(OUT, dpi=100, facecolor=BG, bbox_inches=None)
plt.close()
print(f'OK -> {OUT}  ({OUT.stat().st_size//1024} KB)')
print(f'망우 HH {len(hh)} / HL {len(hl)} / ns {len(ns)} / LL {len(ll)} / LH {len(lh)}')
