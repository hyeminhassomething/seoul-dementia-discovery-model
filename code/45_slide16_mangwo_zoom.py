"""
슬라이드 16 — 망우본동 HH 391 격자 줌인 케이스 스터디
1920×1080 풀 슬라이드 PNG 생성
스타일: 슬라이드 13/15 카드형 레이아웃 일관성
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path('/Users/jeong-hyemin/Downloads/seoul-startup-competition')
OUT = ROOT / 'data/processed/report_viz/v2/SLIDE16_mangwo_zoom.png'
A2_IMG = ROOT / 'data/processed/report_viz/cluster/A2_mangwo_HH_overlay.png'

# ─── Color palette (deck consistent) ─────────────────────
BG = '#0A1A2F'           # 배경 짙은 네이비
BG_BOX = '#142A4A'       # 박스 배경
ACCENT = '#3B82F6'       # 강조 파랑
TEXT_MAIN = '#FFFFFF'    # 흰색 본문
TEXT_SUB = '#B8C5D9'     # 회색 본문
RED = '#EF4444'          # HH 빨강
ORANGE = '#F59E0B'       # 막다른길/HL 주황
CYAN = '#06B6D4'         # 인구 청록
GREEN = '#10B981'        # 인프라
PURPLE = '#A855F7'       # CCTV 보라

# ─── Canvas 1920 × 1080 ───────────────────────────────────
fig = plt.figure(figsize=(19.20, 10.80), dpi=100, facecolor=BG)

# ============ 제목 영역 (상단 ~120px) ============
ax_title = fig.add_axes([0, 0.89, 1, 0.11])
ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1); ax_title.axis('off')
ax_title.set_facecolor(BG)
ax_title.text(0.03, 0.65, '망우본동 HH 391 격자 줌인',
              fontsize=34, fontweight='bold', color=TEXT_MAIN, va='center')
ax_title.text(0.03, 0.22, '65+ 거주지 · CCTV 위치 · 막다른길 동시 오버레이 — 4중 모순의 격자 단위 증거',
              fontsize=14, color=TEXT_SUB, va='center')

# ============ 좌측 4 박스 영역 ============
# 박스 1: 표시 레이어  (y top → 0.80)
def draw_box(left, bottom, width, height, color, title, lines):
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    # 박스 배경
    bg = FancyBboxPatch((0.005, 0.02), 0.99, 0.96,
                        boxstyle="round,pad=0.0,rounding_size=0.015",
                        facecolor=BG_BOX, edgecolor=color, linewidth=2,
                        transform=ax.transAxes)
    ax.add_patch(bg)
    # 좌측 컬러바
    ax.add_patch(plt.Rectangle((0.01, 0.78), 0.008, 0.18,
                               facecolor=color, transform=ax.transAxes))
    # 박스 제목
    ax.text(0.04, 0.87, title, fontsize=15, fontweight='bold',
            color=color, va='center', transform=ax.transAxes)
    # 본문
    y = 0.72
    for line in lines:
        if line.startswith('•'):
            ax.text(0.05, y, line, fontsize=10.5, color=TEXT_MAIN,
                    va='top', transform=ax.transAxes)
        else:
            ax.text(0.05, y, line, fontsize=10.5, color=TEXT_SUB,
                    va='top', transform=ax.transAxes)
        y -= 0.14
    return ax

# 박스 1 — 표시 레이어
draw_box(0.025, 0.55, 0.235, 0.32, ACCENT, '표시 레이어 (4종 동시 오버레이)', [
    '• HH 격자 (빨강): 391개 · 망우본동 1위',
    '• 65+ 거주지: 회색 음영, 인구 비례 농도',
    '• CCTV (파랑 ▲): 동 외곽 편중 — HH와 어긋남',
    '• 막다른길 (주황 ×): 동 내부 깊숙이 분포',
])

# 박스 2 — 정량 진단
draw_box(0.265, 0.55, 0.235, 0.32, RED, '정량 진단 (4중 모순 격자 단위)', [
    '• HH 중 31% CCTV 200m 사각지대',
    '• 막다른길 1.87 / 도로복잡도 1.77 (서울 ↑↑)',
    '• 65+ 11,919명 · 격자당 4.11명 (서울 1.32명)',
    '• INFRA z=-0.84 · STORE z=-0.62 (인프라 30%)',
])

# 박스 3 — 페르소나 분류 (신규)
draw_box(0.025, 0.20, 0.235, 0.32, CYAN, '페르소나 분류 (HDBSCAN × LISA)', [
    '• 망우본동 HH 391 중 인구밀집·인프라소외형 191개',
    '• 우세 페르소나: 49% — 단일 패키지 적용 가능',
    '• 산발형(noise) 187개 (48%), 도심상권 9, 골목 4',
    '• 시그니처: 65+ ↑↑ × INFRA ↓↓',
])

# 박스 4 — 정책 트리거
draw_box(0.265, 0.20, 0.235, 0.32, ORANGE, '정책 트리거 (격자 단위 매핑)', [
    '• CCTV 사각 격자 (HH ∩ CCTV200m 밖) → AI CCTV',
    '• 막다른길 격자 (HH ∩ deadend≥2) → 음성 표지+GPS',
    '• 65+ 밀집 격자 (HH ∩ 65+ ≥4명) → 치매 파수꾼',
    '• 단일 패키지: 1.5억원/년 (4동 동시 처방)',
])

# ============ 우측 산점도 영역 (A2 이미지 임베드) ============
ax_img = fig.add_axes([0.53, 0.10, 0.45, 0.78])
ax_img.axis('off')
if A2_IMG.exists():
    img = mpimg.imread(A2_IMG)
    ax_img.imshow(img)
else:
    ax_img.text(0.5, 0.5, 'A2 이미지 누락', ha='center', va='center',
                fontsize=20, color=TEXT_SUB, transform=ax_img.transAxes)

# ============ 푸터 ============
ax_foot = fig.add_axes([0, 0, 1, 0.07])
ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1); ax_foot.axis('off')
ax_foot.set_facecolor(BG)
ax_foot.text(0.03, 0.55,
             '※ 자치구·동 단위 평균 차트가 아닌 격자 단위 지도 — 정책 집행 단위와 정책 대상 단위가 일치하는 50m 격자 진단의 핵심 케이스 스터디',
             fontsize=9, color=TEXT_SUB, va='center')
ax_foot.text(0.03, 0.20,
             '출처: cluster_labeled_data_v2.csv (250,450 격자 × LISA × HDBSCAN) · T15_top_dong.csv (HH TOP 15) · cluster_naming_v2.csv (96 군집 자동 명명)',
             fontsize=8, color='#6B7682', va='center')

plt.savefig(OUT, dpi=100, facecolor=BG, bbox_inches=None)
plt.close()
print(f'OK -> {OUT}  ({OUT.stat().st_size//1024} KB)')
