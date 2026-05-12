"""
A2_mangwo_HH_overlay.png 재생성
cluster_labeled_data_v2.csv (cuML Queen) 기준 망우본동 HH 391개 정합값.
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

DATA = Path('/Users/jeong-hyemin/Downloads/seoul-startup-competition/data/processed/cluster_labeled_data_v2.csv')
OUT = Path('/Users/jeong-hyemin/Downloads/seoul-startup-competition/data/processed/report_viz/cluster/A2_mangwo_HH_overlay.png')

df = pd.read_csv(DATA)
mangwo = df[(df['GU_NM']=='중랑구') & (df['ADM_NM']=='망우본동')].copy()

n_total = len(mangwo)
hh = mangwo[mangwo['LISA_Cluster']=='HH']
hl = mangwo[mangwo['LISA_Cluster']=='HH']  # placeholder if needed
hl = mangwo[mangwo['LISA_Cluster']=='HL']
lh = mangwo[mangwo['LISA_Cluster']=='LH']
ll = mangwo[mangwo['LISA_Cluster']=='LL']
ns = mangwo[mangwo['LISA_Cluster']=='ns']

n_hh = len(hh)
n_hl = len(hl)
n_ns = len(ns)
n_other = n_total - n_hh - n_hl - n_ns
pop_65 = int(mangwo['pop_65plus_per_grid'].sum())

print(f"망우본동 격자: 총 {n_total} / HH {n_hh} / HL {n_hl} / ns {n_ns}")
print(f"65+ 인구 합: {pop_65:,}")

# === Figure ===
fig, ax = plt.subplots(figsize=(10, 11), dpi=200, facecolor='white')

# 비유의·LL·LH (옅게)
ax.scatter(ns['center_lon'], ns['center_lat'], s=12, c='#D3D9E0', alpha=0.5,
           label=f'ns (n={n_ns})', zorder=1)
ax.scatter(ll['center_lon'], ll['center_lat'], s=12, c='#A8B5C2', alpha=0.6,
           label=f'LL (n={len(ll)})', zorder=1)
ax.scatter(lh['center_lon'], lh['center_lat'], s=12, c='#7C99B5', alpha=0.6,
           label=f'LH (n={len(lh)})', zorder=1)
# HL 강조 (주황)
ax.scatter(hl['center_lon'], hl['center_lat'], s=22, c='#F0AD4E',
           edgecolor='white', linewidth=0.3, label=f'HL (n={n_hl})', zorder=3)
# HH 강조 (빨강)
ax.scatter(hh['center_lon'], hh['center_lat'], s=22, c='#D9534F',
           edgecolor='white', linewidth=0.3, label=f'HH (n={n_hh})', zorder=4)

# 좌측 상단 데이터 박스
data_lines = [
    '실측 데이터',
    f'전체 격자:   {n_total:,}',
    f'HH 격자:    {n_hh}',
    f'HL 격자:    {n_hl}',
    f'비유의 격자: {n_ns}',
    '─────────────',
    f'HH 비율:   {n_hh/n_total*100:.1f}%',
    '동 면적:    294 ha',
    f'65+ 인구:  {pop_65:,}명',
]
box_text = '\n'.join(data_lines)
ax.text(0.02, 0.98, box_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                  edgecolor='#1F3A5F', linewidth=1.5))

# Title
ax.set_title(f'중랑구 망우본동 — HH 핫스팟 격자 분포\n'
             f'전체 {n_total:,} 격자 중 HH {n_hh}개 ({n_hh/n_total*100:.1f}%) — 동 면적의 1/3이 통계적 핫스팟',
             fontsize=13, color='#1F3A5F', pad=12)
ax.set_xlabel('경도 (Longitude)', fontsize=11, color='#1F3A5F')
ax.set_ylabel('위도 (Latitude)', fontsize=11, color='#1F3A5F')

# Legend
leg = ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
                edgecolor='#1F3A5F')

ax.set_aspect('equal')
ax.grid(True, alpha=0.2, linestyle='--')
for spine in ax.spines.values():
    spine.set_edgecolor('#1F3A5F')
    spine.set_linewidth(0.8)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f'OK -> {OUT}  ({OUT.stat().st_size//1024} KB)')
