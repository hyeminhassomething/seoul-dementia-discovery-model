# 서울시 데이터 분석 경진대회 — 치매 노인 실종자 발견 확률 모델

서울시 50m 격자 단위 **치매 노인 실종 시 발견 확률** 모델. 7개 feature 가중합 + 보수적 임계치 기반 사건별 우선 수색 격자 추천.

## 🎯 모델 핵심

```
발견확률 = 교차로_밀도   × 0.35   ← Bayat et al. 실증
        + 도로_복잡도   × 0.25   ← 길을 잃기 쉬움
        + 용도지역_주택가 × 0.15  ← 배회 목적지
        + 공원_쉼터_밀도 × 0.10   ← 휴식지
        + CCTV_밀도     × 0.08   ← 발견 수단
        + 노인복지시설_밀도 × 0.04 ← 약한 신호
        + 65+ 인구밀도   × 0.03   ← 보조 (KT 유동인구 자리, 추후 교체 예정)
```

**보수적 임계치 (사건 후 경과 시간 → 배회 반경):**
- 1시간 → 800m
- 6시간 → 2,500m
- 12시간 → 4,000m
- 24시간 → 6,000m

## 📂 프로젝트 구조

```
seoul-startup-competition/
├─ code/                              # 처리 파이프라인 (실행 순서대로 번호)
│  ├─ 03_unify_elderly_facilities.py     # 노인복지시설 4개 파일 통합
│  ├─ 04_geocode_*.py                    # 카카오 지오코딩
│  ├─ 05_coord_to_grid.py                # (구) 1km 격자 — 보조용
│  ├─ 06_aggregate_*.py                  # (구) 1km 밀도
│  ├─ 07_unify_cctv_points.py            # CCTV 30개 파일 → 9,535개 점
│  ├─ 08_build_50m_grid.py               # 50m 격자 시스템
│  ├─ 09_cctv_coverage_features.py       # 시설별 반경 CCTV 커버리지
│  ├─ 10_extract_road_network.py         # OSM 도로망 → 교차로/복잡도
│  ├─ 11_zoning_to_grid.py               # 용도지역 → 격자 분류
│  ├─ 12_unify_rest_areas.py             # 공원/쉼터 통합
│  ├─ 13_geocode_senior_centers.py       # 경로당 좌표 부착
│  ├─ 14_rest_areas_grid_features.py     # 휴식지 격자 feature
│  ├─ 15_cctv_facility_grid_features.py  # CCTV·시설 격자 feature
│  ├─ 16_population_to_grid.py           # 인구 → 격자 broadcast
│  ├─ 17_discovery_probability_model.py  # ⭐ 7-feature 종합 모델
│  ├─ 18_incident_search_candidates.py   # 사건별 후보 격자 추출
│  └─ 19_visualize_discovery_map.py      # folium 지도 생성
│
├─ data/
│  ├─ raw/             # ❌ gitignore (각자 다운로드 — 아래 안내 참고)
│  ├─ interim/         # 부분 commit (큰 파일은 재생성)
│  └─ processed/       # ✅ 모델 산출물 (commit)
│     ├─ grid50_master_features.csv         (32 cols × 121,051 grids)
│     ├─ grid50_discovery_probability.csv   ⭐ 모델 출력
│     ├─ seoul_discovery_heatmap.html        서울 전체 히트맵
│     └─ incident_demo_map.html              광화문 6h 시나리오 지도
│
└─ docs/               # 파이프라인 문서
```

## 🚀 환경 셋업

```bash
# Python 3.14 + Homebrew 가정
cd seoul-startup-competition
python3 -m venv .venv
source .venv/bin/activate
pip install pandas openpyxl numpy geopandas shapely pyproj osmnx networkx folium
```

## 🔑 카카오 API 키 (지오코딩에 필요)

경로당 좌표 부착 등에 카카오 Local API 사용. 본인 키로 환경변수 설정:

```bash
export KAKAO_REST_API_KEY="여기에_본인_키"
```

키 발급: https://developers.kakao.com/console → 앱 추가 → REST API 키 복사

## 📥 원본 데이터 다운로드 안내

`data/raw/` 폴더는 gitignore. 아래 5종을 직접 받아 폴더 구조 맞춰 저장:

```
data/raw/
├─ elderly_facilities/      # 노인복지시설 4개 (서울 열린데이터광장)
│  ├─ 01_노인주거복지시설.csv
│  ├─ 02_노인의료복지시설.csv
│  ├─ 03_노인의료복지시설현황.xlsx
│  └─ 04_노인여가복지시설.csv
├─ cctv/                    # CCTV 위치 데이터 (구별 + 시 전체)
│  └─ 서울시 불법주정차_전용차로 위반 단속 CCTV 위치정보.csv  (마스터, 4,652점)
│  └─ + 강북·금천·은평·영등포·관악 일반 CCTV
├─ zoning/shp파일/          # 용도지역 폴리곤 (공공데이터포털 / VWorld)
│  ├─ UPIS_C_UQ111.shp + .dbf + .prj + .shx + .sbn + .sbx + .shp.xml
├─ rest_areas/              # 휴식지 4종
│  ├─ 01_seoul_parks.csv
│  ├─ 02_seoul_senior_centers.csv
│  ├─ 03_seoul_cool_shelter.csv
│  └─ 04_seoul_warm_shelters.csv
└─ population/
   └─ 주민등록인구(내국인+각+세별_구별)*.csv
```

각 데이터 출처는 `docs/` 또는 코드 상단 주석 참고.

## ⏯️ 파이프라인 실행

```bash
source .venv/bin/activate
export KAKAO_REST_API_KEY="..."

# 순서대로 실행 (각 스크립트가 직전 출력에 의존)
python code/03_unify_elderly_facilities.py        # 시설 통합
python code/04_geocode_elderly_facilities_kakao.py  # 시설 좌표
python code/07_unify_cctv_points.py               # CCTV 통합
python code/08_build_50m_grid.py                  # 50m 격자
python code/10_extract_road_network.py            # 도로망 (5~10분, OSM 다운로드)
python code/11_zoning_to_grid.py                  # 용도지역
python code/12_unify_rest_areas.py                # 휴식지
python code/13_geocode_senior_centers.py          # 경로당 좌표
python code/14_rest_areas_grid_features.py
python code/15_cctv_facility_grid_features.py
python code/16_population_to_grid.py
python code/17_discovery_probability_model.py     # ⭐ 모델 종합
python code/19_visualize_discovery_map.py         # 지도 생성
```

## 🔍 모델 사용 예시 (사건 후보 격자 추출)

```bash
# 광화문에서 노인 실종, 6시간 경과
python code/18_incident_search_candidates.py \
  --lon 126.9770 --lat 37.5759 --hours 6 --top 20 \
  --out data/processed/incident_광화문.csv

# 또는 30개 가상 사건 시뮬레이션
python code/18_incident_search_candidates.py --simulate-demo
```

## 📊 결과 시각화

브라우저에서 열기:
```bash
open data/processed/seoul_discovery_heatmap.html
open data/processed/incident_demo_map.html
```

## 📈 모델 검증 결과 (스냅샷)

- 격자 수: **121,051** (서울 보행 도로망 있는 격자)
- 평균 발견확률: 0.275, 99 percentile: 0.681
- **자치구별 평균 Top 5:** 동대문 0.394, 중랑 0.394, 서대문 0.368, 은평 0.365, 강북 0.356
- **광화문 6h 시나리오:** 검색 반경 2,500m, 추천 1~10위 모두 종로구 인사동·익선동 (발견확률 0.68~0.69)

## 🧱 추후 작업

- KT 유동인구 데이터 (빅데이터 캠퍼스) → 가중치 0.03 자리 교체
- 안전안내문자 OpenAPI 파싱 → Phase 3 모델 검증 (실제 발견 위치 hit-rate)
