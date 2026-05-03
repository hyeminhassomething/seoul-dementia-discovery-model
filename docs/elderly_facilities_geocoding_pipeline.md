# 노인복지시설 지오코딩/격자/밀도 집계 파이프라인

## 현재 입력 상태

- 입력 파일: `data/interim/elderly_facilities_unified.csv`
- 전체 행: `1,380`
- 고유 주소: `1,259`
- 주소 누락: `5`
- `gu_name` 누락: `13`
- `dong_name` 누락: `410`
- 서울 외 주소로 보이는 행: `23`

## 실제 실행 결과

- 지오코딩 완료 파일: `data/interim/elderly_facilities_geocoded.csv`
- 격자 매핑 완료 파일: `data/interim/elderly_facilities_geocoded_grid.csv`
- 최종 정리 폴더: `data/processed/`
- 전체 시설: `1,380`
- 카카오 지오코딩 성공: `1,315`
- 카카오 지오코딩 실패: `65`
- 서울 1km 격자 매핑 성공: `1,292`
- 생성 격자 수: `1,292`

실패 주소는 `data/interim/elderly_facilities_geocode_failures.csv`에서 확인한다.

## 1. Kakao REST API 키 설정

카카오 디벨로퍼스에서 앱을 만들고, 앱 키 중 `REST API 키`를 환경변수로 설정한다.

```bash
export KAKAO_REST_API_KEY="..."
```

## 2. Kakao 지오코딩 실행

먼저 dry-run으로 입력 상태와 키 인식 여부를 확인한다.

```bash
python3 code/04_geocode_elderly_facilities_kakao.py --dry-run
```

5건만 테스트한다.

```bash
python3 code/04_geocode_elderly_facilities_kakao.py --limit 5
```

전체 지오코딩:

```bash
python3 code/04_geocode_elderly_facilities_kakao.py
```

산출물:

- `data/interim/elderly_facilities_geocoded.csv`
- `data/interim/elderly_facilities_geocode_failures.csv`
- `data/interim/kakao_geocode_cache.json`

중간에 실패해도 `kakao_geocode_cache.json`을 재사용하므로 같은 주소를 다시 호출하지 않는다.

## 참고: Naver Cloud API 키 설정

Naver Cloud Platform에서 Maps Geocoding API가 활성화된 Application의 키를 환경변수로 설정한다.

```bash
export NAVER_MAPS_CLIENT_ID="..."
export NAVER_MAPS_CLIENT_SECRET="..."
```

아래 이름도 지원한다.

```bash
export NCP_APIGW_API_KEY_ID="..."
export NCP_APIGW_API_KEY="..."
```

## 참고: Naver 지오코딩 실행

먼저 dry-run으로 입력 상태와 키 인식 여부를 확인한다.

```bash
python3 code/04_geocode_elderly_facilities_naver.py --dry-run
```

전체 지오코딩:

```bash
python3 code/04_geocode_elderly_facilities_naver.py
```

산출물:

- `data/interim/elderly_facilities_geocoded.csv`
- `data/interim/elderly_facilities_geocode_failures.csv`
- `data/interim/naver_geocode_cache.json`

중간에 실패해도 `naver_geocode_cache.json`을 재사용하므로 같은 주소를 다시 호출하지 않는다.

## 3. 격자 ID 매핑

이미 격자 파일이 있는 경우:

```bash
python3 code/05_coord_to_grid.py \
  --grid data/interim/seoul_grids.csv
```

격자 파일 컬럼은 아래 형태여야 한다.

```text
grid_id,min_lon,min_lat,max_lon,max_lat
B-3,126.9,37.5,126.91,37.51
```

격자 파일이 아직 없으면 서울 bbox 기준 1km 격자를 생성하면서 매핑한다.

```bash
python3 code/05_coord_to_grid.py --generate-grid --cell-size-m 1000
```

산출물:

- `data/interim/elderly_facilities_geocoded_grid.csv`
- `data/interim/seoul_generated_grids.csv`

## 4. 밀도 집계

```bash
python3 code/06_aggregate_elderly_facility_density.py
```

좌표가 성공한 시설만 집계하려면:

```bash
python3 code/06_aggregate_elderly_facility_density.py --only-geocoded
```

산출물:

- `data/interim/facility_density_gu_dong.csv`
- `data/interim/facility_density_grid.csv`
- `data/interim/facility_density_gu_dong_category.csv`

## 권장 확인 순서

1. `elderly_facilities_geocode_failures.csv`에서 `NO_RESULT`, `HTTP_...`, `ERROR` 주소를 확인한다.
2. 실패 주소 중 도로명만 있고 건물번호가 없는 주소는 수작업 보정 후보로 둔다.
3. `geocode_gu`, `geocode_dong`으로 빈 `gu_name`, `dong_name`이 얼마나 보강됐는지 확인한다.
4. 서울 분석만 할 경우 서울 외 주소 후보는 제외하거나 별도 플래그 처리한다.
