"""
13_geocode_senior_centers.py
경로당 주소 → 카카오 지오코딩 → 좌표 부착.

[입력]
  data/interim/senior_centers_addresses_pending_geocode.csv  (3,645 주소)
  data/interim/kakao_geocode_cache.json (재활용)

[출력]
  data/interim/senior_centers_geocoded.csv
  → rest_areas_unified.csv 에 추가됨 (이 스크립트 마지막 단계에서 자동 병합)

[환경변수]
  KAKAO_REST_API_KEY 필수
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM = PROJECT_ROOT / "data/interim"
INPUT = INTERIM / "senior_centers_addresses_pending_geocode.csv"
CACHE_PATH = INTERIM / "kakao_geocode_cache.json"
OUTPUT = INTERIM / "senior_centers_geocoded.csv"
UNIFIED_PATH = INTERIM / "rest_areas_unified.csv"

API_URL = "https://dapi.kakao.com/v2/local/search/address.json"


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with CACHE_PATH.open("w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)


def call_kakao(query: str, key: str) -> dict | None:
    """카카오 주소 검색. 결과 dict 또는 None."""
    url = API_URL + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"KakaoAK {key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}
    docs = data.get("documents", [])
    if not docs:
        return {"error": "no_result"}
    d = docs[0]
    return {
        "lon": float(d["x"]),
        "lat": float(d["y"]),
        "road_address": (d.get("road_address") or {}).get("address_name", ""),
        "jibun_address": (d.get("address") or {}).get("address_name", ""),
        "region_2depth_name": (d.get("address") or {}).get("region_2depth_name", ""),
        "region_3depth_name": (d.get("address") or {}).get("region_3depth_name", ""),
    }


def normalize(addr: str) -> str:
    """주소 정규화. 카카오에 입력하기 좋은 형태로."""
    if not isinstance(addr, str):
        return ""
    s = addr.strip()
    # "(괄호) 내용" 제거 (건물명 일부) — 첫 괄호까지만 남기는 게 더 잘 매칭됨
    # 그러나 도로명주소에서 "(동명)"은 유지가 좋음. 일단 그대로 사용.
    return s


def main() -> None:
    key = os.environ.get("KAKAO_REST_API_KEY")
    if not key:
        print("ERROR: KAKAO_REST_API_KEY 환경변수가 없습니다.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT, encoding="utf-8-sig")
    print(f"[input] {len(df):,} addresses")

    cache = load_cache()
    print(f"[cache] {len(cache):,} entries")

    results = []
    cache_hits = 0
    api_calls = 0
    failures = 0

    for idx, row in df.iterrows():
        name = str(row["name"]).strip()
        addr = normalize(row.get("address", ""))
        if not addr or addr in {"nan", ""}:
            results.append({**row, "lon": None, "lat": None, "geocode_status": "no_address"})
            failures += 1
            continue

        if addr in cache:
            cached = cache[addr]
            if "lon" in cached and "lat" in cached and cached.get("lon") is not None:
                results.append({**row, "lon": cached["lon"], "lat": cached["lat"],
                               "geocode_status": "cache_hit"})
                cache_hits += 1
                continue
            else:
                # 이전에 실패한 캐시. 재시도 안 함.
                results.append({**row, "lon": None, "lat": None,
                               "geocode_status": "cache_miss"})
                failures += 1
                continue

        # API 호출
        time.sleep(0.05)  # rate limit 보호 (20 req/s 정도)
        api_calls += 1
        result = call_kakao(addr, key)
        if result and "lon" in result:
            cache[addr] = result
            results.append({**row, "lon": result["lon"], "lat": result["lat"],
                           "geocode_status": "api_ok"})
        else:
            cache[addr] = {"error": (result or {}).get("error", "unknown"), "lon": None, "lat": None}
            results.append({**row, "lon": None, "lat": None,
                           "geocode_status": "api_fail"})
            failures += 1

        if api_calls % 200 == 0:
            print(f"  api={api_calls}, cache_hits={cache_hits}, failures={failures}")
            save_cache(cache)

    save_cache(cache)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"\n[geocode summary]")
    print(f"  cache_hits: {cache_hits:,}")
    print(f"  api_calls:  {api_calls:,}")
    print(f"  failures:   {failures:,}")
    print(f"  saved:      {OUTPUT.relative_to(PROJECT_ROOT)}")

    # ─── rest_areas_unified.csv 에 병합 ───
    seoul_lat = (37.41, 37.72)
    seoul_lon = (126.74, 127.20)
    unified = pd.read_csv(UNIFIED_PATH)

    sc = out_df[out_df["lon"].notna() & out_df["lat"].notna()].copy()
    sc = sc[
        sc["lat"].between(*seoul_lat) & sc["lon"].between(*seoul_lon)
    ]
    sc_to_add = pd.DataFrame({
        "rest_id": "",  # rebuild after merge
        "kind": "senior_center",
        "name": sc["name"].astype(str).str.strip(),
        "address": sc["address"].astype(str).str.strip(),
        "area_m2": pd.NA,
        "lon": sc["lon"],
        "lat": sc["lat"],
        "source_file": "02_seoul_senior_centers.csv",
    })

    combined = pd.concat([unified.drop(columns=["rest_id"]), sc_to_add.drop(columns=["rest_id"])],
                         ignore_index=True)
    # dedup
    combined["_lat_r"] = combined["lat"].round(4)
    combined["_lon_r"] = combined["lon"].round(4)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["_lat_r", "_lon_r", "name"], keep="first")
    combined = combined.drop(columns=["_lat_r", "_lon_r"]).reset_index(drop=True)
    print(f"\n[merge] unified={len(unified):,} + senior_center={len(sc_to_add):,} → "
          f"{before:,} → dedup={len(combined):,}")

    combined = combined.sort_values(["kind", "lat", "lon"]).reset_index(drop=True)
    combined["rest_id"] = "R-" + (combined.index + 1).astype(str).str.zfill(5)
    combined = combined[["rest_id", "kind", "name", "address", "area_m2",
                         "lon", "lat", "source_file"]]
    combined.to_csv(UNIFIED_PATH, index=False, encoding="utf-8-sig")

    print(f"\n[final] rest_areas_unified.csv updated: {len(combined):,} points")
    print(combined["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
