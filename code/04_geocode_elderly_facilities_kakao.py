"""
04_geocode_elderly_facilities_kakao.py

Kakao Local API로 elderly_facilities_unified.csv의 address를 좌표로 변환한다.

환경변수:
  KAKAO_REST_API_KEY

출력:
  data/interim/elderly_facilities_geocoded.csv
  data/interim/elderly_facilities_geocode_failures.csv
  data/interim/kakao_geocode_cache.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_INPUT = INTERIM_DIR / "elderly_facilities_unified.csv"
DEFAULT_OUTPUT = INTERIM_DIR / "elderly_facilities_geocoded.csv"
DEFAULT_FAILURES = INTERIM_DIR / "elderly_facilities_geocode_failures.csv"
DEFAULT_CACHE = INTERIM_DIR / "kakao_geocode_cache.json"

ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
BASE_FIELDS = [
    "lon",
    "lat",
    "road_address",
    "jibun_address",
    "geocode_status",
    "geocode_provider",
    "geocode_query",
    "geocode_error",
    "geocode_sido",
    "geocode_gu",
    "geocode_dong",
    "kakao_address_type",
    "kakao_h_code",
    "kakao_b_code",
    "kakao_building_name",
    "kakao_zone_no",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kakao Local API로 노인복지시설 주소에 좌표를 붙입니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sleep", type=float, default=0.12, help="API 호출 간 대기 초")
    parser.add_argument("--limit", type=int, default=0, help="앞 n개 행만 처리. 0이면 전체")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 입력 상태만 점검")
    parser.add_argument("--force", action="store_true", help="캐시가 있어도 API를 다시 호출")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalize_query(address: str) -> str:
    query = (address or "").strip()
    query = query.replace("서울시", "서울특별시")
    query = re.sub(r"\s+", " ", query)
    query = re.sub(r"(\d+)\s+번지", r"\1번지", query)
    query = re.sub(r"(\d+)\s+호", r"\1호", query)
    return query.strip()


def fallback_queries(query: str) -> list[str]:
    candidates = [query]
    no_paren = re.sub(r"\([^)]*\)", "", query).strip()
    if no_paren and no_paren not in candidates:
        candidates.append(no_paren)
    no_floor = re.sub(r",?\s*\d+\s*층.*$", "", no_paren).strip()
    if no_floor and no_floor not in candidates:
        candidates.append(no_floor)
    no_building = re.sub(r"\s+[가-힣A-Za-z0-9+·._-]+(?:빌딩|센터|회관|타워|요양원|복지관).*$", "", no_floor).strip()
    if no_building and no_building not in candidates:
        candidates.append(no_building)
    return candidates


def empty_result(status: str, query: str, error: str = "") -> dict[str, Any]:
    return {
        "lon": "",
        "lat": "",
        "road_address": "",
        "jibun_address": "",
        "geocode_status": status,
        "geocode_provider": "kakao",
        "geocode_query": query,
        "geocode_error": error,
        "geocode_sido": "",
        "geocode_gu": "",
        "geocode_dong": "",
        "kakao_address_type": "",
        "kakao_h_code": "",
        "kakao_b_code": "",
        "kakao_building_name": "",
        "kakao_zone_no": "",
    }


def pick_region(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    address = document.get("address") or {}
    road_address = document.get("road_address") or {}
    region_source = address or road_address
    return address, road_address if isinstance(road_address, dict) else {}, region_source


def parse_kakao_document(document: dict[str, Any], query: str) -> dict[str, Any]:
    address, road_address, region_source = pick_region(document)
    return {
        "lon": document.get("x", ""),
        "lat": document.get("y", ""),
        "road_address": road_address.get("address_name", ""),
        "jibun_address": address.get("address_name", ""),
        "geocode_status": "OK",
        "geocode_provider": "kakao",
        "geocode_query": query,
        "geocode_error": "",
        "geocode_sido": region_source.get("region_1depth_name", ""),
        "geocode_gu": region_source.get("region_2depth_name", ""),
        "geocode_dong": address.get("region_3depth_h_name") or region_source.get("region_3depth_name", ""),
        "kakao_address_type": document.get("address_type", ""),
        "kakao_h_code": address.get("h_code", ""),
        "kakao_b_code": address.get("b_code", ""),
        "kakao_building_name": road_address.get("building_name", ""),
        "kakao_zone_no": road_address.get("zone_no", ""),
    }


def call_kakao_address(query: str, rest_api_key: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query": query, "analyze_type": "similar", "size": 10})
    req = urllib.request.Request(f"{ADDRESS_URL}?{params}")
    req.add_header("Authorization", f"KakaoAK {rest_api_key}")

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return empty_result(f"HTTP_{e.code}", query, body[:500])
    except Exception as e:
        return empty_result("ERROR", query, str(e))

    documents = data.get("documents") or []
    if not documents:
        total = (data.get("meta") or {}).get("total_count", 0)
        return empty_result("NO_RESULT", query, f"total_count={total}")

    return parse_kakao_document(documents[0], query)


def geocode_with_fallbacks(
    address: str,
    cache: dict[str, dict[str, Any]],
    rest_api_key: str,
    *,
    sleep_seconds: float,
    force: bool,
) -> dict[str, Any]:
    query = normalize_query(address)
    if not query:
        return empty_result("EMPTY_ADDRESS", query)

    if not force and query in cache:
        return cache[query]

    last = empty_result("NO_RESULT", query)
    for candidate in fallback_queries(query):
        if not force and candidate in cache:
            result = cache[candidate]
        else:
            result = call_kakao_address(candidate, rest_api_key)
            cache[candidate] = result
            time.sleep(sleep_seconds)
        last = result
        if result.get("lat") and result.get("lon"):
            cache[query] = result
            return result

    cache[query] = last
    return last


def print_readiness(rows: list[dict[str, str]], rest_api_key: str | None) -> None:
    addresses = [row.get("address", "").strip() for row in rows]
    print(f"rows={len(rows)}")
    print(f"unique_addresses={len({address for address in addresses if address})}")
    print(f"missing_address={sum(1 for address in addresses if not address)}")
    print(f"missing_gu_name={sum(1 for row in rows if not row.get('gu_name', '').strip())}")
    print(f"missing_dong_name={sum(1 for row in rows if not row.get('dong_name', '').strip())}")
    print(f"kakao_rest_api_key_present={bool(rest_api_key)}")
    print("\n[sample normalized queries]")
    for row in rows[:5]:
        print(f"- {row.get('facility_id')}: {normalize_query(row.get('address', ''))}")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    rest_api_key = os.environ.get("KAKAO_REST_API_KEY")
    print_readiness(rows, rest_api_key)

    if args.dry_run:
        print("\ndry-run: API 호출 없이 종료합니다.")
        return 0

    if not rest_api_key:
        print("\nERROR: KAKAO_REST_API_KEY 환경변수를 설정해 주세요.", file=sys.stderr)
        return 2

    cache = load_cache(args.cache)
    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        result = geocode_with_fallbacks(
            row.get("address", ""),
            cache,
            rest_api_key,
            sleep_seconds=args.sleep,
            force=args.force,
        )
        merged = dict(row)
        merged.update(result)

        if not merged.get("gu_name") and result.get("geocode_gu"):
            merged["gu_name"] = result["geocode_gu"]
        if not merged.get("dong_name") and result.get("geocode_dong"):
            merged["dong_name"] = result["geocode_dong"]

        out_rows.append(merged)
        if not result.get("lat") or not result.get("lon"):
            failures.append(merged)

        if index % 50 == 0 or index == total:
            save_cache(args.cache, cache)
            print(f"processed={index}/{total}, failures={len(failures)}, cache={len(cache)}")

    fieldnames = list(rows[0].keys()) + [field for field in BASE_FIELDS if field not in rows[0]]
    write_rows(args.output, out_rows, fieldnames)
    write_rows(args.failures, failures, fieldnames)
    save_cache(args.cache, cache)

    print(f"\nsaved: {display_path(args.output)}")
    print(f"failures: {display_path(args.failures)} ({len(failures)} rows)")
    print(f"cache: {display_path(args.cache)} ({len(cache)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
