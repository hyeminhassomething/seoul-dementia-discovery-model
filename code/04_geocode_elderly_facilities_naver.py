"""
04_geocode_elderly_facilities_naver.py

Naver Cloud Geocoding API로 elderly_facilities_unified.csv의 address를 좌표로 변환한다.

환경변수:
  NAVER_MAPS_CLIENT_ID      또는 NCP_APIGW_API_KEY_ID
  NAVER_MAPS_CLIENT_SECRET  또는 NCP_APIGW_API_KEY

출력:
  data/interim/elderly_facilities_geocoded.csv
  data/interim/elderly_facilities_geocode_failures.csv
  data/interim/naver_geocode_cache.json
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
DEFAULT_CACHE = INTERIM_DIR / "naver_geocode_cache.json"

GEOCODE_URL = "https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode"
BASE_FIELDS = [
    "lon",
    "lat",
    "road_address",
    "jibun_address",
    "geocode_status",
    "geocode_query",
    "geocode_error",
    "geocode_sido",
    "geocode_gu",
    "geocode_dong",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naver Cloud Geocoding API로 노인복지시설 주소에 좌표를 붙입니다."
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


def get_api_keys() -> tuple[str | None, str | None]:
    key_id = (
        os.environ.get("NAVER_MAPS_CLIENT_ID")
        or os.environ.get("NCP_APIGW_API_KEY_ID")
        or os.environ.get("NAVER_CLIENT_ID")
    )
    key = (
        os.environ.get("NAVER_MAPS_CLIENT_SECRET")
        or os.environ.get("NCP_APIGW_API_KEY")
        or os.environ.get("NAVER_CLIENT_SECRET")
    )
    return key_id, key


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
    return candidates


def extract_address_elements(address: dict[str, Any]) -> dict[str, str]:
    out = {"geocode_sido": "", "geocode_gu": "", "geocode_dong": ""}
    for element in address.get("addressElements", []):
        types = set(element.get("types", []))
        name = element.get("longName") or element.get("shortName") or ""
        if "SIDO" in types:
            out["geocode_sido"] = name
        elif "SIGUGUN" in types:
            out["geocode_gu"] = name
        elif "DONGMYUN" in types:
            out["geocode_dong"] = name
    return out


def empty_result(status: str, query: str, error: str = "") -> dict[str, Any]:
    return {
        "lon": "",
        "lat": "",
        "road_address": "",
        "jibun_address": "",
        "geocode_status": status,
        "geocode_query": query,
        "geocode_error": error,
        "geocode_sido": "",
        "geocode_gu": "",
        "geocode_dong": "",
    }


def call_naver_geocode(query: str, key_id: str, key: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(f"{GEOCODE_URL}?{params}")
    req.add_header("X-NCP-APIGW-API-KEY-ID", key_id)
    req.add_header("X-NCP-APIGW-API-KEY", key)

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return empty_result(f"HTTP_{e.code}", query, body[:500])
    except Exception as e:
        return empty_result("ERROR", query, str(e))

    status = data.get("status") or "UNKNOWN"
    addresses = data.get("addresses") or []
    if not addresses:
        return empty_result("NO_RESULT" if status == "OK" else status, query)

    best = addresses[0]
    result = {
        "lon": best.get("x", ""),
        "lat": best.get("y", ""),
        "road_address": best.get("roadAddress", ""),
        "jibun_address": best.get("jibunAddress", ""),
        "geocode_status": status,
        "geocode_query": query,
        "geocode_error": "",
    }
    result.update(extract_address_elements(best))
    return result


def geocode_with_fallbacks(
    address: str,
    cache: dict[str, dict[str, Any]],
    key_id: str,
    key: str,
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
            result = call_naver_geocode(candidate, key_id, key)
            cache[candidate] = result
            time.sleep(sleep_seconds)
        last = result
        if result.get("lat") and result.get("lon"):
            cache[query] = result
            return result
    cache[query] = last
    return last


def print_readiness(rows: list[dict[str, str]], key_id: str | None, key: str | None) -> None:
    addresses = [row.get("address", "").strip() for row in rows]
    missing_address = sum(1 for address in addresses if not address)
    unique_addresses = len({address for address in addresses if address})
    missing_gu = sum(1 for row in rows if not row.get("gu_name", "").strip())
    missing_dong = sum(1 for row in rows if not row.get("dong_name", "").strip())
    non_seoul = [
        row for row in rows
        if row.get("address", "").strip()
        and "서울" not in row.get("address", "")
        and row.get("gu_name", "").strip() not in {
            "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
            "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
            "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
        }
    ]

    print(f"rows={len(rows)}")
    print(f"unique_addresses={unique_addresses}")
    print(f"missing_address={missing_address}")
    print(f"missing_gu_name={missing_gu}")
    print(f"missing_dong_name={missing_dong}")
    print(f"likely_non_seoul_rows={len(non_seoul)}")
    print(f"api_key_id_present={bool(key_id)}")
    print(f"api_key_present={bool(key)}")
    print("\n[sample normalized queries]")
    for row in rows[:5]:
        print(f"- {row.get('facility_id')}: {normalize_query(row.get('address', ''))}")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    key_id, key = get_api_keys()
    print_readiness(rows, key_id, key)

    if args.dry_run:
        print("\ndry-run: API 호출 없이 종료합니다.")
        return 0

    if not key_id or not key:
        print(
            "\nERROR: Naver API 키가 없습니다. "
            "NAVER_MAPS_CLIENT_ID/NAVER_MAPS_CLIENT_SECRET 환경변수를 설정해 주세요.",
            file=sys.stderr,
        )
        return 2

    cache = load_cache(args.cache)
    out_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    total = len(rows)
    for index, row in enumerate(rows, start=1):
        result = geocode_with_fallbacks(
            row.get("address", ""),
            cache,
            key_id,
            key,
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

    print(f"\nsaved: {args.output.relative_to(PROJECT_ROOT)}")
    print(f"failures: {args.failures.relative_to(PROJECT_ROOT)} ({len(failures)} rows)")
    print(f"cache: {args.cache.relative_to(PROJECT_ROOT)} ({len(cache)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
