"""
05_coord_to_grid.py

좌표가 붙은 노인복지시설 CSV에 grid_id 컬럼을 추가한다.

두 가지 방식 지원:
  1) --grid로 기존 격자 CSV 사용
     필요 컬럼: grid_id,min_lon,min_lat,max_lon,max_lat

  2) --generate-grid로 좌표 범위 기준 격자 생성
     출력: data/interim/seoul_generated_grids.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import string
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_INPUT = INTERIM_DIR / "elderly_facilities_geocoded.csv"
DEFAULT_OUTPUT = INTERIM_DIR / "elderly_facilities_geocoded_grid.csv"
DEFAULT_GENERATED_GRID = INTERIM_DIR / "seoul_generated_grids.csv"

SEOUL_BOUNDS = {
    "min_lon": 126.764,
    "min_lat": 37.413,
    "max_lon": 127.184,
    "max_lat": 37.715,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="lon/lat 좌표를 grid_id로 매핑합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid", type=Path, help="기존 격자 CSV")
    parser.add_argument("--generated-grid-output", type=Path, default=DEFAULT_GENERATED_GRID)
    parser.add_argument("--generate-grid", action="store_true", help="서울 bbox 기준 격자를 생성해서 사용")
    parser.add_argument("--cell-size-m", type=float, default=1000.0, help="생성 격자 한 칸 크기(m)")
    parser.add_argument("--bounds-from-data", action="store_true", help="서울 고정 bbox 대신 입력 좌표 범위로 격자 생성")
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


def parse_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def row_label(index: int) -> str:
    alphabet = string.ascii_uppercase
    label = ""
    n = index
    while True:
        label = alphabet[n % 26] + label
        n = n // 26 - 1
        if n < 0:
            return label


def grid_id(row_index: int, col_index: int) -> str:
    return f"{row_label(row_index)}-{col_index + 1}"


def bounds_from_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    lons = [parse_float(row.get("lon")) for row in rows]
    lats = [parse_float(row.get("lat")) for row in rows]
    lons = [value for value in lons if value is not None]
    lats = [value for value in lats if value is not None]
    if not lons or not lats:
        raise ValueError("입력 파일에 유효한 lon/lat 좌표가 없습니다.")
    margin = 0.005
    return {
        "min_lon": min(lons) - margin,
        "min_lat": min(lats) - margin,
        "max_lon": max(lons) + margin,
        "max_lat": max(lats) + margin,
    }


def generate_grids(bounds: dict[str, float], cell_size_m: float) -> list[dict[str, Any]]:
    mid_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    lat_step = cell_size_m / 111_320
    lon_step = cell_size_m / (111_320 * math.cos(math.radians(mid_lat)))

    row_count = math.ceil((bounds["max_lat"] - bounds["min_lat"]) / lat_step)
    col_count = math.ceil((bounds["max_lon"] - bounds["min_lon"]) / lon_step)

    grids: list[dict[str, Any]] = []
    # A행이 북쪽에 오도록 row 0부터 위에서 아래로 내려간다.
    for row_index in range(row_count):
        max_lat = bounds["max_lat"] - row_index * lat_step
        min_lat = max_lat - lat_step
        for col_index in range(col_count):
            min_lon = bounds["min_lon"] + col_index * lon_step
            max_lon = min_lon + lon_step
            grids.append(
                {
                    "grid_id": grid_id(row_index, col_index),
                    "min_lon": f"{min_lon:.8f}",
                    "min_lat": f"{min_lat:.8f}",
                    "max_lon": f"{max_lon:.8f}",
                    "max_lat": f"{max_lat:.8f}",
                }
            )
    return grids


def find_grid(lon: float | None, lat: float | None, grids: list[dict[str, str]]) -> str:
    if lon is None or lat is None:
        return ""
    for grid in grids:
        if (
            float(grid["min_lon"]) <= lon < float(grid["max_lon"])
            and float(grid["min_lat"]) <= lat < float(grid["max_lat"])
        ):
            return grid["grid_id"]
    return ""


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)

    if args.grid:
        grids = read_rows(args.grid)
    elif args.generate_grid:
        bounds = bounds_from_rows(rows) if args.bounds_from_data else SEOUL_BOUNDS
        grids = generate_grids(bounds, args.cell_size_m)
        write_rows(
            args.generated_grid_output,
            grids,
            ["grid_id", "min_lon", "min_lat", "max_lon", "max_lat"],
        )
        print(
            f"generated grid: {args.generated_grid_output.relative_to(PROJECT_ROOT)} "
            f"({len(grids)} cells, {args.cell_size_m:g}m)"
        )
    else:
        raise SystemExit("--grid 또는 --generate-grid 중 하나가 필요합니다.")

    mapped = 0
    for row in rows:
        lon = parse_float(row.get("lon"))
        lat = parse_float(row.get("lat"))
        row["grid_id"] = find_grid(lon, lat, grids)
        if row["grid_id"]:
            mapped += 1

    fieldnames = list(rows[0].keys())
    if "grid_id" not in fieldnames:
        fieldnames.append("grid_id")
    write_rows(args.output, rows, fieldnames)
    print(f"saved: {args.output.relative_to(PROJECT_ROOT)}")
    print(f"mapped={mapped}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
