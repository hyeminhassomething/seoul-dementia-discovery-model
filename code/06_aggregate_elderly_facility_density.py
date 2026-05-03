"""
06_aggregate_elderly_facility_density.py

좌표/격자가 붙은 노인복지시설 데이터를 기준으로 시설 수를 집계한다.

출력:
  data/interim/facility_density_gu_dong.csv
  data/interim/facility_density_grid.csv
  data/interim/facility_density_gu_dong_category.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DEFAULT_INPUT = INTERIM_DIR / "elderly_facilities_geocoded_grid.csv"
DEFAULT_GU_DONG_OUTPUT = INTERIM_DIR / "facility_density_gu_dong.csv"
DEFAULT_GRID_OUTPUT = INTERIM_DIR / "facility_density_grid.csv"
DEFAULT_CATEGORY_OUTPUT = INTERIM_DIR / "facility_density_gu_dong_category.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="노인복지시설 밀도 집계 CSV를 생성합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gu-dong-output", type=Path, default=DEFAULT_GU_DONG_OUTPUT)
    parser.add_argument("--grid-output", type=Path, default=DEFAULT_GRID_OUTPUT)
    parser.add_argument("--category-output", type=Path, default=DEFAULT_CATEGORY_OUTPUT)
    parser.add_argument("--only-geocoded", action="store_true", help="좌표가 있는 시설만 집계")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_counter(path: Path, columns: list[str], counts: Counter[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns + ["facility_count"])
        for key, count in sorted(counts.items()):
            writer.writerow(list(key) + [count])


def valid_rows(rows: Iterable[dict[str, str]], *, only_geocoded: bool) -> list[dict[str, str]]:
    if not only_geocoded:
        return list(rows)
    return [
        row for row in rows
        if row.get("lon", "").strip() and row.get("lat", "").strip()
    ]


def main() -> int:
    args = parse_args()
    rows = valid_rows(read_rows(args.input), only_geocoded=args.only_geocoded)

    gu_dong_counts: Counter[tuple[str, ...]] = Counter(
        (row.get("gu_name", "").strip(), row.get("dong_name", "").strip())
        for row in rows
        if row.get("gu_name", "").strip()
    )
    grid_counts: Counter[tuple[str, ...]] = Counter(
        (row.get("grid_id", "").strip(),)
        for row in rows
        if row.get("grid_id", "").strip()
    )
    category_counts: Counter[tuple[str, ...]] = Counter(
        (
            row.get("gu_name", "").strip(),
            row.get("dong_name", "").strip(),
            row.get("category_main", "").strip(),
        )
        for row in rows
        if row.get("gu_name", "").strip()
    )

    write_counter(args.gu_dong_output, ["gu_name", "dong_name"], gu_dong_counts)
    write_counter(args.grid_output, ["grid_id"], grid_counts)
    write_counter(args.category_output, ["gu_name", "dong_name", "category_main"], category_counts)

    print(f"input_rows={len(rows)}")
    print(f"saved: {args.gu_dong_output.relative_to(PROJECT_ROOT)} ({len(gu_dong_counts)} rows)")
    print(f"saved: {args.grid_output.relative_to(PROJECT_ROOT)} ({len(grid_counts)} rows)")
    print(f"saved: {args.category_output.relative_to(PROJECT_ROOT)} ({len(category_counts)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
