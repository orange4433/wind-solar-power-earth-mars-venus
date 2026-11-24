from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_YEAR = 2022  # Change this to 2022, 2023, etc. if preferred.


def country_sources(metadata: Path) -> list[tuple[str, str, list[str]]]:
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["country_code"]].append(row)

    countries = []
    for code, zones in grouped.items():
        national = next((row for row in zones if row["zone_key"] == code), None)
        if national:
            selected = [code]
            name = national["country_name"] or national["zone_name"]
        else:
            leaves = [row for row in zones if not row["sub_zone_keys"]]
            selected = [row["zone_key"] for row in (leaves or zones)]
            name = next((row["country_name"] for row in zones if row["country_name"]), code)
        countries.append((name, code, selected))
    return sorted(countries)


def merge(input_path: Path, metadata: Path, output: Path) -> None:
    countries = country_sources(metadata)
    with input_path.open(newline="", encoding="utf-8-sig") as source, output.open(
        "w", newline="", encoding="utf-8-sig"
    ) as target:
        reader = csv.reader(source)
        header = next(reader)
        indexes = {name: index for index, name in enumerate(header)}
        missing = [
            f"{zone}_{kind}_mw"
            for _, _, zones in countries
            for zone in zones
            for kind in ("wind", "solar")
            if f"{zone}_{kind}_mw" not in indexes
        ]
        if missing:
            raise SystemExit(f"Input is missing columns: {', '.join(missing[:10])}")

        writer = csv.writer(target)
        writer.writerow(
            ["datetime_utc"]
            + [
                column
                for name, code, _ in countries
                for column in (f"{name} [{code}]_wind_mw", f"{name} [{code}]_solar_mw")
            ]
        )
        row_count = 0
        for row in reader:
            merged = [row[0]]
            for _, _, zones in countries:
                for kind in ("wind", "solar"):
                    values = [row[indexes[f"{zone}_{kind}_mw"]] for zone in zones]
                    present = [Decimal(value) for value in values if value]
                    merged.append("" if not present else format(sum(present), "f"))
            writer.writerow(merged)
            row_count += 1

    print(f"Saved {row_count} hours x {1 + 2 * len(countries)} columns to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge an Electricity Maps zone-wide table into non-overlapping country columns."
    )
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    merge(
        args.input or HERE / f"global_{args.year}_hourly_wind_solar.csv",
        args.metadata or HERE / f"global_{args.year}_zones.csv",
        args.output or HERE / f"global_{args.year}_hourly_wind_solar_by_country.csv",
    )


if __name__ == "__main__":
    main()
