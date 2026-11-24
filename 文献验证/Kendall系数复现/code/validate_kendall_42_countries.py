from __future__ import annotations

import csv
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data" / "42_major_countries"
PUBLISHED_XLSX = PROJECT / "data" / "Supplementary_Data_2.xlsx"
OUTPUT = PROJECT / "results" / "kendall_validation_42_countries.csv"
COUNTRY_RE = re.compile(r"_mean_(.+)\.csv$")
XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def country_from_path(path: Path) -> str:
    match = COUNTRY_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse country from {path.name}")
    return match.group(1)


def indexed_files(prefix: str) -> dict[str, Path]:
    return {country_from_path(path): path for path in DATA_DIR.glob(f"{prefix}*.csv")}


def read_capacity(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=5)
    frame.columns = ["year", "month", "day", "hour", "capacity"]
    return frame


def read_published_coefficients(path: Path) -> dict[str, float]:
    # XLSX is a ZIP of XML files; this avoids adding openpyxl just for two columns.
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.itertext()) for node in shared_root.findall("x:si", XML_NS)
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    published: dict[str, float] = {}
    for row in sheet.findall(".//x:row", XML_NS)[2:]:
        cells = row.findall("x:c", XML_NS)
        if len(cells) < 2:
            continue
        country_index = int(cells[0].findtext("x:v", namespaces=XML_NS))
        coefficient = float(cells[1].findtext("x:v", namespaces=XML_NS))
        published[shared[country_index]] = coefficient
    return published


def tau(x: pd.Series, y: pd.Series) -> float:
    return float(kendalltau(x, y, nan_policy="omit").statistic)


def grouped_tau(frame: pd.DataFrame, keys: list[pd.Series | str]) -> float:
    grouped = frame.groupby(keys, sort=False)[["wind", "solar"]].mean()
    return tau(grouped["wind"], grouped["solar"])


def tied_pairs(values: pd.Series) -> int:
    counts = values.value_counts(dropna=False).to_numpy()
    return int(((counts * (counts - 1)) // 2).sum())


def main() -> None:
    wind_files = indexed_files("wind_cf_")
    solar_files = indexed_files("solar_cf_")
    # The archive has one mislabeled file: the case inputs request Spain (mean 0.26),
    # while the 42-country folder contains Turkmenistan (mean 0.26) and no Spain.
    if "Spain" not in solar_files and "Turkmenistan" in solar_files:
        solar_files["Spain"] = solar_files.pop("Turkmenistan")
    published = read_published_coefficients(PUBLISHED_XLSX)
    assert len(wind_files) == len(solar_files) == len(published) == 42
    assert wind_files.keys() == solar_files.keys() == published.keys()

    rows: list[dict[str, float | int | str]] = []
    for country in sorted(wind_files):
        wind = read_capacity(wind_files[country])
        solar = read_capacity(solar_files[country])
        time_columns = ["year", "month", "day", "hour"]
        if not wind[time_columns].equals(solar[time_columns]):
            raise ValueError(f"Wind/solar timestamps do not align for {country}")

        frame = wind[time_columns].copy()
        frame["wind"] = wind["capacity"]
        frame["solar"] = solar["capacity"]
        season = (frame["month"] % 12) // 3
        season_year = frame["year"] + (frame["month"] == 12).astype(int)
        daylight = frame["solar"] > 0
        total_pairs = len(frame) * (len(frame) - 1) // 2
        wind_ties = tied_pairs(frame["wind"])
        solar_ties = tied_pairs(frame["solar"])
        tau_b_bound = math.sqrt(
            min(total_pairs - wind_ties, total_pairs - solar_ties)
            / max(total_pairs - wind_ties, total_pairs - solar_ties)
        )

        rows.append(
            {
                "country": country,
                "n_hours": len(frame),
                "published_coefficient": published[country],
                "hourly_kendall_tau_b": tau(frame["wind"], frame["solar"]),
                "hourly_pearson_r": float(pearsonr(frame["wind"], frame["solar"]).statistic),
                "hourly_spearman_rho": float(
                    spearmanr(frame["wind"], frame["solar"]).statistic
                ),
                "daylight_only_tau_b": tau(
                    frame.loc[daylight, "wind"], frame.loc[daylight, "solar"]
                ),
                "daily_mean_tau_b": grouped_tau(frame, ["year", "month", "day"]),
                "monthly_mean_tau_b": grouped_tau(frame, ["year", "month"]),
                "seasonal_mean_tau_b": grouped_tau(frame, [season_year, season]),
                "annual_mean_tau_b": grouped_tau(frame, ["year"]),
                "solar_zero_fraction": float((frame["solar"] == 0).mean()),
                "tau_b_abs_upper_bound_from_ties": tau_b_bound,
            }
        )

    result = pd.DataFrame(rows)
    result["published_minus_hourly_tau_b"] = (
        result["published_coefficient"] - result["hourly_kendall_tau_b"]
    )
    result["absolute_discrepancy"] = result["published_minus_hourly_tau_b"].abs()
    result["published_quantization_k"] = (
        (result["published_coefficient"] + 5 / 6) * result["n_hours"]
    ).round().astype(int)
    result["published_exceeds_tau_b_tie_bound"] = (
        result["published_coefficient"].abs()
        > result["tau_b_abs_upper_bound_from_ties"]
    )
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    columns = [
        "published_coefficient",
        "hourly_kendall_tau_b",
        "daylight_only_tau_b",
        "daily_mean_tau_b",
        "monthly_mean_tau_b",
        "seasonal_mean_tau_b",
        "annual_mean_tau_b",
    ]
    print(f"Wrote {OUTPUT}")
    print(result[columns].agg(["min", "max", "mean"]).T.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nLargest published-vs-standard hourly discrepancies:")
    print(
        result.nlargest(5, "absolute_discrepancy").loc[
            :, ["country", "published_coefficient", "hourly_kendall_tau_b"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(
        "\nPublished values above the mathematical tau-b bound imposed by ties:",
        int(result["published_exceeds_tau_b_tie_bound"].sum()),
    )
    quantization_error = (
        (result["published_coefficient"] + 5 / 6) * result["n_hours"]
        - result["published_quantization_k"]
    ).abs().max()
    print(f"Max error in published = -5/6 + integer/n identity: {quantization_error:.3e}")


if __name__ == "__main__":
    main()
