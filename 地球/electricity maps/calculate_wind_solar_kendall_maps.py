from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import kendalltau


HERE = Path(__file__).resolve().parent
YEARS = (2022, 2025)
MIN_PAIRED_HOURS = 720
MAP_PATH = HERE / "ne_110m_admin_0_countries.geojson"
# Natural Earth 1:110m, public-domain boundaries:
# https://github.com/nvkelso/natural-earth-vector/tree/master/geojson
LONG_CSV = HERE / "wind_solar_kendall_tau_by_country_2022_2025.csv"
CHANGE_CSV = HERE / "wind_solar_kendall_tau_change_2022_to_2025.csv"
MAP_PNG = HERE / "wind_solar_kendall_world_maps_2022_2025.png"
MAP_SVG = HERE / "wind_solar_kendall_world_maps_2022_2025.svg"
NO_DATA_COLOR = "#242424"
PARTIAL_DATA_COLOR = "#e0a137"
# Tiny states/territories omitted by Natural Earth 1:110m are shown as dots.
SMALL_PLACE_POINTS = {
    "AW": (-70.0, 12.5), "BH": (50.6, 26.1), "CV": (-23.6, 15.1),
    "CW": (-69.0, 12.2), "FO": (-6.9, 62.0), "FM": (158.2, 6.9),
    "GP": (-61.6, 16.3), "MQ": (-61.0, 14.7), "MU": (57.6, -20.3),
    "RE": (55.5, -21.1), "SC": (55.5, -4.6),
}


def calculate(year: int, min_hours: int) -> pd.DataFrame:
    path = HERE / f"global_{year}_hourly_wind_solar_by_country.csv"
    data = pd.read_csv(path)
    total_hours = len(data)
    rows = []
    for wind_col in (c for c in data.columns if c.endswith("_wind_mw")):
        solar_col = wind_col.removesuffix("_wind_mw") + "_solar_mw"
        match = re.fullmatch(r"(.+) \[([^]]+)]_wind_mw", wind_col)
        if not match:
            raise ValueError(f"Unexpected column: {wind_col}")
        name, code = match.groups()
        pair = data[[wind_col, solar_col]].dropna()
        wind_available_hours = int(data[wind_col].notna().sum())
        solar_available_hours = int(data[solar_col].notna().sum())
        wind_unique, solar_unique = pair[wind_col].nunique(), pair[solar_col].nunique()
        if len(pair) < min_hours:
            tau = p_value = np.nan
            status = "paired hours < threshold"
        elif wind_unique < 2 or solar_unique < 2:
            tau = p_value = np.nan
            status = "constant wind or solar series"
        else:
            # Tau-b is the slide's Kendall coefficient with tie correction; hourly solar has many night-time ties.
            result = kendalltau(pair[wind_col], pair[solar_col], variant="b")
            tau, p_value, status = float(result.statistic), float(result.pvalue), "valid"
        interpretation = "no result" if not np.isfinite(tau) else (
            "complementary" if tau < 0 else "synchronous" if tau > 0 else "neutral"
        )
        rows.append({
            "year": year,
            "country_name": name,
            "country_code": code,
            "kendall_tau_b": tau,
            "complementarity_score_minus_tau": -tau if np.isfinite(tau) else np.nan,
            "p_value": p_value,
            "interpretation": interpretation,
            "status": status,
            "paired_hours": len(pair),
            "wind_available_hours": wind_available_hours,
            "solar_available_hours": solar_available_hours,
            "coverage_percent": 100 * len(pair) / total_hours,
            "wind_zero_percent": 100 * pair[wind_col].eq(0).mean() if len(pair) else np.nan,
            "solar_zero_percent": 100 * pair[solar_col].eq(0).mean() if len(pair) else np.nan,
            "wind_total_mwh": pair[wind_col].sum(min_count=1),
            "solar_total_mwh": pair[solar_col].sum(min_count=1),
        })
    return pd.DataFrame(rows)


def country_code(feature: dict) -> str:
    properties = feature["properties"]
    code = properties.get("ISO_A2_EH")
    if code and code != "-99":
        return code
    # Natural Earth models Somaliland separately; Electricity Maps reports Somalia nationally.
    return {"Somaliland": "SO"}.get(properties.get("ADMIN"), "")


def polygons(geometry: dict):
    coordinates = geometry["coordinates"]
    if geometry["type"] == "Polygon":
        coordinates = [coordinates]
    for polygon in coordinates:
        yield polygon[0]  # Exterior ring is enough at 1:110m plotting scale.


def draw_map(ax, geojson: dict, results: pd.DataFrame, year: int, cmap, norm) -> None:
    year_results = results.loc[results["year"] == year]
    valid = year_results.loc[year_results["kendall_tau_b"].notna()]
    rows = year_results.set_index("country_code").to_dict("index")

    def map_color(code: str):
        row = rows.get(code)
        if row is None or (row["wind_available_hours"] == 0 and row["solar_available_hours"] == 0):
            return NO_DATA_COLOR
        if not np.isfinite(row["kendall_tau_b"]):
            return PARTIAL_DATA_COLOR
        return cmap(norm(row["kendall_tau_b"]))

    for feature in geojson["features"]:
        if feature["properties"].get("ADMIN") == "Antarctica":
            continue
        color = map_color(country_code(feature))
        for ring in polygons(feature["geometry"]):
            xy = np.asarray(ring)
            ax.fill(xy[:, 0], xy[:, 1], facecolor=color, edgecolor="white", linewidth=0.28)
    for code, (longitude, latitude) in SMALL_PLACE_POINTS.items():
        if code in rows:
            ax.scatter(longitude, latitude, s=17, color=map_color(code),
                       edgecolor="white", linewidth=0.35, zorder=3)
    negative = int((valid["kendall_tau_b"] < 0).sum())
    positive = int((valid["kendall_tau_b"] > 0).sum())
    no_data = int(((year_results["wind_available_hours"] == 0) &
                   (year_results["solar_available_hours"] == 0)).sum())
    partial = len(year_results) - len(valid) - no_data
    ax.set_title(f"{year}年", fontsize=18, fontweight="bold", pad=10)
    ax.text(0.5, -0.04, f"有效 {len(valid)}（互补 {negative}，同步 {positive}）  ·  黄色部分数据 {partial}  ·  黑色风光均缺 {no_data}",
            transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#4d5965")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_map(results: pd.DataFrame) -> None:
    geojson = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    cmap = LinearSegmentedColormap.from_list("complementarity", ["#2166ac", "#f7f7f7", "#b2182b"])
    norm = Normalize(-0.6, 0.6)
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 1, figsize=(15, 9.2), constrained_layout=True)
    fig.suptitle("Electricity Maps 各国风光互补相关系数（小时级 Kendall τ-b）", fontsize=23, fontweight="bold")
    for ax, year in zip(axes, YEARS):
        draw_map(ax, geojson, results, year, cmap, norm)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, orientation="horizontal",
                            fraction=0.035, pad=0.045, aspect=45)
    colorbar.set_label("Kendall τ-b    ← 强互补（负）          无相关（0）          同步（正）→    ·    黄色＝部分数据不足    黑色＝风光均缺", fontsize=12)
    colorbar.set_ticks(np.arange(-0.6, 0.61, 0.2))
    fig.savefig(MAP_PNG, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(MAP_SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate hourly country wind-solar Kendall tau-b and draw maps.")
    parser.add_argument("--min-hours", type=int, default=MIN_PAIRED_HOURS)
    args = parser.parse_args()
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Missing world boundaries: {MAP_PATH}")
    results = pd.concat([calculate(year, args.min_hours) for year in YEARS], ignore_index=True)
    results.to_csv(LONG_CSV, index=False, encoding="utf-8-sig", float_format="%.8f")
    change = results.pivot(index=["country_code", "country_name"], columns="year", values="kendall_tau_b").reset_index()
    change.columns.name = None
    change = change.rename(columns={2022: "kendall_tau_b_2022", 2025: "kendall_tau_b_2025"})
    change["delta_2025_minus_2022"] = change["kendall_tau_b_2025"] - change["kendall_tau_b_2022"]
    change.to_csv(CHANGE_CSV, index=False, encoding="utf-8-sig", float_format="%.8f")
    save_map(results)

    valid = results.dropna(subset=["kendall_tau_b"])
    if args.min_hours == MIN_PAIRED_HOURS:
        assert valid["year"].value_counts().to_dict() == {2025: 101, 2022: 88}
    assert valid["kendall_tau_b"].between(-1, 1).all()
    print(f"Saved {LONG_CSV}")
    print(f"Saved {CHANGE_CSV}")
    print(f"Saved {MAP_PNG}")
    print(f"Saved {MAP_SVG}")


if __name__ == "__main__":
    main()
