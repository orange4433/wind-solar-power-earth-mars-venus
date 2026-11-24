"""Build comparable Akatsuki and Pioneer Venus atmospheric profiles.

Inputs are archived mission products kept in data/raw.  The script writes
tidy CSV files and two figures without modifying the downloaded products.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"

AKATSUKI_RAW = RAW / "rs_20160723_022600_udsc64_l4_ae_v10.tab"
PIONEER_RAW = RAW / "las_low_atm_state.csv"

# The Akatsuki L4 number density is molecular number density.  A Venus lower-
# atmosphere mean molar mass consistent with the source products is used here.
MOLAR_MASS_KG_MOL = 0.04345
AVOGADRO_MOL_INV = 6.02214076e23


def read_akatsuki(path: Path) -> dict[str, np.ndarray]:
    """Read the 24-column Akatsuki PDS3 L4 atmosphere table."""
    rows: list[list[str]] = []
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 24:
                raise ValueError(
                    f"{path.name}:{line_number}: expected 24 fields, got {len(fields)}"
                )
            rows.append(fields)

    # PDS columns are one-based.  Columns 11 and 17 are the official medium
    # (170 K upper-boundary) pressure and temperature solution.
    altitude = np.array([float(row[7]) for row in rows])
    pressure_pa = np.array([float(row[10]) for row in rows])
    temperature_k = np.array([float(row[16]) for row in rows])
    number_density = np.array([float(row[20]) for row in rows])
    density = number_density * MOLAR_MASS_KG_MOL / AVOGADRO_MOL_INV

    valid = (
        (altitude >= 0)
        & (pressure_pa > 0)
        & (temperature_k > 0)
        & (number_density > 1e10)
    )
    return {
        "utc": np.array([row[1] for row in rows], dtype=object)[valid],
        "altitude_km": altitude[valid],
        "pressure_hpa": pressure_pa[valid] / 100.0,
        "temperature_k": temperature_k[valid],
        "density_kg_m3": density[valid],
        "number_density_m3": number_density[valid],
        "latitude_deg_n": np.array([float(row[4]) for row in rows])[valid],
        "longitude_deg_e": np.array([float(row[5]) for row in rows])[valid],
        "solar_zenith_angle_deg": np.array([float(row[22]) for row in rows])[valid],
        "local_solar_time_h": np.array([float(row[23]) for row in rows])[valid],
    }


def read_pioneer(path: Path) -> dict[str, np.ndarray]:
    """Read the Pioneer Venus Large Probe low-atmosphere CSV."""
    rows: list[list[str]] = []
    with path.open("r", encoding="ascii", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) != 7:
                raise ValueError(
                    f"{path.name}:{line_number}: expected 7 fields, got {len(row)}"
                )
            rows.append(row)

    altitude = np.array([float(row[2]) for row in rows])
    pressure_bar = np.array([float(row[3]) for row in rows])
    temperature_k = np.array([float(row[4]) for row in rows])
    density = np.array([float(row[5]) for row in rows])
    valid = (
        (altitude >= 0)
        & (pressure_bar > 0)
        & (temperature_k > 0)
        & (density > 0)
    )
    return {
        "grt_hr_min": np.array([row[0] for row in rows], dtype=object)[valid],
        "grt_seconds": np.array([int(row[1]) for row in rows])[valid],
        "altitude_km": altitude[valid],
        "pressure_hpa": pressure_bar[valid] * 1000.0,
        "temperature_k": temperature_k[valid],
        "density_kg_m3": density[valid],
        "compressibility_factor": np.array([float(row[6]) for row in rows])[valid],
    }


def sort_profile(profile: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = np.argsort(profile["altitude_km"])
    return {key: values[order] for key, values in profile.items()}


def interpolate(profile: dict[str, np.ndarray], field: str, altitude_km: float) -> float:
    altitude = profile["altitude_km"]
    if not altitude.min() <= altitude_km <= altitude.max():
        return float("nan")
    return float(np.interp(altitude_km, altitude, profile[field]))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_processed(ak: dict[str, np.ndarray], pv: dict[str, np.ndarray]) -> None:
    ak_fields = [
        "utc",
        "altitude_km",
        "pressure_hpa",
        "temperature_k",
        "density_kg_m3",
        "number_density_m-3",
        "latitude_deg_n",
        "longitude_deg_e",
        "solar_zenith_angle_deg",
        "local_solar_time_h",
    ]
    ak_rows = []
    for i in range(len(ak["altitude_km"])):
        ak_rows.append(
            {
                "utc": ak["utc"][i],
                "altitude_km": f"{ak['altitude_km'][i]:.3f}",
                "pressure_hpa": f"{ak['pressure_hpa'][i]:.7g}",
                "temperature_k": f"{ak['temperature_k'][i]:.2f}",
                "density_kg_m3": f"{ak['density_kg_m3'][i]:.7g}",
                "number_density_m-3": f"{ak['number_density_m3'][i]:.7e}",
                "latitude_deg_n": f"{ak['latitude_deg_n'][i]:.2f}",
                "longitude_deg_e": f"{ak['longitude_deg_e'][i]:.2f}",
                "solar_zenith_angle_deg": f"{ak['solar_zenith_angle_deg'][i]:.3f}",
                "local_solar_time_h": f"{ak['local_solar_time_h'][i]:.3f}",
            }
        )
    write_csv(PROCESSED / "akatsuki_20160723_egress_profile.csv", ak_fields, ak_rows)

    pv_fields = [
        "grt_hr_min",
        "grt_seconds",
        "altitude_km",
        "pressure_hpa",
        "temperature_k",
        "density_kg_m3",
        "compressibility_factor",
        "latitude_deg_n",
        "longitude_deg_e",
        "solar_zenith_angle_deg",
        "local_solar_time_h",
    ]
    pv_rows = []
    for i in range(len(pv["altitude_km"])):
        pv_rows.append(
            {
                "grt_hr_min": pv["grt_hr_min"][i],
                "grt_seconds": int(pv["grt_seconds"][i]),
                "altitude_km": f"{pv['altitude_km'][i]:.3f}",
                "pressure_hpa": f"{pv['pressure_hpa'][i]:.7g}",
                "temperature_k": f"{pv['temperature_k'][i]:.2f}",
                "density_kg_m3": f"{pv['density_kg_m3'][i]:.7g}",
                "compressibility_factor": f"{pv['compressibility_factor'][i]:.4f}",
                "latitude_deg_n": "4.4",
                "longitude_deg_e": "304.0",
                "solar_zenith_angle_deg": "65.7",
                "local_solar_time_h": f"{7 + 38 / 60:.4f}",
            }
        )
    write_csv(PROCESSED / "pioneer_large_probe_profile.csv", pv_fields, pv_rows)

    target_altitude = 54.0
    comparison_fields = [
        "mission_profile",
        "altitude_km",
        "pressure_hpa",
        "temperature_k",
        "density_kg_m3",
        "latitude_deg_n",
        "local_solar_time_h",
        "solar_zenith_angle_deg",
        "sampling_note",
    ]
    comparison_rows = [
        {
            "mission_profile": "Akatsuki RS 2016-07-23 egress",
            "altitude_km": "54.0",
            "pressure_hpa": f"{interpolate(ak, 'pressure_hpa', target_altitude):.6g}",
            "temperature_k": f"{interpolate(ak, 'temperature_k', target_altitude):.3f}",
            "density_kg_m3": f"{interpolate(ak, 'density_kg_m3', target_altitude):.6g}",
            "latitude_deg_n": f"{interpolate(ak, 'latitude_deg_n', target_altitude):.3f}",
            "local_solar_time_h": f"{interpolate(ak, 'local_solar_time_h', target_altitude):.3f}",
            "solar_zenith_angle_deg": f"{interpolate(ak, 'solar_zenith_angle_deg', target_altitude):.3f}",
            "sampling_note": "linear interpolation to 54 km",
        },
        {
            "mission_profile": "Pioneer Venus Large Probe 1978-12-09",
            "altitude_km": "54.0",
            "pressure_hpa": f"{interpolate(pv, 'pressure_hpa', target_altitude):.6g}",
            "temperature_k": f"{interpolate(pv, 'temperature_k', target_altitude):.3f}",
            "density_kg_m3": f"{interpolate(pv, 'density_kg_m3', target_altitude):.6g}",
            "latitude_deg_n": "4.4",
            "local_solar_time_h": f"{7 + 38 / 60:.3f}",
            "solar_zenith_angle_deg": "65.7",
            "sampling_note": "linear interpolation to 54 km",
        },
    ]
    write_csv(PROCESSED / "comparison_at_54km.csv", comparison_fields, comparison_rows)


def add_profile_lines(axes: np.ndarray, ak: dict[str, np.ndarray], pv: dict[str, np.ndarray]) -> None:
    colors = {"ak": "#0072B2", "pv": "#D55E00"}
    series = [
        ("pressure_hpa", "气压 / Pressure (hPa)", True),
        ("density_kg_m3", "密度 / Density (kg m$^{-3}$)", True),
        ("temperature_k", "温度 / Temperature (K)", False),
    ]
    for ax, (field, xlabel, log_x) in zip(axes, series):
        ax.plot(
            ak[field], ak["altitude_km"], color=colors["ak"], lw=2.2,
            label="Akatsuki RS（2016-07-23，掩出）",
        )
        ax.plot(
            pv[field], pv["altitude_km"], color=colors["pv"], lw=2.2,
            label="Pioneer Venus Large Probe（1978-12-09）",
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color="#c7c7c7", alpha=0.7, lw=0.7)
        ax.grid(True, which="minor", color="#dedede", alpha=0.45, lw=0.5)
        ax.axhspan(51, 55, color="#E69F00", alpha=0.10, zorder=0)
        ax.axhline(54, color="#6b6b6b", ls="--", lw=0.9, alpha=0.8)
        for profile, color in ((ak, colors["ak"]), (pv, colors["pv"])):
            ax.scatter(
                interpolate(profile, field, 54.0), 54.0, s=30,
                facecolor=color, edgecolor="white", linewidth=0.7, zorder=4,
            )


def plot_profiles(
    ak: dict[str, np.ndarray], pv: dict[str, np.ndarray], y_limits: tuple[float, float],
    output_name: str, title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 7.2), sharey=True)
    add_profile_lines(axes, ak, pv)
    axes[0].set_ylabel("距金星表面高度 / Altitude (km)")
    axes[0].set_ylim(*y_limits)
    # Autoscaling otherwise includes the hidden portions of each full profile,
    # making the 45–70 km zoom needlessly wide on the x axes.
    fields = ("pressure_hpa", "density_kg_m3", "temperature_k")
    for ax, field in zip(axes, fields):
        visible = []
        for profile in (ak, pv):
            mask = (
                (profile["altitude_km"] >= y_limits[0])
                & (profile["altitude_km"] <= y_limits[1])
            )
            visible.extend(profile[field][mask])
        x_min, x_max = float(np.min(visible)), float(np.max(visible))
        if field in ("pressure_hpa", "density_kg_m3"):
            ax.set_xlim(x_min / 1.35, x_max * 1.35)
        else:
            padding = max(5.0, 0.06 * (x_max - x_min))
            ax.set_xlim(x_min - padding, x_max + padding)
    axes[0].legend(loc="best", frameon=True, framealpha=0.94, fontsize=9)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.015,
        "橙色浅带：51–55 km；虚线和圆点：54 km。Akatsuki 密度由数密度按 43.45 g mol$^{-1}$ 换算。",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.045, 1, 0.95), w_pad=1.2)
    fig.savefig(FIGURES / output_name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for required in (AKATSUKI_RAW, PIONEER_RAW):
        if not required.exists():
            raise FileNotFoundError(f"Missing raw input: {required}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
            ],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    ak = sort_profile(read_akatsuki(AKATSUKI_RAW))
    pv = sort_profile(read_pioneer(PIONEER_RAW))
    write_processed(ak, pv)
    plot_profiles(
        ak, pv, (0, 96), "akatsuki_pioneer_profiles_full.png",
        "金星大气垂直廓线：Akatsuki 与 Pioneer Venus",
    )
    plot_profiles(
        ak, pv, (45, 70), "akatsuki_pioneer_profiles_45_70km.png",
        "金星云层高度廓线（45–70 km）：Akatsuki 与 Pioneer Venus",
    )

    print(f"Akatsuki: {len(ak['altitude_km'])} rows, "
          f"{ak['altitude_km'].min():.3f}–{ak['altitude_km'].max():.3f} km")
    print(f"Pioneer:  {len(pv['altitude_km'])} rows, "
          f"{pv['altitude_km'].min():.3f}–{pv['altitude_km'].max():.3f} km")
    print("Wrote processed CSV files and two PNG figures.")


if __name__ == "__main__":
    main()
