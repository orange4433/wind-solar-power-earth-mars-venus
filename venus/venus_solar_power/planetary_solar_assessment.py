from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MARS_DIR = HERE / "mars_validation"
OUT = HERE / "planetary_solar_assessment.csv"
FIGURE = HERE / "planetary_solar_power_density.png"

ETA_28C = 0.295                 # SolAero ZTJ beginning-of-life AM0 efficiency
TEMP_COEFF = -0.0022            # relative Pmax / K
SYSTEM_FACTOR = 0.90
INSIGHT_ACTIVE_CELL_M2 = 5.16
INSIGHT_DISC_DIAMETER_M = 2.15
INSIGHT_APERTURE_M2 = 2 * np.pi * (INSIGHT_DISC_DIAMETER_M / 2) ** 2
INSIGHT_SOL4_PEAK_W = 710
INSIGHT_SOL4_ENERGY_WH = 4750
VENUS_TOA_W_M2 = 2605
VENUS_SURFACE_SUBSOLAR_W_M2 = 90
VENUS_LSFR_SZA_DEG = 65.7


def efficiency(cell_temperature_k: float) -> float:
    return max(0.0, ETA_28C * (1 + TEMP_COEFF * (cell_temperature_k - 301.15)))


def main() -> None:
    diagnostics = pd.read_csv(MARS_DIR / "perseverance_insight_quick_check_diagnostics.csv")
    values = dict(zip(diagnostics["metric"], diagnostics["value"]))
    mars_peak = float(values["G_down max"])
    mars_insolation = float(values["daily insolation"])
    mars_air_k = float(values["air temperature mean"])

    lsfr = pd.read_csv(
        HERE / "data" / "lsfr_solar_flux.csv",
        header=None,
        names=["time", "pressure_atm", "altitude_km", "vis_down", "vis_up", "vis_net",
               "combined_down", "combined_up", "combined_net", "narrow_down", "narrow_up", "narrow_net"],
    )
    broadband = lsfr.dropna(subset=["altitude_km", "combined_down"]).sort_values("altitude_km")
    venus_54_ref = float(np.interp(54.0, broadband["altitude_km"], broadband["combined_down"]))
    venus_54_noon = venus_54_ref * (1 / np.cos(np.radians(VENUS_LSFR_SZA_DEG))) ** 0.70

    rows = []
    for irradiance in [450, 550, 650]:
        rows.append({
            "section": "InSight peak sensitivity", "case": f"assumed clear-noon {irradiance} W/m2",
            "irradiance_w_m2": irradiance,
            "active_cell_efficiency_percent": 100 * INSIGHT_SOL4_PEAK_W / (INSIGHT_ACTIVE_CELL_M2 * irradiance),
            "aperture_efficiency_percent": 100 * INSIGHT_SOL4_PEAK_W / (INSIGHT_APERTURE_M2 * irradiance),
        })
    for spectral_factor in [1.15, 1.225, 1.30]:
        broadband_wh_m2 = mars_insolation * spectral_factor
        rows.append({
            "section": "InSight daily sensitivity", "case": f"MEDA 190-1100 nm x {spectral_factor}",
            "irradiance_wh_m2_sol": broadband_wh_m2,
            "active_cell_efficiency_percent": 100 * INSIGHT_SOL4_ENERGY_WH / (INSIGHT_ACTIVE_CELL_M2 * broadband_wh_m2),
            "aperture_efficiency_percent": 100 * INSIGHT_SOL4_ENERGY_WH / (INSIGHT_APERTURE_M2 * broadband_wh_m2),
        })

    cases = [
        ("Venus surface", VENUS_SURFACE_SUBSOLAR_W_M2, 735.0, "Venera 11/12 subsolar measurement"),
        ("Venus 54 km", venus_54_noon, 310.0 + 15.0, "Pioneer Venus LSFR, SZA-scaled"),
        ("Mars surface", mars_peak, mars_air_k + 15.0, "Perseverance MEDA sol 16 peak"),
    ]
    for name, irradiance, cell_k, basis in cases:
        eta = efficiency(cell_k)
        rows.append({
            "section": "Planet surface/altitude comparison", "case": name,
            "irradiance_w_m2": irradiance, "cell_temperature_k": cell_k,
            "cell_efficiency_percent": 100 * eta,
            "electric_power_density_w_m2": irradiance * eta * SYSTEM_FACTOR,
            "effective_transmission_vs_venus_toa": irradiance / VENUS_TOA_W_M2 if name.startswith("Venus") else np.nan,
            "basis": basis,
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")

    comparison = result[result["section"] == "Planet surface/altitude comparison"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    colors = ["#b85c38", "#d9a441", "#8b4a3c"]
    axes[0].bar(comparison["case"], comparison["irradiance_w_m2"], color=colors)
    axes[0].set_ylabel("Incident solar irradiance (W/m²)")
    axes[0].set_title("Observed / calibrated noon-scale sunlight")
    axes[1].bar(comparison["case"], comparison["electric_power_density_w_m2"], color=colors)
    axes[1].set_ylabel("Electrical output (W/m²)")
    axes[1].set_title("Same ZTJ cell model, including temperature")
    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Small checks: Venus is brighter at the top of atmosphere, but this calibrated
    # surface-PV case must be far below Mars; the temperate cloud layer should recover.
    power = comparison.set_index("case")["electric_power_density_w_m2"]
    assert power["Venus surface"] < power["Mars surface"] / 20
    assert power["Venus 54 km"] > power["Mars surface"]
    print(result.to_string(index=False))
    print(f"Saved {OUT}\nSaved {FIGURE}")


if __name__ == "__main__":
    main()
