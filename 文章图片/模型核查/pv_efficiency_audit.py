"""Audit spacecraft photovoltaic parameters and legacy efficiency estimates using ZTJ references."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOLAR = PROJECT/'金星'/'venus_solar_power'

# ===== Device reference: Rocket Lab / SolAero ZTJ datasheet =====
ETA_REF = 0.295
TEMP_REF_K = 301.15  # Space reference conditions at 28 C.
AM0_W_M2 = 1353.0
PMP_TEMP_COEFF_UW_CM2_K = -85.7
# Relative power-temperature coefficient: absolute coefficient / reference power density.
GAMMA_PER_K = PMP_TEMP_COEFF_UW_CM2_K / (ETA_REF*135300.0)
SYSTEM_FACTOR = 0.90  # Scenario assumption for the system factor.
PANEL_DELTA_T_K = 15.0  # Assumed panel temperature offset; a panel thermal model is still required.
INSIGHT_ACTIVE_M2 = 5.16
INSIGHT_ARRAY_M2 = 2*3.49  # Romero-Guzman et al. 2023: 3.49 square metres per deployed array.
INSIGHT_OUTLINE_M2 = 2*np.pi*(2.15/2)**2  # Legacy circular outline approximation, retained for area-sensitivity checks.


def efficiency(cell_temperature_k):
    """First-order scenario for moderate temperatures, without clipping to record efficiencies."""
    return ETA_REF*(1+GAMMA_PER_K*(np.asarray(cell_temperature_k)-TEMP_REF_K))


def venus_reference():
    flux = pd.read_csv(SOLAR/'venus_solar_power_from_lsfr.csv')
    row = flux.loc[flux.channel.eq('combined')].iloc[0]
    air, g = float(row.vega_layer_air_temperature_k), float(row.down_flux_w_m2)
    cell = air+PANEL_DELTA_T_K
    eta = float(efficiency(cell))
    return dict(altitude_km=54, irradiance_w_m2=g, air_temperature_k=air,
                assumed_cell_temperature_k=cell, cell_efficiency=eta,
                system_factor=SYSTEM_FACTOR, power_w_per_active_m2=g*eta*SYSTEM_FACTOR,
                five_active_m2_power_w=5*g*eta*SYSTEM_FACTOR,
                spectral_correction='unresolved; k_spectral=1 is a baseline assumption')


def main():
    assert np.isclose(float(efficiency(TEMP_REF_K)), ETA_REF)
    assert -.003 < GAMMA_PER_K < -.002
    assert efficiency(325) < ETA_REF < efficiency(233)
    profile = pd.read_csv(SOLAR/'mars_validation'/'perseverance_sol0016_power_profile.csv')
    t, g = profile.SCLK.to_numpy(), profile.G_down_W_m2.to_numpy()
    assert np.all(np.diff(t)>0)
    eta = efficiency(profile.air_temp_K.to_numpy()+PANEL_DELTA_T_K)
    h = np.trapezoid(g, t)/3600
    daylight = g >= 20
    weighted = np.trapezoid(g*eta, t)/np.trapezoid(g, t)
    # Cross-site irradiance-proxy sensitivity, without fitting or independent efficiency measurement.
    inferred = []
    for energy_name, energy in [('legacy_telemetry_reconstruction', 4312.),
                                ('published_InSight_Sol4_capability', 4750.)]:
        for spectral_factor in [1., 1.15, 1.225, 1.30]:
            for area_name, area in [('active_cells', INSIGHT_ACTIVE_M2),
                                    ('published_array_surface', INSIGHT_ARRAY_M2),
                                    ('legacy_circle_outline', INSIGHT_OUTLINE_M2)]:
                inferred.append(dict(output_basis=energy_name, energy_wh=energy,
                                     irradiance_proxy='Perseverance MEDA Sol 16; different site/time',
                                     spectral_factor_assumed=spectral_factor, area_basis=area_name,
                                     area_m2=area, incident_wh_m2=h*spectral_factor,
                                     effective_ratio=energy/(area*h*spectral_factor)))
    pd.DataFrame(inferred).to_csv(HERE/'火星_效率口径敏感性.csv', index=False, encoding='utf-8-sig')
    # Reproduce the legacy blackbody bandwidth ratio as a proxy only.
    wavelength = np.linspace(190e-9, 1800e-9, 40000)
    spectrum = 1/(wavelength**5*np.expm1(6.62607015e-34*299792458/(wavelength*1.380649e-23*5772)))
    band = wavelength <= 1100e-9
    ratio = np.trapezoid(spectrum,wavelength)/np.trapezoid(spectrum[band],wavelength[band])
    summary = dict(
        device='ZTJ InGaP/InGaAs/Ge triple-junction', eta_ref=ETA_REF,
        temp_ref_k=TEMP_REF_K, gamma_per_k=GAMMA_PER_K,
        datasheet='https://rocketlabcorp.com/assets/Uploads/RL-SolAero-Data-Sheet-ZTJ.pdf',
        meda_band_insolation_wh_m2=float(h), legacy_blackbody_band_factor=float(ratio),
        legacy_corrected_band_insolation_wh_m2=float(h*ratio),
        legacy_4312wh_effective_ratio=float(4312/(INSIGHT_ACTIVE_M2*h*ratio)),
        paper_4750wh_effective_ratio_same_proxy=float(4750/(INSIGHT_ACTIVE_M2*h*ratio)),
        paper_4750wh_outline_ratio_same_proxy=float(4750/(INSIGHT_OUTLINE_M2*h*ratio)),
        paper_4750wh_array_surface_ratio_same_proxy=float(4750/(INSIGHT_ARRAY_M2*h*ratio)),
        legacy_vs_paper_energy_bias_pct=100*(4312/4750-1),
        meda_temperature_model_cell_eta_daylight_min=float(eta[daylight].min()),
        meda_temperature_model_cell_eta_daylight_max=float(eta[daylight].max()),
        meda_temperature_model_cell_eta_irradiance_weighted=float(weighted),
        meda_forward_active_area_bandlimited_energy_wh=float(INSIGHT_ACTIVE_M2*SYSTEM_FACTOR*np.trapezoid(g*eta,t)/3600),
        venus=venus_reference(),
        interpretation='Cross-site/time and incomplete-spectrum proxies; not an independent InSight cell-efficiency measurement.')
    (HERE/'效率核查结果.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Scenario ranges describe assumptions rather than confidence limits or survivability.
    v, rows = venus_reference(), []
    for delta in [0, 15, 30]:
        for spectral in [.8, 1., 1.2]:
            cell_eta = float(efficiency(v['air_temperature_k']+delta))
            for fill in [.71, .9, 1.]:
                rows.append(dict(delta_t_k=delta, spectral_multiplier_assumed=spectral,
                                 cell_fill_fraction_assumed=fill, panel_gross_area_m2=5,
                                 cell_efficiency_temperature_only=cell_eta,
                                 snapshot_power_w=5*fill*v['irradiance_w_m2']*cell_eta*SYSTEM_FACTOR*spectral))
    pd.DataFrame(rows).to_csv(HERE/'金星54km_假设敏感性.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
