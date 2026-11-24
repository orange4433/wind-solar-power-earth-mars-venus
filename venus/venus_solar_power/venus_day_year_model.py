from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

VENUS_RADIUS_KM = 6051.8
BALLOON_ALTITUDE_KM = 54.0
VENUS_YEAR_EARTH_D = 224.701
VENUS_SOLAR_DAY_EARTH_D = 116.75
VENUS_SIDEREAL_ROTATION_EARTH_D = -243.018
VENUS_OBLIQUITY_DEG = 2.64
VENUS_ECCENTRICITY = 0.0068
VEGA_WIND_MS = 65.0
LSFR_LOCAL_TIME_H = 7.633
LSFR_SZA_DEG = 65.7
ETA_REF = 0.29
TEMP_COEFF_PER_K = -0.0025
PANEL_DELTA_T_K = 15.0
SYSTEM_FACTOR = 0.90
STORAGE_EFF = 0.85
PANEL_AREA_M2 = 5.0
ANGULAR_EXPONENT = 0.70

LSFR_COLUMNS = [
    "grt",
    "pressure_atm",
    "altitude_km",
    "flux_vis_down_w_m2",
    "flux_vis_up_w_m2",
    "flux_vis_net_w_m2",
    "flux_combined_down_w_m2",
    "flux_combined_up_w_m2",
    "flux_combined_net_w_m2",
    "flux_narrowband_down_w_m2",
    "flux_narrowband_up_w_m2",
    "flux_narrowband_net_w_m2",
]


def load_lsfr() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "lsfr_solar_flux.csv", header=None, names=LSFR_COLUMNS)
    for col in LSFR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_vega_temperature() -> float:
    candidates = [
        ROOT.parent / "vega_wind_power" / "vg1bl_rdr_with_time.csv",
        ROOT.parent / "vega_wind_power" / "vg2bl_rdr_with_time.csv",
    ]
    temps = []
    for path in candidates:
        df = pd.read_csv(path)
        valid = df[df["radius_km"] > 0]
        temps.append(valid["atmospheric_temperature_k"].median())
    return float(np.median(temps))


def interpolate_flux_54(lsfr: pd.DataFrame, column: str = "flux_combined_down_w_m2") -> float:
    valid = lsfr.dropna(subset=["altitude_km", column]).sort_values("altitude_km")
    return float(np.interp(BALLOON_ALTITUDE_KM, valid["altitude_km"], valid[column]))


def cell_efficiency(ambient_temp_k: float, eta_ref: float = ETA_REF) -> float:
    cell_temp_k = ambient_temp_k + PANEL_DELTA_T_K
    return max(0.0, eta_ref * (1.0 + TEMP_COEFF_PER_K * (cell_temp_k - 298.15)))


def balloon_solar_cycle_earth_d(wind_ms: float = VEGA_WIND_MS, latitude_deg: float = 0.0) -> float:
    radius_m = (VENUS_RADIUS_KM + BALLOON_ALTITUDE_KM) * 1000.0
    circumference_m = 2.0 * math.pi * radius_m * math.cos(math.radians(latitude_deg))
    circ_earth_d = circumference_m / wind_ms / 86400.0
    return 1.0 / (1.0 / circ_earth_d + 1.0 / VENUS_SOLAR_DAY_EARTH_D)


def solar_declination_rad(t_earth_d: np.ndarray) -> np.ndarray:
    return np.radians(VENUS_OBLIQUITY_DEG) * np.sin(2.0 * np.pi * t_earth_d / VENUS_YEAR_EARTH_D)


def orbit_distance_factor(t_earth_d: np.ndarray) -> np.ndarray:
    mean_anomaly = 2.0 * np.pi * t_earth_d / VENUS_YEAR_EARTH_D
    distance_au_relative = (1.0 - VENUS_ECCENTRICITY * np.cos(mean_anomaly))
    return 1.0 / distance_au_relative**2


def cos_sza_from_local_hour(local_hour: np.ndarray, latitude_deg: float, t_earth_d: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude_deg)
    dec = solar_declination_rad(t_earth_d)
    hour_angle = np.radians(15.0 * (local_hour - 12.0))
    mu = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    return np.clip(mu, 0.0, None)


def flux_from_mu(mu: np.ndarray, ref_flux_w_m2: float, alpha: float = ANGULAR_EXPONENT) -> np.ndarray:
    mu_ref = max(math.cos(math.radians(LSFR_SZA_DEG)), 1e-6)
    flux = ref_flux_w_m2 * (np.clip(mu, 0.0, None) / mu_ref) ** alpha
    return np.where(mu > 0, flux, 0.0)


def power_from_flux(flux_w_m2: np.ndarray, ambient_temp_k: float, area_m2: float = PANEL_AREA_M2) -> np.ndarray:
    return flux_w_m2 * cell_efficiency(ambient_temp_k) * SYSTEM_FACTOR * area_m2


def day_profile(ref_flux: float, ambient_temp_k: float, latitude_deg: float = 0.0) -> pd.DataFrame:
    local_hour = np.linspace(0, 24, 24 * 60 + 1)
    t_surface_d = local_hour / 24.0 * VENUS_SOLAR_DAY_EARTH_D
    t_balloon_d = local_hour / 24.0 * balloon_solar_cycle_earth_d()
    mu_surface = cos_sza_from_local_hour(local_hour, latitude_deg, t_surface_d)
    mu_balloon = cos_sza_from_local_hour(local_hour, latitude_deg, t_balloon_d)
    surface_flux = flux_from_mu(mu_surface, ref_flux) * orbit_distance_factor(t_surface_d)
    balloon_flux = flux_from_mu(mu_balloon, ref_flux) * orbit_distance_factor(t_balloon_d)
    return pd.DataFrame(
        {
            "venus_local_hour": local_hour,
            "surface_elapsed_earth_d": t_surface_d,
            "balloon_elapsed_earth_d": t_balloon_d,
            "surface_flux_w_m2": surface_flux,
            "balloon_flux_w_m2": balloon_flux,
            "surface_power_w": power_from_flux(surface_flux, ambient_temp_k),
            "balloon_power_w": power_from_flux(balloon_flux, ambient_temp_k),
        }
    )


def year_profile(ref_flux: float, ambient_temp_k: float, latitude_deg: float = 0.0) -> pd.DataFrame:
    t = np.linspace(0, VENUS_YEAR_EARTH_D, int(VENUS_YEAR_EARTH_D * 24) + 1)
    surface_hour = (t / VENUS_SOLAR_DAY_EARTH_D * 24.0) % 24.0
    balloon_period = balloon_solar_cycle_earth_d()
    balloon_hour = (t / balloon_period * 24.0) % 24.0
    mu_surface = cos_sza_from_local_hour(surface_hour, latitude_deg, t)
    mu_balloon = cos_sza_from_local_hour(balloon_hour, latitude_deg, t)
    distance_factor = orbit_distance_factor(t)
    surface_flux = flux_from_mu(mu_surface, ref_flux) * distance_factor
    balloon_flux = flux_from_mu(mu_balloon, ref_flux) * distance_factor
    return pd.DataFrame(
        {
            "earth_day": t,
            "venus_year_fraction": t / VENUS_YEAR_EARTH_D,
            "surface_venus_local_hour": surface_hour,
            "balloon_local_hour": balloon_hour,
            "surface_power_w": power_from_flux(surface_flux, ambient_temp_k),
            "balloon_power_w": power_from_flux(balloon_flux, ambient_temp_k),
        }
    )


def integrate_energy_kwh(time_earth_d: np.ndarray, power_w: np.ndarray, storage_eff: float = 1.0) -> float:
    hours = time_earth_d * 24.0
    return float(np.trapezoid(power_w, hours) * storage_eff / 1000.0)


def make_figures(day: pd.DataFrame, year: pd.DataFrame) -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
    axes[0].plot(day["venus_local_hour"], day["surface_power_w"], color="#23777a", lw=2.2)
    axes[0].set_title("Surface-fixed local Venus solar day")
    axes[0].set_xlabel("Venus local solar hour")
    axes[0].set_ylabel("PV power, 5 m2 panel (W)")
    top0 = axes[0].secondary_xaxis(
        "top",
        functions=(
            lambda h: h / 24.0 * VENUS_SOLAR_DAY_EARTH_D,
            lambda d: d / VENUS_SOLAR_DAY_EARTH_D * 24.0,
        ),
    )
    top0.set_xlabel("Elapsed Earth days")

    balloon_period = balloon_solar_cycle_earth_d()
    axes[1].plot(day["venus_local_hour"], day["balloon_power_w"], color="#c07a2a", lw=2.2)
    axes[1].set_title("Vega-like drifting balloon daylight cycle")
    axes[1].set_xlabel("Balloon local solar hour")
    top1 = axes[1].secondary_xaxis(
        "top",
        functions=(
            lambda h: h / 24.0 * balloon_period,
            lambda d: d / balloon_period * 24.0,
        ),
    )
    top1.set_xlabel("Elapsed Earth days")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 24)
    fig.suptitle("Venus solar power over one local day: LSFR-calibrated cosine-power model")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "venus_solar_one_day_power.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.0), sharex=True)
    axes[0].plot(year["earth_day"], year["surface_power_w"], color="#23777a", lw=1.8)
    axes[0].set_title("Surface-fixed point during one Venus orbital year")
    axes[1].plot(year["earth_day"], year["balloon_power_w"], color="#c07a2a", lw=1.5)
    axes[1].set_title("Vega-like drifting balloon during one Venus orbital year")
    axes[1].set_xlabel("Earth days since model start")
    for ax in axes:
        ax.set_ylabel("PV power, 5 m2 panel (W)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, VENUS_YEAR_EARTH_D)
    top = axes[0].secondary_xaxis(
        "top",
        functions=(lambda d: d / VENUS_SOLAR_DAY_EARTH_D, lambda s: s * VENUS_SOLAR_DAY_EARTH_D),
    )
    top.set_xlabel("Surface Venus solar days")
    fig.suptitle("Venus solar power over one Venus year")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "venus_solar_one_year_power.png", bbox_inches="tight")
    plt.close(fig)


def make_notebook(ref_flux: float, ambient_temp_k: float, day: pd.DataFrame, year: pd.DataFrame) -> None:
    balloon_period = balloon_solar_cycle_earth_d()
    day_surface_kwh = integrate_energy_kwh(day["surface_elapsed_earth_d"], day["surface_power_w"], STORAGE_EFF)
    day_balloon_kwh = integrate_energy_kwh(day["balloon_elapsed_earth_d"], day["balloon_power_w"], STORAGE_EFF)
    year_surface_kwh = integrate_energy_kwh(year["earth_day"], year["surface_power_w"], STORAGE_EFF)
    year_balloon_kwh = integrate_energy_kwh(year["earth_day"], year["balloon_power_w"], STORAGE_EFF)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 金星太阳能全天/全年模型\n\n"
            "这个 notebook 用 LSFR 在 54 km 的实测太阳通量作为标定点，构造一个简化的金星日照几何模型。"
        ),
        nbf.v4.new_markdown_cell(
            "## 为什么火星的 cosine-Gaussian 不能直接搬到金星\n\n"
            "你火星 notebook 的 cosine-Gaussian 是在 InSight 已经有本地全天电流曲线时做经验拟合。"
            "它适合补缺测、平滑实测电流，但金星这里没有气球太阳能板全天实测电流。"
            "所以更合适的第一步是用太阳天顶角的余弦项建几何模型，再用 LSFR 的 54 km 实测通量标定。"
        ),
        nbf.v4.new_markdown_cell(
            f"## 常数和标定\n\n"
            f"- 金星恒星自转周期：{VENUS_SIDEREAL_ROTATION_EARTH_D} Earth days，负号表示逆行。\n"
            f"- 金星太阳日：{VENUS_SOLAR_DAY_EARTH_D} Earth days，约 117 天。\n"
            f"- 金星公转周期：{VENUS_YEAR_EARTH_D} Earth days。\n"
            f"- Vega-like 气球按 {VEGA_WIND_MS:.0f} m/s 水平风漂移时，日照循环约 {balloon_period:.2f} Earth days。\n"
            f"- LSFR 54 km combined downward flux：{ref_flux:.1f} W/m2，对应 Pioneer Venus Large Probe 当地约 7:38、SZA≈65.7°。"
        ),
        nbf.v4.new_code_cell("from venus_day_year_model import *"),
        nbf.v4.new_code_cell(
            "lsfr = load_lsfr()\n"
            "ambient_temp_k = load_vega_temperature()\n"
            "ref_flux = interpolate_flux_54(lsfr)\n"
            "day = day_profile(ref_flux, ambient_temp_k)\n"
            "year = year_profile(ref_flux, ambient_temp_k)\n"
            "print('ref_flux W/m2:', round(ref_flux, 1))\n"
            "print('eta_cell:', round(cell_efficiency(ambient_temp_k), 3))"
        ),
        nbf.v4.new_code_cell(
            "print('surface one Venus solar day kWh:', round(integrate_energy_kwh(day.surface_elapsed_earth_d, day.surface_power_w, STORAGE_EFF), 2))\n"
            "print('balloon one daylight cycle kWh:', round(integrate_energy_kwh(day.balloon_elapsed_earth_d, day.balloon_power_w, STORAGE_EFF), 2))\n"
            "print('surface one Venus year kWh:', round(integrate_energy_kwh(year.earth_day, year.surface_power_w, STORAGE_EFF), 2))\n"
            "print('balloon one Venus year kWh:', round(integrate_energy_kwh(year.earth_day, year.balloon_power_w, STORAGE_EFF), 2))"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)\n"
            "axes[0].plot(day.venus_local_hour, day.surface_power_w)\n"
            "axes[1].plot(day.venus_local_hour, day.balloon_power_w, color='tab:orange')\n"
            "axes[0].set_title('Surface-fixed local Venus solar day')\n"
            "axes[1].set_title('Vega-like drifting balloon daylight cycle')\n"
            "for ax in axes:\n"
            "    ax.set_xlabel('Local solar hour')\n"
            "    ax.set_ylabel('Power W')\n"
            "    ax.grid(True, alpha=.25)\n"
            "fig.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)\n"
            "axes[0].plot(year.earth_day, year.surface_power_w)\n"
            "axes[1].plot(year.earth_day, year.balloon_power_w, color='tab:orange')\n"
            "axes[0].set_title('Surface-fixed point')\n"
            "axes[1].set_title('Vega-like balloon')\n"
            "axes[1].set_xlabel('Earth days')\n"
            "for ax in axes:\n"
            "    ax.set_ylabel('Power W')\n"
            "    ax.grid(True, alpha=.25)\n"
            "fig.tight_layout()"
        ),
    ]
    nbf.write(nb, ROOT / "venus_day_year_solar_model.ipynb")


def make_html(ref_flux: float, ambient_temp_k: float, day: pd.DataFrame, year: pd.DataFrame) -> None:
    balloon_period = balloon_solar_cycle_earth_d()
    radius_km = VENUS_RADIUS_KM + BALLOON_ALTITUDE_KM
    circumference_km = 2.0 * math.pi * radius_km
    wind_circ_earth_d = circumference_km * 1000.0 / VEGA_WIND_MS / 86400.0
    solar_day_from_rates = 1.0 / (1.0 / VENUS_YEAR_EARTH_D + 1.0 / abs(VENUS_SIDEREAL_ROTATION_EARTH_D))
    summary = {
        "refFlux": ref_flux,
        "etaCell": cell_efficiency(ambient_temp_k),
        "venusSolarDayEarthD": VENUS_SOLAR_DAY_EARTH_D,
        "venusYearEarthD": VENUS_YEAR_EARTH_D,
        "balloonPeriodEarthD": balloon_period,
        "daySurfaceKwh": integrate_energy_kwh(day["surface_elapsed_earth_d"], day["surface_power_w"], STORAGE_EFF),
        "dayBalloonKwh": integrate_energy_kwh(day["balloon_elapsed_earth_d"], day["balloon_power_w"], STORAGE_EFF),
        "yearSurfaceKwh": integrate_energy_kwh(year["earth_day"], year["surface_power_w"], STORAGE_EFF),
        "yearBalloonKwh": integrate_energy_kwh(year["earth_day"], year["balloon_power_w"], STORAGE_EFF),
    }
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>金星太阳能日照周期模型</title>
  <style>
    :root {{ --bg:#f7f1e4; --ink:#202326; --muted:#5d6570; --line:#d5c8b3; --panel:#fffaf0; --teal:#23777a; --gold:#c07a2a; --wine:#9b4050; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    header, section {{ padding:24px clamp(18px,5vw,58px); border-bottom:1px solid var(--line); }}
    header {{ background:linear-gradient(180deg,#fff8e9,#f7f1e4); }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,4.7vw,46px); letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:24px; }}
    h3 {{ margin:0 0 8px; font-size:17px; }}
    p, li {{ color:var(--muted); }}
    .two {{ display:grid; grid-template-columns:minmax(320px,1fr) minmax(320px,1fr); gap:18px; align-items:start; }}
    .box {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; }}
    .metric {{ background:#fffdf8; border:1px solid var(--line); border-radius:8px; padding:10px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric b {{ display:block; font-size:21px; margin-top:5px; }}
    .formula {{ font-family:Consolas,"Courier New",monospace; white-space:pre-wrap; background:#fff3d6; border:1px solid #dbc28e; border-radius:8px; padding:12px; }}
    canvas {{ width:100%; display:block; background:#fffdf8; border:1px solid var(--line); border-radius:8px; }}
    #animation {{ height:390px; background:linear-gradient(180deg,#292d3a,#1d1f28); }}
    #plot {{ height:420px; }}
    .controls {{ display:grid; gap:10px; margin-top:12px; }}
    .row {{ display:grid; grid-template-columns:160px 1fr 90px; gap:10px; align-items:center; }}
    label {{ color:var(--muted); font-size:13px; }}
    input, select {{ width:100%; min-height:34px; accent-color:var(--teal); }}
    select {{ border:1px solid var(--line); border-radius:6px; padding:0 8px; background:white; font:inherit; }}
    .value {{ text-align:right; font-variant-numeric:tabular-nums; }}
    img {{ width:100%; border:1px solid var(--line); border-radius:8px; background:white; }}
    a {{ color:var(--teal); }}
    @media (max-width:980px) {{ .two,.metrics {{ grid-template-columns:1fr; }} .row {{ grid-template-columns:1fr; gap:4px; }} #animation,#plot {{ height:330px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>金星太阳能：一天、一年和气球漂移</h1>
    <p>用 Pioneer Venus LSFR 在 54 km 的实测通量标定，用太阳天顶角余弦建立日照变化。重点区别：地表固定点的一天约 116.75 个地球日，但随 Vega 风漂移的气球会在约 6-7 个地球日内经历一轮日照变化。</p>
  </header>
  <section>
    <div class="metrics">
      <div class="metric"><span>54 km LSFR combined down</span><b>{summary['refFlux']:.1f} W/m²</b></div>
      <div class="metric"><span>温度修正效率</span><b>{summary['etaCell']*100:.1f}%</b></div>
      <div class="metric"><span>金星太阳日</span><b>{summary['venusSolarDayEarthD']:.2f} d</b></div>
      <div class="metric"><span>Vega-like 气球日照周期</span><b>{summary['balloonPeriodEarthD']:.2f} d</b></div>
    </div>
  </section>
  <section class="two">
    <div>
      <h2>为什么不能直接用火星 cosine-Gaussian</h2>
      <div class="box">
        <p>火星 InSight 那个方法是在已有全天电流数据时，用 cosine-Gaussian 拟合电流曲线。金星这里没有“气球太阳能板全天实测电流”，而且云层散射很强，所以 Gaussian 项没有明确物理来源。</p>
        <div class="formula">推荐第一步:
mu = cos(SZA)
G_down(t) = G_LSFR(54 km) * [max(0, mu(t)) / mu_ref]^alpha
P(t) = A * G_down(t) * eta_cell(T) * factor</div>
      </div>
    </div>
    <div>
      <h2>金星的一天为什么难想象</h2>
      <div class="box">
        <p>金星自转比公转还慢，而且逆行。地表固定点的日出到日落约 58 个地球日，完整太阳日约 117 个地球日。可是 54 km 气球会被超旋转风带着绕行，局地太阳时变化会快得多。</p>
        <div class="formula">地表固定点:
P_solar ≈ 116.75 Earth days

Vega-like 气球:
P_circ ≈ circumference / 65 m/s
P_light ≈ 1 / (1/P_circ + 1/P_solar) ≈ {summary['balloonPeriodEarthD']:.2f} Earth days</div>
      </div>
    </div>
  </section>
  <section class="two">
    <div>
      <h2>动画：地表点 vs 漂浮气球</h2>
      <canvas id="animation" width="980" height="430"></canvas>
      <div class="controls">
        <div class="row"><label for="mode">参考系</label><select id="mode"><option value="surface">地表固定点</option><option value="balloon">Vega-like 气球</option></select><div class="value" id="periodVal"></div></div>
        <div class="row"><label for="speed">播放速度</label><input id="speed" type="range" min="0.2" max="6" step="0.2" value="1.5"><div class="value" id="speedVal"></div></div>
      </div>
    </div>
    <div>
      <h2>实时功率曲线</h2>
      <canvas id="plot" width="980" height="430"></canvas>
      <div class="controls">
        <div class="row"><label for="area">太阳能板面积</label><input id="area" type="range" min="0.5" max="30" step="0.5" value="{PANEL_AREA_M2}"><div class="value" id="areaVal"></div></div>
        <div class="row"><label for="alpha">角度指数 alpha</label><input id="alpha" type="range" min="0.4" max="1.2" step="0.05" value="{ANGULAR_EXPONENT}"><div class="value" id="alphaVal"></div></div>
      </div>
    </div>
  </section>
  <section class="two">
    <div>
      <h2>一天功率曲线</h2>
      <img src="figures/venus_solar_one_day_power.png" alt="Venus one day solar power">
    </div>
    <div>
      <h2>一年功率曲线</h2>
      <img src="figures/venus_solar_one_year_power.png" alt="Venus one year solar power">
    </div>
  </section>
  <section>
    <h2>默认 5 m² 面板的能量估计</h2>
    <div class="metrics">
      <div class="metric"><span>地表固定点一金星太阳日</span><b>{summary['daySurfaceKwh']:.1f} kWh</b></div>
      <div class="metric"><span>气球一日照周期</span><b>{summary['dayBalloonKwh']:.1f} kWh</b></div>
      <div class="metric"><span>地表固定点一金星年</span><b>{summary['yearSurfaceKwh']:.1f} kWh</b></div>
      <div class="metric"><span>气球一金星年</span><b>{summary['yearBalloonKwh']:.1f} kWh</b></div>
    </div>
  </section>
  <section class="two">
    <div class="box">
      <h2>116.75 个地球日怎么来的</h2>
      <p>“金星一天”这里指的是太阳日，也就是站在金星某个固定经度上，从一次正午到下一次正午的时间。金星自转是逆行的，所以太阳在天空中的相对角速度等于公转角速度和自转角速度的绝对值相加。</p>
      <div class="formula">T_orbit = {VENUS_YEAR_EARTH_D:.3f} Earth days
T_rotation = {abs(VENUS_SIDEREAL_ROTATION_EARTH_D):.3f} Earth days

1 / T_solar = 1 / T_orbit + 1 / T_rotation

T_solar = 1 / (1/{VENUS_YEAR_EARTH_D:.3f} + 1/{abs(VENUS_SIDEREAL_ROTATION_EARTH_D):.3f})
        ≈ {solar_day_from_rates:.2f} Earth days</div>
      <p class="note">如果它是顺行自转，就会用角速度相减；金星因为逆行，太阳日反而约 116.75 个地球日，约等于 117 天。</p>
    </div>
    <div class="box">
      <h2>随气球飘的 6.45 天怎么来的</h2>
      <p>Vega 气球不是固定在地表上方，而是被 54 km 附近的超旋转风带着绕金星跑。先算气球沿赤道绕行一圈需要多久，再把这个绕行造成的局地时间变化和金星本身的太阳日变化合并。</p>
      <div class="formula">R = R_venus + z
  = {VENUS_RADIUS_KM:.1f} km + {BALLOON_ALTITUDE_KM:.1f} km
  = {radius_km:.1f} km

C = 2 * pi * R
  ≈ {circumference_km:,.0f} km

U = {VEGA_WIND_MS:.0f} m/s
T_wind = C / U
       ≈ {wind_circ_earth_d:.2f} Earth days

1 / T_balloon = 1 / T_wind + 1 / T_solar

T_balloon = 1 / ({1.0 / wind_circ_earth_d:.4f} + {1.0 / VENUS_SOLAR_DAY_EARTH_D:.4f})
          ≈ {summary['balloonPeriodEarthD']:.2f} Earth days</div>
      <p class="note">这个 6.45 天是“Vega-like、赤道附近、65 m/s 风速、风向与日照相对运动相加”的一阶估计。纬度、风速、风向变了，这个数会跟着变；但它说明了关键点：气球看到的昼夜循环主要由快速超旋转风决定，不再是地表固定点的 116.75 天。</p>
    </div>
  </section>
  <script>
    const BASE = {json.dumps(summary)};
    const ETA = BASE.etaCell;
    const SYS = {SYSTEM_FACTOR};
    const STORAGE = {STORAGE_EFF};
    const LSFR_SZA = {LSFR_SZA_DEG};
    const MU_REF = Math.cos(LSFR_SZA * Math.PI / 180);
    const SOLAR_DAY = BASE.venusSolarDayEarthD;
    const BALLOON_DAY = BASE.balloonPeriodEarthD;
    const YEAR = BASE.venusYearEarthD;
    const REF_FLUX = BASE.refFlux;
    const el = id => document.getElementById(id);
    const anim = el('animation'), actx = anim.getContext('2d');
    const plot = el('plot'), pctx = plot.getContext('2d');
    let elapsedEarthD = 0;
    let last = performance.now();

    function powerAtLocalHour(hour, area, alpha) {{
      const mu = Math.max(0, Math.cos((hour - 12) * Math.PI / 12));
      const flux = mu > 0 ? REF_FLUX * Math.pow(mu / MU_REF, alpha) : 0;
      return area * flux * ETA * SYS;
    }}
    function localHour(tEarthD, period) {{ return (tEarthD / period * 24) % 24; }}
    function drawAnimation(hour, power, mode) {{
      const w = anim.width, h = anim.height;
      actx.clearRect(0,0,w,h);
      actx.fillStyle = '#1d1f28'; actx.fillRect(0,0,w,h);
      actx.fillStyle = '#ffd666'; actx.beginPath(); actx.arc(90,80,38,0,Math.PI*2); actx.fill();
      const cx=500, cy=215, r=120;
      const terminatorX = cx + Math.cos((hour-12)*Math.PI/12)*r;
      const grad = actx.createLinearGradient(cx-r,0,cx+r,0);
      grad.addColorStop(0,'#f3c25f'); grad.addColorStop(0.55,'#a55b42'); grad.addColorStop(1,'#292d3a');
      actx.fillStyle = grad; actx.beginPath(); actx.arc(cx,cy,r,0,Math.PI*2); actx.fill();
      actx.strokeStyle = '#f7f1e4'; actx.lineWidth=2; actx.beginPath(); actx.arc(cx,cy,r,0,Math.PI*2); actx.stroke();
      const a = (hour - 12) * Math.PI / 12;
      const px = cx + Math.cos(a) * r * 0.88;
      const py = cy + Math.sin(a) * r * 0.88;
      actx.fillStyle = mode === 'surface' ? '#23777a' : '#c07a2a';
      actx.beginPath(); actx.arc(px,py,8,0,Math.PI*2); actx.fill();
      if (mode === 'balloon') {{
        actx.strokeStyle = '#c07a2a'; actx.lineWidth=2; actx.beginPath(); actx.arc(cx,cy,r*1.1,0,Math.PI*2); actx.stroke();
      }}
      actx.fillStyle = '#fffaf0'; actx.font='20px Microsoft YaHei, sans-serif';
      actx.fillText(`local hour = ${{hour.toFixed(2)}}`, 40, 340);
      actx.fillText(`elapsed Earth days = ${{elapsedEarthD.toFixed(2)}}`, 40, 370);
      actx.fillText(`power = ${{power.toFixed(0)}} W`, 40, 400);
      actx.fillText(mode === 'surface' ? '地表固定点: 一天很长' : '气球随超旋转风: 日照周期变短', 610, 64);
    }}
    function drawPlot(mode, area, alpha) {{
      const w=plot.width, h=plot.height, left=62, right=w-28, top=32, bottom=h-54;
      pctx.clearRect(0,0,w,h);
      const period = mode === 'surface' ? SOLAR_DAY : BALLOON_DAY;
      const maxP = area * REF_FLUX * Math.pow(1/MU_REF, alpha) * ETA * SYS;
      pctx.strokeStyle='#6b7280'; pctx.lineWidth=1;
      pctx.beginPath(); pctx.moveTo(left,top); pctx.lineTo(left,bottom); pctx.lineTo(right,bottom); pctx.stroke();
      pctx.fillStyle='#5d6570'; pctx.font='12px Microsoft YaHei, sans-serif';
      for (let y=0; y<=Math.ceil(maxP/100)*100; y+=100) {{
        const py = bottom - y/(Math.ceil(maxP/100)*100)*(bottom-top);
        pctx.strokeStyle='rgba(92,100,112,.22)'; pctx.beginPath(); pctx.moveTo(left,py); pctx.lineTo(right,py); pctx.stroke();
        pctx.fillText(String(y), 24, py+4);
      }}
      for (let hr=0; hr<=24; hr+=4) {{
        const x = left + hr/24*(right-left);
        pctx.strokeStyle='rgba(92,100,112,.18)'; pctx.beginPath(); pctx.moveTo(x,top); pctx.lineTo(x,bottom); pctx.stroke();
        pctx.fillText(String(hr), x-5, bottom+18);
      }}
      pctx.fillText('Power (W)', 12, top-8);
      pctx.fillText('local solar hour', right-100, bottom+36);
      pctx.strokeStyle = mode === 'surface' ? '#23777a' : '#c07a2a'; pctx.lineWidth=3; pctx.beginPath();
      for (let i=0;i<=480;i++) {{
        const hr = i/480*24;
        const p = powerAtLocalHour(hr, area, alpha);
        const x = left + hr/24*(right-left);
        const y = bottom - p/(Math.ceil(maxP/100)*100)*(bottom-top);
        if (i===0) pctx.moveTo(x,y); else pctx.lineTo(x,y);
      }}
      pctx.stroke();
      const currentHour = localHour(elapsedEarthD, period);
      const currentPower = powerAtLocalHour(currentHour, area, alpha);
      const x = left + currentHour/24*(right-left);
      const y = bottom - currentPower/(Math.ceil(maxP/100)*100)*(bottom-top);
      pctx.fillStyle='#202326'; pctx.beginPath(); pctx.arc(x,y,5,0,Math.PI*2); pctx.fill();
      return {{ currentHour, currentPower, period }};
    }}
    function frame(now) {{
      const dt = Math.min(0.08, (now-last)/1000); last = now;
      const speed = Number(el('speed').value);
      elapsedEarthD = (elapsedEarthD + dt * speed) % YEAR;
      const mode = el('mode').value;
      const area = Number(el('area').value);
      const alpha = Number(el('alpha').value);
      const result = drawPlot(mode, area, alpha);
      drawAnimation(result.currentHour, result.currentPower, mode);
      el('periodVal').textContent = `${{result.period.toFixed(2)}} d`;
      el('speedVal').textContent = `${{speed.toFixed(1)}} d/s`;
      el('areaVal').textContent = `${{area.toFixed(1)}} m²`;
      el('alphaVal').textContent = alpha.toFixed(2);
      requestAnimationFrame(frame);
    }}
    ['mode','speed','area','alpha'].forEach(id => {{ el(id).addEventListener('input', () => {{}}); el(id).addEventListener('change', () => {{}}); }});
    requestAnimationFrame(frame);
  </script>
</body>
</html>"""
    (ROOT / "venus_solar_day_year_model.html").write_text(html, encoding="utf-8")


def main() -> None:
    lsfr = load_lsfr()
    ambient_temp_k = load_vega_temperature()
    ref_flux = interpolate_flux_54(lsfr)
    day = day_profile(ref_flux, ambient_temp_k)
    year = year_profile(ref_flux, ambient_temp_k)
    day.to_csv(ROOT / "venus_solar_one_day_profile.csv", index=False, encoding="utf-8-sig")
    year.to_csv(ROOT / "venus_solar_one_year_profile.csv", index=False, encoding="utf-8-sig")
    make_figures(day, year)
    make_notebook(ref_flux, ambient_temp_k, day, year)
    make_html(ref_flux, ambient_temp_k, day, year)
    print("Generated venus_solar_day_year_model.html")
    print("Generated venus_day_year_solar_model.ipynb")
    print("Generated figures/venus_solar_one_day_power.png")
    print("Generated figures/venus_solar_one_year_power.png")


if __name__ == "__main__":
    main()
