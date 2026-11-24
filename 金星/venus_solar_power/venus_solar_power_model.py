from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"
TSI_EARTH_W_M2 = 1361.6
VENUS_AU = 0.723
VENUS_SOLAR_CONSTANT_W_M2 = TSI_EARTH_W_M2 / VENUS_AU**2

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


def find_file(name: str) -> Path:
    candidates = [
        ROOT / name,
        DATA_DIR / name,
        ROOT.parent / "venus_solar_power" / name,
        ROOT.parent / "venus_solar_power" / "data" / name,
        ROOT.parent / "vega_wind_power" / name,
        ROOT.parent / "vega_wind_power" / "data" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {name}")


def load_lsfr() -> pd.DataFrame:
    df = pd.read_csv(find_file("lsfr_solar_flux.csv"), header=None, names=LSFR_COLUMNS)
    for col in LSFR_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_vega_temperature() -> float:
    temps = []
    for name in ["vg1bl_rdr_with_time.csv", "vg2bl_rdr_with_time.csv"]:
        df = pd.read_csv(find_file(name))
        valid = df[df["radius_km"] > 0].copy()
        temps.append(valid["atmospheric_temperature_k"].median())
    return float(np.median(temps))


def interpolate_at_altitude(df: pd.DataFrame, altitude_km: float, column: str) -> float:
    valid = df.dropna(subset=["altitude_km", column]).sort_values("altitude_km")
    return float(np.interp(altitude_km, valid["altitude_km"], valid[column]))


def bracket_rows(df: pd.DataFrame, altitude_km: float) -> pd.DataFrame:
    valid = df.dropna(subset=["altitude_km"]).sort_values("altitude_km")
    lower = valid[valid["altitude_km"] <= altitude_km].tail(1)
    upper = valid[valid["altitude_km"] >= altitude_km].head(1)
    return pd.concat([upper, lower]).drop_duplicates().sort_values("altitude_km", ascending=False)


def cell_efficiency(
    ambient_temp_k: float,
    eta_ref: float = 0.30,
    temp_coeff_per_k: float = -0.0025,
    panel_delta_t_k: float = 15.0,
) -> float:
    cell_temp_k = ambient_temp_k + panel_delta_t_k
    return max(0.0, eta_ref * (1.0 + temp_coeff_per_k * (cell_temp_k - 298.15)))


def estimate_power(
    flux_w_m2: float,
    ambient_temp_k: float,
    area_m2: float = 1.0,
    eta_ref: float = 0.30,
    temp_coeff_per_k: float = -0.0025,
    panel_delta_t_k: float = 15.0,
    system_factor: float = 0.90,
) -> dict[str, float]:
    eta_cell = cell_efficiency(ambient_temp_k, eta_ref, temp_coeff_per_k, panel_delta_t_k)
    power_density = flux_w_m2 * eta_cell * system_factor
    return {
        "flux_w_m2": flux_w_m2,
        "cell_efficiency": eta_cell,
        "power_density_w_m2": power_density,
        "area_m2": area_m2,
        "power_w": power_density * area_m2,
    }


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def make_outputs() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )

    lsfr = load_lsfr()
    ambient_temp_k = load_vega_temperature()
    target_altitude = 54.0
    records = []
    for channel in ["vis", "combined", "narrowband"]:
        flux_col = f"flux_{channel}_down_w_m2"
        flux = interpolate_at_altitude(lsfr, target_altitude, flux_col)
        p = estimate_power(flux, ambient_temp_k)
        records.append(
            {
                "altitude_km": target_altitude,
                "channel": channel,
                "down_flux_w_m2": flux,
                "vega_layer_air_temperature_k": ambient_temp_k,
                "cell_efficiency": p["cell_efficiency"],
                "electric_power_density_w_m2": p["power_density_w_m2"],
            }
        )
    summary = pd.DataFrame(records)
    summary.to_csv(ROOT / "venus_solar_power_from_lsfr.csv", index=False, encoding="utf-8-sig")

    make_figure(lsfr, ambient_temp_k, target_altitude)
    make_notebook(summary, ambient_temp_k, target_altitude, bracket_rows(lsfr, target_altitude))
    make_html(lsfr, summary, ambient_temp_k, target_altitude)

    combined = summary[summary["channel"] == "combined"].iloc[0]
    print("Generated venus_solar_power_from_lsfr.csv")
    print("Generated figures/venus_lsfr_solar_power.png")
    print("Generated venus_solar_power_model.ipynb")
    print("Generated venus_solar_principle.html")
    print(
        f"LSFR 54 km combined downward flux={combined['down_flux_w_m2']:.1f} W/m2, "
        f"electric={combined['electric_power_density_w_m2']:.1f} W/m2"
    )


def make_figure(lsfr: pd.DataFrame, ambient_temp_k: float, target_altitude: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    data = lsfr.sort_values("altitude_km")
    axes[0].plot(data["altitude_km"], data["flux_vis_down_w_m2"], lw=2, label="visible down")
    axes[0].plot(data["altitude_km"], data["flux_combined_down_w_m2"], lw=2, label="combined down")
    axes[0].plot(data["altitude_km"], data["flux_narrowband_down_w_m2"], lw=2, label="narrowband down")
    axes[0].axvline(target_altitude, color="k", ls="--", lw=1.3, label="54 km")
    axes[0].set_ylabel("Measured downward solar flux (W/m2)")
    axes[0].set_xlabel("Altitude (km)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    power = data["flux_combined_down_w_m2"].apply(
        lambda flux: estimate_power(flux, ambient_temp_k)["power_density_w_m2"]
    )
    axes[1].plot(data["altitude_km"], power, lw=2.2, color="#c07a2a")
    axes[1].axvline(target_altitude, color="k", ls="--", lw=1.3)
    axes[1].set_ylabel("PV electric power density (W/m2)")
    axes[1].set_xlabel("Altitude (km)")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Pioneer Venus LSFR measured solar flux and 54 km PV estimate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "venus_lsfr_solar_power.png", bbox_inches="tight")
    plt.close(fig)


def make_notebook(summary: pd.DataFrame, ambient_temp_k: float, target_altitude: float, bracket: pd.DataFrame) -> None:
    bracket_table = bracket[
        [
            "grt",
            "pressure_atm",
            "altitude_km",
            "flux_vis_down_w_m2",
            "flux_combined_down_w_m2",
            "flux_narrowband_down_w_m2",
        ]
    ].round(3)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 金星气球高度太阳能：Pioneer Venus LSFR 实测太阳通量模型\n\n"
            "这个 notebook 读取本地 `lsfr_solar_flux.csv`，用 PDS4 XML 标签定义列名，"
            "再把下行太阳通量插值到 VEGA 气球高度附近的 54 km。"
        ),
        nbf.v4.new_markdown_cell(
            "## LSFR 是什么\n\n"
            "LSFR = **Large Probe Solar Flux Radiometer**，即 Pioneer Venus Large Probe 上的太阳通量辐射计。"
            "PDS4 标签说明 CSV 无表头，12 列分别是 `grt, pressure, altitude, visible down/up/net, "
            "combined down/up/net, narrowband down/up/net`。太阳能估算优先看 `combined down`，同时保留 visible 作保守参考。"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd()\n"
            "DATA = ROOT / 'data'\n"
            "LSFR_COLUMNS = ['grt','pressure_atm','altitude_km','flux_vis_down_w_m2','flux_vis_up_w_m2','flux_vis_net_w_m2','flux_combined_down_w_m2','flux_combined_up_w_m2','flux_combined_net_w_m2','flux_narrowband_down_w_m2','flux_narrowband_up_w_m2','flux_narrowband_net_w_m2']\n"
            "lsfr_path = DATA / 'lsfr_solar_flux.csv' if (DATA / 'lsfr_solar_flux.csv').exists() else ROOT / 'lsfr_solar_flux.csv'\n"
            "lsfr = pd.read_csv(lsfr_path, header=None, names=LSFR_COLUMNS)\n"
            "for col in LSFR_COLUMNS[1:]: lsfr[col] = pd.to_numeric(lsfr[col], errors='coerce')\n"
            "lsfr.head()"
        ),
        nbf.v4.new_markdown_cell(
            f"## 54 km 不是直接取一行，而是插值\n\n"
            f"目标高度：**{target_altitude:.1f} km**。LSFR 表中夹住这个高度的两行是：\n\n"
            + dataframe_to_markdown(bracket_table.reset_index().rename(columns={"index": "csv_row_zero_based"}))
        ),
        nbf.v4.new_code_cell(
            "target_altitude = 54.0\n"
            "s = lsfr.dropna(subset=['altitude_km']).sort_values('altitude_km')\n"
            "for col in ['flux_vis_down_w_m2','flux_combined_down_w_m2','flux_narrowband_down_w_m2']:\n"
            "    print(col, np.interp(target_altitude, s.altitude_km, s[col]))"
        ),
        nbf.v4.new_markdown_cell(
            f"## 光伏模型\n\n"
            f"VEGA 气球层温度中位数约 **{ambient_temp_k:.1f} K**。太阳能板输出按：\n\n"
            "$$P = A \\cdot G_{down}(z) \\cdot \\eta_{cell}(T) \\cdot f_{sys}$$\n\n"
            "其中 LSFR 直接给 `G_down(z)`，不再需要人为假设云层透过率。"
        ),
        nbf.v4.new_code_cell(
            "def eta_cell(T_air_k, eta_ref=0.30, gamma=-0.0025, delta_T=15):\n"
            "    T_cell = T_air_k + delta_T\n"
            "    return max(0, eta_ref * (1 + gamma * (T_cell - 298.15)))\n\n"
            f"T_AIR_K = {ambient_temp_k:.6f}\n"
            "F_SYS = 0.90\n"
            "eta = eta_cell(T_AIR_K)\n"
            "combined_flux_54 = np.interp(54.0, s.altitude_km, s.flux_combined_down_w_m2)\n"
            "power_density = combined_flux_54 * eta * F_SYS\n"
            "print('combined down flux at 54 km =', round(combined_flux_54, 1), 'W/m2')\n"
            "print('cell efficiency =', round(eta, 3))\n"
            "print('electric power density =', round(power_density, 1), 'W/m2')\n"
            "for area in [1, 5, 10]: print(area, 'm2:', round(area * power_density, 1), 'W')"
        ),
        nbf.v4.new_markdown_cell(
            "## 这个功率代表什么时间\n\n"
            "LSFR 的 `G_down` 是 Pioneer Venus Large Probe 下降当时、当地太阳几何条件下测到的瞬时下行通量，"
            "不是“金星全球平均”，也不是自动等于“正午最大值”。如果要估算全天能量，第一步可以引入光照时间比例：\n\n"
            "$$P_{avg} \\approx P_{inst} f_{day} \\eta_{storage}$$\n\n"
            "其中 `f_day` 是气球轨迹中处于光照的时间比例，`eta_storage` 是电池充放电/电源管理效率。"
        ),
        nbf.v4.new_code_cell(
            "area = 5\n"
            "f_day = 0.5\n"
            "eta_storage = 0.85\n"
            "p_inst = area * power_density\n"
            "p_avg = p_inst * f_day * eta_storage\n"
            "e_day_kwh = p_avg * 24 / 1000\n"
            "print('instant power W =', round(p_inst, 1))\n"
            "print('24 h averaged power W =', round(p_avg, 1))\n"
            "print('energy per Earth day kWh =', round(e_day_kwh, 2))"
        ),
        nbf.v4.new_markdown_cell(
            "## 面板面积怎么估\n\n"
            "VEGA 气球直径约 3.4-3.54 m，漂浮高度约 53-54 km，气球本体约 12 kg，吊舱约 6.9 kg，"
            "另有约 2.1 kg 氦气，系统量级约 21 kg。原任务不是按携带太阳能板设计的，所以不能直接认为还有大量载荷余量。\n\n"
            "粗估可用：\n\n"
            "$$A_{panel} \\approx m_{panel}/\\sigma_{panel}$$\n\n"
            "其中 `sigma_panel` 是太阳能板、展开机构、支撑、线缆和防酸封装后的面密度。"
            "如果按 0.5-2 kg/m² 的工程假设，2 kg 质量预算对应约 1-4 m²。"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharex=True)\n"
            "for col, label in [('flux_vis_down_w_m2','visible'), ('flux_combined_down_w_m2','combined'), ('flux_narrowband_down_w_m2','narrowband')]:\n"
            "    axes[0].plot(s.altitude_km, s[col], label=label)\n"
            "axes[0].axvline(54, color='k', ls='--')\n"
            "axes[0].set_ylabel('Downward solar flux (W/m2)')\n"
            "axes[0].set_xlabel('Altitude (km)')\n"
            "axes[0].legend()\n"
            "power = s.flux_combined_down_w_m2 * eta * F_SYS\n"
            "axes[1].plot(s.altitude_km, power)\n"
            "axes[1].axvline(54, color='k', ls='--')\n"
            "axes[1].set_ylabel('Electric power density (W/m2)')\n"
            "axes[1].set_xlabel('Altitude (km)')\n"
            "for ax in axes: ax.grid(True, alpha=.25)\n"
            "fig.tight_layout()"
        ),
        nbf.v4.new_markdown_cell("## 结果汇总\n\n" + dataframe_to_markdown(summary.round(3))),
    ]
    nbf.write(nb, ROOT / "venus_solar_power_model.ipynb")


def make_html(lsfr: pd.DataFrame, summary: pd.DataFrame, ambient_temp_k: float, target_altitude: float) -> None:
    lsfr_records = (
        lsfr[["altitude_km", "flux_vis_down_w_m2", "flux_combined_down_w_m2", "flux_narrowband_down_w_m2"]]
        .dropna(subset=["altitude_km"])
        .sort_values("altitude_km")
        .round(4)
        .to_dict(orient="records")
    )
    bracket = bracket_rows(lsfr, target_altitude)
    combined = summary[summary["channel"] == "combined"].iloc[0]
    bracket_html = "".join(
        f"<tr><td>{idx}</td><td>{row['grt']}</td><td>{row['pressure_atm']:.2f}</td>"
        f"<td>{row['altitude_km']:.2f}</td><td>{row['flux_combined_down_w_m2']:.1f}</td></tr>"
        for idx, row in bracket.iterrows()
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>金星气球太阳能：LSFR 实测通量模型</title>
  <style>
    :root {{ --bg:#f7f1e4; --ink:#1f2529; --muted:#5c6470; --line:#d5c8b3; --panel:#fffaf0; --teal:#23777a; --gold:#c07a2a; --wine:#9b4050; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",Arial,sans-serif; line-height:1.55; }}
    header {{ padding:28px clamp(18px,5vw,56px) 18px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,#fff8e9,#f7f1e4); }}
    h1 {{ margin:0 0 8px; font-size:clamp(28px,4.5vw,46px); letter-spacing:0; }}
    header p, .note, td {{ color:var(--muted); }}
    section {{ padding:22px clamp(18px,5vw,56px); border-bottom:1px solid var(--line); }}
    h2 {{ margin:0 0 12px; font-size:24px; }}
    h3 {{ margin:0 0 8px; font-size:17px; }}
    .two {{ display:grid; grid-template-columns:minmax(320px,1fr) minmax(320px,1fr); gap:18px; align-items:start; }}
    .box {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .formula {{ font-family:Consolas,"Courier New",monospace; white-space:pre-wrap; background:#fff3d6; border:1px solid #dbc28e; border-radius:8px; padding:12px; color:#24272a; }}
    canvas {{ width:100%; height:390px; display:block; background:white; border:1px solid var(--line); border-radius:8px; }}
    .controls {{ display:grid; gap:11px; margin-top:12px; }}
    .control-row {{ display:grid; grid-template-columns:150px 1fr 92px; gap:10px; align-items:center; }}
    label {{ color:var(--muted); font-size:13px; }}
    input, select {{ width:100%; min-height:34px; accent-color:var(--teal); }}
    select {{ border:1px solid var(--line); border-radius:6px; background:white; padding:0 8px; font:inherit; }}
    .value {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .metric {{ background:#fffdf8; border:1px solid var(--line); border-radius:8px; padding:10px; }}
    .metric span {{ display:block; color:var(--muted); font-size:12px; }}
    .metric b {{ display:block; margin-top:5px; font-size:22px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }}
    th, td {{ padding:9px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ background:#f3e4ca; color:var(--ink); }}
    a {{ color:var(--teal); }}
    @media (max-width:920px) {{ .two,.metrics {{ grid-template-columns:1fr; }} .control-row {{ grid-template-columns:1fr; gap:4px; }} canvas {{ height:330px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>金星气球太阳能：LSFR 实测通量模型</h1>
    <p>这版已经读取你下载的 <code>lsfr_solar_flux.csv</code>。54 km 的太阳通量由 Pioneer Venus LSFR 实测剖面插值得到，不再用人为云层透过率估计。</p>
  </header>
  <section class="two">
    <div class="box">
      <h2>LSFR 是什么</h2>
      <p>LSFR = <b>Large Probe Solar Flux Radiometer</b>，Pioneer Venus Large Probe 上的太阳通量辐射计。CSV 的第 7 列 <code>flux_combined_down</code> 是可见光+红外合成宽波段的下行太阳通量，单位 W/m²。</p>
      <div class="formula">读取列:
altitude_km = 第 3 列
flux_vis_down = 第 4 列
flux_combined_down = 第 7 列
flux_narrowband_down = 第 10 列

54 km 取值:
G_down(54) = interp(altitude, flux_combined_down)</div>
    </div>
    <div class="box">
      <h2>54 km 插值用到的行</h2>
      <table>
        <thead><tr><th>CSV 行号</th><th>GRT</th><th>压力 atm</th><th>高度 km</th><th>combined down W/m²</th></tr></thead>
        <tbody>{bracket_html}</tbody>
      </table>
      <p class="note">插值结果：54 km 的宽波段下行太阳通量约 <b>{combined['down_flux_w_m2']:.1f} W/m²</b>；默认光伏电功率密度约 <b>{combined['electric_power_density_w_m2']:.1f} W/m²</b>。</p>
      <p class="note">这个数代表 Pioneer Venus 探测器下降当时、当地太阳几何条件下的瞬时下行通量，不是全球平均，也不是自动等于正午最大值。</p>
    </div>
  </section>
  <section class="two">
    <div class="box">
      <h2>up / down / net 是什么</h2>
      <p><b>down</b> 是向下传播的太阳通量，方向大致朝向金星地表；太阳能板朝上时，最该先看它。<b>up</b> 是被云层/大气/地表散射或反射后向上传播的通量。<b>net</b> 是净向下通量，等于 down - up，常用于大气吸收和加热率分析。</p>
      <div class="formula">down: F_down
up:   F_up
net:  F_net = F_down - F_up</div>
    </div>
    <div class="box">
      <h2>这个瞬时对应什么时间</h2>
      <p>Pioneer Venus Large Probe 的公开任务表给出进入/落点几何：太阳天顶角约 65.7°，当地金星时约 7:38。LSFR 在 54 km 的样本 GRT 约 18:56，属于同一次下降的早晨光照条件，不是正午。</p>
      <div class="formula">Large Probe:
local Venus time ≈ 07:38
solar zenith angle ≈ 65.7°
cos(SZA) ≈ 0.41</div>
      <p class="note">金星自转确实很慢，但仍然有当地太阳时。对几十分钟的下降过程来说，当地时变化很小；对长期气球任务来说，需要按气球漂移和金星太阳日重新积分。</p>
    </div>
  </section>
  <section class="two">
    <div>
      <h2>实测通量曲线和交互计算</h2>
      <canvas id="plot" width="980" height="430"></canvas>
      <div class="controls">
        <div class="control-row"><label for="altitude">高度 z (km)</label><input id="altitude" type="range" min="0.2" max="64.3" step="0.1" value="54"><div class="value" id="altitudeVal"></div></div>
        <div class="control-row"><label for="channel">LSFR 通量通道</label><select id="channel"><option value="combined">combined down</option><option value="vis">visible down</option><option value="narrowband">narrowband down</option></select><div class="value" id="fluxVal"></div></div>
        <div class="control-row"><label for="area">太阳能板面积 A</label><input id="area" type="range" min="0.5" max="30" step="0.5" value="5"><div class="value" id="areaVal"></div></div>
        <div class="control-row"><label for="eta">参考效率 eta_ref</label><input id="eta" type="range" min="0.12" max="0.38" step="0.01" value="0.30"><div class="value" id="etaVal"></div></div>
        <div class="control-row"><label for="factor">姿态/系统因子</label><input id="factor" type="range" min="0.3" max="1.0" step="0.05" value="0.90"><div class="value" id="factorVal"></div></div>
        <div class="control-row"><label for="daylight">光照时间比例</label><input id="daylight" type="range" min="0.05" max="1.0" step="0.05" value="0.50"><div class="value" id="daylightVal"></div></div>
        <div class="control-row"><label for="storage">储能效率</label><input id="storage" type="range" min="0.5" max="1.0" step="0.05" value="0.85"><div class="value" id="storageVal"></div></div>
      </div>
    </div>
    <div>
      <h2>功率模型</h2>
      <div class="metrics">
        <div class="metric"><span>LSFR 下行通量</span><b id="fluxOut"></b></div>
        <div class="metric"><span>温度修正效率</span><b id="etaOut"></b></div>
        <div class="metric"><span>单位面积电功率</span><b id="densityOut"></b></div>
        <div class="metric"><span>总电功率</span><b id="powerOut"></b></div>
        <div class="metric"><span>24h 平均功率</span><b id="avgPowerOut"></b></div>
        <div class="metric"><span>每地球日能量</span><b id="dailyEnergyOut"></b></div>
      </div>
      <div class="box" style="margin-top:12px;">
        <h3>数学关系</h3>
        <div class="formula" id="formula"></div>
        <p class="note">这里的 G_down 已经是探测器在金星大气中测到的下行通量；如果太阳能板姿态、谱响应或昼夜平均要更严格，需要再加几何和储能模型。</p>
      </div>
    </div>
  </section>
  <section class="two">
    <div class="box">
      <h2>面板姿态因子是什么</h2>
      <p>“姿态因子”更准确地说是太阳能板接收辐射时的几何/系统折减项。若光线近似为直射，太阳光和面板法线夹角为 theta，则单位面积接收到的通量要乘投影项 <code>max(0, cos(theta))</code>：太阳正对面板时为 1，斜照时小于 1，照到背面时取 0。</p>
      <p>但 LSFR 在金星云层中测到的 <code>G_down</code> 不是一束很干净的直射光，而是向下半球方向的积分通量，里面包含大量云层散射。因此网页里的 <code>factor</code> 是综合因子：包含面板姿态、局部遮挡、硫酸云雾/污染、谱响应不匹配、MPPT、线缆和电源管理损失。若暂时没有更详细结构，取 0.85-0.95 是工程一阶估算；若只想表达姿态，可把系统损失单列，把 <code>factor = f_attitude * eta_system</code>。</p>
      <div class="formula">直射几何:
G_panel = G_direct * max(0, cos(theta))

金星云层简化:
P = A * G_down_LSFR * eta_cell(T) * factor</div>
    </div>
    <div class="box">
      <h2>VEGA 气球能带多大太阳能板</h2>
      <p>VEGA 是直径约 3.4-3.54 m 的球形气球，吊舱约 6.9 kg，气球系统约 12 kg，氦气约 2.1 kg，总量级约 21 kg。原任务没有为太阳能板留明确质量余量，所以面板面积应按质量预算倒推。</p>
      <div class="formula">A_panel ≈ m_panel / sigma_panel

若 sigma_panel = 0.5-2 kg/m²:
2 kg 面板预算 ≈ 1-4 m²
5 kg 面板预算 ≈ 2.5-10 m²</div>
      <p class="note">InSight 的两个太阳能板直径约 2.15-2.2 m，每个面积约 3.6-3.8 m²，两个合计约 7.2-7.6 m²；但它是火星地面着陆器，不代表 Vega 级气球能直接携带同等结构。</p>
    </div>
  </section>
  <section>
    <h2>eta_cell(T) 是怎么算出来的</h2>
    <p>网页里滑块给的是标准测试条件下的参考效率 <code>eta_ref</code>。标准测试通常按电池片温度 25 °C，也就是 298.15 K 来定义。实际在金星 54 km 云层中，电池片会比环境空气更热，所以先估计电池片温度：</p>
    <div class="formula">T_cell = T_air + delta_T

本页默认:
T_air ≈ {ambient_temp_k:.1f} K
delta_T = 15 K
T_cell ≈ {ambient_temp_k + 15.0:.1f} K</div>
    <p>然后用光伏常见的温度系数做线性修正。晶硅或多结电池的效率通常随温度升高而下降，所以温度系数 gamma 取负值；本页默认 <code>gamma = -0.0025 / K</code>，意思是电池片温度每升高 1 K，效率相对降低约 0.25%。</p>
    <div class="formula">eta_cell(T) = eta_ref * [1 + gamma * (T_cell - 298.15 K)]

若 eta_ref = 0.29:
eta_cell = 0.29 * [1 - 0.0025 * ({ambient_temp_k + 15.0:.1f} - 298.15)]
         ≈ {cell_efficiency(ambient_temp_k, eta_ref=0.29):.4f}
         ≈ {cell_efficiency(ambient_temp_k, eta_ref=0.29) * 100:.1f}%</div>
    <p class="note">所以你看到 eta_ref 设成 29%，eta_cell(T) 变成约 27.1%，不是又乘了一个神秘系数，而是因为 54 km 环境温度加上面板受光升温后，电池片温度高于 25 °C，按负温度系数折减了效率。这个系数是工程假设参数，不是 LSFR 直接测出来的；后面可以按具体太阳能电池类型替换。</p>
  </section>
  <section>
    <h2>来源</h2>
    <p class="note">PDS4 bundle: <a href="https://atmos.nmsu.edu/PDS/data/PDS4/pv_sed_bundle/" target="_blank" rel="noreferrer">pv_sed_bundle</a>；LSFR collection: <a href="https://pds.nasa.gov/ds-view/pds/viewCollection.jsp?identifier=urn%3Anasa%3Apds%3Apv_sed%3Adata_lsfr&version=1.0" target="_blank" rel="noreferrer">urn:nasa:pds:pv_sed:data_lsfr</a>；Vega 气球参数参考 <a href="https://www.nationalacademies.org/read/6430/chapter/5" target="_blank" rel="noreferrer">National Academies, A Scientific Rationale for Mobility in Planetary Environments</a> 和 <a href="https://atmos.nmsu.edu/data_and_services/atmospheres_data/VENUS/vega.html" target="_blank" rel="noreferrer">NASA PDS Atmospheres Vega data page</a>。</p>
  </section>
  <script>
    const DATA = {json.dumps(lsfr_records, ensure_ascii=False)};
    const T_AIR_K = {ambient_temp_k:.6f};
    const GAMMA = -0.0025;
    const el = id => document.getElementById(id);
    const canvas = el('plot');
    const ctx = canvas.getContext('2d');
    function column(channel) {{
      return channel === 'vis' ? 'flux_vis_down_w_m2' : channel === 'narrowband' ? 'flux_narrowband_down_w_m2' : 'flux_combined_down_w_m2';
    }}
    function interp(z, col) {{
      const rows = DATA.filter(r => Number.isFinite(r[col])).sort((a,b) => a.altitude_km - b.altitude_km);
      if (z <= rows[0].altitude_km) return rows[0][col];
      if (z >= rows[rows.length - 1].altitude_km) return rows[rows.length - 1][col];
      for (let i=0;i<rows.length-1;i++) {{
        const a = rows[i], b = rows[i+1];
        if (z >= a.altitude_km && z <= b.altitude_km) {{
          const f = (z - a.altitude_km) / (b.altitude_km - a.altitude_km);
          return a[col] + f * (b[col] - a[col]);
        }}
      }}
      return NaN;
    }}
    function etaCell(etaRef) {{
      const tCell = T_AIR_K + 15;
      return Math.max(0, etaRef * (1 + GAMMA * (tCell - 298.15)));
    }}
    function update() {{
      const z = Number(el('altitude').value);
      const ch = el('channel').value;
      const area = Number(el('area').value);
      const etaRef = Number(el('eta').value);
      const factor = Number(el('factor').value);
      const daylight = Number(el('daylight').value);
      const storage = Number(el('storage').value);
      const flux = interp(z, column(ch));
      const eta = etaCell(etaRef);
      const density = flux * eta * factor;
      const power = density * area;
      const avgPower = power * daylight * storage;
      const dailyEnergy = avgPower * 24 / 1000;
      el('altitudeVal').textContent = `${{z.toFixed(1)}} km`;
      el('fluxVal').textContent = `${{flux.toFixed(1)}} W/m²`;
      el('areaVal').textContent = `${{area.toFixed(1)}} m²`;
      el('etaVal').textContent = `${{(etaRef*100).toFixed(0)}}%`;
      el('factorVal').textContent = factor.toFixed(2);
      el('daylightVal').textContent = `${{(daylight*100).toFixed(0)}}%`;
      el('storageVal').textContent = `${{(storage*100).toFixed(0)}}%`;
      el('fluxOut').textContent = `${{flux.toFixed(1)}} W/m²`;
      el('etaOut').textContent = `${{(eta*100).toFixed(1)}}%`;
      el('densityOut').textContent = `${{density.toFixed(1)}} W/m²`;
      el('powerOut').textContent = power >= 1000 ? `${{(power/1000).toFixed(2)}} kW` : `${{power.toFixed(0)}} W`;
      el('avgPowerOut').textContent = avgPower >= 1000 ? `${{(avgPower/1000).toFixed(2)}} kW` : `${{avgPower.toFixed(0)}} W`;
      el('dailyEnergyOut').textContent = `${{dailyEnergy.toFixed(2)}} kWh`;
      el('formula').textContent = `P = A * G_down(z) * eta_cell(T) * factor\\nP = ${{area.toFixed(1)}} * ${{flux.toFixed(1)}} * ${{eta.toFixed(3)}} * ${{factor.toFixed(2)}} = ${{power.toFixed(1)}} W`;
      draw(z, ch, flux);
    }}
    function draw(z, ch, flux) {{
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0,0,w,h);
      const cols = ['flux_vis_down_w_m2','flux_combined_down_w_m2','flux_narrowband_down_w_m2'];
      const labels = ['visible','combined','narrowband'];
      const colors = ['#23777a','#c07a2a','#9b4050'];
      const left=66, right=w-28, top=34, bottom=h-54;
      const maxFluxRaw = Math.max(...DATA.flatMap(r => cols.map(c => r[c] || 0)));
      const maxFlux = Math.ceil(maxFluxRaw / 100) * 100;
      ctx.strokeStyle = '#6b7280'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom); ctx.stroke();
      ctx.fillStyle = '#5c6470'; ctx.font = '13px Microsoft YaHei, sans-serif';
      ctx.fillText('Flux (W/m²)', 14, top - 10);
      ctx.fillText('Altitude (km)', right - 90, bottom + 36);
      ctx.strokeStyle = 'rgba(92,100,112,.22)';
      ctx.fillStyle = '#5c6470';
      ctx.font = '12px Microsoft YaHei, sans-serif';
      for (let yTick = 0; yTick <= maxFlux; yTick += 100) {{
        const y = bottom - (yTick / maxFlux) * (bottom - top);
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(String(yTick), 30, y + 4);
      }}
      for (let xTick = 0; xTick <= 60; xTick += 10) {{
        const x = left + (xTick - 0.2) / (64.3 - 0.2) * (right - left);
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
        ctx.fillText(String(xTick), x - 6, bottom + 18);
      }}
      ctx.strokeStyle = '#6b7280';
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, bottom);
      ctx.lineTo(right, bottom);
      ctx.stroke();
      cols.forEach((col, idx) => {{
        const rows = DATA.filter(r => Number.isFinite(r[col]));
        ctx.strokeStyle = colors[idx]; ctx.lineWidth = col === column(ch) ? 3 : 1.7;
        ctx.beginPath();
        rows.forEach((r,i) => {{
          const x = left + (r.altitude_km - 0.2) / (64.3 - 0.2) * (right-left);
          const y = bottom - (r[col] / maxFlux) * (bottom-top);
          if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
        }});
        ctx.stroke();
        ctx.fillStyle = colors[idx]; ctx.fillText(labels[idx], left + 12 + idx*105, top + 18);
      }});
      const xz = left + (z - 0.2)/(64.3-0.2)*(right-left);
      const yz = bottom - (flux/maxFlux)*(bottom-top);
      ctx.strokeStyle = '#1f2529'; ctx.setLineDash([6,4]); ctx.beginPath(); ctx.moveTo(xz,top); ctx.lineTo(xz,bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#1f2529'; ctx.beginPath(); ctx.arc(xz,yz,5,0,Math.PI*2); ctx.fill();
      ctx.fillText(`${{z.toFixed(1)}} km, ${{flux.toFixed(1)}} W/m²`, Math.min(xz+10, right-160), Math.max(top+28, yz-10));
    }}
    ['altitude','channel','area','eta','factor','daylight','storage'].forEach(id => {{ el(id).addEventListener('input', update); el(id).addEventListener('change', update); }});
    update();
  </script>
</body>
</html>"""
    (ROOT / "venus_solar_principle.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    make_outputs()
