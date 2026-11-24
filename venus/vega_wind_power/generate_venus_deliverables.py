from __future__ import annotations

import base64
import json
import math
import textwrap
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
VENUS_RADIUS_KM = 6051.8
R_CO2 = 8.314462618 / 0.04345  # J kg-1 K-1, CO2-dominated Venus atmosphere


REFERENCES = [
    {
        "key": "PDS Vega Archive",
        "text": "Lorenz, Crisp & Huber, Vega 1 and Vega 2 Balloon and Lander Archive, NASA PDS, 2020, DOI: 10.17189/x3tp-f595.",
        "url": "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=VEGA1%2FVEGA2-V-2%2F3-VENUS-V1.0",
    },
    {
        "key": "PDS Atmospheres Vega",
        "text": "PDS Atmospheres Node, Venus Atmospheric Balloon Data, 11 and 15 June 1985.",
        "url": "https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/VENUS/vega.html",
    },
    {
        "key": "Venus-GRAM",
        "text": "Justh et al., Venus Global Reference Atmospheric Model (Venus-GRAM): User Guide, NASA/TM-20210022168, 2021.",
        "url": "https://ntrs.nasa.gov/citations/20210022168",
    },
    {
        "key": "VEGA vertical winds",
        "text": "Linkin et al., VEGA Balloon Dynamics and Vertical Winds in the Venus Middle Cloud Region, Science, 1986, DOI: 10.1126/science.231.4744.1417.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/17748083/",
    },
    {
        "key": "VEGA turbulence",
        "text": "Kerzhanovich et al., Small-scale turbulence in the Venus middle cloud layer, Soviet Astronomy Letters, 1986; NASA NTRS 19870032285.",
        "url": "https://ntrs.nasa.gov/citations/19870032285",
    },
]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    balloon_frames = []
    for name, label in [
        ("vg1bl_rdr_with_time.csv", "VEGA-1 balloon"),
        ("vg2bl_rdr_with_time.csv", "VEGA-2 balloon"),
    ]:
        df = pd.read_csv(ROOT / name)
        df["probe"] = label
        df["altitude_km"] = df["radius_km"] - VENUS_RADIUS_KM
        df["density_kg_m3"] = df["pressure_hpa"] * 100.0 / (R_CO2 * df["atmospheric_temperature_k"])
        df["horizontal_wind_ms"] = np.sqrt(df["zonal_wind_u_ms"] ** 2 + df["meridional_wind_v_ms"] ** 2)
        df.loc[df["radius_km"] <= 0, ["altitude_km", "density_kg_m3", "horizontal_wind_ms"]] = np.nan
        df.loc[df["horizontal_wind_ms"] == 0, "horizontal_wind_ms"] = np.nan
        balloon_frames.append(df)

    balloon = pd.concat(balloon_frames, ignore_index=True)

    lander = pd.read_csv(ROOT / "vg2lr_with_time.csv")
    lander["probe"] = "VEGA-2 lander"
    lander["altitude_km"] = lander["altitude_m"] / 1000.0
    lander["pressure_hpa"] = lander["pressure_bar"] * 1000.0
    lander["density_kg_m3"] = lander["pressure_bar"] * 1e5 / (R_CO2 * lander["temperature_k"])
    return balloon, lander


def summarize(balloon: pd.DataFrame, lander: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for probe, group in balloon.dropna(subset=["altitude_km"]).groupby("probe"):
        wind = group.dropna(subset=["horizontal_wind_ms"])
        rows.append(
            {
                "dataset": probe,
                "rows": len(group),
                "altitude_min_km": group["altitude_km"].min(),
                "altitude_max_km": group["altitude_km"].max(),
                "pressure_median_hpa": group["pressure_hpa"].median(),
                "temperature_median_k": group["atmospheric_temperature_k"].median(),
                "density_median_kg_m3": group["density_kg_m3"].median(),
                "horizontal_wind_mean_ms": wind["horizontal_wind_ms"].mean(),
                "horizontal_wind_min_ms": wind["horizontal_wind_ms"].min(),
                "horizontal_wind_max_ms": wind["horizontal_wind_ms"].max(),
            }
        )
    rows.append(
        {
            "dataset": "VEGA-2 lander profile",
            "rows": len(lander),
            "altitude_min_km": lander["altitude_km"].min(),
            "altitude_max_km": lander["altitude_km"].max(),
            "pressure_median_hpa": lander["pressure_hpa"].median(),
            "temperature_median_k": lander["temperature_k"].median(),
            "density_median_kg_m3": lander["density_kg_m3"].median(),
            "horizontal_wind_mean_ms": np.nan,
            "horizontal_wind_min_ms": np.nan,
            "horizontal_wind_max_ms": np.nan,
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "venus_environment_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def make_figures(balloon: pd.DataFrame, lander: pd.DataFrame) -> dict[str, Path]:
    setup_matplotlib()
    figures: dict[str, Path] = {}

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), sharey=True)
    env_cols = [
        ("pressure_hpa", "Pressure (hPa)", "log"),
        ("density_kg_m3", "Density (kg/m3)", "log"),
        ("temperature_k", "Temperature (K)", "linear"),
    ]
    lander_plot = lander.sort_values("altitude_km")
    for ax, (col, label, scale) in zip(axes, env_cols):
        if col == "temperature_k":
            x = lander_plot["temperature_k"]
        else:
            x = lander_plot[col]
        ax.plot(x, lander_plot["altitude_km"], color="#6b2fb8", lw=2.2, label="VEGA-2 lander")
        for probe, group in balloon.dropna(subset=["altitude_km"]).groupby("probe"):
            bcol = "atmospheric_temperature_k" if col == "temperature_k" else col
            ax.scatter(group[bcol], group["altitude_km"], s=15, alpha=0.65, label=probe)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.25)
        if scale == "log":
            ax.set_xscale("log")
    axes[0].set_ylabel("Altitude above mean Venus radius (km)")
    axes[0].set_ylim(0, 64)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Venus atmosphere from local VEGA data: pressure, density and temperature")
    fig.tight_layout()
    path = FIG_DIR / "venus_environment_profiles.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures["environment"] = path

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for probe, group in balloon.groupby("probe"):
        valid = group.dropna(subset=["horizontal_wind_ms", "altitude_km"])
        axes[0].plot(valid["elapsed_since_first_h"], valid["horizontal_wind_ms"], lw=2.0, label=probe)
        axes[1].scatter(valid["horizontal_wind_ms"], valid["altitude_km"], s=28, alpha=0.75, label=probe)
    axes[0].set_xlabel("Elapsed time (h)")
    axes[0].set_ylabel("Horizontal wind speed from trajectory inversion (m/s)")
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Horizontal wind speed (m/s)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("VEGA balloon winds near 54 km: absolute wind, not turbine-relative flow")
    fig.tight_layout()
    path = FIG_DIR / "vega_wind_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures["wind"] = path

    results = []
    for scenario in ["step_10ms", "stop_in_60ms"]:
        result = simulate_scenario(scenario=scenario)
        result["scenario"] = scenario
        results.append(result)
    sim = pd.concat(results, ignore_index=True)
    sim.to_csv(ROOT / "venus_balloon_power_scenarios.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    names = {"step_10ms": "10 m/s step gust", "stop_in_60ms": "Balloon reset to 0, air = 60 m/s"}
    for scenario, group in sim.groupby("scenario"):
        axes[0].plot(group["time_s"], np.abs(group["relative_wind_ms"]), lw=2.1, label=names[scenario])
        axes[1].plot(group["time_s"], group["power_w"], lw=2.1, label=names[scenario])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("|relative wind| at turbine (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Electrical power (W)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("0D free-balloon turbine model: transient power only unless a reference-frame difference is maintained")
    fig.tight_layout()
    path = FIG_DIR / "free_balloon_power_scenarios.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    figures["scenarios"] = path
    return figures


def simulate_scenario(
    scenario: str,
    duration_s: float = 180.0,
    dt: float = 0.15,
    rho: float = 0.90,
    balloon_diameter_m: float = 3.54,
    dry_mass_kg: float = 21.0,
    cd_balloon: float = 0.50,
    turbine_area_m2: float = 0.50,
    cp: float = 0.35,
    generator_eff: float = 0.80,
    thrust_factor: float = 0.80,
) -> pd.DataFrame:
    volume = math.pi / 6.0 * balloon_diameter_m**3
    added_mass = 0.5 * rho * volume
    m_eff = dry_mass_kg + added_mass
    balloon_area = math.pi * (balloon_diameter_m / 2.0) ** 2
    s_eff = cd_balloon * balloon_area + thrust_factor * turbine_area_m2
    times = np.arange(0.0, duration_s + dt, dt)
    vb = np.zeros_like(times)
    air = np.zeros_like(times)

    if scenario == "step_10ms":
        air[:] = 10.0
        vb[0] = 0.0
    elif scenario == "stop_in_60ms":
        air[:] = 60.0
        vb[0] = 0.0
    else:
        raise ValueError(scenario)

    for i in range(1, len(times)):
        rel = air[i - 1] - vb[i - 1]
        accel = 0.5 * rho * s_eff * rel * abs(rel) / m_eff
        vb[i] = vb[i - 1] + accel * dt

    rel = air - vb
    power = 0.5 * rho * turbine_area_m2 * cp * generator_eff * np.abs(rel) ** 3
    energy = np.cumsum(power) * dt
    return pd.DataFrame(
        {
            "time_s": times,
            "air_speed_ms": air,
            "balloon_speed_ms": vb,
            "relative_wind_ms": rel,
            "power_w": power,
            "energy_j": energy,
        }
    )


def make_notebook(summary: pd.DataFrame) -> Path:
    refs_md = "\n".join(f"- {ref['text']} {ref['url']}" for ref in REFERENCES)
    summary_md = dataframe_to_markdown(summary.round(3))
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 金星 VEGA 数据：大气密度、风速与自由气球风电模型\n\n"
            "这个 notebook 直接读取当前目录中的 VEGA 气球/着陆器 CSV，完成 PPT 第一部分需要的图："
            "压强、温度、密度随高度分布，以及 54 km 附近 VEGA 气球水平风速统计。"
        ),
        nbf.v4.new_markdown_cell(
            "## 数据与参考文献\n\n"
            "本 notebook 使用本地文件 `vg1bl_rdr_with_time.csv`, `vg2bl_rdr_with_time.csv`, `vg2lr_with_time.csv`。"
            "密度按 CO2 主导大气的理想气体近似计算：$\\rho=P/(R_{CO2}T)$，"
            "$R_{CO2}=8.314462618/0.04345\\ \\mathrm{J\\,kg^{-1}\\,K^{-1}}$。\n\n"
            + refs_md
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd()\n"
            "VENUS_RADIUS_KM = 6051.8\n"
            "R_CO2 = 8.314462618 / 0.04345\n"
            "plt.rcParams.update({'font.family': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})"
        ),
        nbf.v4.new_code_cell(
            "balloons = []\n"
            "for file, probe in [('vg1bl_rdr_with_time.csv', 'VEGA-1 balloon'), ('vg2bl_rdr_with_time.csv', 'VEGA-2 balloon')]:\n"
            "    df = pd.read_csv(ROOT / file)\n"
            "    df['probe'] = probe\n"
            "    df['altitude_km'] = df['radius_km'] - VENUS_RADIUS_KM\n"
            "    df['density_kg_m3'] = df['pressure_hpa'] * 100.0 / (R_CO2 * df['atmospheric_temperature_k'])\n"
            "    df['horizontal_wind_ms'] = np.sqrt(df['zonal_wind_u_ms']**2 + df['meridional_wind_v_ms']**2)\n"
            "    df.loc[df['radius_km'] <= 0, ['altitude_km','density_kg_m3','horizontal_wind_ms']] = np.nan\n"
            "    df.loc[df['horizontal_wind_ms'] == 0, 'horizontal_wind_ms'] = np.nan\n"
            "    balloons.append(df)\n"
            "balloon = pd.concat(balloons, ignore_index=True)\n\n"
            "lander = pd.read_csv(ROOT / 'vg2lr_with_time.csv')\n"
            "lander['altitude_km'] = lander['altitude_m'] / 1000\n"
            "lander['pressure_hpa'] = lander['pressure_bar'] * 1000\n"
            "lander['density_kg_m3'] = lander['pressure_bar'] * 1e5 / (R_CO2 * lander['temperature_k'])"
        ),
        nbf.v4.new_markdown_cell("## 数据概览\n\n" + summary_md),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)\n"
            "lander_plot = lander.sort_values('altitude_km')\n"
            "for ax, col, label, scale in zip(\n"
            "    axes,\n"
            "    ['pressure_hpa', 'density_kg_m3', 'temperature_k'],\n"
            "    ['Pressure (hPa)', 'Density (kg/m3)', 'Temperature (K)'],\n"
            "    ['log', 'log', 'linear'],\n"
            "):\n"
            "    ax.plot(lander_plot[col], lander_plot['altitude_km'], lw=2.2, label='VEGA-2 lander')\n"
            "    for probe, group in balloon.dropna(subset=['altitude_km']).groupby('probe'):\n"
            "        bcol = 'atmospheric_temperature_k' if col == 'temperature_k' else col\n"
            "        ax.scatter(group[bcol], group['altitude_km'], s=14, alpha=0.65, label=probe)\n"
            "    if scale == 'log': ax.set_xscale('log')\n"
            "    ax.grid(True, alpha=0.25)\n"
            "    ax.set_xlabel(label)\n"
            "axes[0].set_ylabel('Altitude above mean Venus radius (km)')\n"
            "axes[0].legend(fontsize=8)\n"
            "fig.suptitle('Venus atmosphere from VEGA data')\n"
            "fig.tight_layout()"
        ),
        nbf.v4.new_code_cell(
            "fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))\n"
            "for probe, group in balloon.groupby('probe'):\n"
            "    valid = group.dropna(subset=['horizontal_wind_ms', 'altitude_km'])\n"
            "    axes[0].plot(valid['elapsed_since_first_h'], valid['horizontal_wind_ms'], lw=2, label=probe)\n"
            "    axes[1].scatter(valid['horizontal_wind_ms'], valid['altitude_km'], s=28, alpha=0.75, label=probe)\n"
            "axes[0].set_xlabel('Elapsed time (h)')\n"
            "axes[0].set_ylabel('Horizontal wind speed (m/s)')\n"
            "axes[1].set_xlabel('Horizontal wind speed (m/s)')\n"
            "axes[1].set_ylabel('Altitude (km)')\n"
            "for ax in axes: ax.grid(True, alpha=0.25); ax.legend(fontsize=8)\n"
            "fig.suptitle('VEGA balloon winds near 54 km')\n"
            "fig.tight_layout()"
        ),
        nbf.v4.new_markdown_cell(
            "## 自由气球风机模型的关键判断\n\n"
            "风机真正使用的是相对风 $w=u_{air}-V_b$。同一高度完全自由漂浮时，平台会被气动力加速并随流，"
            "所以 60-70 m/s 的背景风不能直接当作持续风机来流。简化模型为：\n\n"
            "$$m_{eff}\\frac{dV_b}{dt}=\\frac{1}{2}\\rho S_{eff}w|w|,$$\n\n"
            "$$P_e=\\frac{1}{2}\\rho A_t C_p\\eta |w|^3.$$\n\n"
            "网页 `venus_balloon_power_model.html` 把这个模型做成了可调参数演示。"
        ),
    ]
    out = ROOT / "venus_atmosphere_and_wind.ipynb"
    nbf.write(nb, out)
    return out


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def image_to_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def make_html(figures: dict[str, Path], summary: pd.DataFrame) -> Path:
    env_img = image_to_data_uri(figures["environment"])
    wind_img = image_to_data_uri(figures["wind"])
    summary_records = summary.replace({np.nan: None}).round(3).to_dict(orient="records")
    references = json.dumps(REFERENCES, ensure_ascii=False)
    summary_json = json.dumps(summary_records, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>金星 VEGA 气球风力发电模型</title>
  <style>
    :root {{
      --bg: #f6f2ea;
      --ink: #202326;
      --muted: #5d6570;
      --line: #d6cbb9;
      --panel: #fffaf0;
      --accent: #2c7a7b;
      --accent2: #b44e25;
      --gold: #d79a2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 26px clamp(18px, 4vw, 46px) 14px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #fff7e7, #f6f2ea);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(26px, 4vw, 42px); letter-spacing: 0; }}
    header p {{ margin: 0; color: var(--muted); max-width: 980px; line-height: 1.65; }}
    main {{
      display: grid;
      grid-template-columns: minmax(300px, 420px) minmax(0, 1fr);
      min-height: calc(100vh - 116px);
    }}
    aside {{
      padding: 18px;
      border-right: 1px solid var(--line);
      background: #fbf6eb;
    }}
    section {{ padding: 18px clamp(16px, 3vw, 30px) 28px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 14px;
    }}
    .panel h2, .panel h3 {{ margin: 0 0 10px; font-size: 16px; }}
    label {{ display: block; color: var(--muted); font-size: 13px; margin-bottom: 5px; }}
    select, input[type="range"] {{ width: 100%; }}
    select, button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 6px;
      min-height: 36px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); }}
    .row {{ display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: center; margin: 12px 0; }}
    .value {{ font-variant-numeric: tabular-nums; color: var(--ink); min-width: 76px; text-align: right; }}
    .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fffdf8; }}
    .metric b {{ display: block; font-size: 22px; margin-top: 6px; }}
    canvas {{
      display: block;
      width: 100%;
      height: 390px;
      background: linear-gradient(#f8ddb0 0%, #d4884a 55%, #8d4634 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .charts img {{ width: 100%; border: 1px solid var(--line); border-radius: 8px; background: white; }}
    .source-list {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.55; font-size: 13px; }}
    .source-list a {{ color: var(--accent); }}
    .modal {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      background: rgba(32, 35, 38, .42);
      padding: 20px;
      z-index: 20;
    }}
    .modal.open {{ display: flex; }}
    .modal-body {{
      width: min(860px, 100%);
      max-height: 86vh;
      overflow: auto;
      background: #fffaf0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 18px 60px rgba(40, 28, 14, .22);
    }}
    .modal-body p, .modal-body li {{ color: var(--muted); line-height: 1.65; }}
    .modal-head {{ display:flex; justify-content:space-between; align-items:center; gap: 12px; }}
    .modal-head h2 {{ margin:0; font-size: 20px; }}
    .link-button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 36px;
      padding: 0 12px;
      border-radius: 6px;
      background: var(--accent2);
      color: white;
      border: 1px solid var(--accent2);
      cursor: pointer;
      font: inherit;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .charts, .metrics {{ grid-template-columns: 1fr; }}
      canvas {{ height: 320px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>金星 VEGA 气球风力发电模型</h1>
    <p>基于本地 VEGA 气球/着陆器数据计算 54 km 附近密度、压强和风速；右侧是固定高度 0D 模型，显示自由漂浮气球在扰动风或速度重置后，风机只获得瞬态相对风。</p>
  </header>
  <main>
    <aside>
      <div class="panel">
        <h2>场景</h2>
        <label for="scenario">模型输入</label>
        <select id="scenario">
          <option value="gust">突然来一个风速扰动</option>
          <option value="stop">气球速度突然归零，再被背景风加速</option>
        </select>
        <div class="row">
          <label for="airSpeed">背景/扰动风速 (m/s)</label>
          <span class="value" id="airSpeedValue"></span>
        </div>
        <input id="airSpeed" type="range" min="1" max="80" value="10" step="1">
        <div class="row">
          <label for="rho">大气密度 (kg/m3)</label>
          <span class="value" id="rhoValue"></span>
        </div>
        <input id="rho" type="range" min="0.08" max="1.8" value="0.90" step="0.01">
        <div class="row">
          <label for="area">风机扫掠面积 (m2)</label>
          <span class="value" id="areaValue"></span>
        </div>
        <input id="area" type="range" min="0.05" max="5" value="0.50" step="0.05">
        <div class="row">
          <label for="mass">系统干质量 (kg)</label>
          <span class="value" id="massValue"></span>
        </div>
        <input id="mass" type="range" min="10" max="120" value="21" step="1">
        <div class="row">
          <label for="cp">Cp × 发电效率</label>
          <span class="value" id="cpValue"></span>
        </div>
        <input id="cp" type="range" min="0.05" max="0.45" value="0.28" step="0.01">
        <div style="display:flex; gap:10px; margin-top:14px;">
          <button id="restart">重新播放</button>
          <button id="pause" style="background:#4f5f6d;border-color:#4f5f6d;">暂停</button>
        </div>
      </div>
      <div class="panel">
        <h3>模型说明</h3>
        <p style="margin:.2rem 0 .8rem;color:var(--muted);line-height:1.55;">当前版本固定在 Vega 浮航层附近，不计算高度变化；密度滑块代表选定高度层的大气密度。</p>
        <button class="link-button" id="openPrinciple">查看原理</button>
      </div>
      <div class="panel">
        <h3>数据摘要</h3>
        <div id="summary"></div>
      </div>
      <div class="panel">
        <h3>参考来源</h3>
        <ol class="source-list" id="refs"></ol>
      </div>
    </aside>
    <section>
      <div class="metrics">
        <div class="metric">当前相对风 <b id="relWind">0.0 m/s</b></div>
        <div class="metric">当前电功率 <b id="power">0.0 W</b></div>
        <div class="metric">累计电能 <b id="energy">0.0 kJ</b></div>
      </div>
      <canvas id="scene" width="1100" height="440"></canvas>
      <div class="panel" style="margin-top:14px;">
        <h2>模型核心</h2>
        <p style="margin:0;color:var(--muted);line-height:1.65;">u_air 是背景空气速度或扰动幅值；V_balloon 是气球平台速度；w = u_air - V_balloon 是风机真正看到的相对风。当前模型固定高度，不包含升降、垂直风切变或控制策略。</p>
      </div>
      <div class="charts">
        <img src="{env_img}" alt="VEGA pressure density temperature profile">
        <img src="{wind_img}" alt="VEGA wind summary">
      </div>
    </section>
  </main>
  <div class="modal" id="principleModal" role="dialog" aria-modal="true" aria-labelledby="principleTitle">
    <div class="modal-body">
      <div class="modal-head">
        <h2 id="principleTitle">原理说明</h2>
        <button id="closePrinciple" style="background:#4f5f6d;border-color:#4f5f6d;">关闭</button>
      </div>
      <p>这个模型把气球、吊舱和风机合并成一个沿风向运动的自由平台，只看水平方向。它不是把 Vega 的 60-70 m/s 直接当成风机入流，而是先计算平台被空气阻力推着走的加速过程。</p>
      <p style="font-family:Consolas,monospace;color:#202326;background:#fff3d7;border:1px solid #d6cbb9;border-radius:6px;padding:10px;">w = u_air(t) - V_balloon(t)<br>F_drag = 1/2 rho S_eff w |w|<br>m_eff dV_balloon/dt = F_drag<br>P_e = 1/2 rho A_t Cp eta |w|^3</p>
      <ol>
        <li>u_air(t) 是选定高度层的空气速度；V_balloon(t) 是气球平台速度；w 是风机真正看到的相对风。</li>
        <li>如果 w 为正，空气从后方推平台，V_balloon 增大；如果 w 为负，阻力方向反过来，平台会减速或被推向另一方向。</li>
        <li>m_eff 包含系统干质量和附加质量。这里用球体附加质量近似：m_eff = m_dry + 0.5 rho V_sphere。</li>
        <li>S_eff 是等效阻力面积：S_eff = C_D,b A_b + C_T A_t。风机取能的同时也提供阻力，所以会让自由平台更快随流。</li>
        <li>阶跃风的相对风会衰减：|w(t)| = |w0| / (1 + K |w0| t)，K = rho S_eff / (2 m_eff)。所以初始有功率，但会随着气球被推着走而快速下降。</li>
        <li>边界：当前是固定高度 0D 模型，密度由滑块指定；没有模拟高度变化、垂直风切变、系留绳或主动推进。</li>
      </ol>
    </div>
  </div>
  <script>
    const references = {references};
    const summary = {summary_json};
    const el = id => document.getElementById(id);
    const canvas = el('scene');
    const ctx = canvas.getContext('2d');
    const state = {{ t: 0, vb: 0, energy: 0, running: true, last: performance.now(), samples: [], duration: 180 }};
    const controls = ['scenario','airSpeed','rho','area','mass','cp'];
    const D = 3.54, cdBalloon = 0.50, thrustFactor = 0.80;
    function params() {{
      const scenario = el('scenario').value;
      let air = Number(el('airSpeed').value);
      return {{
        scenario,
        air,
        rho: Number(el('rho').value),
        area: Number(el('area').value),
        mass: Number(el('mass').value),
        cpEff: Number(el('cp').value)
      }};
    }}
    function reset() {{
      const p = params();
      state.t = 0; state.energy = 0; state.samples = [];
      state.vb = 0;
      if (p.scenario === 'gust') {{
        el('airSpeed').min = 1;
        el('airSpeed').max = 40;
        el('airSpeed').value = 10;
      }} else {{
        el('airSpeed').min = 20;
        el('airSpeed').max = 80;
        el('airSpeed').value = 60;
      }}
      state.running = true;
      el('pause').textContent = '暂停';
      syncLabels();
    }}
    function syncLabels() {{
      const p = params();
      el('airSpeedValue').textContent = `${{p.air.toFixed(0)}}`;
      el('rhoValue').textContent = p.rho.toFixed(2);
      el('areaValue').textContent = p.area.toFixed(2);
      el('massValue').textContent = p.mass.toFixed(0);
      el('cpValue').textContent = p.cpEff.toFixed(2);
    }}
    function step(dt) {{
      if (state.t >= state.duration) {{
        state.running = false;
        el('pause').textContent = '继续';
        return;
      }}
      dt = Math.min(dt, state.duration - state.t);
      const p = params();
      const volume = Math.PI / 6 * D**3;
      const addedMass = 0.5 * p.rho * volume;
      const meff = p.mass + addedMass;
      const balloonArea = Math.PI * (D/2)**2;
      const seff = cdBalloon * balloonArea + thrustFactor * p.area;
      const rel = p.air - state.vb;
      const accel = 0.5 * p.rho * seff * rel * Math.abs(rel) / meff;
      state.vb += accel * dt;
      const rel2 = p.air - state.vb;
      const power = 0.5 * p.rho * p.area * p.cpEff * Math.abs(rel2)**3;
      state.energy += power * dt;
      state.t += dt;
      state.samples.push({{t: state.t, rel: Math.abs(rel2), p: power}});
      el('relWind').textContent = `${{Math.abs(rel2).toFixed(2)}} m/s`;
      el('power').textContent = power >= 1000 ? `${{(power/1000).toFixed(2)}} kW` : `${{power.toFixed(1)}} W`;
      el('energy').textContent = `${{(state.energy/1000).toFixed(2)}} kJ`;
    }}
    function draw() {{
      const w = canvas.width, h = canvas.height;
      const p = params();
      ctx.clearRect(0, 0, w, h);
      for (let i=0;i<24;i++) {{
        const y = 30 + i*16;
        const len = 45 + 22*Math.sin(state.t*0.8 + i);
        ctx.strokeStyle = i % 2 ? 'rgba(255,255,255,.32)' : 'rgba(80,45,30,.18)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo((i*71 + state.t*18) % (w+80)-80, y);
        ctx.lineTo((i*71 + state.t*18) % (w+80)-80 + len, y);
        ctx.stroke();
      }}
      const x = 220 + 42*Math.sin(state.t*0.35);
      const y = 165 + 12*Math.sin(state.t*0.9);
      ctx.fillStyle = '#f7df91';
      ctx.strokeStyle = '#5b342b';
      ctx.lineWidth = 3;
      ctx.beginPath(); ctx.ellipse(x, y, 70, 54, 0, 0, Math.PI*2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x-28,y+50); ctx.lineTo(x-18,y+118); ctx.moveTo(x+28,y+50); ctx.lineTo(x+18,y+118); ctx.stroke();
      ctx.fillStyle = '#40484e'; ctx.fillRect(x-36,y+112,72,30);
      const tx = x + 96, ty = y + 116;
      ctx.strokeStyle = '#1f2930'; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(x+36,y+128); ctx.lineTo(tx-24,ty); ctx.stroke();
      ctx.beginPath(); ctx.arc(tx,ty,18,0,Math.PI*2); ctx.stroke();
      for (let k=0;k<3;k++) {{
        const a = state.t*8 + k*2*Math.PI/3;
        ctx.beginPath(); ctx.moveTo(tx,ty); ctx.lineTo(tx+46*Math.cos(a), ty+46*Math.sin(a)); ctx.stroke();
      }}
      const rel = Math.abs(p.air - state.vb);
      ctx.fillStyle = '#202326';
      ctx.font = '20px Microsoft YaHei, sans-serif';
      ctx.fillText(`固定高度约 54 km，rho = ${{p.rho.toFixed(2)}} kg/m3`, 34, 318);
      ctx.fillText(`u_air 背景空气 = ${{p.air.toFixed(1)}} m/s`, 34, 348);
      ctx.fillText(`V_balloon 平台 = ${{state.vb.toFixed(1)}} m/s`, 34, 378);
      ctx.fillText(`w 相对风 = ${{rel.toFixed(2)}} m/s`, 34, 408);
      drawPlot(w - 410, 42, 360, 155, '相对风 |w| (m/s)', s => s.rel, '#2c7a7b', state.duration);
      drawPlot(w - 410, 238, 360, 155, '电功率 P_e (W)', s => s.p, '#b44e25', state.duration);
    }}
    function drawPlot(x, y, w, h, title, getter, color, duration) {{
      ctx.fillStyle = 'rgba(255,250,240,.86)';
      ctx.strokeStyle = '#d6cbb9';
      ctx.lineWidth = 1;
      ctx.fillRect(x,y,w,h); ctx.strokeRect(x,y,w,h);
      ctx.fillStyle = '#5d6570'; ctx.font = '14px Microsoft YaHei, sans-serif'; ctx.fillText(title, x+10, y+22);
      const vals = state.samples.map(getter);
      const max = Math.max(1, ...vals);
      const left = x + 42, right = x + w - 12, top = y + 36, bottom = y + h - 24;
      ctx.strokeStyle = '#6b7280';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, bottom);
      ctx.lineTo(right, bottom);
      ctx.stroke();
      ctx.fillStyle = '#5d6570';
      ctx.font = '11px Microsoft YaHei, sans-serif';
      ctx.fillText(max.toFixed(max >= 100 ? 0 : 1), x + 6, top + 4);
      ctx.fillText('0', x + 25, bottom + 4);
      ctx.fillText('t (s)', right - 28, bottom + 17);
      ctx.fillText('0', left - 2, bottom + 17);
      ctx.fillText(duration.toFixed(0), right - 12, bottom + 4);
      ctx.strokeStyle = color; ctx.lineWidth = 2;
      ctx.beginPath();
      state.samples.forEach((sample,i) => {{
        const v = getter(sample);
        const px = left + Math.min(1, sample.t / duration) * (right-left);
        const py = bottom - (v/max)*(bottom-top);
        if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
      }});
      ctx.stroke();
    }}
    function loop(now) {{
      const dt = Math.min(0.08, (now - state.last)/1000);
      state.last = now;
      syncLabels();
      if (state.running) step(dt);
      draw();
      requestAnimationFrame(loop);
    }}
    controls.forEach(id => el(id).addEventListener('input', () => {{ syncLabels(); }}));
    el('scenario').addEventListener('change', reset);
    el('restart').addEventListener('click', reset);
    el('pause').addEventListener('click', () => {{
      state.running = !state.running;
      el('pause').textContent = state.running ? '暂停' : '继续';
    }});
    el('openPrinciple').addEventListener('click', () => el('principleModal').classList.add('open'));
    el('closePrinciple').addEventListener('click', () => el('principleModal').classList.remove('open'));
    el('principleModal').addEventListener('click', event => {{
      if (event.target === el('principleModal')) el('principleModal').classList.remove('open');
    }});
    el('refs').innerHTML = references.map(r => `<li>${{r.text}} <a href="${{r.url}}" target="_blank" rel="noreferrer">链接</a></li>`).join('');
    el('summary').innerHTML = summary.map(row => `<p style="margin:.45rem 0;color:#5d6570;">${{row.dataset}}：高度 ${{row.altitude_min_km?.toFixed?.(1) ?? ''}}-${{row.altitude_max_km?.toFixed?.(1) ?? ''}} km，密度中位数 ${{row.density_median_kg_m3?.toFixed?.(2) ?? ''}} kg/m3，水平风均值 ${{row.horizontal_wind_mean_ms ? row.horizontal_wind_mean_ms.toFixed(1)+' m/s' : '无'}}</p>`).join('');
    reset();
    requestAnimationFrame(loop);
  </script>
</body>
</html>"""
    out = ROOT / "venus_balloon_power_model.html"
    out.write_text(html, encoding="utf-8")
    return out


EMU_PER_IN = 914400
SLIDE_W = int(13.333 * EMU_PER_IN)
SLIDE_H = int(7.5 * EMU_PER_IN)


def emu(inches: float) -> int:
    return int(inches * EMU_PER_IN)


def ppt_text_box(x: float, y: float, w: float, h: float, text: str, size: int = 22, bold: bool = False, color: str = "1F2933") -> str:
    ppt_text_box.counter += 1
    shape_id = ppt_text_box.counter
    lines = text.split("\n")
    paras = []
    for line in lines:
        paras.append(
            f"""<a:p><a:r><a:rPr lang="zh-CN" sz="{size * 100}" {'b="1"' if bold else ''}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/></a:rPr><a:t>{xml_escape(line)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{size * 100}"/></a:p>"""
        )
    return f"""<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(paras)}</p:txBody></p:sp>"""


ppt_text_box.counter = 10


def ppt_image(rid: str, x: float, y: float, w: float, h: float, name: str) -> str:
    ppt_text_box.counter += 1
    return f"""<p:pic><p:nvPicPr><p:cNvPr id="{ppt_text_box.counter}" name="{xml_escape(name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>"""


def ppt_shape(x: float, y: float, w: float, h: float, fill: str = "FFF4DD", line: str = "D6CBB9") -> str:
    ppt_text_box.counter += 1
    return f"""<p:sp><p:nvSpPr><p:cNvPr id="{ppt_text_box.counter}" name="Shape {ppt_text_box.counter}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="roundRect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill><a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr></p:sp>"""


def make_slide_xml(shapes: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{''.join(shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""


def make_pptx(figures: dict[str, Path], summary: pd.DataFrame, html_path: Path, nb_path: Path) -> Path:
    out = ROOT / "金星Vega气球风力发电_数据与物理模型.pptx"
    ppt_text_box.counter = 10
    v1 = summary[summary["dataset"] == "VEGA-1 balloon"].iloc[0]
    v2 = summary[summary["dataset"] == "VEGA-2 balloon"].iloc[0]
    slides: list[tuple[str, list[tuple[str, Path]]]] = []

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_shape(0, 0, 13.333, 7.5, "FFF1D1", "FFF1D1"),
                    ppt_text_box(0.75, 0.75, 11.8, 0.9, "金星 VEGA 气球风力发电", 34, True, "2A221E"),
                    ppt_text_box(0.78, 1.75, 11.5, 0.8, "实测大气环境 + 自由漂浮气球相对风模型", 24, False, "5C4530"),
                    ppt_text_box(
                        0.85,
                        3.05,
                        11.4,
                        2.6,
                        "核心判断：54 km 云层附近密度接近地球海平面量级，绝对水平风约 60-70 m/s；但自由漂浮气球会随风运动，风机看到的是相对风，持续发电需要扰动、剪切、锚定或主动控制。",
                        23,
                        False,
                        "1F2933",
                    ),
                    ppt_text_box(0.9, 6.5, 10.8, 0.35, "数据：NASA PDS VEGA-1/2 balloon and lander archive；模型：0D 二次阻力响应 + 风机功率方程", 12, False, "5D6570"),
                ]
            ),
            [],
        )
    )

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_text_box(0.55, 0.32, 12.0, 0.5, "1. 金星大气基本环境：压强、温度、密度随高度", 24, True),
                    ppt_image("rId2", 0.55, 1.0, 12.1, 4.8, "environment_profiles"),
                    ppt_text_box(0.75, 6.02, 11.7, 0.75, "密度由 Vega P/T 按 CO2 主导理想气体近似计算：rho = P/(R_CO2 T)。气球工作段约 52-55 km；着陆器剖面补足 0-63 km。", 16),
                ]
            ),
            [("image1.png", figures["environment"])],
        )
    )

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_text_box(0.55, 0.32, 12.0, 0.5, "2. VEGA 气球风速：这是绝对水平风，不是风机相对风", 24, True),
                    ppt_image("rId2", 0.65, 1.0, 11.8, 4.4, "wind_summary"),
                    ppt_text_box(
                        0.75,
                        5.75,
                        11.7,
                        0.95,
                        f"本地数据统计：VEGA-1 水平风均值 {v1['horizontal_wind_mean_ms']:.1f} m/s，范围 {v1['horizontal_wind_min_ms']:.1f}-{v1['horizontal_wind_max_ms']:.1f} m/s；VEGA-2 均值 {v2['horizontal_wind_mean_ms']:.1f} m/s，范围 {v2['horizontal_wind_min_ms']:.1f}-{v2['horizontal_wind_max_ms']:.1f} m/s。",
                        16,
                    ),
                ]
            ),
            [("image2.png", figures["wind"])],
        )
    )

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_text_box(0.55, 0.32, 12.0, 0.5, "3. 自由气球 + 风机：物理模型", 24, True),
                    ppt_shape(0.75, 1.05, 5.8, 2.35),
                    ppt_text_box(1.02, 1.28, 5.25, 1.75, "相对风\nw = u_air - V_b\n\n平台响应\nm_eff dV_b/dt = 1/2 rho S_eff w|w|", 20, True, "2C3E50"),
                    ppt_shape(6.85, 1.05, 5.65, 2.35, "EEF8F6"),
                    ppt_text_box(7.12, 1.28, 5.05, 1.75, "风机电功率\nP_e = 1/2 rho A_t Cp eta |w|^3\n\n自由漂浮稳态：w -> 0", 20, True, "1F5E5F"),
                    ppt_text_box(
                        0.85,
                        4.05,
                        11.6,
                        1.25,
                        "解释老师关心点：Vega 的 U/V 是轨迹反演得到的绝对水平风；如果把风机绑在自由气球上，气球会很快被气动力拖到接近背景风速度。能量来自短时相对运动，而不是长期吃掉 60-70 m/s 背景风。",
                        19,
                    ),
                    ppt_text_box(0.85, 6.15, 11.6, 0.45, "可展示网页：" + str(html_path.name), 16, True, "B44E25"),
                ]
            ),
            [],
        )
    )

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_text_box(0.55, 0.32, 12.0, 0.5, "4. 两种展示场景：扰动风 vs 速度归零", 24, True),
                    ppt_image("rId2", 0.7, 1.0, 11.8, 4.55, "scenario_plot"),
                    ppt_text_box(
                        0.75,
                        5.9,
                        11.8,
                        0.75,
                        "场景 A：突然来一个风速扰动，功率随 |w|^3 短时抬升后衰减。场景 B：气球速度归零后被 60 m/s 背景风加速，初始功率大但不可持续。",
                        16,
                    ),
                ]
            ),
            [("image3.png", figures["scenarios"])],
        )
    )

    slides.append(
        (
            make_slide_xml(
                [
                    ppt_text_box(0.55, 0.32, 12.0, 0.5, "5. 交付材料与下一步", 24, True),
                    ppt_text_box(
                        0.85,
                        1.25,
                        11.8,
                        3.2,
                        "已生成：\n- venus_atmosphere_and_wind.ipynb：数据读取、密度计算、环境/风速绘图\n- venus_balloon_power_model.html：可调参数交互模型，可直接浏览器打开\n- figures/*.png：PPT 可直接引用的图\n\n下一步建议：用 Venus-GRAM 或 VIRA 数据补一条标准大气剖面；若继续做发电，应优先建“垂直风切变/可变高度/系留”三类可持续相对风模型。",
                        20,
                    ),
                    ppt_text_box(
                        0.85,
                        5.45,
                        11.8,
                        0.9,
                        "参考：PDS Vega Archive；PDS Atmospheres Vega；NASA/TM-20210022168 Venus-GRAM；Linkin et al. 1986；Kerzhanovich et al. 1986。",
                        14,
                        False,
                        "5D6570",
                    ),
                ]
            ),
            [],
        )
    )

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        image_exts = []
        for _, images in slides:
            for image_name, _ in images:
                image_exts.append(image_name)
        overrides = "\n".join(
            [
                '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
                '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
                '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
                '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
                '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>',
                '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>',
            ]
            + [
                f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                for i in range(1, len(slides) + 1)
            ]
        )
        z.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>{overrides}</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>""",
        )
        z.writestr(
            "docProps/core.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>金星 VEGA 气球风力发电</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{datetime.now(timezone.utc).isoformat()}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{datetime.now(timezone.utc).isoformat()}</dcterms:modified></cp:coreProperties>""",
        )
        z.writestr(
            "docProps/app.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex generated Open XML</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{len(slides)}</Slides></Properties>""",
        )
        slide_ids = "".join(
            f'<p:sldId id="{256+i}" r:id="rId{i+1}"/>' for i in range(1, len(slides) + 1)
        )
        z.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{slide_ids}</p:sldIdLst><p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>""",
        )
        rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
        for i in range(1, len(slides) + 1):
            rels.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
        z.writestr(
            "ppt/_rels/presentation.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(rels)}</Relationships>""",
        )
        z.writestr(
            "ppt/slideMasters/slideMaster1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="F6F2EA"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>""",
        )
        z.writestr(
            "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>""",
        )
        z.writestr(
            "ppt/slideLayouts/slideLayout1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>""",
        )
        z.writestr(
            "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>""",
        )
        z.writestr(
            "ppt/theme/theme1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Codex"><a:themeElements><a:clrScheme name="Codex"><a:dk1><a:srgbClr val="1F2933"/></a:dk1><a:lt1><a:srgbClr val="F6F2EA"/></a:lt1><a:dk2><a:srgbClr val="2C7A7B"/></a:dk2><a:lt2><a:srgbClr val="FFF4DD"/></a:lt2><a:accent1><a:srgbClr val="2C7A7B"/></a:accent1><a:accent2><a:srgbClr val="B44E25"/></a:accent2><a:accent3><a:srgbClr val="D79A2B"/></a:accent3><a:accent4><a:srgbClr val="6B2FB8"/></a:accent4><a:accent5><a:srgbClr val="4F5F6D"/></a:accent5><a:accent6><a:srgbClr val="5D6570"/></a:accent6><a:hlink><a:srgbClr val="2C7A7B"/></a:hlink><a:folHlink><a:srgbClr val="6B2FB8"/></a:folHlink></a:clrScheme><a:fontScheme name="Codex"><a:majorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:majorFont><a:minorFont><a:latin typeface="Microsoft YaHei"/><a:ea typeface="Microsoft YaHei"/><a:cs typeface="Arial"/></a:minorFont></a:fontScheme><a:fmtScheme name="Codex"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>""",
        )
        for i, (slide_xml, images) in enumerate(slides, start=1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml)
            slide_rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>']
            for j, (image_name, image_path) in enumerate(images, start=2):
                z.writestr(f"ppt/media/{image_name}", image_path.read_bytes())
                slide_rels.append(f'<Relationship Id="rId{j}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image_name}"/>')
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{''.join(slide_rels)}</Relationships>""",
            )
    return out


def main() -> None:
    ensure_dirs()
    balloon, lander = load_data()
    summary = summarize(balloon, lander)
    figures = make_figures(balloon, lander)
    notebook = make_notebook(summary)
    html = make_html(figures, summary)
    pptx = make_pptx(figures, summary, html, notebook)
    print("Generated:")
    for path in [*figures.values(), ROOT / "venus_environment_summary.csv", ROOT / "venus_balloon_power_scenarios.csv", notebook, html, pptx]:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
