from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
R_CO2 = 8.314462618 / 0.04345

AREA_M2 = 0.50
CP = 0.35
GENERATOR_EFFICIENCY = 0.80
BALLOON_DIAMETER_M = 3.54
DRY_MASS_KG = 21.0
CD_BALLOON = 0.50
TURBINE_THRUST_FACTOR = 0.80

GUST_HOLD_S = 10.0
INTEGRATION_DT_S = 1.0
SIMULATION_HOURS = 24.0
SIGMA_SCALES = [0.5, 1.0, 2.0]
RANDOM_SEED = 20260905

PROBES = [
    {"probe": "VEGA-1", "rdr": "vg1bl_rdr_with_time.csv", "edr": "vg1bl_edr_with_time.csv"},
    {"probe": "VEGA-2", "rdr": "vg2bl_rdr_with_time.csv", "edr": "vg2bl_edr_with_time.csv"},
]


def cumulative_trapezoid(values: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    energy = np.zeros_like(values, dtype=float)
    if len(values) > 1:
        energy[1:] = np.cumsum((values[1:] + values[:-1]) * 0.5 * np.diff(seconds))
    return energy


def interval_stats(seconds: pd.Series | np.ndarray) -> dict[str, float]:
    diffs = np.diff(np.asarray(seconds, dtype=float))
    if len(diffs) == 0:
        return {"min_s": float("nan"), "median_s": float("nan"), "max_s": float("nan")}
    return {"min_s": float(np.min(diffs)), "median_s": float(np.median(diffs)), "max_s": float(np.max(diffs))}


def density_from_rdr(df: pd.DataFrame) -> pd.Series:
    return df["pressure_hpa"] * 100.0 / (R_CO2 * df["atmospheric_temperature_k"])


def valid_horizontal_wind(df: pd.DataFrame) -> pd.DataFrame:
    speed = np.hypot(df["zonal_wind_u_ms"], df["meridional_wind_v_ms"])
    valid = (df["radius_km"] > 0) & (speed > 0)
    out = df.loc[valid].copy()
    out["horizontal_wind_ms"] = speed.loc[valid]
    return out


def probe_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_rows = []
    wind_rows = []
    for info in PROBES:
        rdr = pd.read_csv(ROOT / info["rdr"])
        edr = pd.read_csv(ROOT / info["edr"])
        rdr_dt = interval_stats(rdr["elapsed_since_first_s"])
        edr_dt = interval_stats(edr["elapsed_since_first_s"])
        valid = valid_horizontal_wind(rdr)
        valid24 = valid[valid["elapsed_since_first_h"] <= SIMULATION_HOURS]
        valid_dt = interval_stats(valid["elapsed_since_first_s"])
        valid24_dt = interval_stats(valid24["elapsed_since_first_s"])

        density24 = density_from_rdr(rdr[rdr["elapsed_since_first_h"] <= SIMULATION_HOURS])
        file_rows.append(
            {
                "probe": info["probe"],
                "rdr_file": info["rdr"],
                "rdr_rows": len(rdr),
                "rdr_duration_h": rdr["elapsed_since_first_h"].iloc[-1],
                "rdr_median_dt_s": rdr_dt["median_s"],
                "edr_file": info["edr"],
                "edr_rows": len(edr),
                "edr_duration_h": edr["elapsed_since_first_h"].iloc[-1],
                "edr_median_dt_s": edr_dt["median_s"],
                "rdr_density_24h_mean_kg_m3": density24.mean(),
            }
        )
        wind_rows.append(
            {
                "probe": info["probe"],
                "horizontal_valid_rows_all": len(valid),
                "horizontal_valid_rows_24h": len(valid24),
                "horizontal_valid_median_dt_s_all": valid_dt["median_s"],
                "horizontal_valid_median_dt_s_24h": valid24_dt["median_s"],
                "horizontal_valid_max_gap_s_all": valid_dt["max_s"],
                "horizontal_valid_max_gap_s_24h": valid24_dt["max_s"],
                "mean_u_ms_all": valid["zonal_wind_u_ms"].mean(),
                "mean_v_ms_all": valid["meridional_wind_v_ms"].mean(),
                "mean_speed_ms_all": valid["horizontal_wind_ms"].mean(),
                "var_speed_ms2_all": valid["horizontal_wind_ms"].var(ddof=1),
                "std_speed_ms_all": valid["horizontal_wind_ms"].std(ddof=1),
                "std_u_ms_all": valid["zonal_wind_u_ms"].std(ddof=1),
                "std_v_ms_all": valid["meridional_wind_v_ms"].std(ddof=1),
            }
        )
    return pd.DataFrame(file_rows), pd.DataFrame(wind_rows)


def parameter_provenance() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "D",
                "value": BALLOON_DIAMETER_M,
                "unit": "m",
                "role": "气球直径，用于体积和迎风面积",
                "status": "近似实测/文献值",
                "source_or_reason": "VEGA 气球公开资料给出直径约 3.4-3.54 m；本模型沿用 3.54 m。",
            },
            {
                "symbol": "m_dry",
                "value": DRY_MASS_KG,
                "unit": "kg",
                "role": "气球系统干质量，用于有效质量",
                "status": "近似实测/文献值",
                "source_or_reason": "VEGA 气球探测器总质量约 21 kg；不同资料给出 VEGA-1/2 约 20.82/21.11 kg。",
            },
            {
                "symbol": "rho(t)",
                "value": "time series",
                "unit": "kg/m3",
                "role": "金星大气密度",
                "status": "由实测数据计算",
                "source_or_reason": "RDR 压强和温度，rho=P/(R_CO2 T)。",
            },
            {
                "symbol": "U_bar, Sigma",
                "value": "per probe",
                "unit": "m/s, (m/s)^2",
                "role": "背景水平风均值和扰动协方差",
                "status": "由实测数据统计",
                "source_or_reason": "RDR 有效水平风点 zonal_wind_u_ms 与 meridional_wind_v_ms。",
            },
            {
                "symbol": "C_D,b",
                "value": CD_BALLOON,
                "unit": "-",
                "role": "气球阻力系数",
                "status": "工程假设",
                "source_or_reason": "按近似球体高雷诺数阻力系数取 0.5；不是 VEGA 专门风洞标定。",
            },
            {
                "symbol": "C_T",
                "value": TURBINE_THRUST_FACTOR,
                "unit": "-",
                "role": "风机取能带来的附加阻力系数",
                "status": "工程假设",
                "source_or_reason": "用 0.8 表示风机阻力载荷；缺少具体转子设计时用于量级估计。",
            },
            {
                "symbol": "A_t",
                "value": AREA_M2,
                "unit": "m2",
                "role": "风机扫掠面积",
                "status": "设计假设",
                "source_or_reason": "假设一个小型风机面积；不是 VEGA 原始设备。",
            },
            {
                "symbol": "C_p",
                "value": CP,
                "unit": "-",
                "role": "风机功率系数",
                "status": "工程假设",
                "source_or_reason": "低于 Betz 极限，用 0.35 表示小型转子较理想但未优化的取能效率。",
            },
            {
                "symbol": "eta",
                "value": GENERATOR_EFFICIENCY,
                "unit": "-",
                "role": "发电机/电气效率",
                "status": "工程假设",
                "source_or_reason": "取 0.80 做一阶估计；不是 VEGA 实测值。",
            },
            {
                "symbol": "C_A",
                "value": 0.5,
                "unit": "-",
                "role": "球体附加质量系数",
                "status": "理论近似",
                "source_or_reason": "球体势流附加质量近似；用于把被气球带动的周围气体惯性计入 m_eff。",
            },
        ]
    )


def simulate_probe(info: dict[str, str], sigma_scale: float, seed: int) -> pd.DataFrame:
    rdr = pd.read_csv(ROOT / info["rdr"])
    valid = valid_horizontal_wind(rdr)
    density_source = rdr[rdr["elapsed_since_first_h"] <= SIMULATION_HOURS].copy()
    density_source["density_kg_m3"] = density_from_rdr(density_source)

    wind_vectors = valid[["zonal_wind_u_ms", "meridional_wind_v_ms"]].to_numpy(dtype=float)
    mean_wind = wind_vectors.mean(axis=0)
    covariance = np.cov((wind_vectors - mean_wind).T)
    covariance = covariance + np.eye(2) * 1e-9

    total_s = SIMULATION_HOURS * 3600.0
    times = np.arange(0.0, total_s + INTEGRATION_DT_S, INTEGRATION_DT_S)
    hold_index = np.floor(times / GUST_HOLD_S).astype(int)
    n_holds = int(hold_index.max()) + 1
    rng = np.random.default_rng(seed)
    gusts = rng.multivariate_normal(mean=[0.0, 0.0], cov=covariance * sigma_scale**2, size=n_holds)
    speed_std = valid["horizontal_wind_ms"].std(ddof=1)
    clip_radius = 3.0 * sigma_scale * speed_std
    radii = np.linalg.norm(gusts, axis=1)
    too_large = radii > clip_radius
    gusts[too_large] *= (clip_radius / radii[too_large])[:, None]

    air_u = mean_wind[0] + gusts[hold_index, 0]
    air_v = mean_wind[1] + gusts[hold_index, 1]
    background = np.hypot(air_u, air_v)
    density = np.interp(times, density_source["elapsed_since_first_s"], density_source["density_kg_m3"])

    balloon_u = np.zeros_like(times)
    balloon_v = np.zeros_like(times)
    balloon_u[0] = mean_wind[0]
    balloon_v[0] = mean_wind[1]

    volume = math.pi / 6.0 * BALLOON_DIAMETER_M**3
    balloon_area = math.pi * (BALLOON_DIAMETER_M / 2.0) ** 2
    s_eff = CD_BALLOON * balloon_area + TURBINE_THRUST_FACTOR * AREA_M2

    for i in range(1, len(times)):
        rel_u = air_u[i - 1] - balloon_u[i - 1]
        rel_v = air_v[i - 1] - balloon_v[i - 1]
        rel_speed = math.hypot(rel_u, rel_v)
        m_eff = DRY_MASS_KG + 0.5 * density[i - 1] * volume
        accel_factor = 0.5 * density[i - 1] * s_eff * rel_speed / m_eff
        balloon_u[i] = balloon_u[i - 1] + accel_factor * rel_u * INTEGRATION_DT_S
        balloon_v[i] = balloon_v[i - 1] + accel_factor * rel_v * INTEGRATION_DT_S

    relative = np.hypot(air_u - balloon_u, air_v - balloon_v)
    power = 0.5 * density * AREA_M2 * CP * GENERATOR_EFFICIENCY * relative**3
    energy_j = cumulative_trapezoid(power, times)

    return pd.DataFrame(
        {
            "probe": info["probe"],
            "sigma_scale": sigma_scale,
            "time_s": times,
            "time_h": times / 3600.0,
            "density_kg_m3": density,
            "air_u_ms": air_u,
            "air_v_ms": air_v,
            "balloon_u_ms": balloon_u,
            "balloon_v_ms": balloon_v,
            "background_wind_ms": background,
            "relative_wind_ms": relative,
            "power_w": power,
            "cumulative_energy_j": energy_j,
            "cumulative_energy_wh": energy_j / 3600.0,
        }
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (probe, scale), group in results.groupby(["probe", "sigma_scale"], sort=False):
        rows.append(
            {
                "probe": probe,
                "sigma_scale": scale,
                "duration_h": group["time_h"].iloc[-1],
                "calculation_rows": len(group),
                "mean_background_wind_ms": group["background_wind_ms"].mean(),
                "mean_relative_wind_ms": group["relative_wind_ms"].mean(),
                "std_relative_wind_ms": group["relative_wind_ms"].std(ddof=1),
                "max_relative_wind_ms": group["relative_wind_ms"].max(),
                "mean_power_w": group["power_w"].mean(),
                "max_power_w": group["power_w"].max(),
                "energy_wh_per_0p5m2": group["cumulative_energy_wh"].iloc[-1],
                "energy_wh_per_m2": group["cumulative_energy_wh"].iloc[-1] / AREA_M2,
            }
        )
    return pd.DataFrame(rows)


def frame_to_markdown(df: pd.DataFrame, digits: int = 4) -> str:
    rounded = df.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(lambda value: f"{value:.{digits}g}")
    headers = list(rounded.columns)
    rows = rounded.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_notebook(
    file_stats: pd.DataFrame,
    wind_stats: pd.DataFrame,
    params: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    file_md = frame_to_markdown(file_stats.round(4))
    wind_md = frame_to_markdown(wind_stats.round(4))
    params_md = frame_to_markdown(params)
    summary_md = frame_to_markdown(summary.round(6), digits=6)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 金星 VEGA 气球：水平风方差驱动的 10 秒扰动风力发电模型\n\n",
                    "这个 notebook 是当前主模型说明和复现实验。前面基于稀疏水平风点线性插值的版本已经不用；",
                    "这里改为先从 VEGA 水平风反演点估计风速扰动方差，再构造 10 秒保持一次的水平风扰动序列。\n\n",
                    "垂直风暂不纳入主发电量估计。原因是气球本身会随浮力、重力、垂直气流和控制策略上下运动；",
                    "若风机主要面向水平来流，垂直相对风对转子有效迎风面积和安装姿态都不确定。当前先把垂直风作为环境扰动证据，",
                    "主计算只使用水平风。这个选择更稳，也更容易向老师解释模型边界。\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. 数据文件与采样情况\n\n",
                    "水平风只在 RDR 文件中：`zonal_wind_u_ms` 与 `meridional_wind_v_ms`。垂直风在 RDR 和 EDR 中都有：",
                    "RDR 的 `vertical_wind_air_wa_ms`、`vertical_speed_balloon_wb_ms`，以及 EDR 的 `vertical_wind_ms`。\n\n",
                    file_md,
                    "\n\n",
                    "RDR 原始表约每 75.4 s 一行，但水平风有效反演点少得多。筛选条件为：",
                    "`radius_km > 0` 且 $\\sqrt{u^2+v^2}>0$。\n\n",
                    wind_md,
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. 水平风统计模型\n\n",
                    "对每个探测器，令有效水平风矢量为\n\n",
                    "$$\n",
                    "\\mathbf{U}_i = \\begin{bmatrix} u_i \\\\ v_i \\end{bmatrix},\n",
                    "$$\n\n",
                    "其中 $u_i$ 是纬向风速 `zonal_wind_u_ms`，$v_i$ 是经向风速 `meridional_wind_v_ms`。\n\n",
                    "背景风取有效水平风的均值：\n\n",
                    "$$\n",
                    "\\bar{\\mathbf{U}} = \\frac{1}{N}\\sum_{i=1}^{N}\\mathbf{U}_i.\n",
                    "$$\n\n",
                    "扰动协方差取：\n\n",
                    "$$\n",
                    "\\mathbf{\\Sigma} = \\frac{1}{N-1}\\sum_{i=1}^{N}\n",
                    "(\\mathbf{U}_i-\\bar{\\mathbf{U}})(\\mathbf{U}_i-\\bar{\\mathbf{U}})^T.\n",
                    "$$\n\n",
                    "之后每 10 s 生成一个保持不变的扰动：\n\n",
                    "$$\n",
                    "\\delta\\mathbf{U}_k \\sim \\mathcal{N}(\\mathbf{0}, s^2\\mathbf{\\Sigma}),\n",
                    "$$\n\n",
                    "其中 $s$ 是扰动倍率。本文计算 $s=0.5, 1.0, 2.0$ 三档，主档可看 $s=1.0$。为避免极端随机值主导结果，",
                    "扰动矢量幅值截断在约 $3s\\sigma_{|U|}$。\n\n",
                    "空气水平风为\n\n",
                    "$$\n",
                    "\\mathbf{U}_{air}(t)=\\bar{\\mathbf{U}}+\\delta\\mathbf{U}_k,\\quad k=\\left\\lfloor\\frac{t}{10\\ \\mathrm{s}}\\right\\rfloor.\n",
                    "$$\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. 气球动力学积分方程\n\n",
                    "气球水平速度记为 $\\mathbf{V}_b(t)$，风机看到的相对来流为\n\n",
                    "$$\n",
                    "\\mathbf{w}(t)=\\mathbf{U}_{air}(t)-\\mathbf{V}_b(t),\n",
                    "\\qquad W(t)=\\|\\mathbf{w}(t)\\|.\n",
                    "$$\n\n",
                    "自由漂浮气球会被相对来流拖曳，水平动力学写成\n\n",
                    "$$\n",
                    "m_{eff}(t)\\frac{d\\mathbf{V}_b}{dt}=\n",
                    "\\frac{1}{2}\\rho(t)S_{eff}W(t)\\mathbf{w}(t).\n",
                    "$$\n\n",
                    "这个方程的来源是牛顿第二定律 $\\sum \\mathbf{F}=m\\mathbf{a}$，而不是单独的动量守恒方程。右端的阻力项来自标准二次阻力模型：\n\n",
                    "$$\n",
                    "\\mathbf{F}_D=\\frac{1}{2}\\rho C_D A W\\mathbf{w}.\n",
                    "$$\n\n",
                    "它也可以理解为相对来流的动压 $q=\\frac{1}{2}\\rho W^2$ 乘以参考面积和阻力系数；",
                    "$W\\mathbf{w}$ 是把阻力方向写成矢量形式，因为 $\\|W\\mathbf{w}\\|=W^2$。\n\n",
                    "这里各变量含义如下：\n\n",
                    "- $m_{eff}$：有效质量，包含干质量和附加质量。\n",
                    "- $\\rho$：金星大气密度，由 RDR 压强和温度估计，$\\rho=P/(R_{CO2}T)$。\n",
                    "- $S_{eff}$：气球加风机的等效阻力面积。\n",
                    "- $W$：风机处相对风速大小。\n\n",
                    "有效质量采用球体附加质量近似：\n\n",
                    "$$\n",
                    "m_{eff}(t)=m_{dry}+\\frac{1}{2}\\rho(t)V_{balloon},\n",
                    "$$\n\n",
                    "其中\n\n",
                    "$$\n",
                    "V_{balloon}=\\frac{\\pi}{6}D^3.\n",
                    "$$\n\n",
                    "等效阻力面积为\n\n",
                    "$$\n",
                    "S_{eff}=C_{D,b}A_b+C_TA_t,\n",
                    "$$\n\n",
                    "其中 $A_b=\\pi(D/2)^2$ 是气球迎风面积，$A_t$ 是风机扫掠面积，$C_T$ 表示风机取能带来的附加阻力系数。\n\n",
                    "数值积分使用显式欧拉法，内部时间步长 $\\Delta t=1\\ \\mathrm{s}$：\n\n",
                    "$$\n",
                    "\\mathbf{V}_{b,n+1}=\\mathbf{V}_{b,n}+\n",
                    "\\frac{\\rho_nS_{eff}W_n}{2m_{eff,n}}\\mathbf{w}_n\\Delta t.\n",
                    "$$\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. 功率和发电量\n\n",
                    "风机瞬时电功率为\n\n",
                    "$$\n",
                    "P_e(t)=\\frac{1}{2}\\rho(t)A_tC_p\\eta W(t)^3.\n",
                    "$$\n\n",
                    "累计发电量为\n\n",
                    "$$\n",
                    "E(t)=\\int_0^t P_e(\\tau)d\\tau.\n",
                    "$$\n\n",
                    "离散计算采用梯形积分：\n\n",
                    "$$\n",
                    "E_n=E_{n-1}+\\frac{P_{e,n-1}+P_{e,n}}{2}(t_n-t_{n-1}).\n",
                    "$$\n\n",
                    "默认参数：$A_t=0.50\\ \\mathrm{m^2}$，$C_p=0.35$，$\\eta=0.80$，$D=3.54\\ \\mathrm{m}$，",
                    "$m_{dry}=21\\ \\mathrm{kg}$，$C_{D,b}=0.50$，$C_T=0.80$。\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. 参数来源与性质\n\n",
                    "下面这张表区分哪些参数来自 VEGA 数据/公开资料，哪些只是工程估计。结论很重要：",
                    "$m_{dry}$ 和 $D$ 有 VEGA 气球资料依据，$\\rho(t)$、$\\bar{\\mathbf U}$、$\\mathbf\\Sigma$ 来自本地数据；",
                    "$C_{D,b}$、$C_T$、$A_t$、$C_p$、$\\eta$ 不是 VEGA 实测标定值，应在报告里称为假设参数或敏感性参数。\n\n",
                    params_md,
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import math\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n\n",
                    "ROOT = Path.cwd()\n",
                    "if not (ROOT / 'vg1bl_rdr_with_time.csv').exists() and (ROOT / 'vega_wind_power').exists():\n",
                    "    ROOT = ROOT / 'vega_wind_power'\n\n",
                    "results = pd.read_csv(ROOT / 'venus_variance_gust_results.csv')\n",
                    "summary = pd.read_csv(ROOT / 'venus_variance_gust_summary.csv')\n",
                    "params = pd.read_csv(ROOT / 'venus_variance_gust_parameter_sources.csv')\n",
                    "summary\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 6. 本次生成的结果摘要\n\n", summary_md],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n",
                    "for (probe, scale), group in results.groupby(['probe', 'sigma_scale']):\n",
                    "    if scale == 1.0:\n",
                    "        axes[0].plot(group['time_h'], group['relative_wind_ms'], label=probe)\n",
                    "        axes[1].plot(group['time_h'], group['cumulative_energy_wh'], label=probe)\n",
                    "axes[0].set_ylabel('相对水平风速 W (m/s)')\n",
                    "axes[1].set_ylabel('累计发电量 (Wh)')\n",
                    "axes[1].set_xlabel('时间 (h)')\n",
                    "for ax in axes:\n",
                    "    ax.grid(True, alpha=0.25)\n",
                    "    ax.legend()\n",
                    "fig.tight_layout()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. 模型架构\n\n",
                    "当前生成流程由 `build_variance_gust_model.py` 完成：\n\n",
                    "1. 读取 `vg1bl_rdr_with_time.csv`、`vg2bl_rdr_with_time.csv`、`vg1bl_edr_with_time.csv`、`vg2bl_edr_with_time.csv`。\n",
                    "2. 从 RDR 中筛选有效水平风点，计算均值、方差、协方差和数据量统计。\n",
                    "3. 对每个探测器和扰动倍率 $s=0.5,1.0,2.0$，生成 10 秒保持的二维水平扰动。\n",
                    "4. 用 1 秒时间步积分自由气球水平动力学。\n",
                    "5. 由相对风速计算瞬时功率，再用梯形积分得到累计发电量。\n",
                    "6. 摘要使用 1 秒内部积分结果；`venus_variance_gust_results.csv` 为每 10 秒保存一行的结果表。\n",
                    "7. 输出 `venus_variance_gust_results.csv`、`venus_variance_gust_summary.csv`、",
                    "`venus_variance_gust_model.ipynb` 和 `venus_variance_gust_model.html`。\n\n",
                    "模型边界：这是方差驱动的随机扰动估计，不是 Vega 对 10 秒水平阵风的直接测量。它比半小时水平风点直接插值更不容易低估短周期扰动，",
                    "但仍然需要在报告中标注为假设模型。\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "venus_variance_gust_model.ipynb"
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def html_table(df: pd.DataFrame, digits: int = 4) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.{digits}g}")
    head = "".join(f"<th>{col}</th>" for col in view.columns)
    rows = []
    for _, row in view.iterrows():
        rows.append("<tr>" + "".join(f"<td>{row[col]}</td>" for col in view.columns) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def make_html(
    file_stats: pd.DataFrame,
    wind_stats: pd.DataFrame,
    params: pd.DataFrame,
    summary: pd.DataFrame,
    results: pd.DataFrame,
) -> Path:
    web = results[np.isclose(results["time_s"] % GUST_HOLD_S, 0.0)].copy()
    records = []
    for row in web.to_dict(orient="records"):
        records.append(
            {
                "probe": row["probe"],
                "sigma_scale": row["sigma_scale"],
                "time_h": round(float(row["time_h"]), 6),
                "background_wind_ms": round(float(row["background_wind_ms"]), 5),
                "relative_wind_ms": round(float(row["relative_wind_ms"]), 5),
                "power_w": round(max(0.0, float(row["power_w"])), 8),
                "cumulative_energy_wh": round(max(0.0, float(row["cumulative_energy_wh"])), 8),
            }
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>金星 Vega 水平风方差扰动风力发电模型</title>
  <script>
    window.MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}}}};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {{
      --bg: #f4f0e8;
      --ink: #1e2529;
      --muted: #5d6b72;
      --line: #cdd5d1;
      --panel: #fffdf8;
      --teal: #2f7d77;
      --rose: #b23a48;
      --gold: #c78c2d;
      --blue: #466f9f;
    }}
    * {{ box-sizing: border-box; }}
    html {{
      height: 100%;
      overflow: hidden;
    }}
    body {{
      margin: 0;
      height: 100%;
      overflow: hidden;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
    }}
    main {{
      height: 100vh;
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-columns: minmax(340px, 40vw) minmax(480px, 1fr);
    }}
    .scene {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(360px, 1fr) auto;
      border-right: 1px solid var(--line);
      background: linear-gradient(180deg, #dfeceb 0%, #efc482 48%, #9a5d65 100%);
    }}
    .topbar, .metrics, .head, .notes {{
      padding: 16px clamp(16px, 3vw, 28px);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(22px, 3.1vw, 34px);
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 17px;
      letter-spacing: 0;
    }}
    p, li {{
      color: #536269;
      line-height: 1.62;
    }}
    p {{ margin: 7px 0; }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 14px;
    }}
    select, button, .link-button {{
      height: 36px;
      border: 1px solid rgba(30,37,41,.22);
      border-radius: 6px;
      background: rgba(255,255,255,.82);
      color: var(--ink);
      padding: 0 11px;
      font: inherit;
    }}
    .link-button {{
      display: inline-flex;
      align-items: center;
      text-decoration: none;
      background: rgba(255,255,255,.82);
    }}
    button {{
      width: 38px;
      padding: 0;
      display: inline-grid;
      place-items: center;
      background: var(--teal);
      color: #fff;
      border-color: var(--teal);
      cursor: pointer;
    }}
    .speed-control {{
      height: 36px;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 0 10px;
      border: 1px solid rgba(30,37,41,.22);
      border-radius: 6px;
      background: rgba(255,255,255,.82);
      color: var(--muted);
      font-size: 13px;
    }}
    .speed-control input {{
      width: 92px;
    }}
    .speed-control span {{
      color: var(--ink);
      min-width: 32px;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      background: rgba(244,240,232,.76);
      border-top: 1px solid rgba(30,37,41,.16);
    }}
    .metric {{
      min-height: 66px;
      background: rgba(255,255,255,.72);
      border: 1px solid rgba(30,37,41,.14);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .metric span {{
      display: block;
      color: #536269;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .metric strong {{
      display: block;
      font-size: clamp(18px, 2.8vw, 27px);
      line-height: 1.15;
      font-variant-numeric: tabular-nums;
      overflow-wrap: anywhere;
    }}
    .right {{
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(360px, 50vh) minmax(0, 1fr);
      background: #fbfaf5;
    }}
    .head {{
      border-bottom: 1px solid var(--line);
    }}
    .meta {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .chart-wrap {{
      min-height: 0;
      position: relative;
      padding: clamp(12px, 2vw, 20px);
    }}
    #chart {{
      min-height: 420px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .legend {{
      position: absolute;
      left: clamp(30px, 4vw, 52px);
      top: clamp(28px, 4vw, 44px);
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      color: var(--muted);
      font-size: 13px;
      pointer-events: none;
    }}
    .key {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      background: rgba(255,255,255,.78);
      padding: 4px 7px;
      border-radius: 6px;
    }}
    .swatch {{
      width: 24px;
      height: 3px;
      display: inline-block;
      border-radius: 3px;
    }}
    .notes {{
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      display: grid;
      gap: 14px;
      border-top: 1px solid var(--line);
    }}
    .note {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .warning {{
      background: #fff8e8;
      border-color: #d7b36b;
    }}
    .formula {{
      overflow-x: auto;
      padding: 6px 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      min-width: 840px;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e6ece8;
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      color: #3f4f56;
      background: #f1f5f2;
      font-weight: 700;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: #eef3ef;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    @media (max-width: 980px) {{
      body {{ overflow: auto; }}
      main {{ height: auto; min-height: 100vh; overflow: visible; grid-template-columns: 1fr; }}
      .scene {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .right {{ grid-template-rows: auto minmax(400px, 52vh) auto; }}
      .notes {{ overflow: visible; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="scene">
    <div class="topbar">
      <h1>金星 Vega 水平风方差扰动模型</h1>
      <p>用实测水平风方差生成 10 秒保持扰动，计算自由漂浮气球风机的一天发电量。</p>
      <div class="controls">
        <select id="probeSelect" aria-label="选择探测器">
          <option value="VEGA-1">VEGA-1</option>
          <option value="VEGA-2">VEGA-2</option>
        </select>
        <select id="scaleSelect" aria-label="选择扰动倍率">
          <option value="1">s = 1.0</option>
          <option value="0.5">s = 0.5</option>
          <option value="2">s = 2.0</option>
        </select>
        <label class="speed-control">播放速度 <input id="speedRange" type="range" min="0.25" max="4" step="0.25" value="1"><span id="speedValue">1.0x</span></label>
        <a class="link-button" href="venus_variance_gust_principle.html" target="_blank" rel="noreferrer">原理页</a>
        <button id="playButton" type="button" title="播放/暂停" aria-label="播放/暂停">Ⅱ</button>
        <button id="resetButton" type="button" title="回到开始" aria-label="回到开始">↺</button>
      </div>
    </div>
    <canvas id="scene"></canvas>
    <div class="metrics">
      <div class="metric"><span>实时背景水平风速</span><strong id="backgroundValue">0.00 m/s</strong></div>
      <div class="metric"><span>实时相对水平风速</span><strong id="relativeValue">0.00 m/s</strong></div>
      <div class="metric"><span>瞬时功率</span><strong id="powerValue">0.000 W</strong></div>
      <div class="metric"><span>累计发电量</span><strong id="energyValue">0.000 Wh</strong></div>
    </div>
  </section>
  <section class="right">
    <div class="head">
      <h1 id="chartTitle">VEGA-1，s = 1.0</h1>
      <div class="meta">
        <span id="durationText">时长 24 h</span>
        <span id="meanRelText">平均相对风速</span>
        <span id="energyText">总发电量</span>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="chart"></canvas>
      <div class="legend">
        <span class="key"><i class="swatch" style="background: var(--teal)"></i>累计发电量 Wh</span>
        <span class="key"><i class="swatch" style="background: var(--rose)"></i>瞬时功率 W</span>
      </div>
    </div>
    <div class="notes">
      <section class="note warning">
        <h2>为什么不再用水平风线性插值</h2>
        <p>RDR 原始表约每 75.4 s 一行，但有效水平风反演点只有几十个，典型间隔约 30 min。把这些点直接线性插到 1 s，会把 10 s 到数分钟尺度的阵风和湍流滤掉，导致自由气球相对风速被严重低估。</p>
        <p>当前主模型改为：用这些有效水平风点估计扰动方差，再每 10 s 生成一次固定扰动。这样不能证明真实 10 s 阵风一定如此，但比把稀疏点平滑插值更符合“用方差给扰动量”的估计思路。</p>
      </section>
      <section class="note">
        <h2>sigma 与 10 秒扰动怎么来</h2>
        <p>这里的 sigma 不是额外实测传感器，而是由有效水平风样本统计出来的扰动尺度。先把每个有效水平风点写成二维矢量 \(\mathbf{{U}}_i=[u_i,v_i]^T\)，计算均值 \(\bar{{\mathbf{{U}}}}\) 和协方差矩阵 \(\Sigma\)。</p>
        <div class="formula">
        \[
        \bar{{\mathbf{{U}}}}=\frac{{1}}{{N}}\sum_{{i=1}}^{{N}}\mathbf{{U}}_i,\qquad
        \Sigma=\frac{{1}}{{N-1}}\sum_{{i=1}}^{{N}}(\mathbf{{U}}_i-\bar{{\mathbf{{U}}}})(\mathbf{{U}}_i-\bar{{\mathbf{{U}}}})^T
        \]
        \[
        \delta\mathbf{{U}}_k\sim\mathcal{{N}}(\mathbf{{0}},s^2\Sigma),\qquad
        \mathbf{{U}}_{{air}}(t)=\bar{{\mathbf{{U}}}}+\delta\mathbf{{U}}_k,\quad
        k=\left\lfloor\frac{{t}}{{10\ \mathrm{{s}}}}\right\rfloor
        \]
        </div>
        <p>\(s\) 是网页下拉框里的扰动倍率：\(s=1.0\) 表示按实测水平风方差本身生成扰动，\(s=0.5\) 是保守扰动，\(s=2.0\) 是偏强扰动。每 10 s 抽一次 \(\delta\mathbf{{U}}_k\)，这 10 s 内保持不变；为了可复现，随机种子固定为 20260905。</p>
      </section>
      <section class="note">
        <h2>垂直风是否纳入</h2>
        <p>本模型暂不纳入垂直风。垂直风数据更多，但自由气球会随浮力、重力和垂直气流上下运动；若风机面向水平来流，垂直风对转子的有效迎风面积、结构受力和姿态控制都不确定。当前将垂直风保留为后续讨论，不进入主发电量。</p>
      </section>
      <section class="note">
        <h2>气球动力学方程</h2>
        <div class="formula">
        \\[
        \\mathbf{{w}}(t)=\\mathbf{{U}}_{{air}}(t)-\\mathbf{{V}}_b(t),\\qquad W(t)=\\|\\mathbf{{w}}(t)\\|
        \\]
        \\[
        m_{{eff}}(t)\\frac{{d\\mathbf{{V}}_b}}{{dt}}=
        \\frac{{1}}{{2}}\\rho(t)S_{{eff}}W(t)\\mathbf{{w}}(t)
        \\]
        \\[
        m_{{eff}}=m_{{dry}}+\\frac{{1}}{{2}}\\rho V_{{balloon}},\\qquad
        S_{{eff}}=C_{{D,b}}A_b+C_TA_t
        \\]
        \\[
        P_e(t)=\\frac{{1}}{{2}}\\rho(t)A_tC_p\\eta W(t)^3,\qquad
        E(t)=\\int_0^t P_e(\\tau)d\\tau
        \\]
        </div>
        <p>这个方程来自牛顿第二定律加二次阻力模型。二次阻力可理解为动压 \(q=\frac{{1}}{{2}}\rho W^2\) 乘以参考面积和阻力系数；矢量形式中 \(W\mathbf{{w}}\) 的方向沿相对来流，大小为 \(W^2\)。</p>
        <p>变量：\\(\\mathbf{{U}}_{{air}}\\) 是空气水平风，\\(\\mathbf{{V}}_b\\) 是气球水平速度，\\(\\mathbf{{w}}\\) 是风机相对来流，\\(\\rho\\) 是大气密度，\\(A_t\\) 是风机扫掠面积，\\(C_p\\) 是功率系数，\\(\\eta\\) 是发电效率。</p>
      </section>
      <section class="note">
        <h2>参数来源与性质</h2>
        <p>\(D\) 和 \(m_{{dry}}\) 有 VEGA 气球公开资料依据，\(\rho(t)\)、\(\bar{{\mathbf U}}\)、\(\Sigma\) 来自本地 RDR 数据；\(C_{{D,b}}\)、\(C_T\)、\(A_t\)、\(C_p\)、\(\eta\) 是工程估计或设计假设，不是 VEGA 风洞/飞行标定值。</p>
        <div class="table-wrap">{html_table(params, digits=4)}</div>
      </section>
      <section class="note">
        <h2>结果摘要</h2>
        <div class="table-wrap">{html_table(summary, digits=6)}</div>
      </section>
      <section class="note">
        <h2>数据文件统计</h2>
        <div class="table-wrap">{html_table(file_stats.round(4))}</div>
      </section>
      <section class="note">
        <h2>水平风方差统计</h2>
        <div class="table-wrap">{html_table(wind_stats.round(4))}</div>
      </section>
    </div>
  </section>
</main>
<script>
const rawData = __DATA_JSON__;
const summary = __SUMMARY_JSON__;
const byKey = rawData.reduce((acc, row) => {{
  const key = `${{row.probe}}|${{row.sigma_scale}}`;
  if (!acc[key]) acc[key] = [];
  acc[key].push(row);
  return acc;
}}, {{}});
const $ = id => document.getElementById(id);
const scene = $("scene");
const sceneCtx = scene.getContext("2d");
const chart = $("chart");
const chartCtx = chart.getContext("2d");
let probe = "VEGA-1";
let scale = 1;
let index = 0;
let playing = true;
let lastFrame = performance.now();
let phase = 0;
let playbackSpeed = 1;
let playbackCarry = 0;

function key() {{ return `${{probe}}|${{scale}}`; }}
function rows() {{ return byKey[key()] || []; }}
function row() {{
  const data = rows();
  return data[Math.min(index, data.length - 1)] || data[0];
}}
function activeSummary() {{
  return summary.find(item => item.probe === probe && Number(item.sigma_scale) === Number(scale)) || {{}};
}}
function resizeCanvas(canvas, ctx) {{
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {{
    canvas.width = width;
    canvas.height = height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }}
}}
function fmt(value, digits, unit) {{ return `${{value.toFixed(digits)}} ${{unit}}`; }}
function niceStep(maxValue, targetTicks) {{
  if (!Number.isFinite(maxValue) || maxValue <= 0) return 1;
  const raw = maxValue / targetTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const scaled = raw / pow;
  const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * pow;
}}
function makeTicks(maxValue, targetTicks = 5) {{
  const step = niceStep(maxValue, targetTicks);
  const top = Math.ceil(maxValue / step) * step || step;
  const ticks = [];
  for (let v = 0; v <= top + step * 0.5; v += step) ticks.push(v);
  return {{ticks, top}};
}}
function drawScene(now) {{
  resizeCanvas(scene, sceneCtx);
  const rect = scene.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  const r = row();
  if (!r) return;
  const bg = Math.max(0, r.background_wind_ms);
  const rel = Math.max(0, r.relative_wind_ms);
  phase += 0.007 + Math.min(0.12, rel * 0.006);
  const g = sceneCtx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, "#dfeceb");
  g.addColorStop(0.46, "#efc482");
  g.addColorStop(1, "#9a5d65");
  sceneCtx.fillStyle = g;
  sceneCtx.fillRect(0, 0, w, h);
  for (let i = 0; i < 34; i++) {{
    const y = 22 + ((i * 31) % Math.max(80, h - 40));
    const x = ((i * 91 + now * (0.02 + bg * 0.0007)) % (w + 180)) - 120;
    const len = 38 + Math.min(78, bg * 0.8) + 12 * Math.sin(phase + i);
    sceneCtx.strokeStyle = i % 3 === 0 ? "rgba(255,255,255,.52)" : "rgba(43,77,78,.24)";
    sceneCtx.lineWidth = i % 3 === 0 ? 2 : 1.2;
    sceneCtx.beginPath();
    sceneCtx.moveTo(x, y);
    sceneCtx.bezierCurveTo(x + len * 0.33, y - 5, x + len * 0.66, y + 5, x + len, y);
    sceneCtx.stroke();
  }}
  const x = w * 0.47 + Math.sin(phase * 0.7) * (8 + Math.min(18, rel * 1.2));
  const y = h * 0.42 + Math.sin(phase * 1.25) * (5 + Math.min(8, rel * 0.5));
  sceneCtx.lineWidth = 3;
  sceneCtx.strokeStyle = "#253238";
  sceneCtx.fillStyle = "#f4d87d";
  sceneCtx.beginPath();
  sceneCtx.ellipse(x, y, 72, 56, Math.sin(phase) * 0.03, 0, Math.PI * 2);
  sceneCtx.fill();
  sceneCtx.stroke();
  sceneCtx.fillStyle = "rgba(255,255,255,.32)";
  sceneCtx.beginPath();
  sceneCtx.ellipse(x - 24, y - 16, 18, 28, -0.45, 0, Math.PI * 2);
  sceneCtx.fill();
  const gondolaY = y + 106;
  sceneCtx.strokeStyle = "#27343a";
  sceneCtx.lineWidth = 2.2;
  sceneCtx.beginPath();
  sceneCtx.moveTo(x - 34, y + 50);
  sceneCtx.lineTo(x - 22, gondolaY);
  sceneCtx.moveTo(x + 34, y + 50);
  sceneCtx.lineTo(x + 22, gondolaY);
  sceneCtx.stroke();
  sceneCtx.fillStyle = "#27343a";
  sceneCtx.fillRect(x - 38, gondolaY, 76, 26);
  const tx = x + 100;
  const ty = gondolaY + 13;
  sceneCtx.strokeStyle = "#202629";
  sceneCtx.lineWidth = 3.3;
  sceneCtx.beginPath();
  sceneCtx.moveTo(x + 38, ty);
  sceneCtx.lineTo(tx - 22, ty);
  sceneCtx.stroke();
  sceneCtx.beginPath();
  sceneCtx.arc(tx, ty, 17, 0, Math.PI * 2);
  sceneCtx.stroke();
  for (let k = 0; k < 3; k++) {{
    const a = phase * (3.0 + Math.min(4, rel * 0.35)) + k * Math.PI * 2 / 3;
    sceneCtx.beginPath();
    sceneCtx.moveTo(tx, ty);
    sceneCtx.lineTo(tx + 46 * Math.cos(a), ty + 46 * Math.sin(a));
    sceneCtx.stroke();
  }}
  sceneCtx.fillStyle = "rgba(30,37,41,.72)";
  sceneCtx.font = "13px Microsoft YaHei, Segoe UI, sans-serif";
  sceneCtx.fillText(`${{probe}}  s=${{scale.toFixed(1)}}  t=${{r.time_h.toFixed(2)}} h`, 22, h - 24);
}}
function drawChart() {{
  resizeCanvas(chart, chartCtx);
  const rect = chart.getBoundingClientRect();
  const w = rect.width;
  const h = rect.height;
  const data = rows();
  if (!data.length) return;
  chartCtx.clearRect(0, 0, w, h);
  chartCtx.fillStyle = "#fff";
  chartCtx.fillRect(0, 0, w, h);
  const pad = {{left: 76, right: 78, top: 58, bottom: 54}};
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const maxEnergy = Math.max(...data.map(d => Math.max(0, d.cumulative_energy_wh)));
  const maxPower = Math.max(...data.map(d => Math.max(0, d.power_w)));
  const energyScale = makeTicks(maxEnergy, 5);
  const powerScale = makeTicks(maxPower, 5);
  const x = t => pad.left + (t / 24) * plotW;
  const yEnergy = v => pad.top + plotH - (v / energyScale.top) * plotH;
  const yPower = v => pad.top + plotH - (v / powerScale.top) * plotH;
  chartCtx.strokeStyle = "#d8dfdb";
  chartCtx.lineWidth = 1;
  chartCtx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";
  chartCtx.textBaseline = "middle";
  energyScale.ticks.forEach(tick => {{
    const yy = yEnergy(tick);
    chartCtx.beginPath();
    chartCtx.moveTo(pad.left, yy);
    chartCtx.lineTo(w - pad.right, yy);
    chartCtx.stroke();
    chartCtx.fillStyle = "#2f7d77";
    chartCtx.textAlign = "right";
    chartCtx.fillText(tick.toPrecision(tick < 0.01 ? 2 : 3), pad.left - 10, yy);
  }});
  powerScale.ticks.forEach(tick => {{
    const yy = yPower(tick);
    chartCtx.fillStyle = "#b23a48";
    chartCtx.textAlign = "left";
    chartCtx.fillText(tick.toPrecision(tick < 0.01 ? 2 : 3), w - pad.right + 10, yy);
  }});
  chartCtx.strokeStyle = "#87938f";
  chartCtx.beginPath();
  chartCtx.moveTo(pad.left, pad.top);
  chartCtx.lineTo(pad.left, pad.top + plotH);
  chartCtx.lineTo(pad.left + plotW, pad.top + plotH);
  chartCtx.lineTo(pad.left + plotW, pad.top);
  chartCtx.stroke();
  chartCtx.fillStyle = "#5d6b72";
  chartCtx.textAlign = "center";
  chartCtx.textBaseline = "top";
  for (let hr = 0; hr <= 24; hr += 4) {{
    const xx = x(hr);
    chartCtx.beginPath();
    chartCtx.moveTo(xx, pad.top + plotH);
    chartCtx.lineTo(xx, pad.top + plotH + 6);
    chartCtx.stroke();
    chartCtx.fillText(String(hr), xx, pad.top + plotH + 12);
  }}
  chartCtx.fillText("时间 (h)", pad.left + plotW / 2, h - 24);
  chartCtx.save();
  chartCtx.translate(22, pad.top + plotH / 2);
  chartCtx.rotate(-Math.PI / 2);
  chartCtx.fillStyle = "#2f7d77";
  chartCtx.textAlign = "center";
  chartCtx.fillText("累计发电量 (Wh)", 0, 0);
  chartCtx.restore();
  chartCtx.save();
  chartCtx.translate(w - 22, pad.top + plotH / 2);
  chartCtx.rotate(Math.PI / 2);
  chartCtx.fillStyle = "#b23a48";
  chartCtx.textAlign = "center";
  chartCtx.fillText("瞬时功率 (W)", 0, 0);
  chartCtx.restore();
  function line(getY, color, width) {{
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = width;
    chartCtx.beginPath();
    data.forEach((d, i) => {{
      const xx = x(d.time_h);
      const yy = getY(d);
      if (i === 0) chartCtx.moveTo(xx, yy);
      else chartCtx.lineTo(xx, yy);
    }});
    chartCtx.stroke();
  }}
  line(d => yEnergy(Math.max(0, d.cumulative_energy_wh)), "#2f7d77", 2.5);
  line(d => yPower(Math.max(0, d.power_w)), "#b23a48", 1.5);
  const r = row();
  const cx = x(r.time_h);
  chartCtx.strokeStyle = "rgba(30,37,41,.28)";
  chartCtx.setLineDash([5, 5]);
  chartCtx.beginPath();
  chartCtx.moveTo(cx, pad.top);
  chartCtx.lineTo(cx, pad.top + plotH);
  chartCtx.stroke();
  chartCtx.setLineDash([]);
}}
function updateText() {{
  const r = row();
  const s = activeSummary();
  if (!r) return;
  const power = Math.max(0, r.power_w);
  const energy = Math.max(0, r.cumulative_energy_wh);
  $("backgroundValue").textContent = fmt(Math.max(0, r.background_wind_ms), 2, "m/s");
  $("relativeValue").textContent = fmt(Math.max(0, r.relative_wind_ms), 3, "m/s");
  $("powerValue").textContent = fmt(power, power < 0.1 ? 4 : 2, "W");
  $("energyValue").textContent = fmt(energy, energy < 0.01 ? 6 : 3, "Wh");
  $("chartTitle").textContent = `${{probe}}，s = ${{scale.toFixed(1)}}`;
  $("durationText").textContent = `时长 ${{s.duration_h.toFixed(1)}} h`;
  $("meanRelText").textContent = `平均相对风速 ${{s.mean_relative_wind_ms.toFixed(3)}} m/s`;
  $("energyText").textContent = `总发电量 ${{s.energy_wh_per_0p5m2.toFixed(4)}} Wh`;
}}
function frame(now) {{
  const data = rows();
  const elapsed = now - lastFrame;
  lastFrame = now;
  if (playing && data.length) {{
    playbackCarry += (elapsed / 42) * playbackSpeed;
    const step = Math.floor(playbackCarry);
    if (step > 0) {{
      index = (index + step) % data.length;
      playbackCarry -= step;
    }}
  }}
  updateText();
  drawScene(now);
  drawChart();
  requestAnimationFrame(frame);
}}
$("probeSelect").addEventListener("change", event => {{ probe = event.target.value; index = 0; }});
$("scaleSelect").addEventListener("change", event => {{ scale = Number(event.target.value); index = 0; }});
$("speedRange").addEventListener("input", event => {{
  playbackSpeed = Number(event.target.value);
  $("speedValue").textContent = `${{playbackSpeed.toFixed(2).replace(/\\.00$/, ".0")}}x`;
}});
$("playButton").addEventListener("click", () => {{
  playing = !playing;
  $("playButton").textContent = playing ? "Ⅱ" : "▶";
}});
$("resetButton").addEventListener("click", () => {{ index = 0; }});
window.addEventListener("resize", () => {{ drawScene(performance.now()); drawChart(); }});
requestAnimationFrame(frame);
</script>
</body>
</html>"""
    html = html.replace("__DATA_JSON__", json.dumps(records, ensure_ascii=False))
    html = html.replace("__SUMMARY_JSON__", json.dumps(summary.to_dict(orient="records"), ensure_ascii=False))
    out = ROOT / "venus_variance_gust_model.html"
    out.write_text(html, encoding="utf-8")
    return out


def make_principle_html(file_stats: pd.DataFrame, wind_stats: pd.DataFrame, params: pd.DataFrame, summary: pd.DataFrame) -> Path:
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>金星 Vega 风力发电模型原理</title>
  <script>
    window.MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']]}}}};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {{
      --bg: #f4f0e8;
      --ink: #1e2529;
      --muted: #5d6b72;
      --line: #cdd5d1;
      --panel: #fffdf8;
      --teal: #2f7d77;
      --rose: #b23a48;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      line-height: 1.65;
    }}
    header {{
      padding: 26px clamp(18px, 5vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfaf5;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 18px clamp(16px, 3vw, 28px) 36px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 4vw, 42px);
      line-height: 1.18;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 21px;
      letter-spacing: 0;
    }}
    p, li {{
      color: #536269;
    }}
    a {{
      color: var(--teal);
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: clamp(14px, 2.4vw, 22px);
      margin: 16px 0;
    }}
    .formula {{
      overflow-x: auto;
      padding: 8px 0;
      font-size: 1.08rem;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table {{
      width: 100%;
      min-width: 860px;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid #e6ece8;
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #3f4f56;
      background: #f1f5f2;
      font-weight: 700;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      background: #eef3ef;
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
<header>
  <h1>金星 Vega 风力发电模型原理</h1>
  <p>这里展开说明 10 秒水平风方差扰动模型、气球动力学积分、变量物理意义和参数来源。</p>
  <p><a href="venus_variance_gust_model.html">返回主网站</a></p>
</header>
<main>
  <section class="section">
    <h2>1. 扰动风如何生成</h2>
    <p>有效水平风来自 RDR 文件的 <code>zonal_wind_u_ms</code> 与 <code>meridional_wind_v_ms</code>。每个有效水平风点写成二维矢量：</p>
    <div class="formula">
    \\[
    \\mathbf{{U}}_i=\\begin{{bmatrix}}u_i\\\\v_i\\end{{bmatrix}}
    \\]
    </div>
    <p>先计算样本均值和协方差：</p>
    <div class="formula">
    \\[
    \\bar{{\\mathbf{{U}}}}=\\frac{{1}}{{N}}\\sum_{{i=1}}^N\\mathbf{{U}}_i
    \\]
    \\[
    \\Sigma=\\frac{{1}}{{N-1}}\\sum_{{i=1}}^N
    (\\mathbf{{U}}_i-\\bar{{\\mathbf{{U}}}})
    (\\mathbf{{U}}_i-\\bar{{\\mathbf{{U}}}})^T
    \\]
    </div>
    <p>然后每 10 s 抽一次二维扰动，10 s 内保持不变：</p>
    <div class="formula">
    \\[
    \\delta\\mathbf{{U}}_k\\sim\\mathcal{{N}}(\\mathbf{{0}},s^2\\Sigma)
    \\]
    \\[
    \\mathbf{{U}}_{{air}}(t)=\\bar{{\\mathbf{{U}}}}+\\delta\\mathbf{{U}}_k,
    \\qquad
    k=\\left\\lfloor\\frac{{t}}{{10\\ \\mathrm{{s}}}}\\right\\rfloor
    \\]
    </div>
    <p>\(s\) 是扰动倍率：\(s=1.0\) 表示使用实测水平风方差本身；\(s=0.5\) 为保守扰动；\(s=2.0\) 为偏强扰动。随机种子固定为 20260905，因此结果可复现。</p>
  </section>

  <section class="section">
    <h2>2. 气球动力学方程从哪里来</h2>
    <p>模型的核心不是直接套动量守恒，而是牛顿第二定律加二次阻力模型。相对来流为：</p>
    <div class="formula">
    \\[
    \\mathbf{{w}}(t)=\\mathbf{{U}}_{{air}}(t)-\\mathbf{{V}}_b(t),
    \\qquad
    W(t)=\\|\\mathbf{{w}}(t)\\|
    \\]
    </div>
    <p>二次阻力可写为：</p>
    <div class="formula">
    \\[
    \\mathbf{{F}}_D=\\frac{{1}}{{2}}\\rho C_D A W\\mathbf{{w}}
    \\]
    </div>
    <p>这等价于常见标量形式 \(F_D=\frac{{1}}{{2}}\rho C_D A W^2\)。其中 \(W\mathbf{{w}}\) 的方向沿相对来流，大小为 \(W^2\)。所以气球水平运动方程写成：</p>
    <div class="formula">
    \\[
    m_{{eff}}(t)\\frac{{d\\mathbf{{V}}_b}}{{dt}}=
    \\frac{{1}}{{2}}\\rho(t)S_{{eff}}W(t)\\mathbf{{w}}(t)
    \\]
    </div>
    <p>有效质量和等效阻力面积为：</p>
    <div class="formula">
    \\[
    m_{{eff}}(t)=m_{{dry}}+\\frac{{1}}{{2}}\\rho(t)V_{{balloon}},
    \\qquad
    V_{{balloon}}=\\frac{{\\pi}}{{6}}D^3
    \\]
    \\[
    S_{{eff}}=C_{{D,b}}A_b+C_TA_t,
    \\qquad
    A_b=\\pi\\left(\\frac{{D}}{{2}}\\right)^2
    \\]
    </div>
  </section>

  <section class="section">
    <h2>3. 数值积分和功率</h2>
    <p>内部积分步长为 \(\Delta t=1\,\mathrm{{s}}\)，用显式欧拉法更新气球水平速度：</p>
    <div class="formula">
    \\[
    \\mathbf{{V}}_{{b,n+1}}=\\mathbf{{V}}_{{b,n}}+
    \\frac{{\\rho_nS_{{eff}}W_n}}{{2m_{{eff,n}}}}
    \\mathbf{{w}}_n\\Delta t
    \\]
    </div>
    <p>风机电功率和累计发电量为：</p>
    <div class="formula">
    \\[
    P_e(t)=\\frac{{1}}{{2}}\\rho(t)A_tC_p\\eta W(t)^3
    \\]
    \\[
    E_n=E_{{n-1}}+\\frac{{P_{{e,n-1}}+P_{{e,n}}}}{{2}}(t_n-t_{{n-1}})
    \\]
    </div>
    <p>因为 \(W\) 是速度大小，所以 \(P_e\ge 0\)。如果主图里曾看到负的瞬时功率，那是显示层问题，不是物理模型输出。</p>
  </section>

  <section class="section">
    <h2>4. 参数来源</h2>
    <div class="table-wrap">{html_table(params, digits=4)}</div>
  </section>

  <section class="section">
    <h2>5. 数据统计</h2>
    <div class="table-wrap">{html_table(file_stats.round(4))}</div>
  </section>

  <section class="section">
    <h2>6. 水平风统计</h2>
    <div class="table-wrap">{html_table(wind_stats.round(4))}</div>
  </section>

  <section class="section">
    <h2>7. 结果摘要</h2>
    <div class="table-wrap">{html_table(summary, digits=6)}</div>
  </section>
</main>
</body>
</html>"""
    out = ROOT / "venus_variance_gust_principle.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    file_stats, wind_stats = probe_statistics()
    params = parameter_provenance()
    frames = []
    for info in PROBES:
        for scale in SIGMA_SCALES:
            seed = RANDOM_SEED + (1 if info["probe"] == "VEGA-1" else 2) * 100 + int(scale * 10)
            frames.append(simulate_probe(info, scale, seed))
    results = pd.concat(frames, ignore_index=True)
    summary = summarize_results(results)

    results_path = ROOT / "venus_variance_gust_results.csv"
    summary_path = ROOT / "venus_variance_gust_summary.csv"
    file_stats_path = ROOT / "venus_variance_gust_data_stats.csv"
    wind_stats_path = ROOT / "venus_variance_gust_wind_stats.csv"
    params_path = ROOT / "venus_variance_gust_parameter_sources.csv"
    sampled_results = results[np.isclose(results["time_s"] % GUST_HOLD_S, 0.0)].copy()
    sampled_results.to_csv(results_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    file_stats.to_csv(file_stats_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    wind_stats.to_csv(wind_stats_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    params.to_csv(params_path, index=False, encoding="utf-8-sig")
    notebook = make_notebook(file_stats, wind_stats, params, summary)
    html = make_html(file_stats, wind_stats, params, summary, sampled_results)
    principle_html = make_principle_html(file_stats, wind_stats, params, summary)

    print(f"wrote {results_path.name}")
    print(f"wrote {summary_path.name}")
    print(f"wrote {file_stats_path.name}")
    print(f"wrote {wind_stats_path.name}")
    print(f"wrote {params_path.name}")
    print(f"wrote {notebook.name}")
    print(f"wrote {html.name}")
    print(f"wrote {principle_html.name}")
    print(summary.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
