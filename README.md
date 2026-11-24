# Wind and Solar Power across Earth, Mars and Venus

Selected Python code from an ongoing cross-planetary wind and solar energy research project at Peking University, advised by Jiaqi Li.

This public repository is a **code-only release**. The unpublished manuscript, methods draft, figures, result tables and research datasets are withheld. They have never been committed to this repository's Git history.

## Code layout

| Directory | Analysis |
| --- | --- |
| `mars/` | InSight telemetry collection and solar reconstruction, MCD-based wind generation, and mission-load digitization |
| `venus/` | Vega wind scenarios, Akatsuki/Pioneer atmospheric profiles, and photovoltaic models |
| `earth/electricity_maps/` | Generation-data aggregation and wind/solar Kendall correlation mapping |
| `figures/figure_1_mars/` | Figure 1: Mars |
| `figures/figure_2_venus/` | Figure 2: Venus |
| `figures/figure_3_earth/` | Figure 3: Earth |
| `figures/model_validation/` | Physical-model audit helpers |
| `literature_validation/kendall_reproduction/code/` | Literature comparison and country-level correlation validation |

Repository directories, code filenames, source comments and this documentation are in English. Code references and cross-directory imports use the same English paths. External data basenames and data-column names retain their original spelling; place independently supplied inputs under the corresponding English directories.

The figure launcher is `figures/redraw_all_figures.py`. Its targets are `model_validation/pv_efficiency_audit.py`, `figure_1_mars/plot_figure_1_mars.py`, `figure_2_venus/plot_figure_2_venus_context.py` and `figure_3_earth/plot_figure_3_earth_globes.py`, resolved relative to the launcher.

## Environment and external inputs

Use Python 3.11 and install `requirements.txt`. The code requires independently supplied inputs at its documented relative paths. A fresh clone does not contain the data needed to regenerate the research figures or manuscript results.

- Mars calculations require engineering telemetry, wind arrays, or the extracted MCD NetCDF fields. The wind notebook expects `Data_MCD/mcd_data.nc` and runs from its containing Mars directory.
- Venus calculations require Vega, Akatsuki and Pioneer Venus tables; context mapping also needs the Magellan topography raster.
- Earth analyses require the hourly country-level generation cache, geographic layers and comparison tables.
- Digitization requires its source mission-energy images. Network collectors depend on their original mission-data services.

The figure redraw launcher and model notebooks are research entry points, not standalone demos with bundled sample data. Model assumptions remain under development.

## Validation and attribution

After the English-path migration, all 23 Python files and 18 notebook code cells were syntax-checked. The redraw launcher was exercised with its data-dependent jobs intercepted, confirming all four target scripts exist and resolve correctly. An AST comparison verified that executable changes were limited to path strings and the launcher's English progress message. Both public notebooks remain byte-identical and have no saved outputs. Full scientific analyses have not been rerun from this data-free release.

Mission data and external models retain their original source terms. The separate [Hartwick Mars-Wind-Energy project](https://github.com/vhartwick/Mars-Wind-Energy) is third-party work and is not included as original code here.
