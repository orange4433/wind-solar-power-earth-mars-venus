# Wind and Solar Power across Earth, Mars and Venus

Selected Python code from an ongoing cross-planetary wind and solar energy research project at Peking University, advised by Jiaqi Li.

This public repository is a **code-only release**. The unpublished manuscript, methods draft, figures, result tables and research datasets are withheld. They have never been committed to this repository's Git history.

## Code layout

| Directory | Analysis |
| --- | --- |
| `火星/` | InSight telemetry collection and solar reconstruction, MCD-based wind generation, and mission-load digitization |
| `金星/` | Vega wind scenarios, Akatsuki/Pioneer atmospheric profiles, and photovoltaic models |
| `地球/` | Generation-data aggregation and wind/solar Kendall correlation mapping |
| `文章图片/` | Figure-generation code and physical-model audit helpers |
| `文献验证/Kendall系数复现/code/` | Literature comparison and country-level correlation validation |

Source comments and this documentation are in English. Original relative paths and data-column names are retained so that the scripts remain compatible with the research working archive.

## Environment and external inputs

Use Python 3.11 and install `requirements.txt`. The code requires independently supplied inputs at its documented relative paths. A fresh clone does not contain the data needed to regenerate the research figures or manuscript results.

- Mars calculations require engineering telemetry, wind arrays, or the extracted MCD NetCDF fields. The wind notebook expects `Data_MCD/mcd_data.nc` and runs from its containing Mars directory.
- Venus calculations require Vega, Akatsuki and Pioneer Venus tables; context mapping also needs the Magellan topography raster.
- Earth analyses require the hourly country-level generation cache, geographic layers and comparison tables.
- Digitization requires its source mission-energy images. Network collectors depend on their original mission-data services.

The figure redraw launcher and model notebooks are research entry points, not standalone demos with bundled sample data. Model assumptions remain under development.

## Validation and attribution

All included Python files and notebook code cells were syntax-checked. The public notebooks have no saved outputs. Full scientific analyses have not been rerun from this data-free release.

Mission data and external models retain their original source terms. The separate [Hartwick Mars-Wind-Energy project](https://github.com/vhartwick/Mars-Wind-Energy) is third-party work and is not included as original code here.
