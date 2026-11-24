"""
Figure 3: two-view globe maps and country complementarity trends, with widths set by 2025 wind-plus-solar generation.
Run directly with the required geographic plotting dependencies installed; preserve original figures and data.
"""
from pathlib import Path
import sys
import site
import importlib.util
import json
import hashlib

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
site.addsitedir(str(ROOT/'.venv-earth-plot/Lib/site-packages'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
import cartopy.crs as ccrs
from shapely.geometry import shape
from shapely import make_valid

spec = importlib.util.spec_from_file_location('earth_original', HERE/'plot_figure_3_earth.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
sys.path.insert(0, str(original.MAP_SOURCE))
from calculate_wind_solar_kendall_maps import country_code, SMALL_PLACE_POINTS

FIGSIZE = (19.2, 10.8)
# ===== Adjustable legend settings for panel b =====
LEGEND_LEFT = .815             # Horizontal legend position as a canvas-width fraction; smaller values move it left.
LEGEND_COLUMN_GAP = .030       # Column-center spacing as a canvas-width fraction; smaller values bring columns closer.
LEGEND_ROWS = 6                # Put six entries in the left column and the remaining five in the right.
LEGEND_LINE_HEIGHT_PT = 24     # Legend sample-line length in pt.
LEGEND_TEXT_GAP_PT = 6         # Vertical offset for country names below the line, on the same centerline, in pt.
LEGEND_ITEM_GAP_PT = 24        # Uniform gap from each country name to the next divider, in pt.
LEGEND_FONT_SIZE = 13          # Country-name font size.
OUTPUT = HERE/'图3_地球_20261116.png'
METRICS = [('solar_twh', 'Solar'), ('wind_twh', 'Wind')]
YEARS = [2022, 2023, 2024, 2025]
COUNTRY_TREND_FILE = original.SOURCE/'各国风光互补系数_2022-2025.csv'
QUALITY_FILE = original.MAP_SOURCE/'Electricity_Maps_国家数据等级_2022-2025.csv'
US_TIER_SHARE_FILE = original.SOURCE/'独立核查'/'美国加拿大子区A_C级发电权重_2022-2025.csv'
COUNTRIES = [
    ('IN', 'India'), ('JP', 'Japan'), ('TR', 'Turkey'),
    ('DE', 'Germany'), ('ES', 'Spain'), ('GB', 'Great Britain'),
    ('ZA', 'South Africa'), ('US', 'United States'), ('MX', 'Mexico'),
    ('BR', 'Brazil'), ('AU', 'Australia'),
]
# Use square, circle, and cross markers within each continent, repeating the same set across continents.
MARKERS = ['s', 'o', 'X', 's', 'o', 'X', 's', 's', 'o', 's', 's']
# Opposite hemisphere centres put the dividing meridians at 20 W and 160 E.
# Lambert equal-area hemispheres reduce the severe edge compression of orthographic views.
VIEWS = [(-110, 0, 'Western hemisphere'), (70, 0, 'Eastern hemisphere')]


def load_totals():
    annual = pd.read_csv(original.MAP_TOTALS_FILE, keep_default_na=False, na_values=[''])
    assert annual.country_code.is_unique and 'NA' in annual.country_code.values
    # Validate the cached totals against available hourly data without rewriting files.
    hourly = pd.read_csv(original.MAP_SOURCE/'global_2025_hourly_wind_solar_by_country.csv')
    expected = pd.date_range('2025-01-01', '2026-01-01', freq='h', inclusive='left', tz='UTC')
    assert pd.DatetimeIndex(pd.to_datetime(hourly.datetime_utc, utc=True)).equals(expected)
    for row in annual.itertuples():
        for metric, _ in METRICS:
            kind = metric.removesuffix('_twh')
            series = hourly[f'{row.country_name} [{row.country_code}]_{kind}_mw']
            np.testing.assert_allclose(series.sum(min_count=1)/1e6, getattr(row, metric), equal_nan=True)
            assert int(series.notna().sum()) == getattr(row, kind+'_hours')
    return annual.set_index('country_code').to_dict('index')


def draw_globes(fig, indexed):
    geo = json.loads((original.MAP_SOURCE/'ne_110m_admin_0_countries.geojson').read_text(encoding='utf-8'))
    geometries = [(country_code(f), make_valid(shape(f['geometry']))) for f in geo['features']]
    cmap = plt.get_cmap('YlGnBu')
    norm = SymLogNorm(linthresh=.01, linscale=.8, vmin=0, vmax=3000, base=10)
    source_crs = ccrs.PlateCarree()
    radius = 6371000.0
    globe = ccrs.Globe(semimajor_axis=radius, semiminor_axis=radius, ellipse=None)
    hemisphere_limit = np.sqrt(2) * radius
    axes = []
    for group, (metric, label) in enumerate(METRICS):
        for view, (lon, lat, view_label) in enumerate(VIEWS):
            left, bottom = [.035, .325][view], [.57, .12][group]
            projection = ccrs.LambertAzimuthalEqualArea(lon, lat, globe=globe)
            ax = fig.add_axes([left, bottom, .235, .235*FIGSIZE[0]/FIGSIZE[1]],
                              projection=projection)
            ax.set_xlim(-hemisphere_limit, hemisphere_limit)
            ax.set_ylim(-hemisphere_limit, hemisphere_limit)
            theta = np.linspace(0, 2*np.pi, 721)
            circle = np.column_stack([.5+.5*np.cos(theta), .5+.5*np.sin(theta)])
            ax.set_boundary(MplPath(circle), transform=ax.transAxes)
            ax.set_facecolor('#EDF3F5')
            ax.spines['geo'].set_edgecolor('#71808A')
            ax.spines['geo'].set_linewidth(.9)
            def color(code):
                value = indexed.get(code, {}).get(metric, np.nan)
                if not np.isfinite(value):
                    return original.NO_DATA
                assert 0 <= value < norm.vmax
                return cmap(norm(value))
            for code, geometry in geometries:
                ax.add_geometries([geometry], source_crs, facecolor=color(code),
                                  edgecolor='#6A7880', linewidth=.30, zorder=2)
            for code, (x, y) in SMALL_PLACE_POINTS.items():
                if code in indexed:
                    ax.scatter(x, y, transform=source_crs, s=12, facecolor=color(code),
                               edgecolor='#6A7880', linewidth=.35, zorder=4)
            ax.gridlines(crs=source_crs, xlocs=np.arange(-180, 181, 20),
                         ylocs=np.arange(-80, 81, 20), linewidth=.5, color='#70818B', alpha=.25)
            ax.text(.5, -.025, view_label, transform=ax.transAxes,
                    ha='center', va='top', fontsize=13, color='#48545E')
            axes.append(ax)
        fig.text(.297, [.995, .555][group], label, ha='center',
                 va='top', fontsize=23, fontweight='bold')
    cax = fig.add_axes([.10, .042, .36, .014])
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    bar.set_ticks([0,.01,.1,1,10,100,1000,3000], labels=['0','0.01','0.1','1','10','100','1000','3000'])
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=11, length=4)
    bar.ax.xaxis.set_label_position('top')
    bar.set_label('2025 available-data total (TWh)', fontsize=13, labelpad=5)
    bar.outline.set_linewidth(.8)
    fig.legend(handles=[Patch(facecolor=original.NO_DATA, edgecolor='#6A7880', label='No data')],
               loc='center', bbox_to_anchor=(.525, .049), fontsize=12, frameon=False)
    assert len(axes) == 4 and len({id(ax.projection) for ax in axes}) == 4
    return axes


def load_country_trends():
    data = pd.read_csv(COUNTRY_TREND_FILE)
    selected_codes = [code for code, _ in COUNTRIES]
    data = data[data.country_code.isin(selected_codes)].copy()
    assert len(data) == len(COUNTRIES) * len(YEARS)
    assert not data.duplicated(['country_code', 'year']).any()
    assert data.status.eq('valid').all() and data.kendall_tau_b.notna().all()
    assert data.groupby('country_code').year.apply(lambda values: values.tolist() == YEARS).all()

    quality = pd.read_csv(QUALITY_FILE).set_index('国家代码').loc[selected_codes]
    child_tiers = quality['子分区等级构成'].fillna('')
    assert quality['纳入洲际固定81国样本'].eq('是').all()
    assert quality['Electricity Maps级别'].ne('C').all()
    assert ~child_tiers.drop('US').str.contains('C×').any()
    us_share = pd.read_csv(US_TIER_SHARE_FILE)
    us_share = us_share[us_share['国家代码'].eq('US')].sort_values('年份')
    assert us_share['年份'].tolist() == YEARS
    assert us_share['C级风光合计占比_pct'].max() < 1
    return data


def draw_country_trends(ax, data, totals):
    ax.axvspan(-1, 0, color='#ECF5F1', alpha=.80, zorder=0)
    ax.axvspan(0, 1, color='#FBF1E8', alpha=.80, zorder=0)
    ax.axvline(0, color='#71818C', lw=1.0, ls='--')
    energy = np.array([totals[code]['wind_solar_twh'] for code, _ in COUNTRIES], dtype=float)
    assert np.isfinite(energy).all() and (energy > 0).all()
    # Square-root scaling balances generation differences with readability for smaller countries; share widths across years.
    widths = 1.5*(1.2 + 3.8*np.sqrt(energy/energy.max()))
    assert np.all(np.diff(widths[np.argsort(energy)]) >= 0)
    handles, labels, width_records = [], [], []
    for (code, name), marker, total, width in zip(COUNTRIES, MARKERS, energy, widths):
        rows = data[data.country_code.eq(code)].sort_values('year')
        continent = rows.continent.iloc[0]
        color = original.STYLES[continent][0]
        values = rows.kendall_tau_b.to_numpy()
        ax.plot(values, YEARS, color=color, lw=width, marker=marker, ms=6,
                markeredgecolor='white', markeredgewidth=.45, alpha=.70)
        handles.append(Line2D([], [], color=color, lw=width, marker=marker, ms=6,
                              markeredgecolor='white', markeredgewidth=.45))
        labels.append(name)
        width_records.append(dict(country_code=code, country=name, year=2025,
                                  wind_solar_available_twh=total, linewidth_pt=width))
    ax.set(xlim=(-.52, .12), ylim=(2025.35, 2021.65), yticks=YEARS,
           xlabel='Wind–solar association (Kendall $\\tau_b$)', ylabel='Year')
    ax.grid(axis='x', color='#D7DFE4', lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=16, direction='in', top=True, right=True)
    ax.xaxis.label.set_size(18)
    ax.yaxis.label.set_size(18)
    return handles, labels, width_records


def draw_country_legend(fig, trend_ax, handles, labels):
    position = trend_ax.get_position()
    legend = fig.add_axes([LEGEND_LEFT, position.y0, 2*LEGEND_COLUMN_GAP, position.height])
    legend.set(xlim=(0, 1), ylim=(0, 1))
    legend.set_axis_off()
    assert len(handles) == len(labels) == 11
    entries = list(zip(handles, labels))
    entries.append(entries.pop(LEGEND_ROWS))  # Move South Africa to the end of the second column.
    handles, labels = zip(*entries)
    labels = ['U.S.A.' if label == 'United States' else label for label in labels]
    points_to_axes = fig.dpi/72/legend.bbox.height
    line_height = LEGEND_LINE_HEIGHT_PT*points_to_axes
    gap = LEGEND_TEXT_GAP_PT*points_to_axes
    item_gap = LEGEND_ITEM_GAP_PT*points_to_axes
    texts = []
    item_heights = []
    for i, (handle, label) in enumerate(zip(handles, labels)):
        column = i//LEGEND_ROWS
        x = [.25, .75][column]
        text = legend.annotate(label, (x, 0), xytext=(0, -LEGEND_TEXT_GAP_PT),
                               textcoords='offset points', rotation=90, ha='center', va='top',
                               fontsize=LEGEND_FONT_SIZE)
        text_height = text.get_window_extent().height/legend.bbox.height
        item_heights.append(line_height+gap+text_height)
        texts.append(text)
    tops = np.zeros(len(handles))
    for row in range(1, len(handles)-LEGEND_ROWS):
        step = max(item_heights[row-1], item_heights[LEGEND_ROWS+row-1])+item_gap
        tops[row] = tops[LEGEND_ROWS+row] = tops[row-1]-step
    # Place the UK after Spain using its own gap, independent of South Africa's text width.
    tops[LEGEND_ROWS-1] = tops[LEGEND_ROWS-2]-item_heights[LEGEND_ROWS-2]-item_gap
    total_height = max(height-top for height, top in zip(item_heights, tops))
    assert total_height <= 1, 'Country legend exceeds the available height'
    tops += (1+total_height)/2
    for i, (handle, text) in enumerate(zip(handles, texts)):
        column = i//LEGEND_ROWS
        x = [.25, .75][column]
        top = tops[i]
        bottom = top-line_height
        text.xy = (x, bottom)
        legend.plot([x, x, x], [bottom, (top+bottom)/2, top],
                    color=handle.get_color(), lw=handle.get_linewidth(), solid_capstyle='butt',
                    marker=handle.get_marker(), markevery=[1], ms=6,
                    markeredgecolor=handle.get_markeredgecolor(),
                    markeredgewidth=handle.get_markeredgewidth())
    return legend


def main():
    protected = [HERE/'plot_figure_3_earth.py', HERE/'图3_地球_组合.png',
                 original.MAP_TOTALS_FILE, HERE/original.DATA_NAME]
    hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    original.setup()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':'black',
                         'axes.labelcolor':'black', 'axes.edgecolor':'black',
                         'xtick.color':'black', 'ytick.color':'black',
                         'axes.spines.top':True, 'axes.spines.right':True, 'axes.linewidth':1.3})
    fig = plt.figure(figsize=FIGSIZE)
    totals = load_totals()
    globes = draw_globes(fig, totals)
    trend_top = globes[0].get_position().y1
    ax = fig.add_axes([.64, .105, .17, trend_top-.105])
    handles, labels, width_records = draw_country_trends(ax, load_country_trends(), totals)
    draw_country_legend(fig, ax, handles, labels)
    fig.text(.012, .962, 'a', fontsize=30, fontweight='bold')
    fig.text(.615, .962, 'b', fontsize=30, fontweight='bold')
    fig.savefig(OUTPUT, dpi=240, bbox_inches='tight', facecolor='white')
    pd.DataFrame(width_records).to_csv(HERE/'b图_线宽与发电量_20261116.csv', index=False, encoding='utf-8-sig')
    plt.close(fig)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p,digest in hashes.items())
    print('Saved four-globe PNG with 11 verified country trends for 2022-2025.')


if __name__ == '__main__':
    main()
