"""Figure 3: three 2025 generation maps above the existing complementarity trends.

Hourly validation plots are generated separately by the hourly-data audit script.
"""
from pathlib import Path
import sys
import shutil
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import SymLogNorm
from matplotlib.cm import ScalarMappable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import setup, save

# ===== Frequently adjusted parameters =====
FIGSIZE = (26, 12.6)
MAP_WIDTH = .28  # Each map is about 7.28 inches wide, about 22% wider than the previous version.
MAP_STEP = .285  # Leave only 0.005 canvas width between adjacent maps.
YEARS = [2022, 2023, 2024, 2025]
OUTPUT = HERE/'图3_地球_组合'
SOURCE = HERE.parents[1]/'地球'/'electricity maps'/'wind_solar_continent_trend_2022_2025'
DATA_NAME = '七大洲风光互补系数_2022-2025_详细.csv'
MAP_SOURCE = SOURCE.parent
MAP_TOTALS_FILE = HERE/'2025各国风光累计发电量_地图用.csv'
MAP_METRICS = [('wind_twh', 'Wind'), ('solar_twh', 'Solar'), ('wind_solar_twh', 'Wind + solar')]
NO_DATA = '#FFFFFF'
# Preserve Kendall tau-b: more negative values indicate stronger inverse variation.
STYLES = {
    'Asia': ('#CD7242','o'), 'Europe': ('#376D9E','s'),
    'Africa': ('#80963C','^'), 'North America': ('#8862A7','D'),
    'South America': ('#C05063','v'), 'Oceania': ('#198D8D','P'),
}


def draw_trend(ax, d):
    ax.axhspan(-.55,0,color='#ECF5F1',alpha=.75,zorder=0)
    ax.axhspan(0,.15,color='#FBF1E8',alpha=.75,zorder=0)
    ax.axhline(0,color='#71818C',lw=1.1,ls='--')
    for name,(color,marker) in STYLES.items():
        rows=d[d.continent.eq(name)].sort_values('year')
        assert rows.year.tolist()==YEARS and rows.country_count.nunique()==1
        y=rows.kendall_tau_b.to_numpy()
        assert np.isfinite(y).all() and (np.abs(y)<=1).all()
        ax.plot(YEARS,y,color=color,lw=2.4,marker=marker,ms=8,label=name)
    # Label line ends directly to distinguish nearby European and African trends.
    label_y={'Asia':.083,'Europe':-.205,'Africa':-.170,'North America':-.245,
             'South America':-.481,'Oceania':-.283}
    for name,(color,_) in STYLES.items():
        row=d[(d.continent==name)&(d.year==2025)].iloc[0]
        value=float(row.kendall_tau_b)
        ax.annotate(f'{name}  {value:+.3f}',xy=(2025,value),xytext=(2025.17,label_y[name]),
                    va='center',color=color,fontsize=16,
                    arrowprops=dict(arrowstyle='-',color=color,lw=.8))
    ax.set(xlim=(2021.85,2026.04),ylim=(-.55,.15),xticks=YEARS,
           xlabel='Year',ylabel='Wind–solar association (Kendall $\\tau_b$)')
    ax.grid(axis='y',color='#D7DFE4',lw=.7)
    ax.set_axisbelow(True)
    ax.text(2021.94,.126,'Positive: co-variation',fontsize=14,color='#9A6844')
    ax.text(2021.94,-.525,'More negative: stronger temporal complementarity',fontsize=14,color='#4F7665')
    handles,labels=ax.get_legend_handles_labels()
    handles.append(Line2D([],[],ls='None',marker='x',color='#8F969C',ms=8))
    labels.append('Antarctica: no data')
    ax.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.18),ncols=7,
              fontsize=14,columnspacing=1.2,handlelength=1.6,frameon=False)
    ax.tick_params(labelsize=18, direction='in', top=True, right=True)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)


def available_totals():
    """Sum available observations without coverage filtering or extrapolation to a complete year."""
    hourly = pd.read_csv(MAP_SOURCE/'global_2025_hourly_wind_solar_by_country.csv')
    times = pd.to_datetime(hourly.datetime_utc, utc=True)
    expected = pd.date_range('2025-01-01','2026-01-01',freq='h',inclusive='left',tz='UTC')
    assert pd.DatetimeIndex(times).equals(expected)
    rows = []
    for column in hourly.columns:
        if not column.endswith('_wind_mw'):
            continue
        name, code = re.fullmatch(r'(.+) \[([^]]+)]_wind_mw', column).groups()
        solar_column = column.removesuffix('_wind_mw') + '_solar_mw'
        pair = hourly[[column, solar_column]]
        # min_count=1 retains NaN for wholly missing series and sums available records otherwise.
        totals = pair.sum(min_count=1)/1_000_000  # MW × 1 h → MWh → TWh
        rows.append(dict(country_code=code, country_name=name, expected_hours=len(hourly),
                         wind_hours=int(pair[column].notna().sum()),
                         solar_hours=int(pair[solar_column].notna().sum()),
                         wind_twh=totals.iloc[0], solar_twh=totals.iloc[1],
                         wind_solar_twh=totals.sum(min_count=1)))
    annual = pd.DataFrame(rows)
    assert annual.country_code.is_unique and 'NA' in annual.country_code.values
    assert annual.loc[annual.wind_hours.eq(0),'wind_twh'].isna().all()
    assert annual.loc[annual.solar_hours.eq(0),'solar_twh'].isna().all()
    np.testing.assert_allclose(annual.wind_solar_twh,
                               annual[['wind_twh','solar_twh']].sum(axis=1,min_count=1),equal_nan=True)
    annual.to_csv(MAP_TOTALS_FILE,index=False,encoding='utf-8-sig')
    return annual


def draw_generation_maps(fig):
    metrics = [name for name, _ in MAP_METRICS]
    annual = available_totals()
    indexed = annual.set_index('country_code').to_dict('index')
    geo = json.loads((MAP_SOURCE/'ne_110m_admin_0_countries.geojson').read_text(encoding='utf-8'))
    sys.path.insert(0, str(MAP_SOURCE))
    from calculate_wind_solar_kendall_maps import country_code, polygons, SMALL_PLACE_POINTS
    # Use a linear segment from 0 to 0.01 TWh and a logarithmic segment above it.
    cmap = plt.get_cmap('YlGnBu')
    norm = SymLogNorm(linthresh=.01, linscale=.8, vmin=0, vmax=3000, base=10)
    assert annual[metrics].max().max() < norm.vmax
    axes = []
    for i, (metric, label) in enumerate(MAP_METRICS):
        ax = fig.add_axes([.06 + i*MAP_STEP, .61, MAP_WIDTH, .34])
        def color(code):
            row = indexed.get(code)
            if row is None or not np.isfinite(row[metric]):
                return NO_DATA
            value = row[metric]
            assert np.isfinite(value) and value >= 0
            return cmap(norm(value))
        shapes, colors = [], []
        for feature in geo['features']:
            if feature['properties'].get('ADMIN') == 'Antarctica':
                continue
            for ring in polygons(feature['geometry']):
                shapes.append(Polygon(np.asarray(ring), closed=True))
                colors.append(color(country_code(feature)))
        ax.add_collection(PatchCollection(shapes, facecolor=colors, edgecolor='#68747D', linewidth=.25))
        for code, (lon, lat) in SMALL_PLACE_POINTS.items():
            if code in indexed:
                ax.scatter(lon,lat,s=10,facecolor=color(code),edgecolor='#68747D',linewidth=.3,zorder=3)
        ax.set(xlim=(-180,180),ylim=(-60,85),aspect='equal')
        ax.set_axis_off()
        # Type labels belong to the map legend rather than separate panel titles.
        ax.text(.5,-.08,label,transform=ax.transAxes,fontsize=20,fontweight='bold',ha='center')
        axes.append(ax)
    bar_ax = fig.add_axes([.93, .65, .011, .27])
    bar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=bar_ax,orientation='vertical')
    bar.set_ticks([0,.01,.1,1,10,100,1000,3000], labels=['0','0.01','0.1','1','10','100','1000','3000'])
    bar.ax.tick_params(labelsize=14)
    bar.ax.minorticks_off()
    bar.set_label('2025 available-data total (TWh)',fontsize=16,labelpad=10)
    fig.legend(handles=[Patch(facecolor=NO_DATA,edgecolor='#68747D',label='No data')],
               loc='center',bbox_to_anchor=(.95,.605),frameon=False,fontsize=14)
    return axes


def main():
    setup()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'text.color':'black',
                         'axes.labelcolor':'black','axes.edgecolor':'black',
                         'xtick.color':'black','ytick.color':'black',
                         'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':1.6})
    data_file = HERE/DATA_NAME
    if not data_file.exists():
        shutil.copy2(SOURCE/DATA_NAME,data_file)
    d = pd.read_csv(data_file)
    assert len(d)==28 and not d.duplicated(['year','continent']).any()
    assert d.loc[d.continent.eq('Antarctica'),'kendall_tau_b'].isna().all()
    fig = plt.figure(figsize=FIGSIZE)
    maps = draw_generation_maps(fig)
    ax = fig.add_axes([.075,.145,.91,.365])
    draw_trend(ax,d)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(item.get_window_extent(renderer).x0 for item in [ax.yaxis.label,*ax.get_yticklabels()])
    label_right = left-6*fig.dpi/72
    x = label_right/(fig.get_figwidth()*fig.dpi)
    ax.set_title('b',loc='left',x=(label_right-ax.bbox.x0)/ax.bbox.width,ha='right',
                 fontsize=28,fontweight='bold',pad=12)
    fig.text(x,.93,'a',fontsize=28,fontweight='bold',ha='right',va='bottom')
    save(fig,OUTPUT)


if __name__ == '__main__':
    main()
