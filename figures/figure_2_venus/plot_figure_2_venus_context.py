"""Figure 2: blue/red balloon tracks, black/gray mission profiles, and curve labels; preserve source data."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.path import Path as MarkerPath
from matplotlib.transforms import Affine2D
from matplotlib.ticker import FuncFormatter
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent/'model_validation'))
from figure_style import setup, SOLAR
from pv_efficiency_audit import venus_reference
from venus_profile_audit import load_audit_data

# ===== Frequently adjusted parameters =====
FIGSIZE = (22.6, 18.0)
PANEL_LABEL_SIZE = 26
PANEL_LABEL_GAP_PT = 6  # As for Mars, place panel letters left of y-axis text.
PANEL_LABEL_PAD_PT = 12
AXIS_LABEL_SIZE = 21
TICK_SIZE = 19
LEGEND_SIZE = 19
SIDE_PANEL_GAP = .07  # Panels c and d form two nearly square plots stacked on the right.
FLUX_XLIM = (50, 65)
TARGET_ALTITUDE = 54.0
OUTPUT = HERE/'图2_金星_20261116'
PROFILE_STYLES = {
    'Akatsuki RS': ('Akatsuki', '#111111', None),
    'Pioneer Venus Large Probe': ('Pioneer Venus', '#737373', None),
    'Vega 2 Lander': ('Vega 2 lander', '#AAAAAA', None),
    'Vega 1 Balloon': ('Vega 1 balloon', '#1565C0', 'o'),
    'Vega 2 Balloon': ('Vega 2 balloon', '#D32F2F', 'o'),
}
BALLOON_COLORS = [PROFILE_STYLES[name][1] for name in ('Vega 1 Balloon', 'Vega 2 Balloon')]
ZOOM_PRESSURE = (475, 735)
ZOOM_ALTITUDE = (53.1, 55.1)


def draw_profiles(ax, data, labels=True):
    for name, (label, color, marker) in PROFILE_STYLES.items():
        d = data[data.mission.eq(name)].sort_values('altitude_km')
        assert len(d) and (d.pressure_hpa > 0).all()
        if marker:
            ax.scatter(d.pressure_hpa, d.altitude_km, s=85, marker=marker,
                       color=color, edgecolor='white', linewidth=.4,
                       label=label if labels else None, zorder=4)
        else:
            ax.plot(d.pressure_hpa, d.altitude_km, lw=2.2, color=color,
                    ls='-',
                    label='_nolegend_')


def label_curve(ax, x, y, label, anchor_x, color, *, size=15, align='left', offset=4):
    """Rotate text using the displayed curve slope without moving or fitting the original curve."""
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    assert x.min() <= anchor_x <= x.max()
    half_span = .04 * np.ptp(ax.get_xlim())
    neighbors = np.clip([anchor_x-half_span, anchor_x+half_span], x.min(), x.max())
    screen = ax.transData.transform(np.column_stack([neighbors, np.interp(neighbors, x, y)]))
    dx, dy = screen[1]-screen[0]
    angle = np.degrees(np.arctan2(dy, dx))
    return ax.annotate(label, xy=(anchor_x, np.interp(anchor_x, x, y)),
                       xytext=(0, offset), textcoords='offset points', rotation=angle,
                       rotation_mode='anchor', ha=align,
                       va='bottom' if offset >= 0 else 'top', fontsize=size,
                       color=color, zorder=8, clip_on=True,
                       path_effects=[pe.withStroke(linewidth=3, foreground='white')])


def draw_context(ax, locations):
    # USGS GTDR v02: DN is elevation in metres relative to R=6051 km.
    # https://astrogeology.usgs.gov/search/map/Venus/Magellan/RadarProperties/Venus_Magellan_Topography_Global_4641m_v02.cub.xml
    path = HERE/'地图与数据说明/Venus_Magellan_Topography_Global_4641m_v02.tif'
    with Image.open(path) as image:
        width, height = image.size
        offsets = np.asarray(image.tag_v2[273])
        counts = np.asarray(image.tag_v2[279])
        # Read validated uncompressed signed-int16 strips; Pillow's image decoder
        # does not handle this file's planar-configuration tag correctly.
        assert (width, height) == (8192, 4096)
        assert image.tag_v2[259] == 1 and image.tag_v2[339] == (2,)
        assert np.all(counts == width*2) and np.all(np.diff(offsets) == width*2)
        assert offsets[-1]+counts[-1] == path.stat().st_size
        raw = np.memmap(path, dtype='<i2', mode='r', offset=int(offsets[0]), shape=(height, width))
        first, last = 1365, 2731
        # Native longitude -180..180 becomes 0..360 by swapping equal halves.
        crop = np.roll(raw[first:last], -width//2, axis=1)
        elevation = np.ma.masked_equal(crop, -32768).astype(float)/1000
        assert -3 <= elevation.min() < elevation.max() <= 12
        # Fill only GTDR missing-data strips by linear interpolation along latitude rows; preserve measured pixels.
        elevation = elevation.filled(np.nan)
        missing = ~np.isfinite(elevation)
        columns = np.arange(elevation.shape[1])
        for row in elevation:
            valid = np.isfinite(row)
            assert valid.any()
            row[~valid] = np.interp(columns[~valid], columns[valid], row[valid])
        assert missing.any() and np.isfinite(elevation).all()
    cmap = plt.get_cmap('gray').copy()
    cmap.set_bad('#e5e5e5')
    mapped = ax.imshow(elevation, extent=(0, 360, 90-last*180/height, 90-first*180/height),
                       origin='upper', aspect='auto', cmap=cmap,
                       norm=Normalize(-3, 10), interpolation='nearest', zorder=0)
    map_position = ax.get_position()
    colorbar_width = .40
    colorbar_left = map_position.x0+(map_position.width-colorbar_width)/2
    cax = ax.figure.add_axes([colorbar_left, .735, colorbar_width, .013])
    colorbar = ax.figure.colorbar(mapped, cax=cax, orientation='horizontal', ticks=[-3, 0, 3, 6, 9])
    colorbar.set_label('Elevation (km)', fontsize=AXIS_LABEL_SIZE-1, labelpad=3)
    colorbar.ax.tick_params(labelsize=TICK_SIZE-1, length=3, width=.7)
    colorbar.outline.set_linewidth(.8)
    ax.set(xlim=(0, 360), ylim=(-30, 30), xlabel='Longitude', ylabel='Latitude',
           xticks=np.arange(0, 361, 60), yticks=np.arange(-30, 31, 15))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, pos: '0' if v in (0, 360) else '180°' if v == 180
        else f'{v if v < 180 else 360-v:g}°' + ('E' if v < 180 else 'W')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: '0' if v == 0 else f'{abs(v):g}°' + ('N' if v > 0 else 'S')))
    ax.grid(color='#555555', linewidth=.01, alpha=.12)
    labels = {
        'Vega 1 Balloon': ((110, 21), 'Vega 1 balloon'),
        'Vega 2 Balloon': ((110, -22), 'Vega 2 balloon')}
    for name, (text_xy, text) in labels.items():
        g = locations[locations.mission.eq(name)].sort_values('utc')
        color = PROFILE_STYLES[name][1]
        ax.plot(g.longitude_deg_e, g.latitude_deg_n, color=color, lw=2.5)
        ax.scatter(g.longitude_deg_e.iloc[0], g.latitude_deg_n.iloc[0], marker='o', s=210,
                   facecolor=color, edgecolor='white', linewidth=1.3, zorder=4)
        ax.scatter(g.longitude_deg_e.iloc[-1], g.latitude_deg_n.iloc[-1], marker='o', s=175,
                   facecolor=color, edgecolor='white', linewidth=1.3, zorder=4)
        middle = g.iloc[len(g)//2]
        ax.annotate(text, (middle.longitude_deg_e, middle.latitude_deg_n), xytext=text_xy,
                    ha='center', va='center', fontsize=LEGEND_SIZE, color=color,
                    arrowprops=dict(arrowstyle='-', color=color))
        i, j = int(len(g)*.55), int(len(g)*.68)
        assert g.utc.iloc[i] < g.utc.iloc[j]
        # Indicate direction with an open arrowhead consisting of two strokes, without a shaft.
        positions = g[['longitude_deg_e', 'latitude_deg_n']].iloc[[i, j]].to_numpy()
        dx, dy = np.diff(ax.transData.transform(positions), axis=0)[0]
        head = MarkerPath([[-.5, .5], [.5, 0], [-.5, -.5]]).transformed(
            Affine2D().rotate(np.arctan2(dy, dx)))
        ax.plot(g.longitude_deg_e.iloc[j], g.latitude_deg_n.iloc[j], linestyle='none',
                color='white', marker=head, markersize=16, markerfacecolor='none',
                markeredgecolor='white', markeredgewidth=3.6, zorder=4.9)
        ax.plot(g.longitude_deg_e.iloc[j], g.latitude_deg_n.iloc[j], linestyle='none',
                color=color, marker=head, markersize=16, markerfacecolor='none',
                markeredgecolor=color, markeredgewidth=1.8, zorder=5)
    ak = locations[locations.mission.eq('Akatsuki RS')].sort_values('altitude_km')
    lon, lat = [np.interp(54, ak.altitude_km, ak[field]) for field in ['longitude_deg_e', 'latitude_deg_n']]
    ax.plot(ak.longitude_deg_e, ak.latitude_deg_n, color='black', lw=2)
    ax.scatter(lon, lat, facecolor='black', edgecolor='white', linewidth=1.4, marker='^', s=280, zorder=5)
    ax.annotate('Akatsuki', (lon, lat), xytext=(225, 21),
                ha='center', va='center', fontsize=LEGEND_SIZE, color='black',
                arrowprops=dict(arrowstyle='-', color='black'))
    for name, text, text_xy in [
        ('Pioneer Venus Large Probe', 'Pioneer Venus large probe', (284, -17)),
        ('Vega 2 Lander', 'Vega 2 lander', (211, -22))]:
        row = locations[locations.mission.eq(name)].iloc[0]
        ax.scatter(row.longitude_deg_e, row.latitude_deg_n, facecolor='black', edgecolor='white',
                   linewidth=1.4, marker='^', s=280, zorder=5)
        ax.annotate(text, (row.longitude_deg_e, row.latitude_deg_n), xytext=text_xy,
                    ha='center', va='center', fontsize=LEGEND_SIZE, color='black',
                    arrowprops=dict(arrowstyle='-', color='black'))
    for spine in ax.spines.values():
        spine.set_visible(True)
    for text in ax.texts:
        if not text.get_text():
            continue
        mission_color = text.get_color()
        text.set_color('#20262c')
        text.set_fontweight('bold')
        text.set_bbox(dict(boxstyle='round,pad=.3', facecolor='white',
                           edgecolor=mission_color, linewidth=1.1, alpha=.96))
        text.set_zorder(6)
    for line in ax.lines:
        if line.get_linestyle() == 'None':
            line.set_path_effects([])
            continue
        line.set_path_effects([pe.Stroke(linewidth=line.get_linewidth()+2,
                                       foreground='white', alpha=.8), pe.Normal()])



def main():
    setup()
    # Override only the Venus figure: four-sided frames, inward ticks, and unparenthesized letters match Mars.
    plt.rcParams.update({
        'font.family': 'Arial', 'text.color': 'black',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.linewidth': 1.6, 'legend.frameon': True,
        'legend.fancybox': False, 'legend.framealpha': 1,
        'legend.edgecolor': 'black', 'legend.facecolor': 'white',
    })
    # Re-read mission data, correct Akatsuki's altitude definition, and filter balloons by pressure/radius quality flags.
    _, profiles, locations, _, _ = load_audit_data()
    winds = pd.read_csv(HERE/'金星-2.csv')
    flux = pd.read_csv(HERE/'金星-3.csv')
    raw_flux = pd.read_csv(HERE.parents[1]/'venus/venus_solar_power/data/lsfr_solar_flux.csv', header=None)
    for series, column in [('Visible downward flux', 3), ('Combined downward flux', 6),
                           ('Narrowband downward flux', 9)]:
        expected = set(map(tuple, raw_flux[[2, column]].dropna().round(6).to_numpy()))
        actual = set(map(tuple, flux.loc[flux.series.eq(series), ['altitude_km', 'flux_w_m2']].round(6).to_numpy()))
        assert actual and actual <= expected, f'{series}: plotted values differ from the PDS table'
    fig = plt.figure(figsize=FIGSIZE)
    # Positions are fractions of the full canvas; panels c and d share width and height and are nearly square.
    bottom, top = .075, .675
    side_height = (top-bottom-SIDE_PANEL_GAP)/2
    right_width = side_height*FIGSIZE[1]/FIGSIZE[0]
    right_left = .555
    right_edge = right_left+right_width
    context = fig.add_axes([.075, .80, right_edge-.075, .175])
    draw_context(context, locations)
    a = fig.add_axes([.075, bottom, .38, top-bottom])
    c = fig.add_axes([right_left, bottom, right_width, side_height])
    wind_bottom = bottom + side_height + SIDE_PANEL_GAP
    b = fig.add_axes([right_left, wind_bottom, right_width, side_height])
    for ax in [context, a, b, c]:
        ax.tick_params(direction='in', top=True, right=True, length=7,
                       width=1.1, labelsize=TICK_SIZE)
        ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
        ax.tick_params(axis='x', pad=12)
        ax.tick_params(axis='y', pad=10)
        ax.xaxis.labelpad = 12
        ax.yaxis.labelpad = 16
        ax.grid(color='#555555', linewidth=.01, alpha=.12)
        ax.set_axisbelow(True)
    c.set_facecolor('white')

    draw_profiles(a, profiles)
    a.set(xscale='log', xlim=(.05, 2e5), ylim=(0, 96),
          xlabel='Pressure (hPa)', ylabel='Altitude (km)')
    a.axhspan(51, 55, color=SOLAR, alpha=.11)
    a.axhline(TARGET_ALTITUDE, color='#6C7983', ls='--', lw=1)
    a.legend(loc='upper right', fontsize=LEGEND_SIZE, labelspacing=.6).get_frame().set_linewidth(1.3)
    a.text(.02, .59, 'Balloon layer', transform=a.transAxes, color='#99702C', fontsize=16)
    # Draw insets with readable fonts rather than pasting PNG screenshots.
    zoom = a.inset_axes([.12, .10, .50, .40])
    draw_profiles(zoom, profiles, labels=False)
    zoom.set(xlim=ZOOM_PRESSURE, ylim=ZOOM_ALTITUDE)
    zoom.set_xlabel('Pressure (hPa)', fontsize=16)
    zoom.set_ylabel('Altitude (km)', fontsize=15, labelpad=5)
    zoom.tick_params(labelsize=15, direction='in', top=True, right=True)
    zoom.set_xticks([500, 550, 600, 650, 700])
    zoom.set_yticks([53.5, 54, 54.5, 55])
    zoom.set_facecolor('white')
    zoom.axhline(TARGET_ALTITUDE, color='#6C7983', lw=.8, ls='--')
    zoom.grid(linewidth=.01, alpha=.12)
    a.add_patch(Rectangle((ZOOM_PRESSURE[0], ZOOM_ALTITUDE[0]),
                          np.ptp(ZOOM_PRESSURE), np.ptp(ZOOM_ALTITUDE), fill=False,
                          edgecolor='#555555', linewidth=1.2, zorder=5))
    for source, dest in [((ZOOM_PRESSURE[0], ZOOM_ALTITUDE[0]), (0, 1)),
                         ((ZOOM_PRESSURE[1], ZOOM_ALTITUDE[0]), (1, 1))]:
        fig.add_artist(ConnectionPatch(xyA=source, coordsA=a.transData,
                       xyB=dest, coordsB=zoom.transAxes, color='#888888',
                       linewidth=.8, zorder=1))
    for spine in zoom.spines.values():
        spine.set_visible(True)
    for name, label_altitude, offset in [('Akatsuki RS', 53.35, 4),
                                         ('Pioneer Venus Large Probe', 53.2, 4),
                                         ('Vega 2 Lander', 53.35, -4)]:
        d = profiles.loc[profiles.mission.eq(name)].sort_values('altitude_km')
        pressure = np.interp(label_altitude, d.altitude_km, d.pressure_hpa)
        label, color, _ = PROFILE_STYLES[name]
        label_curve(zoom, d.pressure_hpa, d.altitude_km, label, pressure,
                    color, size=16, align='right', offset=offset)

    rows = []
    for name, color in zip(['Vega 1 Balloon', 'Vega 2 Balloon'], BALLOON_COLORS):
        d = winds[winds.balloon.eq(name)].sort_values('elapsed_time_h')
        speed = d.horizontal_wind_speed_ms.to_numpy()
        b.plot(d.elapsed_time_h, speed, lw=2.1, color=color, label=name.replace(' Balloon',' balloon'))
        rows.append(dict(balloon=name, n=len(speed), mean_ms=speed.mean(),
                         sd_ms=speed.std(ddof=1), variance_m2_s2=speed.var(ddof=1)))
    b.set(xlabel='Elapsed time (h)', ylabel='Ground-relative speed (m s$^{-1}$)',
          xlim=(0, 50), ylim=(60, 74), xticks=[0, 10, 20, 30, 40],
          yticks=np.arange(60, 75, 2))
    b.legend(loc='upper left', fontsize=LEGEND_SIZE).get_frame().set_linewidth(1.3)
    # Place two mean +/- sample-standard-deviation bars to the right of the time series within panel c.
    bar_x = 47.5
    for row, color in zip(rows, BALLOON_COLORS):
        b.vlines(bar_x, row['mean_ms']-row['sd_ms'], row['mean_ms']+row['sd_ms'],
                 color=color, linewidth=11, alpha=.4, zorder=3)
        b.scatter(bar_x, row['mean_ms'], s=70, facecolor=color,
                  edgecolor='white', linewidth=1.5, zorder=4)

    # Product-specific approximate passbands, which vary with temperature:
    # https://atmos.nmsu.edu/PDS/data/PDS4/pv_sed_bundle/data_lsfr/lsfr_solar_flux_descr.txt
    specs = [('Visible downward flux', 'Visible', '#737373'),
             ('Combined downward flux', 'Combined', '#111111'),
             ('Narrowband downward flux', 'Narrowband', '#AAAAAA')]
    for series, label, color in specs:
        d = flux[(flux.record_type=='solar_flux') & (flux.series==series)].sort_values('altitude_km')
        assert len(d)
        if series == 'Combined downward flux':
            # Archive explicitly cautions against interpreting Combined below ~50 km.
            # Shared boundary sample keeps the measured polyline continuous.
            boundary = np.searchsorted(d.altitude_km.to_numpy(), 50)
            c.plot(d.altitude_km.iloc[boundary:], d.flux_w_m2.iloc[boundary:],
                   color=color, lw=2.2, label=label)
            c.plot(d.altitude_km.iloc[:boundary+1], d.flux_w_m2.iloc[:boundary+1],
                   color=color, lw=2.2, ls=(0, (4, 2)))
        else:
            c.plot(d.altitude_km, d.flux_w_m2, color=color, lw=2.2, label=label)
    c.axvspan(51, 55, color=SOLAR, alpha=.09, linewidth=0, zorder=0)
    c.text(53, 105, 'Balloon layer', ha='center', va='bottom', fontsize=14, color='#99702C')
    ref = venus_reference()
    c.axvline(TARGET_ALTITUDE, color='#52636E', ls='--', lw=1.2)
    c.scatter(TARGET_ALTITUDE,ref['irradiance_w_m2'],color='black',s=65,edgecolor='white',zorder=5)
    c.set(xlabel='Altitude above surface (km)', ylabel='Downward solar flux (W m$^{-2}$)',
          xlim=FLUX_XLIM, ylim=(0,570), xticks=np.arange(50,66,5), yticks=np.arange(0,501,100))
    for series, label, color in specs:
        d = flux[(flux.record_type=='solar_flux') & (flux.series==series)].sort_values('altitude_km')
        label_curve(c, d.altitude_km, d.flux_w_m2, label, 58.5, color,
                    size=16, offset=-5 if label == 'Visible' else 4)
    # Place panel letters left of their y-axis text: 26 pt with a 6 pt gap.
    fig.canvas.draw()
    # Align b/d x-axis titles and c/d y-axis titles using d's actual text positions.
    xlabel_y = c.xaxis.label.get_transform().transform(c.xaxis.label.get_position())[1]
    a.xaxis.set_label_coords(.5, (xlabel_y-a.bbox.y0)/a.bbox.height)
    ylabel_x = c.yaxis.label.get_transform().transform(c.yaxis.label.get_position())[0]
    b.yaxis.set_label_coords((ylabel_x-b.bbox.x0)/b.bbox.width, .5)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_edges = [min(item.get_window_extent(renderer).x0
                      for item in [ax.yaxis.label, *ax.get_yticklabels()])
                  for ax in [context, a, b, c]]
    # Align Mars b/c letters vertically; for Venus use a common left edge despite differing y-axis text widths.
    left_edges[2:] = [min(left_edges[2:])]*2
    for ax, letter, left_edge in zip([context, a, b, c], 'abcd', left_edges):
        label_x = (left_edge - PANEL_LABEL_GAP_PT*fig.dpi/72 - ax.bbox.x0) / ax.bbox.width
        ax.set_title(letter, loc='left', x=label_x, ha='right', pad=PANEL_LABEL_PAD_PT,
                     fontsize=PANEL_LABEL_SIZE, fontweight='bold', fontfamily='Arial')
    assert np.isclose(b.bbox.height, c.bbox.height)
    assert np.isclose(b.bbox.width, b.bbox.height)
    assert np.isclose(c.bbox.width, c.bbox.height)
    assert np.isclose(b.bbox.x1, c.bbox.x1) and np.isclose(c.bbox.x1, context.bbox.x1)
    assert all(line.get_linewidth() == .01 for line in c.get_xgridlines())
    assert [text.get_text() for text in a.get_legend().get_texts()] == ['Vega 1 balloon', 'Vega 2 balloon']
    assert c.get_legend() is None and all(text.get_bbox_patch() is None for text in c.texts)
    assert all(text.get_fontfamily() == ['Arial']
               for text in fig.findobj(match=plt.Text) if text.get_text())
    for row in rows:
        print(row)
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved context PNG.')


if __name__ == '__main__':
    main()
