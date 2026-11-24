"""Figure 1: long-term energy evolution and two representative sols, panels a-c."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from figure_style import setup, panel, save
from figure_style import SOLAR as SIDE_SOLAR, WIND as SIDE_WIND, HYBRID as SIDE_HYBRID

# ===== Frequently adjusted parameters =====
FIGSIZE = (22.6, 10)
# Panel a has an aspect ratio of about 1.49, matching the reference; b and c are stacked on the right.
COLUMN_RATIOS = [2, 1]
PANEL_LABEL_GAP_PT = 6  # Gap between the panel letter's right edge and the y-axis text's left edge.
PANEL_LABEL_SIZE = 28
CALENDAR_AXIS_OFFSET = -.11  # Earth-time axis position relative to the height of panel a.
MAIN_LABEL_SIZE = 21
MAIN_TICK_SIZE = 19
SIDE_LABEL_SIZE = 19
SIDE_TICK_SIZE = 17
SIDE_LEGEND_SIZE = 14
SIDE_NOTE_SIZE = 14.5
SOLAR = '#008000'
HYBRID = '#A51C24'
PANEL_BACKGROUND = '#ECFFFF'
# Mars northern-hemisphere seasons and Earth UTC calendar months use separate labels.
SEASONS = [('N. spring', '#CAE8CB', '#39724B'),
           ('N. summer', '#FFE4AD', '#9A691C'),
           ('N. autumn', '#F5CDD1', '#9B4D59'),
           ('N. winter', '#D5D0ED', '#6C5B95')]
SEASON_ALPHA = .42
# Derived from engineering telemetry: UTC=2018-334T15:01:26.932, Sol 4 07:00:42.021.
SOL0_UTC = pd.Timestamp('2018-11-26T05:10:50.336037Z')
SOL_SECONDS = 88775.244
SMOOTH_SOLS = 7
EXAMPLE_SOLS = [1080, 1248]
SOLAR_DATA = HERE / '火星-2.csv'
HOURLY_DATA = HERE / '火星-1.csv'
OUTPUT = HERE / '图1_火星_组合'
# Upstream integration uses 0.01 Mars-local hours; Wh conversion requires 3600 seconds per hour.
# Preserve the CSV and convert wind energy after reading; solar energy is already daily Wh and is not rescaled.
SOL_TO_EARTH_HOUR = 88775.244 / 86400


def rolling_segments(series, window=SMOOTH_SOLS):
    """Smooth only contiguous valid sols; do not bridge gaps or zero placeholders."""
    output = pd.DataFrame(index=series.index, columns=['median'], dtype=float)
    valid = series.notna()
    group = (~valid).cumsum()
    for _, values in series[valid].groupby(group[valid]):
        roll = values.rolling(window, center=True, min_periods=1)
        output.loc[values.index, 'median'] = roll.median()
    return output


def solar_longitude(days_j2000_tt):
    """NASA Mars24 algorithms B-1 through B-5; return unwrapped solar longitude in degrees."""
    d = np.asarray(days_j2000_tt)
    m = np.deg2rad(19.3871 + .52402073*d)
    perturbers = [(0.0071, 2.2353, 49.409), (0.0057, 2.7543, 168.173),
                 (0.0039, 1.1177, 191.837), (0.0037, 15.7866, 21.736),
                 (0.0021, 2.1354, 15.704), (0.0020, 2.4694, 95.528),
                 (0.0018, 32.8493, 49.095)]
    pbs = sum(a*np.cos(np.deg2rad(.985626*d/tau + phi)) for a, tau, phi in perturbers)
    center = ((10.691 + 3e-7*d)*np.sin(m) + .623*np.sin(2*m)
              + .050*np.sin(3*m) + .005*np.sin(4*m) + .0005*np.sin(5*m) + pbs)
    return 270.3871 + .524038496*d + center


def add_calendar(ax):
    """Seasons are bounded by 90-degree Ls intervals, not four equal time intervals."""
    # NASA examples independently check the calculation; TT-UTC is 69.184 seconds for this 2018-2022 figure.
    assert abs(solar_longitude(4.50074) % 360 - 277.18758) < 2e-5
    assert abs(solar_longitude(1463.07471) % 360 - 327.32416) < 2e-5
    sols = np.linspace(0, 1350, 5401)
    epoch_tt = SOL0_UTC.to_julian_date() + 69.184/86400 - 2451545
    ls = solar_longitude(epoch_tt + sols*SOL_SECONDS/86400)
    assert (np.diff(ls) > 0).all()
    crossings = np.arange(np.ceil(ls[0]/90)*90, ls[-1], 90)
    boundaries = np.r_[0, np.interp(crossings, ls, sols), 1350]
    assert np.max(np.abs(solar_longitude(epoch_tt + boundaries[1:-1]*SOL_SECONDS/86400)
                         - crossings)) < 1e-4
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        mid = (start + end)/2
        season_index = int((solar_longitude(epoch_tt + mid*SOL_SECONDS/86400) % 360)//90)
        label, color, text_color = SEASONS[season_index]
        ax.axvspan(start, end, facecolor=color, alpha=SEASON_ALPHA, zorder=0)
        if start > 0:
            ax.axvline(start, color=text_color, lw=.8, alpha=.3, zorder=.5)
        if end-start > 50:
            # Lower the text to avoid the steep decline during the initial winter.
            ax.text(mid, .48 if start == 0 else .67, label, rotation=90, ha='center', va='center',
                    transform=ax.get_xaxis_transform(), fontsize=16, color=text_color)
        rows.append(dict(season=label, display_start_sol=start, display_end_sol=end,
                         start_utc=str(SOL0_UTC+pd.Timedelta(seconds=start*SOL_SECONDS)),
                         end_utc=str(SOL0_UTC+pd.Timedelta(seconds=end*SOL_SECONDS))))
    pd.DataFrame(rows).to_csv(HERE/'图1_季节分区.csv', index=False, encoding='utf-8-sig')
    months = pd.date_range('2019-01-01', '2022-07-01', freq='QS', tz='UTC')
    ticks = (months-SOL0_UTC).total_seconds()/SOL_SECONDS
    calendar = ax.secondary_xaxis(CALENDAR_AXIS_OFFSET)
    calendar.set_xticks(ticks, [date.strftime('%b\n%Y') for date in months])
    calendar.tick_params(axis='x', labelsize=12, length=4, pad=5, direction='out')
    calendar.set_xlabel('Earth calendar (UTC)', fontsize=13, labelpad=6)
    calendar.spines['bottom'].set_linewidth(.9)


def main():
    setup()
    # Override styles for the Mars figure only.
    plt.rcParams.update({
        'font.family': 'Arial', 'text.color': 'black',
        'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.linewidth': 1.6, 'legend.frameon': True,
        'legend.fancybox': False, 'legend.framealpha': 1,
        'legend.edgecolor': 'black', 'legend.facecolor': 'white',
    })
    source = pd.read_csv(SOLAR_DATA).set_index('sol_number').sort_index()
    assert source.index.is_unique
    # Insert missing sol rows as NaN so curves do not bridge observation gaps.
    d = source.reindex(range(int(source.index.min()), int(source.index.max()) + 1))
    solar = d.solar_daily_wh.where(d.solar_daily_wh > 0)
    wind_all = d.all_day_wind_daily_wh.where(d.all_day_wind_available.eq(1))
    wind_strict = d.continuous10_wind_daily_wh.where(d.continuous10_wind_available.eq(1))
    # The two columns use different reconstruction schemes; continuity-filled data need not form a strict subset.
    assert (wind_all.dropna() >= 0).all() and (wind_strict.dropna() >= 0).all()
    paired = solar.notna() & wind_strict.notna()
    assert np.allclose((solar+wind_strict)[paired], d.continuous10_solar_plus_wind_daily_wh[paired])
    wind_all = wind_all * SOL_TO_EARTH_HOUR
    wind_strict = wind_strict * SOL_TO_EARTH_HOUR
    # These are existing reconstructions/scenarios, not independently measured power or confidence intervals.
    ss = rolling_segments(solar)
    hybrid = rolling_segments(solar + wind_strict)
    x = d.index.to_numpy()
    fig = plt.figure(figsize=FIGSIZE)
    grid = fig.add_gridspec(2, 2, width_ratios=COLUMN_RATIOS, hspace=.35, wspace=.23)
    a = fig.add_subplot(grid[:, 0])
    a.set_facecolor(PANEL_BACKGROUND)
    a.set_axisbelow(True)
    a.grid(axis='both', color='#555555', linewidth=.8, alpha=.65)
    a.tick_params(direction='in', top=True, right=True, length=7, width=1.1)
    a.set_title('a', loc='left', pad=12, fontsize=PANEL_LABEL_SIZE, fontweight='bold')

    # Use Wh as in the reference; retain the original numerical and statistical conventions.
    a.plot(x, solar, linestyle='none', marker='o', ms=2.5, alpha=.45,
           color=SOLAR, rasterized=True)
    a.plot(x, ss['median'], color=SOLAR, lw=2.1, label='Solar reconstruction')
    a.plot(x, hybrid['median'], color=HYBRID, lw=2.6, label='Solar and wind scenario')
    a.set(ylabel='Energy per sol (Wh)', ylim=(0, 5000), xlim=(0, 1350),
          xlabel='Sol', xticks=np.arange(0, 1351, 150), yticks=np.arange(0, 5001, 1000))
    a.tick_params(labelsize=MAIN_TICK_SIZE)
    a.xaxis.label.set_size(MAIN_LABEL_SIZE)
    a.yaxis.label.set_size(MAIN_LABEL_SIZE)
    a.legend(loc='upper right', fontsize=16, borderpad=.5, handlelength=2).get_frame().set_linewidth(1.5)
    add_calendar(a)

    for index, sol in enumerate(EXAMPLE_SOLS):
        a.axvline(sol, color='#606060', lw=1.1, ls=(0, (4, 4)), alpha=.65)
        a.text(sol-18, 2200-index*650, f'Sol {sol}\nsee {chr(98+index)}',
               ha='right', color='black', fontsize=15,
               bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=3))

    # Keep white backgrounds and original curve colors in b and c; match frames and fonts to a.
    setup()
    plt.rcParams.update({'font.family': 'Arial', 'text.color': 'black',
                         'axes.labelcolor': 'black', 'xtick.color': 'black',
                         'ytick.color': 'black'})
    b, c = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])
    hourly = pd.read_csv(HOURLY_DATA)
    summaries = []
    for ax, sol, letter in zip([b, c], EXAMPLE_SOLS, ['b', 'c']):
        h = hourly[hourly.sol.eq(f'SOL-{sol:04d}')].sort_values('hour')
        assert h.hour.tolist() == list(range(24))
        assert np.allclose(h.solar_only_wh+h.wind_added_wh, h.solar_plus_wind_wh)
        h = h.copy()
        h['wind_added_wh'] *= SOL_TO_EARTH_HOUR
        h['solar_plus_wind_wh'] = h.solar_only_wh+h.wind_added_wh
        # Plot hourly energy at interval centers and extend edges to 0/24 without adding energy samples.
        hour = np.r_[0., h.hour.to_numpy() + .5, 24.]
        solar_line = np.pad(h.solar_only_wh.to_numpy(), (1, 1), mode='edge')
        total_line = np.pad(h.solar_plus_wind_wh.to_numpy(), (1, 1), mode='edge')
        np.testing.assert_array_equal(solar_line[1:-1], h.solar_only_wh)
        np.testing.assert_array_equal(total_line[1:-1], h.solar_plus_wind_wh)
        ax.fill_between(hour, solar_line, total_line,
                        color=SIDE_WIND, alpha=.25, label='Wind increment')
        ax.plot(hour, solar_line, '-o', color=SIDE_SOLAR, lw=2, ms=3.2,
                markevery=list(range(1, 25)), label='Solar')
        ax.plot(hour, total_line, '-', color=SIDE_HYBRID, lw=2, label='Solar and wind')
        ax.set(xlabel='Local solar time (h)', ylabel='Energy per hour (Wh)',
               xlim=(0, 24), ylim=(0, 160), xticks=np.arange(0, 25, 4),
               yticks=[0, 50, 100, 150])
        panel(ax, letter)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.6)
        ax.tick_params(direction='in', top=True, right=True, length=6,
                       width=1.1, labelsize=SIDE_TICK_SIZE)
        ax.xaxis.label.set_size(SIDE_LABEL_SIZE)
        ax.yaxis.label.set_size(SIDE_LABEL_SIZE)
        ax.set_title(letter, loc='left', pad=12, fontsize=PANEL_LABEL_SIZE, fontweight='bold')
        total_solar, total_wind = h.solar_only_wh.sum(), h.wind_added_wh.sum()
        share = total_wind / (total_solar+total_wind)*100
        ax.text(.025, .97, f'Solar: {total_solar:.0f} Wh\nWind: {total_wind:.1f} Wh\nWind share: {share:.1f}%',
                va='top', transform=ax.transAxes, fontsize=SIDE_NOTE_SIZE,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=4))
        ax.legend(loc='upper right', fontsize=SIDE_LEGEND_SIZE, frameon=True,
                  fancybox=False, facecolor='white', edgecolor='black',
                  framealpha=1, borderpad=.4, handlelength=1.5,
                  labelspacing=.3).get_frame().set_linewidth(1.3)
        summaries.append(dict(sol=sol, solar_wh=total_solar, wind_wh=total_wind, wind_share_pct=share))

    fig.subplots_adjust(left=.065, right=.985, top=.94, bottom=.105)
    # Right-hand panels have equal height; the lower edge of c aligns with a's Earth-time axis.
    pa, pb, pc = a.get_position(), b.get_position(), c.get_position()
    calendar_bottom = pa.y0 + CALENDAR_AXIS_OFFSET*pa.height
    gap = pb.y0 - pc.y1
    side_height = (pa.y1 - calendar_bottom - gap)/2
    b.set_position([pb.x0, pa.y1-side_height, pb.width, side_height])
    c.set_position([pc.x0, calendar_bottom, pc.width, side_height])
    # Position panel letters left of y-axis titles and tick labels using actual text bounds.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, letter in zip([a, b, c], 'abc'):
        left_edge = min(text.get_window_extent(renderer).x0
                        for text in [ax.yaxis.label, *ax.get_yticklabels()])
        label_right = left_edge - PANEL_LABEL_GAP_PT*fig.dpi/72
        label_x = (label_right-ax.bbox.x0)/ax.bbox.width
        ax.set_title(letter, loc='left', x=label_x, ha='right',
                     pad=12, fontsize=PANEL_LABEL_SIZE, fontweight='bold')
    save(fig, OUTPUT)
    pd.DataFrame(summaries).to_csv(HERE/'图1_代表日统计.csv', index=False, encoding='utf-8-sig')
    valid = solar.notna() & wind_strict.notna()
    summary = dict(input_sols=len(source), omitted_zero_placeholders=int((source.solar_daily_wh==0).sum()),
                   continuity_paired_sols=int(valid.sum()),
                   continuity_wind_share_pct=float(100*wind_strict[valid].sum()/(solar[valid]+wind_strict[valid]).sum()),
                   continuity_available_but_all_day_missing=int((wind_strict.notna() & wind_all.isna()).sum()),
                   wind_wh_time_unit_factor=SOL_TO_EARTH_HOUR,
                   upstream_wind_reuse_cycle_sols=668,
                   interpolated_points=0, band=None)
    (HERE/'图1_统计口径.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # Gap check: smoothing must not create values at missing observations.
    test = rolling_segments(pd.Series([1., 2., np.nan, 100., 101.]))
    assert test.iloc[2].isna().all() and test.iloc[1]['median'] < 3


if __name__ == '__main__':
    main()
