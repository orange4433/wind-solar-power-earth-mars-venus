"""
Audit Figure 2a source data, quality flags, and altitude definitions; plot positions and differences.
Run this file directly. Original mission files and the historical Venus profile CSV are not overwritten.
"""
from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / '数据核查'
RAW = PROJECT / 'venus/akatsuki_pioneer_profiles/data/raw'
VEGA = PROJECT / 'venus/vega_wind_power'
sys.path.insert(0, str(VEGA))
sys.path.insert(0, str(HERE.parent))
from read_vega_data import read_table
from figure_style import setup, save

REFERENCE_RADIUS_KM = 6051.8
PIONEER_SITE_RADIUS_KM = 6052.0  # Official documentation gives R0 = 6052.0 +/- 0.25 km.
COLORS = {'Akatsuki RS': '#2477A5', 'Pioneer Venus Large Probe': '#D77D29',
          'Vega 2 Lander': '#229879', 'Vega 1 Balloon': '#C09B2D',
          'Vega 2 Balloon': '#9367AC'}


def load_audit_data():
    """Return original profiles, corrected profiles, and positions, with filtering based on quality flags."""
    original = pd.read_csv(HERE/'金星-1.csv')
    ak = pd.read_csv(RAW/'rs_20160723_022600_udsc64_l4_ae_v10.tab', sep=r'\s+', header=None)
    pv = pd.read_csv(RAW/'las_low_atm_state.csv', header=None)
    lander = read_table('vg2lr.dat')
    lander = lander[lander.altitude_m.ge(0)].copy()  # As in the original figure, exclude four post-landing records with negative altitude.
    balloon = {n: read_table(f'vg{n}bl_rdr.dat') for n in [1, 2]}
    original_sources = {
        'Akatsuki RS': (ak[7].to_numpy(), ak[10].to_numpy()/100),
        'Pioneer Venus Large Probe': (pv[2].to_numpy(), pv[3].to_numpy()*1000),
        'Vega 2 Lander': (lander.altitude_m.to_numpy()/1000, lander.pressure_bar.to_numpy()*1000)}
    checks = []
    for name, (height, pressure) in original_sources.items():
        old = original[original.mission.eq(name)]
        error = (abs(old.altitude_km.to_numpy()[:, None]-height)
                 + abs(old.pressure_hpa.to_numpy()[:, None]-pressure))
        assert len(old) == len(height) and error.min(axis=1).max() < 1e-7
        checks.append(dict(mission=name, old_points=len(old), retained_points=len(old),
                           no_pressure_flag_points=0, max_source_match_error=float(error.min(axis=1).max())))
    corrected = [
        pd.DataFrame(dict(mission='Akatsuki RS', altitude_km=ak[3]-REFERENCE_RADIUS_KM,
                          pressure_hpa=ak[10]/100, height_basis='geometric above R=6051.8 km')),
        pd.DataFrame(dict(mission='Pioneer Venus Large Probe',
                          altitude_km=pv[2]+PIONEER_SITE_RADIUS_KM-REFERENCE_RADIUS_KM,
                          pressure_hpa=pv[3]*1000, height_basis='geometric above R=6051.8 km; R0 uncertainty 0.25 km')),
        # Label this as reconstructed altitude; information is insufficient to unify absolute altitude datums.
        pd.DataFrame(dict(mission='Vega 2 Lander', altitude_km=lander.altitude_m/1000,
                          pressure_hpa=lander.pressure_bar*1000, height_basis='native reconstructed altitude; datum not unified'))]
    locations = [pd.DataFrame(dict(mission='Akatsuki RS', longitude_deg_e=ak[5],
                                  latitude_deg_n=ak[4], altitude_km=ak[3]-REFERENCE_RADIUS_KM,
                                  utc=ak[1], kind='occultation tangent point'))]
    flags = []
    for n, raw in balloon.items():
        name = f'Vega {n} Balloon'
        old = original[original.mission.eq(name)]
        height = raw.radius_km.to_numpy()-REFERENCE_RADIUS_KM
        error = (abs(old.altitude_km.to_numpy()[:, None]-height)
                 + abs(old.pressure_hpa.to_numpy()[:, None]-raw.pressure_hpa.to_numpy()))
        indices = error.argmin(axis=1)
        assert error.min(axis=1).max() < 1e-7
        selected = raw.iloc[indices].copy()
        selected.insert(0, 'mission', name)
        selected['old_altitude_km'] = old.altitude_km.to_numpy()
        selected['retained_in_pressure_plot'] = selected.pressure_flag.between(2, 4) & selected.radius_flag.between(2, 4)
        flags.append(selected)
        good = raw[(raw.radius_km > 6000) & raw.pressure_hpa.gt(0)
                   & raw.pressure_flag.between(2, 4) & raw.radius_flag.between(2, 4)]
        checks.append(dict(mission=name, old_points=len(old), retained_points=len(good),
                           no_pressure_flag_points=int(selected.pressure_flag.eq(0).sum()),
                           max_source_match_error=float(error.min(axis=1).max())))
        corrected.append(pd.DataFrame(dict(mission=name, altitude_km=good.radius_km-REFERENCE_RADIUS_KM,
                                            pressure_hpa=good.pressure_hpa,
                                            height_basis='model-assisted radius minus 6051.8 km')))
        track = raw[(raw.radius_km > 6000) & raw.latitude_flag.between(2, 4)
                    & raw.longitude_flag.between(2, 4)].sort_values('time_s')
        # Filter locations by position quality; missing pressure must not discard valid positions.
        locations.append(pd.DataFrame(dict(mission=name, longitude_deg_e=track.longitude_deg,
                                            latitude_deg_n=track.latitude_deg,
                                            altitude_km=track.radius_km-REFERENCE_RADIUS_KM,
                                            utc=pd.Timestamp(f'1985-06-{11 if n == 1 else 15}T00:00:00Z')
                                                + pd.to_timedelta(track.time_s, unit='s'),
                                            kind='radio-tracking trajectory with model-assisted altitude')))
    locations.append(pd.DataFrame([
        dict(mission='Pioneer Venus Large Probe', longitude_deg_e=304.0, latitude_deg_n=4.4,
             altitude_km=np.nan, utc='1978-12-09', kind='impact site; not a measured descent track'),
        dict(mission='Vega 2 Lander', longitude_deg_e=181.08, latitude_deg_n=-6.45,
             altitude_km=np.nan, utc='1985-06-15', kind='landing site; not a measured descent track')]))
    return (original, pd.concat(corrected, ignore_index=True), pd.concat(locations, ignore_index=True),
            pd.DataFrame(checks), pd.concat(flags, ignore_index=True))


def load_corrected_profiles():
    return load_audit_data()[1]


def pressure_at(profile, altitude):
    ordered = profile.sort_values('altitude_km')
    assert np.min(altitude) >= ordered.altitude_km.min() and np.max(altitude) <= ordered.altitude_km.max()
    return np.interp(altitude, ordered.altitude_km, ordered.pressure_hpa)


def plot_locations(locations):
    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.set(xlim=(55, 330), ylim=(-30, 30), xlabel='Longitude (°E)', ylabel='Latitude (°N)',
           xticks=np.arange(60, 331, 30), yticks=np.arange(-30, 31, 10))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=.25)
    ax.axhline(0, color='#7D8994', lw=.9)
    labels = {
        'Vega 1 Balloon': ((110, 21), 'Vega 1 balloon\n11–12 Jun 1985'),
        'Vega 2 Balloon': ((110, -22), 'Vega 2 balloon\n15–16 Jun 1985')}
    for name, (text_xy, text) in labels.items():
        g = locations[locations.mission.eq(name)].sort_values('utc')
        color = COLORS[name]
        ax.plot(g.longitude_deg_e, g.latitude_deg_n, color=color, lw=2.5)
        ax.scatter(g.longitude_deg_e.iloc[0], g.latitude_deg_n.iloc[0], marker='o', s=65,
                   facecolor='white', edgecolor=color, linewidth=1.8, zorder=4)
        ax.scatter(g.longitude_deg_e.iloc[-1], g.latitude_deg_n.iloc[-1], marker='s', s=50, color=color, zorder=4)
        middle = g.iloc[len(g)//2]
        ax.annotate(text, (middle.longitude_deg_e, middle.latitude_deg_n), xytext=text_xy,
                    ha='center', va='center', fontsize=13, color=color,
                    arrowprops=dict(arrowstyle='-', color=color))
        i = len(g)//3
        ax.annotate('', xy=(g.longitude_deg_e.iloc[i+5], g.latitude_deg_n.iloc[i+5]),
                    xytext=(g.longitude_deg_e.iloc[i], g.latitude_deg_n.iloc[i]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ak = locations[locations.mission.eq('Akatsuki RS')].sort_values('altitude_km')
    lon, lat = [np.interp(54, ak.altitude_km, ak[field]) for field in ['longitude_deg_e', 'latitude_deg_n']]
    ax.plot(ak.longitude_deg_e, ak.latitude_deg_n, color=COLORS['Akatsuki RS'], lw=3)
    ax.scatter(lon, lat, color=COLORS['Akatsuki RS'], marker='D', s=65, zorder=4)
    ax.annotate('Akatsuki tangent point\n23 Jul 2016 · 54 km', (lon, lat), xytext=(225, 21),
                ha='center', va='center', fontsize=13, color=COLORS['Akatsuki RS'],
                arrowprops=dict(arrowstyle='-', color=COLORS['Akatsuki RS']))
    for name, marker, text, text_xy in [
        ('Pioneer Venus Large Probe', 'o', 'Pioneer Venus large probe\n9 Dec 1978 · impact site', (284, -17)),
        ('Vega 2 Lander', '*', 'Vega 2 lander\n15 Jun 1985 · landing site', (211, -22))]:
        row = locations[locations.mission.eq(name)].iloc[0]
        ax.scatter(row.longitude_deg_e, row.latitude_deg_n, color=COLORS[name], marker=marker, s=100, zorder=4)
        ax.annotate(text, (row.longitude_deg_e, row.latitude_deg_n), xytext=text_xy,
                    ha='center', va='center', fontsize=13, color=COLORS[name],
                    arrowprops=dict(arrowstyle='-', color=COLORS[name]))
    for spine in ax.spines.values():
        spine.set_visible(True)
    fig.tight_layout()
    save(fig, HERE/'图2a_任务位置')


def main():
    OUT.mkdir(exist_ok=True)
    setup()
    old, corrected, locations, checks, flags = load_audit_data()
    for name, frame in [('修正后剖面', corrected), ('任务位置', locations), ('逐源核对', checks), ('气球原散点质量标记', flags)]:
        frame.to_csv(OUT/f'{name}.csv', index=False, encoding='utf-8-sig')
    rows = []
    for name in ['Akatsuki RS', 'Pioneer Venus Large Probe', 'Vega 2 Lander']:
        rows.append(dict(mission=name, original_pressure_at_54_hpa=pressure_at(old[old.mission.eq(name)], 54),
                         corrected_pressure_at_54_hpa=pressure_at(corrected[corrected.mission.eq(name)], 54),
                         height_basis=corrected[corrected.mission.eq(name)].height_basis.iloc[0]))
    pd.DataFrame(rows).to_csv(OUT/'54km气压比较.csv', index=False, encoding='utf-8-sig')
    plot_locations(locations)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    heights = np.linspace(49, 62, 261)
    reference = corrected[corrected.mission.eq('Pioneer Venus Large Probe')]
    differences = []
    for name, label in [('Akatsuki RS', 'Akatsuki, geometric height'),
                        ('Pioneer Venus Large Probe', 'Pioneer, geometric height'),
                        ('Vega 2 Lander', 'Vega 2, native height')]:
        group = corrected[corrected.mission.eq(name)]
        axes[0].plot(group.pressure_hpa, group.altitude_km, color=COLORS[name], lw=2, label=label,
                     ls='--' if name == 'Vega 2 Lander' else '-')
        if name != 'Pioneer Venus Large Probe':
            difference = 100*(pressure_at(group, heights)/pressure_at(reference, heights)-1)
            axes[1].plot(difference, heights, color=COLORS[name], lw=2,
                         ls='--' if name == 'Vega 2 Lander' else '-', label=label)
            differences.append(pd.DataFrame(dict(mission=name, altitude_km=heights, relative_difference_pct=difference)))
    for name, marker in [('Vega 1 Balloon', '^'), ('Vega 2 Balloon', 'D')]:
        g = corrected[corrected.mission.eq(name)]
        axes[0].scatter(g.pressure_hpa, g.altitude_km, marker=marker, color=COLORS[name], s=22)
    axes[0].set(xlim=(100, 1500), ylim=(49, 62), xlabel='Pressure (hPa)', ylabel='Altitude (km)')
    axes[1].set(xlabel='Pressure difference from Pioneer (%)', ylim=(49, 62))
    axes[1].axvline(0, color='gray', lw=1)
    for ax, letter in zip(axes, 'ab'):
        ax.set_title(letter, loc='left', fontsize=22, fontweight='bold')
        ax.grid(alpha=.25)
        ax.legend(fontsize=10, loc='upper right' if letter == 'a' else 'upper center')
    fig.tight_layout()
    save(fig, HERE/'图2a_剖面差异核查')
    pd.concat(differences).to_csv(OUT/'气压相对差异.csv', index=False, encoding='utf-8-sig')
    paths = [HERE/'金星-1.csv', RAW/'rs_20160723_022600_udsc64_l4_ae_v10.tab',
             RAW/'rs_20160723_022600_udsc64_l4_ae_v10.lbl', RAW/'las_low_atm_state.csv',
             RAW/'las_low_atm_state_descr.txt', VEGA/'vg1bl_rdr.dat', VEGA/'vg2bl_rdr.dat', VEGA/'vg2lr.dat']
    manifest = {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'原始文件SHA256.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(checks.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
