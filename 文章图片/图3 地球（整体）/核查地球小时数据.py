"""
Audit aggregation from local Electricity Maps zone caches; write only to a separate verification directory.
Run this script directly. Downloaded data, country tables, and original coefficient tables are not overwritten.
"""
from pathlib import Path
import sys
import json
import hashlib
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / '地球/electricity maps'
OUT = HERE / '独立验证'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(HERE.parent))
from merge_zones_to_countries import country_sources
from figure_style import setup

YEARS = [2022, 2023, 2024, 2025]
CASES = [('South America', 2025, '南美_2025'), ('Asia', 2025, '亚洲_2025'), ('Asia', 2022, '亚洲_2022')]
WIND, SOLAR = '#2477A5', '#D88B24'


def tau(frame):
    pair = frame[['wind_mw', 'solar_mw']].dropna()
    if len(pair) < 2 or (pair.nunique() < 2).any():
        return np.nan
    return float(kendalltau(pair.wind_mw, pair.solar_mw, variant='b').statistic)


def brute_tau(pair):
    """Independently verify tau-b tie handling by enumerating all pairs in a small sample."""
    x, y = pair.dropna().to_numpy().T
    i, j = np.triu_indices(len(x), 1)
    dx, dy = np.sign(x[i]-x[j]), np.sign(y[i]-y[j])
    return np.sum(dx*dy)/np.sqrt(np.count_nonzero(dx)*np.count_nonzero(dy))


def load_strict(year):
    path = SOURCE / f'global_{year}_hourly_wind_solar.csv'
    zones = pd.read_csv(path, index_col='datetime_utc', parse_dates=True)
    expected = pd.date_range(f'{year}-01-01', f'{year+1}-01-01', freq='h', inclusive='left', tz='UTC')
    assert zones.index.equals(expected) and zones.columns.is_unique
    assert (zones.fillna(0).to_numpy() >= 0).all()
    np.testing.assert_allclose(zones, zones.round(2), rtol=0, atol=0, equal_nan=True)
    countries, info, issues = {}, [], []
    saved = pd.read_csv(SOURCE/f'global_{year}_hourly_wind_solar_by_country.csv', index_col=0, parse_dates=True)
    assert saved.index.equals(expected)
    for name, code, selected in country_sources(SOURCE/f'global_{year}_zones.csv'):
        assert len(selected) == len(set(selected))
        info.append(dict(country_code=code, country_name=name, zones=';'.join(selected)))
        for kind in ['wind', 'solar']:
            columns = [f'{zone}_{kind}_mw' for zone in selected]
            values = zones[columns]
            present = values.notna().sum(axis=1)
            partial = present.gt(0) & present.lt(len(selected))
            # Require every selected zone when aggregating; a partial-zone sum is not a national total.
            countries[f'{code}_{kind}_mw'] = values.sum(axis=1, min_count=len(selected))
            old_name = next(c for c in saved if c.endswith(f'[{code}]_{kind}_mw'))
            np.testing.assert_allclose(saved[old_name], values.sum(axis=1, min_count=1),
                                       rtol=0, atol=1e-7, equal_nan=True)
            if partial.any():
                issues.append(dict(year=year, country_code=code, resource=kind,
                                   partial_hours=int(partial.sum()), zones=';'.join(selected)))
    return pd.DataFrame(countries), pd.DataFrame(info), zones, issues


def pool(data, codes):
    if not codes:
        return pd.DataFrame(index=data.index, columns=['wind_mw', 'solar_mw'], dtype=float)
    values = np.stack([data[[f'{code}_{kind}_mw' for code in codes]].to_numpy()
                       for kind in ['wind', 'solar']], axis=2)
    complete = np.isfinite(values).all(axis=(1, 2))
    pair = pd.DataFrame(values.sum(axis=1), index=data.index, columns=['wind_mw', 'solar_mw'])
    pair.loc[~complete] = np.nan
    return pair


def plot_case(pair, continent, year, stem):
    """Plot the full-year hourly series, fixed first weeks of January/July, diurnal cycles, and all-hour scatter."""
    setup()
    fig = plt.figure(figsize=(17, 12))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05], hspace=.48, wspace=.24)
    full = fig.add_subplot(grid[0, :])
    jan, jul = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    cycle, scatter = fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])
    for ax, data, title in [(full, pair, 'All hourly values (UTC)'),
                            (jan, pair.loc[f'{year}-01-01':f'{year}-01-07'], '1–7 January (UTC)'),
                            (jul, pair.loc[f'{year}-07-01':f'{year}-07-07'], '1–7 July (UTC)')]:
        for key, label, color in [('wind_mw', 'Wind', WIND), ('solar_mw', 'Solar', SOLAR)]:
            ax.plot(data.index, data[key]/1000, color=color, lw=.6 if ax is full else 1.6,
                    alpha=.7 if ax is full else 1, label=label)
        ax.set(ylabel='Hourly mean power (GW)', title=title)
        ax.xaxis.set_major_locator(mdates.MonthLocator() if ax is full else mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b' if ax is full else '%d %b', tz=pair.index.tz))
        ax.legend(loc='upper right', ncols=2)
    mean_cycle = pair.groupby(pair.index.hour).mean()/1000
    for key, label, color in [('wind_mw', 'Wind', WIND), ('solar_mw', 'Solar', SOLAR)]:
        cycle.plot(mean_cycle.index, mean_cycle[key], '-o', color=color, label=label, ms=4)
    cycle.set(xlabel='UTC hour', ylabel='Mean power (GW)', title='Annual mean diurnal cycle', xticks=range(0,24,3))
    cycle.legend()
    valid = pair.dropna()/1000
    density = scatter.hexbin(valid.wind_mw, valid.solar_mw, gridsize=45, mincnt=1, bins='log', cmap='viridis')
    fig.colorbar(density, ax=scatter, label='Hours per bin', pad=.02)
    scatter.set(xlabel='Wind (GW)', ylabel='Solar (GW)', title=f'All paired hours: Kendall τ-b = {tau(pair):+.4f}')
    for ax in [full, jan, jul, cycle, scatter]:
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(True)
    fig.suptitle(f'{continent} · {year} · fixed country sample · {len(valid):,} paired hours', fontsize=19, y=.99)
    fig.subplots_adjust(top=.94, bottom=.07, left=.075, right=.97)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'{stem}_小时验证.{extension}', dpi=250, bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    cohort_path = SOURCE/'wind_solar_continent_trend_2022_2025/大洲固定国家样本.csv'
    members = pd.read_csv(cohort_path, keep_default_na=False)  # NA is Namibia's code, not a missing value.
    assert len(members) == 81 and 'NA' in members.country_code.values
    old = pd.read_csv(HERE/'七大洲风光互补系数_2022-2025_详细.csv')
    results, diagnostics, monthly, issues, eligibility = [], [], [], [], []
    annual_rows, contributions = [], []
    for year in YEARS:
        data, info, zones, year_issues = load_strict(year)
        issues.extend(year_issues)
        for row in info.itertuples(index=False):
            pair = data[[f'{row.country_code}_wind_mw', f'{row.country_code}_solar_mw']].dropna()
            eligibility.append(dict(year=year, code=row.country_code,
                                    eligible=len(pair)/len(data) >= .95 and pair.to_numpy().sum() > 0))
        saved_pool = pd.read_csv(SOURCE/f'wind_solar_continent_trend_2022_2025/大洲合并小时发电量_{year}.csv',
                                 index_col=0, parse_dates=True)
        for continent in old.continent.unique():
            codes = members.loc[members.continent.eq(continent), 'country_code'].tolist()
            pair = pool(data, codes)
            stored = old[(old.year == year) & old.continent.eq(continent)].iloc[0]
            unrounded_tau = tau(pair)
            assert np.isclose(unrounded_tau, stored.kendall_tau_b, rtol=0, atol=1e-6, equal_nan=True)
            # Downloads have 0.01 MW precision; remove floating-point summation residues to identify genuine ties.
            pair = pair.round(2)
            value = tau(pair)
            assert len(pair.dropna()) == stored.paired_hours
            np.testing.assert_allclose(pair, saved_pool[[f'{continent}_wind_mw', f'{continent}_solar_mw']],
                                       rtol=0, atol=1.1e-6, equal_nan=True)
            results.append(dict(year=year, continent=continent, country_count=len(codes),
                                paired_hours=len(pair.dropna()), expected_hours=len(pair),
                                recalculated_tau_b=value, stored_tau_b=stored.kendall_tau_b,
                                unrounded_tau_b=unrounded_tau,
                                difference=value-stored.kendall_tau_b))
            target = next((stem for cont, yr, stem in CASES if cont == continent and yr == year), None)
            if target:
                # Trace results to downloaded caches rather than only recalculating continental summary tables.
                source_rows = info[info.country_code.isin(codes)]
                for selected in source_rows.zones:
                    for zone in selected.split(';'):
                        cache = pd.read_csv(SOURCE/f'global_{year}_zone_cache/{zone}.csv', index_col=0, parse_dates=True)
                        assert cache.index.equals(zones.index)
                        np.testing.assert_allclose(cache[['wind_mw','solar_mw']], zones[[f'{zone}_wind_mw',f'{zone}_solar_mw']],
                                                   rtol=0, atol=1e-9, equal_nan=True)
                probe = pair.dropna().iloc[:168]
                assert np.isclose(brute_tau(probe), tau(probe))
                export = pair.copy()
                export['wind_solar_mw'] = pair.wind_mw + pair.solar_mw
                # Each bin is one hour: mean MW times 1 h equals MWh, numerically equal but physically distinct.
                export['wind_energy_mwh'] = pair.wind_mw
                export['solar_energy_mwh'] = pair.solar_mw
                export['paired_valid'] = pair.notna().all(axis=1)
                export.rename_axis('datetime_utc').to_csv(OUT/f'{target}_逐小时.csv', encoding='utf-8-sig')
                national = data[[f'{code}_{kind}_mw' for code in codes for kind in ['wind','solar']]]
                national.rename_axis('datetime_utc').to_csv(OUT/f'{target}_各国逐小时.csv', encoding='utf-8-sig')
                source_rows.to_csv(OUT/f'{target}_成员与分区.csv', index=False, encoding='utf-8-sig')
                for code in codes:
                    contributions.append(dict(case=target, country_code=code,
                                              wind_twh=data.loc[pair.wind_mw.notna(), f'{code}_wind_mw'].sum()/1e6,
                                              solar_twh=data.loc[pair.solar_mw.notna(), f'{code}_solar_mw'].sum()/1e6))
                daily = pair.resample('D').mean()
                daily[pair.resample('D').count().min(axis=1) != 24] = np.nan
                cycle = pair.groupby([pair.index.month, pair.index.hour]).transform('mean')
                diagnostics.append(dict(case=target, paired_hours=len(pair.dropna()), missing_hours=int(pair.isna().any(axis=1).sum()),
                                        hourly_tau_b=value, daily_mean_tau_b=tau(daily),
                                        month_hour_anomaly_tau_b=tau(pair-cycle),
                                        wind_twh=pair.wind_mw.sum()/1e6, solar_twh=pair.solar_mw.sum()/1e6))
                for month, group in pair.groupby(pair.index.month):
                    monthly.append(dict(case=target, month=month, paired_hours=len(group.dropna()), tau_b=tau(group)))
                plot_case(pair, continent, year, target)
                print('Verified', target, f'tau={value:+.8f}', flush=True)
        if year == 2025:
            for row in info.itertuples(index=False):
                record = dict(country_code=row.country_code, country_name=row.country_name, expected_hours=len(data), zones=row.zones)
                for kind in ['wind','solar']:
                    values = data[f'{row.country_code}_{kind}_mw']
                    count = int(values.notna().sum())
                    record[f'{kind}_hours'] = count
                    record[f'{kind}_observed_twh'] = values.sum(min_count=1)/1e6
                    record[f'{kind}_twh'] = record[f'{kind}_observed_twh'] if count == len(data) else np.nan
                    record[f'{kind}_status'] = 'complete' if count == len(data) else 'missing' if count == 0 else 'incomplete'
                    partial = any(i['country_code']==row.country_code and i['resource']==kind for i in year_issues)
                    if partial and count == 0:
                        record[f'{kind}_status'] = 'incomplete'
                record['wind_solar_twh'] = record['wind_twh'] + record['solar_twh']
                record['wind_solar_status'] = ('complete' if np.isfinite(record['wind_solar_twh']) else
                                                'missing' if record['wind_status']==record['solar_status']=='missing' else 'incomplete')
                annual_rows.append(record)
    eligible = pd.DataFrame(eligibility).groupby('code').eligible.agg(['all','size'])
    assert set(eligible.index[eligible['all'] & eligible['size'].eq(4)]) == set(members.country_code)
    for name, rows in [('全洲系数复核',results),('三个案例统计',diagnostics),('分月互补系数',monthly),
                       ('分区不完整问题',issues),('案例各国贡献',contributions)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(annual_rows).to_csv(HERE/'2025各国风光年发电量_核查后.csv', index=False, encoding='utf-8-sig')
    files = [cohort_path, HERE/'七大洲风光互补系数_2022-2025_详细.csv']
    files += [SOURCE/f'global_{year}_{suffix}.csv' for year in YEARS for suffix in
              ['hourly_wind_solar','hourly_wind_solar_by_country','zones']]
    (OUT/'输入文件SHA256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                                                    ensure_ascii=False, indent=2), encoding='utf-8')
    print(pd.DataFrame(diagnostics).to_string(index=False))
    print('PASS: 28 continent-year rows checked (unrounded tau tolerance 1e-6); fixed 81-member cohort independently reproduced.')


if __name__ == '__main__':
    main()
