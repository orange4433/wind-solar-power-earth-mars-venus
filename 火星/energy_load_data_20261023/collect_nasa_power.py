"""Collect NASA's dated InSight energy summaries; do not interpolate daily data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import csv
import re

from bs4 import BeautifulSoup
import requests

HERE = Path(__file__).resolve().parent
RAW = HERE / 'nasa_source_pages'
SOL0 = datetime.fromisoformat('2018-11-26T05:10:50.336037+00:00')
SOL_SECONDS = 88775.244
ENERGY = re.compile(r'average\s+(?:of|between)\s*~?(\d+)(?:\s*(?:and|[-–])\s*(\d+))?\s+watt.hours', re.I)
DATE = re.compile(r'(?:As of|On)\s+([A-Za-z]+\.?\s+\d{1,2},\s+2022)', re.I)


def read_page(url):
    path = RAW / ((url.rstrip('/').split('/')[-1] or 'index') + '.html')
    if not path.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        path.write_bytes(response.content)
    return BeautifulSoup(path.read_bytes(), 'html.parser')


def parse_summary(text):
    energy, date = ENERGY.search(text), DATE.search(text)
    if not energy or not date:
        raise ValueError(f'Unrecognized NASA energy summary: {text[:300]}')
    date_text = date[1].replace('.', '').replace('Sept ', 'Sep ')
    for fmt in ('%B %d, %Y', '%b %d, %Y'):
        try:
            day = datetime.strptime(date_text, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    else:
        raise ValueError(date_text)
    low, high = int(energy[1]), int(energy[2] or energy[1])
    assert 0 < low <= high < 5000
    tau = re.search(r'was estimated at\s+([\d.]+)', text)
    return dict(
        observation_date=day.date().isoformat(),
        solar_energy_lower_wh_per_sol=low,
        solar_energy_upper_wh_per_sol=high,
        tau=float(tau[1].rstrip('.')) if tau else '',
        sol_at_utc_date_start=round((day-SOL0).total_seconds()/SOL_SECONDS, 3),
        sol_at_utc_date_end=round((day+timedelta(days=1)-SOL0).total_seconds()/SOL_SECONDS, 3),
        record_type='NASA reported approximate average; not a continuous daily telemetry series',
    )


def collect_post(url):
    soup = read_page(url)
    # Read the body, not the title: the July 20 post reports July 16 observations.
    paragraphs = [p.get_text(' ', strip=True) for p in soup.find_all('p')]
    paragraph = next(p for p in paragraphs if ENERGY.search(p) and DATE.search(p))
    row = parse_summary(paragraph)
    row.update(source_title=soup.find('h1').get_text(' ', strip=True),
               source_url=url, source_paragraph=paragraph, checked_date='2026-10-23')
    return row


def main():
    RAW.mkdir(exist_ok=True)
    urls = set()
    for page in range(1, 4):
        url = 'https://science.nasa.gov/blogs/insight/' + (f'page/{page}/' if page > 1 else '')
        soup = read_page(url)
        urls.update(a['href'] for a in soup.find_all('a', href=True)
                    if '/2022/' in a['href'] and 'power-level' in a['href'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(pool.map(collect_post, sorted(urls)), key=lambda r: (r['observation_date'], r['source_url']))
    assert rows and rows[-1]['observation_date'] == '2022-12-12'
    unique = {}
    for row in rows:
        date = row['observation_date']
        if date in unique:
            previous = unique[date]
            for field in ('solar_energy_lower_wh_per_sol', 'solar_energy_upper_wh_per_sol', 'tau'):
                assert previous[field] == row[field], (date, field)
            previous['source_url'] += ' | ' + row['source_url']
        else:
            unique[date] = row.copy()
    for name, records in [('nasa_power_posts.csv', rows), ('nasa_solar_energy_2022.csv', list(unique.values()))]:
        with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f'{len(rows)} source posts; {len(unique)} distinct observation dates')
    for row in unique.values():
        print(row['observation_date'], row['solar_energy_lower_wh_per_sol'], row['solar_energy_upper_wh_per_sol'])


if __name__ == '__main__':
    assert parse_summary('On July 16, 2022, generating an average of 385-400 watt-hours')['solar_energy_upper_wh_per_sol'] == 400
    assert parse_summary('As of Nov. 27, 2022, average between 285 and 295 watt-hours')['observation_date'] == '2022-11-27'
    assert parse_summary('As of Dec. 12, 2022, average of ~285 watt-hours')['solar_energy_lower_wh_per_sol'] == 285
    main()
