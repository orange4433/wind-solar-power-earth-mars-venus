import os
import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import pandas as pd
import numpy as np
import requests
from tqdm import tqdm


# ==========================
# Parameters
# ==========================

BASE_URL = (
    "https://pds-ppi.igpp.ucla.edu/data/"
    "insight-ifg-mars/data-sc-engineering/"
)

SAVE_DIR = "./insight_engineering_data"

OUTPUT = "InSight_hourly_solar_energy.txt"

REQUEST_TIMEOUT = 60


class LinkParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return

        for name,value in attrs:
            if name.lower() == "href":
                self.links.append(value)


def local_tab_files():

    if not os.path.isdir(SAVE_DIR):
        return []

    paths = []

    for root,dirs,files in os.walk(SAVE_DIR):
        for file in files:
            if file.lower().endswith(".tab"):
                paths.append(
                    os.path.join(root,file)
                )

    return paths


def url_candidates(url):

    yield url

    if url.startswith("https://"):
        yield "http://" + url[len("https://"):]


def fetch_text(session,url):

    last_error = None

    for candidate in url_candidates(url):
        try:
            r = session.get(
                candidate,
                timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()
            return r.text,candidate

        except requests.RequestException as e:
            last_error = e

    raise RuntimeError(
        f"Could not open PDS directory: {url}\n"
        f"Last error: {last_error}"
    )


def find_remote_tab_files(base_url):

    session = requests.Session()
    base = urlparse(base_url)
    base_path = base.path.rstrip("/") + "/"
    pages = [base_url]
    seen_pages = set()
    tab_urls = []

    while pages:
        page_url = pages.pop()

        if page_url in seen_pages:
            continue

        seen_pages.add(page_url)
        html,real_page_url = fetch_text(session,page_url)

        parser = LinkParser()
        parser.feed(html)

        for href in parser.links:
            if not href or href.startswith("?") or href.startswith("#"):
                continue

            full_url = urljoin(real_page_url,href)
            parsed = urlparse(full_url)

            if parsed.netloc != base.netloc:
                continue

            clean_url = parsed._replace(
                query="",
                fragment=""
            ).geturl()

            clean_path = urlparse(clean_url).path

            if not clean_path.startswith(base_path):
                continue

            if clean_path.lower().endswith(".tab"):
                tab_urls.append(clean_url)

            elif clean_path.endswith("/") and clean_url not in seen_pages:
                pages.append(clean_url)

    return sorted(set(tab_urls))


def local_path_for_url(url):

    base_path = urlparse(BASE_URL).path.rstrip("/") + "/"
    path = urlparse(url).path

    if path.startswith(base_path):
        rel_path = path[len(base_path):]
    else:
        rel_path = os.path.basename(path)

    return os.path.join(
        SAVE_DIR,
        *rel_path.split("/")
    )


# ==========================
# Download files.
# ==========================

def download_file(url, filename, session=None):

    if os.path.exists(filename):
        return True

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    session = session or requests.Session()
    last_error = None

    for candidate in url_candidates(url):
        try:
            with session.get(
                candidate,
                stream=True,
                timeout=REQUEST_TIMEOUT
            ) as r:
                r.raise_for_status()

                with open(filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)

                return True

        except requests.RequestException as e:
            last_error = e

    print(
        "Download failed:",
        url,
        last_error
    )

    return False


def ensure_tab_files():

    if local_tab_files():
        return

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    print(
        "No local .tab files found. Searching PDS archive:",
        BASE_URL
    )

    urls = find_remote_tab_files(BASE_URL)

    if not urls:
        raise RuntimeError(
            f"No .tab files were found at PDS URL: {BASE_URL}"
        )

    print(
        "Found",
        len(urls),
        ".tab files. Downloading to:",
        os.path.abspath(SAVE_DIR)
    )

    session = requests.Session()
    downloaded = 0

    for url in tqdm(urls):
        if download_file(
            url,
            local_path_for_url(url),
            session=session
        ):
            downloaded += 1

    if downloaded == 0:
        raise RuntimeError(
            "Could not download any .tab files. "
            "Please check your network connection or download the data manually."
        )



# ==========================
# Read TAB files.
# ==========================

def read_tab(filename):

    df = pd.read_csv(
        filename,
        sep=",",
        skipinitialspace=True,
        low_memory=False
    )

    # Strip whitespace from column names.
    df.columns = [
        c.strip()
        for c in df.columns
    ]

    return df



# ==========================
# Calculate power.
# ==========================

def process_file(filename):

    df = read_tab(filename)


    # Missing values
    df = df.replace(
        [9999.900,9999.999],
        np.nan
    )


    # Time
    df["SCET_UTC"] = pd.to_datetime(
        df["SCET_UTC"],
        errors="coerce"
    )


    # Extract the sol.
    # SOL-0004M07:00:42
    sol_time = (
        df["SOL/Mars local time"]
        .astype(str)
    )


    df["SOL"] = (
        sol_time
        .str.extract(r"(SOL-\d+)")
    )


    df["MLST_hour"] = (
        sol_time
        .str.extract(r"M(\d+):")
        .astype(float)
    )


    # Voltage
    V1 = df["E-0769"]
    V2 = df["E-0789"]

    # Total solar-array current
    I = df["G-0036"]


    # Power in W
    df["Power_W"] = (
        (V1+V2)/2
        *
        I
    )


    # Time interval
    df = df.sort_values(
        "SCET_UTC"
    )


    df["dt"] = (
        df["SCET_UTC"]
        .diff()
        .dt.total_seconds()
    )


    # Remove anomalous intervals.
    df.loc[
        (df["dt"]<=0) |
        (df["dt"]>3600),
        "dt"
    ] = np.nan


    # Wh
    df["Energy_Wh"] = (
        df["Power_W"]
        *
        df["dt"]
        /
        3600
    )


    # Report Mars local hours.
    result = (
        df
        .groupby(
            [
                "SOL",
                "MLST_hour"
            ]
        )
        ["Energy_Wh"]
        .sum()
        .reset_index()
    )


    return result



# ==========================
# Main program
# ==========================


all_results=[]

ensure_tab_files()


# Assume all TAB files are already downloaded.
# Download them first if necessary.

for root,dirs,files in os.walk(SAVE_DIR):

    for file in files:

        if file.lower().endswith(".tab"):

            path=os.path.join(
                root,
                file
            )

            print(
                "processing:",
                file
            )

            try:

                r=process_file(path)

                all_results.append(r)

            except Exception as e:

                print(
                    "Error:",
                    file,
                    e
                )



# Merge.

if not all_results:
    raise RuntimeError(
        f"No .tab files were successfully processed under: {os.path.abspath(SAVE_DIR)}\n"
        "Check that the folder contains InSight engineering .tab files and that "
        "their columns include SCET_UTC, SOL/Mars local time, E-0769, E-0789, and G-0036."
    )


final=pd.concat(
    all_results,
    ignore_index=True
)


# Sort.

final=final.sort_values(
    [
        "SOL",
        "MLST_hour"
    ]
)



# Write text output.

with open(
    OUTPUT,
    "w"
) as f:

    f.write(
        "SOL_hour\tEnergy_Wh\n"
    )

    for _,row in final.iterrows():

        name=(
            f"{row['SOL']}_hour_"
            f"{int(row['MLST_hour']):02d}"
        )

        f.write(
            f"{name}\t"
            f"{row['Energy_Wh']:.4f}\n"
        )


print(
    "Finished:",
    OUTPUT
)
