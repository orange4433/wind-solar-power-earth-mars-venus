from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent


EDR_COLUMNS = [
    "time_s",
    "record_type",
    "pressure_mbar",
    "pressure_flag",
    "vertical_wind_ms",
    "vertical_wind_flag",
    "ambient_temperature_k",
    "ambient_temperature_flag",
    "nephelometer_temperature_k",
    "nephelometer_temperature_flag",
    "nephelometer_backscatter_1e-8_m_sr",
    "nephelometer_backscatter_flag",
]

RDR_COLUMNS = [
    "time_s",
    "pressure_hpa",
    "pressure_flag",
    "atmospheric_temperature_k",
    "atmospheric_temperature_flag",
    "entropy_j_per_k",
    "entropy_flag",
    "nephelometer_backscatter_1e-8_m_sr",
    "nephelometer_backscatter_flag",
    "radius_km",
    "radius_flag",
    "latitude_rad",
    "latitude_flag",
    "longitude_rad",
    "longitude_flag",
    "zonal_wind_u_ms",
    "zonal_wind_u_flag",
    "meridional_wind_v_ms",
    "meridional_wind_v_flag",
    "vertical_wind_air_wa_ms",
    "vertical_wind_air_wa_flag",
    "vertical_speed_balloon_wb_ms",
    "vertical_speed_balloon_wb_flag",
]

LANDER_COLUMNS = [
    "record_number",
    "time_s",
    "pressure_bar",
    "temperature_k",
    "altitude_m",
]

DATA_FILES = {
    "vg1bl_edr.dat": EDR_COLUMNS,
    "vg2bl_edr.dat": EDR_COLUMNS,
    "vg1bl_rdr.dat": RDR_COLUMNS,
    "vg2bl_rdr.dat": RDR_COLUMNS,
    "vg2lr.dat": LANDER_COLUMNS,
}


def read_table(filename: str) -> pd.DataFrame:
    """Read one VEGA PDS ASCII table into a DataFrame."""
    path = DATA_DIR / filename
    columns = DATA_FILES[filename]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns)

    if "time_s" in df.columns:
        df.insert(df.columns.get_loc("time_s") + 1, "time_h", df["time_s"] / 3600.0)

    if "latitude_rad" in df.columns:
        df["latitude_deg"] = df["latitude_rad"] * 180.0 / 3.141592653589793
    if "longitude_rad" in df.columns:
        df["longitude_deg"] = df["longitude_rad"] * 180.0 / 3.141592653589793

    if "nephelometer_backscatter_1e-8_m_sr" in df.columns:
        df["nephelometer_backscatter_m-1_sr-1"] = (
            df["nephelometer_backscatter_1e-8_m_sr"] / 1e8
        )

    return df


def read_all() -> dict[str, pd.DataFrame]:
    """Read all DAT files in this folder."""
    return {filename: read_table(filename) for filename in DATA_FILES}


def main() -> None:
    tables = read_all()

    for filename, df in tables.items():
        print(f"\n=== {filename} ===")
        print(f"rows: {len(df)}, columns: {len(df.columns)}")
        print(df.head(5).to_string(index=False))

        csv_path = DATA_DIR / f"{Path(filename).stem}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"saved: {csv_path.name}")

    print("\nFlag meaning used by the PDS labels:")
    print("0 = no data; 1 = erroneous; 2/3 = good but uncertain; 4 = high quality")


if __name__ == "__main__":
    main()
