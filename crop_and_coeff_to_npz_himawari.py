from __future__ import annotations

import csv
import math
import numpy as np
import functools as ft
from pathlib import Path
import time
import datetime
from collections import defaultdict
from satpy import Scene
import crop
import common
from common import log, rgb, reset, blue, orange, bold
import gc


AHI_channels = ['B07', 'B11', 'B13', 'B14', 'B15']
lat_long = ['latitude', 'longitude']
crop_channels = AHI_channels
all_channels = AHI_channels


def parse_filename_himawari(path):
    """
    Example:
    HS_H09_20230113_1500_B07_FLDK_R20_S0110.DAT.bz2
    """
    name = path if isinstance(path, str) else path.name
    parts = name.split('_')

    if len(parts) < 6:
        raise ValueError(f"Unexpected Himawari filename format: {name}")

    satellite = parts[1]          # H09
    date_str = parts[2]           # 20230113
    time_str = parts[3]           # 1500
    band = parts[4]               # B07
    area = parts[5]               # FLDK

    start = datetime.datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
    startfix = start.strftime("%Y%m%d%H%M%S")

    return {
        'filename': name,
        'channels': band,
        'satellite': satellite,
        'sat_and_start': f'{satellite} {start}',
        'start': start,
        'end': start,
        'path': str(path),
        'datetime': startfix,
        'area': area,
    }


def group_himawari_by_time_sat(him_dir):
    hims = [
        parse_filename_himawari(f)
        for f in him_dir.iterdir()
        if f.is_file() and f.name.startswith("HS_H09_") and f.name.endswith(".DAT.bz2")
    ]

    def red(d, him):
        d[him['datetime']].append(him)
        return d

    d = ft.reduce(red, hims, defaultdict(list))
    return d


def hourly_targets(date_str: str, start_z: str, end_z: str) -> list[datetime.datetime]:
    start_dt = datetime.datetime.strptime(
        f"{date_str} {start_z.zfill(4)}",
        "%m/%d/%Y %H%M"
    )
    end_dt = datetime.datetime.strptime(
        f"{date_str} {end_z.zfill(4)}",
        "%m/%d/%Y %H%M"
    )
    if end_dt < start_dt:
        end_dt += datetime.timedelta(days=1)

    t = start_dt.replace(minute=0, second=0, microsecond=0)
    if t < start_dt:
        t += datetime.timedelta(hours=1)

    targets = []
    while t <= end_dt:
        targets.append(t)
        t += datetime.timedelta(hours=1)

    return targets


def folder_name_from_dt(dt: datetime.datetime) -> str:
    return f"{dt.strftime('%Y')}{dt.strftime('%j')}_{dt.strftime('%H')}Z"


def build_case_folder_latlon_lookup(csv_file: Path) -> dict[str, dict[str, tuple[float, float]]]:
    lookup: dict[str, dict[str, tuple[float, float]]] = {}

    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        rows = []
        for row in reader:
            clean_row = {k.strip(): v for k, v in row.items()}
            rows.append(clean_row)

    for i, row in enumerate(rows, start=1):
        case_name = f"Case_{i}"

        date_str = row["Date"].strip()
        start_z = row["Start (Zulu)"].strip()
        end_z = row["End (Zulu)"].strip()
        lat = float(row["Lat"])
        lon = float(row["Lon"])

        lookup[case_name] = {}

        for dt in hourly_targets(date_str, start_z, end_z):
            folder_name = folder_name_from_dt(dt)
            lookup[case_name][folder_name] = (lat, lon)

    return lookup


def make_km_bbox(lat_center: float, lon_center: float, box_size_km: float = 1000.0):
    half_side_km = box_size_km / 2.0

    dlat = half_side_km / 111.32
    dlon = half_side_km / (111.32 * math.cos(math.radians(lat_center)))

    lat_min = lat_center - dlat
    lat_max = lat_center + dlat
    lon_min = lon_center - dlon
    lon_max = lon_center + dlon

    return (lon_min, lat_min, lon_max, lat_max)


def apply_sbaf_coefficients_himawari(data: dict, satellite: str) -> dict:
    """
    Himawari-9 -> VIIRS-like matched channels
    Using ALL-season coefficients from SBAF Table.xlsx
    """
    coeffs = {
        "H09": {
            "M13": ("B07", 1.0316, -5.689),
            "M14": ("B11", 1.0009, -0.169),
            "M15": ("B13", 1.0064, -1.153),
            "M16": ("B15", 0.9795, 3.698),
        }
    }

    if satellite not in coeffs:
        return data

    for viirs_band, (source_band, m, b) in coeffs[satellite].items():
        data[viirs_band] = m * data[source_band] + b

    return data


def process_set_himawari(grouped_files, curr_idx, total_groups, lat, lon):
    log.info(
        f'{rgb(255,0,0)}Processing{reset} timestep '
        f'{bold}{curr_idx + 1}/{total_groups}{reset}'
    )

    him_dt = grouped_files[0]["datetime"]
    satellite = grouped_files[0]["satellite"]
    print("the Himawari dt is", him_dt)

    himfiles = [f["path"] for f in grouped_files]
    print("Himawari files are", himfiles)

    master_scene = Scene(filenames={'ahi_hsd': himfiles})
    master_scene.load(AHI_channels)
    print(master_scene.available_dataset_names())

    bbox = make_km_bbox(lat, lon, box_size_km=1000.0)
    print("Cropping with bbox:", bbox)

    master_scene_cropped = master_scene.crop(ll_bbox=bbox)

    measurement = master_scene_cropped['B07']

    global longitude
    global latitude
    longitude, latitude = measurement.attrs['area'].get_lonlats()

    print(master_scene_cropped.available_dataset_names())
    print("datasets are now loaded")




    log.info(f'Extracting cropped Himawari bands for {blue}{him_dt}{reset}')
    t = time.time()

    data = {}

    for ch in all_channels:
        arr = master_scene_cropped[ch].values

        # Convert masked arrays to normal numpy arrays with NaN fill
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        arr = np.asarray(arr, dtype=np.float32)

        print(
            f"{ch}: shape={arr.shape}, finite={np.isfinite(arr).sum()}, "
            f"min={np.nanmin(arr)}, max={np.nanmax(arr)}"
        )

        data[ch] = arr

    log.debug(
        f'Band extraction took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds'
    )

    data = apply_sbaf_coefficients_himawari(data, satellite)
    data['channels'] = list(data)
    data['filenames'] = himfiles
    data["datetime"] = him_dt
    return data






def process_folder(folder_path: Path, lat: float, lon: float):
    print(f"\nProcessing folder: {folder_path}")
    print(f"Using center lat/lon: {lat}, {lon}")

    HIM_dict = group_himawari_by_time_sat(folder_path)
    matched = HIM_dict

    if not matched:
        print(f"No Himawari files found in {folder_path}")
        return

    datas = []
    dcount = 0

    for dt_key in sorted(matched):
        log.info(f"DTG is {dt_key}, matched files are {matched[dt_key]}")
        try:
            datas.append(process_set_himawari(matched[dt_key], dcount, len(matched), lat, lon))
            dcount += 1
        except KeyError:
            log.info(f"skipped {dcount} as there was no matching data")
            continue
        except IndexError:
            log.info(f"skipped {dcount} due to an incomplete dataset")
            continue
        except ValueError as e:
            log.info(f"skipped {dcount} due to crop/value error: {e}")
            continue

    if not datas:
        print(f"No valid datasets produced for {folder_path}")
        return

    channels = ['M13', 'M14', 'M15', 'M16']
    print("the channels are", channels)
    print("starting the stacking")

    for c in channels:
        if c not in datas[0]:
            raise KeyError(f"Expected coefficient-adjusted channel {c} was not created")

    case1 = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
    case1['latitude'] = latitude
    case1['longitude'] = longitude
    case1['channels'] = channels
    case1['samples'] = [d["datetime"] for d in datas]
    case1['ABItimes'] = [d['datetime'] for d in datas]

    filename = folder_path / f'{folder_path.name}_FD_REDUCEDv2ALL.npz'
    log.info(
        f'Writing {blue}{filename.name}{reset}\n'
        f'{orange}Channels{reset} {channels}\n{orange}'
    )
    np.savez_compressed(filename, **case1)
    log.info(f'Wrote {blue}{filename.name}{reset}')

    del case1, datas
    gc.collect()


def main():
    root_dir = Path(r"D:\MLNVI Batch\AFIT_BROWN").resolve()
    csv_file = Path(r"D:\MLNVI Batch\Fog_Cases.csv").resolve()

    case_lookup = build_case_folder_latlon_lookup(csv_file)

    print("Built lookup for", len(case_lookup), "cases")

    case_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("Case_")]
    )

    # Himawari cases only for now
    for case_dir in [p for p in case_dirs if p.name in {'Case_27', 'Case_28', 'Case_29', 'Case_30'}]:
    #for case_dir in [p for p in case_dirs if p.name in {'Case_30'}]:

        case_name = case_dir.name

        if case_name not in case_lookup:
            print(f"Skipping {case_name}: no matching CSV row found")
            continue

        print(f"\n===== Processing {case_name} =====")

        timestamp_dirs = sorted([p for p in case_dir.iterdir() if p.is_dir()])

        for folder in timestamp_dirs:
            folder_name = folder.name

            if folder_name not in case_lookup[case_name]:
                print(f"Skipping {folder}: no matching lat/lon found in CSV lookup")
                continue

            lat, lon = case_lookup[case_name][folder_name]
            process_folder(folder, lat, lon)


if __name__ == "__main__":
    main()
