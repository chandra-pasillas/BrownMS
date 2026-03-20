from __future__ import annotations

import csv
import math
import re
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


ABI_channels = ['C07', 'C11', 'C13', 'C14', 'C15']
lat_long = ['latitude', 'longitude']
crop_channels = ABI_channels
all_channels = ABI_channels

format_abi = '%Y%j%H%M%S'


def parse_time(time_str):
    return datetime.datetime.strptime(time_str, format_abi)


def parse_filename_abi(path):
    name = path if isinstance(path, str) else path.name
    n = name.split('_')
    start = parse_time(n[3][1:-1])
    startfix = start.strftime('%Y%m%d%H%M%S')
    end = parse_time(n[4][1:-1])
    return {
        'filename': name,
        'channels': n[0],
        'satellite': n[2],
        'sat_and_start': f'{n[2]} {start}',
        'start': start,
        'end': end,
        'path': str(path),
        'datetime': startfix
    }


def group_abi_by_time_sat(abi_dir):
    abis = [
        parse_filename_abi(f)
        for f in abi_dir.iterdir()
        if f.is_file() and f.suffix == '.nc' and 'OR_ABI' in f.name
    ]

    def red(d, abi):
        d[abi['datetime']].append(abi)
        return d

    d = ft.reduce(red, abis, defaultdict(list))
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
    """
    Build a nested lookup like:
    {
        "Case_1": {
            "2023039_03Z": (lat, lon),
            "2023039_04Z": (lat, lon),
        },
        "Case_2": {
            ...
        }
    }

    Assumes CSV row order matches Case_1, Case_2, ...
    """
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
    """
    Return ll_bbox in Satpy order:
    (lon_min, lat_min, lon_max, lat_max)
    """
    half_side_km = box_size_km / 2.0

    dlat = half_side_km / 111.32
    dlon = half_side_km / (111.32 * math.cos(math.radians(lat_center)))

    lat_min = lat_center - dlat
    lat_max = lat_center + dlat
    lon_min = lon_center - dlon
    lon_max = lon_center + dlon

    return (lon_min, lat_min, lon_max, lat_max)


# Add coefficients

def apply_sbaf_coefficients(data: dict, satellite: str) -> dict:
    """
    Convert ABI channels to VIIRS-like matched channels using:
    Y = m*x + b

    Using ALL-season coefficients.
    """

    coeffs = {
        "G18": {
            "M13": ("C07", 1.0193, -3.466),
            "M14": ("C11", 0.9989, 0.199),
            "M15": ("C13", 1.0023, -0.410),
            "M16": ("C15", 1.0009, -0.074),
        },
        "G16": {
            "M13": ("C07", 1.0254, -4.580),
            "M14": ("C11", 0.9975, 0.447),
            "M15": ("C13", 1.0043, -0.765),
            "M16": ("C15", 1.0025, -0.145),
        },
    }

    if satellite not in coeffs:
        return data

    for viirs_band, (source_band, m, b) in coeffs[satellite].items():
        data[viirs_band] = m * data[source_band] + b

    return data






def process_set_ABI(grouped_files, curr_idx, total_groups, lat, lon):
    log.info(
        f'{rgb(255,0,0)}Processing{reset} timestep '
        f'{bold}{curr_idx + 1}/{total_groups}{reset}'
    )

    abi_dt = grouped_files[0]["datetime"]
    satellite = grouped_files[0]["satellite"]
    print("the ABI dt is", abi_dt)


    abifiles = [f["path"] for f in grouped_files]
    print("ABI files are", abifiles)

    master_scene = Scene(filenames={'abi_l1b': abifiles})
    master_scene.load(ABI_channels)
    print(master_scene.available_dataset_names())

    bbox = make_km_bbox(lat, lon, box_size_km=1000.0)
    print("Cropping with bbox:", bbox)

    master_scene_cropped = master_scene.crop(ll_bbox=bbox)

    measurement = master_scene_cropped['C07']

    global longitude
    global latitude
    longitude, latitude = measurement.attrs['area'].get_lonlats()

    print(master_scene_cropped.available_dataset_names())
    print("datasets are now loaded")

    log.info(f'Cropping nan edges of {blue}{abi_dt}{reset}')
    t = time.time()
    data = crop.crop_nan_edges_GOES(master_scene_cropped, all_channels, all_channels)
    log.debug(
        f'Cropping nan edges took {rgb(255,0,0)}{time.time() - t:.2f}{reset} seconds'
    )

# Add this data apply sbaf
    #data = apply_sbaf_coefficients(data, grouped_files[0]["satellite"])
    data = apply_sbaf_coefficients(data, satellite)
    data['channels'] = list(data)
    data['filenames'] = abifiles
    data["datetime"] = abi_dt
    return data


def process_folder(folder_path: Path, lat: float, lon: float):
    """
    Process one timestamp folder containing .nc files.
    """
    print(f"\nProcessing folder: {folder_path}")
    print(f"Using center lat/lon: {lat}, {lon}")

    ABI_dict = group_abi_by_time_sat(folder_path)
    matched = ABI_dict

    if not matched:
        print(f"No ABI files found in {folder_path}")
        return

    datas = []
    dcount = 0

    for dt_key in sorted(matched):
        log.info(f"DTG is {dt_key}, matched files are {matched[dt_key]}")
        try:
            datas.append(process_set_ABI(matched[dt_key], dcount, len(matched), lat, lon))
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

    #channels = datas[0]['channels']
    #print("the channels are", channels)
    #print("starting the stacking")

    #case1 = {c: np.stack(tuple(d[c] for d in datas)) for c in channels}
    #case1['latitude'] = latitude
    #case1['longitude'] = longitude
    #case1['channels'] = channels

# trying this
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
    # Root folder containing Case_1, Case_2, ...
    #root_dir = Path("/home/nathanstanford/AFIT_Brown/AFIT_BROWN").resolve()

    root_dir = Path(r"D:\MLNVI Batch\AFIT_BROWN").resolve()

    # CSV containing Date, Start (Zulu), End (Zulu), Lat, Lon
    #csv_file = Path("/home/nathanstanford/AFIT_Brown/Fog_Cases.csv").resolve()

    csv_file = Path(r"D:\MLNVI Batch\Fog_Cases.csv").resolve()

    case_lookup = build_case_folder_latlon_lookup(csv_file)

    print("Built lookup for", len(case_lookup), "cases")

    case_dirs = sorted(
        [p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("Case_")]
    )

    for case_dir in case_dirs:
    #for case_dir in [p for p in case_dirs if p.name in {'Case_14','Case_15','Case_16','Case_17'}]:

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