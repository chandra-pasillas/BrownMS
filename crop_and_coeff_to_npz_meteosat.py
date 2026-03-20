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


SEVIRI_channels = ['IR_039', 'IR_087', 'IR_108', 'IR_120']
lat_long = ['latitude', 'longitude']
crop_channels = SEVIRI_channels
all_channels = SEVIRI_channels


def parse_filename_meteosat(path):
    """
    Example:
    MSG2-SEVI-MSG15-0100-NA-20231023221240.579000000Z-NA.nat
    """
    name = path if isinstance(path, str) else path.name

    if not name.endswith(".nat"):
        raise ValueError(f"Unexpected Meteosat filename format: {name}")

    stem = name[:-4]
    parts = stem.split('-')

    if len(parts) < 6:
        raise ValueError(f"Unexpected Meteosat filename format: {name}")

    satellite = parts[0]   # MSG2
    date_token = parts[5]  # 20231023221240.579000000Z

    if '.' in date_token:
        date_token = date_token.split('.')[0]
    date_token = date_token.replace('Z', '')

    start = datetime.datetime.strptime(date_token, "%Y%m%d%H%M%S")
    startfix = start.strftime("%Y%m%d%H%M%S")

    return {
        'filename': name,
        'channels': "SEVIRI",
        'satellite': satellite,
        'sat_and_start': f'{satellite} {start}',
        'start': start,
        'end': start,
        'path': str(path),
        'datetime': startfix
    }


def group_meteosat_by_time_sat(msg_dir):
    mets = [
        parse_filename_meteosat(f)
        for f in msg_dir.iterdir()
        if f.is_file() and f.suffix == '.nat' and f.name.startswith("MSG") and "SEVI" in f.name
    ]

    def red(d, met):
        d[met['datetime']].append(met)
        return d

    d = ft.reduce(red, mets, defaultdict(list))
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


def apply_sbaf_coefficients_meteosat(data: dict, satellite: str) -> dict:
    """
    Meteosat-9 / SEVIRI -> VIIRS-like matched channels
    Using ALL-season coefficients from SBAF table
    Y = m*x + b
    """
    coeffs = {
        "MSG2": {
            "M13": ("IR_039", 1.0034, -0.610),
            "M14": ("IR_087", 1.0046, -0.823),
            "M15": ("IR_108", 0.9982,  0.325),
            "M16": ("IR_120", 0.9977,  0.419),
        },
        "Meteosat-9": {
            "M13": ("IR_039", 1.0034, -0.610),
            "M14": ("IR_087", 1.0046, -0.823),
            "M15": ("IR_108", 0.9982,  0.325),
            "M16": ("IR_120", 0.9977,  0.419),
        },
    }

    if satellite not in coeffs:
        return data

    for viirs_band, (source_band, m, b) in coeffs[satellite].items():
        data[viirs_band] = m * data[source_band] + b

    return data


def process_set_meteosat(grouped_files, curr_idx, total_groups, lat, lon):
    log.info(
        f'{rgb(255,0,0)}Processing{reset} timestep '
        f'{bold}{curr_idx + 1}/{total_groups}{reset}'
    )

    met_dt = grouped_files[0]["datetime"]
    satellite = grouped_files[0]["satellite"]
    print("the Meteosat dt is", met_dt)

    metfiles = [f["path"] for f in grouped_files]
    print("Meteosat files are", metfiles)

    master_scene = Scene(filenames={'seviri_l1b_native': metfiles})
    master_scene.load(SEVIRI_channels)
    print(master_scene.available_dataset_names())

    bbox = make_km_bbox(lat, lon, box_size_km=1000.0)
    print("Cropping with bbox:", bbox)

    master_scene_cropped = master_scene.crop(ll_bbox=bbox)

    measurement = master_scene_cropped['IR_039']

    global longitude
    global latitude
    longitude, latitude = measurement.attrs['area'].get_lonlats()

    print(master_scene_cropped.available_dataset_names())
    print("datasets are now loaded")

    log.info(f'Extracting cropped Meteosat bands for {blue}{met_dt}{reset}')
    t = time.time()

    data = {}

    for ch in all_channels:
        arr = master_scene_cropped[ch].values

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

    data = apply_sbaf_coefficients_meteosat(data, satellite)
    data['channels'] = list(data)
    data['filenames'] = metfiles
    data["datetime"] = met_dt
    return data


def process_folder(folder_path: Path, lat: float, lon: float):
    print(f"\nProcessing folder: {folder_path}")
    print(f"Using center lat/lon: {lat}, {lon}")

    MET_dict = group_meteosat_by_time_sat(folder_path)
    matched = MET_dict

    if not matched:
        print(f"No Meteosat files found in {folder_path}")
        return

    datas = []
    dcount = 0

    for dt_key in sorted(matched):
        log.info(f"DTG is {dt_key}, matched files are {matched[dt_key]}")
        try:
            datas.append(process_set_meteosat(matched[dt_key], dcount, len(matched), lat, lon))
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

    for case_dir in [p for p in case_dirs if p.name in {
        'Case_18', 'Case_19', 'Case_20',
        'Case_31', 'Case_32', 'Case_33', 'Case_34', 'Case_35', 'Case_36', 'Case_37', 'Case_38'
    }]:
    # for case_dir in [p for p in case_dirs if p.name in {'Case_38'}]:

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