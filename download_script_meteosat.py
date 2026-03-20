from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import eumdac

CSV_FILE = r"D:\MLNVI Batch\Fog_Cases.csv"
BASE_OUTDIR = Path(r"D:\MLNVI Batch\AFIT_BROWN")

TARGET_SATELLITE = "Meteosat-9"
# Leave empty to process every Meteosat-9 row in the CSV.
# Example: {"Case_31", "Case_32"}
TARGET_CASES: set[str] = set()

# Pick the collection that matches your Meteosat-9 service.
# Common choices from the EUMETSAT Data Store are:
#   EO:EUM:DAT:MSG:HRSEVIRI       -> MSG 0 degree
#   EO:EUM:DAT:MSG:HRSEVIRI-IODC  -> MSG Indian Ocean
COLLECTION_ID = "EO:EUM:DAT:MSG:HRSEVIRI-IODC"

# Meteosat/SEVIRI scans are usually every 15 minutes.
SEARCH_WINDOW_MINUTES = 20

# Your downstream Meteosat -> VIIRS-like mapping uses these SEVIRI channels:
# IR039, IR087, IR108, IR120
REQUIRED_CHANNEL_TOKENS = ("IR_039", "IR_087", "IR_108", "IR_120", "IR039", "IR087", "IR108", "IR120")

# Put your EUMETSAT API credentials in environment variables before running:
#   set EUMETSAT_KEY=your_consumer_key
#   set EUMETSAT_SECRET=your_consumer_secret
# Or hard-code them below if you prefer.
EUMETSAT_KEY = os.environ.get("EUMETSAT_KEY", "")
EUMETSAT_SECRET = os.environ.get("EUMETSAT_SECRET", "")


def hourly_targets(date_str: str, start_z: str, end_z: str) -> list[datetime]:
    start_dt = datetime.strptime(f"{date_str} {start_z.zfill(4)}", "%m/%d/%Y %H%M")
    end_dt = datetime.strptime(f"{date_str} {end_z.zfill(4)}", "%m/%d/%Y %H%M")

    if end_dt < start_dt:
        end_dt += timedelta(days=1)

    t = start_dt.replace(minute=0, second=0, microsecond=0)
    if t < start_dt:
        t += timedelta(hours=1)

    targets = []
    while t <= end_dt:
        targets.append(t)
        t += timedelta(hours=1)

    return targets


def target_folder_name(dt: datetime) -> str:
    return f"{dt.strftime('%Y')}{dt.strftime('%j')}_{dt.strftime('%H')}Z"


def build_token() -> eumdac.AccessToken:
    if not EUMETSAT_KEY or not EUMETSAT_SECRET:
        raise RuntimeError(
            "Missing EUMETSAT credentials. Set EUMETSAT_KEY and EUMETSAT_SECRET first."
        )
    return eumdac.AccessToken((EUMETSAT_KEY, EUMETSAT_SECRET))


def get_collection(token: eumdac.AccessToken):
    datastore = eumdac.DataStore(token)
    return datastore.get_collection(COLLECTION_ID)


def iter_product_entries(product) -> list:
    try:
        entries = list(product.entries)
    except Exception:
        entries = []
    return entries


def entry_name(entry) -> str:
    for attr in ("name", "filename", "id"):
        value = getattr(entry, attr, None)
        if value:
            return str(value)
    return str(entry)


def product_name(product) -> str:
    for attr in ("title", "id", "name"):
        value = getattr(product, attr, None)
        if value:
            return str(value)
    return str(product)


def product_time(product, fallback: datetime) -> datetime:
    for attr in (
        "sensing_start",
        "start_time",
        "start",
        "dtstart",
        "publication_date",
    ):
        value = getattr(product, attr, None)
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
    return fallback


def product_has_required_channels(product) -> bool:
    entries = iter_product_entries(product)
    if not entries:
        # Many MSG products are single native files; keep them.
        return True

    names = [entry_name(e).upper() for e in entries]
    return any(tok in n for tok in REQUIRED_CHANNEL_TOKENS for n in names)


def find_nearest_product(collection, target: datetime):
    start = target - timedelta(minutes=SEARCH_WINDOW_MINUTES)
    end = target + timedelta(minutes=SEARCH_WINDOW_MINUTES)

    try:
        products = list(collection.search(dtstart=start, dtend=end))
    except TypeError:
        # Some eumdac versions use start/end instead.
        products = list(collection.search(start=start, end=end))

    if not products:
        return None

    filtered = [p for p in products if product_has_required_channels(p)]
    if not filtered:
        filtered = products

    return min(filtered, key=lambda p: abs((product_time(p, target) - target).total_seconds()))


def download_product(product, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    entries = iter_product_entries(product)
    if entries:
        for entry in entries:
            name = entry_name(entry)
            outpath = outdir / name
            if outpath.exists():
                print(f"      already exists: {name}")
                continue

            print(f"      downloading entry: {name}")
            with product.open(entry=entry) as src, open(outpath, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return

    # Fallback: download the whole product as a single file.
    pname = product_name(product)
    outpath = outdir / f"{pname}.nat"
    if outpath.exists():
        print(f"      already exists: {outpath.name}")
        return

    print(f"      downloading full product: {outpath.name}")
    with product.open() as src, open(outpath, "wb") as dst:
        shutil.copyfileobj(src, dst)


def main() -> None:
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]

    print("CSV headers detected:", rows[0].keys() if rows else "No rows found")
    print(f"Using collection: {COLLECTION_ID}")

    token = build_token()
    collection = get_collection(token)

    for i, row in enumerate(rows, start=1):
        case_name = f"Case_{i}"

        if TARGET_CASES and case_name not in TARGET_CASES:
            continue

        date_str = row.get("Date", "").strip()
        start_z = row.get("Start (Zulu)", "").strip()
        end_z = row.get("End (Zulu)", "").strip()
        satellite = row.get("Satellite", "").strip()

        if satellite != TARGET_SATELLITE:
            continue

        if not date_str or not start_z or not end_z:
            print(f"Skipping incomplete Meteosat row: {case_name}")
            continue

        case_outdir = BASE_OUTDIR / case_name
        case_outdir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {case_name}: {date_str} {start_z}-{end_z}Z | {satellite}")
        print(f"Case folder: {case_outdir}")

        targets = hourly_targets(date_str, start_z, end_z)

        for target in targets:
            folder = case_outdir / target_folder_name(target)
            folder.mkdir(parents=True, exist_ok=True)

            print(f"\n  Target time: {target:%Y-%m-%d %H:%MZ}")
            print(f"  Folder: {folder}")

            product = find_nearest_product(collection, target)
            if product is None:
                print("    no Meteosat product found in search window")
                continue

            scan_time = product_time(product, target)
            print(f"    nearest product: target {target:%H:%MZ} -> scan {scan_time:%H:%MZ} :: {product_name(product)}")
            download_product(product, folder)

    print("DONE")


if __name__ == "__main__":
    main()