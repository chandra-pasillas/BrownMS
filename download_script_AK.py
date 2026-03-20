from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

CSV_FILE = r"D:\MLNVI Batch\Fog_Cases.csv"
BASE_OUTDIR = Path("AFIT_BROWN")

# Only run these Anchorage cases
ANCHORAGE_CASES = {"14", "15", "16", "17"}

SATELLITE_CONFIG = {
    "GOES-18": {
        "bucket": "s3://noaa-goes18",
        "product": "ABI-L1b-RadF",   # Full Disk for Alaska/Anchorage
        "mode": "M6",
        "sat_id": "G18",
        "channels": ["C07", "C11", "C13", "C14", "C15"],
    },
}

NOT_IMPLEMENTED_SATELLITES = {"Himawari-9", "Meteosat-9"}

# Example:
# OR_ABI-L1b-RadF-M6C13_G18_s20233001600211_e20233001609519_c20233001609587.nc
FILENAME_RE = re.compile(
    r"OR_ABI-L1b-RadF-(M\d)(C\d{2})_(G18)_"
    r"s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_"
    r"e\d{4}\d{3}\d{2}\d{2}\d{2}\d_"
    r"c\d{4}\d{3}\d{2}\d{2}\d{2}\d\.nc$"
)

def run_cmd(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout

def list_s3(prefix: str) -> list[str]:
    out = run_cmd(["aws", "s3", "ls", "--no-sign-request", prefix])
    return [line for line in out.splitlines() if line.strip()]

def download_s3(s3_uri: str, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", "--no-sign-request", s3_uri, str(outpath)],
        check=True
    )

def parse_goes_filename(filename: str) -> dict | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None

    mode, channel, sat_id, year, jday, hh, mm, ss, tenth = m.groups()

    scan_start = datetime(int(year), 1, 1) + timedelta(
        days=int(jday) - 1,
        hours=int(hh),
        minutes=int(mm),
        seconds=int(ss),
        milliseconds=int(tenth) * 100
    )

    return {
        "mode": mode,
        "channel": channel,
        "sat_id": sat_id,
        "year": int(year),
        "jday": int(jday),
        "hour": int(hh),
        "minute": int(mm),
        "second": int(ss),
        "scan_start": scan_start,
    }

def hourly_targets(date_str: str, start_z: str, end_z: str) -> list[datetime]:
    start_dt = datetime.strptime(
        f"{date_str} {start_z.zfill(4)}",
        "%m/%d/%Y %H%M"
    )
    end_dt = datetime.strptime(
        f"{date_str} {end_z.zfill(4)}",
        "%m/%d/%Y %H%M"
    )

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

def build_hour_prefix(dt: datetime, sat_cfg: dict) -> str:
    year = dt.strftime("%Y")
    jday = dt.strftime("%j")
    hour = dt.strftime("%H")
    return f"{sat_cfg['bucket']}/{sat_cfg['product']}/{year}/{jday}/{hour}/"

def target_folder_name(dt: datetime) -> str:
    return f"{dt.strftime('%Y')}{dt.strftime('%j')}_{dt.strftime('%H')}Z"

def find_nearest_file_for_channel(
    target: datetime,
    channel: str,
    sat_cfg: dict
) -> tuple[datetime, str] | None:
    prefix = build_hour_prefix(target, sat_cfg)
    lines = list_s3(prefix)

    candidates: list[tuple[datetime, str]] = []

    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue

        fname = parts[-1]
        parsed = parse_goes_filename(fname)
        if parsed is None:
            continue

        if parsed["mode"] != sat_cfg["mode"]:
            continue
        if parsed["channel"] != channel:
            continue
        if parsed["sat_id"] != sat_cfg["sat_id"]:
            continue

        candidates.append((parsed["scan_start"], prefix + fname))

    if not candidates:
        return None

    return min(candidates, key=lambda x: abs((x[0] - target).total_seconds()))

def main() -> None:
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]

        rows = []
        for row in reader:
            clean_row = {k.strip(): v for k, v in row.items()}
            rows.append(clean_row)

    print("CSV headers detected:", rows[0].keys() if rows else "No rows found")

    for i, row in enumerate(rows, start=1):
        case_id = str(i)

        # Only Anchorage cases
        if case_id not in ANCHORAGE_CASES:
            continue

        date_str = row.get("Date", "").strip()
        start_z = row.get("Start (Zulu)", "").strip()
        end_z = row.get("End (Zulu)", "").strip()
        satellite = row.get("Satellite", "").strip()

        if not date_str or not start_z or not end_z or not satellite:
            print(f"Skipping incomplete row: {row}")
            continue
        if satellite in NOT_IMPLEMENTED_SATELLITES:
            print(f"Skipping Case {case_id}: {satellite} not implemented yet")
            continue
        if satellite not in SATELLITE_CONFIG:
            print(f"Skipping Case {case_id}: unsupported satellite '{satellite}'")
            continue

        sat_cfg = SATELLITE_CONFIG[satellite]

        case_outdir = BASE_OUTDIR / f"Case_{case_id}"
        case_outdir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing Case {case_id}: {date_str} {start_z}-{end_z}Z | {satellite} | {sat_cfg['product']}")
        print(f"Case folder: {case_outdir}")

        targets = hourly_targets(date_str, start_z, end_z)

        for target in targets:
            folder = case_outdir / target_folder_name(target)
            folder.mkdir(parents=True, exist_ok=True)

            print(f"\n  Target time: {target:%Y-%m-%d %H:%MZ}")
            print(f"  Folder: {folder}")

            for channel in sat_cfg["channels"]:
                result = find_nearest_file_for_channel(target, channel, sat_cfg)

                if result is None:
                    print(f"    {channel}: no file found")
                    continue

                scan_time, s3_uri = result
                fname = Path(s3_uri).name
                outpath = folder / fname

                print(
                    f"    {channel}: target {target:%H:%MZ} -> "
                    f"scan {scan_time:%H:%M:%SZ} :: {fname}"
                )

                if outpath.exists():
                    print("      already exists")
                    continue

                download_s3(s3_uri, outpath)

if __name__ == "__main__":
    main()
