from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

CSV_FILE = r"D:\MLNVI Batch\Fog_Cases.csv"
BASE_OUTDIR = Path(r"D:\MLNVI Batch\AFIT_BROWN")

TARGET_SATELLITE = "Himawari-9"
TARGET_CASES = {"Case_27", "Case_28", "Case_29", "Case_30"}

HIMAWARI_CFG = {
    "bucket": "s3://noaa-himawari9",
    "product": "AHI-L1b-FLDK",
    "bands": ["B07", "B11", "B13", "B14", "B15"],
}


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


def minute_candidates(target: datetime) -> list[datetime]:
    mins = [0, 10, 20, 30, 40, 50]
    base = target.replace(second=0, microsecond=0)
    out = []

    for hour_offset in (-1, 0, 1):
        h = base + timedelta(hours=hour_offset)
        for m in mins:
            out.append(h.replace(minute=m))

    return sorted(out)


def build_minute_prefix(dt: datetime) -> str:
    return dt.strftime(
        f"{HIMAWARI_CFG['bucket']}/{HIMAWARI_CFG['product']}/%Y/%m/%d/%H%M/"
    )


def parse_himawari_filename(fname: str) -> dict | None:
    # Example:
    # HS_H09_20231101_2200_B07_FLDK_R20_S0110.DAT.bz2
    parts = fname.split("_")
    if len(parts) < 7:
        return None
    if not fname.endswith(".DAT.bz2"):
        return None
    if parts[0] != "HS" or parts[1] != "H09":
        return None

    try:
        dt = datetime.strptime(parts[2] + parts[3], "%Y%m%d%H%M")
    except Exception:
        return None

    band = parts[4]
    area = parts[5]
    segment = parts[7].replace(".DAT.bz2", "") if len(parts) > 7 else ""

    return {
        "scan_start": dt,
        "band": band,
        "area": area,
        "segment": segment,
    }


def find_nearest_band_segments(target: datetime, band: str) -> tuple[datetime, list[str]] | None:
    grouped: dict[datetime, list[str]] = {}

    for dt in minute_candidates(target):
        prefix = build_minute_prefix(dt)

        try:
            lines = list_s3(prefix)
        except subprocess.CalledProcessError:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue

            fname = parts[-1]
            parsed = parse_himawari_filename(fname)
            if parsed is None:
                continue
            if parsed["band"] != band:
                continue
            if parsed["area"] != "FLDK":
                continue

            grouped.setdefault(parsed["scan_start"], []).append(prefix + fname)

    if not grouped:
        return None

    nearest_dt = min(grouped.keys(), key=lambda x: abs((x - target).total_seconds()))
    files = sorted(grouped[nearest_dt])

    return nearest_dt, files


def main() -> None:
    BASE_OUTDIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = [{k.strip(): v for k, v in row.items()} for row in reader]

    print("CSV headers detected:", rows[0].keys() if rows else "No rows found")

    for i, row in enumerate(rows, start=1):
        case_name = f"Case_{i}"

        if case_name not in TARGET_CASES:
            continue

        date_str = row.get("Date", "").strip()
        start_z = row.get("Start (Zulu)", "").strip()
        end_z = row.get("End (Zulu)", "").strip()
        satellite = row.get("Satellite", "").strip()

        if satellite != TARGET_SATELLITE:
            continue

        if not date_str or not start_z or not end_z:
            print(f"Skipping incomplete Himawari row: {case_name}")
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

            for band in HIMAWARI_CFG["bands"]:
                result = find_nearest_band_segments(target, band)

                if result is None:
                    print(f"    {band}: no files found")
                    continue

                scan_time, s3_uris = result
                print(f"    {band}: target {target:%H:%MZ} -> scan {scan_time:%H:%MZ} :: {len(s3_uris)} segments")

                for s3_uri in s3_uris:
                    fname = Path(s3_uri).name
                    outpath = folder / fname

                    if outpath.exists():
                        continue

                    download_s3(s3_uri, outpath)

    print("DONE")


if __name__ == "__main__":
    main()