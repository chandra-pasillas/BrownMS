import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import cartopy.mpl.ticker as cticker

ROOT = Path(r"D:\MLNVI Batch\AFIT_BROWN")
CSV_FILE = Path(r"D:\MLNVI Batch\Fog_Cases.csv")

HIMAWARI_CASES = {"Case_27", "Case_28", "Case_29", "Case_30"}


def load_case_centers(csv_file: Path):
    import csv

    centers = {}

    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]

        for i, row in enumerate(reader, start=1):
            row = {k.strip(): v for k, v in row.items()}
            case_name = f"Case_{i}"

            centers[case_name] = {
                "lat": float(row["Lat"]),
                "lon": float(row["Lon"]),
                "date": row["Date"].strip(),
                "location": row["Location"].strip(),
            }

    return centers


def make_plot(pred_path: Path, pm_path: Path, center_lat: float, center_lon: float, case_label: str, date_str: str):
    pred = np.load(pred_path)
    base = np.load(pm_path)

    ml = np.squeeze(pred["MLtruth"])
    lat = base["latitude"]
    lon = base["longitude"]

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(ml)

    if valid.sum() == 0:
        print(f"Skipping {pred_path} - no finite points at all")
        return

    west = np.nanmin(lon[valid])
    east = np.nanmax(lon[valid])
    south = np.nanmin(lat[valid])
    north = np.nanmax(lat[valid])

    folder_time = pred_path.parent.name.split("_")[-1]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    mesh = ax.pcolormesh(
        lon,
        lat,
        ml,
        transform=ccrs.PlateCarree(),
        cmap="gray",
        vmin=0,
        vmax=1,
        shading="auto",
    )

    ax.coastlines(resolution="10m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.6)
    ax.set_extent([west, east, south, north], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax.plot(
        center_lon,
        center_lat,
        marker="x",
        color="red",
        markersize=10,
        mew=2,
        transform=ccrs.PlateCarree(),
    )

    title = f"ML-NVI Prediction\n{case_label} | {date_str} {folder_time}"
    ax.set_title(title, fontsize=14)

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("ML-NVI")

    out_png = pred_path.with_name(pred_path.stem + "_plot.png")
    plt.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


def main():
    centers = load_case_centers(CSV_FILE)

    for pred_path in ROOT.rglob("*_PM_MLpred.npz"):
        case_name = pred_path.parent.parent.name

        if case_name not in HIMAWARI_CASES:
            continue

        if case_name not in centers:
            print(f"Skipping {pred_path} - case not found in CSV")
            continue

        pm_path = pred_path.with_name(pred_path.name.replace("_MLpred.npz", ".npz"))
        if not pm_path.exists():
            print(f"Skipping {pred_path} - matching PM file not found")
            continue

        center_lat = centers[case_name]["lat"]
        center_lon = centers[case_name]["lon"]
        date_str = centers[case_name]["date"]
        case_label = centers[case_name]["location"]

        try:
            make_plot(pred_path, pm_path, center_lat, center_lon, case_label, date_str)
        except Exception as e:
            print(f"Failed on {pred_path}: {e}")


if __name__ == "__main__":
    main()
