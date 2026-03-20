import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import cartopy.mpl.ticker as cticker

ROOT = Path(r"D:\MLNVI Batch\AFIT_BROWN")
CSV_FILE = Path(r"D:\MLNVI Batch\Fog_Cases.csv")

METEOSAT_CASES = {
    "Case_18", "Case_19", "Case_20",
    "Case_31", "Case_32", "Case_33", "Case_34",
    "Case_35", "Case_36", "Case_37", "Case_38"
}


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


def extract_finite_core(lat, lon, ml):
    """
    Find a rectangular core where lat/lon are fully finite.
    ML can still contain NaNs; those will be masked later.
    """
    geo_finite = np.isfinite(lat) & np.isfinite(lon)

    if geo_finite.sum() == 0:
        return None, None, None

    good_rows = np.where(np.all(geo_finite, axis=1))[0]
    good_cols = np.where(np.all(geo_finite, axis=0))[0]

    if len(good_rows) == 0 or len(good_cols) == 0:
        return None, None, None

    lat2 = lat[np.ix_(good_rows, good_cols)]
    lon2 = lon[np.ix_(good_rows, good_cols)]
    ml2 = ml[np.ix_(good_rows, good_cols)]

    return lat2, lon2, ml2


def make_plot(pred_path: Path, pm_path: Path, center_lat: float, center_lon: float, case_label: str, date_str: str):
    pred = np.load(pred_path)
    base = np.load(pm_path)

    ml = np.squeeze(pred["MLtruth"]).astype(np.float32)
    lat = np.asarray(base["latitude"], dtype=np.float32)
    lon = np.asarray(base["longitude"], dtype=np.float32)

    lat, lon, ml = extract_finite_core(lat, lon, ml)

    if lat is None:
        print(f"Skipping {pred_path} - no fully finite lat/lon core")
        return

    # Mask only ML invalid values
    ml_masked = np.ma.masked_invalid(ml)

    if ml_masked.count() == 0:
        print(f"Skipping {pred_path} - no finite ML values in valid core")
        return

    west = np.nanmin(lon)
    east = np.nanmax(lon)
    south = np.nanmin(lat)
    north = np.nanmax(lat)

    if not np.isfinite([west, east, south, north]).all():
        print(f"Skipping {pred_path} - invalid extent")
        return

    if east <= west or north <= south:
        print(f"Skipping {pred_path} - bad extent")
        return

    folder_time = pred_path.parent.name.split("_")[-1]

    fig = plt.figure(figsize=(10, 8))
    try:
        ax = plt.axes(projection=ccrs.PlateCarree())

        mesh = ax.pcolormesh(
            lon,
            lat,
            ml_masked,
            transform=ccrs.PlateCarree(),
            cmap="gray",
            vmin=0,
            vmax=1,
            shading="auto",
            rasterized=True,
        )

        ax.coastlines(resolution="10m", linewidth=1.0)
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
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

        if np.isfinite(center_lat) and np.isfinite(center_lon):
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
        plt.savefig(out_png, dpi=500, bbox_inches="tight")
        print(f"Saved: {out_png}")

    finally:
        plt.close(fig)


def main():
    centers = load_case_centers(CSV_FILE)

    for pred_path in ROOT.rglob("*_PM_MLpred.npz"):
        case_name = pred_path.parent.parent.name

        if case_name not in METEOSAT_CASES:
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