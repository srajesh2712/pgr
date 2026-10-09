"""
ERA5 Hourly Storm Analysis Script
=================================
This script performs multi-hourly meteorological analysis on ERA5 NetCDF dataset:
1. Storm Trajectory & Central Pressure Evolution (Map Plot)
2. Time-Series Meteogram at Malin Head (Pressure vs Wind)
3. Domain-Wide Peak Wind Speed & Central MSLP Trends
4. Hourly Animated Visualization (Saved as storm_evolution.gif)
"""

from interface_meta import skip
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from matplotlib.animation import FuncAnimation


def load_dataset():
    default_path = Path(__file__).parent / "data" / "week4" / "2f4b34b00ecc91be5a6954d53f379fdb.nc"
    if default_path.exists():
        file_path = default_path
    else:
        # Fallback search in current working directory
        nc_files = list(Path(".").glob("*.nc"))
        file_path = nc_files[0] if nc_files else "data.nc"
    
    print(f"Loading dataset from: {file_path}")
    return xr.open_dataset(file_path)


def plot_storm_trajectory(ds):
    print("Generating Storm Trajectory Plot...")
    storm_lats, storm_lons, min_press = [], [], []

    for t in ds.valid_time:
        mslp_t = ds["msl"].sel(valid_time=t) / 100.0  # hPa
        idx = np.unravel_index(np.argmin(mslp_t.values), mslp_t.shape)
        
        storm_lats.append(float(mslp_t.latitude[idx[0]]))
        storm_lons.append(float(mslp_t.longitude[idx[1]]))
        min_press.append(float(mslp_t.min()))

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor="#e6e6e6")

    sc = ax.scatter(
        storm_lons, storm_lats, c=min_press, cmap="Blues_r", 
        s=60, zorder=5, transform=ccrs.PlateCarree(), edgecolor="black", linewidth=0.5
    )
    ax.plot(storm_lons, storm_lats, color="crimson", linestyle="--", linewidth=1.8, transform=ccrs.PlateCarree())

    # Mark Start and End positions
    ax.plot(storm_lons[0], storm_lats[0], "go", markersize=10, label="Start", transform=ccrs.PlateCarree())
    ax.plot(storm_lons[-1], storm_lats[-1], "rx", markersize=10, markeredgewidth=2, label="End", transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label("Central MSLP (hPa)")

    plt.legend(loc="upper left")
    plt.title("Storm Center Trajectory & Pressure Evolution", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("storm_trajectory.png", dpi=300)
    plt.show()


def plot_meteogram_malin_head(ds, malin_lat=55.37, malin_lon=-7.34):
    print("Generating Meteogram at Malin Head...")
    malin = ds.sel(latitude=malin_lat, longitude=malin_lon, method="nearest")

    malin_mslp = malin["msl"] / 100.0
    malin_wind = np.sqrt(malin["u10"]**2 + malin["v10"]**2)
    times = ds.valid_time.values

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color = "#1f77b4"
    ax1.set_xlabel("Time (UTC)", fontweight="bold")
    ax1.set_ylabel("MSLP (hPa)", color=color, fontweight="bold")
    ax1.plot(times, malin_mslp, color=color, linewidth=2.5, label="MSLP")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()  
    color = "#d62728"
    ax2.set_ylabel("10m Wind Speed (m/s)", color=color, fontweight="bold")
    ax2.plot(times, malin_wind, color=color, linewidth=2.5, linestyle="-.", label="Wind Speed")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title(f"Meteogram at Malin Head ({malin_lat}°N, {malin_lon}°E)", fontsize=14, weight="bold")
    fig.tight_layout()
    plt.savefig("meteogram_malin_head.png", dpi=300)
    plt.show()


def plot_domain_summary(ds):
    print("Generating Domain Wind & Pressure Summary...")
    v_actual_all = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
    max_actual = v_actual_all.max(dim=["latitude", "longitude"]).values
    min_press = (ds["msl"] / 100.0).min(dim=["latitude", "longitude"]).values
    times = ds.valid_time.values

    fig, ax1 = plt.subplots(figsize=(11, 5))

    color = "darkorange"
    ax1.set_xlabel("Time (UTC)", fontweight="bold")
    ax1.set_ylabel("Max Domain Wind Speed (m/s)", color=color, fontweight="bold")
    ax1.plot(times, max_actual, color=color, linewidth=2.5, label="Max Wind Speed")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color = "navy"
    ax2.set_ylabel("Min Domain MSLP (hPa)", color=color, fontweight="bold")
    ax2.plot(times, min_press, color=color, linewidth=2.5, linestyle="--", label="Min MSLP")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Hourly Domain Intensity Parameters", fontsize=14, weight="bold")
    fig.tight_layout()
    plt.savefig("domain_summary.png", dpi=300)
    plt.show()


def create_storm_animation(ds, output_gif="storm_evolution.gif"):
    print(f"Generating Storm Evolution Animation: {output_gif}...")
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    def update(t_idx):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="#e6e6e6")
        
        mslp_t = ds["msl"].isel(valid_time=t_idx) / 100.0
        u_t = ds["u10"].isel(valid_time=t_idx)
        v_t = ds["v10"].isel(valid_time=t_idx)
        w_speed = np.sqrt(u_t**2 + v_t**2)
        
        # Wind speed shading
        ax.contourf(ds.longitude, ds.latitude, w_speed, levels=np.arange(0, 36, 2), cmap="YlOrRd", alpha=0.7, transform=ccrs.PlateCarree())
        
        # MSLP contours
        cs = ax.contour(ds.longitude, ds.latitude, mslp_t, levels=12, colors="black", linewidths=1.2, transform=ccrs.PlateCarree())
        ax.clabel(cs, inline=True, fontsize=8, fmt="%d")
        
        # Malin Head marker
        ax.plot(-7.34, 55.37, "ro", markersize=6, transform=ccrs.PlateCarree())
        
        time_str = str(ds.valid_time.values[t_idx])[:16]
        ax.set_title(f"Storm Dynamics - {time_str} UTC", fontsize=13, weight="bold")

        skip = 5

        ax.quiver(
            ds.longitude.values[::skip],
            ds.latitude.values[::skip],
            u_t.values[::skip, ::skip],
            v_t.values[::skip, ::skip],
            color="navy",
            scale=350,
            width=0.002,
            transform=ccrs.PlateCarree()
        )   
        skip = 5

        ax.quiver(
            ds.longitude.values[::skip],
            ds.latitude.values[::skip],
            u_t.values[::skip, ::skip],
            v_t.values[::skip, ::skip],
            color="navy",
            scale=350,
            width=0.002,
            transform=ccrs.PlateCarree()
        )

    anim = FuncAnimation(fig, update, frames=len(ds.valid_time), interval=300)
    anim.save(output_gif, writer="pillow")
    plt.close()
    print("Animation successfully saved!")


if __name__ == "__main__":
    dataset = load_dataset()
    print(dataset)
    
    # Run all analysis modules
    plot_storm_trajectory(dataset)
    plot_meteogram_malin_head(dataset)
    plot_domain_summary(dataset)
    create_storm_animation(dataset)