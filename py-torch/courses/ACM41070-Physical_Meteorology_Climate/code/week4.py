import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
output_dir = Path(__file__).parent / "output" / "week4"
output_dir.mkdir(parents=True, exist_ok=True)   
# ==========================================================
# LOAD DATA
# ==========================================================

file_path = (
    Path(__file__).parent
    / "data"
    / "week4"
    / "1e54f08eb13e7d9b9d174498f892d1be.nc"
)

ds = xr.open_dataset(file_path)

print(ds)

lat = ds.latitude.values
lon = ds.longitude.values

# ==========================================================
# GRID GEOMETRY (shared by both panels)
# ==========================================================

Omega = 7.292115e-5
rho = 1.225
R = 6371000

lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)

f = 2 * Omega * np.sin(lat_rad)

dlat = np.gradient(lat_rad)
dlon = np.gradient(lon_rad)

dy = dlat[:, None] * R
dx = dlon[None, :] * R * np.cos(lat_rad[:, None])

f2d = f[:, None].astype(float)
f2d[np.abs(f2d) < 1e-5] = np.nan

# ==========================================================
# SELECT TIME FOR MAP PANEL
# ==========================================================

t = 91

mslp = ds["msl"].isel(valid_time=t)
u10  = ds["u10"].isel(valid_time=t)
v10  = ds["v10"].isel(valid_time=t)

# ==========================================================
# GEOSTROPHIC WIND — SINGLE TIME STEP (for map)
# ==========================================================

p = mslp.values

dpdy, dpdx = np.gradient(p)
dpdy = dpdy / dy
dpdx = dpdx / dx

ug = -(1 / (rho * f2d)) * dpdy
vg =  (1 / (rho * f2d)) * dpdx

Vg = np.sqrt(ug**2 + vg**2)

print(f"Maximum geostrophic wind = {np.nanmax(Vg):.2f} m/s")
print(f"Mean geostrophic wind    = {np.nanmean(Vg):.2f} m/s")

# ==========================================================
# ACTUAL WINDS — SINGLE TIME STEP (for map)
# ==========================================================

wind_speed = np.sqrt(u10.values**2 + v10.values**2)

print(f"Maximum actual wind = {np.nanmax(wind_speed):.2f} m/s")
print(f"Mean actual wind    = {np.nanmean(wind_speed):.2f} m/s")

# ==========================================================
# STORM CENTRE
# ==========================================================

idx = np.unravel_index(np.argmin(mslp.values), mslp.shape)

storm_lat   = lat[idx[0]]
storm_lon   = lon[idx[1]]
storm_press = mslp.values[idx] / 100.0

print(
    f"Storm Centre: {storm_lat:.2f}N, "
    f"{storm_lon:.2f}E, {storm_press:.1f} hPa"
)

mslp_hpa = mslp / 100.0

# ==========================================================
# MAP PANEL (top-left)
# ==========================================================

fig1 = plt.figure(figsize=(12, 8))
ax1  = plt.axes(projection=ccrs.PlateCarree())

ax1.add_feature(cfeature.LAND,      facecolor="lightgrey")
ax1.add_feature(cfeature.COASTLINE, linewidth=1)
ax1.add_feature(cfeature.BORDERS,   linewidth=0.5)

wind_fill = ax1.contourf(
    lon, lat, wind_speed,
    levels=np.arange(0, 32, 2),
    cmap="YlOrRd",
    extend="max",
    alpha=0.8,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(wind_fill, ax=ax1, shrink=0.8, pad=0.02)
cbar.set_label("10 m Wind Speed (m/s)")

levels = np.arange(int(mslp_hpa.min()), int(mslp_hpa.max()) + 4, 4)

cs = ax1.contour(
    lon, lat, mslp_hpa,
    levels=levels,
    colors="black",
    linewidths=1.2,
    transform=ccrs.PlateCarree()
)

ax1.clabel(cs, inline=True, fontsize=8, fmt="%d")

skip = 8

ax1.quiver(
    lon[::skip], lat[::skip],
    u10.values[::skip, ::skip],
    v10.values[::skip, ::skip],
    color="navy",
    scale=400, width=0.002,
    transform=ccrs.PlateCarree()
)

ax1.quiver(
    lon[::skip], lat[::skip],
    ug[::skip, ::skip],
    vg[::skip, ::skip],
    color="cyan",
    scale=700, width=0.002, alpha=0.9,
    transform=ccrs.PlateCarree()
)

malin_lat = 55.37
malin_lon = -7.34

ax1.plot(
    malin_lon, malin_lat, "ro",
    markersize=7,
    transform=ccrs.PlateCarree()
)

ax1.text(
    malin_lon + 0.5, malin_lat,
    "Malin Head",
    fontsize=10, color="red", weight="bold",
    transform=ccrs.PlateCarree()
)

ax1.plot(
    storm_lon, storm_lat,
    marker="X", color="magenta", markersize=14,
    transform=ccrs.PlateCarree()
)

ax1.text(
    storm_lon, storm_lat,
    f"L {storm_press:.0f}",
    fontsize=12, color="magenta", weight="bold",
    transform=ccrs.PlateCarree()
)

gl = ax1.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
gl.top_labels   = False
gl.right_labels = False

ax1.set_extent(
    [lon.min(), lon.max(), lat.min(), lat.max()],
    crs=ccrs.PlateCarree()
)

ax1.set_title(
    f"Storm Bert ({str(ds.valid_time[t].values)[:16]})\n"
    "MSLP, Actual Wind Field and Geostrophic Wind",
    fontsize=15, weight="bold"
)

plt.tight_layout()
plt.savefig(output_dir / "storm_bert_map.png", dpi=150)
plt.close()

print("Map panel saved.")

# ==========================================================
# ERA5 TIME SERIES AT MALIN HEAD (bottom-right panel)
# ==========================================================

ds_malin = ds.sel(
    latitude=malin_lat,
    longitude=malin_lon,
    method="nearest"
)

times    = ds_malin["valid_time"].values
u10_ts   = ds_malin["u10"].values
v10_ts   = ds_malin["v10"].values

wind_speed_ts = np.sqrt(u10_ts**2 + v10_ts**2) * 1.944  # knots

# Malin Head grid indices
malin_lat_idx = np.argmin(np.abs(lat - malin_lat))
malin_lon_idx = np.argmin(np.abs(lon - malin_lon))

# Vectorised geostrophic wind over all time steps
p_all    = ds["msl"].values                       # (time, lat, lon)
dpdy_all = np.gradient(p_all, axis=1) / dy[None, :, :]
dpdx_all = np.gradient(p_all, axis=2) / dx[None, :, :]

f3d   = f2d[None, :, :]
ug_all = -(1 / (rho * f3d)) * dpdy_all
vg_all =  (1 / (rho * f3d)) * dpdx_all

vg_ts = (
    np.sqrt(ug_all**2 + vg_all**2)
    [:, malin_lat_idx, malin_lon_idx]
    * 1.944
)

# --- Plot ---
fig4, ax4 = plt.subplots(figsize=(10, 5))

ax4.plot(
    times, vg_ts,
    color="red", linewidth=1.5, linestyle="-",
    label="Geostrophic Wind ERA5 (kt)"
)

ax4.plot(
    times, wind_speed_ts,
    color="blue", linewidth=1.5, linestyle="--",
    label="ERA5 10m Wind at Malin Head (kt)"
)

ax4.set_xlabel("Date & Time (UTC)")
ax4.set_ylabel("Wind Speed (knots)")
ax4.set_title(
    "Storm Bert: ERA5 Wind at Malin Head\n"
    "(Model vs Geostrophic)",
    fontsize=13, weight="bold"
)

ax4.legend()
ax4.grid(True, alpha=0.4)
ax4.set_xlim(times[0], times[-1])

plt.tight_layout()
plt.savefig(output_dir / "era5_malin_head_timeseries.png", dpi=150)
plt.close()

print("Time series panel saved.")