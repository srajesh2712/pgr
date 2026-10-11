import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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

# ==========================================================
# SELECT TIME
# ==========================================================

t = 91

mslp = ds["msl"].isel(valid_time=t)  # Pa
u10 = ds["u10"].isel(valid_time=t)
v10 = ds["v10"].isel(valid_time=t)

lat = ds.latitude.values
lon = ds.longitude.values

# ==========================================================
# GEOSTROPHIC WIND FROM PRESSURE GRADIENT
# ==========================================================

Omega = 7.292115e-5
rho = 1.225
R = 6371000

lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)

# Coriolis parameter

f = 2 * Omega * np.sin(lat_rad)

# Grid spacing (metres)

dlat = np.gradient(lat_rad)
dlon = np.gradient(lon_rad)

dy = dlat[:, None] * R
dx = dlon[None, :] * R * np.cos(lat_rad[:, None])

# Pressure field (Pa)

p = mslp.values

# Pressure gradients

dpdy, dpdx = np.gradient(p)

dpdy = dpdy / dy
dpdx = dpdx / dx

# Expand Coriolis parameter

f2d = f[:, None].astype(float)

# Avoid singularity near equator

f2d[np.abs(f2d) < 1e-5] = np.nan

# Geostrophic wind equations

ug = -(1 / (rho * f2d)) * dpdy
vg = (1 / (rho * f2d)) * dpdx

Vg = np.sqrt(ug**2 + vg**2)
print("ug range:", np.nanmin(ug), np.nanmax(ug))
print("vg range:", np.nanmin(vg), np.nanmax(vg))
print("f range :", np.nanmin(f2d), np.nanmax(f2d))
print("dpdx range:", np.nanmin(dpdx), np.nanmax(dpdx))
print("dpdy range:", np.nanmin(dpdy), np.nanmax(dpdy))
print(
    f"Maximum geostrophic wind = {np.nanmax(Vg):.2f} m/s"
)
print("95th percentile:", np.nanpercentile(Vg, 95))
print("99th percentile:", np.nanpercentile(Vg, 99))
print("99.5th percentile:", np.nanpercentile(Vg, 99.5))
print("99.9th percentile:", np.nanpercentile(Vg, 99.9))
print(
    f"Mean geostrophic wind = {np.nanmean(Vg):.2f} m/s"
)

# ==========================================================
# ACTUAL WINDS
# ==========================================================

wind_speed = np.sqrt(
    u10.values**2 +
    v10.values**2
)

print(
    f"Maximum actual wind = {np.nanmax(wind_speed):.2f} m/s"
)

print(
    f"Mean actual wind = {np.nanmean(wind_speed):.2f} m/s"
)

# ==========================================================
# STORM CENTRE
# ==========================================================

idx = np.unravel_index(
    np.argmin(mslp.values),
    mslp.shape
)

storm_lat = lat[idx[0]]
storm_lon = lon[idx[1]]
storm_press = mslp.values[idx] / 100.0

print(
    f"Storm Centre: "
    f"{storm_lat:.2f}N, "
    f"{storm_lon:.2f}E, "
    f"{storm_press:.1f} hPa"
)

# ==========================================================
# CONVERT MSLP TO hPa FOR PLOTTING
# ==========================================================

mslp_hpa = mslp / 100.0

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(15, 10))

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

# ==========================================================
# MAP FEATURES
# ==========================================================

ax.add_feature(cfeature.LAND, facecolor="lightgrey")
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)

# ==========================================================
# WIND SPEED SHADING
# ==========================================================

wind_fill = ax.contourf(
    lon,
    lat,
    wind_speed,
    levels=np.arange(0, 32, 2),
    cmap="YlOrRd",
    extend="max",
    alpha=0.8,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(
    wind_fill,
    ax=ax,
    shrink=0.8,
    pad=0.02
)

cbar.set_label("10 m Wind Speed (m/s)")

# ==========================================================
# PRESSURE CONTOURS
# ==========================================================

levels = np.arange(
    int(mslp_hpa.min()),
    int(mslp_hpa.max()) + 4,
    4
)

cs = ax.contour(
    lon,
    lat,
    mslp_hpa,
    levels=levels,
    colors="black",
    linewidths=1.2,
    transform=ccrs.PlateCarree()
)

ax.clabel(
    cs,
    inline=True,
    fontsize=8,
    fmt="%d"
)

# ==========================================================
# ACTUAL WIND VECTORS
# ==========================================================

skip = 8

ax.quiver(
    lon[::skip],
    lat[::skip],
    u10.values[::skip, ::skip],
    v10.values[::skip, ::skip],
    color="navy",
    scale=400,
    width=0.002,
    transform=ccrs.PlateCarree()
)

# ==========================================================
# GEOSTROPHIC WIND VECTORS
# ==========================================================

ax.quiver(
    lon[::skip],
    lat[::skip],
    ug[::skip, ::skip],
    vg[::skip, ::skip],
    color="cyan",
    scale=700,
    width=0.002,
    alpha=0.9,
    transform=ccrs.PlateCarree()
)

# ==========================================================
# MALIN HEAD
# ==========================================================

malin_lat = 55.37
malin_lon = -7.34

ax.plot(
    malin_lon,
    malin_lat,
    "ro",
    markersize=7,
    transform=ccrs.PlateCarree()
)

ax.text(
    malin_lon + 0.5,
    malin_lat,
    "Malin Head",
    fontsize=10,
    color="red",
    weight="bold",
    transform=ccrs.PlateCarree()
)

# ==========================================================
# STORM CENTRE MARKER
# ==========================================================

ax.plot(
    storm_lon,
    storm_lat,
    marker="X",
    color="magenta",
    markersize=14,
    transform=ccrs.PlateCarree()
)

ax.text(
    storm_lon,
    storm_lat,
    f"L {storm_press:.0f}",
    fontsize=12,
    color="magenta",
    weight="bold",
    transform=ccrs.PlateCarree()
)

# ==========================================================
# GRIDLINES
# ==========================================================

gl = ax.gridlines(
    draw_labels=True,
    linestyle="--",
    alpha=0.5
)

gl.top_labels = False
gl.right_labels = False

# ==========================================================
# DOMAIN LIMITS
# ==========================================================

ax.set_extent(
    [
        lon.min(),
        lon.max(),
        lat.min(),
        lat.max()
    ],
    crs=ccrs.PlateCarree()
)

# ==========================================================
# TITLE
# ==========================================================

plt.title(
    f"Storm Bert ({str(ds.valid_time[t].values)[:16]})\n"
    "MSLP, Actual Wind Field and Geostrophic Wind",
    fontsize=15,
    weight="bold"
)

plt.tight_layout()
plt.show()