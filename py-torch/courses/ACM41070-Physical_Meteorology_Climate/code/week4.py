import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file_path = Path(__file__).parent / "data" /"week4" / "2d6b96deddd5146bdc8783ba042ed9b9.nc"
ds = xr.open_dataset(file_path)
print(ds)

# Select one hour
t = 15 # e.g. 12 UTC

msl = ds["msl"].isel(valid_time=t)

lat = ds["latitude"].values
lon = ds["longitude"].values

# Earth constants
Omega = 7.292115e-5 # rad/s
g = 9.81 # m/s²

# Convert pressure to geopotential height approximation
# Z = p/(rho*g)
# Using rho ≈ 1.225 kg/m³
rho = 1.225

Z = msl.values / (rho * g)

# Convert coordinates to radians
lat_rad = np.deg2rad(lat)
lon_rad = np.deg2rad(lon)

# Coriolis parameter
f = 2 * Omega * np.sin(lat_rad)

# Grid spacing in metres
R = 6371000

dlat = np.gradient(lat_rad)
dlon = np.gradient(lon_rad)

dy = dlat[:, None] * R
dx = dlon[None, :] * R * np.cos(lat_rad[:, None])

# Height gradients
dZdy, dZdx = np.gradient(Z)

dZdy = dZdy / dy
dZdx = dZdx / dx

# Expand f to 2D
f2d = f[:, None]

# Avoid division by zero
f2d[np.abs(f2d) < 1e-5] = np.nan

# Geostrophic wind equations
ug = -(g / f2d) * dZdy
vg = (g / f2d) * dZdx

# Geostrophic wind speed
Vg = np.sqrt(ug**2 + vg**2)

print("Maximum geostrophic wind:", np.nanmax(Vg), "m/s")
print("Mean geostrophic wind:", np.nanmean(Vg), "m/s")

u10 = ds["u10"].isel(valid_time=12)
v10 = ds["v10"].isel(valid_time=12)

V_actual = np.sqrt(u10**2 + v10**2)

print("Max actual wind:", float(V_actual.max()))
print("Mean actual wind:", float(V_actual.mean()))
t = 13

# Data
mslp = ds["msl"].isel(valid_time=t) / 100.0  # hPa
u10 = ds["u10"].isel(valid_time=t)
v10 = ds["v10"].isel(valid_time=t)
tcc = ds["tcc"].isel(valid_time=t)           # 0-1

wind_speed = np.sqrt(u10**2 + v10**2)

# -----------------------------
# Create Figure
# -----------------------------

fig = plt.figure(figsize=(14,10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAND, facecolor="lightgray")

# -----------------------------
# Cloud Cover Shading
# -----------------------------

cloud = ax.contourf(
    ds.longitude,
    ds.latitude,
    tcc,
    levels=np.arange(0, 1.1, 0.1),
    cmap="Greys",
    alpha=0.5,
    transform=ccrs.PlateCarree()
)

cbar1 = plt.colorbar(
    cloud,
    ax=ax,
    shrink=0.75,
    pad=0.02
)

cbar1.set_label("Total Cloud Cover")

# -----------------------------
# Wind Speed Shading
# -----------------------------

wind = ax.contourf(
    ds.longitude,
    ds.latitude,
    wind_speed,
    levels=np.arange(0, 32, 2),
    cmap="YlOrRd",
    alpha=0.7,
    transform=ccrs.PlateCarree()
)

cbar2 = plt.colorbar(
    wind,
    ax=ax,
    shrink=0.75,
    pad=0.08
)

cbar2.set_label("Wind Speed (m/s)")

# -----------------------------
# Pressure Contours
# -----------------------------

levels = np.arange(
    int(mslp.min()) - 4,
    int(mslp.max()) + 4,
    4
)

cs = ax.contour(
    ds.longitude,
    ds.latitude,
    mslp,
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

# -----------------------------
# Wind Vectors
# -----------------------------

skip = 5

ax.quiver(
    ds.longitude.values[::skip],
    ds.latitude.values[::skip],
    u10.values[::skip, ::skip],
    v10.values[::skip, ::skip],
    color="navy",
    scale=350,
    width=0.002,
    transform=ccrs.PlateCarree()
)

# -----------------------------
# Malin Head
# -----------------------------

malin_lat = 55.37
malin_lon = -7.34

ax.plot(
    malin_lon,
    malin_lat,
    "ro",
    markersize=8,
    transform=ccrs.PlateCarree()
)

ax.text(
    malin_lon + 0.5,
    malin_lat,
    "Malin Head",
    color="red",
    fontsize=10,
    weight="bold",
    transform=ccrs.PlateCarree()
)

# -----------------------------
# Storm Centre
# -----------------------------

idx = np.unravel_index(
    np.argmin(mslp.values),
    mslp.shape
)

storm_lat = float(mslp.latitude[idx[0]])
storm_lon = float(mslp.longitude[idx[1]])
storm_press = float(mslp.min())

ax.plot(
    storm_lon,
    storm_lat,
    marker='X',
    color='magenta',
    markersize=15,
    transform=ccrs.PlateCarree()
)

ax.text(
    storm_lon,
    storm_lat,
    f"L {storm_press:.0f} hPa",
    color="magenta",
    fontsize=11,
    weight="bold",
    transform=ccrs.PlateCarree()
)

# -----------------------------
# Gridlines
# -----------------------------

gl = ax.gridlines(
    draw_labels=True,
    linestyle="--",
    alpha=0.5
)

gl.top_labels = False
gl.right_labels = False

# -----------------------------
# Title
# -----------------------------

plt.title(
    f"Storm Bert (23 Nov 2014 {t:02d}:00 UTC)\n"
    "MSLP, Cloud Cover and 10m Wind Field",
    fontsize=15,
    weight="bold"
)

plt.show()