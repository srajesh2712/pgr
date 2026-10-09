"""
Storm Bert - presentation graphics from hourly ERA5 data.

Outputs (written to ./output next to this script):
  1. storm_bert_animation.gif / .mp4   hourly map + live time-series panels
  2. storm_bert_snapshots.png          4 key hours side by side (for a slide)
  3. storm_bert_timeseries.png         pressure + wind through the storm
  4. storm_bert_track.png              storm-centre track coloured by pressure

Usage:  python storm_bert_presentation.py [path/to/file.nc]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib import animation, cm
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "data" / "week4" / "4bc0b7d8457a7070d63b5b44c837ec34.nc"
OUT_DIR = Path(__file__).parent / "output"

DATE = None            # e.g. "2024-11-23" to keep a single day if the file holds several
MALIN = (55.37, -7.34)  # lat, lon
CENTRE_GUESS = None    # (lat, lon, time_index) of the storm centre if auto-detect picks the wrong low
SEARCH_BOX = 4.0       # deg: how far the centre may move between hours when tracking
FPS = 4                # animation speed (frames per second); 24 hourly frames -> 6 s
MAKE_GIF = True        # GIF pastes into PowerPoint/Keynote/Slides everywhere
MAKE_MP4 = True        # needs ffmpeg on PATH; skipped automatically if missing
KEY_HOURS = None       # e.g. ["2024-11-23 06:00", "2024-11-23 12:00", ...] (4 entries) for snapshots

# Palette: one hue (orange) for wind magnitude, two fixed identity colours
# that are reused on the map AND in the time-series so they read as one system.
WIND_CMAP = matplotlib.colormaps["Oranges"]
NAVY = "#1d3557"    # storm centre
TEAL = "#0b7285"    # Malin Head
ORANGE = "#c2410c"  # domain-peak wind (matches the wind shading)
INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "figure.facecolor": "white",
})

PC = ccrs.PlateCarree()

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------
if len(sys.argv) > 1:
    DATA_FILE = Path(sys.argv[1])
OUT_DIR.mkdir(exist_ok=True)

ds = xr.open_dataset(DATA_FILE)
if DATE:
    ds = ds.sel(valid_time=DATE)


def get(name):
    da = ds[name]
    for extra in ("number", "expver"):
        if extra in da.dims:
            da = da.isel({extra: 0})
    return da


times = pd.to_datetime(ds["valid_time"].values)
T = len(times)
lat = ds["latitude"].values
lon = ds["longitude"].values
print(f"{T} time steps: {times[0]} -> {times[-1]}")

mslp = get("msl").values / 100.0                      # hPa, (T, Y, X)
u10, v10 = get("u10").values, get("v10").values
ws = np.hypot(u10, v10)                                # m/s
gust_name = "i10fg" if "i10fg" in ds else None
gust = get(gust_name).values if gust_name else None

# --------------------------------------------------------------------------
# Derived hourly series
# --------------------------------------------------------------------------
def track_centre(mslp, lat, lon, guess=None, box=SEARCH_BOX):
    """Follow the pressure minimum hour by hour, starting from the deepest hour."""
    if guess:
        t0 = guess[2]
        i0 = (np.abs(lat - guess[0]).argmin(), np.abs(lon - guess[1]).argmin())
    else:
        t0 = int(np.nanargmin(mslp.reshape(len(mslp), -1).min(axis=1)))
        i0 = np.unravel_index(np.nanargmin(mslp[t0]), mslp[t0].shape)

    def nearest_min(frame, la, lo):
        mask = (np.abs(lat[:, None] - la) <= box) & (np.abs(lon[None, :] - lo) <= box)
        return np.unravel_index(np.argmin(np.where(mask, frame, np.inf)), frame.shape)

    idx = [None] * len(mslp)
    idx[t0] = i0
    for t in range(t0 + 1, len(mslp)):
        idx[t] = nearest_min(mslp[t], lat[idx[t - 1][0]], lon[idx[t - 1][1]])
    for t in range(t0 - 1, -1, -1):
        idx[t] = nearest_min(mslp[t], lat[idx[t + 1][0]], lon[idx[t + 1][1]])
    return np.array(idx)


idx = track_centre(mslp, lat, lon, CENTRE_GUESS)
c_lat, c_lon = lat[idx[:, 0]], lon[idx[:, 1]]
c_p = mslp[np.arange(T), idx[:, 0], idx[:, 1]]

ml = (np.abs(lat - MALIN[0]).argmin(), np.abs(lon - MALIN[1]).argmin())
malin_p = mslp[:, ml[0], ml[1]]
malin_ws = ws[:, ml[0], ml[1]]
malin_gust = gust[:, ml[0], ml[1]] if gust is not None else None
peak_ws = ws.reshape(T, -1).max(axis=1)

deepest = int(c_p.argmin())
print(f"Deepest: {c_p.min():.0f} hPa at {times[deepest]:%d %b %H:%M} UTC")
print(f"Malin Head peak wind: {malin_ws.max():.1f} m/s"
      + (f", peak gust {malin_gust.max():.1f} m/s" if gust is not None else ""))

# Fixed scales across ALL hours, so frames are comparable
W_MAX = float(np.ceil(peak_ws.max() / 4) * 4)
W_LEVELS = np.arange(0, W_MAX + 2, 2)
W_NORM = Normalize(0, W_MAX)
P_LEVELS = np.arange(np.floor(mslp.min() / 4) * 4, np.ceil(mslp.max() / 4) * 4 + 4, 4)
EXTENT = [lon.min(), lon.max(), lat.min(), lat.max()]
STEP = max(1, int(round(1.0 / abs(np.diff(lon).mean()))))   # ~1 degree between arrows


# --------------------------------------------------------------------------
# Map helpers
# --------------------------------------------------------------------------
def setup_map(ax, labels=True):
    ax.set_extent(EXTENT, crs=PC)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor=INK, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor=MUTED, zorder=4)
    gl = ax.gridlines(draw_labels=labels, linewidth=0.4, color="white", alpha=0.6, linestyle="--")
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 9, "color": MUTED}
    ax.plot(MALIN[1], MALIN[0], "o", ms=8, mfc=TEAL, mec="white", mew=1.5, transform=PC, zorder=6)
    ax.text(MALIN[1] + 0.35, MALIN[0] + 0.25, "Malin Head", color=TEAL, fontsize=10, weight="bold",
            transform=PC, zorder=6, path_effects=[pe.withStroke(linewidth=3, foreground="white")])


def _remove(artist):
    try:
        artist.remove()
    except Exception:
        for c in getattr(artist, "collections", []):
            c.remove()


def draw_frame(ax, i, trail=True):
    """Draw hour i. Returns the list of artists so an animation can remove them."""
    art = []
    cf = ax.contourf(lon, lat, ws[i], levels=W_LEVELS, cmap=WIND_CMAP, norm=W_NORM,
                     extend="max", alpha=0.9, transform=PC, zorder=1)
    art.append(cf)

    cs = ax.contour(lon, lat, mslp[i], levels=P_LEVELS, colors=INK, linewidths=0.9,
                    transform=PC, zorder=3)
    art.append(cs)
    try:
        art.extend(ax.clabel(cs, fmt="%d", fontsize=8, inline=True))
    except Exception:
        pass

    s = slice(None, None, STEP)
    art.append(ax.quiver(lon[s], lat[s], u10[i][s, s], v10[i][s, s], color=INK, alpha=0.65,
                         scale=W_MAX * 14, width=0.0022, transform=PC, zorder=5))

    if trail and i > 0:
        art.extend(ax.plot(c_lon[: i + 1], c_lat[: i + 1], "-", color=NAVY, lw=2,
                           transform=PC, zorder=6))
    art.extend(ax.plot(c_lon[i], c_lat[i], "X", ms=15, mfc=NAVY, mec="white", mew=1.5,
                       transform=PC, zorder=7))
    art.append(ax.text(c_lon[i] + 0.4, c_lat[i] - 0.6, f"L {c_p[i]:.0f} hPa", color=NAVY,
                       fontsize=11, weight="bold", transform=PC, zorder=7,
                       path_effects=[pe.withStroke(linewidth=3, foreground="white")]))
    return art


def wind_colorbar(fig, ax, **kw):
    sm = cm.ScalarMappable(norm=W_NORM, cmap=WIND_CMAP)
    cb = fig.colorbar(sm, ax=ax, extend="max", **kw)
    cb.set_label("10 m wind speed (m/s)")
    cb.outline.set_visible(False)
    return cb


# --------------------------------------------------------------------------
# Time-series panels (shared by the animation and the static figure)
# --------------------------------------------------------------------------
PANELS = [
    ("Pressure (hPa)", [("Storm centre", c_p, NAVY, "-"),
                        ("Malin Head", malin_p, TEAL, "-")]),
    ("10 m wind (m/s)", [("Peak anywhere in domain", peak_ws, ORANGE, "-"),
                         ("Malin Head", malin_ws, TEAL, "-")]
     + ([("Malin Head gust", malin_gust, TEAL, ":")] if gust is not None else [])),
]


def draw_timeseries(axes, cursor=True):
    """Draw static lines; return per-axis (vline, [(dot, y_array), ...]) cursors."""
    cursors = []
    for ax, (ylabel, series) in zip(axes, PANELS):
        for name, y, col, ls in series:
            ax.plot(times, y, ls, color=col, lw=2, label=name)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6 if T > 12 else 2)))
        ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0),
                  ncol=len(series), borderaxespad=0.2, handlelength=1.6, columnspacing=1.2)
        ax.margins(y=0.25)
        if not cursor:
            continue
        vl = ax.axvline(times[0], color=MUTED, lw=1, zorder=0)
        dots = [(ax.plot([], [], "o", ms=7, mfc=col, mec="white", mew=1.2, zorder=5)[0], y)
                for _, y, col, _ in series]
        cursors.append((vl, dots))
    axes[-1].set_xlabel("Time (UTC)")
    return cursors


def move_cursors(cursors, i):
    for vl, dots in cursors:
        vl.set_xdata([times[i], times[i]])
        for dot, y in dots:
            dot.set_data([times[i]], [y[i]])


def stamp(i):
    return f"{times[i]:%a %d %b %Y, %H:%M} UTC"


# --------------------------------------------------------------------------
# 1. Animation
# --------------------------------------------------------------------------
def make_animation():
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.9, 1], left=0.05, right=0.97, top=0.88,
                          bottom=0.09, hspace=0.5, wspace=0.14)
    ax_map = fig.add_subplot(gs[:, 0], projection=PC)
    ax_p, ax_w = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1], sharex=None)

    setup_map(ax_map)
    wind_colorbar(fig, ax_map, orientation="horizontal", pad=0.07, shrink=0.8, aspect=40)
    ax_map.set_title("Mean sea-level pressure (contours, hPa)  ·  10 m wind (colour + arrows)",
                     fontsize=11, color=MUTED)
    title = fig.suptitle("", fontsize=20, weight="bold", y=0.96)
    cursors = draw_timeseries([ax_p, ax_w])

    state = {"art": []}

    def update(i):
        for a in state["art"]:
            _remove(a)
        state["art"] = draw_frame(ax_map, i)
        title.set_text(f"Storm Bert  ·  {stamp(i)}")
        move_cursors(cursors, i)

    anim = animation.FuncAnimation(fig, update, frames=T, interval=1000 / FPS)

    if MAKE_GIF:
        out = OUT_DIR / "storm_bert_animation.gif"
        anim.save(out, writer=animation.PillowWriter(fps=FPS), dpi=100)
        print("saved", out)
    if MAKE_MP4 and animation.writers.is_available("ffmpeg"):
        out = OUT_DIR / "storm_bert_animation.mp4"
        anim.save(out, writer=animation.FFMpegWriter(fps=FPS, bitrate=6000), dpi=150)
        print("saved", out)
    elif MAKE_MP4:
        print("ffmpeg not found - skipped MP4 (the GIF was still made)")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Snapshots
# --------------------------------------------------------------------------
def make_snapshots():
    if KEY_HOURS:
        picks = [int(np.abs(times - pd.Timestamp(h)).argmin()) for h in KEY_HOURS]
    else:
        picks = sorted({int(round(p)) for p in np.linspace(0, T - 1, 4)})
    n = len(picks)
    cols = 2 if n > 2 else n
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7.5 * cols, 6.2 * rows),
                            subplot_kw={"projection": PC}, squeeze=False)
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    for ax, i in zip(axs.flat, picks):
        setup_map(ax)
        draw_frame(ax, i, trail=False)
        ax.set_title(f"{times[i]:%H:%M} UTC  ·  centre {c_p[i]:.0f} hPa", fontsize=14, weight="bold")
    fig.suptitle(f"Storm Bert  ·  {times[0]:%d %b %Y}", fontsize=20, weight="bold")
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.97, hspace=0.18, wspace=0.12)
    wind_colorbar(fig, axs.ravel().tolist(), orientation="horizontal", pad=0.04,
                  shrink=0.5, aspect=40)
    out = OUT_DIR / "storm_bert_snapshots.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Time-series figure
# --------------------------------------------------------------------------
def make_timeseries():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    draw_timeseries(list(axes), cursor=False)
    ax_p, ax_w = axes
    ax_p.annotate(f"Deepest: {c_p[deepest]:.0f} hPa\n{times[deepest]:%H:%M} UTC",
                  (times[deepest], c_p[deepest]), xytext=(-110, 40), textcoords="offset points",
                  fontsize=10, color=NAVY, arrowprops=dict(arrowstyle="-", color=NAVY))
    y, label = (malin_gust, "gust") if gust is not None else (malin_ws, "wind")
    k = int(y.argmax())
    ax_w.annotate(f"Malin Head peak {label}: {y[k]:.0f} m/s\n{times[k]:%H:%M} UTC",
                  (times[k], y[k]), xytext=(40, 22), textcoords="offset points",
                  fontsize=10, color=TEAL, arrowprops=dict(arrowstyle="-", color=TEAL))
    fig.suptitle(f"Storm Bert  ·  {times[0]:%d %b %Y}", x=0.01, ha="left", fontsize=18, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "storm_bert_timeseries.png"
    fig.savefig(out, dpi=200)
    print("saved", out)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Track
# --------------------------------------------------------------------------
def make_track():
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=PC)
    setup_map(ax)
    ax.plot(c_lon, c_lat, "-", color=NAVY, lw=1.5, transform=PC, zorder=5)
    sc = ax.scatter(c_lon, c_lat, c=c_p, cmap="Purples_r", vmin=c_p.min() - 4, vmax=c_p.max() + 4,
                    s=70, edgecolors=NAVY, linewidths=1, transform=PC, zorder=6)
    for i in range(0, T, 3):
        ax.text(c_lon[i], c_lat[i] + 0.35, f"{times[i]:%H}", fontsize=9, ha="center",
                transform=PC, zorder=7, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
    cb.set_label("Central pressure (hPa)")
    cb.outline.set_visible(False)
    ax.set_title(f"Storm Bert centre track  ·  {times[0]:%d %b %Y}\n(labels = hour UTC)",
                 fontsize=15, weight="bold")
    out = OUT_DIR / "storm_bert_track.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)
    plt.close(fig)


if __name__ == "__main__":
    make_timeseries()
    make_track()
    make_snapshots()
    make_animation()
