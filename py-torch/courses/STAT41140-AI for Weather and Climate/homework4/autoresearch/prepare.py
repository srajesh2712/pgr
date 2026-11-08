"""
prepare.py — FIXED data + evaluation for the autoresearch loop.

!!! The agent must NOT edit this file. It defines the benchmark. !!!

Task   : given station observations up to hour t, predict temp/rhum/msl/wdsp
         at hour t + HORIZON at all 25 Met Éireann stations.
Metric : val_score = mean over targets of  RMSE_model / RMSE_persistence
         (persistence = "value at t+H equals value at t"). < 1.0 beats persistence.

Run once:  python prepare.py        (builds cache/data.npz, prints baselines)
Import  :  from prepare import load_data, evaluate, HORIZON, ...
"""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE       = Path(__file__).resolve().parent
DATA_DIR   = HERE.parent / "data"
DIST_CSV   = DATA_DIR / "edge_matrices" / "station_distances.csv"
CACHE      = HERE / "cache" / "data.npz"

HORIZON    = int(os.environ.get("HORIZON", 6))   # hours ahead (benchmark default 6; run.sh --horizon)
MAX_LOOKBACK = 168                                # eval times start >= 1 week into split
TARGETS    = ["temp", "rhum", "msl", "wdsp"]
# Input channels in X (normalised). First 8 = teammate's NODE_FEATURES, then wind vector.
FEATURES   = ["temp", "rhum", "msl", "wdsp", "rain", "dewpt", "vappr", "wetb", "wind_u", "wind_v"]
TARGET_IDX = [FEATURES.index(t) for t in TARGETS]

START, END = "2011-01-01 00:00", "2026-09-30 23:00"
SPLITS = {                                         # inclusive time ranges (UTC)
    "train": ("2011-01-01", "2022-12-31 23:00"),
    "val":   ("2023-01-01", "2024-12-31 23:00"),
    "test":  ("2025-01-01", "2026-09-30 23:00"),
}
ADJ_SIGMA_KM = 100.0                               # Gaussian kernel width for default adjacency


# ── Loading ─────────────────────────────────────────────────────────────────────

def _read_station(fp):
    with open(fp, encoding="utf-8", errors="replace") as fh:
        head = [next(fh) for _ in range(80)]
    hdr = next(i for i, l in enumerate(head) if l.lower().startswith("date,"))
    name = next((re.match(r"Station Name:\s*(.+)", l).group(1).strip()
                 for l in head if l.startswith("Station Name")), fp.stem)
    lat, lon = next((float(m.group(1)), float(m.group(2)))
                    for l in head if (m := re.match(r"Latitude:\s*([-\d.]+)\s*,\s*Longitude:\s*([-\d.]+)", l)))
    df = pd.read_csv(fp, skiprows=hdr, low_memory=False, na_values=["", " "])
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y %H:%M")
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated()]
    return name.title(), lat, lon, df


def build_cache():
    hours = pd.date_range(START, END, freq="h")
    files = sorted(DATA_DIR.glob("hly*.csv"))
    raw_cols = ["temp", "rhum", "msl", "wdsp", "rain", "dewpt", "vappr", "wetb", "wddir"]
    names, lats, lons, arrs = [], [], [], []
    for fp in files:
        t0 = time.time()
        name, lat, lon, df = _read_station(fp)
        df = df.reindex(hours)
        a = np.stack([pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32)
                      if c in df.columns else np.full(len(hours), np.nan, np.float32)
                      for c in raw_cols], axis=1)
        names.append(name); lats.append(lat); lons.append(lon); arrs.append(a)
        print(f"  {name:22s} missing={np.isnan(a[:, :4]).mean():6.1%}  ({time.time()-t0:.1f}s)")
    raw9 = np.stack(arrs, axis=1)                                  # (H, S, 9)
    wdsp, wddir = raw9[..., 3], np.deg2rad(raw9[..., 8])
    u, v = -wdsp * np.sin(wddir), -wdsp * np.cos(wddir)            # wind *towards* (meteo convention)
    raw = np.concatenate([raw9[..., :8], u[..., None], v[..., None]], axis=-1)  # (H, S, 10)

    # distance matrix, reordered to station order
    dist = pd.read_csv(DIST_CSV, index_col=0)
    key = lambda s: re.sub(r"[^a-z]", "", s.lower())             # "Sherkin Island" == "Sherkinisland"
    dist.index = dist.index.map(key); dist.columns = dist.columns.map(key)
    missing = [n for n in names if key(n) not in dist.index]
    if missing:
        raise KeyError(f"stations not in distance matrix: {missing}; have {list(dist.index)}")
    dist_km = dist.loc[[key(n) for n in names], [key(n) for n in names]].to_numpy(np.float32)

    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, raw=raw, hours=hours.values.astype("datetime64[h]").astype(np.int64),
                        names=np.array(names), lat=np.array(lats), lon=np.array(lons), dist_km=dist_km)


# ── Public API ──────────────────────────────────────────────────────────────────

def load_data():
    """
    Returns dict with:
      X        (H, S, F) float32 normalised inputs, NaN replaced by 0
      mask     (H, S, F) bool, True where the observation exists
      raw      (H, S, F) float32 physical units with NaN
      mu, std  (F,) train-split statistics used for normalisation
      hours    (H,) datetime64[h];  hour_of_day, day_of_year (H,) ints
      names, lat, lon (S,);  dist_km (S, S);  adj (S, S) row-normalised Gaussian kernel
      split_idx {split: np.ndarray of valid "now" indices t (target t+HORIZON in range)}
    """
    if not CACHE.exists():
        build_cache()
    z = np.load(CACHE)
    raw = z["raw"]; hours = z["hours"].astype("datetime64[h]")
    H = len(hours)
    tr = _range(hours, *SPLITS["train"])
    mu = np.nanmean(raw[tr], axis=(0, 1)).astype(np.float32)
    std = np.nanstd(raw[tr], axis=(0, 1)).astype(np.float32)
    mask = ~np.isnan(raw)
    X = np.nan_to_num((raw - mu) / std, nan=0.0).astype(np.float32)

    dist = z["dist_km"]
    w = np.exp(-(dist / ADJ_SIGMA_KM) ** 2)
    adj = (w / w.sum(1, keepdims=True)).astype(np.float32)

    split_idx = {}
    for s, (a, b) in SPLITS.items():
        idx = _range(hours, a, b)
        idx = idx[(idx >= idx[0] + MAX_LOOKBACK) & (idx + HORIZON <= idx[-1])]
        split_idx[s] = idx
    hrs = pd.DatetimeIndex(hours)
    return dict(X=X, mask=mask, raw=raw, mu=mu, std=std, hours=hours,
                hour_of_day=hrs.hour.to_numpy(), day_of_year=hrs.dayofyear.to_numpy(),
                names=z["names"], lat=z["lat"], lon=z["lon"], dist_km=dist, adj=adj,
                split_idx=split_idx)


def _range(hours, a, b):
    return np.where((hours >= np.datetime64(a, "h")) & (hours <= np.datetime64(pd.Timestamp(b), "h")))[0]


def evaluate(predict_fn, data, split="val", batch=256, verbose=True):
    """
    predict_fn(t_idx: np.ndarray[int] of shape (B,)) -> np.ndarray (B, S, len(TARGETS))
        Predictions for hour t+HORIZON in PHYSICAL units, using only data at hours <= t.
    Scored on (t, station, target) where both the truth at t+H and the persistence
    value at t exist, so model and persistence are compared on identical samples.
    """
    raw = data["raw"][..., TARGET_IDX]
    idx = data["split_idx"][split]
    se_m = np.zeros(len(TARGETS)); se_p = np.zeros(len(TARGETS)); n = np.zeros(len(TARGETS))
    for i in range(0, len(idx), batch):
        t = idx[i:i + batch]
        y, p = raw[t + HORIZON], raw[t]
        pred = np.asarray(predict_fn(t), dtype=np.float64)
        ok = ~np.isnan(y) & ~np.isnan(p)
        if np.isnan(pred[ok]).any():
            raise ValueError("predict_fn returned NaN on a scored sample")
        se_m += np.where(ok, (pred - y) ** 2, 0).sum((0, 1))
        se_p += np.where(ok, (p - y) ** 2, 0).sum((0, 1))
        n += ok.sum((0, 1))
    rmse_m, rmse_p = np.sqrt(se_m / n), np.sqrt(se_p / n)
    ratio = rmse_m / rmse_p
    score = float(ratio.mean())
    if verbose:
        for k, tname in enumerate(TARGETS):
            print(f"  {tname:5s} rmse={rmse_m[k]:8.4f}  persistence={rmse_p[k]:8.4f}  ratio={ratio[k]:.4f}")
        print(f"{split}_score: {score:.6f}")
    return score, dict(zip(TARGETS, rmse_m.tolist()))


if __name__ == "__main__":
    t0 = time.time()
    d = load_data()
    print(f"X {d['X'].shape}  stations={len(d['names'])}  loaded in {time.time()-t0:.1f}s")
    for s, i in d["split_idx"].items():
        print(f"  {s:5s}: {len(i):6d} forecast times")
    raw = d["raw"][..., TARGET_IDX]
    print("\nPersistence (sanity, ratio must be 1.0):")
    evaluate(lambda t: np.nan_to_num(raw[t]), d)
