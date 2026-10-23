"""
build_edge_matrices.py  –  GPU-accelerated edition
====================================================
Uses CuPy (CUDA) for the Pearson correlation matrix computation.
Falls back to NumPy automatically if no CUDA GPU is available.

Install deps:
  pip install pandas numpy scipy
  # For GPU (pick the right CUDA version for your system):
  pip install cupy-cuda12x      # CUDA 12.x
  pip install cupy-cuda11x      # CUDA 11.x
  # Check your CUDA version with:  nvcc --version  or  nvidia-smi

Outputs (in ./edge_matrices/):
  distance_weight_matrix.csv          – 1 static file
  pearson_correlation/corr_<ts>.csv   – 1 per hour
  hybrid/hybrid_<ts>.csv              – 1 per hour
"""

import math
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# ── GPU setup: use PyTorch CUDA (already installed), fall back to NumPy ───────
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() returned False")
    DEVICE = torch.device("cuda")
    GPU = True
    print(f"✓ PyTorch CUDA detected – running on {torch.cuda.get_device_name(0)}")
except Exception as _e:
    GPU = False
    DEVICE = None
    print(f"⚠  PyTorch CUDA not available ({_e})\n   Falling back to NumPy (CPU).")

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR     = Path(__file__).parent / "data"
OUT_DIR      = Path(__file__).parent / "data/edge_matrices"
CORR_WINDOW  = 0      # 0 = single hour only; increase for rolling-window correlation
GAUSS_SIGMA  = 150    # km – Gaussian decay for distance weight

# Hybrid weight: distance drives structure, correlation modulates it
# 0.7 distance + 0.3 correlation = geography-first, weather-aware
HYBRID_ALPHA      = 0.3    # weight given to correlation  (1-HYBRID_ALPHA = distance)
EDGE_THRESHOLD    = 0.1    # zero out edges below this weight (sparsity)
ROW_NORMALISE     = True   # row-normalise adjacency (standard for GCN)

WRITE_WORKERS = 4     # parallel CSV writers (I/O bound, not GPU bound)

CORR_FEATURES = [
    "temp", "rhum", "msl", "wdsp", "rain", "dewpt", "vappr", "wetb"
]

HEADER_KEYWORDS = {
    "date", "temp", "rain", "rhum", "wdsp", "msl",
    "dewpt", "wetb", "wddir", "sun", "vis", "clht", "clamt", "vappr"
}

# ── Header parsing ────────────────────────────────────────────────────────────

def parse_header(filepath: Path) -> dict:
    meta = {"name": None, "height_m": None, "lat": None, "lon": None}
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 60:
                break
            s = line.strip()
            m = re.match(r"Station Name:\s*(.+)", s, re.I)
            if m:
                meta["name"] = m.group(1).strip()
            m = re.match(r"Station Height:\s*([\d.]+)", s, re.I)
            if m:
                meta["height_m"] = float(m.group(1))
            m = re.match(r"Latitude:\s*([-\d.]+)\s*,\s*Longitude:\s*([-\d.]+)", s, re.I)
            if m:
                meta["lat"] = float(m.group(1))
                meta["lon"] = float(m.group(2))
    return meta


def find_data_bounds(filepath: Path):
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        tokens = {t.strip().lower().strip('"') for t in line.split(",")}
        if len(tokens & HEADER_KEYWORDS) >= 3:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Cannot locate column-header row in {filepath}")
    return header_idx, lines


def load_csv(filepath: Path) -> pd.DataFrame:
    header_idx, lines = find_data_bounds(filepath)
    raw_cols = [c.strip().strip('"').lower() for c in lines[header_idx].split(",")]

    seen_ind = 0
    col_names = []
    for c in raw_cols:
        if c == "ind":
            col_names.append(f"ind_{seen_ind}")
            seen_ind += 1
        else:
            col_names.append(c)

    clean_lines = []
    for ln in lines[header_idx + 1:]:
        s = ln.strip()
        if s and s[0] in "0123456789\"'-":
            clean_lines.append(ln)

    df = pd.read_csv(
        StringIO("".join(clean_lines)),
        header=None,
        names=col_names,
        na_values=["", " "],
        on_bad_lines="warn",
        engine="python",
    )

    date_col = next((c for c in ("date", "datetime", col_names[0]) if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"Cannot identify date column in {filepath}")
    if date_col != "datetime":
        df = df.rename(columns={date_col: "datetime"})

    sample = df["datetime"].dropna().iloc[0] if not df["datetime"].dropna().empty else None
    fmt = None
    for candidate_fmt in (
        "%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
        "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S",
    ):
        try:
            pd.to_datetime(str(sample), format=candidate_fmt)
            fmt = candidate_fmt
            break
        except (ValueError, TypeError):
            pass

    df["datetime"] = (
        pd.to_datetime(df["datetime"], format=fmt, utc=True) if fmt
        else pd.to_datetime(df["datetime"], utc=True)
    )
    df = df.set_index("datetime").sort_index()
    for col in df.columns:
        if not col.startswith("ind"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Distance matrix (CPU, computed once) ─────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(stations: list) -> pd.DataFrame:
    names = [s["name"] for s in stations]
    n = len(names)
    W = np.zeros((n, n))
    for i, si in enumerate(stations):
        for j, sj in enumerate(stations):
            if i == j:
                W[i, j] = 1.0
            else:
                d = haversine_km(si["lat"], si["lon"], sj["lat"], sj["lon"])
                W[i, j] = math.exp(-d ** 2 / (2 * GAUSS_SIGMA ** 2))
    return pd.DataFrame(W, index=names, columns=names)


# ── GPU-accelerated Pearson correlation ───────────────────────────────────────

def pearson_matrix_gpu(mat: np.ndarray) -> np.ndarray:
    """
    mat : (n_stations, n_features*window_steps)  float32.
    Returns (n_stations, n_stations) Pearson r matrix clipped to [0,1].
    Runs on CUDA via PyTorch if available, otherwise pure NumPy.
    """
    if GPU:
        X = torch.tensor(mat, dtype=torch.float32, device=DEVICE)

        # replace NaN with 0 before centering
        X = torch.nan_to_num(X, nan=0.0)

        # zero-centre each row
        X -= X.mean(dim=1, keepdim=True)

        # row norms
        norms = torch.linalg.norm(X, dim=1, keepdim=True)
        norms = torch.where(norms < 1e-9, torch.ones_like(norms), norms)
        X = X / norms

        # correlation via matmul
        C = X @ X.T
        C = torch.clamp(C, 0.0, 1.0)
        C.fill_diagonal_(1.0)

        return C.cpu().numpy()
    else:
        # NumPy fallback
        X = np.nan_to_num(mat.astype(np.float32), nan=0.0)
        X -= X.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        X /= norms
        C = X @ X.T
        C = np.clip(C, 0.0, 1.0)
        np.fill_diagonal(C, 1.0)
        return C


# ── Pre-build feature tensor for all stations × all hours ────────────────────

def build_feature_tensor(stations, data, all_hours, window):
    """
    Returns a 3-D numpy array:
        feat[h, s, :]  =  Z-score normalised feature vector for station s at hour h
                          (NaN where data is missing)
    Shape: (H, S, n_features * window_steps)
    where window_steps = 2*window+1

    Each feature is normalised independently across ALL stations and hours
    so that temperature, pressure, humidity etc. contribute equally to
    the Pearson correlation (avoids one dominant variable inflating r).
    """
    H = len(all_hours)
    S = len(stations)
    window_steps = 2 * window + 1
    F = len(CORR_FEATURES)
    feat = np.full((H, S, F * window_steps), np.nan, dtype=np.float32)

    hour_to_idx = {ts: i for i, ts in enumerate(all_hours)}

    for s_idx, station in enumerate(stations):
        name = station["name"]
        df = data.get(name)
        if df is None:
            continue
        available = [c for c in CORR_FEATURES if c in df.columns]
        if not available:
            continue

        for ts, row in df[available].iterrows():
            base_hour = ts.floor("h")
            if base_hour not in hour_to_idx:
                continue
            for delta in range(-window, window + 1):
                target_hour = base_hour + pd.Timedelta(hours=delta)
                if target_hour not in hour_to_idx:
                    continue
                t_idx     = hour_to_idx[target_hour]
                step_slot = delta + window
                f_start   = step_slot * len(available)
                f_end     = f_start + len(available)
                vals = row.values.astype(np.float32)
                slot = feat[t_idx, s_idx, f_start:f_end]
                mask = np.isnan(slot)
                slot[mask] = vals[mask]

    # ── Z-score normalise each feature across all stations & hours ────────────
    # feat shape: (H, S, F*W)  — normalise along the H*S axis per feature slot
    feat_flat = feat.reshape(-1, feat.shape[-1])          # (H*S, F*W)
    with np.errstate(invalid="ignore"):
        mu  = np.nanmean(feat_flat, axis=0, keepdims=True)
        std = np.nanstd (feat_flat, axis=0, keepdims=True)
        std = np.where(std < 1e-9, 1.0, std)
        feat_flat = (feat_flat - mu) / std
    feat = feat_flat.reshape(H, S, -1)

    return feat


# ── Write helpers (run in thread pool) ───────────────────────────────────────

def write_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. load all station CSVs
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR.resolve()}")

    stations, data = [], {}
    print(f"\nLoading {len(csv_files)} station file(s) …\n")
    for fp in csv_files:
        meta = parse_header(fp)
        if not meta["name"]:
            meta["name"] = fp.stem
        print(f"  {meta['name']:30s}  lat={meta['lat']}  lon={meta['lon']}")
        try:
            df = load_csv(fp)
            stations.append(meta)
            data[meta["name"]] = df
            print(f"    → {len(df)} rows, cols: {[c for c in df.columns if not c.startswith('ind')]}")
        except Exception as e:
            print(f"    [ERROR] {e}")

    valid_stations = [s for s in stations if s["lat"] is not None]
    if not valid_stations:
        raise RuntimeError("No valid stations loaded.")
    names = [s["name"] for s in valid_stations]

    # 2. all unique hours
    all_times = pd.DatetimeIndex(
        pd.concat([pd.Series(index=df.index, dtype=float)
                   for df in data.values()]).index
    )
    all_hours = all_times.floor("h").unique().sort_values()
    H = len(all_hours)
    print(f"\nTotal hours: {H}")

    # 3. static distance matrix
    dist_W = build_distance_matrix(valid_stations)
    dist_W.to_csv(OUT_DIR / "distance_weight_matrix.csv")
    print(f"Saved distance matrix.\n")

    # 4. pre-build feature tensor  (H, S, F*W)  — done once on CPU
    print("Pre-building feature tensor (CPU) …")
    t0 = time.time()
    feat = build_feature_tensor(valid_stations, data, all_hours, CORR_WINDOW)
    print(f"  Done in {time.time()-t0:.1f}s  shape={feat.shape}\n")

    # 5. transfer full tensor to GPU once
    if GPU:
        print("Transferring tensor to GPU …")
        feat_gpu = torch.tensor(feat, dtype=torch.float32, device=DEVICE)
        print(f"  Done. Tensor on GPU: {feat_gpu.shape}  ({feat_gpu.element_size() * feat_gpu.nelement() / 1e6:.1f} MB)\n")

    # 6. compute all matrices on GPU, accumulate in RAM, save to single files
    dist_np = dist_W.values.astype(np.float32)

    corr_all   = np.zeros((H, len(names), len(names)), dtype=np.float32)
    hybrid_all = np.zeros((H, len(names), len(names)), dtype=np.float32)

    print(f"Computing per-hour matrices {'(GPU)' if GPU else '(CPU)'} …")
    t0 = time.time()

    for h_idx, ts in enumerate(all_hours):
        if GPU:
            C = pearson_matrix_gpu(feat_gpu[h_idx].cpu().numpy())
        else:
            C = pearson_matrix_gpu(feat[h_idx])

        corr_all[h_idx] = C

        # geography-first hybrid: distance is the base, correlation modulates
        W = (1 - HYBRID_ALPHA) * dist_np + HYBRID_ALPHA * C

        # threshold – zero out weak edges
        W[W < EDGE_THRESHOLD] = 0.0
        np.fill_diagonal(W, 1.0)   # always keep self-loops

        # row-normalise so each node's incoming weights sum to 1
        if ROW_NORMALISE:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            W = W / row_sums

        hybrid_all[h_idx] = W

        if (h_idx + 1) % 500 == 0 or (h_idx + 1) == H:
            elapsed = time.time() - t0
            rate = (h_idx + 1) / elapsed
            eta  = (H - h_idx - 1) / rate
            print(f"  {h_idx+1:>6}/{H}  {rate:.0f} hr/s  ETA {eta/60:.1f} min")

    print(f"\nMatrices computed in {time.time()-t0:.1f}s — saving to single files …")

    # timestamps as ISO strings for the index
    timestamps = [ts.isoformat() for ts in all_hours]

    # binary adjacency: 1 where edge exists, 0 elsewhere (for GAT / edge-index models)
    binary_all = (hybrid_all > 0).astype(np.float32)

    # ── Option A: NPZ ─────────────────────────────────────────────────────────
    npz_path = OUT_DIR / "edge_matrices.npz"
    np.savez_compressed(
        npz_path,
        corr=corr_all,           # (H, S, S) raw Pearson correlation
        adj=hybrid_all,          # (H, S, S) final adjacency: thresholded, row-normalised
        binary_adj=binary_all,   # (H, S, S) 0/1 structure only
        distance=dist_np,        # (S, S)    static distance weights
        timestamps=np.array(timestamps),
        station_names=np.array(names),
    )
    print(f"  Saved → {npz_path}  ({npz_path.stat().st_size/1e6:.1f} MB)")

    # ── Option B: HDF5 ────────────────────────────────────────────────────────
    try:
        import h5py
        h5_path = OUT_DIR / "edge_matrices.h5"
        S = len(names)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("corr",       data=corr_all,   compression="gzip", chunks=(1, S, S))
            f.create_dataset("adj",        data=hybrid_all, compression="gzip", chunks=(1, S, S))
            f.create_dataset("binary_adj", data=binary_all, compression="gzip", chunks=(1, S, S))
            f.create_dataset("distance",   data=dist_np)
            f.create_dataset("timestamps",    data=np.array(timestamps, dtype="S32"))
            f.create_dataset("station_names", data=np.array(names,      dtype="S64"))
            # metadata attributes
            f.attrs["hybrid_alpha"]   = HYBRID_ALPHA
            f.attrs["edge_threshold"] = EDGE_THRESHOLD
            f.attrs["row_normalised"] = ROW_NORMALISE
            f.attrs["gauss_sigma_km"] = GAUSS_SIGMA
        print(f"  Saved → {h5_path}  ({h5_path.stat().st_size/1e6:.1f} MB)")
    except ImportError:
        print("  (h5py not installed — skipped HDF5; NPZ is available)")

    # ── static distance CSV still useful as a quick reference ─────────────────
    dist_W.to_csv(OUT_DIR / "distance_weight_matrix.csv")

    total = time.time() - t0
    print(f"\nAll done in {total/60:.1f} min  →  {OUT_DIR.resolve()}/")
    print("""
How to load in your GNN:
  import numpy as np
  data = np.load('edge_matrices/edge_matrices.npz', allow_pickle=True)
  A_t        = data['adj'][t]         # (S, S) row-normalised weighted adjacency
  A_bin_t    = data['binary_adj'][t]  # (S, S) 0/1 structure (for GAT/edge_index)
  A_static   = data['distance']       # (S, S) static distance weights

  # HDF5 — load one hour without reading the full file:
  import h5py
  with h5py.File('edge_matrices/edge_matrices.h5', 'r') as f:
      A_t = f['adj'][t]

  # Convert to PyG edge_index format:
  import torch
  A = torch.tensor(data['adj'][t])
  edge_index = A.nonzero(as_tuple=False).t().contiguous()  # (2, E)
  edge_weight = A[edge_index[0], edge_index[1]]             # (E,)
""")


if __name__ == "__main__":
    main()