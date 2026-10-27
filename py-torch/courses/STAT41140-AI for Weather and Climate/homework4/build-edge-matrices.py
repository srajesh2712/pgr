"""
build_distance_edges.py
=======================
Builds a single static 25×25 distance-based adjacency matrix for the
Irish weather station GNN.

Method:
  - Haversine distance between every pair of stations
  - Gaussian decay: w = exp(-d^2 / 2*sigma^2)
  - Zero out edges below threshold (sparsity)
  - Self-loops set to 1.0 (diagonal)
  - Row-normalise (standard for GCN message passing)

Outputs (in ./data/edge_matrices/):
  distance_adj.csv       – row-normalised weighted adjacency (25×25)
  distance_adj.npy       – same, as numpy array
  station_distances.csv  – raw distances in km (25×25)
"""

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR       = Path(__file__).parent / "data"
OUT_DIR        = Path(__file__).parent / "data/edge_matrices"
GAUSS_SIGMA    = 150.0   # km — Gaussian decay bandwidth
EDGE_THRESHOLD = 0.1     # zero out weights below this
ROW_NORMALISE  = True
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Station list ───────────────────────────────────────────────────────────────
STATIONS = [
    (1875, "Athenry"),
    (675,  "Ballyhaise"),
    (2375, "Belmullet"),
    (3723, "Casement"),
    (2175, "Claremorris"),
    (3904, "Cork Airport"),
    (532,  "Dublin Airport"),
    (1375, "Dunsany"),
    (2075, "Finner"),
    (1475, "Gurteen"),
    (1775, "Johnstown Castle 2"),
    (4935, "Knock Airport"),
    (275,  "Mace Head"),
    (1575, "Malin Head"),
    (1275, "Markree"),
    (575,  "Moore Park"),
    (1975, "Mt Dillon"),
    (875,  "Mullingar"),
    (1175, "Newport"),
    (375,  "Oak Park"),
    (175,  "Phoenix Park"),
    (1075, "Roches Point"),
    (518,  "Shannon Airport"),
    (775,  "Sherkinisland"),
    (2275, "Valentia Observatory"),
]

# ── Helper: parse lat/lon from CSV header ──────────────────────────────────────
def parse_latlon(stno: int, name: str):
    path = DATA_DIR / f"hly{stno}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # e.g. "Latitude: 53.298" or "Lat: 53.298"
            m = re.search(r"[Ll]at(?:itude)?[:\s,]+(-?\d+\.?\d*)", line)
            if m:
                lat = float(m.group(1))
                # next line or same line for lon
                rest = f.readline()
                m2 = re.search(r"[Ll]on(?:gitude)?[:\s,]+(-?\d+\.?\d*)", line + rest)
                if m2:
                    return lat, float(m2.group(1))
    raise ValueError(f"Could not parse lat/lon for {name} ({stno})")

# ── Haversine distance ─────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Reading station coordinates …")
    coords = {}
    for stno, name in STATIONS:
        try:
            lat, lon = parse_latlon(stno, name)
            coords[name] = (lat, lon)
            print(f"  {name:25s}  lat={lat:.3f}  lon={lon:.3f}")
        except Exception as e:
            print(f"  [WARN] {name}: {e}")

    names = [n for _, n in STATIONS if n in coords]
    S = len(names)
    print(f"\n{S} stations with coordinates.\n")

    # Distance matrix
    dist_km = np.zeros((S, S), dtype=np.float32)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                dist_km[i, j] = haversine_km(*coords[a], *coords[b])

    # Gaussian decay
    adj = np.exp(-dist_km**2 / (2 * GAUSS_SIGMA**2)).astype(np.float32)

    # Self-loops
    np.fill_diagonal(adj, 1.0)

    # Threshold
    adj[adj < EDGE_THRESHOLD] = 0.0
    np.fill_diagonal(adj, 1.0)  # keep self-loop after threshold

    # Row-normalise
    if ROW_NORMALISE:
        row_sums = adj.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        adj = adj / row_sums

    # Save
    dist_df = pd.DataFrame(dist_km, index=names, columns=names)
    dist_df.to_csv(OUT_DIR / "station_distances.csv")

    adj_df = pd.DataFrame(adj, index=names, columns=names)
    adj_df.to_csv(OUT_DIR / "distance_adj.csv")
    np.save(OUT_DIR / "distance_adj.npy", adj)

    print(f"Saved:")
    print(f"  {OUT_DIR / 'station_distances.csv'}  (distances in km)")
    print(f"  {OUT_DIR / 'distance_adj.csv'}        (adjacency matrix)")
    print(f"  {OUT_DIR / 'distance_adj.npy'}        (numpy array)")
    print(f"\nAdjacency stats:")
    print(f"  Non-zero edges : {(adj > 0).sum()} / {S*S}")
    print(f"  Min weight     : {adj[adj > 0].min():.4f}")
    print(f"  Max weight     : {adj.max():.4f}")

if __name__ == "__main__":
    main()