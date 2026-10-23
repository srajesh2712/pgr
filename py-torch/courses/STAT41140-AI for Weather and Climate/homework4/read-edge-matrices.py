"""
inspect_edge_matrices.py
========================
Query and inspect the contents of edge_matrices.npz and edge_matrices.h5
produced by build_edge_matrices.py.

Usage:
    uv run python inspect_edge_matrices.py
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path

NPZ_PATH = Path(__file__).parent /"data/edge_matrices/edge_matrices.npz"
H5_PATH  = Path(__file__).parent / "data/edge_matrices/edge_matrices.h5"

# ── NPZ summary ───────────────────────────────────────────────────────────────

print("=" * 60)
print("NPZ FILE SUMMARY")
print("=" * 60)
d = np.load(NPZ_PATH, allow_pickle=True)
for k in d.files:
    arr = d[k]
    print(f"  {k:20s}  shape={str(arr.shape):25s}  dtype={arr.dtype}")

# ── HDF5 summary ──────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("HDF5 FILE SUMMARY  (+ metadata attributes)")
print("=" * 60)
with h5py.File(H5_PATH, "r") as f:
    for k in f.keys():
        ds = f[k]
        print(f"  {k:20s}  shape={str(ds.shape):25s}  dtype={ds.dtype}")
    print()
    print("  Attributes:")
    for k, v in f.attrs.items():
        print(f"    {k} = {v}")

# ── Station names & date range ────────────────────────────────────────────────

names      = [s.decode() if isinstance(s, bytes) else s for s in d["station_names"]]
timestamps = [t.decode() if isinstance(t, bytes) else t for t in d["timestamps"]]

print()
print("=" * 60)
print("STATIONS")
print("=" * 60)
for i, n in enumerate(names):
    print(f"  [{i}] {n}")

print()
print("=" * 60)
print("DATE RANGE")
print("=" * 60)
print(f"  First : {timestamps[0]}")
print(f"  Last  : {timestamps[-1]}")
print(f"  Total : {len(timestamps)} hours")

# ── Helper ────────────────────────────────────────────────────────────────────

def show_matrix(label, mat, names):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)
    df = pd.DataFrame(mat, index=names, columns=names).round(4)
    print(df.to_string())

# ── Sample matrices ───────────────────────────────────────────────────────────

show_matrix(f"FINAL ADJ (thresholded, row-normalised) — hour 0  ({timestamps[0]})",
            d["adj"][0], names)

show_matrix(f"FINAL ADJ (thresholded, row-normalised) — hour 100  ({timestamps[100]})",
            d["adj"][100], names)

show_matrix(f"BINARY ADJ (edge structure) — hour 0  ({timestamps[0]})",
            d["binary_adj"][0], names)

show_matrix(f"RAW CORRELATION — hour 0  ({timestamps[0]})",
            d["corr"][0], names)

show_matrix("DISTANCE WEIGHT MATRIX (static)",
            d["distance"], names)

# ── Sparsity check ────────────────────────────────────────────────────────────

print()
print("=" * 60)
print("SPARSITY CHECK (binary_adj)")
print("=" * 60)
S = len(names)
possible_edges = S * S
for h_idx in [0, 100, 1000, 5000]:
    nnz = int((d["binary_adj"][h_idx] > 0).sum())
    print(f"  Hour {h_idx:>5} ({timestamps[h_idx][:10]}): {nnz}/{possible_edges} edges  "
          f"({100*nnz/possible_edges:.0f}% dense)")

# ── Correlation stats across all hours ───────────────────────────────────────

print()
print("=" * 60)
print("ADJ WEIGHT STATS ACROSS ALL HOURS  (off-diagonal pairs)")
print("=" * 60)
adj = d["adj"]   # (H, S, S)
rows = []
for i in range(S):
    for j in range(S):
        if i == j:
            continue
        vals = adj[:, i, j]
        rows.append({
            "from":      names[i],
            "to":        names[j],
            "mean_w":    round(float(vals.mean()), 4),
            "min_w":     round(float(vals.min()),  4),
            "max_w":     round(float(vals.max()),  4),
            "std_w":     round(float(vals.std()),  4),
            "pct_zero":  round(float((vals == 0).mean() * 100), 1),
        })

stats_df = pd.DataFrame(rows).sort_values("mean_w", ascending=False)
print(stats_df.to_string(index=False))

# ── Lookup by timestamp ───────────────────────────────────────────────────────

print()
print("=" * 60)
print("LOOKUP BY TIMESTAMP")
print("=" * 60)

def get_matrix(ts_str: str, kind: str = "adj"):
    """
    Return adjacency matrix for a given ISO timestamp string.
    kind: 'adj' | 'corr' | 'binary_adj' | 'distance'
    """
    try:
        idx = timestamps.index(ts_str)
    except ValueError:
        matches = [i for i, t in enumerate(timestamps) if t.startswith(ts_str)]
        if not matches:
            raise KeyError(f"Timestamp '{ts_str}' not found")
        idx = matches[0]
        print(f"  (matched '{timestamps[idx]}')")
    if kind == "distance":
        return pd.DataFrame(d["distance"], index=names, columns=names)
    return pd.DataFrame(d[kind][idx], index=names, columns=names)

example_ts = timestamps[500]
print(f"\n  Example: adj matrix at {example_ts}")
print(get_matrix(example_ts, "adj").round(4).to_string())

# ── PyG edge_index example ────────────────────────────────────────────────────

print()
print("=" * 60)
print("PyG EDGE_INDEX EXAMPLE  (hour 0)")
print("=" * 60)
import torch
A = torch.tensor(d["adj"][0], dtype=torch.float32)
edge_index  = A.nonzero(as_tuple=False).t().contiguous()   # (2, E)
edge_weight = A[edge_index[0], edge_index[1]]               # (E,)
print(f"  edge_index  shape : {tuple(edge_index.shape)}")
print(f"  edge_weight shape : {tuple(edge_weight.shape)}")
print(f"  edge_index:\n{edge_index}")
print(f"  edge_weight: {edge_weight.round(decimals=4)}")
print(f"\n  Station index map: { {i: n for i, n in enumerate(names)} }")

print()
print("Done.")