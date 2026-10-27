# Homework 4 — Spatio-Temporal GNN for Irish Weather Forecasting

**Course:** STAT41140 — AI for Weather and Climate  
**University:** University College Dublin (UCD)

---

## Overview

This project implements a **Spatio-Temporal Graph Convolutional Network (ST-GCN)** to predict next-hour weather conditions at 25 Irish Met Éireann stations across Ireland.

Given the last N hours of multi-variate weather readings across all stations, the model predicts the next hour's values at every station simultaneously — treating the station network as a spatial graph.

---

## Architecture

```
Input: (Batch, Time, Stations, Features)
         ↓
    GCN Layers × 2     ← spatial message passing using distance-based adjacency
         ↓
    GRU (2 layers)     ← temporal sequence modelling
         ↓
    MLP Head           ← per-station output
         ↓
Output: (Batch, Stations, Target Features)
```

- **Graph:** Static 25×25 distance-based adjacency matrix (Gaussian decay, row-normalised)
- **Spatial:** 2-layer Graph Convolution (`H = σ(A × X × W)`)
- **Temporal:** 2-layer GRU over the sequence of GCN embeddings
- **Input features:** `temp, rhum, msl, wdsp, rain, dewpt, vappr, wetb`
- **Target features:** `temp, rhum, msl, wdsp`

---

## Stations (25 total)

| Station | Station | Station |
|---|---|---|
| Athenry | Ballyhaise | Belmullet |
| Casement | Claremorris | Cork Airport |
| Dublin Airport | Dunsany | Finner |
| Gurteen | Johnstown Castle 2 | Knock Airport |
| Mace Head | Malin Head | Markree |
| Moore Park | Mt Dillon | Mullingar |
| Newport | Oak Park | Phoenix Park |
| Roches Point | Shannon Airport | Sherkinisland |
| Valentia Observatory | | |

---

## Project Structure

```
homework4/
├── download-met-data.py      # Step 1: download raw CSVs from Met Éireann
├── build-edge-matrices.py    # Step 2: build the distance adjacency matrix
├── train-gnn.py              # Step 3: train the ST-GCN model
├── read-edge-matrices.py     # Utility: inspect edge matrix files
└── data/
    ├── hly<stno>.csv         # Raw hourly station data (one per station)
    └── edge_matrices/
        ├── distance_adj.csv          # Row-normalised adjacency (25×25)
        ├── distance_adj.npy          # Same, as numpy array
        ├── station_distances.csv     # Raw distances in km (25×25)
        └── gnn_model.pt              # Saved model checkpoint
```

---

## Setup

### Requirements

```bash
pip install torch numpy pandas requests h5py
```

> Uses `uv` as the package runner. Install from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv) if needed.

---

## Quickstart (Run in Order)

### Step 1 — Download Station Data

Downloads hourly CSVs for all 25 stations from Met Éireann's climate data portal.

```bash
python download-met-data.py
```

Output: `data/hly<stno>.csv` for each station (~10–60 MB each).  
Already downloaded files are skipped automatically.

---

### Step 2 — Build Edge Matrices

Computes the static 25×25 adjacency matrix using Haversine distances between stations and a Gaussian decay kernel.

```bash
python build-edge-matrices.py
```

Output in `data/edge_matrices/`:
- `distance_adj.npy` — the adjacency matrix used during training
- `distance_adj.csv` — human-readable version
- `station_distances.csv` — raw distances in km

**Key parameters** (edit in script):

| Parameter | Default | Description |
|---|---|---|
| `GAUSS_SIGMA` | 150 km | Gaussian decay bandwidth |
| `EDGE_THRESHOLD` | 0.1 | Zero out weights below this |
| `ROW_NORMALISE` | True | Normalise each row to sum to 1 |

---

### Step 3 — Train the Model

```bash
# Basic run (uses defaults)
python train-gnn.py

# Custom hyperparameters
python train-gnn.py --epochs 50 --lr 0.001 --seq-len 6

# Evaluate only (load saved model, skip training)
python train-gnn.py --eval-only

# Resume training from checkpoint
python train-gnn.py --resume

# Run inference after training (predict next hour)
python train-gnn.py --predict

# Force CPU
python train-gnn.py --no-gpu
```

**Default hyperparameters:**

| Parameter | Value |
|---|---|
| Sequence length | 24 hours |
| GCN hidden dim | 64 |
| GRU hidden dim | 128 |
| GCN layers | 2 |
| GRU layers | 2 |
| Epochs | 30 |
| Learning rate | 5e-4 |
| Batch size | 256 |
| Dropout | 0.2 |
| Early stopping patience | 8 epochs |
| Train / Val / Test split | 75% / 15% / 10% |

---

## Training Output

After training, the script prints per-feature and per-station RMSE and MAE in original units (denormalised):

```
Per-feature  RMSE  |  MAE  (denormalised, original units)
  Feature         RMSE       MAE
  ───────────────────────────────────
  temp          1.2345    0.9876
  rhum          4.5678    3.2100
  msl           2.1234    1.5678
  wdsp          1.8765    1.3210

Per-station  RMSE  (all target features averaged)
  Dublin Airport          1.4321
  ...
```

The best model (lowest validation loss) is saved to `data/edge_matrices/gnn_model.pt`.

---

## Inspecting Edge Matrices

To inspect the adjacency matrix and distance files:

```bash
python read-edge-matrices.py
```

This shows matrix shapes, station names, date ranges, sparsity stats, and example PyG-compatible `edge_index` tensors.

---

## Notes for New Team Members

- **Large files:** Several CSV files are 50–60 MB. The `.h5` edge matrix file exceeds GitHub's 100 MB limit — use Git LFS or regenerate it locally with `build-edge-matrices.py`.
- **Data is not included in the repo:** Run `download-met-data.py` first to populate `data/`.
- **GPU recommended:** Training on 25 stations × 24-hour sequences is feasible on CPU but significantly faster on CUDA.
- **Model checkpoint:** `gnn_model.pt` stores model weights, normalisation statistics (`mu`, `std`), and hyperparameters — load it with `--eval-only` to skip retraining.
