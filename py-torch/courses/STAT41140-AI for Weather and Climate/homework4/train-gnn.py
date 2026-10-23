"""
train_gnn.py  –  Spatio-Temporal GNN for Irish Weather Station Next-Hour Prediction
=====================================================================================
Architecture : ST-GCN (Spatial Graph Conv + Temporal GRU)
Task         : Given the last N hours of features at all stations,
               predict the next hour's features at all stations.
Dataset      : edge_matrices/edge_matrices.npz  (built by build_edge_matrices.py)
               homework4/data/*.csv              (raw station CSVs, for node features)

Usage:
    uv run python train_gnn.py
    uv run python train_gnn.py --epochs 50 --lr 0.001 --seq-len 6
    uv run python train_gnn.py --predict  # run inference on last window after training
"""

import argparse
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ─────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).parent          # homework4/
DATA_DIR   = _HERE / "data"               # homework4/data/
NPZ_PATH   = DATA_DIR / "edge_matrices" / "edge_matrices.npz"   # homework4/edge_matrices/
MODEL_PATH = DATA_DIR / "edge_matrices" / "gnn_model.pt"

# Features to use as node inputs (must exist in station CSVs)
NODE_FEATURES = ["temp", "rhum", "msl", "wdsp", "rain", "dewpt", "vappr", "wetb"]

# Features to predict at the next hour (subset of NODE_FEATURES)
TARGET_FEATURES = ["temp", "rhum", "msl", "wdsp"]

SEQ_LEN      = 24      # hours of history fed into the model
HIDDEN_DIM   = 64     # GCN hidden size
GRU_HIDDEN   = 128    # GRU hidden size
NUM_GCN_LAYERS = 2
EPOCHS       = 30
LR           = 5e-4
BATCH_SIZE   = 32
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.10
DROPOUT      = 0.2
EARLY_STOP   = 8      # patience (epochs)

HEADER_KEYWORDS = {
    "date","temp","rain","rhum","wdsp","msl",
    "dewpt","wetb","wddir","sun","vis","clht","clamt","vappr"
}

# ── Argument parsing ───────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",   type=int,   default=EPOCHS)
    p.add_argument("--lr",       type=float, default=LR)
    p.add_argument("--seq-len",  type=int,   default=SEQ_LEN)
    p.add_argument("--hidden",   type=int,   default=HIDDEN_DIM)
    p.add_argument("--batch",    type=int,   default=BATCH_SIZE)
    p.add_argument("--predict",   action="store_true", help="run inference after training")
    p.add_argument("--eval-only", action="store_true", help="skip training, load saved model and evaluate")
    p.add_argument("--no-gpu",    action="store_true", help="force CPU")
    return p.parse_args()

# ── CSV loading (same as build_edge_matrices.py) ───────────────────────────────

def find_data_bounds(filepath):
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        tokens = {t.strip().lower().strip('"') for t in line.split(",")}
        if len(tokens & HEADER_KEYWORDS) >= 3:
            return i, lines
    raise ValueError(f"Cannot locate column-header row in {filepath}")

def load_csv(filepath):
    header_idx, lines = find_data_bounds(filepath)
    raw_cols = [c.strip().strip('"').lower() for c in lines[header_idx].split(",")]
    seen_ind, col_names = 0, []
    for c in raw_cols:
        if c == "ind":
            col_names.append(f"ind_{seen_ind}"); seen_ind += 1
        else:
            col_names.append(c)
    clean = [ln for ln in lines[header_idx+1:] if ln.strip() and ln.strip()[0] in "0123456789\"'-"]
    df = pd.read_csv(StringIO("".join(clean)), header=None, names=col_names,
                     na_values=["", " "], on_bad_lines="warn", engine="python")
    date_col = next((c for c in ("date","datetime",col_names[0]) if c in df.columns), None)
    if date_col != "datetime":
        df = df.rename(columns={date_col: "datetime"})
    sample = df["datetime"].dropna().iloc[0] if not df["datetime"].dropna().empty else None
    fmt = None
    for f in ("%d-%b-%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
              "%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            pd.to_datetime(str(sample), format=f); fmt = f; break
        except (ValueError, TypeError): pass
    df["datetime"] = (pd.to_datetime(df["datetime"], format=fmt, utc=True) if fmt
                      else pd.to_datetime(df["datetime"], utc=True))
    df = df.set_index("datetime").sort_index()
    for col in df.columns:
        if not col.startswith("ind"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def parse_header(filepath):
    meta = {"name": None, "lat": None, "lon": None, "height_m": None}
    import re
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 60: break
            s = line.strip()
            m = re.match(r"Station Name:\s*(.+)", s, re.I)
            if m: meta["name"] = m.group(1).strip()
            m = re.match(r"Latitude:\s*([-\d.]+)\s*,\s*Longitude:\s*([-\d.]+)", s, re.I)
            if m: meta["lat"], meta["lon"] = float(m.group(1)), float(m.group(2))
    return meta

# ── Build node-feature tensor (H, S, F) ───────────────────────────────────────

def build_node_features(stations, data, all_hours, features):
    """
    Returns feat (H, S, F) float32 — one row per hour per station.
    Missing values are forward-filled then backward-filled per station.
    """
    H, S, F = len(all_hours), len(stations), len(features)
    feat = np.full((H, S, F), np.nan, dtype=np.float32)
    hour_idx = {ts: i for i, ts in enumerate(all_hours)}

    for s_idx, station in enumerate(stations):
        df = data.get(station["name"])
        if df is None:
            continue
        avail = [c for c in features if c in df.columns]
        if not avail:
            continue
        # resample to hourly mean
        hourly = df[avail].resample("1h").mean()
        for ts, row in hourly.iterrows():
            ts_floor = ts.floor("h")
            if ts_floor in hour_idx:
                h = hour_idx[ts_floor]
                for fi, col in enumerate(avail):
                    feat[h, s_idx, fi] = float(row[col]) if not np.isnan(row[col]) else np.nan

        # fill missing within this station column by column
        for fi in range(F):
            col_vals = feat[:, s_idx, fi]
            mask = ~np.isnan(col_vals)
            if mask.sum() == 0:
                continue
            idxs = np.where(mask)[0]
            # forward fill
            for k in range(len(idxs)-1):
                feat[idxs[k]+1:idxs[k+1], s_idx, fi] = col_vals[idxs[k]]
            feat[idxs[-1]+1:, s_idx, fi] = col_vals[idxs[-1]]
            # backward fill
            feat[:idxs[0], s_idx, fi] = col_vals[idxs[0]]

    # z-score normalise globally per feature
    flat = feat.reshape(-1, F)
    mu  = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    feat = (feat - mu) / std
    feat = np.nan_to_num(feat, nan=0.0)
    return feat, mu, std

# ── PyTorch model ──────────────────────────────────────────────────────────────

def build_model(in_dim, out_dim, hidden, gru_hidden, num_gcn, dropout, device):
    """
    ST-GCN:
      For each timestep in the sequence:
        x → GCN layers (graph convolution with adj) → node embeddings
      Stack embeddings over time → GRU → predict next-hour features
    """
    import torch
    import torch.nn as nn

    class GraphConv(nn.Module):
        """Simple spectral-style GCN: H = σ(A * X * W)"""
        def __init__(self, in_f, out_f, dropout):
            super().__init__()
            self.lin = nn.Linear(in_f, out_f)
            self.bn  = nn.BatchNorm1d(out_f)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, adj):
            # x: (B, S, in_f),  adj: (B, S, S)
            x = torch.bmm(adj, x)          # aggregate neighbours
            x = self.lin(x)                # linear transform
            B, S, F = x.shape
            x = self.bn(x.reshape(B*S, F)).reshape(B, S, F)
            x = torch.relu(x)
            return self.drop(x)

    class STGCN(nn.Module):
        def __init__(self, in_dim, out_dim, hidden, gru_hidden, num_gcn, dropout):
            super().__init__()
            self.gcn_layers = nn.ModuleList()
            dim = in_dim
            for i in range(num_gcn):
                self.gcn_layers.append(GraphConv(dim, hidden, dropout))
                dim = hidden

            self.gru = nn.GRU(
                input_size  = hidden,
                hidden_size = gru_hidden,
                num_layers  = 2,
                batch_first = True,
                dropout     = dropout,
            )
            self.head = nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gru_hidden // 2, out_dim),
            )

        def forward(self, x_seq, adj_seq):
            """
            x_seq   : (B, T, S, F_in)
            adj_seq : (B, T, S, S)
            Returns : (B, S, F_out) — predicted features at next hour
            """
            B, T, S, F_in = x_seq.shape
            embeddings = []
            for t in range(T):
                x   = x_seq[:, t]       # (B, S, F_in)
                adj = adj_seq[:, t]     # (B, S, S)
                for gcn in self.gcn_layers:
                    x = gcn(x, adj)     # (B, S, hidden)
                embeddings.append(x)   # (B, S, hidden)

            # stack time → (B, S, T, hidden) → per-station GRU
            emb = torch.stack(embeddings, dim=2)  # (B, S, T, hidden)
            emb = emb.reshape(B * S, T, -1)       # (B*S, T, hidden)
            _, h = self.gru(emb)                  # h: (2, B*S, gru_hidden)
            h = h[-1]                             # last layer: (B*S, gru_hidden)
            out = self.head(h)                    # (B*S, out_dim)
            out = out.reshape(B, S, -1)           # (B, S, out_dim)
            return out

    model = STGCN(in_dim, out_dim, hidden, gru_hidden, num_gcn, dropout).to(device)
    return model

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_sequences_gpu(feat, adj, seq_len, target_feat_indices, device):
    """
    GPU-accelerated sliding window using torch.unfold / as_strided.
    feat : (H, S, F)  numpy float32
    adj  : (H, S, S)  numpy float32
    Returns torch tensors on CPU ready for TensorDataset.
    """
    import torch
    H, S, F = feat.shape
    F_out = len(target_feat_indices)

    feat_t = torch.tensor(feat, dtype=torch.float32, device=device)  # (H, S, F)
    adj_t  = torch.tensor(adj,  dtype=torch.float32, device=device)  # (H, S, S)

    # sliding windows via unfold on the time dimension
    # feat_t.permute(1,2,0) → (S, F, H) → unfold → (S, F, N, T)
    feat_unf = feat_t.permute(1, 2, 0).unfold(-1, seq_len, 1)   # (S, F, N+1, T)
    feat_unf = feat_unf.permute(2, 3, 0, 1)                      # (N+1, T, S, F)

    adj_unf  = adj_t.permute(1, 2, 0).unfold(-1, seq_len, 1)    # (S, S, N+1, T)
    adj_unf  = adj_unf.permute(2, 3, 0, 1)                       # (N+1, T, S, S)

    # target is the hour immediately after each window — drop last window (no target)
    feat_unf = feat_unf[:-1]                                      # (N, T, S, F)
    adj_unf  = adj_unf[:-1]                                       # (N, T, S, S)

    tgt_idx  = torch.tensor(target_feat_indices, device=device)
    N = H - seq_len
    Y        = feat_t[seq_len:, :, :][:, :, tgt_idx]             # (N, S, F_out)

    # drop samples where any target is NaN
    valid = ~Y.isnan().any(dim=-1).any(dim=-1)                    # (N,)
    feat_unf = feat_unf[valid]
    adj_unf  = adj_unf[valid]
    Y        = Y[valid]

    print(f"  {valid.sum().item()} valid samples  (dropped {(~valid).sum().item()} with NaN targets)")
    return feat_unf.cpu(), adj_unf.cpu(), Y.cpu()

# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available()
                          else "cuda")
    print(f"\n{'='*60}")
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Seq len: {args.seq_len}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # 1. Load edge matrices
    print("Loading edge matrices …")
    d = np.load(NPZ_PATH, allow_pickle=True)
    adj_all    = d["adj"]          # (H, S, S)
    timestamps = [t.decode() if isinstance(t, bytes) else t for t in d["timestamps"]]
    names      = [s.decode() if isinstance(s, bytes) else s for s in d["station_names"]]
    H, S, _ = adj_all.shape
    print(f"  {H} hours,  {S} stations: {names}")

    # 2. Load and align node features
    print("\nLoading station CSVs …")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    stations, data = [], {}
    for fp in csv_files:
        meta = parse_header(fp)
        if not meta["name"]: meta["name"] = fp.stem
        try:
            df = load_csv(fp)
            stations.append(meta)
            data[meta["name"]] = df
            print(f"  {meta['name']:25s}  {len(df)} rows")
        except Exception as e:
            print(f"  [SKIP] {fp.name}: {e}")

    # keep only stations that appear in the edge matrix
    stations = [s for s in stations if s["name"] in names]
    stations.sort(key=lambda s: names.index(s["name"]))

    all_times = pd.DatetimeIndex(
        pd.concat([pd.Series(index=df.index, dtype=float) for df in data.values()]).index
    )
    all_hours = all_times.floor("h").unique().sort_values()

    print("\nBuilding node-feature tensor …")
    feat, mu, std = build_node_features(stations, data, all_hours, NODE_FEATURES)
    print(f"  feat shape : {feat.shape}   (H, S, F_in={len(NODE_FEATURES)})")

    # feat is already built over all_hours in the same order as adj timestamps
    # just use it directly — no alignment loop needed
    feat_aligned = feat

    # target feature indices
    tgt_idx = [NODE_FEATURES.index(f) for f in TARGET_FEATURES if f in NODE_FEATURES]
    F_out   = len(tgt_idx)
    F_in    = len(NODE_FEATURES)

    print(f"\nBuilding sequences on GPU (seq_len={args.seq_len}) …")
    X, A, Y = make_sequences_gpu(feat_aligned, adj_all, args.seq_len, tgt_idx, device)
    N = len(X)
    print(f"  {N} samples  |  X:{X.shape}  A:{A.shape}  Y:{Y.shape}")

    # 3. Train / val / test split (chronological)
    n_test = max(1, int(N * TEST_SPLIT))
    n_val  = max(1, int(N * VAL_SPLIT))
    n_train = N - n_val - n_test

    X_tr, A_tr, Y_tr = X[:n_train], A[:n_train], Y[:n_train]
    X_v,  A_v,  Y_v  = X[n_train:n_train+n_val], A[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_te, A_te, Y_te = X[-n_test:], A[-n_test:], Y[-n_test:]
    print(f"  Train:{n_train}  Val:{n_val}  Test:{n_test}")

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    tr_ds = TensorDataset(to_t(X_tr), to_t(A_tr), to_t(Y_tr))
    v_ds  = TensorDataset(to_t(X_v),  to_t(A_v),  to_t(Y_v))
    te_ds = TensorDataset(to_t(X_te), to_t(A_te), to_t(Y_te))

    pin = (device.type == "cuda")
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  drop_last=True, pin_memory=pin, num_workers=0)
    v_dl  = DataLoader(v_ds,  batch_size=args.batch, shuffle=False, pin_memory=pin, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False, pin_memory=pin, num_workers=0)

    # 4. Model
    model = build_model(F_in, F_out, args.hidden, GRU_HIDDEN, NUM_GCN_LAYERS, DROPOUT, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    opt      = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    criterion = nn.MSELoss()

    # 5. Training (skip if --eval-only)
    if args.eval_only:
        print(f"\n  --eval-only: skipping training, loading saved model from {MODEL_PATH}")
    else:
        print(f"\n{'─'*60}")
        print(f"  Training  (predicting: {TARGET_FEATURES})")
        print(f"{'─'*60}")

        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            # ── train
            model.train()
            tr_loss = 0.0
            t0 = time.time()
            for xb, ab, yb in tr_dl:
                xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb, ab)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_dl)

            # ── val
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, ab, yb in v_dl:
                    xb, ab, yb = xb.to(device), ab.to(device), yb.to(device)
                    val_loss += criterion(model(xb, ab), yb).item()
            val_loss /= len(v_dl)

            scheduler.step(val_loss)
            elapsed = time.time() - t0

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={tr_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}  "
                  f"({elapsed:.1f}s)")

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "mu": mu, "std": std,
                    "names": names,
                    "node_features": NODE_FEATURES,
                    "target_features": TARGET_FEATURES,
                    "seq_len": args.seq_len,
                    "hidden": args.hidden,
                    "gru_hidden": GRU_HIDDEN,
                    "num_gcn": NUM_GCN_LAYERS,
                    "dropout": DROPOUT,
                    "epoch": epoch,
                }, MODEL_PATH)
                print(f"    ✓ saved best model (val={best_val:.5f})")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP:
                    print(f"\n  Early stopping at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
                    break

    # 6. Test evaluation
    print(f"\n{'─'*60}")
    print("  Test evaluation  (loading best model)")
    print(f"{'─'*60}")
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, ab, yb in te_dl:
            xb, ab = xb.to(device), ab.to(device)
            pred = model(xb, ab).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())

    all_pred = np.concatenate(all_pred, axis=0)   # (N_test, S, F_out)
    all_true = np.concatenate(all_true, axis=0)

    # denormalise
    tgt_mu  = mu[tgt_idx]
    tgt_std = std[tgt_idx]
    pred_dn = all_pred * tgt_std + tgt_mu
    true_dn = all_true * tgt_std + tgt_mu

    print(f"\n  Per-feature  RMSE  |  MAE  (denormalised, original units)")
    print(f"  {'Feature':<12}  {'RMSE':>8}  {'MAE':>8}")
    print(f"  {'─'*35}")
    for fi, feat_name in enumerate(TARGET_FEATURES):
        rmse = float(np.sqrt(np.mean((pred_dn[:, :, fi] - true_dn[:, :, fi])**2)))
        mae  = float(np.mean(np.abs(pred_dn[:, :, fi]  - true_dn[:, :, fi])))
        print(f"  {feat_name:<12}  {rmse:8.4f}  {mae:8.4f}")

    print(f"\n  Per-station  RMSE  (all target features averaged)")
    print(f"  {'Station':<25}  {'RMSE':>8}")
    print(f"  {'─'*35}")
    for si, sname in enumerate(names):
        rmse = float(np.sqrt(np.mean((pred_dn[:, si, :] - true_dn[:, si, :])**2)))
        print(f"  {sname:<25}  {rmse:8.4f}")

    # 7. Predict next hour
    if args.predict:
        print(f"\n{'─'*60}")
        print("  INFERENCE  – predicting the next hour from the last window")
        print(f"{'─'*60}")

        # last seq_len hours of features and adj
        last_x   = feat_aligned[-args.seq_len:]       # (T, S, F)
        last_adj = adj_all[-args.seq_len:]             # (T, S, S)
        last_ts  = timestamps[-1]

        xb = torch.tensor(last_x[None], dtype=torch.float32).to(device)   # (1, T, S, F)
        ab = torch.tensor(last_adj[None], dtype=torch.float32).to(device)

        with torch.no_grad():
            pred = model(xb, ab).cpu().numpy()[0]   # (S, F_out)

        pred_dn = pred * tgt_std + tgt_mu

        print(f"\n  Last known timestamp : {last_ts}")
        print(f"  Predicting for       : +1 hour\n")
        print(f"  {'Station':<25}  " + "  ".join(f"{f:>8}" for f in TARGET_FEATURES))
        print(f"  {'─'*65}")
        for si, sname in enumerate(names):
            vals = "  ".join(f"{pred_dn[si, fi]:8.3f}" for fi in range(F_out))
            print(f"  {sname:<25}  {vals}")

    if not args.eval_only:
        print(f"\nDone. Model saved to {MODEL_PATH}")
    else:
        print(f"\nDone.")


if __name__ == "__main__":
    args = get_args()
    train(args)