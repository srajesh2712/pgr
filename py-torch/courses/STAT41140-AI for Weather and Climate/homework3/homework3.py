from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEEDS = [0, 1, 2]
SEQ_LEN = 8
FEATURES = ["t2m", "msl", "u10", "v10"]
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15  # remaining 15% is test (~1 year each for val/test)
EPOCHS = 50
PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- data
DATA_DIR = Path(__file__).parent / "data"


def load_point(path):
    """Series of FEATURES at the Dublin-ish grid point for one ERA5 file."""
    with xr.open_dataset(path) as ds:
        pt = ds.sel(latitude=53.25, longitude=-6.25, method="nearest")
        if "expver" in pt.dims:  # ERA5 can carry two experiment versions with NaNs
            pt = pt.mean("expver", skipna=True)
        time_name = next(n for n in ("valid_time", "time") if n in pt.coords)
        part = pt[FEATURES].to_dataframe()[FEATURES]
        part.index = pt[time_name].to_index()
        return part


parts = []
for path in sorted(DATA_DIR.glob("*_instant.nc")):
    part = load_point(path)
    print(f"{path.name}: {part.index.min()} -> {part.index.max()}  ({len(part)} rows)")
    parts.append(part)

df = pd.concat(parts).sort_index()
n_dup = int(df.index.duplicated().sum())
if n_dup:
    print(f"WARNING: dropping {n_dup} duplicate timestamps (a file may be mislabeled)")
    df = df[~df.index.duplicated(keep="first")]
df = df.dropna()

# files have an extra 05:00 step each day; keep the regular 3-hourly grid
df = df[df.index.hour % 3 == 0]
gaps, counts = np.unique(np.diff(df.index.values), return_counts=True)
print("time gaps:", dict(zip(gaps.astype("timedelta64[h]").astype(int), counts)))
print(len(df), "rows,", df.index.min(), "->", df.index.max())

# cyclical time features (not scaled; already in [-1, 1])
hour = df.index.hour.values
doy = df.index.dayofyear.values
time_feats = np.stack(
    [
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * doy / 366),
        np.cos(2 * np.pi * doy / 366),
    ],
    axis=1,
)

# chronological split on rows; scaler fit on TRAIN rows only (no leakage)
n = len(df)
n_train = int(TRAIN_FRAC * n)
n_val = int((TRAIN_FRAC + VAL_FRAC) * n)

raw = df[FEATURES].values
scaler = StandardScaler().fit(raw[:n_train])
data = np.concatenate([scaler.transform(raw), time_feats], axis=1)
N_FEATURES = data.shape[1]

# window i uses rows [i, i+SEQ_LEN) and predicts t2m at row i+SEQ_LEN
X = np.stack([data[i : i + SEQ_LEN] for i in range(n - SEQ_LEN)])
target_idx = np.arange(SEQ_LEN, n)

# drop windows that span a time gap (window start -> target must be exactly SEQ_LEN*3h)
times = df.index.values
valid = (times[SEQ_LEN:] - times[: n - SEQ_LEN]) == np.timedelta64(3 * SEQ_LEN, "h")
print(f"dropping {int((~valid).sum())} windows that span a gap")
X, target_idx = X[valid], target_idx[valid]

train_m = target_idx < n_train
val_m = (target_idx >= n_train) & (target_idx < n_val)
test_m = target_idx >= n_val

# target = CHANGE in (scaled) t2m from the last input step, rescaled by the
# train-set std of that change so the model learns corrections to persistence
delta = data[target_idx, 0] - X[:, -1, 0]
delta_std = delta[train_m].std()
y = delta / delta_std

X_train, y_train = X[train_m], y[train_m]
X_val, y_val = X[val_m], y[val_m]
X_test, y_test = X[test_m], y[test_m]
print("windows  train/val/test:", len(X_train), len(X_val), len(X_test))
print("val starts", df.index[n_train], "| test starts", df.index[n_val])


class WeatherDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


val_loader = DataLoader(WeatherDataset(X_val, y_val), batch_size=64)
test_loader = DataLoader(WeatherDataset(X_test, y_test), batch_size=64)


# ---------------------------------------------------------------- model
class WeatherTransformer(nn.Module):
    def __init__(self, n_features=N_FEATURES, d_model=32, nhead=4, num_layers=2, seq_len=SEQ_LEN):
        super().__init__()
        self.embedding = nn.Linear(n_features, d_model)
        # learned positional embedding so the model knows timestep order
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos
        x = self.transformer(x)
        return self.fc(x[:, -1, :]).squeeze(-1)  # squeeze(-1): safe for batch size 1


def evaluate(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(preds)


def train_transformer(seed):
    """Train one model with early stopping; return its test-set prediction (scaled change)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = DataLoader(WeatherDataset(X_train, y_train), batch_size=32, shuffle=True)
    model = WeatherTransformer().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        val_loss = float(np.mean((evaluate(model, val_loader) - y_val) ** 2))
        scheduler.step(val_loss)
        print(f"seed {seed}  epoch {epoch:02d}  train {running / len(train_loader):.4f}  val {val_loss:.4f}")
        if val_loss < best_val:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    return evaluate(model, test_loader)


# ---------------------------------------------------------------- helpers
to_k = lambda v: v * scaler.scale_[0] + scaler.mean_[0]  # scaled t2m -> Kelvin
change_to_k = lambda c: c * delta_std * scaler.scale_[0]  # model output -> change in K

persist_k = to_k(X_test[:, -1, 0])  # "next value = last value"
actual_k = raw[target_idx[test_m], 0]  # true t2m in K at the target steps
test_hour = df.index.hour.values[target_idx[test_m]]
test_month = df.index.month.values[target_idx[test_m]]
train_hour = df.index.hour.values[target_idx[train_m]]


def scores(p):
    return (
        np.sqrt(mean_squared_error(actual_k, p)),
        mean_absolute_error(actual_k, p),
    )


results = {}  # name -> list of (rmse, mae), one per seed

# ---------------------------------------------------------------- baselines
results["persistence"] = [scores(persist_k)]

# persistence + average change for that hour of day (learned from train windows only)
mean_change_by_hour = (
    pd.Series(change_to_k(y_train)).groupby(train_hour).mean().reindex(range(24)).fillna(0.0)
)
hour_k = persist_k + mean_change_by_hour.loc[test_hour].values
results["persist+hour"] = [scores(hour_k)]

# ridge regression on the flattened window (linear model, same inputs and target)
ridge = Ridge(alpha=1.0).fit(X_train.reshape(len(X_train), -1), y_train)
ridge_k = persist_k + change_to_k(ridge.predict(X_test.reshape(len(X_test), -1)))
results["linear (ridge)"] = [scores(ridge_k)]

# ---------------------------------------------------------------- transformer, several seeds
tf_preds_k = []
for seed in SEEDS:
    p_k = persist_k + change_to_k(train_transformer(seed))
    tf_preds_k.append(p_k)
    results.setdefault("transformer", []).append(scores(p_k))

# ---------------------------------------------------------------- report
print("\n=== test results (K) ===")
for name, runs in results.items():
    r = np.array(runs)
    rmse_txt = f"{r[:, 0].mean():.3f}" + (f" +/- {r[:, 0].std():.3f}" if len(r) > 1 else "")
    mae_txt = f"{r[:, 1].mean():.3f}" + (f" +/- {r[:, 1].std():.3f}" if len(r) > 1 else "")
    print(f"{name:16s} RMSE = {rmse_txt:16s} MAE = {mae_txt}")

# error breakdown for the seed-averaged transformer vs persistence+hour
ens_k = np.mean(tf_preds_k, axis=0)
print(f"\ntransformer ensemble of {len(SEEDS)} seeds: RMSE = {scores(ens_k)[0]:.3f}  MAE = {scores(ens_k)[1]:.3f}")

breakdown = pd.DataFrame(
    {
        "hour": test_hour,
        "month": test_month,
        "persist+hour": np.abs(actual_k - hour_k),
        "transformer": np.abs(actual_k - ens_k),
    }
)
print("\nMAE (K) by hour of day:")
print(breakdown.groupby("hour")[["persist+hour", "transformer"]].mean().round(3).to_string())
print("\nMAE (K) by month:")
print(breakdown.groupby("month")[["persist+hour", "transformer"]].mean().round(3).to_string())
