from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SEED = 0
SEQ_LEN = 8
FEATURES = ["t2m", "msl", "u10", "v10"]
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1  # remaining 20% is test
EPOCHS = 50
PATIENCE = 7

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ---------------------------------------------------------------- data
default_path = Path(__file__).parent / "data" / "2024_instant.nc"
ds = xr.open_dataset(default_path)
print(ds.sizes)

print(ds.valid_time.values[:12], ds.valid_time.values[-12:])
print(np.unique(np.diff(ds.valid_time.values), return_counts=True))