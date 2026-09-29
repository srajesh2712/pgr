import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# =====================================================
# CONFIGURATION
# =====================================================

FILE_PATH = "homework2/data/dly1475.csv"
MODEL_PATH = "weather_cnn_lstm.pth"
PLOT_PATH = "soil_temp_forecast_presentation.png"

PATIENCE = 15
LOOKBACK_DAYS = 30
TARGET_COL = "soil"  # Target: Soil Temperature (°C)

BATCH_SIZE = 512
EPOCHS = 200
LEARNING_RATE = 0.001


# =====================================================
# LOAD & PREPROCESS DATA
# =====================================================

print("Loading weather data...")

df = pd.read_csv(FILE_PATH, skiprows=24)
df.columns = df.columns.str.strip()

feature_cols = [
    "maxtp",
    "mintp",
    "gmin",
    "rain",
    "cbl",
    "wdsp",
    "hm",
    "ddhm",
    "hg",
    "soil",
    "pe",
    "evap",
    "smd_wd",
    "smd_md",
    "smd_pd",
    "glorad",
]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[feature_cols] = df[feature_cols].ffill().fillna(0)
target_idx = feature_cols.index(TARGET_COL)


# =====================================================
# TEMPORAL SPLIT & SCALING
# =====================================================

raw_data = df[feature_cols].values
total_samples = len(raw_data)

train_split_idx = int(total_samples * 0.70)
val_split_idx = int(total_samples * 0.85)

train_raw = raw_data[:train_split_idx]
val_raw = raw_data[train_split_idx:val_split_idx]
test_raw = raw_data[val_split_idx:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_raw)
val_scaled = scaler.transform(val_raw) # we use transform here to ensure the validation data is scaled using the training data's parameters
test_scaled = scaler.transform(test_raw) 

scaled_features = scaler.transform(raw_data)


# =====================================================
# CREATE SEQUENCES
# =====================================================
"""
This function creates sequences of data for the CNN-LSTM model. Each sequence consists of `seq_length` days of features, and the target is the soil temperature on the day immediately following the sequence. The function returns two numpy arrays: `X` containing the input sequences and `y` containing the corresponding target values.
"""

def create_sequences(data, seq_length, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)


X_train, y_train = create_sequences(train_scaled, LOOKBACK_DAYS, target_idx)
X_val, y_val = create_sequences(val_scaled, LOOKBACK_DAYS, target_idx)
X_test, y_test = create_sequences(test_scaled, LOOKBACK_DAYS, target_idx)


# =====================================================
# DATASET & DATALOADERS
# =====================================================


class WeatherDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    WeatherDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False
)
val_loader = DataLoader(
    WeatherDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    WeatherDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
)


# =====================================================
# CNN-LSTM MODEL ARCHITECTURE
# =====================================================


class WeatherCNNLSTM(nn.Module):

    def __init__(self, num_features, cnn_out_channels=32, lstm_hidden_dim=64):
        super().__init__()

        self.conv1d = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=cnn_out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = WeatherCNNLSTM(num_features=len(feature_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-4)

best_val_loss = float("inf")

# Load existing weights if present
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))


# =====================================================
# TRAINING
# =====================================================

patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(X_train)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)

    val_loss /= len(X_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            MODEL_PATH,
        )
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        break


# =====================================================
# INVERSE TRANSFORM UTILITY
# =====================================================


def inverse_transform_target(
    scaled_array, scaler, target_idx, total_features
):
    """Converts scaled 1D array back to original units (°C)."""
    dummy = np.zeros((len(scaled_array), total_features))
    dummy[:, target_idx] = scaled_array.flatten()
    return scaler.inverse_transform(dummy)[:, target_idx]


# =====================================================
#  EVALUATION
# =====================================================

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_preds_scaled, y_actuals_scaled = [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        preds = model(batch_X)
        y_preds_scaled.extend(preds.cpu().numpy())
        y_actuals_scaled.extend(batch_y.numpy())

y_preds_scaled = np.array(y_preds_scaled)
y_actuals_scaled = np.array(y_actuals_scaled)

# Convert from scaled z-scores back to real degrees Celsius (°C)
preds_deg = inverse_transform_target(
    y_preds_scaled, scaler, target_idx, len(feature_cols)
)
actuals_deg = inverse_transform_target(
    y_actuals_scaled, scaler, target_idx, len(feature_cols)
)

# Calculate metrics in real physical units (°C)
mae_deg = mean_absolute_error(actuals_deg, preds_deg)
mse_deg = np.mean((actuals_deg - preds_deg) ** 2)
rmse_deg = np.sqrt(mse_deg)
r2_val = r2_score(actuals_deg, preds_deg)

# Print executive report
print("=" * 60)
print("     CNN-LSTM SOIL TEMPERATURE FORECAST MODEL REPORT        ")
print("=" * 60)
print(f"  Test Samples Evaluated : {len(actuals_deg)}")
print(f"  Mean Absolute Error    : {mae_deg:.3f} °C")
print(f"  Root Mean Sq Error     : {rmse_deg:.3f} °C")
print(f"  R-Squared (R²) Score   : {r2_val:.3f}")
print("-" * 60)

# Print sample test comparison
sample_df = pd.DataFrame(
    {
        "Actual Temp (°C)": np.round(actuals_deg[:10], 2),
        "Predicted Temp (°C)": np.round(preds_deg[:10], 2),
        "Abs Error (°C)": np.round(np.abs(actuals_deg[:10] - preds_deg[:10]), 2),
    }
)

print("\n--- SAMPLE TEST PREDICTIONS (First 10 Test Days) ---")
print(sample_df.to_string(index=False))

# Forecast for Tomorrow
with torch.no_grad():
    latest_days = scaled_features[-LOOKBACK_DAYS:]
    latest_tensor = (
        torch.tensor(latest_days, dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    pred_scaled_tomorrow = model(latest_tensor).item()

pred_tomorrow_deg = inverse_transform_target(
    np.array([pred_scaled_tomorrow]), scaler, target_idx, len(feature_cols)
)[0]

print("\n" + "=" * 60)
print(f"  TOMORROW'S SOIL TEMP FORECAST: {pred_tomorrow_deg:.2f} °C")
print("=" * 60)


# =====================================================
# GENERATE GRAPH
# =====================================================

plt.figure(figsize=(12, 5), dpi=150)
plot_days = min(120, len(actuals_deg))  # Plot last 120 days for clear presentation

plt.plot(
    actuals_deg[:plot_days],
    label="Actual Soil Temp (°C)",
    color="#1f77b4",
    linewidth=1.8,
)
plt.plot(
    preds_deg[:plot_days],
    label="Predicted Soil Temp (°C)",
    color="#ff7f0e",
    linestyle="--",
    linewidth=1.8,
)

plt.title(
    f"CNN-LSTM Soil Temperature Forecast vs Actuals (Test Set)\nRMSE: {rmse_deg:.2f} °C | MAE: {mae_deg:.2f} °C | R²: {r2_val:.2f}",
    fontsize=13,
    pad=12,
)
plt.xlabel("Days in Test Period", fontsize=11)
plt.ylabel("Soil Temperature (°C)", fontsize=11)
plt.legend(frameon=True, loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

plt.savefig(PLOT_PATH)
print(f"\nPresentation plot saved to '{PLOT_PATH}'")